"""Causal-information, physical-execution and fair-comparison integration tests."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from recovery.model import (METHODS, Scenario, apply_action, issue_forecast, make_profile,
                            simulate_history, simulate_recovery)
from recovery.optimization import BatterySpec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def test_future_actual_measurements_cannot_leak_into_forecast():
    s = Scenario("causal", 342)
    p = make_profile(s)
    t = 9
    before = issue_forecast(s, p, t)
    altered = deepcopy(p)
    altered.actual_load[t + 1:] += 100
    altered.actual_pv[t + 1:] += 100
    after = issue_forecast(s, altered, t)
    for name in ("load_kw", "pv_kw", "price_per_kwh", "grid_limit_kw"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert before.load_kw[0] == pytest.approx(p.actual_load[t])
    assert before.pv_kw[0] == pytest.approx(p.actual_pv[t])


def test_attack_changes_only_future_tariff_input():
    s = Scenario("poison", 342, strength=.8)
    p = make_profile(s)
    clean = issue_forecast(s, p, s.attack_t)
    corrupt = issue_forecast(s, p, s.attack_t, poisoned=True)
    for name in ("load_kw", "pv_kw", "grid_limit_kw"):
        np.testing.assert_array_equal(getattr(clean, name), getattr(corrupt, name))
    assert clean.price_per_kwh[0] == corrupt.price_per_kwh[0]
    np.testing.assert_allclose(corrupt.price_per_kwh[1:], .2 * clean.price_per_kwh[1:])


def test_zero_attack_has_identical_clean_and_actual_physical_history():
    s = Scenario("negative_control", 93, strength=0)
    p, spec = make_profile(s), BatterySpec()
    ea, actual, _ = simulate_history(s, p, spec)
    er, clean, _ = simulate_history(s, p, spec, clean=True)
    assert ea == er
    for a, r in zip(actual, clean, strict=True):
        for name in ("energy_before_kwh", "energy_after_kwh", "charge_kw", "discharge_kw",
                     "grid_import_kw", "unserved_kw", "step_cost"):
            assert a[name] == r[name]


def test_actuator_clipping_preserves_bounds_and_balance():
    spec = BatterySpec()
    rng = np.random.default_rng(582)
    for _ in range(100):
        energy = rng.uniform(spec.min_energy_kwh, spec.capacity_kwh)
        charge, discharge = rng.uniform(0, 12), 0.
        if rng.random() < .5:
            charge, discharge = 0., charge
        row = apply_action(energy, charge, discharge, rng.uniform(0, 9),
                           rng.uniform(0, 4), rng.uniform(0, 8), .3, spec)
        assert spec.min_energy_kwh <= row["energy_after_kwh"] <= spec.capacity_kwh
        assert 0 <= row["charge_kw"] <= spec.max_charge_kw
        assert 0 <= row["discharge_kw"] <= spec.max_discharge_kw
        assert row["unserved_kw"] >= 0
        assert row["load_served_kw"] >= -1e-8
        assert abs(row["balance_error_kw"]) < 1e-7
        expected = energy + (spec.eta_charge * row["charge_kw"] - row["discharge_kw"] / spec.eta_discharge) * spec.dt_hours
        assert row["energy_after_kwh"] == pytest.approx(expected, abs=1e-8)
    empty = apply_action(spec.min_energy_kwh, 0, 4, 4, 0, 0, .3, spec)
    assert empty["energy_after_kwh"] == spec.min_energy_kwh
    assert empty["discharge_kw"] == 0
    assert empty["unserved_kw"] == 4


def test_ledger_hashes_bind_forecasts_decisions_and_executed_effects():
    s = Scenario("ledger", 41)
    _, _, ledger = simulate_history(s, make_profile(s), BatterySpec())
    previous = "0" * 64
    for event in ledger:
        assert event["previous_event_hash"] == previous
        assert event["forecast_id"] == digest(event["forecast"])
        content = {key: value for key, value in event.items() if key != "event_hash"}
        assert event["event_hash"] == digest(content)
        effect = event["executed_effect"]
        assert effect["t"] == event["t"]
        assert effect["scenario_id"] == event["scenario_id"]
        altered = deepcopy(content)
        altered["executed_effect"]["energy_after_kwh"] += .01
        assert digest(altered) != event["event_hash"]
        previous = event["event_hash"]


def test_all_methods_start_same_real_state_and_fair_mpc_variants_match():
    s = Scenario("fair", 1057, strength=.95, delay_steps=4, grid_cap_kw=7)
    spec, p = BatterySpec(), make_profile(s)
    ea, _, _ = simulate_history(s, p, spec)
    er, _, _ = simulate_history(s, p, spec, clean=True)
    _, _, ref_states, _ = simulate_recovery(s, p, spec, er, "clean_reference")
    outputs = {}
    for method in METHODS:
        outputs[method] = simulate_recovery(s, p, spec, ea, method,
                                           reference_energy=er, reference_states=ref_states)
        metrics, rows, states, _ = outputs[method]
        assert metrics["initial_actual_energy_kwh"] == ea
        assert rows[0]["energy_before_kwh"] == ea
        assert states[s.detection_t] == ea
        for row in rows:
            assert row["energy_after_kwh"] == pytest.approx(
                row["energy_before_kwh"] + (.95 * row["charge_kw"] - row["discharge_kw"] / .95) * .25,
                abs=1e-7)
    for other in ("attribution_mpc", "reset_measured"):
        a, b = outputs["state_mpc"], outputs[other]
        for name in ("future_cost", "future_unserved_kwh", "physical_recovery_success",
                     "deadline_success", "final_energy_kwh", "target_shortfall_kwh"):
            assert a[0][name] == pytest.approx(b[0][name], abs=1e-8)
        for ra, rb in zip(a[1], b[1], strict=True):
            for name in ("charge_kw", "discharge_kw", "energy_after_kwh"):
                assert ra[name] == pytest.approx(rb[name], abs=1e-8)
    # The replay's software belief can start at the clean value; its plant cannot.
    replay_rows = outputs["software_replay"][1]
    assert replay_rows[0]["controller_belief_kwh"] == er
    assert replay_rows[0]["energy_before_kwh"] == ea


def test_late_recharge_does_not_retroactively_meet_deadline():
    s = Scenario("late", 31, delay_steps=2, deadline_steps=4, grid_cap_kw=9, noise_scale=0)
    spec, p = BatterySpec(), make_profile(s)
    metrics, _, states, _ = simulate_recovery(s, p, spec, 1.2, "max_charge")
    assert states[s.deadline_t] < s.target_energy_kwh
    assert states[s.horizon] >= s.target_energy_kwh - 1e-6
    assert metrics["time_to_target_steps"] > s.deadline_steps
    assert not metrics["deadline_success"]
    assert not metrics["physical_recovery_success"]


def test_confirmation_is_clean_equivalent_for_static_posted_tariffs_only():
    s = Scenario("gating", 851, strength=.95, delay_steps=4)
    p, spec = make_profile(s), BatterySpec()
    clean_e, clean_rows, _ = simulate_history(s, p, spec, clean=True)
    gate_e, gate_rows, _ = simulate_history(s, p, spec, prevention="confirmation_gate")
    assert gate_e == pytest.approx(clean_e, abs=1e-8)
    np.testing.assert_allclose([r["step_cost"] for r in gate_rows],
                               [r["step_cost"] for r in clean_rows], atol=1e-8)
    attack_e, _, _ = simulate_history(s, p, spec)
    authorized_e, _, _ = simulate_history(s, p, spec, prevention="effect_authorization")
    assert authorized_e == pytest.approx(attack_e, abs=1e-8)
