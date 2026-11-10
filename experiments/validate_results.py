#!/usr/bin/env python3
"""Check saved experimental outputs without rerunning or changing the study.

This verifies bookkeeping, physical invariants, paired comparisons and hash
integrity. It does not certify the scientific assumptions or prove authenticity
of a log against a malicious author who can rewrite the entire hash chain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METHODS = {"attribution_mpc", "state_mpc", "software_replay", "reset_measured", "max_charge", "safe_stop"}
KEYS = ["scenario_id", "method", "t"]
LEDGER_METHODS = {"poisoned_history", "clean_history", "clean_reference", "attribution_mpc", "software_replay"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def booleans(values):
    s = pd.Series(values).reset_index(drop=True)
    result = s.astype(str).str.lower().map({"true": True, "false": False})
    if result.isna().any():
        raise ValueError(f"Invalid/missing boolean values: {s[result.isna()].head().tolist()}")
    return result.to_numpy(dtype=bool)


class Audit:
    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.checks = []
        self.counts = {}

    def check(self, name, condition, **details):
        ok = bool(np.all(condition))
        self.checks.append({"name": name, "passed": ok, **details})
        return ok

    def close(self, name, actual, expected, **details):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        ok = np.isclose(a, b, rtol=1e-8, atol=self.tolerance, equal_nan=True)
        finite = np.isfinite(a) & np.isfinite(b)
        difference = np.abs(a[finite] - b[finite])
        self.check(name, ok, compared=int(ok.size), failed=int(np.count_nonzero(~ok)),
                   max_absolute_error=float(difference.max()) if difference.size else None, **details)


def validate(results: Path, audit: Audit):
    cfg = json.loads((results / "scenario_config.json").read_text())
    metadata = json.loads((results / "run_metadata.json").read_text())
    cases = cfg["scenarios"]
    case_map = {s["scenario_id"]: s for s in cases}
    spec = cfg["battery"]
    ep = pd.read_csv(results / "episodes.csv")
    ctrl = pd.read_csv(results / "controllers.csv")
    traces = pd.read_csv(results / "traces.csv")
    profiles = pd.read_csv(results / "profiles.csv")
    audit.counts.update(scenarios=len(cases), episodes=len(ep), controllers=len(ctrl),
                        trace_rows=len(traces), profile_rows=len(profiles))
    audit.check("run_finished", metadata.get("status") == "complete", recorded_status=metadata.get("status"))
    audit.check("scenario_count_and_ids", len(cases) == len(case_map) == len(ep) == metadata["episodes"]
                and not ep.scenario_id.duplicated().any() and set(ep.scenario_id) == set(case_map))
    audit.check("synthetic_data_declared", cfg.get("data_type") == "synthetic")
    for name, frame, keys in (("controllers", ctrl, ["scenario_id", "method"]),
                               ("traces", traces, KEYS), ("profiles", profiles, ["scenario_id", "t"])):
        audit.check(f"{name}_unique_keys", not frame.duplicated(keys).any())
        audit.check(f"{name}_scenario_ids", set(frame.scenario_id) == set(case_map))
    audit.check("exact_six_controllers_per_scenario", len(ctrl) == len(cases) * 6 and
                all(set(g.method) == METHODS for _, g in ctrl.groupby("scenario_id")))
    audit.check("expected_total_trace_count", len(traces) == sum(7 * s["horizon"] - 5 * (s["attack_t"] + s["delay_steps"]) for s in cases))
    audit.check("expected_total_profile_count", len(profiles) == sum(s["horizon"] for s in cases))
    epi = ep.set_index("scenario_id")
    ctrli = ctrl.set_index(["scenario_id", "method"])
    tri = traces.set_index(KEYS)
    pri = profiles.set_index(["scenario_id", "t"])
    # Strictly deterministic invariant checks; solver success is not presumed.
    config_errors, trajectory_errors, summary_errors, episode_errors = [], [], [], []
    summary_numeric = ["future_cost", "future_unserved_kwh", "final_energy_kwh", "time_to_target_steps",
                       "target_shortfall_kwh", "infeasible_plan_steps", "requested_clipped_kwh",
                       "battery_throughput_kwh", "initial_actual_energy_kwh", "solver_seconds"]
    recomputed, saved = [], []
    for sid, s in case_map.items():
        e = epi.loc[sid]
        detection = s["attack_t"] + s["delay_steps"]
        deadline = detection + s["deadline_steps"]
        for key, value in s.items():
            if key == "scenario_id":
                continue
            same = e[key] == value if isinstance(value, str) else np.isclose(e[key], value, atol=audit.tolerance, rtol=1e-8)
            if not same:
                config_errors.append(f"{sid}:{key}")
        if e.detection_t != detection or e.deadline_t != deadline:
            config_errors.append(f"{sid}:derived_times")
        p = profiles[profiles.scenario_id == sid].sort_values("t")
        if not np.array_equal(p.t, np.arange(s["horizon"])):
            trajectory_errors.append(f"{sid}:profile_times")
        histories = {}
        for method in METHODS | {"clean_reference", "clean_history", "poisoned_history"}:
            g = traces[(traces.scenario_id == sid) & (traces.method == method)].sort_values("t")
            history = method.endswith("history")
            expected_t = np.arange(detection) if history else np.arange(detection, s["horizon"])
            if not np.array_equal(g.t, expected_t):
                trajectory_errors.append(f"{sid}:{method}:times")
                continue
            if not np.allclose(g.energy_before_kwh.iloc[1:], g.energy_after_kwh.iloc[:-1], atol=audit.tolerance, rtol=1e-8):
                trajectory_errors.append(f"{sid}:{method}:state_continuity")
            expected_start = s["initial_energy_kwh"] if history else e.e_reference_detect if method == "clean_reference" else e.e_actual_detect
            if not np.isclose(g.energy_before_kwh.iloc[0], expected_start, atol=audit.tolerance, rtol=1e-8):
                trajectory_errors.append(f"{sid}:{method}:initial_energy")
            expected_phase = "pre_detection" if history else "post_detection"
            if not (g.phase == expected_phase).all():
                trajectory_errors.append(f"{sid}:{method}:phase")
            if history:
                histories[method] = g
                continue
            end_energy = float(g.energy_after_kwh.iloc[-1])
            deadline_energy = float(g.loc[g.t == deadline - 1, "energy_after_kwh"].iloc[0])
            missed = float((g.unserved_kw * g.dt_hours).sum())
            deadline_success = deadline_energy >= s["target_energy_kwh"] - 1e-6
            success = deadline_success and missed <= 1e-6 and end_energy >= s["target_energy_kwh"] - 1e-6
            reached = g.loc[g.energy_after_kwh >= s["target_energy_kwh"] - 1e-6, "t"]
            first = 0 if expected_start >= s["target_energy_kwh"] - 1e-6 else (int(reached.iloc[0]) + 1 - detection if len(reached) else np.nan)
            values = {"future_cost": float(g.step_cost.sum()), "future_unserved_kwh": missed,
                      "final_energy_kwh": end_energy, "time_to_target_steps": first,
                      "target_shortfall_kwh": max(0., s["target_energy_kwh"] - deadline_energy),
                      "infeasible_plan_steps": int((~booleans(g.plan_strict_feasible)).sum()),
                      "requested_clipped_kwh": float(g.requested_clipped_kwh.sum()),
                      "battery_throughput_kwh": float(((g.charge_kw + g.discharge_kw) * g.dt_hours).sum()),
                      "initial_actual_energy_kwh": expected_start, "solver_seconds": float(g.solver_seconds.sum())}
            if method in METHODS:
                recorded = ctrli.loc[(sid, method)]
                saved.extend(float(recorded[name]) for name in summary_numeric)
                recomputed.extend(values[name] for name in summary_numeric)
                if bool(recorded.deadline_success) != deadline_success or bool(recorded.physical_recovery_success) != success:
                    summary_errors.append(f"{sid}:{method}:success_flags")
            else:
                if not np.isclose(e.reference_future_cost, values["future_cost"], atol=audit.tolerance, rtol=1e-8):
                    episode_errors.append(f"{sid}:reference_future_cost")
                if not np.isclose(e.reference_future_unserved_kwh, missed, atol=audit.tolerance, rtol=1e-8):
                    episode_errors.append(f"{sid}:reference_future_unserved")
                if bool(e.reference_recovery_success) != success:
                    episode_errors.append(f"{sid}:reference_success")
        for name, prefix in (("poisoned_history", "actual"), ("clean_history", "reference")):
            g = histories[name]
            for field, expected in ((f"e_{prefix}_detect", g.energy_after_kwh.iloc[-1]),
                                    (f"pre_{prefix}_cost", g.step_cost.sum()),
                                    (f"pre_{prefix}_unserved_kwh", (g.unserved_kw * g.dt_hours).sum())):
                if not np.isclose(e[field], expected, atol=audit.tolerance, rtol=1e-8):
                    episode_errors.append(f"{sid}:{field}")
    audit.check("scenario_configuration_matches_rows", not config_errors, errors=config_errors[:12])
    audit.check("trajectory_times_initial_states_and_continuity", not trajectory_errors, errors=trajectory_errors[:12])
    audit.close("controller_summaries_recomputed_from_traces", saved, recomputed)
    audit.check("controller_success_flags_recomputed", not summary_errors, errors=summary_errors[:12])
    audit.check("episode_history_and_reference_accounting", not episode_errors, errors=episode_errors[:12])
    numeric = ["energy_before_kwh", "energy_after_kwh", "charge_kw", "discharge_kw", "grid_import_kw",
               "load_kw", "load_served_kw", "pv_available_kw", "pv_used_kw", "unserved_kw", "true_price", "step_cost", "dt_hours"]
    audit.check("finite_physical_trace_values", np.isfinite(traces[numeric].to_numpy()).all())
    tol = audit.tolerance
    audit.check("hard_energy_bounds", ((traces[["energy_before_kwh", "energy_after_kwh"]] >= spec["min_energy_kwh"] - tol)
                                       & (traces[["energy_before_kwh", "energy_after_kwh"]] <= spec["capacity_kwh"] + tol)).all().all())
    audit.check("hard_power_and_service_bounds", (traces[numeric[2:]] >= -tol).all().all()
                and (traces.charge_kw <= spec["max_charge_kw"] + tol).all()
                and (traces.discharge_kw <= spec["max_discharge_kw"] + tol).all()
                and (traces.grid_import_kw <= traces.grid_limit_kw + tol).all()
                and (traces.pv_used_kw <= traces.pv_available_kw + tol).all()
                and (traces.unserved_kw <= traces.load_kw + tol).all())
    audit.check("no_simultaneous_charging_and_discharging", np.minimum(traces.charge_kw, traces.discharge_kw) <= tol)
    audit.close("sampling_interval", traces.dt_hours, np.full(len(traces), spec["dt_hours"]))
    audit.close("battery_energy_balance", traces.energy_after_kwh, traces.energy_before_kwh
                + spec["eta_charge"] * traces.charge_kw * traces.dt_hours
                - traces.discharge_kw * traces.dt_hours / spec["eta_discharge"])
    audit.close("power_balance", traces.grid_import_kw + traces.pv_used_kw + traces.discharge_kw,
                traces.load_served_kw + traces.charge_kw)
    audit.close("critical_service_balance", traces.load_served_kw + traces.unserved_kw, traces.load_kw)
    audit.close("grid_purchase_cost", traces.step_cost, traces.grid_import_kw * traces.dt_hours * traces.true_price)
    audit.close("actuator_clipping_accounting", traces.requested_clipped_kwh,
                (abs(traces.requested_charge_kw - traces.charge_kw) + abs(traces.requested_discharge_kw - traces.discharge_kw)) * traces.dt_hours)
    paired_profiles = profiles.merge(traces, on=["scenario_id", "t"], suffixes=("_profile", ""), validate="one_to_many")
    for trace_name, profile_name in (("load_kw", "actual_load"), ("pv_available_kw", "actual_pv"),
                                     ("true_price", "tariff"), ("grid_limit_kw", "grid_limit"),
                                     ("physical_grid_limit_kw", "physical_grid_limit")):
        audit.close(f"shared_exogenous_{trace_name}", paired_profiles[trace_name], paired_profiles[profile_name])
    comparison_columns = [c for c in traces.columns if c not in ("method", "solver_seconds", "reference_solver_seconds", "solver_kind")]
    base = traces[traces.method == "state_mpc"].sort_values(["scenario_id", "t"]).reset_index(drop=True)
    for method in ("attribution_mpc", "reset_measured"):
        other = traces[traces.method == method].sort_values(["scenario_id", "t"]).reset_index(drop=True)
        mismatches = []
        for col in comparison_columns:
            if pd.api.types.is_numeric_dtype(base[col]):
                same = np.isclose(base[col], other[col], atol=tol, rtol=1e-8, equal_nan=True).all()
            else:
                same = base[col].fillna("<missing>").equals(other[col].fillna("<missing>"))
            if not same:
                mismatches.append(col)
        audit.check(f"{method}_same_actions_and_outcomes_as_state_mpc", not mismatches, unequal_columns=mismatches)
    audit.close("paired_soc_deficit", ep.soc_deficit_kwh, ep.e_reference_detect - ep.e_actual_detect)
    audit.close("realized_cost_difference", ep.realized_extra_cost, ep.actual_future_cost - ep.reference_future_cost)
    actual_summary = ep.merge(ctrl[ctrl.method == "state_mpc"], on="scenario_id", validate="one_to_one")
    audit.close("episode_actual_future_cost", actual_summary.actual_future_cost, actual_summary.future_cost)
    audit.check("realized_pair_success_flag", booleans(actual_summary.realized_paired_success) ==
                (booleans(actual_summary.physical_recovery_success) & booleans(actual_summary.reference_recovery_success)))
    af, rf = booleans(ep.actual_feasible), booleans(ep.reference_feasible)
    both = af & rf
    audit.check("estimated_delta_defined_iff_both_strictly_feasible", np.isfinite(ep.estimated_extra_cost) == both)
    audit.check("branch_optima_defined_iff_strictly_feasible", np.array_equal(np.isfinite(ep.estimated_actual_cost), af)
                and np.array_equal(np.isfinite(ep.estimated_reference_cost), rf))
    audit.close("estimated_paired_cost_difference", ep.loc[both, "estimated_extra_cost"],
                ep.loc[both, "estimated_actual_cost"] - ep.loc[both, "estimated_reference_cost"])
    classes = np.select([both, ~af & rf, af & ~rf], ["both_feasible", "actual_only_infeasible", "reference_only_infeasible"], default="both_infeasible")
    audit.check("feasibility_classes", ep.feasibility_class.to_numpy() == classes)
    zero = ep.strength == 0
    audit.close("no_attack_same_detection_state", ep.loc[zero, "e_actual_detect"], ep.loc[zero, "e_reference_detect"])
    audit.close("no_attack_zero_realized_cost_difference", ep.loc[zero, "realized_extra_cost"], np.zeros(zero.sum()))
    audit.close("no_attack_zero_defined_estimated_difference", ep.loc[zero & both, "estimated_extra_cost"], np.zeros((zero & both).sum()))
    ledger_checks(results, audit, tri, pri, case_map)
    ac_checks(results, audit, traces, metadata)
    current_hash_errors = []
    for relative, recorded_hash in metadata.get("code_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != recorded_hash:
            current_hash_errors.append(relative)
    audit.check("recorded_code_hashes_match_current_files", bool(metadata.get("code_sha256")) and not current_hash_errors,
                changed_or_missing=current_hash_errors, note="Changed postprocessing also requires a new provenance record.")


def ledger_checks(results, audit, traces, profiles, cases):
    seen, previous, previous_t = set(), {}, {}
    errors, total = [], 0
    max_trace_error = 0.0
    for line_number, line in enumerate(gzip.open(results / "effect_ledger.jsonl.gz", "rt", encoding="utf-8"), 1):
        event = json.loads(line)
        total += 1
        key = (event["scenario_id"], event["method"], event["t"])
        chain = key[:2]
        sid, method, t = key
        s = cases[sid]
        effect = event["executed_effect"]
        if key in seen or method not in LEDGER_METHODS:
            errors.append(f"line {line_number}:duplicate_or_unknown_event")
        seen.add(key)
        check_hash = event.copy()
        event_hash = check_hash.pop("event_hash")
        if digest(check_hash) != event_hash or digest(event["forecast"]) != event["forecast_id"]:
            errors.append(f"line {line_number}:event_or_forecast_hash")
        decision = {"forecast_id": event["forecast_id"], "belief": event["controller_belief_kwh"],
                    "charge": effect["requested_charge_kw"], "discharge": effect["requested_discharge_kw"]}
        if digest(decision) != event["decision_id"]:
            errors.append(f"line {line_number}:decision_hash")
        if event["previous_event_hash"] != previous.get(chain, "0" * 64):
            errors.append(f"line {line_number}:chain_link")
        if chain in previous_t and t != previous_t[chain] + 1:
            errors.append(f"line {line_number}:chain_time")
        previous[chain], previous_t[chain] = event_hash, t
        if event["observation_id"] != f"{sid}:{method}:{t}":
            errors.append(f"line {line_number}:observation_id")
        if key not in traces.index:
            errors.append(f"line {line_number}:missing_trace")
            continue
        row = traces.loc[key]
        for name, value in effect.items():
            other = dict(zip(KEYS, key))[name] if name in KEYS else row[name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                close = np.isclose(value, other, atol=audit.tolerance, rtol=1e-8)
                max_trace_error = max(max_trace_error, abs(float(value) - float(other)))
            else:
                close = value == other
            if not close:
                errors.append(f"line {line_number}:trace:{name}")
        f = event["forecast"]
        n = s["horizon"] - t
        if set(f) != {"load_kw", "pv_kw", "price_per_kwh", "grid_limit_kw"} or any(len(a) != n for a in f.values()):
            errors.append(f"line {line_number}:forecast_shape")
        elif not all(np.isfinite(a).all() and (np.asarray(a) >= -audit.tolerance).all() for a in f.values()):
            errors.append(f"line {line_number}:forecast_finite_nonnegative")
        else:
            p = profiles.loc[sid].sort_index().iloc[t:]
            if not (np.isclose(f["load_kw"][0], p.actual_load.iloc[0], atol=audit.tolerance)
                    and np.isclose(f["pv_kw"][0], p.actual_pv.iloc[0], atol=audit.tolerance)
                    and np.allclose(f["grid_limit_kw"], p.grid_limit, atol=audit.tolerance)):
                errors.append(f"line {line_number}:forecast_current_observation_or_grid")
            prices = p.tariff.to_numpy().copy()
            poisoned = method == "poisoned_history" and s["attack_t"] <= t < s["attack_t"] + s["delay_steps"] and s["strength"] > 0
            expected_channel = "sanitized_forecast" if method not in {"clean_history", "poisoned_history"} else "poisoned_future_tariff" if poisoned else "authentic_forecast"
            if poisoned:
                prices[1:] *= 1 - s["strength"]
            if event["input_channel"] != expected_channel or not np.allclose(f["price_per_kwh"], prices, atol=audit.tolerance):
                errors.append(f"line {line_number}:forecast_channel_or_tariff")
    expected_keys = set(traces.index[traces.index.get_level_values("method").isin(LEDGER_METHODS)].tolist())
    audit.counts.update(ledger_events=total, ledger_chains=len(previous))
    audit.check("ledger_complete_unique_trace_coverage", seen == expected_keys and total == len(expected_keys),
                expected_events=len(expected_keys), actual_events=total, missing=len(expected_keys - seen), extra=len(seen - expected_keys))
    audit.check("ledger_hashes_chains_forecasts_and_effect_correspondence", not errors,
                error_count=len(errors), examples=errors[:15], max_trace_numeric_error=max_trace_error,
                scope="Internal consistency only; an unanchored hash chain is not proof against complete rewriting.")


def ac_checks(results, audit, traces, metadata):
    path = results / "ac_validation.csv"
    if not metadata.get("stages", {}).get("ac", path.exists()):
        audit.counts["ac_validation"] = "not_requested"
        return
    if not audit.check("requested_ac_results_present", path.exists()):
        return
    cfg = json.loads((results / "ac_config.json").read_text())["config"]
    ac = pd.read_csv(path)
    expected = traces[(traces.method == "state_mpc") | traces.scenario_id.isin(["case_000", "case_001"])].copy()
    audit.check("ac_selection_count_and_keys", not ac.duplicated(KEYS).any() and
                set(ac.set_index(KEYS).index) == set(expected.set_index(KEYS).index) and len(ac) == len(expected))
    joined = ac.merge(expected, on=KEYS, suffixes=("_ac", "_trace"), validate="one_to_one")
    for field in ("charge_kw", "discharge_kw", "load_served_kw", "pv_used_kw", "grid_import_kw"):
        audit.close(f"ac_uses_executed_{field}", joined[f"{field}_ac"], joined[f"{field}_trace"])
    audit.close("ac_uses_physical_source_cap", ac.grid_limit_kw, ac.physical_grid_limit_kw)
    audit.close("ac_charge_storage_sign_mapping", ac.battery_charge_kw, ac.charge_kw)
    audit.close("ac_discharge_storage_sign_mapping", ac.battery_discharge_kw, ac.discharge_kw)
    island = ac.grid_limit_kw <= cfg["power_tolerance_kw"]
    checked = ac.ac_status == "checked_grid_connected"
    allowed = {"not_checked_islanded", "checked_grid_connected", "power_flow_not_converged", "unsupplied_or_nonfinite_bus"}
    audit.check("ac_status_and_island_scope", ac.ac_status.isin(allowed).all()
                and ((ac.ac_status == "not_checked_islanded") == island).all()
                and not booleans(ac.loc[island, "ac_converged"]).any()
                and ac.loc[island, ["ac_passed", "ac_network_passed", "ac_grid_kw", "ac_min_voltage_pu"]].isna().all().all())
    connected = ac.loc[checked]
    audit.check("ac_checked_rows_converged", booleans(connected.ac_converged).all())
    physical_cols = ["ac_grid_kw", "ac_loss_kw", "ac_ideal_grid_kw", "ac_min_voltage_pu", "ac_max_voltage_pu",
                     "ac_max_line_loading_percent", "ac_max_trafo_loading_percent"]
    audit.check("ac_checked_values_finite", np.isfinite(connected[physical_cols].to_numpy()).all())
    audit.close("ac_executed_lossless_balance", ac.ac_ideal_grid_kw, ac.load_served_kw + ac.battery_charge_kw
                - ac.battery_discharge_kw - ac.pv_used_kw)
    audit.close("ac_loss_accounting", connected.ac_grid_kw - connected.ac_ideal_grid_kw - connected.ac_loss_kw,
                connected.ac_power_balance_error_kw)
    audit.check("ac_solver_power_balance_within_tolerance", abs(connected.ac_power_balance_error_kw) <= 1e-4,
                tolerance_kw=1e-4, max_absolute_error_kw=float(abs(connected.ac_power_balance_error_kw).max()) if len(connected) else None)
    audit.close("ac_import_excess_accounting", connected.ac_grid_cap_excess_kw,
                np.maximum(0, connected.ac_grid_kw - connected.grid_limit_kw))
    voltage = (connected.ac_min_voltage_pu < cfg["voltage_min_pu"] - 1e-8) | (connected.ac_max_voltage_pu > cfg["voltage_max_pu"] + 1e-8)
    loading = connected[["ac_max_line_loading_percent", "ac_max_trafo_loading_percent"]].max(axis=1) > cfg["loading_max_percent"] + 1e-6
    cap = connected.ac_grid_cap_excess_kw > cfg["power_tolerance_kw"]
    audit.check("ac_violations_and_pass_flags_recomputed", np.array_equal(booleans(connected.ac_voltage_violation), voltage)
                and np.array_equal(booleans(connected.ac_loading_violation), loading)
                and np.array_equal(booleans(connected.ac_grid_cap_violation), cap)
                and np.array_equal(booleans(connected.ac_network_passed), ~(voltage | loading))
                and np.array_equal(booleans(connected.ac_passed), ~(voltage | loading | cap)))
    failed = ~(checked | island)
    audit.check("ac_failed_solves_not_counted_as_passes", not booleans(ac.loc[failed, "ac_passed"]).any()
                and not booleans(ac.loc[failed, "ac_converged"]).any())
    audit.counts.update(ac_rows=len(ac), ac_checked_grid_connected=int(checked.sum()), ac_not_checked_islanded=int(island.sum()),
                        ac_unsuccessful_solves=int(failed.sum()), ac_passed_checked=int(booleans(connected.ac_passed).sum()),
                        ac_voltage_violations=int(voltage.sum()), ac_loading_violations=int(loading.sum()), ac_import_cap_violations=int(cap.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/main")
    parser.add_argument("--output", type=Path, help="Verification JSON (default: RESULTS/validation.json)")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")
    results = args.results.resolve()
    out = args.output.resolve() if args.output else results / "validation.json"
    audit = Audit(args.tolerance)
    try:
        validate(results, audit)
    except Exception as exc:
        audit.check("validation_completed_without_exception", False, exception_type=type(exc).__name__, error=str(exc))
    passed = all(c["passed"] for c in audit.checks)
    report = {"status": "passed" if passed else "failed", "validated_utc": datetime.now(timezone.utc).isoformat(),
              "results_directory": str(results), "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "absolute_tolerance": args.tolerance, "counts": audit.counts,
              "checks_passed": sum(c["passed"] for c in audit.checks), "checks_total": len(audit.checks),
              "checks": audit.checks,
              "scope": "Saved-output consistency, physical constraints, paired accounting, hash-chain integrity and AC reporting; no new optimization runs or assumption validation."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Validation {report['status']}: {report['checks_passed']}/{report['checks_total']} checks; {out}")
    for check in audit.checks:
        if not check["passed"]:
            print(json.dumps(check, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
