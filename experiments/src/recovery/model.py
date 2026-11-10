"""Synthetic forecast-poisoning interventions and identical-state recovery trials.

No LLM, detector, measured campus data or future realization oracle is used.
The energy agent is a deterministic MPC planner receiving a forecast document.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .optimization import BatteryForecast, BatterySpec, solve_battery

METHODS = ("attribution_mpc", "state_mpc", "software_replay", "reset_measured", "max_charge", "safe_stop")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    strength: float = 0.95
    delay_steps: int = 4
    grid_cap_kw: float = 7.0
    deadline_steps: int = 8
    noise_scale: float = 0.07
    tariff_scale: float = 1.0
    horizon: int = 40
    attack_t: int = 4
    initial_energy_kwh: float = 9.0
    target_energy_kwh: float = 9.0
    loss_allowance_kw: float = 1.0

    @property
    def detection_t(self):
        return self.attack_t + self.delay_steps

    @property
    def deadline_t(self):
        return self.detection_t + self.deadline_steps


@dataclass
class Profile:
    nominal_load: np.ndarray
    nominal_pv: np.ndarray
    actual_load: np.ndarray
    actual_pv: np.ndarray
    tariff: np.ndarray
    grid_limit: np.ndarray
    physical_grid_limit: np.ndarray


def make_profile(s: Scenario) -> Profile:
    """All values are declared synthetic experimental choices, not UAE data."""
    rng = np.random.default_rng(s.seed)
    t = np.arange(s.horizon)
    load_scale = rng.uniform(0.94, 1.06)
    nominal_load = load_scale * (3.4 + 0.35 * np.sin((t + 3) * np.pi / 20))
    nominal_pv = 1.5 * np.maximum(0.0, np.sin(np.pi * (t - 5) / 35))
    load_noise, pv_noise = np.zeros(s.horizon), np.zeros(s.horizon)
    for k in range(s.horizon):
        load_noise[k] = (0.65 * load_noise[k - 1] if k else 0) + rng.normal(0, s.noise_scale * 2.5)
        pv_noise[k] = (0.65 * pv_noise[k - 1] if k else 0) + rng.normal(0, s.noise_scale * 0.8)
    actual_load = np.maximum(0.2, nominal_load + np.clip(load_noise, -0.7, 0.7))
    actual_pv = np.maximum(0, nominal_pv + np.clip(pv_noise, -0.4, 0.4))
    tariff = np.select([t < 4, t < 12, t < 24, t < 32], [0.10, 0.18, 0.52, 0.34], default=0.12)
    # Tiny known tariff variation removes most arbitrary timing ties, not a penalty.
    tariff = s.tariff_scale * (tariff + 0.0002 * t)
    physical = np.full(s.horizon, s.grid_cap_kw)
    physical[20:24] = 0.0  # Announced critical-service outage, known to every planner.
    usable = np.maximum(physical - s.loss_allowance_kw, 0.0)
    return Profile(nominal_load, nominal_pv, actual_load, actual_pv, tariff, usable, physical)


def issue_forecast(s: Scenario, p: Profile, t: int, *, poisoned=False) -> BatteryForecast:
    """Uses only nominal future profiles plus the residual observed at time t."""
    leads = np.arange(s.horizon - t)
    residual_load = p.actual_load[t] - p.nominal_load[t]
    residual_pv = p.actual_pv[t] - p.nominal_pv[t]
    load = np.maximum(0.0, p.nominal_load[t:] + residual_load * 0.65 ** leads)
    pv = np.maximum(0.0, p.nominal_pv[t:] + residual_pv * 0.65 ** leads)
    price = p.tariff[t:].copy()  # Posted tariff, not an oracle realization forecast.
    if poisoned:
        price[1:] *= 1.0 - s.strength
    return BatteryForecast(load, pv, price, p.grid_limit[t:])


def apply_action(energy, charge, discharge, load, pv, grid_limit, price, spec: BatterySpec):
    """Identical actuator clipping for every method; never rolls physical SOC back."""
    requested_c, requested_d = max(0.0, float(charge)), max(0.0, float(discharge))
    if min(requested_c, requested_d) > 1e-6:
        raise ValueError("A controller requested simultaneous charging and discharging.")
    c = min(requested_c, spec.max_charge_kw,
            max(0.0, (spec.capacity_kwh - energy) / (spec.eta_charge * spec.dt_hours)),
            max(0.0, grid_limit + pv - load))
    d = min(requested_d, spec.max_discharge_kw,
            max(0.0, (energy - spec.min_energy_kwh) * spec.eta_discharge / spec.dt_hours), load + c)
    pv_used = min(pv, max(0.0, load + c - d))
    grid = min(grid_limit, max(0.0, load + c - d - pv_used))
    unmet = max(0.0, load + c - d - pv_used - grid)
    next_e = energy + spec.eta_charge * c * spec.dt_hours - d * spec.dt_hours / spec.eta_discharge
    balance = grid + pv_used + d + unmet - load - c
    assert abs(balance) < 1e-7
    assert spec.min_energy_kwh - 1e-7 <= next_e <= spec.capacity_kwh + 1e-7
    return {
        "energy_before_kwh": float(energy), "energy_after_kwh": float(np.clip(next_e, spec.min_energy_kwh, spec.capacity_kwh)),
        "charge_kw": float(c), "discharge_kw": float(d), "grid_import_kw": float(grid),
        "load_served_kw": float(load - unmet), "load_kw": float(load), "pv_available_kw": float(pv),
        "pv_used_kw": float(pv_used), "unserved_kw": float(unmet), "true_price": float(price),
        "step_cost": float(grid * spec.dt_hours * price),
        "requested_charge_kw": requested_c, "requested_discharge_kw": requested_d,
        "requested_clipped_kwh": float((abs(requested_c - c) + abs(requested_d - d)) * spec.dt_hours),
        "balance_error_kw": float(balance),
    }


def _hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def forecast_document(forecast):
    return {name: getattr(forecast, name).tolist() for name in
            ("load_kw", "pv_kw", "price_per_kwh", "grid_limit_kw")}


def event_record(row, forecast, belief, previous_hash, *, input_channel):
    document = forecast_document(forecast)
    event = {"scenario_id": row["scenario_id"], "method": row["method"], "t": row["t"],
             "observation_id": f'{row["scenario_id"]}:{row["method"]}:{row["t"]}',
             "input_channel": input_channel, "forecast_id": _hash(document), "forecast": document,
             "controller_belief_kwh": float(belief), "executed_effect": row,
             "previous_event_hash": previous_hash}
    event["decision_id"] = _hash({"forecast_id": event["forecast_id"], "belief": float(belief),
                                  "charge": row["requested_charge_kw"], "discharge": row["requested_discharge_kw"]})
    event["event_hash"] = _hash(event)
    return event


def _row(s, p, spec, t, method, phase, outcome, plan=None):
    return {"scenario_id": s.scenario_id, "seed": s.seed, "method": method, "t": int(t),
            "dt_hours": spec.dt_hours, "phase": phase, "deadline_t": s.deadline_t,
            "target_energy_kwh": s.target_energy_kwh, "grid_limit_kw": float(p.grid_limit[t]),
            "physical_grid_limit_kw": float(p.physical_grid_limit[t]),
            "plan_strict_feasible": bool(plan.strict_feasible) if plan else True,
            "solver_seconds": float(plan.runtime_seconds) if plan else 0.0,
            "solver_kind": plan.solver_kind if plan else "rule", **outcome}


def simulate_history(s: Scenario, p: Profile, spec: BatterySpec, *, clean=False,
                     prevention=None, confirmation_delay=2):
    energy = s.initial_energy_kwh
    rows, ledger = [], []
    method = "clean_history" if clean else (prevention or "poisoned_history")
    previous = "0" * 64
    for t in range(s.detection_t):
        corrupted = not clean and s.attack_t <= t < s.detection_t
        f = issue_forecast(s, p, t, poisoned=corrupted)
        channel = "poisoned_future_tariff" if corrupted and s.strength > 0 else "authentic_forecast"
        if prevention == "confirmation_gate":
            # No attack flag is inspected. Use a delayed authenticated published tariff
            # snapshot for *all* requests. Current meter/load observations stay current.
            verified_issue = max(0, t - confirmation_delay)
            verified = issue_forecast(s, p, verified_issue, poisoned=False)
            old_prices = verified.price_per_kwh[t - verified_issue:]
            f = BatteryForecast(f.load_kw, f.pv_kw, old_prices, f.grid_limit_kw)
            channel = f"confirmed_posted_tariff_from_t{verified_issue}"
        plan = solve_battery(energy, f, spec, target_energy_kwh=s.target_energy_kwh, allow_rescue=True)
        if not plan.success:
            raise RuntimeError(plan.status)
        c, d = float(plan.charge_kw[0]), float(plan.discharge_kw[0])
        if prevention == "effect_authorization":
            # Permit is bound to forecast hash, action and actuator. It validates
            # identity/integrity/bounds, not semantic correctness of the forecast.
            permit = {"forecast_id": _hash(forecast_document(f)), "actuator": "battery",
                      "charge_kw": c, "discharge_kw": d}
            assert permit["actuator"] == "battery" and c <= spec.max_charge_kw + 1e-6 and d <= spec.max_discharge_kw + 1e-6
        outcome = apply_action(energy, c, d, p.actual_load[t], p.actual_pv[t], p.grid_limit[t], p.tariff[t], spec)
        row = _row(s, p, spec, t, method, "pre_detection", outcome, plan)
        event = event_record(row, f, energy, previous, input_channel=channel)
        ledger.append(event); previous = event["event_hash"]
        rows.append(row); energy = row["energy_after_kwh"]
    return energy, rows, ledger


def simulate_recovery(s: Scenario, p: Profile, spec: BatterySpec, initial_energy,
                      method, *, reference_energy=None, reference_states=None):
    energy = float(initial_energy)
    belief = float(reference_energy) if method == "software_replay" else energy
    rows, ledger = [], []
    previous = "0" * 64
    reached = 0 if energy >= s.target_energy_kwh - 1e-6 else None
    states = {s.detection_t: energy}
    for t in range(s.detection_t, s.horizon):
        f = issue_forecast(s, p, t)
        local_deadline = s.deadline_t - t if t <= s.deadline_t else None
        plan = None
        reference_solver_seconds = 0.0
        if method in ("max_charge", "safe_stop"):
            c = d = 0.0
            if method == "max_charge":
                if p.actual_load[t] > p.actual_pv[t] + p.grid_limit[t]:
                    d = min(spec.max_discharge_kw, p.actual_load[t] - p.actual_pv[t] - p.grid_limit[t])
                elif energy < s.target_energy_kwh:
                    c = min(spec.max_charge_kw, (s.target_energy_kwh - energy) / (spec.eta_charge * spec.dt_hours))
            belief = energy
        else:
            if method != "software_replay":
                belief = energy
            offset = 0.0
            if method == "attribution_mpc" and reference_states is not None:
                ref_plan = solve_battery(reference_states[t], f, spec, target_energy_kwh=s.target_energy_kwh,
                                         deadline_index=local_deadline, allow_rescue=False)
                reference_solver_seconds = ref_plan.runtime_seconds
                if ref_plan.success:
                    offset = ref_plan.cost
            plan = solve_battery(belief, f, spec, target_energy_kwh=s.target_energy_kwh,
                                 deadline_index=local_deadline, allow_rescue=True, objective_offset=offset)
            if not plan.success:
                raise RuntimeError(f"{s.scenario_id}/{method}: {plan.status}")
            c, d = plan.charge_kw[0], plan.discharge_kw[0]
        outcome = apply_action(energy, c, d, p.actual_load[t], p.actual_pv[t], p.grid_limit[t], p.tariff[t], spec)
        row = _row(s, p, spec, t, method, "post_detection", outcome, plan)
        row["reference_solver_seconds"] = reference_solver_seconds
        row["solver_seconds"] += reference_solver_seconds
        row["controller_belief_kwh"] = belief
        if method in ("attribution_mpc", "software_replay", "clean_reference"):
            event = event_record(row, f, belief, previous, input_channel="sanitized_forecast")
            ledger.append(event); previous = event["event_hash"]
        rows.append(row)
        energy = outcome["energy_after_kwh"]
        states[t + 1] = energy
        if method == "software_replay":
            belief = float(np.clip(plan.energy_kwh[1], spec.min_energy_kwh, spec.capacity_kwh))
        if reached is None and energy >= s.target_energy_kwh - 1e-6:
            reached = t + 1 - s.detection_t
    missed = sum(r["unserved_kw"] * spec.dt_hours for r in rows)
    deadline_energy = states[s.deadline_t]
    deadline_success = deadline_energy >= s.target_energy_kwh - 1e-6
    success = deadline_success and missed <= 1e-6 and energy >= s.target_energy_kwh - 1e-6
    metrics = {
        "scenario_id": s.scenario_id, "seed": s.seed, "method": method,
        "future_cost": sum(r["step_cost"] for r in rows), "future_unserved_kwh": missed,
        "deadline_success": bool(deadline_success), "physical_recovery_success": bool(success),
        "final_energy_kwh": energy, "time_to_target_steps": reached,
        "target_shortfall_kwh": max(0.0, s.target_energy_kwh - deadline_energy),
        "infeasible_plan_steps": sum(not r["plan_strict_feasible"] for r in rows),
        "requested_clipped_kwh": sum(r["requested_clipped_kwh"] for r in rows),
        "battery_throughput_kwh": sum((r["charge_kw"] + r["discharge_kw"]) * spec.dt_hours for r in rows),
        "solver_seconds": sum(r["solver_seconds"] for r in rows),
        "initial_actual_energy_kwh": initial_energy,
    }
    return metrics, rows, states, ledger


def scenario_grid(n=100, master_seed=20260924):
    rng = np.random.default_rng(master_seed)
    cases = []
    for i in range(n):
        cases.append(Scenario(
            scenario_id=f"case_{i:03d}", seed=1000 + i,
            strength=float([0.0, 0.55, 0.8, 0.95][i % 4]),
            delay_steps=int(rng.choice([2, 4, 8])), grid_cap_kw=float(rng.choice([5.5, 7.0, 9.0])),
            deadline_steps=int(rng.choice([4, 8, 12])), noise_scale=float(rng.choice([0.0, 0.05, 0.12])),
            tariff_scale=float(rng.choice([0.7, 1.0, 1.5])),
        ))
    return cases
