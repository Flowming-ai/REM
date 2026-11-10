#!/usr/bin/env python3
"""Sensor-noise cohort: relax the exact-observation assumption without modifying the original study.

The original study assumes the recovery controller reads the true battery energy. This run repeats the
same 500 independent test scenarios while the controller observes energy through a noisy sensor, and
measures (a) how attribution accuracy degrades, (b) whether the analytical control equivalence survives,
and (c) how the optimising and heuristic controllers compare as observation quality falls.

Noise is applied only to the measured actual state. The clean reference is a simulated counterfactual and
does not pass through a sensor; its separate uncertainty is covered by the existing reference-perturbation
analysis. Within one scenario and noise level every method receives the identical noise sequence, so the
equivalence comparison is not confounded by different draws.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "src"))

from recovery.model import (METHODS, Scenario, make_profile, simulate_history, issue_forecast,
                            apply_action, _row, simulate_recovery)
from recovery.optimization import BatterySpec, solve_battery

SIGMAS = [0.0, 0.06, 0.12, 0.30, 0.60]          # kWh standard deviation on the observed battery energy
NOISE_SEED_BASE = 20261008                       # fixed before execution


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def noise_sequence(scenario_id: str, sigma: float, horizon: int) -> np.ndarray:
    """Deterministic per-(scenario, sigma) sequence, shared by every method."""
    if sigma <= 0:
        return np.zeros(horizon + 1)
    key = f"{scenario_id}|{sigma}".encode()
    seed = NOISE_SEED_BASE + int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return np.random.default_rng(seed).normal(0.0, sigma, horizon + 1)


def observe(true_energy, draw, spec):
    """Sensor reading: true energy plus noise, clipped to the physically admissible range."""
    return float(np.clip(true_energy + draw, spec.min_energy_kwh, spec.capacity_kwh))


def simulate_recovery_noisy(s, p, spec, initial_energy, method, *, reference_energy=None,
                            reference_states=None, noise=None):
    """simulate_recovery with an observation channel on the measured actual state.

    With noise=None (or all zeros) the belief equals the true energy at every step, which is the original
    behaviour; the reproduction gate in main() checks this against the saved results.
    """
    energy = float(initial_energy)
    if noise is None:
        noise = np.zeros(s.horizon + 1)
    belief = float(reference_energy) if method == "software_replay" else observe(energy, noise[s.detection_t], spec)
    rows = []
    reached = 0 if energy >= s.target_energy_kwh - 1e-6 else None
    states = {s.detection_t: energy}
    for t in range(s.detection_t, s.horizon):
        f = issue_forecast(s, p, t)
        local_deadline = s.deadline_t - t if t <= s.deadline_t else None
        plan = None
        if method in ("max_charge", "safe_stop"):
            c = d = 0.0
            observed = observe(energy, noise[t], spec)
            if method == "max_charge":
                if p.actual_load[t] > p.actual_pv[t] + p.grid_limit[t]:
                    d = min(spec.max_discharge_kw, p.actual_load[t] - p.actual_pv[t] - p.grid_limit[t])
                elif observed < s.target_energy_kwh:
                    c = min(spec.max_charge_kw,
                            (s.target_energy_kwh - observed) / (spec.eta_charge * spec.dt_hours))
            belief = observed
        else:
            if method != "software_replay":
                belief = observe(energy, noise[t], spec)
            offset = 0.0
            if method == "attribution_mpc" and reference_states is not None:
                ref_plan = solve_battery(reference_states[t], f, spec, target_energy_kwh=s.target_energy_kwh,
                                         deadline_index=local_deadline, allow_rescue=False)
                if ref_plan.success:
                    offset = ref_plan.cost
            plan = solve_battery(belief, f, spec, target_energy_kwh=s.target_energy_kwh,
                                 deadline_index=local_deadline, allow_rescue=True, objective_offset=offset)
            if not plan.success:
                raise RuntimeError(f"{s.scenario_id}/{method}: {plan.status}")
            c, d = plan.charge_kw[0], plan.discharge_kw[0]
        outcome = apply_action(energy, c, d, p.actual_load[t], p.actual_pv[t], p.grid_limit[t], p.tariff[t], spec)
        row = _row(s, p, spec, t, method, "post_detection", outcome, plan)
        row["controller_belief_kwh"] = belief
        rows.append(row)
        energy = outcome["energy_after_kwh"]
        states[t + 1] = energy
        if method == "software_replay":
            belief = float(np.clip(plan.energy_kwh[1], spec.min_energy_kwh, spec.capacity_kwh))
        if reached is None and energy >= s.target_energy_kwh - 1e-6:
            reached = t + 1 - s.detection_t
    missed = sum(r["unserved_kw"] * spec.dt_hours for r in rows)
    deadline_energy = states[s.deadline_t]
    terminal_energy = states[s.horizon]
    success = (deadline_energy >= s.target_energy_kwh - 1e-6 and
               terminal_energy >= s.target_energy_kwh - 1e-6 and missed <= 1e-6)
    metrics = {"scenario_id": s.scenario_id, "seed": s.seed, "method": method,
               "future_cost": sum(r["step_cost"] for r in rows),
               "future_unserved_kwh": missed, "deadline_energy_kwh": deadline_energy,
               "terminal_energy_kwh": terminal_energy, "physical_recovery_success": bool(success),
               "steps_to_target": reached}
    actions = [(round(r["charge_kw"], 9), round(r["discharge_kw"], 9)) for r in rows]
    return metrics, actions


def build_cases():
    """Identical construction to the independent test cohort; seeds 10000-10499."""
    rng = np.random.default_rng(20260926)
    return [Scenario(
        scenario_id=f"external_{i:03d}", seed=10000 + i,
        strength=float((.55, .8, .95)[i % 3]),
        delay_steps=int(rng.choice([2, 4, 8])), grid_cap_kw=float(rng.choice([5.5, 7.0, 9.0])),
        deadline_steps=int(rng.choice([4, 8, 12])), noise_scale=float(rng.choice([0.0, .05, .12])),
        tariff_scale=float(rng.choice([.7, 1.0, 1.5])),
    ) for i in range(500)]


def main():
    out = HERE / "results"
    if (HERE / "protocol.json").exists():
        raise RuntimeError("This fixed run already exists; do not overwrite or reselect after seeing outcomes.")
    out.mkdir(parents=True, exist_ok=True)
    spec = BatterySpec()
    cases = build_cases()

    # Confirm the original study's source and results are untouched by this run.
    reference_hashes = json.loads((EXPERIMENTS / "expanded_validation" / "original_assets_sha256.json").read_text())
    drift = {k: v for k, v in reference_hashes.items()
             if (EXPERIMENTS / k).is_file() and digest(EXPERIMENTS / k) != v}
    if drift:
        raise RuntimeError(f"Original assets changed; refusing to run: {sorted(drift)[:5]}")

    protocol = {
        "purpose": "Relax the exact-observation assumption and measure its effect on attribution accuracy, "
                   "the analytical control equivalence, and controller comparison.",
        "cohort": "the 500 independent test scenarios, seeds 10000-10499, unchanged",
        "sigmas_kwh": SIGMAS,
        "noise_model": "i.i.d. Gaussian on the observed actual battery energy, clipped to physical bounds; "
                       "applied at the detection-time estimate and at every recovery step",
        "shared_draws": "one sequence per (scenario, sigma), identical across all six methods",
        "reference_state": "exact; the simulated counterfactual does not pass through a sensor",
        "noise_seed_base": NOISE_SEED_BASE,
        "reproduction_gate": "sigma=0 must reproduce the saved controller outcomes of the independent "
                             "test cohort exactly; the run aborts otherwise",
        "primary_outcomes": ["attribution absolute error vs realised cost difference",
                             "identical-action rate between attribution_mpc and state_mpc",
                             "complete recovery count per method",
                             "mean post-detection unserved critical energy per method"],
        "analysis": "descriptive by sigma; no model is fitted and no scenario is excluded after execution",
    }
    write_json(HERE / "protocol.json", protocol)
    protocol_hash = digest(HERE / "protocol.json")
    (HERE / "protocol.sha256").write_text(protocol_hash + "\n")
    print(f"Protocol frozen: {protocol_hash}; starting {len(cases)} cases x {len(SIGMAS)} noise levels")

    started = time.perf_counter()
    rows, episodes = [], []
    gate_prev = pd.read_csv(EXPERIMENTS / "expanded_validation" / "results" / "controllers.csv")
    for sigma in SIGMAS:
        for i, s in enumerate(cases):
            p = make_profile(s)
            ea, _, _ = simulate_history(s, p, spec)
            er, _, _ = simulate_history(s, p, spec, clean=True)
            seq = noise_sequence(s.scenario_id, sigma, s.horizon)
            f = issue_forecast(s, p, s.detection_t)
            ea_obs = observe(ea, seq[s.detection_t], spec)
            pa = solve_battery(ea_obs, f, spec, target_energy_kwh=s.target_energy_kwh,
                               deadline_index=s.deadline_steps)
            pr = solve_battery(er, f, spec, target_energy_kwh=s.target_energy_kwh,
                               deadline_index=s.deadline_steps)
            _, _, reference_states, _ = simulate_recovery(s, p, spec, er, "clean_reference")
            ref_metrics, _ = simulate_recovery_noisy(s, p, spec, er, "clean_reference", noise=None)
            per_method, actions = {}, {}
            for method in METHODS:
                m, acts = simulate_recovery_noisy(s, p, spec, ea, method, reference_energy=er,
                                                  reference_states=reference_states, noise=seq)
                m["sigma_kwh"] = sigma
                per_method[method] = m; actions[method] = acts
                rows.append(m)
            ma = per_method["state_mpc"]
            episodes.append({
                "scenario_id": s.scenario_id, "seed": s.seed, "sigma_kwh": sigma,
                "strength": s.strength, "e_actual_detect": ea, "e_actual_observed": ea_obs,
                "e_reference_detect": er, "soc_deficit_kwh": er - ea,
                "actual_feasible": bool(pa.success), "reference_feasible": bool(pr.success),
                "estimated_extra_cost": pa.cost - pr.cost if pa.success and pr.success else np.nan,
                "realized_extra_cost": ma["future_cost"] - ref_metrics["future_cost"],
                "realized_paired_success": bool(ma["physical_recovery_success"]
                                                and ref_metrics["physical_recovery_success"]),
                "identical_actions_attr_vs_state": actions["attribution_mpc"] == actions["state_mpc"],
            })
            if (i + 1) % 250 == 0:
                print(f"  sigma={sigma}: {i+1}/{len(cases)}; elapsed {time.perf_counter()-started:.1f}s")
        if sigma == 0.0:   # reproduction gate
            new0 = pd.DataFrame([r for r in rows if r["sigma_kwh"] == 0.0])
            merged = new0.merge(gate_prev, on=["scenario_id", "method"], suffixes=("_new", "_old"))
            assert len(merged) == 500 * len(METHODS), f"gate join size {len(merged)}"
            for new_col, old_col in (("future_cost_new", "future_cost_old"),
                                     ("future_unserved_kwh_new", "future_unserved_kwh_old"),
                                     ("terminal_energy_kwh", "final_energy_kwh")):
                delta = float(np.max(np.abs(merged[new_col] - merged[old_col])))
                if delta > 1e-9:
                    raise RuntimeError(f"Reproduction gate failed on {new_col}: max |diff| = {delta}")
            mism = int((merged["physical_recovery_success_new"] != merged["physical_recovery_success_old"]).sum())
            if mism:
                raise RuntimeError(f"Reproduction gate failed: {mism} success flags differ")
            print("  reproduction gate passed: sigma=0 reproduces the saved cohort")

    pd.DataFrame(rows).to_csv(out / "controllers.csv", index=False)
    pd.DataFrame(episodes).to_csv(out / "episodes.csv", index=False)
    meta = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started, "sigmas": SIGMAS,
            "episodes_per_sigma": len(cases), "methods": list(METHODS),
            "protocol_sha256": protocol_hash, "python": platform.python_version(),
            "packages": {m: importlib.metadata.version(m) for m in ("numpy", "scipy", "pandas")},
            "original_assets_unchanged": True}
    write_json(out / "run_metadata.json", meta)
    print(f"Completed in {meta['elapsed_seconds']:.1f}s; original assets unchanged.")


if __name__ == "__main__":
    main()
