#!/usr/bin/env python3
"""Run the complete reproducible study; output contains actual solver results."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from recovery.model import (METHODS, Scenario, issue_forecast, make_profile,
                            scenario_grid, simulate_history, simulate_recovery)
from recovery.optimization import BatterySpec, minimum_recovery_steps, solve_battery

ROOT = Path(__file__).resolve().parent


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def feasibility(a, r):
    return ("both_feasible" if a and r else "actual_only_infeasible" if r
            else "reference_only_infeasible" if a else "both_infeasible")


def run_main(cases, out):
    spec = BatterySpec()
    episodes, controllers, traces, profiles = [], [], [], []
    started = time.perf_counter()
    with gzip.open(out / "effect_ledger.jsonl.gz", "wt", encoding="utf-8") as ledger_file:
        for i, s in enumerate(cases):
            p = make_profile(s)
            ea, actual_history, la = simulate_history(s, p, spec)
            er, reference_history, lr = simulate_history(s, p, spec, clean=True)
            f = issue_forecast(s, p, s.detection_t)
            pa = solve_battery(ea, f, spec, target_energy_kwh=s.target_energy_kwh, deadline_index=s.deadline_steps)
            pr = solve_battery(er, f, spec, target_energy_kwh=s.target_energy_kwh, deadline_index=s.deadline_steps)
            reference_metrics, reference_trace, reference_states, reference_ledger = simulate_recovery(
                s, p, spec, er, "clean_reference")
            traces.extend(actual_history + reference_history + reference_trace)
            all_ledger = la + lr + reference_ledger
            method_metrics = {}
            for method in METHODS:
                metrics, trace, _, ledger = simulate_recovery(
                    s, p, spec, ea, method, reference_energy=er, reference_states=reference_states)
                controllers.append(metrics); method_metrics[method] = metrics
                traces.extend(trace); all_ledger.extend(ledger)
            for event in all_ledger:
                ledger_file.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            for t in range(s.horizon):
                profiles.append({"scenario_id": s.scenario_id, "seed": s.seed, "t": t,
                                 **{name: float(getattr(p, name)[t]) for name in vars(p)}})
            ma = method_metrics["state_mpc"]
            row = {**asdict(s), "detection_t": s.detection_t, "deadline_t": s.deadline_t,
                   "eta_charge": spec.eta_charge, "e_actual_detect": ea, "e_reference_detect": er,
                   "soc_deficit_kwh": er - ea, "actual_feasible": pa.success,
                   "reference_feasible": pr.success, "feasibility_class": feasibility(pa.success, pr.success),
                   "estimated_extra_cost": pa.cost - pr.cost if pa.success and pr.success else np.nan,
                   "estimated_actual_cost": pa.cost, "estimated_reference_cost": pr.cost,
                   "min_recovery_steps_actual": minimum_recovery_steps(ea, f, spec, s.target_energy_kwh),
                   "min_recovery_steps_reference": minimum_recovery_steps(er, f, spec, s.target_energy_kwh),
                   "pre_actual_cost": sum(r["step_cost"] for r in actual_history),
                   "pre_reference_cost": sum(r["step_cost"] for r in reference_history),
                   "pre_actual_unserved_kwh": sum(r["unserved_kw"] * spec.dt_hours for r in actual_history),
                   "pre_reference_unserved_kwh": sum(r["unserved_kw"] * spec.dt_hours for r in reference_history),
                   "realized_extra_cost": ma["future_cost"] - reference_metrics["future_cost"],
                   "realized_paired_success": ma["physical_recovery_success"] and reference_metrics["physical_recovery_success"],
                   "actual_future_cost": ma["future_cost"], "reference_future_cost": reference_metrics["future_cost"],
                   "current_price": f.price_per_kwh[0], "mean_forecast_price": float(np.mean(f.price_per_kwh)),
                   "reference_future_unserved_kwh": reference_metrics["future_unserved_kwh"],
                   "reference_recovery_success": reference_metrics["physical_recovery_success"]}
            episodes.append(row)
            if (i + 1) % 10 == 0 or i + 1 == len(cases):
                print(f'Main episodes {i+1}/{len(cases)}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    pd.DataFrame(episodes).to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(controllers).to_csv(out / "controllers.csv", index=False)
    pd.DataFrame(traces).to_csv(out / "traces.csv", index=False)
    pd.DataFrame(profiles).to_csv(out / "profiles.csv", index=False)
    return pd.DataFrame(episodes), pd.DataFrame(controllers), pd.DataFrame(traces)


def run_sensitivity(out, episodes):
    spec = BatterySpec()
    rows = []
    for cap in np.arange(4.5, 10.01, 0.5):
        for deadline in range(2, 21, 2):
            s = Scenario("controlled", 4242, noise_scale=0.0, grid_cap_kw=float(cap), deadline_steps=deadline)
            f = issue_forecast(s, make_profile(s), s.detection_t)
            a = solve_battery(4.0, f, spec, target_energy_kwh=9, deadline_index=deadline)
            r = solve_battery(9.0, f, spec, target_energy_kwh=9, deadline_index=deadline)
            rows.append({"grid_cap_kw": cap, "deadline_steps": deadline, "deadline_hours": deadline * .25,
                         "actual_energy": 4, "reference_energy": 9, "actual_feasible": a.success,
                         "reference_feasible": r.success, "feasibility_class": feasibility(a.success, r.success),
                         "extra_cost": a.cost - r.cost if a.success and r.success else np.nan})
    pd.DataFrame(rows).to_csv(out / "feasibility_sweep.csv", index=False)
    equal = []
    for cap in (5.5, 7, 9):
        for deadline in (4, 8, 12, 16):
            for price_scale in (.5, 1, 2):
                s = Scenario("equal_deficit", 4242, noise_scale=0, grid_cap_kw=cap,
                             deadline_steps=deadline, tariff_scale=price_scale)
                f = issue_forecast(s, make_profile(s), s.detection_t)
                a = solve_battery(4, f, spec, target_energy_kwh=9, deadline_index=deadline)
                r = solve_battery(9, f, spec, target_energy_kwh=9, deadline_index=deadline)
                equal.append({"deficit_kwh": 5, "grid_cap_kw": cap, "deadline_steps": deadline,
                              "tariff_scale": price_scale, "feasibility_class": feasibility(a.success, r.success),
                              "extra_cost": a.cost - r.cost if a.success and r.success else np.nan})
    pd.DataFrame(equal).to_csv(out / "equal_deficit.csv", index=False)
    sensitivity = []
    for r in episodes.loc[episodes.actual_feasible & episodes.reference_feasible & (episodes.strength > 0)].head(20).to_dict("records"):
        s = Scenario(**{key: r[key] for key in Scenario.__dataclass_fields__})
        f = issue_forecast(s, make_profile(s), s.detection_t)
        for perturb in (-.05, -.02, 0, .02, .05):
            er = np.clip(r["e_reference_detect"] + perturb * spec.capacity_kwh, spec.min_energy_kwh, spec.capacity_kwh)
            plan = solve_battery(er, f, spec, target_energy_kwh=s.target_energy_kwh, deadline_index=s.deadline_steps)
            extra = r["estimated_actual_cost"] - plan.cost if plan.success else np.nan
            sensitivity.append({"scenario_id": s.scenario_id, "reference_error_fraction_capacity": perturb,
                                "perturbed_reference_kwh": er, "reference_feasible": plan.success,
                                "extra_cost": extra, "change_from_nominal": extra - r["estimated_extra_cost"]})
    pd.DataFrame(sensitivity).to_csv(out / "reference_sensitivity.csv", index=False)
    severity = []
    for delay in (2, 4, 8):
        for strength in (0, .25, .5, .65, .8, .95):
            s = Scenario("severity", 4242, delay_steps=delay, strength=strength, noise_scale=0)
            p = make_profile(s)
            ea, _, _ = simulate_history(s, p, spec)
            er, _, _ = simulate_history(s, p, spec, clean=True)
            f = issue_forecast(s, p, s.detection_t)
            a = solve_battery(ea, f, spec, target_energy_kwh=9, deadline_index=s.deadline_steps)
            r = solve_battery(er, f, spec, target_energy_kwh=9, deadline_index=s.deadline_steps)
            severity.append({"strength": strength, "delay_steps": delay, "deficit_kwh": er - ea,
                             "feasibility_class": feasibility(a.success, r.success),
                             "extra_cost": a.cost - r.cost if a.success and r.success else np.nan})
    pd.DataFrame(severity).to_csv(out / "attack_sweep.csv", index=False)
    print('Sensitivity sweeps completed.', flush=True)


def run_prevention(out):
    rows = []
    spec = BatterySpec()
    for seed in range(2000, 2008):
        for strength in (0, .95):
            s = Scenario(f"prevention_{seed}_{strength}", seed, strength=strength, delay_steps=8,
                         noise_scale=.07, grid_cap_kw=7, deadline_steps=8)
            p = make_profile(s)
            for mechanism in ("none", "confirmation_gate", "effect_authorization"):
                energy, history, _ = simulate_history(s, p, spec, prevention=None if mechanism == "none" else mechanism)
                metrics, _, _, _ = simulate_recovery(s, p, spec, energy, "state_mpc")
                rows.append({"seed": seed, "strength": strength, "mechanism": mechanism,
                             "energy_at_detection_kwh": energy, "whole_episode_cost": sum(r["step_cost"] for r in history) + metrics["future_cost"],
                             "whole_episode_unserved_kwh": sum(r["unserved_kw"] * .25 for r in history) + metrics["future_unserved_kwh"],
                             "physical_recovery_success": metrics["physical_recovery_success"],
                             "confirmation_delay_steps": 2 if mechanism == "confirmation_gate" else 0})
    pd.DataFrame(rows).to_csv(out / "prevention.csv", index=False)
    print('Separate prevention trials completed.', flush=True)


def run_ac(out, traces):
    from recovery.platform import ACValidationConfig, validate_ac_dataframe
    # All state-MPC snapshots, plus every baseline for the first two scenarios.
    selected = traces[(traces.method == "state_mpc") | traces.scenario_id.isin(["case_000", "case_001"])].copy()
    selected["dispatch_grid_limit_kw"] = selected.grid_limit_kw
    selected["grid_limit_kw"] = selected.physical_grid_limit_kw
    selected["battery_charge_kw"] = selected.charge_kw
    selected["battery_discharge_kw"] = selected.discharge_kw
    config = ACValidationConfig(storage_capacity_kwh=12)
    result = validate_ac_dataframe(selected, config=config)
    result.to_csv(out / "ac_validation.csv", index=False)
    write_json(out / "ac_config.json", {"config": asdict(config), "selection": "all state_mpc steps plus all traces for case_000 and case_001",
                                       "loss_allowance_kw": 1.0, "source_limit_used": "physical_grid_limit_kw",
                                       "scope": "ex-post balanced grid-connected snapshots; outage/island snapshots not validated"})
    print(f'AC checks completed: {len(result)} snapshots.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results/main")
    parser.add_argument("--figures", type=Path, help="Default: experiments/figures for main; otherwise OUTPUT/figures")
    parser.add_argument("--skip-extras", action="store_true")
    parser.add_argument("--skip-ac", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    figures = args.figures.resolve() if args.figures else (ROOT / "figures" if out == ROOT / "results/main" else out / "figures")
    cases = scenario_grid(args.episodes)
    write_json(out / "scenario_config.json", {"master_seed": 20260924, "battery": asdict(BatterySpec()),
                                             "scenarios": [asdict(s) for s in cases], "data_type": "synthetic"})
    code_files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve(), ROOT / "uv.lock", ROOT / "pyproject.toml"]
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "platform": platform.platform(), "processor": platform.processor(),
                "output_directory": str(out), "figure_directory": str(figures),
                "stages": {"main": True, "sensitivity_and_prevention": not args.skip_extras, "ac": not args.skip_ac, "analysis": not args.skip_analysis},
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "pandapower")},
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in code_files}, "episodes": args.episodes}
    write_json(out / "run_metadata.json", metadata)
    started = time.perf_counter()
    episodes, controllers, traces = run_main(cases, out)
    if not args.skip_extras:
        run_sensitivity(out, episodes); run_prevention(out)
    if not args.skip_ac:
        run_ac(out, traces)
    if not args.skip_analysis:
        from recovery.analysis import analyze_results
        analyze_results(out, figures)
        if not args.skip_extras and not args.skip_ac:
            from recovery.extra_analysis import analyze_extra_results
            analyze_extra_results(out, figures)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
    write_json(out / "run_metadata.json", metadata)
    print(f'Finished. Results: {out}. Elapsed: {metadata["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == "__main__":
    main()
