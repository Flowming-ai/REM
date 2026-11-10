"""Two stronger comparison estimates of the realised extra cost, added after the tests.

cheapest hours: the energy deficit bought at the cheapest forecast prices, first before the deadline, at most at full
                charging power in each step and never while the grid is disconnected;
target level:   the lowest cost of the recovery task from the attacked battery level minus the lowest cost from the
                target level, V(e_a) - V(e_tar): the debt of Eq. (8) with the reference run replaced by the target.
Both are scored like the debt, by the mean absolute error of Eq. (10), on the test set, on the measured profiles and at
every level of meter error. The script rebuilds each scenario with the unchanged experiment code and stops if it does
not reproduce the saved debt. Run from the project folder (RMProj):
    experiments/.venv/bin/python "assignment 3/code/stronger_comparators.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
sys.path[:0] = [str(EXP / "real_data_validation"), str(EXP / "src"), str(EXP)]
sys.dont_write_bytecode = True                                # write no cache files into the experiment folders
from run import real_profile                                 # the measured-data run, unchanged
from recovery.model import Scenario, issue_forecast, make_profile
from recovery.optimization import BatterySpec, solve_battery

DRAWS = 20000                                                 # bootstrap resamples, as for every other interval
rng = np.random.default_rng(20261110)
ESTIMATES = ["debt", "price_at_detection", "cheapest_hours", "target_level"]


def cheapest_hours(deficit, forecast, spec, deadline):
    """Cost of buying the energy deficit at the cheapest forecast prices, first before the deadline."""
    need = abs(deficit) / spec.eta_charge                     # grid energy that restores the deficit, in kWh
    per_step = spec.max_charge_kw * spec.dt_hours             # the most that one step can charge, in kWh
    cost = 0.0
    for steps in (range(deadline), range(deadline, len(forecast))):
        connected = [k for k in steps if forecast.grid_limit_kw[k] > 0]
        for k in sorted(connected, key=lambda k: forecast.price_per_kwh[k]):
            bought = min(need, per_step)
            cost += bought * forecast.price_per_kwh[k]
            need -= bought
    return float(np.sign(deficit) * cost)


def estimate(row, e_attacked, forecast, spec, target, deadline):
    """All four estimates of one scenario; stops if the rebuilt scenario does not reproduce the saved debt."""
    lowest = lambda energy: solve_battery(energy, forecast, spec, target_energy_kwh=target, deadline_index=deadline)
    from_attacked, from_reference = lowest(e_attacked), lowest(row["e_reference_detect"])
    from_target = lowest(target)
    if abs(from_attacked.cost - from_reference.cost - row["estimated_extra_cost"]) > 1e-9:
        raise SystemExit(f"{row['scenario_id']}: the rebuilt scenario does not reproduce the saved debt")
    deficit = row["e_reference_detect"] - e_attacked
    return {"realised": row["realized_extra_cost"], "debt": row["estimated_extra_cost"],
            "price_at_detection": deficit * forecast.price_per_kwh[0] / spec.eta_charge,
            "cheapest_hours": cheapest_hours(deficit, forecast, spec, deadline),
            "target_level": from_attacked.cost - from_target.cost if from_target.success else np.nan}


def score(rows):
    """Mean absolute error of each estimate, its 95% interval, and the paired difference from the debt."""
    table = pd.DataFrame(rows)
    errors = {name: (table[name] - table.realised).abs().to_numpy() for name in ESTIMATES}
    picks = rng.integers(0, len(table), (DRAWS, len(table)))  # the same resamples for every estimate
    out = {"n": len(table), "target_level_defined": int(table.target_level.notna().sum())}
    for name in ESTIMATES:
        gap = errors[name] - errors["debt"]
        out[name] = {"mae": float(errors[name].mean()), "median": float(np.median(errors[name])),
                     "max": float(errors[name].max()), "below": int((table[name] < table.realised).sum()),
                     "ci95": np.percentile(errors[name][picks].mean(axis=1), [2.5, 97.5]).tolist(),
                     "gap": float(gap.mean()),
                     "gap_ci95": np.percentile(gap[picks].mean(axis=1), [2.5, 97.5]).tolist()}
    return out


R = {}
# ---------------------------------------------------------------- synthetic test set, exact meter and meter error
spec = BatterySpec()
test = pd.read_csv(EXP / "expanded_validation/results/episodes.csv")
forecasts = {}
for r in test.to_dict("records"):
    s = Scenario(**{key: r[key] for key in Scenario.__dataclass_fields__})
    forecast = issue_forecast(s, make_profile(s), s.detection_t)
    forecasts[s.scenario_id] = (forecast, s.target_energy_kwh, s.deadline_steps)
used = test[(test.feasibility_class == "both_feasible") & test.realized_paired_success]
rows = []
for r in used.to_dict("records"):
    forecast, target, deadline = forecasts[r["scenario_id"]]
    rows.append(estimate(r, r["e_actual_detect"], forecast, spec, target, deadline))
R["test"] = score(rows)
R["test_random_variation"] = score([row for row, noise in zip(rows, used.noise_scale) if noise > 0])
above = used.e_reference_detect > used.target_energy_kwh        # reference run above the target at detection
R["test"]["reference_above_target"] = [int(above.sum()), len(used), float(used.e_reference_detect.mean())]
noise = pd.read_csv(EXP / "noise_validation/results/episodes.csv")
noise = noise[noise.actual_feasible & noise.reference_feasible & noise.realized_paired_success
              & noise.estimated_extra_cost.notna() & (noise.sigma_kwh > 0)]
R["meter_error"] = {}
for sigma, level in noise.groupby("sigma_kwh"):
    rows = []
    for r in level.to_dict("records"):
        forecast, target, deadline = forecasts[r["scenario_id"]]
        rows.append(estimate(r, r["e_actual_observed"], forecast, spec, target, deadline))
    R["meter_error"][f"{sigma:g}"] = score(rows)

# ---------------------------------------------------------------- measured profiles
data = pd.read_csv(EXP / "real_data_validation/data/real_profiles.csv")
plant = json.loads((EXP / "real_data_validation/data/data_metadata.json").read_text())["plant"]
spec = BatterySpec(capacity_kwh=plant["capacity_kwh"], min_energy_kwh=plant["min_energy_kwh"],
                   max_charge_kw=plant["max_charge_kw"], max_discharge_kw=plant["max_discharge_kw"])
measured = pd.read_csv(EXP / "real_data_validation/results/episodes.csv")
measured = measured[(measured.feasibility_class == "both_feasible") & measured.realized_paired_success
                    & measured.estimated_extra_cost.notna()]
rows = []
for r in measured.to_dict("records"):
    s = Scenario(scenario_id=r["scenario_id"], seed=r["seed"], strength=r["strength"], delay_steps=r["delay_steps"],
                 grid_cap_kw=r["grid_cap_kw"], deadline_steps=r["deadline_steps"], tariff_scale=1.0,
                 initial_energy_kwh=plant["target_energy_kwh"], target_energy_kwh=plant["target_energy_kwh"],
                 loss_allowance_kw=plant["loss_allowance_kw"])
    profile = real_profile(s, data[data.scenario_index == int(s.scenario_id[5:])])
    forecast = issue_forecast(s, profile, s.detection_t)
    rows.append(estimate(r, r["e_actual_detect"], forecast, spec, s.target_energy_kwh, s.deadline_steps))
R["measured"] = score(rows)
R["measured_changed"] = score([row for row, gap in zip(rows, measured.soc_deficit_kwh) if abs(gap) > 1e-9])

(ROOT / "assignment 3" / "build" / "stronger_comparators.json").write_text(json.dumps(R, indent=1) + "\n")
for name, part in (("test", R["test"]), ("measured", R["measured"])):
    print(name, part["n"], {key: round(part[key]["mae"], 4) for key in ESTIMATES})
