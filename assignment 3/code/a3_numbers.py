"""Recompute every number reported in Assignment 3 from the saved experiment results.

Run from the project folder (RMProj):
    experiments/.venv/bin/python "assignment 3/code/a3_numbers.py"
The script reads only saved result files; it never runs a simulation. It writes
"assignment 3/build/a3_numbers.json" and stops with an error if a recomputed value
disagrees with a summary saved by the original experiment scripts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "assignment 3" / "build"
DRAWS = 20000          # bootstrap resamples, as fixed in Assignment 2
SEED = 20261029        # seed for the new intervals computed here
ETA_C = 0.95           # charging efficiency, Table 2 of Assignment 2


def interval(values):
    """95% bootstrap interval of a mean, resampling scenarios with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(values), size=(DRAWS, len(values)))
    means = values[picks].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def mae_gap(errors_simple, errors_debt):
    """MAE(simple estimate) - MAE(debt) with its 95% paired bootstrap interval."""
    gap = np.abs(np.asarray(errors_simple)) - np.abs(np.asarray(errors_debt))
    return {"n": int(len(gap)), "difference": float(gap.mean()), "ci95": interval(gap)}


def check(name, value, saved, tol=1e-6):
    if abs(value - saved) > tol:
        raise SystemExit(f"{name}: recomputed {value} differs from saved {saved}")


R = {}

# ---------------------------------------------------------------- development set (100 scenarios)
dev = pd.read_csv(EXP / "results/main/episodes.csv")
clean = dev[dev.strength == 0]
hit = dev[dev.strength > 0]
both = hit[hit.feasibility_class == "both_feasible"]
R["dev"] = {
    "n": len(dev), "no_attack": len(clean),
    "no_attack_identical": int(((clean.soc_deficit_kwh.abs() < 1e-12) & (clean.estimated_extra_cost.abs() < 1e-12)).sum()),
    "attacked": len(hit),
    "gap_mean": float(hit.soc_deficit_kwh.mean()), "gap_min": float(hit.soc_deficit_kwh.min()),
    "gap_max": float(hit.soc_deficit_kwh.max()),
    "classes": hit.feasibility_class.value_counts().to_dict(),
    "debt_mean": float(both.estimated_extra_cost.mean()), "debt_min": float(both.estimated_extra_cost.min()),
    "debt_max": float(both.estimated_extra_cost.max()),
    "spent_less_before_detection_mean": float((hit.pre_reference_cost - hit.pre_actual_cost).mean()),
}
ex = dev[dev.scenario_id == "case_070"].iloc[0]
R["example"] = {"e_attacked": float(ex.e_actual_detect), "e_clean": float(ex.e_reference_detect),
                "debt": float(ex.estimated_extra_cost), "realised": float(ex.realized_extra_cost),
                "spent_less_before": float(ex.pre_reference_cost - ex.pre_actual_cost),
                "detection_step": int(ex.detection_t), "delay": int(ex.delay_steps)}
check("example gap", R["example"]["e_clean"] - R["example"]["e_attacked"], 7.305, tol=5e-4)

# ---------------------------------------------------------------- test set (500 scenarios)
ep = pd.read_csv(EXP / "expanded_validation/results/episodes.csv")
tp = pd.read_csv(EXP / "expanded_validation/test_predictions.csv")
saved = pd.read_csv(EXP / "expanded_validation/prediction_metrics.csv")
R["test"] = {"n": len(ep), "classes": ep.feasibility_class.value_counts().to_dict(),
             "eligible": len(tp),
             "debt_negative_among_both_feasible": int((ep[ep.feasibility_class == "both_feasible"].estimated_extra_cost < 0).sum()),
             "spent_less_before_detection_mean": float((ep.pre_reference_cost - ep.pre_actual_cost).mean())}
labels = {"counterfactual_cost": "debt", "current_price": "current_price",
          "forecast_price": "mean_price", "soc_ols": "linear_fit"}
pred = {}
for col, name in labels.items():
    err = tp[f"error_{col}"]
    row = saved[(saved.group == "all_eligible") & (saved.predictor == col)].iloc[0]
    check(f"MAE {name}", float(err.abs().mean()), row.mae)
    pred[name] = {"mae": float(err.abs().mean()), "ci95": [float(row.mae_ci95_low), float(row.mae_ci95_high)],
                  "median": float(err.abs().median()), "max": float(err.abs().max())}
R["test"]["prediction"] = pred
R["test"]["h1_gap_current_price"] = mae_gap(tp.error_current_price, tp.error_counterfactual_cost)
R["test"]["no_random_variation"] = {"n": int((tp.noise_scale == 0).sum()),
                                    "debt_mae": float(tp[tp.noise_scale == 0].error_counterfactual_cost.abs().mean())}
R["test"]["random_variation"] = {"n": int((tp.noise_scale > 0).sum()),
                                 "debt_mae": float(tp[tp.noise_scale > 0].error_counterfactual_cost.abs().mean())}

# by detection delay (Assignment 2, Section 2.6): feasibility on all 500, accuracy on the eligible cases
delay = []
for d in sorted(ep.delay_steps.unique()):
    e, t = ep[ep.delay_steps == d], tp[tp.delay_steps == d]
    bf = e.feasibility_class == "both_feasible"
    delay.append({"delay_steps": int(d), "n": len(e),
                  "attacked_cannot_recover": int((~bf).sum()),
                  "gap_mean": float(e.soc_deficit_kwh.mean()),
                  "debt_mean": float(t.estimated_extra_cost.mean()),
                  "eligible": len(t), "realised_mean": float(t.realized_extra_cost.mean()),
                  "debt_mae": float(t.error_counterfactual_cost.abs().mean()),
                  "current_price_mae": float(t.error_current_price.abs().mean()),
                  "gap_vs_current_price": mae_gap(t.error_current_price, t.error_counterfactual_cost)})
R["test"]["by_delay"] = delay

# ---------------------------------------------------------------- recovery methods on the test set
ctrl = pd.read_csv(EXP / "expanded_validation/controller_summary.csv")
comp = pd.read_csv(EXP / "expanded_validation/controller_comparisons.csv")
R["methods"] = {r.method: {"success": int(r.successful), "unmet_kwh": float(r.mean_unserved_kwh),
                           "solver_s": float(r.mean_cumulative_solver_seconds_per_scenario)}
                for r in ctrl.itertuples()}
mx = comp[(comp.baseline == "max_charge") & (comp.metric == "future_cost")].iloc[0]
ms = comp[(comp.baseline == "max_charge") & (comp.metric == "physical_recovery_success")].iloc[0]
R["methods_max_vs_mpc"] = {"cost_n": int(mx.n_paired), "mpc_cost": float(mx.reference_mean),
                           "max_cost": float(mx.baseline_mean), "cost_gap": float(mx.mean_difference_baseline_minus_reference),
                           "cost_ci95": [float(mx.ci95_low), float(mx.ci95_high)],
                           "success_gap": float(ms.mean_difference_baseline_minus_reference),
                           "success_ci95": [float(ms.ci95_low), float(ms.ci95_high)]}
dev_ctrl = pd.read_csv(EXP / "results/main/controllers.csv")
dev_hit = dev_ctrl[dev_ctrl.scenario_id.isin(hit.scenario_id)]
R["methods_dev"] = dev_hit.groupby("method").physical_recovery_success.sum().astype(int).to_dict()

# ---------------------------------------------------------------- measured data (400 scenarios)
me = pd.read_csv(EXP / "real_data_validation/results/episodes.csv")
mc = pd.read_csv(EXP / "real_data_validation/results/controllers.csv")
changed = me[me.soc_deficit_kwh.abs() > 1e-9]
use = me[(me.feasibility_class == "both_feasible") & me.realized_paired_success & me.estimated_extra_cost.notna()].copy()
use["err_debt"] = use.estimated_extra_cost - use.realized_extra_cost
use["err_price"] = use.soc_deficit_kwh * use.current_price / use.eta_charge - use.realized_extra_cost
use_changed = use[use.soc_deficit_kwh.abs() > 1e-9]
no_neg = use[~use.has_floored_price]
R["measured"] = {"n": len(me), "changed": len(changed), "attacked_fuller": int((changed.soc_deficit_kwh < 0).sum()),
                 "debt_negative": int((use.estimated_extra_cost < -1e-9).sum()),
                 "eligible": len(use), "debt_mae": float(use.err_debt.abs().mean()),
                 "current_price_mae": float(use.err_price.abs().mean()),
                 "gap_vs_current_price": mae_gap(use.err_price, use.err_debt),
                 "changed_eligible": len(use_changed),
                 "changed_debt_mae": float(use_changed.err_debt.abs().mean()),
                 "changed_current_price_mae": float(use_changed.err_price.abs().mean()),
                 "changed_gap": mae_gap(use_changed.err_price, use_changed.err_debt),
                 "no_negative_price": len(no_neg), "no_negative_price_debt_mae": float(no_neg.err_debt.abs().mean()),
                 "success": mc.groupby("method").physical_recovery_success.sum().astype(int).to_dict()}
saved_real = json.loads((EXP / "real_data_validation/results/summary_real.json").read_text())
check("measured MAE", R["measured"]["debt_mae"], saved_real["mae"], tol=6e-6)
s = mc.pivot(index="scenario_id", columns="method", values="physical_recovery_success")
cost = mc.pivot(index="scenario_id", columns="method", values="future_cost")
both_ok = s.state_mpc & s.max_charge
gap = (cost.max_charge - cost.state_mpc)[both_ok]
R["measured"]["max_vs_mpc_cost"] = {"n": int(both_ok.sum()), "mpc": float(cost.state_mpc[both_ok].mean()),
    "max": float(cost.max_charge[both_ok].mean()), "gap": float(gap.mean()), "ci95": interval(gap)}
check("measured price MAE", R["measured"]["current_price_mae"], saved_real["baseline_current"], tol=6e-6)
check("measured cost gap", float(gap.mean()), saved_real["maxchg_minus_mpc"], tol=6e-6)
# ---------------------------------------------------------------- sensor error (500 scenarios x 5 levels)
ne = pd.read_csv(EXP / "expanded_validation/results/episodes.csv")[["scenario_id", "current_price"]]
noise = pd.read_csv(EXP / "noise_validation/results/episodes.csv").merge(ne, on="scenario_id")
summ = json.loads((EXP / "noise_validation/results/summary_noise.json").read_text())
resv = json.loads((EXP / "noise_validation/results/summary_reserve.json").read_text())
sens = []
for sigma, g in noise.groupby("sigma_kwh"):
    g = g[g.actual_feasible & g.reference_feasible & g.realized_paired_success & g.estimated_extra_cost.notna()]
    err_debt = g.estimated_extra_cost - g.realized_extra_cost
    # The simple estimate prices the energy gap the controller sees: clean level minus the meter reading.
    err_price = (g.e_reference_detect - g.e_actual_observed) * g.current_price / ETA_C - g.realized_extra_cost
    key = f"{sigma:g}"
    key = {"0": "0.0"}.get(key, key)
    check(f"sensor MAE {sigma}", float(err_debt.abs().mean()), summ[key]["mae"], tol=6e-6)
    check(f"sensor baseline {sigma}", float(err_price.abs().mean()), summ[key]["baseline_current"], tol=6e-6)
    sens.append({"sigma": float(sigma), "eligible": len(g), "debt_mae": float(err_debt.abs().mean()),
                 "current_price_mae": float(err_price.abs().mean()), "gap": mae_gap(err_price, err_debt),
                 "identical_actions": int(summ[key]["identical_actions"]),
                 "state_mpc_success": int(summ[key]["mpc_success"]), "max_charge_success": int(summ[key]["maxchg_success"]),
                 "reserve": resv[key]})
R["sensor"] = sens

# money saved by the planning controller with a reserve of twice the meter error, against charging at full
# power, on scenarios where both recover fully
plain = pd.read_csv(EXP / "noise_validation/results/controllers.csv")
plain = plain[plain.method == "max_charge"]
margin = pd.read_csv(EXP / "noise_validation/results/reserve_controllers.csv")
margin = margin[margin.rule == "reserve_2sigma"]
saved_saving = {0.0: 2.033, 0.06: 2.006, 0.12: 1.985, 0.3: 1.883, 0.6: 1.770}
for entry in sens:
    a = plain[np.isclose(plain.sigma_kwh, entry["sigma"])].set_index("scenario_id")
    b = margin[np.isclose(margin.sigma_kwh, entry["sigma"])].set_index("scenario_id")
    ok = a.physical_recovery_success & b.physical_recovery_success.reindex(a.index)
    saving = (a.future_cost - b.future_cost.reindex(a.index))[ok]
    check(f"reserve saving {entry['sigma']}", float(saving.mean()), saved_saving[entry["sigma"]], tol=6e-4)
    entry["reserve_saving"] = {"n": int(ok.sum()), "mean": float(saving.mean()), "ci95": interval(saving)}

# ---------------------------------------------------------------- reference sensitivity, sweeps, network check
R["reference_shift"] = pd.read_csv(EXP / "results/main/reference_sensitivity_summary.csv").to_dict(orient="records")
extra = json.loads((EXP / "results/main/extra_summary.json").read_text())
R["sweeps"] = {"grid_deadline_cells": extra["controlled_states"]["n_feasibility_cells"],
               "grid_deadline_classes": extra["controlled_states"]["feasibility_counts"],
               "equal_gap_cells": extra["controlled_states"]["n_equal_deficit_cells"],
               "equal_gap_min": extra["controlled_states"]["equal_deficit_finite_min_usd"],
               "equal_gap_max": extra["controlled_states"]["equal_deficit_finite_max_usd"],
               "earliest_deadline_h": extra["controlled_states"]["earliest_feasible_deadline_hours_by_cap"],
               "clipped_shifts": extra["reference_sensitivity"]["n_clipped_perturbations"]}
ac = pd.read_csv(EXP / "results/main/ac_validation.csv", low_memory=False)
on = ac[ac.ac_status == "checked_grid_connected"]
R["network"] = {"snapshots": len(ac), "grid_connected": len(on), "passed": int(on.ac_passed.sum()),
                "outage_unchecked": int((ac.ac_status != "checked_grid_connected").sum()),
                "v_min": float(on.ac_min_voltage_pu.min()), "v_max": float(on.ac_max_voltage_pu.max()),
                "line_max": float(on.ac_max_line_loading_percent.max()),
                "trafo_max": float(on.ac_max_trafo_loading_percent.max()), "loss_max": float(on.ac_loss_kw.max())}

# ---------------------------------------------------------------- language-model test (300 scenarios x 4 policies)
llm = "llm_agent_validation/expanded/results/"
R["llm"] = {"outcomes": pd.read_csv(EXP / (llm + "agent_outcomes.csv")).iloc[:, :12].to_dict(orient="records"),
            "contrasts": pd.read_csv(EXP / (llm + "paired_agent_contrasts.csv")).iloc[:, :8].to_dict(orient="records"),
            "prediction": pd.read_csv(EXP / (llm + "prediction_metrics.csv")).iloc[:, :7].to_dict(orient="records")}

# ---------------------------------------------------------------- target reached by the deadline (Table 3)
runs = pd.read_csv(EXP / "expanded_validation/results/controllers.csv")       # one row per scenario and method
deadline = runs.groupby("method").deadline_success.sum().astype(int).to_dict()
recovered = runs.groupby("method").physical_recovery_success.sum().astype(int).to_dict()
for row in json.loads((EXP / "expanded_validation/controls_summary.json").read_text())["outcomes"]:
    check(f"deadline target {row['method']}", deadline[row["method"]], row["deadline_success_count"])
    check(f"full recoveries {row['method']}", recovered[row["method"]], row["recovery_success_count"])
mpc = runs[runs.method == "state_mpc"]
end_only = mpc.deadline_success & ~mpc.physical_recovery_success & (mpc.future_unserved_kwh <= 1e-6)
R["deadline"] = {"test": deadline, "state_mpc_missed_only_end_target": int(end_only.sum())}

# ---------------------------------------------------------------- whole-period cost, unrecoverable cases, feasibility counts
used = ep.set_index("scenario_id").loc[tp.scenario_id]                  # the 329 scenarios used for prediction
before = used.pre_actual_cost - used.pre_reference_cost
R["test"]["whole_period"] = {"before_detection": float(before.mean()),
                             "after_detection": float(used.realized_extra_cost.mean()),
                             "total": float((before + used.realized_extra_cost).mean())}
bad = ep[ep.feasibility_class != "both_feasible"]
good = ep[ep.feasibility_class == "both_feasible"]
lost = mpc[mpc.scenario_id.isin(bad.scenario_id)]                      # State MPC where the attacked run cannot recover
late = lost[lost.time_to_target_steps.notna() & ~lost.deadline_success]
R["test"]["unrecoverable"] = {"n": len(bad), "shortfall_mean": float(lost.target_shortfall_kwh.mean()),
                              "shortfall_max": float(lost.target_shortfall_kwh.max()),
                              "unmet_mean": float(lost.future_unserved_kwh.mean()),
                              "full_recoveries": int(lost.physical_recovery_success.sum()),
                              "target_after_deadline": len(late),
                              "steps_to_target_after_deadline": float(late.time_to_target_steps.mean()),
                              "target_never": int(lost.time_to_target_steps.isna().sum()),
                              "deficit_range": [float(bad.soc_deficit_kwh.min()), float(bad.soc_deficit_kwh.max())],
                              "recoverable_deficit_range": [float(good.soc_deficit_kwh.min()), float(good.soc_deficit_kwh.max())],
                              "recoverable_at_least_smallest": int((good.soc_deficit_kwh >= bad.soc_deficit_kwh.min()).sum())}
R["measured"]["classes"] = me.feasibility_class.value_counts().to_dict()
for name, count in saved_real["feasibility"].items():
    check(f"measured class {name}", R["measured"]["classes"][name], count)
both = noise[noise.actual_feasible & noise.reference_feasible].groupby("sigma_kwh").size()
R["sensor_both_feasible"] = {f"{k:g}": int(v) for k, v in both.items()}

# ---------------------------------------------------------------- further numbers quoted in the text
R["example"]["strength"] = float(ex.strength)                          # 0.8: future prices at a fifth
# identical actions of the three MPC methods in every recovery step of the test set
traces = pd.read_csv(EXP / "expanded_validation/results/traces.csv").sort_values(["scenario_id", "t"])
cols = ["requested_charge_kw", "requested_discharge_kw", "charge_kw", "discharge_kw",
        "energy_after_kwh", "grid_import_kw", "unserved_kw", "step_cost"]
mpc_steps = traces[traces.method == "state_mpc"]
saved_steps = json.loads((EXP / "expanded_validation/controls_summary.json").read_text())["action_equivalence"]
for method in ("attribution_mpc", "reset_measured"):
    other = traces[traces.method == method]
    same_steps = (other.scenario_id.values == mpc_steps.scenario_id.values).all() and (other.t.values == mpc_steps.t.values).all()
    largest = float(np.abs(other[cols].to_numpy() - mpc_steps[cols].to_numpy()).max())
    check(f"identical actions {method}", largest + (0 if same_steps else 1), 0.0, tol=1e-9)
    check(f"compared steps {method}", len(other), saved_steps[method]["compared_actions"])
R["test"]["compared_steps"] = len(mpc_steps)
# Fig. 2: scenarios in which the price-at-detection estimate is below 70% of the realised extra cost
low = tp[tp.prediction_current_price < 0.7 * tp.realized_extra_cost]
R["test"]["low_group"] = {"n": len(low), "detection_steps": sorted(int(x) for x in low.detection_t.unique()),
                          "price_at_detection": float(low.current_price.mean()),
                          "cost_per_kwh_bought": float((low.realized_extra_cost * ETA_C / low.soc_deficit_kwh).mean())}
# Table II: differences in MAE for the other two estimates, checked against the saved analysis
saved_gaps = pd.read_csv(EXP / "expanded_validation/paired_mae_improvements.csv")
saved_gaps = saved_gaps[saved_gaps.group == "all_eligible"].set_index("baseline")
for col, name in (("forecast_price", "mean_price"), ("soc_ols", "linear_fit")):
    R["test"][f"gap_{name}"] = mae_gap(tp[f"error_{col}"], tp.error_counterfactual_cost)
    check(f"gap {name}", R["test"][f"gap_{name}"]["difference"], saved_gaps.loc[col, "mae_improvement_baseline_minus_reference"])
# meter error: where State MPC and Maximum charging end, per error level
runs_noise = pd.read_csv(EXP / "noise_validation/results/controllers.csv")
target = float(tp.target_energy_kwh.iloc[0])                           # 9 kWh in every synthetic scenario
for entry in sens:
    key = {"0": "0.0"}.get(f"{entry['sigma']:g}", f"{entry['sigma']:g}")
    level = runs_noise[np.isclose(runs_noise.sigma_kwh, entry["sigma"])]
    mpc_level, max_level = level[level.method == "state_mpc"], level[level.method == "max_charge"]
    failed = mpc_level[~mpc_level.physical_recovery_success]
    miss = np.maximum(target - failed.deadline_energy_kwh, target - failed.terminal_energy_kwh)
    entry["end_levels"] = {"state_mpc_median": float(mpc_level.terminal_energy_kwh.median()),
                           "state_mpc_below_target": float((mpc_level.terminal_energy_kwh < target - 1e-6).mean()),
                           "failures_missing_by_under_0.1": float((miss < 0.1).mean()),
                           "max_charge_lowest": float(max_level.terminal_energy_kwh.min())}
    check(f"median end level {key}", entry["end_levels"]["state_mpc_median"] - target, summ[key]["mpc_terminal_median_dev"], tol=6e-6)
    check(f"lowest end level {key}", entry["end_levels"]["max_charge_lowest"], summ[key]["maxchg_terminal_min"], tol=6e-5)
# Fig. 4: with a 6 kW limit, the deadlines (hours) at which the attacked run can recover
sweep = pd.read_csv(EXP / "results/main/feasibility_sweep.csv")
six = sweep[np.isclose(sweep.grid_cap_kw, 6.0)]
R["sweeps"]["six_kw_feasible_hours"] = [float(h) for h in six[six.extra_cost.notna()].deadline_hours]
# language-model test: decision steps, tool use, deficits and the failures of the reference runs
choices = pd.read_csv(EXP / (llm + "tool_choices.csv"))
pairs = pd.read_csv(EXP / (llm + "paired_episodes.csv"))
saved_outcomes = pd.read_csv(EXP / (llm + "agent_outcomes.csv")).set_index("arm")
saved_contrast = pd.read_csv(EXP / (llm + "paired_agent_contrasts.csv")).set_index(["minuend_arm", "subtrahend_arm"])
model_steps = choices[choices.arm.str.startswith("llm_")]
tool = choices[(choices.arm == "llm_available") & (choices.condition == "poisoned")].iloc[0]
deficit = pairs.pivot(index="scenario_id", columns="arm", values="energy_deficit_kwh")
with_tool = pairs[pairs.arm == "llm_available"]
reference_failed = with_tool[~with_tool.clean_physical_recovery_success]
gap_change = saved_contrast.loc[("llm_unavailable", "llm_available")]
R["llm_text"] = {"decision_steps": int(model_steps.decision_cycles.sum()), "holds": int(model_steps.hold_or_failure_cycles.sum()),
                 "verified_steps": int(tool.verified_execution_cycles), "attacked_steps": int(tool.decision_cycles),
                 "verified_fraction": [float(tool[f"verified_execution_scenario_fraction_{x}"]) for x in ("mean", "ci95_low", "ci95_high")],
                 "mean_deficit": {arm: [float(deficit[arm].mean())] + [float(saved_outcomes.loc[arm, f"signed_deficit_kwh_{x}"])
                                  for x in ("ci95_low", "ci95_high")] for arm in deficit.columns},
                 "deficit_by_strength_with_tool": {f"{k:g}": float(v) for k, v in with_tool.groupby("strength").energy_deficit_kwh.mean().items()},
                 "deficit_reduction": [float((deficit.llm_unavailable - deficit.llm_available).mean()),
                                       float(gap_change.signed_deficit_difference_kwh_ci95_low), float(gap_change.signed_deficit_difference_kwh_ci95_high)],
                 "debt_reduction": [float(gap_change[f"estimated_extra_cost_difference_usd_{x}"]) for x in ("mean", "ci95_low", "ci95_high")],
                 "reference_failures": len(reference_failed),
                 "reference_end_shortfall": [float((target - reference_failed.clean_final_energy_kwh).min()),
                                             float((target - reference_failed.clean_final_energy_kwh).max())]}
R["llm_text"]["reference_failures_served_deadline_met"] = int(((reference_failed.clean_future_unserved_kwh <= 1e-6)
                                                               & reference_failed.clean_deadline_success).sum())
for arm in deficit.columns:
    check(f"mean deficit {arm}", R["llm_text"]["mean_deficit"][arm][0], saved_outcomes.loc[arm, "signed_deficit_kwh_mean"])
check("deficit reduction", R["llm_text"]["deficit_reduction"][0], gap_change.signed_deficit_difference_kwh_mean)


def rate_gap(first, second):
    """Paired difference of two success rates (first minus second) with its 95% bootstrap interval."""
    diff = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return {"n": len(diff), "difference": float(diff.mean()), "ci95": interval(diff)}


# practical size of the errors, and the scenarios with random variation, where agreement is not built in
R["test"]["mean_realised"] = float(tp.realized_extra_cost.mean())
noisy = tp[tp.noise_scale > 0]
R["test"]["random_variation"]["current_price_mae"] = float(noisy.error_current_price.abs().mean())
R["test"]["random_variation"]["gap"] = mae_gap(noisy.error_current_price, noisy.error_counterfactual_cost)
# how well the energy deficit alone separates unrecoverable from recoverable test scenarios (area under the ROC curve)
bad_flag = (ep.feasibility_class != "both_feasible").to_numpy()
deficits = ep.soc_deficit_kwh.to_numpy()
rng = np.random.default_rng(SEED)
def auc(flag, score):
    ranks = pd.Series(score).rank().to_numpy()
    n_bad, n_good = flag.sum(), (~flag).sum()
    return (ranks[flag].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_good)
draws = [auc(bad_flag[i], deficits[i]) for i in (rng.integers(0, len(ep), len(ep)) for _ in range(DRAWS))]
R["test"]["deficit_auc"] = {"auc": float(auc(bad_flag, deficits)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
# paired differences in success behind statements in the text
by_method = runs.pivot(index="scenario_id", columns="method", values="deadline_success")
R["methods_deadline_gap"] = rate_gap(by_method.max_charge, by_method.state_mpc)          # 360 against 387
R["measured"]["max_vs_mpc_success"] = rate_gap(s.max_charge, s.state_mpc)                 # 361 against 353
mpc_by_level = runs_noise[runs_noise.method == "state_mpc"].pivot(index="scenario_id", columns="sigma_kwh",
                                                                  values="physical_recovery_success")
R["sensor_mpc_drop"] = rate_gap(mpc_by_level[0.06], mpc_by_level[0.0])                     # 331 to 118
margin_ok = margin.pivot(index="scenario_id", columns="sigma_kwh", values="physical_recovery_success")
full_ok = runs_noise[runs_noise.method == "max_charge"].pivot(index="scenario_id", columns="sigma_kwh",
                                                               values="physical_recovery_success")
for entry in sens:
    entry["reserve_vs_max_success"] = rate_gap(margin_ok[entry["sigma"]], full_ok[entry["sigma"]].reindex(margin_ok.index))
    level = runs_noise[(runs_noise.method == "state_mpc") & np.isclose(runs_noise.sigma_kwh, entry["sigma"])]
    short = np.maximum(target - level.deadline_energy_kwh, target - level.terminal_energy_kwh)
    entry["state_mpc_success_within_0.1"] = int(((short < 0.1) & (level.future_unserved_kwh <= 1e-6)).sum())
# measured profiles: correlated neighbouring windows, resampled in blocks of five consecutive scenarios
blocks = use.scenario_id.str[5:].astype(int).to_numpy() // 5
gaps = (use.err_price.abs() - use.err_debt.abs()).to_numpy()
block_ids = np.unique(blocks)
means = []
for _ in range(DRAWS):
    chosen = rng.choice(block_ids, len(block_ids))
    picked = np.concatenate([gaps[blocks == b] for b in chosen])
    means.append(picked.mean())
R["measured"]["block_bootstrap_gap"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
# measured profiles: negative deficits, and Software replay split by whether the attack changed the battery
fuller = use[use.soc_deficit_kwh < -1e-9]
replay = mc[mc.method == "software_replay"].merge(me[["scenario_id", "soc_deficit_kwh"]], on="scenario_id")
moved = replay.soc_deficit_kwh.abs() > 1e-9
R["measured"]["fuller_used"] = {"n": len(fuller), "debt_negative": int((fuller.estimated_extra_cost < -1e-9).sum()),
                                "debt_zero": int((fuller.estimated_extra_cost.abs() <= 1e-9).sum())}
R["measured"]["software_replay"] = {"unchanged": [int(replay[~moved].physical_recovery_success.sum()), int((~moved).sum())],
                                    "changed": [int(replay[moved].physical_recovery_success.sum()), int(moved.sum())]}
# reference shifts: only the upward shifts can be cut off at the battery capacity
shifts = pd.read_csv(EXP / "results/main/reference_sensitivity_summary.csv")
R["sweeps"]["upward_shifts"] = int(shifts[shifts.reference_error_fraction_capacity > 0].n.sum())
# language model: share of attacked steps executed with verified prices, by attack strength
per_scenario = pd.read_csv(EXP / (llm + "tool_choices_by_scenario.csv"))
per_scenario = per_scenario[(per_scenario.arm == "llm_available") & (per_scenario.condition == "poisoned")]
per_scenario = per_scenario.merge(with_tool[["seed", "strength"]], on="seed")
per_scenario["verified_steps"] = per_scenario.verified_execution * per_scenario.decision_cycles   # share times steps
share = per_scenario.groupby("strength")[["verified_steps", "decision_cycles"]].sum()
R["llm_text"]["verified_share_by_strength"] = {f"{k:g}": float(r.verified_steps / r.decision_cycles) for k, r in share.iterrows()}
check("verified steps by strength", float(share.verified_steps.sum()), R["llm_text"]["verified_steps"], tol=1e-6)
# the other deficit-based estimate (missing energy times the mean remaining price) on measured data and under meter
# error, to show which of the two is the better one in each data set; checked against the saved summaries
mean_prices = pd.read_csv(EXP / "expanded_validation/results/episodes.csv")[["scenario_id", "mean_forecast_price"]]
use_mean = use.soc_deficit_kwh * use.mean_forecast_price / use.eta_charge - use.realized_extra_cost
R["measured"]["mean_price_mae"] = float(use_mean.abs().mean())
check("measured mean-price MAE", R["measured"]["mean_price_mae"], 0.11692, tol=6e-6)      # value in real_data_validation/README.md
noise_mean = noise.merge(mean_prices, on="scenario_id")
for entry in sens:
    key = {"0": "0.0"}.get(f"{entry['sigma']:g}", f"{entry['sigma']:g}")
    g = noise_mean[np.isclose(noise_mean.sigma_kwh, entry["sigma"])]
    g = g[g.actual_feasible & g.reference_feasible & g.realized_paired_success & g.estimated_extra_cost.notna()]
    err = (g.e_reference_detect - g.e_actual_observed) * g.mean_forecast_price / ETA_C - g.realized_extra_cost
    entry["mean_price_mae"] = float(err.abs().mean())
    check(f"sensor mean-price MAE {key}", entry["mean_price_mae"], summ[key]["baseline_meanfc"], tol=6e-6)
# the stored plans: checksum and freezing time of each experiment
R["plans"] = {}
for name, folder in (("test", "expanded_validation"), ("meter_error", "noise_validation"),
                     ("measured", "real_data_validation"), ("language_model", "llm_agent_validation/expanded")):
    plan = json.loads((EXP / folder / "protocol.json").read_text())
    R["plans"][name] = {"sha256": (EXP / folder / "protocol.sha256").read_text().split()[0],
                        "frozen_utc": plan.get("frozen_utc", plan.get("frozen_at_utc"))}

# measured profiles: purchases before detection in the scenarios in which the attacked run held more energy
fuller_all = me[me.soc_deficit_kwh < -1e-9]
bought = (fuller_all.pre_actual_cost.abs() > 1e-9) | (fuller_all.pre_reference_cost.abs() > 1e-9)
R["measured"]["fuller_no_purchase"] = [int((~bought).sum()), len(fuller_all)]                     # 97 of 97
# test set: did the State MPC scenarios that missed only the end target have random variation in demand and solar output
varied = mpc[end_only].merge(ep[["scenario_id", "noise_scale"]], on="scenario_id").noise_scale > 0
R["deadline"]["end_target_only_with_random_variation"] = [int(varied.sum()), len(varied)]             # 56 of 56

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "a3_numbers.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
print(json.dumps({k: R[k] for k in ("dev", "example")}, indent=1, default=float))
print("All recomputed values agree with the saved summaries.")
