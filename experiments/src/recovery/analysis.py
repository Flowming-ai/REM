"""Paired analysis and exportable figures for the recovery experiments.

No failed run is rewarded for its artificially low purchase cost. Cost
comparisons use the same scenarios on which both compared controllers succeed.
Prediction uses an untouched seed split, with one training-only linear model
and three fixed, unfitted cost predictors.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


METHODS = [
    "attribution_mpc", "state_mpc", "software_replay", "reset_measured",
    "max_charge", "safe_stop",
]
LABELS = {
    "attribution_mpc": "Attribution + MPC",
    "state_mpc": "Measured state + MPC",
    "software_replay": "Software replay",
    "reset_measured": "Reset + measured state",
    "max_charge": "Maximum charging",
    "safe_stop": "Safe stop",
}
COLORS = dict(zip(METHODS, ["#0072B2", "#222222", "#D55E00", "#009E73", "#CC79A7", "#A87800"]))
PREDICTORS = {
    "counterfactual_cost": "Counterfactual cost",
    "soc_ols": "SoC deficit: train-only OLS",
    "current_price": "Deficit / efficiency × current price",
    "forecast_price": "Deficit / efficiency × mean forecast price",
}


def _bool_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = normalized.isin(["true", "false", "1", "0", "1.0", "0.0", "nan", "none"])
    if not allowed.all():
        raise ValueError(f"Invalid boolean values in {series.name}.")
    return normalized.isin(["true", "1", "1.0"])


def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI of a paired difference; each draw resamples whole scenarios."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def _save(fig, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _paired_controller_analysis(controllers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if controllers.duplicated(["scenario_id", "method"]).any():
        raise ValueError("Each scenario/method must occur once in controllers.csv.")
    methods = [m for m in METHODS if m in set(controllers.method)]
    successful = controllers.pivot(index="scenario_id", columns="method", values="physical_recovery_success")
    complete = successful.reindex(columns=methods).notna().all(axis=1)
    common_ids = successful.index[complete & successful.reindex(columns=methods).fillna(False).all(axis=1)]
    rows = []
    for method in methods:
        g = controllers[controllers.method == method]
        common = g[g.scenario_id.isin(common_ids)]
        successes = g[g.physical_recovery_success]
        rows.append({
            "method": method, "n_scenarios": len(g),
            "physical_recovery_success_rate": g.physical_recovery_success.mean(),
            "deadline_success_rate": g.deadline_success.mean(),
            "future_unserved_kwh_mean": g.future_unserved_kwh.mean(),
            "future_unserved_kwh_median": g.future_unserved_kwh.median(),
            "target_shortfall_kwh_mean": g.target_shortfall_kwh.mean(),
            "requested_clipped_kwh_mean": g.requested_clipped_kwh.mean(),
            "time_to_target_steps_successful_mean": successes.time_to_target_steps.mean(),
            "n_successful_for_time": len(successes),
            "cost_common_all_methods_success_mean": common.future_cost.mean(),
            "n_common_all_methods_success": len(common),
        })
    rng = np.random.default_rng(20260924)
    proposed = controllers[controllers.method == "attribution_mpc"].set_index("scenario_id")
    paired_rows = []
    for method in methods:
        if method == "attribution_mpc":
            continue
        baseline = controllers[controllers.method == method].set_index("scenario_id")
        pair = proposed.join(baseline, how="inner", lsuffix="_proposed", rsuffix="_baseline")
        common_success = pair.physical_recovery_success_proposed & pair.physical_recovery_success_baseline
        for metric in ["future_cost", "future_unserved_kwh", "physical_recovery_success", "target_shortfall_kwh", "requested_clipped_kwh"]:
            selected = pair[common_success] if metric == "future_cost" else pair
            p = selected[f"{metric}_proposed"].to_numpy(dtype=float)
            b = selected[f"{metric}_baseline"].to_numpy(dtype=float)
            keep = np.isfinite(p) & np.isfinite(b)
            p, b = p[keep], b[keep]
            difference = b - p
            lower, upper = _bootstrap_mean(difference, rng)
            paired_rows.append({
                "baseline": method, "reference_method": "attribution_mpc", "metric": metric,
                "cohort": "both_methods_physical_recovery_success" if metric == "future_cost" else "all_matched_scenarios",
                "n_paired": len(difference), "n_scenarios_available": len(pair),
                "mean_proposed": float(p.mean()) if len(p) else np.nan,
                "mean_baseline": float(b.mean()) if len(b) else np.nan,
                "mean_difference_baseline_minus_proposed": float(difference.mean()) if len(difference) else np.nan,
                "ci95_low": lower, "ci95_high": upper,
                "bootstrap_unit": "scenario", "bootstrap_draws": 2000,
                "small_sample": len(difference) < 10,
            })
    summary = {
        "cost_comparison_rule": "Same scenario, both methods physical_recovery_success; failed cases are not compared on purchase cost.",
        "n_all_method_common_success": len(common_ids),
        "bootstrap": {"unit": "scenario", "draws": 2000, "seed": 20260924, "interval": "percentile 95%; descriptive sensitivity interval, no multiplicity-adjusted hypothesis tests"},
    }
    return pd.DataFrame(rows), pd.DataFrame(paired_rows), summary


def _prediction_analysis(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = ["realized_extra_cost", "estimated_extra_cost", "soc_deficit_kwh", "current_price", "mean_forecast_price", "eta_charge"]
    finite = np.isfinite(episodes[required].to_numpy(dtype=float)).all(axis=1)
    positive_efficiency = episodes.eta_charge.to_numpy(dtype=float) > 0
    eligible = episodes.realized_paired_success & finite & positive_efficiency
    frame = episodes.loc[eligible].copy()
    frame["heldout"] = frame.seed.astype(int) % 5 == 0
    train = frame[~frame.heldout]
    test = frame[frame.heldout].copy()
    coefficients = None
    if len(train):
        x = np.column_stack([np.ones(len(train)), train.soc_deficit_kwh.to_numpy(float)])
        coefficients = np.linalg.lstsq(x, train.realized_extra_cost.to_numpy(float), rcond=None)[0]
    test["prediction_counterfactual_cost"] = test.estimated_extra_cost
    test["prediction_soc_ols"] = np.nan if coefficients is None else coefficients[0] + coefficients[1] * test.soc_deficit_kwh
    test["prediction_current_price"] = test.soc_deficit_kwh / test.eta_charge * test.current_price
    test["prediction_forecast_price"] = test.soc_deficit_kwh / test.eta_charge * test.mean_forecast_price
    if len(test) and coefficients is not None:
        shared = np.isfinite(test[[f"prediction_{key}" for key in PREDICTORS]].to_numpy(float)).all(axis=1)
        test = test[shared].copy()
    records = []
    actual = test.realized_extra_cost.to_numpy(float)
    for key, label in PREDICTORS.items():
        predicted = test[f"prediction_{key}"].to_numpy(float)
        keep = np.isfinite(actual) & np.isfinite(predicted)
        y, yhat = actual[keep], predicted[keep]
        residual = yhat - y
        denom = float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0
        rho = float(spearmanr(y, yhat).statistic) if len(y) >= 2 and np.ptp(y) > 1e-12 and np.ptp(yhat) > 1e-12 else np.nan
        records.append({
            "predictor": key, "label": label, "n_test": len(y), "n_train": len(train),
            "mae": float(np.abs(residual).mean()) if len(y) else np.nan,
            "rmse": float(np.sqrt(np.mean(residual ** 2))) if len(y) else np.nan,
            "r2": float(1 - np.sum(residual ** 2) / denom) if len(y) >= 2 and denom > 1e-12 else np.nan,
            "spearman": rho, "small_sample": len(y) < 10,
            "fitted_on_training_only": key == "soc_ols",
        })
    details = {
        "target": "realized_extra_cost", "split": "heldout iff seed % 5 == 0; all other seeds train",
        "cohort": "realized_paired_success; finite target and all fixed predictor inputs; eta_charge > 0",
        "n_total": len(episodes), "n_eligible": len(frame), "n_train": len(train), "n_heldout": len(test),
        "n_excluded_failed_pair": int((~episodes.realized_paired_success).sum()),
        "n_excluded_nonfinite_or_efficiency": int((episodes.realized_paired_success & ~(finite & positive_efficiency)).sum()),
        "training_seeds": sorted(train.seed.astype(int).unique().tolist()),
        "heldout_seeds": sorted(test.seed.astype(int).unique().tolist()),
        "soc_ols_intercept": None if coefficients is None else float(coefficients[0]),
        "soc_ols_slope": None if coefficients is None else float(coefficients[1]),
        "limitations": "Exploratory fixed seed holdout; no calibration of the raw metric or price baselines. Small samples are flagged, and undefined R2/correlation is left missing.",
    }
    return pd.DataFrame(records), test, details


def _plot_controllers(summary: pd.DataFrame, pairs: pd.DataFrame, directory: Path, cohort: str = "All scenarios") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    fig.suptitle(f"{cohort} (n={int(summary.n_scenarios.max()) if len(summary) else 0})", fontsize=11)
    costs = pairs[pairs.metric == "future_cost"]
    if len(costs):
        for y, row in enumerate(costs.itertuples()):
            if row.n_paired:
                mean = row.mean_difference_baseline_minus_proposed
                axes[0, 0].plot([row.ci95_low, row.ci95_high], [y, y], color=COLORS[row.baseline], lw=2)
                axes[0, 0].plot(mean, y, "o", color=COLORS[row.baseline])
        axes[0, 0].set_yticks(range(len(costs)), [f"{LABELS[r.baseline]} (n={r.n_paired})" for r in costs.itertuples()])
        axes[0, 0].axvline(0, color="0.4", lw=0.8)
        axes[0, 0].invert_yaxis()
    axes[0, 0].set(title="(a) Paired cost: both controllers succeed", xlabel="Baseline cost − attribution + MPC cost ($)")
    methods = summary.method.tolist()
    names = [LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    y = np.arange(len(methods))
    for ax, column, title, xlabel, factor in [
        (axes[0, 1], "future_unserved_kwh_mean", "(b) Missed service: all cases in cohort", "Mean unserved energy (kWh)", 1),
        (axes[1, 0], "physical_recovery_success_rate", "(c) Complete recovery: all cases in cohort", "Successful scenarios (%)", 100),
        (axes[1, 1], "time_to_target_steps_successful_mean", "(d) First target crossing: successful only", "Mean time from detection (steps)", 1),
    ]:
        ax.barh(y, summary[column].to_numpy(float) * factor, color=colors, alpha=0.85)
        labels = names
        if column == "time_to_target_steps_successful_mean":
            labels = [f"{n} (n={count})" for n, count in zip(names, summary.n_successful_for_time)]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set(title=title, xlabel=xlabel)
        ax.grid(axis="x", alpha=0.2)
    axes[1, 0].set_xlim(0, 105)
    _save(fig, directory, "controller_comparison")


def _plot_prediction(metrics: pd.DataFrame, heldout: pd.DataFrame, directory: Path, cohort: str = "All scenarios") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    fig.suptitle(cohort, fontsize=11)
    if len(heldout):
        actual = heldout.realized_extra_cost.to_numpy(float)
        prediction = heldout.prediction_counterfactual_cost.to_numpy(float)
        dots = axes[0].scatter(actual, prediction, c=heldout.soc_deficit_kwh, cmap="viridis", s=28, alpha=0.75, edgecolors="none")
        bounds = [min(actual.min(), prediction.min()), max(actual.max(), prediction.max())]
        if bounds[0] == bounds[1]:
            bounds = [bounds[0] - 0.1, bounds[1] + 0.1]
        axes[0].plot(bounds, bounds, "--", color="0.4", lw=1, label="Equal predicted and realized cost")
        fig.colorbar(dots, ax=axes[0], label="SoC deficit at detection (kWh)")
        axes[0].legend(fontsize=7, loc="best")
    else:
        axes[0].text(0.5, 0.5, "No eligible held-out pairs", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set(title=f"(a) Raw model prediction: held-out n={len(heldout)}", xlabel="Realized paired extra cost ($)", ylabel="Estimated extra cost ($)")
    plot_labels = ["Counterfactual cost", "SoC deficit: OLS", "Current-price estimate", "Mean-forecast-price estimate"]
    axes[1].barh(np.arange(len(metrics)), metrics.mae, color=["#0072B2", "#999999", "#D55E00", "#009E73"])
    axes[1].set_yticks(np.arange(len(metrics)), [f"{label} (n={n})" for label, n in zip(plot_labels, metrics.n_test)])
    axes[1].invert_yaxis()
    axes[1].set(title="(b) Same held-out cohort; no test-set fitting", xlabel="Mean absolute prediction error ($)")
    axes[1].grid(axis="x", alpha=0.2)
    _save(fig, directory, "heldout_prediction")


def _plot_cost_soc(episodes: pd.DataFrame, directory: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    finite = np.isfinite(episodes.estimated_extra_cost) & np.isfinite(episodes.soc_deficit_kwh)
    matched = episodes.actual_feasible & episodes.reference_feasible & finite
    valid = episodes[matched]
    if len(valid):
        points = axes[0].scatter(valid.soc_deficit_kwh, valid.estimated_extra_cost, c=valid.mean_forecast_price, cmap="plasma", s=25, alpha=0.7, edgecolors="none")
        fig.colorbar(points, ax=axes[0], label="Mean forecast price ($/kWh)")
    else:
        axes[0].text(0.5, 0.5, "No feasible paired cost estimates", ha="center", transform=axes[0].transAxes)
    axes[0].set(title=f"(a) Feasible paired estimates (n={len(valid)})", xlabel="SoC deficit at detection (kWh)", ylabel="Estimated extra recovery cost ($)")
    good = episodes[np.isfinite(episodes.soc_deficit_kwh)].copy()
    if len(good):
        if good.soc_deficit_kwh.nunique() > 1:
            good["bin"] = pd.cut(good.soc_deficit_kwh, bins=min(6, good.soc_deficit_kwh.nunique()), include_lowest=True, duplicates="drop")
            grouped = good.groupby("bin", observed=True).agg(x=("soc_deficit_kwh", "mean"), feasible=("actual_feasible", "mean"), n=("actual_feasible", "size"))
        else:
            grouped = pd.DataFrame({"x": [good.soc_deficit_kwh.mean()], "feasible": [good.actual_feasible.mean()], "n": [len(good)]})
        axes[1].plot(grouped.x, grouped.feasible * 100, "o-", color="#0072B2")
        for row in grouped.itertuples():
            axes[1].annotate(f"n={row.n}", (row.x, row.feasible * 100), xytext=(0, -15 if row.feasible > 0.9 else 7), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set(title="(b) Hard-model feasibility: all scenarios", xlabel="SoC deficit at detection (kWh; binned)", ylabel="Actual-state recovery feasible (%)", ylim=(-5, 108))
    axes[1].grid(alpha=0.2)
    _save(fig, directory, "cost_and_feasibility_vs_soc")


def _plot_trajectory(episodes: pd.DataFrame, traces: pd.DataFrame, directory: Path) -> dict:
    candidates = episodes[episodes.actual_feasible & episodes.reference_feasible]
    rule = "largest SoC deficit among estimated paired-feasible scenarios"
    if not len(candidates):
        candidates = episodes
        rule = "fallback: largest SoC deficit, no estimated paired-feasible scenario available"
    if not len(candidates) or not len(traces):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No representative trajectory available", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, directory, "representative_recovery")
        return {"scenario_id": None, "selection_rule": rule}
    candidate = candidates.sort_values(["soc_deficit_kwh", "scenario_id"], ascending=[False, True]).iloc[0]
    selected = traces[traces.scenario_id == candidate.scenario_id].copy()
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 9), sharex=True, constrained_layout=True)
    for method in METHODS:
        g = selected[selected.method == method].sort_values("t")
        if not len(g):
            continue
        dt = g.dt_hours.to_numpy(float)
        time = g.t.to_numpy(float) * dt
        end = time + dt
        style = "--" if method == "state_mpc" else "-"
        zorder = 6 if method == "state_mpc" else 5 if method == "attribution_mpc" else 3
        kw = {"color": COLORS[method], "linestyle": style, "lw": 1.8 if method == "attribution_mpc" else 1.1, "label": LABELS[method], "zorder": zorder}
        axes[0].plot(np.r_[time[0], end], np.r_[g.energy_before_kwh.iloc[0], g.energy_after_kwh], **kw)
        axes[1].step(time, g.charge_kw - g.discharge_kw, where="post", **kw)
        axes[2].step(time, g.grid_import_kw, where="post", **kw)
        cost = np.cumsum(g.grid_import_kw.to_numpy(float) * g.true_price.to_numpy(float) * dt)
        axes[3].plot(np.r_[time[0], end], np.r_[0, cost], **kw)
    primary = selected[selected.method == "attribution_mpc"].sort_values("t")
    if not len(primary):
        primary = selected[selected.method == selected.method.iloc[0]].sort_values("t")
    time = primary.t.to_numpy(float) * primary.dt_hours.to_numpy(float)
    target = float(primary.target_energy_kwh.iloc[0])
    axes[0].axhline(target, color="0.45", linestyle=":", lw=1, label="Recovery target")
    detection_step = float(candidate.get("detection_t", primary.t.iloc[0]))
    detection_time = detection_step * float(primary.dt_hours.iloc[0])
    for ax in axes:
        ax.axvline(detection_time, color="0.5", linestyle=":", lw=0.8)
    axes[0].plot(detection_time, candidate.e_reference_detect, marker="D", ms=5, color="#009E73", linestyle="none", label="Unpoisoned state at detection", zorder=7)
    axes[2].step(time, primary.grid_limit_kw, where="post", color="0.3", linestyle=":", lw=1.2, label="Dispatch import limit")
    deadline = float(primary.deadline_t.iloc[0] * primary.dt_hours.iloc[0])
    for ax in axes:
        ax.axvline(deadline, color="#D55E00", linestyle=":", lw=0.8)
        ax.grid(alpha=0.15)
    axes[0].set(title=f"Representative executed recovery: {candidate.scenario_id}\nDetection: {detection_time:g} h; recovery deadline: {deadline:g} h", ylabel="Battery energy (kWh)")
    axes[1].set(ylabel="Charge − discharge (kW)")
    axes[2].set(ylabel="Grid import (kW)")
    axes[3].set(ylabel="Cumulative grid cost ($)", xlabel="Simulation time (h)")
    axes[0].legend(ncol=3, fontsize=7, loc="best")
    axes[2].legend(handles=axes[2].get_legend_handles_labels()[0][-1:], loc="best", fontsize=7)
    _save(fig, directory, "representative_recovery")
    return {"scenario_id": str(candidate.scenario_id), "selection_rule": rule, "soc_deficit_kwh": float(candidate.soc_deficit_kwh), "cumulative_cost_definition": "Executed grid_import_kw × true_price × dt_hours; excludes any throughput fee."}


def analyze_results(results_dir: Path, figures_dir: Path) -> dict:
    """Read main CSVs, save auditable summaries and four PNG/PDF figures."""
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(results_dir / "episodes.csv")
    controllers = pd.read_csv(results_dir / "controllers.csv")
    traces = pd.read_csv(results_dir / "traces.csv")
    if episodes.scenario_id.duplicated().any():
        raise ValueError("Each scenario_id must occur once in episodes.csv.")
    for column in ["actual_feasible", "reference_feasible", "realized_paired_success"]:
        episodes[column] = _bool_values(episodes[column])
    for column in ["deadline_success", "physical_recovery_success"]:
        controllers[column] = _bool_values(controllers[column])
    controller_summary, comparisons, controller_info = _paired_controller_analysis(controllers)
    prediction_metrics, heldout, prediction_info = _prediction_analysis(episodes)
    controller_summary.to_csv(results_dir / "controller_summary.csv", index=False)
    comparisons.to_csv(results_dir / "paired_comparisons.csv", index=False)
    prediction_metrics.to_csv(results_dir / "prediction_metrics.csv", index=False)
    heldout.to_csv(results_dir / "heldout_predictions.csv", index=False)
    attacked_episodes = episodes[episodes.strength > 0] if "strength" in episodes else episodes.iloc[:0]
    attacked_details = None
    plot_controller_summary, plot_comparisons = controller_summary, comparisons
    plot_prediction_metrics, plot_heldout = prediction_metrics, heldout
    plot_cohort = "All scenarios"
    if len(attacked_episodes):
        attacked_controllers = controllers[controllers.scenario_id.isin(attacked_episodes.scenario_id)]
        attacked_controller_summary, attacked_comparisons, attacked_controller_info = _paired_controller_analysis(attacked_controllers)
        attacked_metrics, attacked_heldout, attacked_prediction_info = _prediction_analysis(attacked_episodes)
        attacked_controller_summary.to_csv(results_dir / "controller_summary_attacked.csv", index=False)
        attacked_comparisons.to_csv(results_dir / "paired_comparisons_attacked.csv", index=False)
        attacked_metrics.to_csv(results_dir / "prediction_metrics_attacked.csv", index=False)
        attacked_heldout.to_csv(results_dir / "heldout_predictions_attacked.csv", index=False)
        attacked_details = {
            "cohort": "strength > 0; excludes zero-attack controls",
            "n_scenarios": len(attacked_episodes),
            "controller_analysis": attacked_controller_info,
            "controller_summary": attacked_controller_summary.to_dict(orient="records"),
            "prediction_analysis": attacked_prediction_info,
            "prediction_metrics": attacked_metrics.to_dict(orient="records"),
        }
        plot_controller_summary, plot_comparisons = attacked_controller_summary, attacked_comparisons
        plot_prediction_metrics, plot_heldout = attacked_metrics, attacked_heldout
        plot_cohort = "Attacked scenarios only (strength > 0)"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        _plot_controllers(plot_controller_summary, plot_comparisons, figures_dir, plot_cohort)
        _plot_prediction(plot_prediction_metrics, plot_heldout, figures_dir, plot_cohort)
        _plot_cost_soc(episodes, figures_dir)
        representative = _plot_trajectory(episodes, traces, figures_dir)
    summary = {
        "n_scenarios": len(episodes), "n_controller_runs": len(controllers), "n_trace_rows": len(traces),
        "feasibility_classes": episodes.feasibility_class.value_counts(dropna=False).to_dict(),
        "n_negative_estimated_extra_cost": int((episodes.estimated_extra_cost < -1e-8).sum()),
        "n_finite_estimated_extra_cost": int(np.isfinite(episodes.estimated_extra_cost).sum()),
        "controller_analysis": controller_info, "prediction_analysis": prediction_info,
        "prediction_metrics": prediction_metrics.to_dict(orient="records"),
        "attacked_only": attacked_details,
        "controller_and_prediction_figure_cohort": plot_cohort,
        "representative_trajectory": representative,
        "figures": ["representative_recovery", "controller_comparison", "heldout_prediction", "cost_and_feasibility_vs_soc"],
    }
    # Test the project's explicit null: attribution alone should not change MPC.
    prop = controllers[controllers.method == "attribution_mpc"].set_index("scenario_id")
    action_keys = ["scenario_id", "t"]
    prop_trace = traces[traces.method == "attribution_mpc"].set_index(action_keys)
    for baseline, output_key in [("state_mpc", "ordinary_mpc_comparison"), ("reset_measured", "measured_reset_comparison")]:
        ordinary = controllers[controllers.method == baseline].set_index("scenario_id")
        match = prop.join(ordinary, how="inner", lsuffix="_proposed", rsuffix="_state")
        state_trace = traces[traces.method == baseline].set_index(action_keys)
        action_match = prop_trace[["charge_kw", "discharge_kw"]].join(state_trace[["charge_kw", "discharge_kw"]], how="inner", lsuffix="_proposed", rsuffix="_state")
        summary[output_key] = {
            "baseline": baseline, "n_matched_scenarios": len(match),
            "max_abs_future_cost_difference": float(np.abs(match.future_cost_proposed - match.future_cost_state).max()) if len(match) else None,
            "n_success_disagreements": int((match.physical_recovery_success_proposed != match.physical_recovery_success_state).sum()),
            "n_matched_action_rows": len(action_match),
            "max_abs_charge_kw_difference": float(np.abs(action_match.charge_kw_proposed - action_match.charge_kw_state).max()) if len(action_match) else None,
            "max_abs_discharge_kw_difference": float(np.abs(action_match.discharge_kw_proposed - action_match.discharge_kw_state).max()) if len(action_match) else None,
            "interpretation_rule": "Equality supports cost attribution as an evaluation layer; it does not support a new-controller performance advantage.",
        }
    clean_summary = _finite_json(summary)
    (results_dir / "summary.json").write_text(json.dumps(clean_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return clean_summary
