"""Controlled sweeps, reference sensitivity, prevention and AC-check figures."""
from __future__ import annotations

import json
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from .analysis import _finite_json, _save


def analyze_extra_results(results_dir: Path, figures_dir: Path) -> dict:
    out, figs = Path(results_dir), Path(figures_dir)
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                "axes.titleweight": "bold", "pdf.fonttype": 42})
    summary = {}
    sweep = pd.read_csv(out / "feasibility_sweep.csv")
    equal = pd.read_csv(out / "equal_deficit.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    table = sweep.pivot(index="grid_cap_kw", columns="deadline_hours", values="extra_cost")
    cmap = plt.colormaps["viridis"].copy(); cmap.set_bad("#e5e5e5")
    im = axes[0].imshow(table.values, origin="lower", aspect="auto", cmap=cmap)
    axes[0].set_xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
    axes[0].set_yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
    for i, row in enumerate(table.values):
        for j, val in enumerate(row):
            if not np.isfinite(val):
                axes[0].add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                           hatch="///", linewidth=0, edgecolor="#aaa"))
    fig.colorbar(im, ax=axes[0], label="Additional recovery cost (USD)", shrink=.8)
    axes[0].set(xlabel="Deadline after detection (h)", ylabel="Physical grid import cap (kW)",
                title="A  Feasibility and cost")
    axes[0].legend(handles=[Patch(facecolor="#e5e5e5", hatch="///", label="Actual recovery infeasible")],
                   fontsize=8, loc="upper right")
    for cap, color in zip([5.5, 7, 9], ["#0072B2", "#D55E00", "#009E73"]):
        part = sweep[sweep.grid_cap_kw == cap]
        axes[1].plot(part.deadline_hours, part.extra_cost, "o-", color=color, label=f"{cap:g} kW cap", ms=4)
    axes[1].set(xlabel="Deadline after detection (h)", ylabel="Additional recovery cost (USD)",
                title="B  Cost at each required deadline")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=.2)
    for deadline, color in zip([8, 12, 16], ["#0072B2", "#D55E00", "#009E73"]):
        part = equal[(equal.grid_cap_kw == 7) & (equal.deadline_steps == deadline)]
        axes[2].plot(part.tariff_scale, part.extra_cost, "o-", color=color,
                     label=f"{deadline*.25:g} h deadline", ms=4)
    axes[2].set(xlabel="Multiplier of the same posted tariff", ylabel="Additional recovery cost (USD)",
                title="C  Same energy deficit, different price")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=.2)
    fig.suptitle("Controlled states: actual 4 kWh; reference 9 kWh; target 9 kWh\n"
                 "Same 5 kWh deficit in every cell; infeasible costs remain undefined", fontsize=12)
    _save(fig, figs, "feasibility_and_cost_frontier")
    summary["controlled_states"] = {
        "n_feasibility_cells": len(sweep), "feasibility_counts": sweep.feasibility_class.value_counts().to_dict(),
        "finite_extra_cost_min_usd": sweep.extra_cost.min(), "finite_extra_cost_max_usd": sweep.extra_cost.max(),
        "n_equal_deficit_cells": len(equal), "equal_deficit_finite_min_usd": equal.extra_cost.min(),
        "equal_deficit_finite_max_usd": equal.extra_cost.max(),
        "earliest_feasible_deadline_hours_by_cap": sweep[sweep.actual_feasible].groupby("grid_cap_kw").deadline_hours.min().to_dict(),
    }

    attacks = pd.read_csv(out / "attack_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for delay, color in zip([2, 4, 8], ["#0072B2", "#D55E00", "#009E73"]):
        part = attacks[attacks.delay_steps == delay]
        axes[0].plot(part.strength, part.deficit_kwh, "o-", color=color, label=f"{delay*.25:g} h detection delay")
        axes[1].plot(part.strength, part.extra_cost, "o-", color=color)
    axes[0].set(ylabel="Reference minus actual energy (kWh)", title="A  State changed by executed actions")
    axes[1].set(ylabel="Additional recovery cost (USD)", title="B  Cost under a common recovery task")
    for ax in axes:
        ax.set_xlabel("Forecast-price reduction fraction"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Attack sweep: one fixed, noise-free profile; 18 executed histories", fontsize=12)
    _save(fig, figs, "attack_strength_and_delay")
    summary["attack_sweep"] = {"n_cases": len(attacks), "max_deficit_kwh": attacks.deficit_kwh.max(),
                               "feasibility_counts": attacks.feasibility_class.value_counts().to_dict()}

    ref = pd.read_csv(out / "reference_sensitivity.csv")
    episodes = pd.read_csv(out / "episodes.csv")
    ref = ref.merge(episodes[["scenario_id", "e_reference_detect"]], on="scenario_id", validate="many_to_one")
    ref["effective_error_kwh"] = ref.perturbed_reference_kwh - ref.e_reference_detect
    stats = ref.groupby("reference_error_fraction_capacity").agg(
        n=("scenario_id", "size"), n_finite=("extra_cost", "count"),
        mean_change_usd=("change_from_nominal", "mean"), min_change_usd=("change_from_nominal", "min"),
        max_change_usd=("change_from_nominal", "max"), mean_effective_error_kwh=("effective_error_kwh", "mean"))
    stats.to_csv(out / "reference_sensitivity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    for _, part in ref.groupby("scenario_id", sort=False):
        axes[0].plot(part.reference_error_fraction_capacity * 100, part.change_from_nominal,
                     "o-", lw=.8, ms=3, alpha=.45, color="#0072B2")
    axes[0].plot(stats.index * 100, stats.mean_change_usd, "o-", color="#D55E00", lw=2.5, label="Mean over 20 cases")
    axes[0].set(xlabel="Requested reference perturbation (% of capacity)", ylabel="Change in attributed cost (USD)",
                title="A  Attribution sensitivity")
    axes[0].legend(fontsize=9)
    axes[1].scatter(ref.effective_error_kwh, ref.change_from_nominal, alpha=.5, color="#0072B2", s=22)
    axes[1].set(xlabel="Effective reference-energy error after clipping (kWh)",
                ylabel="Change in attributed cost (USD)", title="B  Effect of physical capacity clipping")
    for ax in axes:
        ax.axhline(0, color="black", lw=.7); ax.grid(alpha=.2)
    fig.suptitle("First 20 attacked, paired-feasible cases; reference clipped to [1.2, 12] kWh", fontsize=12)
    _save(fig, figs, "reference_sensitivity")
    summary["reference_sensitivity"] = {
        "n_cases": ref.scenario_id.nunique(), "n_perturbations": len(ref),
        "max_absolute_cost_change_usd": ref.change_from_nominal.abs().max(),
        "n_infeasible_perturbations": int((~ref.reference_feasible).sum()),
        "n_clipped_perturbations": int((np.abs(ref.effective_error_kwh-ref.reference_error_fraction_capacity*12)>1e-8).sum()),
        "by_requested_fraction_capacity": stats.reset_index().to_dict("records"),
    }

    prevention = pd.read_csv(out / "prevention.csv")
    pre = prevention.groupby(["strength", "mechanism"]).agg(
        n=("seed", "size"), energy_mean=("energy_at_detection_kwh", "mean"),
        cost_mean=("whole_episode_cost", "mean"), unserved_mean=("whole_episode_unserved_kwh", "mean"),
        success_rate=("physical_recovery_success", "mean")).reset_index()
    pre.to_csv(out / "prevention_summary.csv", index=False)
    names = ["none", "confirmation_gate", "effect_authorization"]
    labels = ["No prevention", "Trusted tariff gate", "Effect-permit proxy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    x = np.arange(3)
    for strength, shift, color, label in [(0, -.18, "#0072B2", "No attack (8 seeds)"),
                                        (.95, .18, "#D55E00", "Attack (8 paired seeds)")]:
        group = pre[pre.strength == strength].set_index("mechanism").loc[names]
        axes[0].bar(x + shift, group.energy_mean, .34, color=color, label=label)
        axes[1].bar(x + shift, group.success_rate*100, .34, color=color)
    axes[0].set(ylabel="Energy at detection (kWh)", ylim=(0, 13), title="A  Before recovery starts")
    axes[1].set(ylabel="Recovery success (%)", ylim=(0, 110), title="B  Same subsequent state-based MPC")
    for ax in axes:
        ax.set_xticks(x, labels); ax.tick_params(axis="x", labelsize=9); ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Separate prevention experiment: static posted tariff; trusted gate has no staleness cost", fontsize=12)
    _save(fig, figs, "prevention_mechanisms")
    summary["prevention"] = pre.to_dict("records")

    ac = pd.read_csv(out / "ac_validation.csv")
    checked = ac[ac.ac_status == "checked_grid_connected"].copy()
    numeric_pass = checked.ac_passed.astype(str).str.lower().isin(["true", "1", "1.0"])
    ac_summary = {
        "n_snapshots_selected": len(ac), "status_counts": ac.ac_status.value_counts().to_dict(),
        "n_grid_connected_passed": int(numeric_pass.sum()), "n_grid_connected_checked": len(checked),
        "min_voltage_pu": checked.ac_min_voltage_pu.min(), "max_voltage_pu": checked.ac_max_voltage_pu.max(),
        "max_line_loading_percent": checked.ac_max_line_loading_percent.max(),
        "max_trafo_loading_percent": checked.ac_max_trafo_loading_percent.max(),
        "max_import_cap_excess_kw": checked.ac_grid_cap_excess_kw.max(),
        "max_loss_kw": checked.ac_loss_kw.max(),
        "max_power_balance_error_abs_kw": checked.ac_power_balance_error_kw.abs().max(),
        "scope": "state_mpc all cases plus all recorded traces of case_000 and case_001; island snapshots excluded from pass denominator",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    axes[0].scatter(checked.ac_grid_kw, checked.ac_min_voltage_pu, s=8, alpha=.35, color="#0072B2", rasterized=True)
    axes[0].axhline(.95, ls="--", color="#D55E00", label="Declared lower limit: 0.95 pu")
    axes[0].set(xlabel="AC source import including losses (kW)", ylabel="Minimum bus voltage (pu)",
                title="A  Grid-connected voltage checks")
    axes[0].legend(fontsize=9)
    margin = checked.grid_limit_kw - checked.ac_grid_kw
    axes[1].hist(margin, bins=28, color="#0072B2", alpha=.8)
    axes[1].axvline(0, ls="--", color="#D55E00")
    axes[1].set(xlabel="Physical source cap minus AC import (kW)", ylabel="Number of checked snapshots",
                title="B  Import headroom after network losses")
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle(f"pandapower: {len(checked):,} grid-connected checks; "
                 f"{(ac.ac_status=='not_checked_islanded').sum():,} island snapshots not checked", fontsize=12)
    _save(fig, figs, "ac_network_validation")
    summary["ac"] = ac_summary

    controllers = pd.read_csv(out / "controllers.csv")
    traces = pd.read_csv(out / "traces.csv")
    analysis_summary = json.loads((out / "summary.json").read_text())
    representative = analysis_summary["representative_trajectory"]["scenario_id"]
    case = episodes.set_index("scenario_id").loc[representative]
    paired_traces = []
    for history_method, recovery_method, color, label in [
        ("poisoned_history", "state_mpc", "#D55E00", "Executed attack, then measured-state MPC"),
        ("clean_history", "clean_reference", "#0072B2", "Unpoisoned reference: same observations and controller")]:
        rows = traces[(traces.scenario_id == representative) & traces.method.isin([history_method, recovery_method])].sort_values("t")
        paired_traces.append((rows, color, label))
    event = None
    with gzip.open(out / "effect_ledger.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if record["scenario_id"] == representative and record["method"] == "poisoned_history" and record["t"] == case.attack_t:
                event = record; break
    if event is None:
        raise ValueError("Representative attacked forecast absent from ledger")
    profiles = pd.read_csv(out / "profiles.csv")
    prices = profiles[(profiles.scenario_id == representative) & (profiles.t >= case.attack_t)]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True, layout="constrained")
    axes[0].step(prices.t*.25, prices.tariff, where="post", color="#0072B2", label="Authentic posted tariff")
    axes[0].step(prices.t*.25, event["forecast"]["price_per_kwh"], where="post", color="#D55E00",
                 label="Poisoned forecast issued at attack onset")
    axes[0].set(ylabel="Price (USD/kWh)", title="A  Recorded input: current price stays true; future prices are reduced")
    for rows, color, label in paired_traces:
        state_time = np.r_[rows.t.iloc[0], rows.t.to_numpy()+1]*.25
        axes[1].plot(state_time, np.r_[rows.energy_before_kwh.iloc[0], rows.energy_after_kwh], color=color, label=label)
        axes[2].plot(state_time, np.r_[0, rows.step_cost.cumsum()], color=color)
    axes[1].axhline(9, color="#555", ls=":", lw=1)
    axes[1].set(ylabel="Battery energy (kWh)", title="B  Real state divergence remains when the forecast is sanitized")
    axes[2].set(ylabel="Purchases since t = 0 (USD)", xlabel="Simulation time (h)",
                title="C  Earlier purchase savings and later recovery expenditure are distinct")
    for ax in axes:
        ax.axvspan(case.attack_t*.25, case.detection_t*.25, color="#F0AD4E", alpha=.12)
        ax.axvspan(5, 6, color="grey", alpha=.1)
        ax.axvline(case.detection_t*.25, ls="--", color="#555", lw=1)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Complete attack-to-recovery record: {representative}\n"
                 "Orange band: corrupted input; dashed line: detection; grey band: announced grid outage", fontsize=12)
    _save(fig, figs, "attack_to_recovery_timeline")
    attacked = episodes[episodes.strength > 0]
    main = {
        "all_n": len(episodes), "attacked_n": len(attacked),
        "paired_feasible_attacked_n": int(attacked.estimated_extra_cost.notna().sum()),
        "attack_deficit_kwh_mean": attacked.soc_deficit_kwh.mean(),
        "attack_deficit_kwh_min": attacked.soc_deficit_kwh.min(),
        "attack_deficit_kwh_max": attacked.soc_deficit_kwh.max(),
        "estimated_extra_cost_attacked_feasible_mean_usd": attacked.estimated_extra_cost.mean(),
        "estimated_extra_cost_attacked_feasible_min_usd": attacked.estimated_extra_cost.min(),
        "estimated_extra_cost_attacked_feasible_max_usd": attacked.estimated_extra_cost.max(),
        "actual_min_recovery_hours_attacked_median": attacked.min_recovery_steps_actual.median()*.25,
        "actual_min_recovery_hours_attacked_max": attacked.min_recovery_steps_actual.max()*.25,
        "n_strict_feasible_but_rollout_failed": int((episodes.actual_feasible & ~controllers[controllers.method=='state_mpc'].set_index('scenario_id').physical_recovery_success.reindex(episodes.scenario_id).to_numpy()).sum()),
        "pre_detection_purchase_difference_attacked_mean_usd": (attacked.pre_actual_cost-attacked.pre_reference_cost).mean(),
        "pre_detection_unserved_actual_total_kwh": episodes.pre_actual_unserved_kwh.sum(),
        "representative_case": {key: case[key] for key in ["seed", "e_actual_detect", "e_reference_detect", "soc_deficit_kwh",
             "estimated_extra_cost", "realized_extra_cost", "pre_actual_cost", "pre_reference_cost", "actual_future_cost", "reference_future_cost"]},
        "solver_seconds_mean_per_episode": controllers.groupby("method").solver_seconds.mean().to_dict(),
        "solver_kind_counts_in_recorded_traces": traces.solver_kind.value_counts().to_dict(),
    }
    summary["main_additional"] = main
    summary = _finite_json(summary)
    (out / "extra_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return summary
