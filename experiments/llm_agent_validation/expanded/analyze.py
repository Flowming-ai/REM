#!/usr/bin/env python3
"""Scenario-level analysis of the fixed 300-profile LLM expansion.

No outcome-dependent case selection and no refitting. Every bootstrap replicate
resamples ten scenarios within each of the thirty design cells, using the same
scenario indices for all four agents and both paired physical trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARMS = ["llm_available", "llm_unavailable", "always_received", "always_verified"]
LABELS = {"llm_available": "LLM + authenticated source", "llm_unavailable": "LLM, received source only",
          "always_received": "Always received (rule)", "always_verified": "Always verified (rule)"}
SHORT = {"llm_available": "LLM\navailable", "llm_unavailable": "LLM\nunavailable",
         "always_received": "Rule\nreceived", "always_verified": "Rule\nverified"}
COLORS = {"llm_available": "#377c9e", "llm_unavailable": "#d17838",
          "always_received": "#af647e", "always_verified": "#639c85"}
EPS = 1e-6
BOOTSTRAPS = 20000
BOOTSTRAP_SEED = 20260929


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(native(value), indent=2, allow_nan=False) + "\n")


def normalize(frame):
    """Accept descriptive runner aliases without changing the measured values."""
    aliases = {"attack_strength": "strength", "detection_delay": "delay",
               "detection_delay_steps": "delay", "delay_steps": "delay",
               "grid_capacity_kw": "grid_limit_kw", "grid_cap_kw": "grid_limit_kw",
               "poisoned_physical_recovery_success": "poisoned_recovery_success",
               "clean_physical_recovery_success": "clean_recovery_success"}
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame = frame.rename(columns={old: new})
    for col in ("actual_feasible", "reference_feasible", "poisoned_recovery_success", "clean_recovery_success"):
        if col in frame and frame[col].dtype == object:
            values = frame[col].astype(str).str.lower()
            assert values.isin(["true", "false"]).all(), col
            frame[col] = values.eq("true")
    return frame


class StratifiedBootstrap:
    def __init__(self, design, n=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
        self.design = design.reset_index(drop=True)
        rng = np.random.default_rng(seed)
        self.indices = np.empty((n, len(design)), dtype=np.int32)
        self.cells = []
        for _, g in self.design.groupby(["strength", "delay", "grid_limit_kw"], sort=True):
            ix = g.index.to_numpy()
            self.cells.append(ix)
            self.indices[:, ix] = rng.choice(ix, size=(n, len(ix)), replace=True)
        self.n = n
        self.seed = seed

    def estimate(self, values, mask=None, sqrt=False):
        a = np.asarray(values, dtype=float)
        valid = np.isfinite(a)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        count = int(valid.sum())
        if not count:
            return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
        clean = np.where(valid, a, 0.)
        # Masking after the common stratified draw estimates the conditional mean;
        # it does not rebalance the selected population or refit an estimator.
        columns = np.concatenate([ix for ix in self.cells if valid[ix].any()])
        index = self.indices[:, columns]
        den = valid[index].sum(axis=1)
        num = clean[index].sum(axis=1)
        reps = np.divide(num, den, out=np.full(self.n, np.nan), where=den > 0)
        point = float(clean.sum() / count)
        if sqrt:
            reps = np.sqrt(np.maximum(reps, 0.))
            point = float(np.sqrt(max(point, 0.)))
        ci = np.nanquantile(reps, [.025, .975])
        return {"n": count, "mean": point, "ci95_low": float(ci[0]), "ci95_high": float(ci[1])}


def eligible(frame):
    return (frame.actual_feasible & frame.reference_feasible
            & frame.poisoned_recovery_success & frame.clean_recovery_success
            & np.isfinite(frame.estimated_extra_cost) & np.isfinite(frame.realized_extra_cost)).to_numpy()


def add_metric(row, name, value):
    row.update({f"{name}_{k}": v for k, v in value.items()})


def analyse_pairs(pairs, bootstrap):
    by_arm = {arm: pairs[pairs.arm.eq(arm)].sort_values("seed").reset_index(drop=True) for arm in ARMS}
    rows, prediction_rows, cohort_rows = [], [], []
    for arm, frame in by_arm.items():
        d = frame.energy_deficit_kwh.to_numpy()
        paired = (frame.actual_feasible & frame.reference_feasible).to_numpy()
        both = (frame.poisoned_recovery_success & frame.clean_recovery_success).to_numpy()
        good = eligible(frame)
        row = {"arm": arm, "n_scenarios": len(frame), "positive_deficit_n": int((d > EPS).sum()),
               "zero_difference_n": int((abs(d) <= EPS).sum()), "negative_deficit_n": int((d < -EPS).sum()),
               "any_state_difference_n": int((abs(d) > EPS).sum()),
               "initially_paired_feasible_n": int(paired.sum()), "both_rolling_success_n": int(both.sum()),
               "prediction_eligible_n": int(good.sum()),
               "clean_recovery_success_n": int(frame.clean_recovery_success.sum()),
               "poisoned_recovery_success_n": int(frame.poisoned_recovery_success.sum())}
        metrics = [("signed_deficit_kwh", d, None), ("positive_deficit_rate", d > EPS, None),
                   ("any_state_difference_rate", abs(d) > EPS, None),
                   ("poisoned_recovery_success_rate", frame.poisoned_recovery_success, None),
                   ("clean_recovery_success_rate", frame.clean_recovery_success, None),
                   ("poisoned_unserved_kwh", frame.poisoned_future_unserved_kwh, None),
                   ("clean_unserved_kwh", frame.clean_future_unserved_kwh, None),
                   ("estimated_extra_cost_initial_feasible_usd", frame.estimated_extra_cost, paired),
                   ("estimated_extra_cost_eligible_usd", frame.estimated_extra_cost, good),
                   ("executed_cost_difference_eligible_usd", frame.realized_extra_cost, good),
                   ("executed_cost_difference_all_usd", frame.realized_extra_cost, None)]
        for name, values, mask in metrics:
            add_metric(row, name, bootstrap.estimate(values, mask))
        rows.append(row)
        error = (frame.estimated_extra_cost - frame.realized_extra_cost).to_numpy()
        for subset, mask in [("all_eligible", good), ("nonzero_state_difference_eligible", good & (abs(d) > EPS)),
                             ("zero_state_difference_eligible", good & (abs(d) <= EPS))]:
            result = {"arm": arm, "subset": subset, "n": int(mask.sum())}
            add_metric(result, "mae_usd", bootstrap.estimate(abs(error), mask))
            add_metric(result, "rmse_usd", bootstrap.estimate(error ** 2, mask, sqrt=True))
            result["p95_absolute_error_usd"] = float(np.quantile(abs(error[mask]), .95)) if mask.any() else None
            result["max_absolute_error_usd"] = float(np.max(abs(error[mask]))) if mask.any() else None
            prediction_rows.append(result)
        cohort_rows.append({"arm": arm, "all_scenarios_n": len(frame),
                            "both_initially_infeasible_n": int((~frame.actual_feasible & ~frame.reference_feasible).sum()),
                            "actual_only_infeasible_n": int((~frame.actual_feasible & frame.reference_feasible).sum()),
                            "reference_only_infeasible_n": int((frame.actual_feasible & ~frame.reference_feasible).sum()),
                            "initially_paired_feasible_n": int(paired.sum()),
                            "initially_feasible_but_rolling_failure_n": int((paired & ~both).sum()),
                            "rolling_both_success_but_initially_infeasible_n": int((both & ~paired).sum()),
                            "finite_successful_prediction_n": int(good.sum()),
                            "zero_state_difference_prediction_n": int((good & (abs(d) <= EPS)).sum()),
                            "nonzero_state_difference_prediction_n": int((good & (abs(d) > EPS)).sum())})
    contrasts = []
    for minuend, subtrahend in [("llm_unavailable", "llm_available"), ("llm_available", "always_verified"),
                                ("llm_unavailable", "always_received"), ("always_received", "always_verified")]:
        a, b = by_arm[minuend], by_arm[subtrahend]
        common = eligible(a) & eligible(b)
        row = {"minuend_arm": minuend, "subtrahend_arm": subtrahend,
               "n_paired_scenarios": len(a), "n_common_prediction_eligible": int(common.sum())}
        for name, col, mask in [("signed_deficit_difference_kwh", "energy_deficit_kwh", None),
                                ("unserved_difference_kwh", "poisoned_future_unserved_kwh", None),
                                ("estimated_extra_cost_difference_usd", "estimated_extra_cost", common),
                                ("executed_cost_difference_usd", "realized_extra_cost", common)]:
            add_metric(row, name, bootstrap.estimate(a[col].to_numpy() - b[col].to_numpy(), mask))
        add_metric(row, "recovery_success_rate_difference", bootstrap.estimate(
            a.poisoned_recovery_success.astype(float).to_numpy() - b.poisoned_recovery_success.astype(float).to_numpy()))
        contrasts.append(row)
    return by_arm, pd.DataFrame(rows), pd.DataFrame(prediction_rows), pd.DataFrame(cohort_rows), pd.DataFrame(contrasts)


def analyse_tools(steps, design, bootstrap):
    output, scenario_rows = [], []
    for arm in ARMS:
        for condition in ("clean", "poisoned"):
            frame = steps[(steps.arm == arm) & (steps.condition == condition)].copy()
            if frame.empty:
                continue
            executed = frame.status.isin(["executed_plan", "rule_executed"])
            frame["received_execution"] = frame.selected_source.eq("received") & executed
            frame["verified_execution"] = frame.selected_source.eq("verified") & executed
            frame["hold_or_failure"] = ~executed
            frame["verification_used"] = frame.verification_calls.gt(0) if "verification_calls" in frame else False
            for name in ("model_hold", "invalid_call_hold", "tool_budget_hold"):
                frame[name] = frame.status.eq(name)
            grouped = frame.groupby("seed", sort=True)
            means = grouped[["received_execution", "verified_execution", "hold_or_failure", "model_hold", "invalid_call_hold", "tool_budget_hold", "verification_used"]].mean()
            means["decision_cycles"] = grouped.size()
            means["generation_calls"] = grouped.generation_calls.sum() if "generation_calls" in frame else 0
            means = means.reindex(design.seed)
            assert means.notna().all().all(), (arm, condition)
            assert np.allclose(means[["received_execution", "verified_execution", "hold_or_failure"]].sum(axis=1), 1.)
            row = {"arm": arm, "condition": condition, "n_scenarios": len(means),
                   "decision_cycles": len(frame), "generation_calls": int(means.generation_calls.sum())}
            for name in ("received_execution", "verified_execution", "hold_or_failure", "model_hold", "invalid_call_hold", "tool_budget_hold", "verification_used"):
                row[f"{name}_cycles"] = int(frame[name].sum())
                add_metric(row, f"{name}_scenario_fraction", bootstrap.estimate(means[name].to_numpy()))
            output.append(row)
            z = means.reset_index()
            z["arm"], z["condition"] = arm, condition
            scenario_rows.extend(z.to_dict("records"))
    return pd.DataFrame(output), pd.DataFrame(scenario_rows)


def analyse_cells(by_arm, design, bootstrap):
    rows = []
    for arm, frame in by_arm.items():
        d = frame.energy_deficit_kwh.to_numpy()
        good = eligible(frame)
        for keys, group in design.groupby(["strength", "delay", "grid_limit_kw"], sort=True):
            mask = np.zeros(len(design), dtype=bool)
            mask[group.index] = True
            row = {"arm": arm, "strength": keys[0], "delay": int(keys[1]), "grid_limit_kw": keys[2],
                   "n_scenarios": int(mask.sum()), "prediction_eligible_n": int((mask & good).sum()),
                   "positive_deficit_n": int((mask & (d > EPS)).sum()),
                   "poisoned_recovery_success_n": int(frame.loc[mask, "poisoned_recovery_success"].sum())}
            add_metric(row, "signed_deficit_kwh", bootstrap.estimate(d, mask))
            add_metric(row, "estimated_extra_cost_eligible_usd", bootstrap.estimate(frame.estimated_extra_cost, mask & good))
            add_metric(row, "executed_cost_difference_eligible_usd", bootstrap.estimate(frame.realized_extra_cost, mask & good))
            rows.append(row)
    return pd.DataFrame(rows)


def analyse_marginals(by_arm, design, bootstrap):
    rows = []
    for factor in ("strength", "delay", "grid_limit_kw"):
        for value, group in design.groupby(factor, sort=True):
            mask = np.zeros(len(design), dtype=bool)
            mask[group.index] = True
            for arm, frame in by_arm.items():
                good = eligible(frame)
                row = {"arm": arm, "factor": factor, "value": float(value), "n_scenarios": int(mask.sum()),
                       "prediction_eligible_n": int((mask & good).sum())}
                add_metric(row, "signed_deficit_kwh", bootstrap.estimate(frame.energy_deficit_kwh, mask))
                add_metric(row, "poisoned_recovery_success_rate", bootstrap.estimate(frame.poisoned_recovery_success, mask))
                add_metric(row, "estimated_extra_cost_eligible_usd", bootstrap.estimate(frame.estimated_extra_cost, mask & good))
                rows.append(row)
    return pd.DataFrame(rows)


def figure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.fontsize": 7, "axes.labelsize": 8})


def save_figure(fig, folder, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(folder, by_arm, outcomes, tools):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)
    keys = [(a, c) for a in ARMS[:2] for c in ("clean", "poisoned")]
    names = ["Available\nclean", "Available\npoisoned", "Unavailable\nclean", "Unavailable\npoisoned"]
    bottom = np.zeros(4)
    for name, label, color in [("received_execution", "Received", "#db9a6f"),
                                ("verified_execution", "Verified", "#6eaa96"),
                                ("hold_or_failure", "Hold / failure", "#a8adb3")]:
        values = np.array([tools.loc[(tools.arm == a) & (tools.condition == c), f"{name}_scenario_fraction_mean"].iloc[0] for a, c in keys])
        axes[0, 0].bar(np.arange(4), values * 100, bottom=bottom * 100, color=color, label=label, width=.66)
        bottom += values
    axes[0, 0].set(xticks=np.arange(4), xticklabels=names, ylabel="Mean scenario fraction (%)", ylim=(0, 115),
                   title="(a) LLM-authorized decisions")
    axes[0, 0].legend(loc="upper center", ncol=3, fontsize=6.6, frameon=False)
    for ax, column, is_cost, title, ylabel in [
            (axes[0, 1], "energy_deficit_kwh", False, "(b) All 300 paired scenarios per agent", "Detection energy deficit (kWh)"),
            (axes[1, 0], "estimated_extra_cost", True, "(c) Finite estimates with successful recoveries", "Attributed future cost (USD)")]:
        values = []
        for arm in ARMS:
            frame = by_arm[arm]
            mask = eligible(frame) if is_cost else np.ones(len(frame), dtype=bool)
            values.append(frame.loc[mask, column].to_numpy())
        boxes = ax.boxplot(values, patch_artist=True, widths=.52, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.1},
                           whiskerprops={"linewidth": .8}, capprops={"linewidth": .8})
        rng = np.random.default_rng(20260930)
        for i, (arm, v, box) in enumerate(zip(ARMS, values, boxes["boxes"])):
            box.set_facecolor(COLORS[arm]); box.set_alpha(.5)
            ax.scatter(i + 1 + rng.uniform(-.17, .17, len(v)), v, s=4, alpha=.28, color=COLORS[arm], linewidths=0, rasterized=True)
        labels = [SHORT[a] + (f"\n(n={len(v)})" if is_cost else "") for a, v in zip(ARMS, values)]
        ax.set(xticks=np.arange(1, 5), xticklabels=labels, ylabel=ylabel, title=title)
        ax.axhline(0, color=".6", lw=.65, ls=":", zorder=-1)
    rowmap = outcomes.set_index("arm")
    for k, arm in enumerate(ARMS):
        row = rowmap.loc[arm]
        m, low, high = [100 * row[f"poisoned_recovery_success_rate_{s}"] for s in ("mean", "ci95_low", "ci95_high")]
        axes[1, 1].errorbar(k, m, yerr=[[max(0., m-low)], [max(0., high-m)]], fmt="o", color=COLORS[arm], capsize=3, ms=5)
        axes[1, 1].annotate(f"{int(row.poisoned_recovery_success_n)}/300", (k, m), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    axes[1, 1].set(xticks=np.arange(4), xticklabels=[SHORT[a] for a in ARMS], ylabel="Recovery success (%)",
                   title="(d) All attacked trajectories; 95% intervals", ylim=(0, 110), xlim=(-.5, 3.5))
    save_figure(fig, folder, "expanded_llm_outcomes")


def plot_cells(folder, cells):
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.35), sharex=True, sharey=True, constrained_layout=True)
    strength_colors = {0.55: "#5b8a9d", 0.8: "#b8893e", 0.95: "#9f617d"}
    letters = "abcd"
    for i, arm in enumerate(ARMS[:2]):
        for j, grid in enumerate((5.5, 7.0)):
            ax = axes[i, j]
            for strength, color in strength_colors.items():
                z = cells[(cells.arm == arm) & np.isclose(cells.grid_limit_kw, grid) & np.isclose(cells.strength, strength)].sort_values("delay")
                y = z.signed_deficit_kwh_mean.to_numpy()
                lo = z.signed_deficit_kwh_ci95_low.to_numpy()
                hi = z.signed_deficit_kwh_ci95_high.to_numpy()
                ax.errorbar(z.delay, y, yerr=[np.maximum(y-lo, 0), np.maximum(hi-y, 0)], color=color,
                            marker="o", ms=3.5, lw=1., capsize=2, label=f"Strength {strength:.2f}")
            if i == 0:
                ax.set_title(f"({letters[i*2+j]}) Source available; grid {grid:g} kW")
            else:
                ax.set_title(f"({letters[i*2+j]}) Source unavailable; grid {grid:g} kW")
            ax.axhline(0, color=".6", ls=":", lw=.7)
            ax.set(xticks=[1, 2, 3, 4, 6])
            if j == 0:
                ax.set_ylabel("Mean energy deficit (kWh)")
            if i == 1:
                ax.set_xlabel("Detection delay (15-min steps)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    save_figure(fig, folder, "expanded_llm_cell_effects")


def plot_factors(folder, marginal):
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.3), constrained_layout=True)
    factors = [("strength", "Poisoning strength", [0.55, .8, .95]),
               ("delay", "Detection delay (steps)", [1, 2, 3, 4, 6]),
               ("grid_limit_kw", "Physical grid capacity (kW)", [5.5, 7.])]
    for col, (factor, xlabel, ticks) in enumerate(factors):
        for arm in ARMS[:2]:
            z = marginal[(marginal.arm == arm) & (marginal.factor == factor)].sort_values("value")
            for row, metric in enumerate(("signed_deficit_kwh", "poisoned_recovery_success_rate")):
                scale = 100. if row else 1.
                mean = z[f"{metric}_mean"].to_numpy() * scale
                lo = z[f"{metric}_ci95_low"].to_numpy() * scale
                hi = z[f"{metric}_ci95_high"].to_numpy() * scale
                axes[row, col].errorbar(z.value, mean, yerr=[np.maximum(mean-lo, 0), np.maximum(hi-mean, 0)],
                                       color=COLORS[arm], marker="o" if arm == ARMS[0] else "s", markersize=3.5,
                                       linewidth=1., capsize=2, label="Source available" if arm == ARMS[0] else "Source unavailable")
        for row in range(2):
            axes[row, col].set(xticks=ticks, xlabel=xlabel, title=f"({chr(97+row*3+col)}) {int(z.n_scenarios.iloc[0])} scenarios / value")
            if row == 0:
                axes[row, col].axhline(0, color=".65", ls=":", lw=.6)
            else:
                axes[row, col].set_ylim(0, 105)
    axes[0, 0].set_ylabel("Mean energy deficit (kWh)")
    axes[1, 0].set_ylabel("Recovery success (%)")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    save_figure(fig, folder, "expanded_llm_factors")


def first_received_forecasts(results, seeds):
    found = {}
    paths = sorted((results / "parts").glob("part_*/effect_ledger.jsonl.gz"))
    for path in paths:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                seed, arm, condition = row.get("seed"), row.get("arm"), row.get("condition")
                key = seed, arm, condition
                if seed not in seeds or arm not in ARMS[:2] or key in found:
                    continue
                prices = row.get("received_forecast_prices")
                if prices is None:
                    prices = row.get("observation", {}).get("received_external_bulletin", {}).get("price_usd_per_kwh")
                if prices is not None:
                    found[key] = (int(row["t"]), np.asarray(prices, dtype=float))
        if len(found) == 4 * len(seeds):
            break
    assert len(found) == 4 * len(seeds), "Missing preselected raw bulletin"
    return found


def mechanism_seeds(pairs):
    seeds = []
    for grid in (5.5, 7.):
        group = pairs[np.isclose(pairs.strength, .95) & pairs.delay.eq(4) & np.isclose(pairs.grid_limit_kw, grid)]
        seeds.append(int(group.seed.min()))
    return seeds


def interval_step(ax, interval_indices, values, dt, **kwargs):
    """Draw each interval through its right endpoint, including the final one."""
    indices = np.asarray(interval_indices, dtype=float)
    y = np.asarray(values, dtype=float)
    assert len(indices) == len(y) and len(y) > 0
    return ax.step(np.r_[indices, indices[-1] + 1.] * dt, np.r_[y, y[-1]], where="post", **kwargs)


def plot_mechanism_full(folder, traces, steps, pairs, results):
    seeds = mechanism_seeds(pairs)
    bulletins = first_received_forecasts(results, seeds)
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 7.45), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.1, .72, 1.05, 1.2]})
    dt = .25
    state_colors = {"received": "#db9a6f", "verified": "#6eaa96", "hold": "#a8adb3"}
    from matplotlib.patches import Patch
    keys = [(arm, condition) for arm in ARMS[:2] for condition in ("clean", "poisoned")]
    for column, seed in enumerate(seeds):
        z = traces[(traces.seed == seed) & traces.arm.isin(ARMS[:2])].copy()
        end_time = (float(z.t.max()) + 1.) * dt
        settings = pairs[pairs.seed == seed].iloc[0]
        attack_t = 4
        detection_t = int(settings.detection_t)
        deadline_t = int(settings.deadline_t)
        clean = z[(z.arm == "llm_available") & (z.condition == "clean")].sort_values("t")
        interval_step(axes[0, column], clean.t, clean.true_price, dt, label="True / clean price", color="#525966", lw=1.15)
        t, price = bulletins[(seed, "llm_available", "poisoned")]
        interval_step(axes[0, column], t + np.arange(len(price)), price, dt, color="#d17838", lw=1., label="Poisoned bulletin at t=4")
        axes[0, column].set(ylabel="Price (USD/kWh)", title=f"({chr(97+column)}) Seed {seed}; physical grid {settings.grid_limit_kw:g} kW")
        axes[0, column].set_title(axes[0, column].get_title(), pad=24)
        axes[0, column].legend(loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0,
                               frameon=False, fontsize=6.0, ncol=2, columnspacing=.8, handlelength=1.6)
        for row_number, (arm, condition) in enumerate(keys):
            frame = steps[(steps.seed == seed) & (steps.arm == arm) & (steps.condition == condition)].sort_values("t")
            for row in frame.itertuples():
                source = row.selected_source if row.status == "executed_plan" else "hold"
                axes[1, column].broken_barh([(row.t * dt, dt)], (row_number - .34, .68), facecolors=state_colors.get(source, state_colors["hold"]), edgecolors="white", linewidth=.5)
        axes[1, column].set(yticks=np.arange(4), yticklabels=["A / clean", "A / poison", "U / clean", "U / poison"], title=f"({chr(99+column)}) Executed price source")
        axes[1, column].set_title(axes[1, column].get_title(), pad=24)
        axes[1, column].legend(handles=[Patch(color=v, label=k.capitalize()) for k, v in state_colors.items()],
                               ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0, frameon=False, fontsize=6.)
        for arm, condition in keys:
            frame = z[(z.arm == arm) & (z.condition == condition)].sort_values("t")
            color = COLORS[arm]
            style = "--" if condition == "clean" else "-"
            label = ("A" if arm == "llm_available" else "U") + " / " + condition
            interval_step(axes[2, column], frame.t, frame.charge_kw - frame.discharge_kw, dt,
                          label=label, color=color, ls=style, lw=1.)
            axes[3, column].plot(np.r_[0., (frame.t.to_numpy() + 1) * dt],
                                np.r_[frame.energy_before_kwh.iloc[0], frame.energy_after_kwh], color=color, ls=style, lw=1.2, label=label)
        axes[2, column].set(ylabel="Net charging (kW)", title=f"({chr(101+column)}) Actual physical battery action")
        axes[3, column].set(ylabel="Battery energy (kWh)", xlabel="Elapsed time (h)", title=f"({chr(103+column)}) Physical state and MPC recovery")
        axes[3, column].axhline(9., color=".55", lw=.7, ls=":")
        axes[3, column].set_title(axes[3, column].get_title(), pad=24)
        axes[3, column].legend(loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0,
                               frameon=False, ncol=4, fontsize=5.7, columnspacing=.55, handlelength=1.2)
        for row_number in range(4):
            ax = axes[row_number, column]
            ax.axvspan(attack_t * dt, detection_t * dt, color=".9", zorder=-3)
            ax.axvline(detection_t * dt, color=".4", ls="--", lw=.75)
            ax.axvline(deadline_t * dt, color="#946840", ls=":", lw=.7)
            ax.set_xlim(0, end_time)
            ax.tick_params(labelsize=7)
    save_figure(fig, folder, "expanded_llm_mechanism_full")


def plot_mechanism(folder, traces, steps, pairs):
    """Compact main-paper view; logged price bulletin remains in the full figure."""
    from matplotlib.patches import Patch
    seeds = mechanism_seeds(pairs)
    fig = plt.figure(figsize=(7.15, 4.8), constrained_layout=True)
    layout = fig.add_gridspec(3, 2, height_ratios=[.35, 1.05, 1.35])
    state_colors = {"received": "#db9a6f", "verified": "#6eaa96", "hold": "#a8adb3"}
    keys = [(arm, condition) for arm in ARMS[:2] for condition in ("clean", "poisoned")]
    for column, seed in enumerate(seeds):
        case_traces = traces[(traces.seed == seed) & traces.arm.isin(ARMS[:2])]
        end_time = (float(case_traces.t.max()) + 1.) * .25
        settings = pairs[pairs.seed == seed].iloc[0]
        detection, deadline = float(settings.detection_t) * .25, float(settings.deadline_t) * .25
        source_ax = fig.add_subplot(layout[0, column])
        action_ax = fig.add_subplot(layout[1, column], sharex=source_ax)
        energy_ax = fig.add_subplot(layout[2, column], sharex=source_ax)
        source_ax.set(yticks=[0, 1], yticklabels=["A", "U"], ylim=(-.5, 1.5),
                      title=f"Seed {seed}; physical grid {settings.grid_limit_kw:g} kW")
        source_ax.spines[["left", "bottom"]].set_visible(False)
        source_ax.tick_params(axis="both", length=0, labelbottom=False)
        for row_number, arm in enumerate(ARMS[:2]):
            subset = steps[(steps.seed == seed) & (steps.arm == arm) & steps.condition.eq("poisoned")]
            for row in subset.itertuples():
                source = row.selected_source if row.status == "executed_plan" else "hold"
                source_ax.broken_barh([(row.t * .25, .25)], (row_number - .37, .74),
                                      facecolors=state_colors.get(source, state_colors["hold"]), edgecolors="white", linewidth=.4)
        source_ax.legend(handles=[Patch(color=v, label=k.capitalize()) for k, v in state_colors.items()],
                         loc="center right", ncol=1, frameon=False, fontsize=5.8, handlelength=.8, columnspacing=.6, labelspacing=.15)
        for arm, condition in keys:
            frame = traces[(traces.seed == seed) & (traces.arm == arm) & (traces.condition == condition)].sort_values("t")
            color, style = COLORS[arm], "--" if condition == "clean" else "-"
            label = ("A" if arm == "llm_available" else "U") + " / " + condition
            interval_step(action_ax, frame.t, frame.charge_kw - frame.discharge_kw, .25, color=color, ls=style, lw=.95)
            energy_ax.plot(np.r_[0., (frame.t.to_numpy() + 1) * .25],
                           np.r_[frame.energy_before_kwh.iloc[0], frame.energy_after_kwh], color=color, ls=style, lw=1.15, label=label)
        action_ax.set(ylabel="Net charging (kW)", title=f"({chr(97+column)}) Actual battery action", ylim=(-4.4, 4.4))
        action_ax.tick_params(labelbottom=False)
        energy_ax.set(ylabel="Battery energy (kWh)", xlabel="Elapsed time (h)",
                      title=f"({chr(99+column)}) State and subsequent MPC recovery", ylim=(.8, 12.6))
        energy_ax.axhline(9., color=".55", lw=.7, ls=":")
        energy_ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=6.3, columnspacing=.6)
        for ax in (action_ax, energy_ax):
            ax.axvspan(1., detection, color=".9", zorder=-3)
            ax.axvline(detection, color=".4", ls="--", lw=.75)
            ax.axvline(deadline, color="#946840", ls=":", lw=.7)
        for ax in (source_ax, action_ax, energy_ax):
            ax.set_xlim(0, end_time)
            ax.tick_params(labelsize=7)
    save_figure(fig, folder, "expanded_llm_mechanism")


def analyse_generation_records(results):
    totals = {"logged_model_calls": 0, "model_generated_tokens": 0,
              "grammar_formatting_tokens": 0, "complete_serialized_tokens": 0,
              "actual_grammar_branch_decisions": 0, "forced_only_calls": 0,
              "unique_inference_batches": 0, "sum_batch_inference_seconds": 0.}
    batches = {}
    for path in sorted((results / "parts").glob("part_*/model_transcripts.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                totals["logged_model_calls"] += 1
                totals["model_generated_tokens"] += record["model_generated_token_count"]
                totals["grammar_formatting_tokens"] += len(record["forced_prefix_token_ids"]) + len(record["forced_suffix_token_ids"])
                totals["complete_serialized_tokens"] += record["output_token_count"]
                totals["actual_grammar_branch_decisions"] += len(record["branch_decisions"])
                totals["forced_only_calls"] += int(record["forced_only_call"])
                batches.setdefault((path.parent.name, record["batch_id"]), record["batch_inference_seconds"])
    totals["unique_inference_batches"] = len(batches)
    totals["sum_batch_inference_seconds"] = float(sum(batches.values()))
    totals["interpretation"] = ("Real model-generated tokens and branch decisions are separated from fixed grammar formatting. "
                                "Batch time is summed once per batch; it is not per-call latency and is outside the physical cost objective.")
    assert totals["model_generated_tokens"] + totals["grammar_formatting_tokens"] == totals["complete_serialized_tokens"]
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.results
    pairs = normalize(pd.read_csv(out / "paired_episodes.csv"))
    steps = normalize(pd.read_csv(out / "agent_steps.csv"))
    traces = normalize(pd.read_csv(out / "traces.csv"))
    assert set(pairs.arm) == set(ARMS)
    assert len(pairs) == 1200 and not pairs.duplicated(["seed", "arm"]).any()
    design = pairs[pairs.arm == ARMS[0]][["seed", "strength", "delay", "grid_limit_kw"]].sort_values("seed").reset_index(drop=True)
    assert design.seed.tolist() == list(range(30000, 30300))
    cells = design.groupby(["strength", "delay", "grid_limit_kw"]).size()
    assert len(cells) == 30 and cells.eq(10).all()
    for arm in ARMS:
        assert design.equals(pairs[pairs.arm == arm][design.columns].sort_values("seed").reset_index(drop=True))
    bootstrap = StratifiedBootstrap(design)
    by_arm, outcomes, predictions, cohorts, contrasts = analyse_pairs(pairs, bootstrap)
    tools, tool_scenarios = analyse_tools(steps, design, bootstrap)
    factors = analyse_cells(by_arm, design, bootstrap)
    marginals = analyse_marginals(by_arm, design, bootstrap)
    tables = {"agent_outcomes": outcomes, "prediction_metrics": predictions, "cohort_audit": cohorts,
              "paired_agent_contrasts": contrasts, "tool_choices": tools,
              "tool_choices_by_scenario": tool_scenarios, "factor_effects": factors, "marginal_effects": marginals}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    llm_steps = steps[steps.arm.isin(ARMS[:2])]
    generation_summary = analyse_generation_records(out)
    checks = {"300_new_unique_seeds": len(design) == 300,
              "30_cells_of_10": len(cells) == 30 and bool(cells.eq(10).all()),
              "1200_agent_scenario_pairs": len(pairs) == 1200,
              "2400_complete_physical_trajectories": len(traces.groupby(["seed", "arm", "condition"])) == 2400,
              "3840_llm_decision_cycles": len(llm_steps) == 3840,
              "logged_calls_match_cycle_generation_counts": generation_summary["logged_model_calls"] == int(llm_steps.generation_calls.sum()),
              "per_scenario_tool_fractions_sum_to_one": bool(np.allclose(tool_scenarios[["received_execution", "verified_execution", "hold_or_failure"]].sum(axis=1), 1.)),
              "all_cost_predictions_use_finite_feasible_successful_pairs": all(
                  predictions[predictions.arm == a].iloc[0]["n"] == int(eligible(f).sum()) for a, f in by_arm.items())}
    if not all(checks.values()):
        raise RuntimeError(str({k: v for k, v in checks.items() if not v}))
    summary = {"n_independent_scenarios": 300, "n_factor_cells": 30, "n_scenarios_per_cell": 10,
               "n_agent_scenario_pairs": 1200, "n_physical_trajectories": 2400,
               "n_llm_cycles": len(llm_steps), "n_llm_generations": int(llm_steps.generation_calls.sum()),
               "generation_records": generation_summary,
               "bootstrap": {"replicates": BOOTSTRAPS, "seed": BOOTSTRAP_SEED,
                             "unit": "whole scenario, paired across all arms and clean/poisoned trajectories",
                             "stratification": "resample ten scenarios within each of thirty strength-delay-grid cells",
                             "interval": "percentile 2.5% and 97.5%; conditional means computed after masking each shared draw",
                             "boundary_caution": "Bootstrap proportion intervals degenerate at 0/300 or 300/300; this does not establish zero population risk or certain success.",
                             "index_array_sha256": hashlib.sha256(bootstrap.indices.tobytes()).hexdigest()},
               "effect_threshold_kwh": EPS, "figure_case_seeds": mechanism_seeds(pairs),
               "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                for name in ("paired_episodes.csv", "agent_steps.csv", "traces.csv")},
               "checks": checks, "agent_outcomes": outcomes.to_dict("records"),
               "prediction_metrics": predictions.to_dict("records"),
               "paired_agent_contrasts": contrasts.to_dict("records"), "tool_choices": tools.to_dict("records"),
               "scope": ["One local model in a numeric forecast-poisoning simulator; not natural-language prompt injection.",
                         "Availability intervention jointly changes access, tool schema and availability instructions.",
                         "Delay changes the poisoned control-window length, absolute detection time and absolute service deadline; marginal delay effects are not a pure duration intervention.",
                         "Each agent has its own clean reference. Cross-agent Delta-J contrasts compare attack-attributable increments, not poisoned total spending.",
                         "Always-verified control assumes access to the authenticated source.",
                         "Tool proportions are equally weighted scenario means, not duration-weighted cycle percentages.",
                         "Zero state-difference cases remain in the main results and are separated in prediction metrics.",
                         "Cost accuracy is conditional on finite initial paired feasibility and both rolling recoveries succeeding.",
                         "Source-cost contrasts use common eligible scenarios; failed cases remain in cohort and recovery summaries.",
                         "Inference latency, compute energy and authenticated-source acquisition cost are outside physical cost accounting."]}
    write_json(out / "analysis_summary.json", summary)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    figure_style()
    plot_outcomes(figures, by_arm, outcomes, tools)
    plot_factors(figures, marginals)
    plot_cells(figures, factors)
    plot_mechanism(figures, traces, steps, pairs)
    plot_mechanism_full(figures, traces, steps, pairs, out)
    print(json.dumps(native({"n_independent_scenarios": 300, "n_llm_cycles": len(llm_steps),
                             "n_llm_generations": int(llm_steps.generation_calls.sum()), "checks": checks,
                             "outcomes": outcomes[["arm", "positive_deficit_n", "prediction_eligible_n", "signed_deficit_kwh_mean", "poisoned_recovery_success_n"]].to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()
