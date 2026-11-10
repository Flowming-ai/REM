#!/usr/bin/env python3
"""Independently recompute expanded-study statistics without importing analyze.py."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ARMS = ("llm_available", "llm_unavailable", "always_received", "always_verified")
EPS = 1e-6


def main():
    raw = pd.read_csv(OUT / "paired_episodes.csv")
    steps = pd.read_csv(OUT / "agent_steps.csv")
    frames = {arm: raw[raw.arm.eq(arm)].sort_values("seed").reset_index(drop=True) for arm in ARMS}
    design = frames[ARMS[0]][["seed", "strength", "delay_steps", "grid_cap_kw"]]
    rng = np.random.default_rng(20260929)
    index = np.empty((20000, 300), dtype=np.int32)
    cell_indices = []
    for _, frame in design.groupby(["strength", "delay_steps", "grid_cap_kw"], sort=True):
        positions = frame.index.to_numpy()
        index[:, positions] = rng.choice(positions, size=(20000, len(positions)), replace=True)
        cell_indices.append(positions)
    checks = defaultdict(lambda: True)
    failures = defaultdict(list)
    numerical_count = 0
    metric_count = 0
    cache = {}

    def compare(label, actual, expected):
        nonlocal numerical_count
        numerical_count += 1
        if expected is None:
            good = pd.isna(actual)
        else:
            good = bool(np.isclose(float(actual), float(expected), rtol=0, atol=2e-11, equal_nan=True))
        checks[label.split(":", 1)[0]] &= good
        if not good and len(failures[label.split(":", 1)[0]]) < 5:
            failures[label.split(":", 1)[0]].append({"field": label, "actual": str(actual), "expected": str(expected)})

    def statistic(values, subset=None, root=False):
        values = np.asarray(values, dtype=float)
        keep = np.isfinite(values)
        if subset is not None:
            keep &= np.asarray(subset, dtype=bool)
        key = (values.tobytes(), keep.tobytes(), root)
        if key in cache:
            return cache[key]
        n = int(keep.sum())
        if not n:
            result = {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
        else:
            # Sum per stratum rather than using the analysis implementation's
            # reduced-column vectorization. Every draw keeps whole paired seeds.
            numerator = np.zeros(20000)
            denominator = np.zeros(20000, dtype=np.int32)
            for positions in cell_indices:
                if not keep[positions].any():
                    continue
                sampled = index[:, positions]
                selected = keep[sampled]
                numerator += np.where(selected, values[sampled], 0.).sum(axis=1)
                denominator += selected.sum(axis=1).astype(np.int32)
            replicates = np.divide(numerator, denominator, out=np.full(20000, np.nan), where=denominator > 0)
            point = float(np.mean(values[keep]))
            if root:
                point = float(np.sqrt(max(0., point)))
                replicates = np.sqrt(np.maximum(0., replicates))
            lower, upper = np.nanquantile(replicates, [.025, .975])
            result = {"n": n, "mean": point, "ci95_low": float(lower), "ci95_high": float(upper)}
        cache[key] = result
        return result

    def verify_metric(table, row, name, values, subset=None, root=False):
        nonlocal metric_count
        metric_count += 1
        answer = statistic(values, subset, root)
        for suffix, expected in answer.items():
            compare(f"{table}:{name}_{suffix}", row[f"{name}_{suffix}"], expected)

    def flags(frame):
        feasible = (frame.actual_feasible & frame.reference_feasible).to_numpy()
        success = (frame.poisoned_physical_recovery_success & frame.clean_physical_recovery_success).to_numpy()
        finite = np.isfinite(frame.estimated_extra_cost.to_numpy()) & np.isfinite(frame.realized_extra_cost.to_numpy())
        return feasible, success, feasible & success & finite

    outcomes = pd.read_csv(OUT / "agent_outcomes.csv").set_index("arm")
    predictions = pd.read_csv(OUT / "prediction_metrics.csv").set_index(["arm", "subset"])
    cohorts = pd.read_csv(OUT / "cohort_audit.csv").set_index("arm")
    for arm, frame in frames.items():
        f, s, eligible = flags(frame)
        deficit = frame.energy_deficit_kwh.to_numpy()
        out = outcomes.loc[arm]
        expected_counts = {
            "n_scenarios": 300, "positive_deficit_n": int((deficit > EPS).sum()),
            "zero_difference_n": int((abs(deficit) <= EPS).sum()),
            "negative_deficit_n": int((deficit < -EPS).sum()),
            "any_state_difference_n": int((abs(deficit) > EPS).sum()),
            "initially_paired_feasible_n": int(f.sum()), "both_rolling_success_n": int(s.sum()),
            "prediction_eligible_n": int(eligible.sum()),
            "clean_recovery_success_n": int(frame.clean_physical_recovery_success.sum()),
            "poisoned_recovery_success_n": int(frame.poisoned_physical_recovery_success.sum()),
        }
        for name, value in expected_counts.items():
            compare("agent_outcomes:"+name, out[name], value)
        fields = [
            ("signed_deficit_kwh", deficit, None), ("positive_deficit_rate", deficit > EPS, None),
            ("any_state_difference_rate", abs(deficit) > EPS, None),
            ("poisoned_recovery_success_rate", frame.poisoned_physical_recovery_success, None),
            ("clean_recovery_success_rate", frame.clean_physical_recovery_success, None),
            ("poisoned_unserved_kwh", frame.poisoned_future_unserved_kwh, None),
            ("clean_unserved_kwh", frame.clean_future_unserved_kwh, None),
            ("estimated_extra_cost_initial_feasible_usd", frame.estimated_extra_cost, f),
            ("estimated_extra_cost_eligible_usd", frame.estimated_extra_cost, eligible),
            ("executed_cost_difference_eligible_usd", frame.realized_extra_cost, eligible),
            ("executed_cost_difference_all_usd", frame.realized_extra_cost, None),
        ]
        for name, values, mask in fields:
            verify_metric("agent_outcomes", out, name, values, mask)
        for subset, mask in [
            ("all_eligible", eligible),
            ("nonzero_state_difference_eligible", eligible & (abs(deficit) > EPS)),
            ("zero_state_difference_eligible", eligible & (abs(deficit) <= EPS)),
        ]:
            row = predictions.loc[(arm, subset)]
            error = frame.estimated_extra_cost.to_numpy()-frame.realized_extra_cost.to_numpy()
            compare("prediction_metrics:n", row["n"], int(mask.sum()))
            verify_metric("prediction_metrics", row, "mae_usd", abs(error), mask)
            verify_metric("prediction_metrics", row, "rmse_usd", error**2, mask, True)
            compare("prediction_metrics:p95_absolute_error_usd", row.p95_absolute_error_usd,
                    float(np.quantile(abs(error[mask]), .95)) if mask.any() else None)
            compare("prediction_metrics:max_absolute_error_usd", row.max_absolute_error_usd,
                    float(np.max(abs(error[mask]))) if mask.any() else None)
        counts = {
            "all_scenarios_n": 300,
            "both_initially_infeasible_n": int((~frame.actual_feasible & ~frame.reference_feasible).sum()),
            "actual_only_infeasible_n": int((~frame.actual_feasible & frame.reference_feasible).sum()),
            "reference_only_infeasible_n": int((frame.actual_feasible & ~frame.reference_feasible).sum()),
            "initially_paired_feasible_n": int(f.sum()),
            "initially_feasible_but_rolling_failure_n": int((f & ~s).sum()),
            "rolling_both_success_but_initially_infeasible_n": int((s & ~f).sum()),
            "finite_successful_prediction_n": int(eligible.sum()),
            "zero_state_difference_prediction_n": int((eligible & (abs(deficit) <= EPS)).sum()),
            "nonzero_state_difference_prediction_n": int((eligible & (abs(deficit) > EPS)).sum()),
        }
        for name, value in counts.items():
            compare("cohort_audit:"+name, cohorts.loc[arm][name], value)

    contrasts = pd.read_csv(OUT / "paired_agent_contrasts.csv")
    for row in contrasts.to_dict("records"):
        a, b = (frames[row[k]] for k in ("minuend_arm", "subtrahend_arm"))
        common = flags(a)[2] & flags(b)[2]
        compare("paired_agent_contrasts:n_paired_scenarios", row["n_paired_scenarios"], 300)
        compare("paired_agent_contrasts:n_common_prediction_eligible", row["n_common_prediction_eligible"], int(common.sum()))
        for name, column, mask in [
            ("signed_deficit_difference_kwh", "energy_deficit_kwh", None),
            ("unserved_difference_kwh", "poisoned_future_unserved_kwh", None),
            ("estimated_extra_cost_difference_usd", "estimated_extra_cost", common),
            ("executed_cost_difference_usd", "realized_extra_cost", common),
        ]:
            verify_metric("paired_agent_contrasts", row, name, a[column].to_numpy()-b[column].to_numpy(), mask)
        verify_metric("paired_agent_contrasts", row, "recovery_success_rate_difference",
                      a.poisoned_physical_recovery_success.astype(float).to_numpy()-b.poisoned_physical_recovery_success.astype(float).to_numpy())

    tools = pd.read_csv(OUT / "tool_choices.csv")
    tool_cases = pd.read_csv(OUT / "tool_choices_by_scenario.csv").set_index(["arm", "condition", "seed"])
    for row in tools.to_dict("records"):
        z = steps[(steps.arm == row["arm"]) & (steps.condition == row["condition"])].copy()
        executed = z.status.isin(["executed_plan", "rule_executed"])
        indicators = {
            "received_execution": z.selected_source.eq("received") & executed,
            "verified_execution": z.selected_source.eq("verified") & executed,
            "hold_or_failure": ~executed,
            "model_hold": z.status.eq("model_hold"),
            "invalid_call_hold": z.status.eq("invalid_call_hold"),
            "tool_budget_hold": z.status.eq("tool_budget_hold"),
            "verification_used": z.verification_calls.gt(0),
        }
        compare("tool_choices:n_scenarios", row["n_scenarios"], 300)
        compare("tool_choices:decision_cycles", row["decision_cycles"], len(z))
        compare("tool_choices:generation_calls", row["generation_calls"], int(z.generation_calls.sum()))
        for seed, group in z.groupby("seed", sort=True):
            case = tool_cases.loc[(row["arm"], row["condition"], seed)]
            compare("tool_choices_by_scenario:decision_cycles", case["decision_cycles"], len(group))
            compare("tool_choices_by_scenario:generation_calls", case["generation_calls"], int(group.generation_calls.sum()))
        for name, values in indicators.items():
            by_seed = pd.DataFrame({"seed": z.seed, "value": values}).groupby("seed", sort=True).value.mean()
            compare("tool_choices:"+name+"_cycles", row[name+"_cycles"], int(values.sum()))
            verify_metric("tool_choices", row, name+"_scenario_fraction", by_seed.to_numpy())
            for seed, fraction in by_seed.items():
                compare("tool_choices_by_scenario:"+name, tool_cases.loc[(row["arm"], row["condition"], seed)][name], fraction)

    for table_name in ("factor_effects", "marginal_effects"):
        table = pd.read_csv(OUT / (table_name+".csv"))
        for row in table.to_dict("records"):
            frame = frames[row["arm"]]
            eligible = flags(frame)[2]
            if table_name == "factor_effects":
                mask = (np.isclose(design.strength, row["strength"]) & design.delay_steps.eq(row["delay"])
                        & np.isclose(design.grid_cap_kw, row["grid_limit_kw"])).to_numpy()
            else:
                column = {"strength": "strength", "delay": "delay_steps", "grid_limit_kw": "grid_cap_kw"}[row["factor"]]
                mask = np.isclose(design[column], row["value"])
            compare(table_name+":n_scenarios", row["n_scenarios"], int(mask.sum()))
            compare(table_name+":prediction_eligible_n", row["prediction_eligible_n"], int((mask & eligible).sum()))
            verify_metric(table_name, row, "signed_deficit_kwh", frame.energy_deficit_kwh, mask)
            verify_metric(table_name, row, "estimated_extra_cost_eligible_usd", frame.estimated_extra_cost, mask & eligible)
            if table_name == "factor_effects":
                compare(table_name+":positive_deficit_n", row["positive_deficit_n"], int((mask & (frame.energy_deficit_kwh > EPS)).sum()))
                compare(table_name+":poisoned_recovery_success_n", row["poisoned_recovery_success_n"], int(frame.loc[mask, "poisoned_physical_recovery_success"].sum()))
                verify_metric(table_name, row, "executed_cost_difference_eligible_usd", frame.realized_extra_cost, mask & eligible)
            else:
                verify_metric(table_name, row, "poisoned_recovery_success_rate", frame.poisoned_physical_recovery_success, mask)

    summary = json.loads((OUT / "analysis_summary.json").read_text())
    checks["bootstrap_index_digest"] = hashlib.sha256(index.tobytes()).hexdigest() == summary["bootstrap"]["index_array_sha256"]
    checks["figure_case_selection_predeclared"] = summary["figure_case_seeds"] == [
        int(design[np.isclose(design.strength, .95) & design.delay_steps.eq(4) & np.isclose(design.grid_cap_kw, grid)].seed.min())
        for grid in (5.5, 7.)]
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": all(checks.values()), "checks": dict(checks), "failures": dict(failures),
        "numerical_quantities_recomputed": numerical_count, "metric_groups_recomputed": metric_count,
        "bootstrap_draws": 20000, "bootstrap_seed": 20260929,
        "absolute_comparison_tolerance": 2e-11,
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256((HERE / "analyze.py").read_bytes()).hexdigest(),
        "input_file_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                              for name in ("paired_episodes.csv", "agent_steps.csv", "agent_outcomes.csv",
                                           "prediction_metrics.csv", "cohort_audit.csv", "paired_agent_contrasts.csv",
                                           "tool_choices.csv", "tool_choices_by_scenario.csv", "factor_effects.csv",
                                           "marginal_effects.csv", "analysis_summary.json")},
        "scope": "All eight analysis tables independently recomputed from raw paired outcomes and cycle records; no import of analyze.py or its statistical helpers.",
    }
    (OUT / "independent_statistics_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
