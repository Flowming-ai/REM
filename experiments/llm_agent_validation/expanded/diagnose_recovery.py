#!/usr/bin/env python3
"""Describe saved recovery failures; no simulations, fitting or threshold changes.

This is a retrospective descriptive diagnostic, not a newly declared primary
endpoint. The success definition is the frozen experiment's 9 kWh deadline and
terminal target with at most 1e-6 kWh unserved energy / target tolerance.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TARGET_KWH = 9.0
TOLERANCE_KWH = 1e-6
ARMS = ("llm_available", "llm_unavailable", "always_received", "always_verified")
CONDITIONS = ("clean", "poisoned")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def booleans(series):
    values = series.astype(str).str.lower()
    if not values.isin(("true", "false")).all():
        raise ValueError("Unexpected Boolean outcome representation")
    return values.eq("true")


def main():
    inputs = {"recovery_outcomes.csv": RESULTS / "recovery_outcomes.csv",
              "traces.csv": RESULTS / "traces.csv",
              "completion.json": RESULTS / "completion.json",
              "protocol.json": HERE / "protocol.json"}
    initial_hashes = {name: sha256(path) for name, path in inputs.items()}
    completion = json.loads(inputs["completion.json"].read_text())
    frozen = json.loads(inputs["protocol.json"].read_text())
    assert completion["status"] == "complete" and completion["independent_profiles"] == 300
    assert completion["protocol_sha256"] == frozen["protocol_sha256"]
    assert {case["target_energy_kwh"] for case in frozen["protocol"]["cases"]} == {TARGET_KWH}
    for name in ("recovery_outcomes.csv", "traces.csv"):
        assert initial_hashes[name] == completion["result_sha256"][name], name

    outcomes = pd.read_csv(inputs["recovery_outcomes.csv"])
    traces = pd.read_csv(inputs["traces.csv"])
    assert len(outcomes) == 2400 and len(traces) == 96000
    assert not outcomes.duplicated(["seed", "arm", "condition"]).any()
    for column in ("deadline_success", "physical_recovery_success"):
        outcomes[column] = booleans(outcomes[column])
    outcomes["deadline_failure"] = ~outcomes.deadline_success
    outcomes["terminal_failure"] = outcomes.final_energy_kwh < TARGET_KWH - TOLERANCE_KWH
    outcomes["service_failure"] = outcomes.future_unserved_kwh > TOLERANCE_KWH
    outcomes["terminal_shortfall_kwh"] = np.maximum(0., TARGET_KWH - outcomes.final_energy_kwh)
    assert (outcomes.deadline_success == (outcomes.target_shortfall_kwh <= TOLERANCE_KWH)).all()
    reconstructed_success = ~(outcomes.deadline_failure | outcomes.terminal_failure | outcomes.service_failure)
    assert reconstructed_success.equals(outcomes.physical_recovery_success)

    groups = []
    for arm in ARMS:
        for condition in CONDITIONS:
            group = outcomes[(outcomes.arm == arm) & (outcomes.condition == condition)]
            assert len(group) == 300 and set(group.seed) == set(range(30000, 30300))
            d, t, s = (group[name] for name in ("deadline_failure", "terminal_failure", "service_failure"))
            categories = {
                "success": int(group.physical_recovery_success.sum()),
                "deadline_only_failure": int((d & ~t & ~s).sum()),
                "terminal_only_failure": int((t & ~d & ~s).sum()),
                "deadline_and_terminal_failure_without_service_failure": int((d & t & ~s).sum()),
                "any_service_failure": int(s.sum()),
            }
            assert sum(categories.values()) == 300
            groups.append({
                "arm": arm, "condition": condition, "n": len(group),
                "mutually_exclusive_outcomes": categories,
                "component_failure_counts": {"deadline": int(d.sum()), "terminal": int(t.sum()), "service": int(s.sum())},
                "deadline_shortfall_max_kwh": float(group.target_shortfall_kwh.max()),
                "terminal_shortfall_max_kwh": float(group.terminal_shortfall_kwh.max()),
                "unserved_energy_max_kwh": float(group.future_unserved_kwh.max()),
                "scenarios_with_any_strictly_infeasible_recovery_plan": int(group.infeasible_plan_steps.gt(0).sum()),
            })

    baseline = outcomes[(outcomes.arm == "always_verified") & (outcomes.condition == "clean")]
    failed = baseline[~baseline.physical_recovery_success]
    baseline_trace = traces[(traces.arm == "always_verified") & (traces.condition == "clean")
                            & (traces.phase == "post_detection")].copy()
    baseline_trace["strict_feasible"] = booleans(baseline_trace.plan_strict_feasible)
    first_infeasible = baseline_trace[~baseline_trace.strict_feasible].groupby("seed").t.min()
    failed_first = first_infeasible.reindex(failed.seed)
    assert failed_first.notna().all()
    last_infeasible = baseline_trace[(baseline_trace.t == 39) & ~baseline_trace.strict_feasible]
    assert set(last_infeasible.seed) == set(failed.seed)
    baseline_diagnosis = {
        "arm": "always_verified", "condition": "clean", "n": 300,
        "failed_n": len(failed),
        "all_failures_are_terminal_only": bool((failed.terminal_failure & ~failed.deadline_failure & ~failed.service_failure).all()),
        "failed_terminal_shortfall_min_kwh": float(failed.terminal_shortfall_kwh.min()),
        "failed_terminal_shortfall_median_kwh": float(failed.terminal_shortfall_kwh.median()),
        "failed_terminal_shortfall_max_kwh": float(failed.terminal_shortfall_kwh.max()),
        "failed_seeds_with_strictly_infeasible_recovery_plan": int(failed.infeasible_plan_steps.gt(0).sum()),
        "first_strictly_infeasible_interval_histogram_among_failures": {
            str(int(k)): int(v) for k, v in sorted(Counter(failed_first).items())},
        "final_interval_strict_infeasibility_matches_failed_seed_set": True,
        "failed_seeds": [int(s) for s in sorted(failed.seed)],
    }

    paired = []
    for arm in ARMS:
        clean = outcomes[(outcomes.arm == arm) & (outcomes.condition == "clean")].set_index("seed").sort_index()
        poisoned = outcomes[(outcomes.arm == arm) & (outcomes.condition == "poisoned")].set_index("seed").sort_index()
        a, b = clean.physical_recovery_success, poisoned.physical_recovery_success
        added = a & ~b
        paired.append({
            "arm": arm, "n_paired_profiles": len(clean),
            "clean_success_poisoned_failure": int(added.sum()),
            "clean_failure_poisoned_success": int((~a & b).sum()),
            "both_failure": int((~a & ~b).sum()), "both_success": int((a & b).sum()),
            "added_failure_seeds": [int(seed) for seed in clean.index[added]],
        })

    final_hashes = {name: sha256(path) for name, path in inputs.items()}
    assert initial_hashes == final_hashes, "Input changed during diagnosis"
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Retrospective descriptive diagnosis of the existing frozen 300-profile experiment. No simulation, model inference, fitting, outcome replacement or criterion changes.",
        "criterion": {"target_energy_kwh": TARGET_KWH, "tolerance_kwh": TOLERANCE_KWH,
                      "success": "Deadline target AND terminal target AND post-detection unserved energy <= tolerance.",
                      "exclusive_categories": "Service failure includes any energy failures in that trajectory. The other failure categories require service success, so the five categories partition all trajectories."},
        "input_file_sha256": initial_hashes, "inputs_unchanged_during_diagnosis": True,
        "script_sha256": sha256(Path(__file__)),
        "arm_condition_outcomes": groups, "baseline_terminal_failure_diagnosis": baseline_diagnosis,
        "paired_failure_changes": paired,
        "interpretation": [
            "Overall recovery failure is not an attack-success rate: the corresponding clean controller already fails in some profiles.",
            "The late strict-infeasibility records are consistent with limited remaining charge capacity under rolling point forecasts. This descriptive diagnosis does not isolate each causal mechanism or demonstrate that another controller would succeed.",
            "Observed target shortfalls and service outcomes remain distinct; no failure is relabeled and prediction eligibility is unchanged.",
        ],
    }
    output = RESULTS / "recovery_failure_diagnostics.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "arm_condition_outcomes": groups,
                      "baseline_terminal_failure_diagnosis": {k: v for k, v in baseline_diagnosis.items() if k != "failed_seeds"},
                      "paired_failure_changes": paired}, indent=2))


if __name__ == "__main__":
    main()
