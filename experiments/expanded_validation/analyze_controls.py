#!/usr/bin/env python3
"""Secondary paired controller analysis of the already frozen 500-case run.

This reports recovery/service outcomes on all 500 cases, and purchase cost only
on scenarios where both controllers succeed. It changes no simulation or model.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
METHODS=["attribution_mpc","state_mpc","software_replay","reset_measured","max_charge","safe_stop"]
SEED=20260932
DRAWS=20000


def main():
    ctrl=pd.read_csv(HERE / "results/controllers.csv")
    assert len(ctrl)==3000 and not ctrl.duplicated(["scenario_id","method"]).any()
    outcomes=[]
    for method in METHODS:
        part=ctrl[ctrl.method==method]
        outcomes.append({"method":method,"n_scenarios":len(part),"recovery_success_count":int(part.physical_recovery_success.sum()),
                         "deadline_success_count":int(part.deadline_success.sum()),
                         "mean_unserved_kwh":float(part.future_unserved_kwh.mean()),
                         "mean_deadline_shortfall_kwh":float(part.target_shortfall_kwh.mean()),
                         "mean_cumulative_solver_seconds_per_scenario":float(part.solver_seconds.mean())})
    pd.DataFrame(outcomes).to_csv(HERE / "controller_outcomes.csv",index=False,float_format="%.17g")
    base=ctrl[ctrl.method=="state_mpc"].set_index("scenario_id").sort_index()
    rows=[]
    rng=np.random.default_rng(SEED)
    for method in METHODS:
        if method=="state_mpc":continue
        other=ctrl[ctrl.method==method].set_index("scenario_id").sort_index()
        assert base.index.equals(other.index)
        common=base.physical_recovery_success & other.physical_recovery_success
        for metric in ["future_cost","physical_recovery_success","deadline_success","future_unserved_kwh","target_shortfall_kwh"]:
            mask=common if metric=="future_cost" else np.ones(len(base),dtype=bool)
            proposed=base.loc[mask,metric].to_numpy(float)
            baseline=other.loc[mask,metric].to_numpy(float)
            difference=baseline-proposed
            if len(difference):
                indexes=rng.integers(0,len(difference),size=(DRAWS,len(difference)))
                lo,hi=np.quantile(difference[indexes].mean(axis=1),[.025,.975],method="linear")
                m=float(difference.mean());a=float(proposed.mean());b=float(baseline.mean());lo=float(lo);hi=float(hi)
            else:
                m=a=b=lo=hi=None
            rows.append({"baseline":method,"reference":"state_mpc","metric":metric,
                         "cohort":"both_controllers_successful" if metric=="future_cost" else "all_500_scenarios",
                         "n_paired":len(difference),"reference_mean":a,"baseline_mean":b,
                         "mean_difference_baseline_minus_reference":m,"ci95_low":lo,"ci95_high":hi,
                         "bootstrap_draws":DRAWS,"bootstrap_seed":SEED})
    pd.DataFrame(rows).to_csv(HERE / "controller_comparisons.csv",index=False,float_format="%.17g")
    primary=json.loads((HERE / "summary.json").read_text())
    report={"analysis":"Secondary descriptive controller comparisons; no simulation or parameter changes",
            "analyzed_utc":datetime.now(timezone.utc).isoformat(),"outcomes":outcomes,"comparisons":rows,
            "action_equivalence":primary["action_equivalence"],
            "bootstrap":{"seed":SEED,"draws":DRAWS,"unit":"paired whole scenario","interval":"95% percentile","draw_order":"METHODS excluding state_mpc, then listed metric order, consuming one NumPy default_rng stream"},
            "scope":{"cost":"Only successful common scenarios; never reward failed runs for low energy purchase costs",
                     "timing":"Observed cumulative optimizer wall time across recovery per scenario; excludes simulation/ledger/plotting. Attribution includes extra reference solves; machine/run-dependent, not per-action latency.",
                     "physical_reset":"Measured-state reset changes controller belief to actual measured energy, never restores battery energy.",
                     "uncertainty":"Conditional on this synthetic generator and paired scenario sampling; no real-world or independent hardware claim."},
            "source_sha256":{"controllers.csv":hashlib.sha256((HERE / "results/controllers.csv").read_bytes()).hexdigest(),
                             "analyze_controls.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (HERE / "controls_summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"outcomes":outcomes,"max_charge_comparison":[r for r in rows if r["baseline"]=="max_charge"],"actions":report["action_equivalence"]},indent=2))


if __name__=="__main__":main()
