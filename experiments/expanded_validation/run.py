#!/usr/bin/env python3
"""Freeze and execute a new-seed synthetic test without modifying the original study."""
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
from run_experiments import run_main
from recovery.model import METHODS, Scenario
from recovery.optimization import BatterySpec


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    out = HERE / "results"
    if (out / "episodes.csv").exists() or (HERE / "protocol.json").exists():
        raise RuntimeError("This fixed run already exists; do not overwrite or select a new seed after outcomes.")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260926)
    cases = [Scenario(
        scenario_id=f"external_{i:03d}", seed=10000+i,
        strength=float((.55, .8, .95)[i % 3]),
        delay_steps=int(rng.choice([2, 4, 8])), grid_cap_kw=float(rng.choice([5.5, 7.0, 9.0])),
        deadline_steps=int(rng.choice([4, 8, 12])), noise_scale=float(rng.choice([0.0, .05, .12])),
        tariff_scale=float(rng.choice([.7, 1.0, 1.5])),
    ) for i in range(500)]
    old = pd.read_csv(EXPERIMENTS / "results/main/episodes.csv")
    required = ["realized_extra_cost", "estimated_extra_cost", "soc_deficit_kwh", "current_price", "mean_forecast_price", "eta_charge"]
    eligible = ((old.strength > 0) & old.realized_paired_success &
                np.isfinite(old[required]).all(axis=1) & (old.eta_charge > 0))
    train = old[eligible & (old.seed.astype(int) % 5 != 0)].copy()
    assert len(train) == 42
    coefficient = np.linalg.lstsq(np.column_stack([np.ones(len(train)), train.soc_deficit_kwh]),
                                 train.realized_extra_cost, rcond=None)[0]
    assert not set(old.seed) & {s.seed for s in cases}
    protected = sorted((EXPERIMENTS / "src").rglob("*.py")) + sorted((EXPERIMENTS / "results/main").rglob("*"))
    protected += sorted((EXPERIMENTS / "supplementary").rglob("*"))
    protection = {str(p.relative_to(EXPERIMENTS)): digest(p) for p in protected if p.is_file() and "__pycache__" not in p.parts}
    write_json(HERE / "original_assets_sha256.json", protection)
    config = {"master_seed": 20260926, "battery": asdict(BatterySpec()), "scenarios": [asdict(s) for s in cases], "data_type": "synthetic"}
    write_json(out / "scenario_config.json", config)
    protocol = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "title": "New-seed independent synthetic test from the existing generator",
        "n_scenarios": 500, "scenario_ids": "external_000 through external_499",
        "profile_seeds": [10000, 10499], "factor_seed": 20260926,
        "generation": "500 cases fixed once before simulation; no outcome-dependent replacement or extension; six recovery methods per case",
        "factor_support": {"strength_cycle": [.55, .8, .95], "delay_steps": [2,4,8], "grid_cap_kw": [5.5,7.,9.], "deadline_steps": [4,8,12], "noise_scale": [0.,.05,.12], "tariff_scale": [.7,1.,1.5]},
        "prediction_training": {"source": "../results/main/episodes.csv", "n":42,
            "rule":"Original attacked successful finite pairs with seed % 5 != 0; no new-case fitting or selection",
            "scenario_ids": train.scenario_id.tolist(), "intercept": float(coefficient[0]), "slope": float(coefficient[1]),
            "source_sha256":digest(EXPERIMENTS / "results/main/episodes.csv")},
        "test_cohort": "All 500 are new test cases; finite paired-cost prediction comparison requires both initial recovery problems feasible, both rolling recoveries successful, finite inputs/target and positive efficiency. All exclusions and failures reported separately.",
        "predictors": ["counterfactual_cost", "soc_ols", "current_price", "forecast_price"],
        "endpoints": ["MAE", "RMSE", "median_absolute_error", "p90_absolute_error", "p95_absolute_error", "maximum_absolute_error", "paired_MAE_improvement"],
        "subgroups": ["all_eligible", "noise_scale_positive", "noise_scale_zero", "noise_scale_0.05", "noise_scale_0.12"],
        "bootstrap": {"seed":20260927, "draws":20000, "unit":"whole scenario, paired across all predictors", "interval":"95% percentile, linear quantiles", "no_refitting":True},
        "network_scope": "No new AC validation or feasibility/prevention sweeps; original network evidence does not validate these 500 cases.",
        "scientific_scope": "Independent profile seeds from the same synthetic generator and parameter support; not measured, out-of-distribution, cross-platform or LLM-agent validation.",
        "scenario_config_sha256": digest(out / "scenario_config.json"),
        "run_script_sha256": digest(Path(__file__)),
    }
    write_json(HERE / "protocol.json", protocol)
    (HERE / "protocol.sha256").write_text(digest(HERE / "protocol.json") + "  protocol.json\n")
    code_files = sorted((EXPERIMENTS / "src").rglob("*.py")) + [EXPERIMENTS / "run_experiments.py", EXPERIMENTS / "uv.lock", EXPERIMENTS / "pyproject.toml", Path(__file__)]
    metadata = {"status":"running", "started_utc":datetime.now(timezone.utc).isoformat(),
                "python":platform.python_version(), "platform":platform.platform(),
                "episodes":len(cases), "methods":list(METHODS),
                "stages":{"main":True,"sensitivity_and_prevention":False,"ac":False,"analysis":False},
                "protocol_sha256":digest(HERE / "protocol.json"),
                "code_sha256":{str(p.relative_to(EXPERIMENTS)):digest(p) for p in code_files},
                "packages":{name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","matplotlib","pandapower")}}
    write_json(out / "run_metadata.json", metadata)
    print(f"Protocol frozen: {metadata['protocol_sha256']}; starting all {len(cases)} cases", flush=True)
    started = time.perf_counter()
    run_main(cases, out)
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started)
    assert all(digest(EXPERIMENTS / path) == sha for path, sha in protection.items())
    metadata["original_assets_unchanged"] = True
    write_json(out / "run_metadata.json", metadata)
    print(f"Completed {len(cases)} cases in {metadata['elapsed_seconds']:.1f}s; original assets unchanged.", flush=True)


if __name__ == "__main__":
    main()
