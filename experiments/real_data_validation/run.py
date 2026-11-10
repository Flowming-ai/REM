#!/usr/bin/env python3
"""Measured-data cohort: repeat the study on public metered profiles and a settled dynamic tariff.

Only the inputs change. The plant model, attack model, forecast construction, attribution and all six
recovery methods are the unmodified functions of the original study, so any difference in outcome is a
property of the data rather than of a re-implementation.

Measured load and generation come from the Open Power System Data household package; prices come from the
Octopus Agile half-hourly tariff. Scenario construction and the declared scaling rules are documented in
prepare_data.py and recorded in data/data_metadata.json.
"""
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
sys.path.insert(0, str(EXPERIMENTS / "src"))

from recovery.model import (METHODS, Scenario, Profile, simulate_history, simulate_recovery,
                            issue_forecast)
from run_experiments import feasibility
from recovery.optimization import BatterySpec, solve_battery

FACTOR_SEED = 20261009          # fixed before execution


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def real_profile(s: Scenario, w: pd.DataFrame) -> Profile:
    """Measured arrays in the Profile the original code expects; outage window unchanged.

    The plant model admits only nonnegative purchase prices, which the settled tariff violates in 4.23 per
    cent of intervals. Those intervals are floored at zero rather than dropped. Flooring is conservative for
    this study: it removes the payment a consumer would receive for charging, so a recovery cost computed
    this way is never below the cost under the published price. Scenarios containing a floored interval are
    flagged so that the unaffected majority can be reported separately.
    """
    physical = np.full(s.horizon, s.grid_cap_kw)
    physical[20:24] = 0.0                      # same announced outage as the synthetic study
    usable = np.maximum(physical - s.loss_allowance_kw, 0.0)
    tariff = np.maximum(w.tariff_gbp_per_kwh.to_numpy(), 0.0)
    return Profile(w.load_nom_kw.to_numpy(), w.pv_nom_kw.to_numpy(),
                   w.load_kw.to_numpy(), w.pv_kw.to_numpy(),
                   tariff, usable, physical)


def main():
    out = HERE / "results"
    if (HERE / "protocol.json").exists():
        raise RuntimeError("This fixed run already exists; do not overwrite or reselect after outcomes.")
    out.mkdir(parents=True, exist_ok=True)

    # the original study's source and saved results must be untouched
    ref = json.loads((EXPERIMENTS / "expanded_validation" / "original_assets_sha256.json").read_text())
    drift = [k for k, v in ref.items() if not (EXPERIMENTS / k).is_file()
             or digest(EXPERIMENTS / k) != v]
    if drift:
        raise RuntimeError(f"Original assets changed; refusing to run: {sorted(drift)[:5]}")

    data = pd.read_csv(HERE / "data" / "real_profiles.csv")
    dmeta = json.loads((HERE / "data" / "data_metadata.json").read_text())
    n = int(dmeta["scenarios"])
    pl = dmeta["plant"]
    # The plant is sized to the measured community, at the same multiples of the 99th-percentile load
    # that the synthetic study uses of its own. Efficiencies and the interval length are unchanged.
    spec = BatterySpec(capacity_kwh=pl["capacity_kwh"], min_energy_kwh=pl["min_energy_kwh"],
                       max_charge_kw=pl["max_charge_kw"], max_discharge_kw=pl["max_discharge_kw"])

    rng = np.random.default_rng(FACTOR_SEED)
    cases = [Scenario(
        scenario_id=f"real_{i:03d}", seed=90000 + i,
        strength=float((.55, .8, .95)[i % 3]),
        delay_steps=int(rng.choice([2, 4, 8])),
        grid_cap_kw=float(rng.choice(pl["grid_caps_kw"])),
        deadline_steps=int(rng.choice([4, 8, 12])),
        tariff_scale=1.0,                      # measured prices are used as published
        initial_energy_kwh=pl["target_energy_kwh"], target_energy_kwh=pl["target_energy_kwh"],
        loss_allowance_kw=pl["loss_allowance_kw"],
    ) for i in range(n)]

    protocol = {
        "purpose": "Repeat the study on measured load, measured generation and a settled dynamic tariff, "
                   "changing only the inputs.",
        "data": dmeta,
        "scenarios": n, "factor_seed": FACTOR_SEED,
        "attack_strengths": [0.55, 0.80, 0.95], "delays": [2, 4, 8],
        "grid_caps_kw": [5.5, 7.0, 9.0], "deadlines": [4, 8, 12],
        "unchanged": ["plant model equations", "attack model", "forecast construction",
                      "attribution definition", "all six recovery methods", "rescue hierarchy",
                      "success criterion", "efficiencies", "interval length", "outage window"],
        "plant_sizing": "battery power, capacity, minimum energy, recovery target, grid caps and loss "
                        "allowance are set to the same multiples of the 99th-percentile load that the "
                        "synthetic study uses of its own, because measured community demand has a "
                        "peak-to-mean ratio of 5.2 against 1.3 for the synthetic generator",
        "currency": "GBP; measured prices are not rescaled, so costs are not comparable in level with the "
                    "synthetic cohorts",
        "price_treatment": "the plant model admits only nonnegative purchase prices; 4.23 per cent of "
                           "intervals in the settled tariff are negative and are floored at zero, which is "
                           "conservative for recovery cost. 355 of 400 scenarios contain no floored "
                           "interval and are reported separately.",
        "note_on_freezing": "an earlier attempt aborted on the first scenario because the unmodified plant "
                            "model rejected negative prices; no outcome was produced or inspected, so this "
                            "protocol adds the declared price treatment and is frozen before any result.",
        "analysis": "descriptive; no model is fitted and no scenario is excluded after execution",
    }
    write_json(HERE / "protocol.json", protocol)
    phash = digest(HERE / "protocol.json")
    (HERE / "protocol.sha256").write_text(phash + "\n")
    print(f"Protocol frozen: {phash}; starting {n} measured-data cases")

    started = time.perf_counter()
    episodes, controllers = [], []
    for i, s in enumerate(cases):
        w = data[data.scenario_index == i]
        assert len(w) == s.horizon, f"{s.scenario_id}: {len(w)} rows"
        p = real_profile(s, w)
        ea, ah, _ = simulate_history(s, p, spec)
        er, rh, _ = simulate_history(s, p, spec, clean=True)
        f = issue_forecast(s, p, s.detection_t)
        pa = solve_battery(ea, f, spec, target_energy_kwh=s.target_energy_kwh,
                           deadline_index=s.deadline_steps)
        pr = solve_battery(er, f, spec, target_energy_kwh=s.target_energy_kwh,
                           deadline_index=s.deadline_steps)
        ref_metrics, _, reference_states, _ = simulate_recovery(s, p, spec, er, "clean_reference")
        per = {}
        for method in METHODS:
            m, _, _, _ = simulate_recovery(s, p, spec, ea, method,
                                           reference_energy=er, reference_states=reference_states)
            m.update(scenario_id=s.scenario_id, method=method)
            per[method] = m
            controllers.append(m)
        ma = per["state_mpc"]
        episodes.append({
            "scenario_id": s.scenario_id, "seed": s.seed, "strength": s.strength,
            "grid_cap_kw": s.grid_cap_kw, "deadline_steps": s.deadline_steps,
            "delay_steps": s.delay_steps, "eta_charge": spec.eta_charge,
            "e_actual_detect": ea, "e_reference_detect": er, "soc_deficit_kwh": er - ea,
            "actual_feasible": bool(pa.success), "reference_feasible": bool(pr.success),
            "feasibility_class": feasibility(pa.success, pr.success),
            "estimated_extra_cost": pa.cost - pr.cost if pa.success and pr.success else np.nan,
            "realized_extra_cost": ma["future_cost"] - ref_metrics["future_cost"],
            "realized_paired_success": bool(ma["physical_recovery_success"]
                                            and ref_metrics["physical_recovery_success"]),
            "current_price": float(f.price_per_kwh[0]),
            "mean_forecast_price": float(np.mean(f.price_per_kwh)),
            "pre_actual_cost": sum(r["step_cost"] for r in ah),
            "pre_reference_cost": sum(r["step_cost"] for r in rh),
            "has_floored_price": bool((w.tariff_gbp_per_kwh < 0).any()),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}; elapsed {time.perf_counter()-started:.1f}s")

    pd.DataFrame(episodes).to_csv(out / "episodes.csv", index=False)
    pd.DataFrame(controllers).to_csv(out / "controllers.csv", index=False)
    write_json(out / "run_metadata.json", {
        "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "scenarios": n,
        "methods": list(METHODS), "protocol_sha256": phash,
        "python": platform.python_version(),
        "packages": {m: importlib.metadata.version(m) for m in ("numpy", "scipy", "pandas")},
        "original_assets_unchanged": True})
    print(f"Completed {n} cases in {time.perf_counter()-started:.1f}s; original assets unchanged.")


if __name__ == "__main__":
    main()
