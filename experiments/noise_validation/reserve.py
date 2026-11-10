#!/usr/bin/env python3
"""Reserve-margin recovery control under sensor noise.

The noise cohort showed that cost-minimal recovery places the terminal state exactly on the target
constraint, so any observation error drives it below the target about half the time, usually by a margin
far smaller than the measurement error. The aggressive heuristic succeeds more often not because it
controls better but because charging at full power leaves an implicit reserve.

This run tests the corresponding explicit remedy: plan against a raised target while still being judged
against the original one. Two margin rules are declared before execution so that the margin is not tuned
to the outcome:

    rule A  m = 2 sigma    scaled to the known sensor specification; m = 0 when sigma = 0
    rule B  m = 0.25 kWh   fixed at about 2 percent of capacity, independent of sigma

Success, unserved energy and cost are always evaluated against the original 9 kWh target, so a margin buys
nothing in the metric except through the physical state it actually produces.
"""
from __future__ import annotations

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

from recovery.model import make_profile, simulate_history, issue_forecast, apply_action, _row
from recovery.optimization import BatterySpec, solve_battery
from run import SIGMAS, build_cases, noise_sequence, observe, digest, write_json

MARGIN_RULES = {"reserve_2sigma": None, "reserve_fixed_0p25": 0.25}   # None = 2*sigma


def simulate_reserve(s, p, spec, initial_energy, *, margin, noise):
    """Measured-state MPC planning against target+margin, evaluated against the original target."""
    energy = float(initial_energy)
    planning_target = s.target_energy_kwh + margin
    rows = []
    states = {s.detection_t: energy}
    for t in range(s.detection_t, s.horizon):
        f = issue_forecast(s, p, t)
        local_deadline = s.deadline_t - t if t <= s.deadline_t else None
        belief = observe(energy, noise[t], spec)
        plan = solve_battery(belief, f, spec, target_energy_kwh=planning_target,
                             deadline_index=local_deadline, allow_rescue=True)
        if not plan.success:
            raise RuntimeError(f"{s.scenario_id}/reserve: {plan.status}")
        outcome = apply_action(energy, plan.charge_kw[0], plan.discharge_kw[0],
                               p.actual_load[t], p.actual_pv[t], p.grid_limit[t], p.tariff[t], spec)
        row = _row(s, p, spec, t, "reserve_mpc", "post_detection", outcome, plan)
        rows.append(row)
        energy = outcome["energy_after_kwh"]
        states[t + 1] = energy
    missed = sum(r["unserved_kw"] * spec.dt_hours for r in rows)
    deadline_energy = states[s.deadline_t]
    terminal_energy = states[s.horizon]
    # judged against the ORIGINAL target, not the raised planning target
    success = (deadline_energy >= s.target_energy_kwh - 1e-6 and
               terminal_energy >= s.target_energy_kwh - 1e-6 and missed <= 1e-6)
    return {"scenario_id": s.scenario_id, "seed": s.seed,
            "future_cost": sum(r["step_cost"] for r in rows), "future_unserved_kwh": missed,
            "deadline_energy_kwh": deadline_energy, "terminal_energy_kwh": terminal_energy,
            "physical_recovery_success": bool(success)}


def main():
    out = HERE / "results"
    if (HERE / "protocol_reserve.json").exists():
        raise RuntimeError("This fixed run already exists; do not overwrite or reselect after outcomes.")
    spec = BatterySpec()
    cases = build_cases()
    protocol = {
        "purpose": "Test whether an explicit reserve margin recovers the success rate that cost-minimal "
                   "recovery loses to observation error, and at what cost.",
        "cohort": "the same 500 independent test scenarios and the same noise draws as the noise cohort",
        "sigmas_kwh": SIGMAS,
        "margin_rules": {"reserve_2sigma": "margin = 2*sigma kWh", "reserve_fixed_0p25": "margin = 0.25 kWh"},
        "evaluation": "success, unserved energy and cost judged against the original 9 kWh target; the "
                      "raised target enters the planning constraints only",
        "declared_before_execution": True,
        "primary_outcomes": ["complete recovery count", "mean unserved critical energy",
                             "post-detection purchase cost on jointly successful scenarios"],
    }
    write_json(HERE / "protocol_reserve.json", protocol)
    phash = digest(HERE / "protocol_reserve.json")
    (HERE / "protocol_reserve.sha256").write_text(phash + "\n")
    print(f"Protocol frozen: {phash}; {len(cases)} cases x {len(SIGMAS)} sigmas x {len(MARGIN_RULES)} rules")

    started = time.perf_counter()
    rows = []
    for sigma in SIGMAS:
        for rule, fixed in MARGIN_RULES.items():
            margin = 2.0 * sigma if fixed is None else fixed
            for s in cases:
                p = make_profile(s)
                ea, _, _ = simulate_history(s, p, spec)
                seq = noise_sequence(s.scenario_id, sigma, s.horizon)
                m = simulate_reserve(s, p, spec, ea, margin=margin, noise=seq)
                m.update(sigma_kwh=sigma, rule=rule, margin_kwh=margin)
                rows.append(m)
            print(f"  sigma={sigma} {rule} (m={margin:.3f}): done, elapsed {time.perf_counter()-started:.1f}s")

    pd.DataFrame(rows).to_csv(out / "reserve_controllers.csv", index=False)
    write_json(out / "reserve_metadata.json", {
        "status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "protocol_sha256": phash,
        "python": platform.python_version(),
        "packages": {m: importlib.metadata.version(m) for m in ("numpy", "scipy", "pandas")}})
    print(f"Completed in {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
