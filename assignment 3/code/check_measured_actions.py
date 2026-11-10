"""Compare the actions of Attribution MPC and State MPC step by step on the 400 measured-data scenarios.

The measured-data run (experiments/real_data_validation/run.py) saved only the outcome of each recovery, not the
actions. This script rebuilds the same scenarios with the unchanged experiment code, reruns the two controllers,
stops if a rerun does not reproduce the saved cost and end level, and compares the two controllers at every step.
Run from the project folder (RMProj):
    experiments/.venv/bin/python "assignment 3/code/check_measured_actions.py"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
sys.path[:0] = [str(EXP / "real_data_validation"), str(EXP / "src"), str(EXP)]
sys.dont_write_bytecode = True                                # write no cache files into the experiment folders
from run import FACTOR_SEED, real_profile                    # the measured-data run, unchanged
from recovery.model import Scenario, simulate_history, simulate_recovery
from recovery.optimization import BatterySpec

# the quantities compared at every step, the same as for the synthetic test set
COLUMNS = ["requested_charge_kw", "requested_discharge_kw", "charge_kw", "discharge_kw",
           "energy_after_kwh", "grid_import_kw", "unserved_kw", "step_cost"]
data = pd.read_csv(EXP / "real_data_validation/data/real_profiles.csv")
meta = json.loads((EXP / "real_data_validation/data/data_metadata.json").read_text())
plant = meta["plant"]
spec = BatterySpec(capacity_kwh=plant["capacity_kwh"], min_energy_kwh=plant["min_energy_kwh"],
                   max_charge_kw=plant["max_charge_kw"], max_discharge_kw=plant["max_discharge_kw"])
saved = pd.read_csv(EXP / "real_data_validation/results/controllers.csv").set_index(["scenario_id", "method"])

rng = np.random.default_rng(FACTOR_SEED)       # the same random draws, in the same order, as run.py
steps, largest = 0, 0.0
for i in range(int(meta["scenarios"])):
    s = Scenario(scenario_id=f"real_{i:03d}", seed=90000 + i, strength=float((.55, .8, .95)[i % 3]),
                 delay_steps=int(rng.choice([2, 4, 8])), grid_cap_kw=float(rng.choice(plant["grid_caps_kw"])),
                 deadline_steps=int(rng.choice([4, 8, 12])), tariff_scale=1.0,
                 initial_energy_kwh=plant["target_energy_kwh"], target_energy_kwh=plant["target_energy_kwh"],
                 loss_allowance_kw=plant["loss_allowance_kw"])
    p = real_profile(s, data[data.scenario_index == i])
    e_attacked = simulate_history(s, p, spec)[0]
    e_clean = simulate_history(s, p, spec, clean=True)[0]
    clean_states = simulate_recovery(s, p, spec, e_clean, "clean_reference")[2]
    actions = {}
    for method in ("attribution_mpc", "state_mpc"):
        outcome, rows, _, _ = simulate_recovery(s, p, spec, e_attacked, method,
                                                reference_energy=e_clean, reference_states=clean_states)
        old = saved.loc[(s.scenario_id, method)]
        if abs(outcome["future_cost"] - old.future_cost) > 1e-9 or \
                abs(outcome["final_energy_kwh"] - old.final_energy_kwh) > 1e-9:
            raise SystemExit(f"{s.scenario_id} {method}: the rerun does not reproduce the saved result")
        actions[method] = pd.DataFrame(rows)[COLUMNS].to_numpy()
    steps += len(actions["state_mpc"])
    largest = max(largest, float(np.abs(actions["attribution_mpc"] - actions["state_mpc"]).max()))

summary = {"scenarios": int(meta["scenarios"]), "compared_steps": steps, "largest_difference": largest,
           "identical": largest < 1e-9}
(ROOT / "assignment 3" / "build" / "measured_actions.json").write_text(json.dumps(summary, indent=1) + "\n")
print(summary)
