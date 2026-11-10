#!/usr/bin/env python3
"""Build a measured-data scenario set and size the plant to it.

Load and generation come from the Open Power System Data household package at 15-minute resolution, which
reports cumulative energy counters for individual German buildings. Prices come from the Octopus Agile
half-hourly tariff published through the Octopus Energy public API, a settled dynamic retail tariff that
includes negative intervals.

Measured power is used as recorded; no series is rescaled. A single building's demand has a peak-to-mean
ratio near 11, against 1.3 for the synthetic generator, so one building cannot ride through the announced
outage on any plausibly sized battery. Aggregating the available buildings reduces that ratio to about 5,
which is the community level that this literature models. The plant is then sized to the measured community
rather than the measured community being scaled into a plant sized for smooth synthetic demand.

Declared rules, fixed before any result is read:
  load       sum of all residential grid-import series available in the package
  generation sum of all residential PV series available in the package
  period     the longest continuous 15-minute segment on which every selected series is present
  windows    the first N non-overlapping 40-step windows of that segment
  prices     the first N non-overlapping 20-half-hour windows of the Agile series, chronological
  plant      battery power, capacity, minimum energy, recovery target and grid caps set to the same
             multiples of the 99th-percentile load that the synthetic study uses of its own
  forecast   climatological nominal by time of day over the whole segment, plus the current residual
             decayed by 0.65 per step; no future value of the scenario window is used
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_SCENARIOS = 400
HORIZON = 40

# Plant-to-load ratios of the synthetic study, measured against its own 99th-percentile load (4.196 kW).
R_BATTERY_POWER = 0.953      # x p99 load
R_CAPACITY_H = 2.860         # hours of p99 load
R_MIN_ENERGY_H = 0.286
R_TARGET_H = 2.145
R_GRID_CAPS = (1.311, 1.668, 2.145)


def build(raw_opsd: Path, raw_prices: Path, out: Path):
    head = pd.read_csv(raw_opsd, nrows=0).columns
    load_cols = sorted(c for c in head if "residential" in c and c.endswith("_grid_import"))
    pv_cols = sorted(c for c in head if "residential" in c and c.endswith("_pv"))
    df = pd.read_csv(raw_opsd, usecols=["utc_timestamp"] + load_cols + pv_cols)
    df["t"] = pd.to_datetime(df.utc_timestamp)
    df = df.sort_values("t").reset_index(drop=True)

    # cumulative kWh counters -> average power over each 15-minute interval, then aggregate
    rate = lambda c: df[c].diff() * 4.0
    load = sum(rate(c) for c in load_cols)
    pv = sum(rate(c) for c in pv_cols)
    w = pd.DataFrame({"t": df.t, "load_kw": load, "pv_kw": pv}).dropna()
    w = w[(w.load_kw >= 0) & (w.pv_kw >= 0) & (w.load_kw < 200) & (w.pv_kw < 200)].reset_index(drop=True)

    w["seg"] = (w.t.diff().dt.total_seconds() != 900).cumsum()
    seg = w.seg.value_counts().idxmax()
    w = w[w.seg == seg].reset_index(drop=True)

    p99 = float(np.percentile(w.load_kw, 99))
    plant = {
        "p99_load_kw": p99,
        "max_charge_kw": R_BATTERY_POWER * p99, "max_discharge_kw": R_BATTERY_POWER * p99,
        "capacity_kwh": R_CAPACITY_H * p99, "min_energy_kwh": R_MIN_ENERGY_H * p99,
        "target_energy_kwh": R_TARGET_H * p99,
        "grid_caps_kw": [r * p99 for r in R_GRID_CAPS],
        "loss_allowance_kw": 1.0 * p99 / 4.196,          # same fraction of p99 as the synthetic 1 kW
    }

    # climatological nominal by time of day over the whole segment
    w["tod"] = w.t.dt.hour * 4 + w.t.dt.minute // 15
    nom = w.groupby("tod")[["load_kw", "pv_kw"]].mean().rename(
        columns={"load_kw": "load_nom", "pv_kw": "pv_nom"})
    w = w.join(nom, on="tod")

    prices = json.loads(raw_prices.read_text())
    prices.sort(key=lambda r: r["valid_from"])
    half = np.array([r["value_inc_vat"] for r in prices], dtype=float) / 100.0
    per_scenario_half = HORIZON // 2
    n = min(N_SCENARIOS, len(w) // HORIZON, len(half) // per_scenario_half)

    rows = []
    for i in range(n):
        win = w.iloc[i * HORIZON:(i + 1) * HORIZON]
        tariff = np.repeat(half[i * per_scenario_half:(i + 1) * per_scenario_half], 2)
        for k in range(HORIZON):
            r = win.iloc[k]
            rows.append({"scenario_index": i, "k": k, "utc_timestamp": r.t.isoformat(),
                         "load_kw": float(r.load_kw), "pv_kw": float(r.pv_kw),
                         "load_nom_kw": float(r.load_nom), "pv_nom_kw": float(r.pv_nom),
                         "tariff_gbp_per_kwh": float(tariff[k])})
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "real_profiles.csv", index=False)

    meta = {
        "load_series": load_cols, "pv_series": pv_cols,
        "segment_start_utc": str(w.t.min()), "segment_end_utc": str(w.t.max()),
        "segment_steps": int(len(w)), "scenarios": int(n), "horizon": HORIZON,
        "rescaling": "none; measured power is used as recorded",
        "load_mean_kw": float(w.load_kw.mean()), "load_p99_kw": p99,
        "load_max_kw": float(w.load_kw.max()),
        "load_peak_to_mean": float(w.load_kw.max() / w.load_kw.mean()),
        "pv_mean_kw": float(w.pv_kw.mean()), "pv_max_kw": float(w.pv_kw.max()),
        "plant": plant,
        "price_source": "Octopus Agile AGILE-24-10-01 region C, value_inc_vat, p/kWh to GBP/kWh",
        "price_first_valid_from": prices[0]["valid_from"], "price_last_valid_from": prices[-1]["valid_from"],
        "price_min_gbp": float(half.min()), "price_max_gbp": float(half.max()),
        "price_mean_gbp": float(half.mean()), "negative_price_halfhours": int((half < 0).sum()),
        "forecast": "climatological nominal by time of day plus current residual decayed by 0.65 per step",
        "data_type": "measured",
        "opsd_sha256": hashlib.sha256(raw_opsd.read_bytes()).hexdigest(),
        "prices_sha256": hashlib.sha256(raw_prices.read_bytes()).hexdigest(),
    }
    (out / "data_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {len(frame)} rows for {n} scenarios -> {out/'real_profiles.csv'}")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]), HERE / "data")
