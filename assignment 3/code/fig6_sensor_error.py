"""Fig. 6: the 500 test scenarios rerun with a random error on every battery reading."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a3_style import ATTACKED, CLEAN, EXP, GREEN, GREY, PURPLE, SIMPLE, save

ETA_C = 0.95
noise = pd.read_csv(EXP / "noise_validation/results/episodes.csv")
prices = pd.read_csv(EXP / "expanded_validation/results/episodes.csv")[["scenario_id", "current_price"]]
noise = noise.merge(prices, on="scenario_id")
runs = json.loads((EXP / "noise_validation/results/summary_noise.json").read_text())
reserve = json.loads((EXP / "noise_validation/results/summary_reserve.json").read_text())

rng = np.random.default_rng(20261029)


def mean_and_interval(values):
    """Mean and 95% bootstrap interval (20,000 resamples of scenarios), as in Fig. 4."""
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def line(ax, x, stats, fmt, color, label, shift):
    """Points joined by a line with 95% error bars, moved sideways by shift so that two lines' bars do not overlap."""
    means = [s[0] for s in stats]
    errors = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
    ax.errorbar([v + shift for v in x], means, yerr=errors, fmt=fmt, color=color, ms=4, capsize=2.5,
                elinewidth=0.8, label=label)


sigmas, err_debt, err_price = [], [], []
for sigma, group in noise.groupby("sigma_kwh"):
    ok = group[group.actual_feasible & group.reference_feasible & group.realized_paired_success]
    sigmas.append(sigma)
    err_debt.append(mean_and_interval((ok.estimated_extra_cost - ok.realized_extra_cost).abs()))
    seen_gap = ok.e_reference_detect - ok.e_actual_observed          # gap as seen through the meter
    err_price.append(mean_and_interval((seen_gap * ok.current_price / ETA_C - ok.realized_extra_cost).abs()))

keys = ["0.0", "0.06", "0.12", "0.3", "0.6"]
fig, (left, right) = plt.subplots(1, 2, figsize=(6.9, 2.95), constrained_layout=True)

line(left, sigmas, err_debt, "o-", CLEAN, "Debt", -0.008)
line(left, sigmas, err_price, "s-", SIMPLE, "Energy deficit times price at detection", 0.008)
left.set_xlabel("Standard deviation of the meter error (kWh)")
left.set_ylabel("Mean absolute error (USD)")
left.set_title("(a) Error of the cost estimate", loc="left")
left.set_ylim(0, 0.5)
left.legend(loc="upper left", frameon=False)

right.plot(sigmas, [runs[k]["mpc_success"] for k in keys], "o-", color=GREY, ms=4,
           label="Planning controller, exact target")
right.plot(sigmas, [runs[k]["maxchg_success"] for k in keys], "s-", color=ATTACKED, ms=4,
           label="Charge at full power")
right.plot(sigmas, [reserve[k]["reserve2s_success"] for k in keys], "^-", color=GREEN, ms=4,
           label="Planning controller, reserve twice the error")
right.plot(sigmas, [reserve[k]["reservefix_success"] for k in keys], "D-", color=PURPLE, ms=3.5,
           label="Planning controller, fixed 0.25 kWh reserve")
right.set_xlabel("Standard deviation of the meter error (kWh)")
right.set_ylabel("Full recoveries (out of 500)")
right.set_title("(b) Recovery success", loc="left")
right.set_ylim(0, 500)
fig.legend(*right.get_legend_handles_labels(), loc="outside lower right", ncol=2, frameon=False)

save(fig, "fig6_sensor_error")
