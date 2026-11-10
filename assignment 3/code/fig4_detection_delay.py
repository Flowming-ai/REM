"""Fig. 4: effect of the detection delay on the 500 test scenarios."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a3_style import ATTACKED, CLEAN, EXP, SIMPLE, save

episodes = pd.read_csv(EXP / "expanded_validation/results/episodes.csv")     # all 500 scenarios
pred = pd.read_csv(EXP / "expanded_validation/test_predictions.csv")          # 329 usable for prediction
delays = [2, 4, 8]
ticks = ["30 min", "1 hour", "2 hours"]
rng = np.random.default_rng(20261029)


def mean_and_interval(values):
    """Mean and 95% bootstrap interval (20,000 resamples of scenarios)."""
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(20000, len(values)))].mean(axis=1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def bars(ax, x, stats, color, label=None, width=0.6):
    means = [s[0] for s in stats]
    errors = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
    ax.bar(x, means, width=width, color=color, yerr=errors, capsize=2.5, label=label,
           error_kw={"elinewidth": 0.8, "ecolor": "black"})


fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.3), constrained_layout=True)
x = np.arange(3)

gap = [mean_and_interval(episodes[episodes.delay_steps == d].soc_deficit_kwh) for d in delays]
bars(axes[0, 0], x, gap, ATTACKED)
axes[0, 0].set_ylabel("Energy deficit at detection (kWh)")
axes[0, 0].set_title("(a) Energy deficit caused by the attack", loc="left")

counts = [(int((episodes[episodes.delay_steps == d].feasibility_class != "both_feasible").sum()),
           int((episodes.delay_steps == d).sum())) for d in delays]
share = [100 * bad / total for bad, total in counts]
axes[0, 1].bar(x, share, width=0.6, color=ATTACKED)
for i, (bad, total) in enumerate(counts):
    axes[0, 1].text(i, share[i] + 1.2, f"{bad} of {total}", ha="center", fontsize=7)
axes[0, 1].set_ylim(0, 45)
axes[0, 1].set_ylabel("Scenarios (%)")
axes[0, 1].set_title("(b) Attacked run can no longer recover", loc="left")

debt = [mean_and_interval(pred[pred.delay_steps == d].prediction_counterfactual_cost) for d in delays]
real = [mean_and_interval(pred[pred.delay_steps == d].realized_extra_cost) for d in delays]
bars(axes[1, 0], x - 0.2, debt, CLEAN, "Debt computed at detection", width=0.38)
bars(axes[1, 0], x + 0.2, real, "#9BBFDB", "Realised extra cost", width=0.38)
axes[1, 0].set_ylabel("Extra recovery cost (USD)")
axes[1, 0].set_title("(c) Extra cost of recovery", loc="left")
axes[1, 0].legend(loc="upper left", frameon=False)

err_debt = [mean_and_interval(pred[pred.delay_steps == d].absolute_error_counterfactual_cost) for d in delays]
err_price = [mean_and_interval(pred[pred.delay_steps == d].absolute_error_current_price) for d in delays]
bars(axes[1, 1], x - 0.2, err_debt, CLEAN, "Debt", width=0.38)
for position, (mean, low, high) in zip(x - 0.2, err_debt):              # the debt's bars are short: print the values
    axes[1, 1].text(position, high + 0.015, f"{mean:.3f}", ha="center", fontsize=6.5)
bars(axes[1, 1], x + 0.2, err_price, SIMPLE, "Energy deficit times price at detection", width=0.38)
axes[1, 1].set_ylabel("Mean absolute error (USD)")
axes[1, 1].set_title("(d) Error of the estimate", loc="left")
axes[1, 1].legend(loc="upper left", frameon=False)
axes[1, 1].set_ylim(0, 0.8)

for ax in axes.flat:
    ax.set_xticks(x, ticks)
    ax.set_xlabel("Time from attack to detection")
    ax.grid(axis="x", visible=False)

save(fig, "fig4_detection_delay")
