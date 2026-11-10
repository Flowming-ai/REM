"""Fig. 3: how well the debt and the price-at-detection estimate predict the realised extra cost (329 test scenarios)."""
import pandas as pd
import matplotlib.pyplot as plt

from a3_style import CLEAN, EXP, GREY, SIMPLE, save

pred = pd.read_csv(EXP / "expanded_validation/test_predictions.csv")
low = pred.prediction_current_price < 0.7 * pred.realized_extra_cost       # the group discussed in the text

fig, ax = plt.subplots(figsize=(3.5, 3.1), constrained_layout=True)
top = 1.05 * max(pred.realized_extra_cost.max(), pred.prediction_current_price.max(),
                 pred.prediction_counterfactual_cost.max())
ax.plot([0, top], [0, top], color=GREY, lw=0.8, ls="--", label="Perfect prediction")
ax.plot([0, top], [0, 0.7 * top], color=GREY, lw=0.6, ls=":", label="70% of the realised extra cost")
ax.scatter(pred.realized_extra_cost[~low], pred.prediction_current_price[~low], s=10, color=SIMPLE, alpha=0.75,
           edgecolor="none", label="Energy deficit times price at detection")
ax.scatter(pred.realized_extra_cost[low], pred.prediction_current_price[low], s=12, facecolor="none",
           edgecolor=SIMPLE, lw=0.7, label=f"Same estimate, below 70% ({low.sum()} scenarios)")
ax.scatter(pred.realized_extra_cost, pred.prediction_counterfactual_cost, s=10, color=CLEAN, edgecolor="none",
           zorder=3, label="Debt (difference of optimal recovery costs)")
ax.set_xlim(-0.3, top)
ax.set_ylim(-0.3, top)
ax.set_xlabel("Realised extra cost after detection (USD)")
ax.set_ylabel("Estimate made at detection (USD)")
ax.legend(loc="upper left", frameon=False, markerscale=1.8, fontsize=6.5)

save(fig, "fig3_prediction")
