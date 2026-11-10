"""Fig. 5: extra recovery cost from 4 kWh against a reference run at 9 kWh (120 settings)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from a3_style import EXP, save

sweep = pd.read_csv(EXP / "results/main/feasibility_sweep.csv")
table = sweep.pivot(index="grid_cap_kw", columns="deadline_hours", values="extra_cost")
limits, deadlines = table.index.values, table.columns.values
cost = np.ma.masked_invalid(table.values)              # empty cells: the attacked run cannot recover

fig, ax = plt.subplots(figsize=(3.5, 2.75), constrained_layout=True)
ax.set_facecolor("#EFEFEF")
x_edges = np.append(deadlines - 0.25, deadlines[-1] + 0.25)
y_edges = np.append(limits - 0.25, limits[-1] + 0.25)
mesh = ax.pcolormesh(x_edges, y_edges, cost, cmap="Blues", vmin=1.8, vmax=2.8, shading="flat")
for i, limit in enumerate(limits):
    for j, deadline in enumerate(deadlines):
        if np.ma.is_masked(cost[i, j]):
            ax.add_patch(plt.Rectangle((deadline - 0.25, limit - 0.25), 0.5, 0.5, fill=False,
                                       hatch="////", edgecolor="#9A9A9A", lw=0))
ax.set_xticks(deadlines)
ax.set_yticks(limits)
ax.set_xlabel("Time allowed after detection (hours)")
ax.set_ylabel("Grid import limit (kW)")
ax.grid(False)
bar = fig.colorbar(mesh, ax=ax, pad=0.02)
bar.set_label("Extra recovery cost (USD)")
ax.legend(handles=[Patch(facecolor="#EFEFEF", hatch="////", edgecolor="#9A9A9A",
                         label="Attacked run cannot recover")],
          loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)

save(fig, "fig5_recovery_boundary")
