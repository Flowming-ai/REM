"""Fig. 2: one development scenario from the attack to the end of recovery (scenario case_070)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from a3_style import ATTACKED, CLEAN, EXP, GREY, save

CASE = "case_070"
episodes = pd.read_csv(EXP / "results/main/episodes.csv")
info = episodes[episodes.scenario_id == CASE].iloc[0]
traces = pd.read_csv(EXP / "results/main/traces.csv")
case = traces[traces.scenario_id == CASE]

# Attacked run: the poisoned history before detection, then the standard planning controller.
attacked = pd.concat([case[case.method == "poisoned_history"], case[case.method == "state_mpc"]]).sort_values("t")
# Reference run (no attack): the clean history, then the same controller from the clean battery level.
clean = pd.concat([case[case.method == "clean_history"], case[case.method == "clean_reference"]]).sort_values("t")

hours = np.arange(len(attacked) + 1) * 0.25                     # 15-minute steps in hours
energy_a = np.append(attacked.energy_before_kwh.values, attacked.energy_after_kwh.values[-1])
energy_c = np.append(clean.energy_before_kwh.values, clean.energy_after_kwh.values[-1])
spent_a = np.append(0, attacked.step_cost.cumsum().values)
spent_c = np.append(0, clean.step_cost.cumsum().values)

price = attacked.true_price.values
step_hours = attacked.t.values * 0.25
t_attack, t_detect = int(info.attack_t), int(info.detection_t)
forecast = price * (1 - info.strength)                            # Eq. (5): every future price lowered
forecast[: t_attack + 1] = price[: t_attack + 1]                  # the current price stays correct

fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15), constrained_layout=True)
for ax in axes:
    ax.axvspan(t_attack * 0.25, t_detect * 0.25, color=ATTACKED, alpha=0.10, lw=0)
    ax.axvspan(20 * 0.25, 24 * 0.25, color=GREY, alpha=0.15, lw=0)
    ax.axvline(t_detect * 0.25, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Hours from start")
    ax.set_xlim(0, 10)

axes[0].step(step_hours, price, where="post", color="black", lw=1.2, label="True price")
axes[0].step(step_hours[t_attack:], forecast[t_attack:], where="post", color=ATTACKED, ls="--",
             label="Forecast received when the attack starts")
axes[0].set_ylabel("Electricity price (USD per kWh)")
axes[0].set_title("(a) Price forecast", loc="left")

axes[1].plot(hours, energy_c, color=CLEAN, label="Reference run (no attack)")
axes[1].plot(hours, energy_a, color=ATTACKED, label="Attacked run")
axes[1].axhline(info.target_energy_kwh, color=GREY, lw=0.8, ls=":")
axes[1].text(9.9, info.target_energy_kwh + 0.25, "Target", ha="right", color=GREY, fontsize=7)
axes[1].set_ylabel("Stored energy (kWh)")
axes[1].set_ylim(0, 14.6)
axes[1].text(t_attack * 0.25 + 0.05, 13.4, "Attack", color=ATTACKED, fontsize=7)
axes[1].text(t_detect * 0.25 + 0.08, 13.4, "Detected", fontsize=7)
axes[1].text(5.5, 12.4, "Grid\noutage", color=GREY, fontsize=7, ha="center", va="top")
axes[1].set_yticks([0, 3, 6, 9, 12])
axes[1].set_title("(b) Stored energy", loc="left")

axes[2].plot(hours, spent_c, color=CLEAN, label="Reference run (no attack)")
axes[2].plot(hours, spent_a, color=ATTACKED, label="Attacked run")
axes[2].set_ylabel("Money spent so far (USD)")
axes[2].set_title("(c) Spending", loc="left")
handles_a, labels_a = axes[0].get_legend_handles_labels()
handles_b, labels_b = axes[1].get_legend_handles_labels()
fig.legend(handles_a + handles_b, labels_a + labels_b, loc="outside lower center", ncol=4, frameon=False)

save(fig, "fig2_example_scenario")
