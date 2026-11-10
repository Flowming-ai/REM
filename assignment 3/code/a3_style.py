"""Shared look of the Assignment 3 figures: fonts, colours and saving."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIGURES = ROOT / "assignment 3" / "figures"

ATTACKED = "#D55E00"   # orange: the run that received the false forecast
CLEAN = "#0072B2"      # blue: the run without the attack
SIMPLE = "#E69F00"     # yellow-orange: simple estimates
GREY = "#6B6B6B"
GREEN = "#009E73"
PURPLE = "#CC79A7"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E3E3E3",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
})


def save(fig, name):
    """Write the figure as PDF (for the report) and PNG (for checking)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
