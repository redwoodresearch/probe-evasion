"""Accuracy cost vs attack strength after 16 rounds, one panel per probe (post v9's four tasks). Per writer: attack
strength = gap closed (mean of 3 chains), cost = OFF correct share minus ON correct share in points (mean of chains with a
template; maths by m45's correctness-only re-judge, the others by their validity flag). Bars: range over the 3 chains.
Values from the independent recomputation (numbers/numbers.json).

    .venv/bin/python analysis/post_v9/fig_cost.py      # writes analysis/post_v9/figures/cost_2x2.png
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_2x2", HERE / "fig_2x2.py")
FIG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIG)
F, NUMBERS = FIG.F, FIG.NUMBERS
LABELLED = {**FIG.KEY_WRITERS, "llama4maverick": "Llama 4 Maverick"}
X_LIMITS = (-0.12, 0.7)
Y_LIMITS = {"maths": (-8, 48), "ordering": (-8, 48), "false_belief": (-12, 22), "contradiction": (-12, 22)}  # per row: room for labels


def panel(ax, task: str, title: str, meta: dict) -> None:
    t = NUMBERS[task]["B16"]
    strength, chains = t["writer_means"], t["writer_chains"]
    for w, cost in t["accuracy_cost"].items():
        colour = F.LAB_COLOUR[meta[w]["lab"]]
        x, y = strength[w], cost["cost_points"]
        ax.hlines(y, min(chains[w]), max(chains[w]), color=colour, lw=1.5, alpha=0.45)
        ax.vlines(x, min(cost["cost_points_chains"]), max(cost["cost_points_chains"]), color=colour, lw=1.5, alpha=0.45)
        ax.plot(x, y, "o", ms=9, color=colour, zorder=3)
    xs = [strength[w] for w in t["accuracy_cost"]]
    ys = [c["cost_points"] for c in t["accuracy_cost"].values()]
    texts = [ax.text(strength[w], t["accuracy_cost"][w]["cost_points"], LABELLED[w], fontsize=10, color=F.v6.TEXT, zorder=4)
             for w in LABELLED]
    # labels next to their dots, nudged off every dot and each other, with a thin pointer where moved
    adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.5, 2.0), force_text=(0.6, 0.9), force_static=(0.5, 0.8),
                arrowprops=dict(arrowstyle="-", color=F.v6.MUTED, lw=0.6))
    ax.axhline(0, color=F.v6.MUTED, lw=0.8)
    ax.axvline(0, color=F.v6.MUTED, lw=0.8)
    ax.set_title(title, fontsize=15)
    ax.grid(color=F.v6.GRID)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS[task])
    top = Y_LIMITS[task][1]
    for w, cost in t["accuracy_cost"].items():  # a search bar that runs off the top: say where it ends
        hi = max(cost["cost_points_chains"])
        if hi > top:
            ax.annotate(f"to {hi:.0f}", (strength[w], top), xytext=(4, -12), textcoords="offset points", fontsize=9, color=F.v6.MUTED)


def main() -> None:
    meta = {r["writer"]: r for r in F.writer_rows(F.load_m45(), "maths")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey="row", gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    for ax, (task, title) in zip(axes.flat, FIG.PANELS.items()):
        panel(ax, task, title, meta)
    for ax in axes[-1]:
        ax.set_xlabel("attack strength after 16 rounds (probe-dodging score)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy cost (points)\nOFF correct − ON correct", fontsize=12)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, 0.045, "One dot per writer (mean of 3 searches); bars = range over the 3 searches. The two rows have different y-scales. "
             "Maths correctness by m45's re-judge; other tasks by their validity check.", ha="center", fontsize=10.5, color=F.v6.MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.11)
    fig.savefig(HERE / "figures" / "cost_2x2.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
