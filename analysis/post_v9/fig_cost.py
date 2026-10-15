"""Accuracy cost vs attack strength after 16 rounds, one panel per probe (post v9's four tasks). Per writer: attack
strength = gap closed (mean of 3 chains), cost = OFF correct share minus ON correct share in points (mean of chains with a
template; arithmetic by a correctness-only re-check, the others by their validity flag). Bars: range over the 3 chains.
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
CLIPPED_LABEL_OFFSET = (-0.03, 6)  # (score units, points) from the dot to the label's bottom-right corner
Y_LIMITS = (-12, 48)
Y_LABEL = "accuracy cost (percentage points)"
Y_LABEL_NOTE = "% correct with attack OFF − % correct with attack ON"  # smaller, beside the main label
Y_NOTE_X = -0.085  # axes fraction, between the tick labels and the main label  # one scale for all four panels, so tasks compare directly


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
    # a writer whose search bar runs off the top is labelled by hand, up and to the left (empty space), saying where the bar
    # ends; adjustText would push that label out of the panel
    top = Y_LIMITS[1]
    clipped = {w for w in LABELLED if max(t["accuracy_cost"][w]["cost_points_chains"]) > top}
    for w in clipped:
        x, y = strength[w], t["accuracy_cost"][w]["cost_points"]
        hi = max(t["accuracy_cost"][w]["cost_points_chains"])
        ax.annotate(f"{LABELLED[w]}\n(bar runs to {hi:.0f})", (x, y), xytext=(x + CLIPPED_LABEL_OFFSET[0], y + CLIPPED_LABEL_OFFSET[1]),
                    ha="right", va="bottom", fontsize=11.5, color=F.v6.TEXT, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=F.v6.MUTED, lw=0.6, relpos=(1, 0), shrinkB=6))
    texts = [ax.text(strength[w], t["accuracy_cost"][w]["cost_points"], LABELLED[w], fontsize=11.5, color=F.v6.TEXT, zorder=4)
             for w in LABELLED if w not in clipped]
    # labels next to their dots, nudged off every dot and each other, with a thin pointer where moved
    adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.5, 2.0), force_text=(0.6, 0.9), force_static=(0.5, 0.8),
                arrowprops=dict(arrowstyle="-", color=F.v6.MUTED, lw=0.6))
    ax.axhline(0, color=F.v6.MUTED, lw=0.8)
    ax.axvline(0, color=F.v6.MUTED, lw=0.8)
    ax.set_title(title, fontsize=17)
    ax.tick_params(labelsize=12)
    ax.grid(color=F.v6.GRID)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)


def main() -> None:
    np.random.seed(0)  # adjustText nudges overlapping labels with numpy's global RNG; fix it so reruns match
    meta = {r["writer"]: r for r in F.writer_rows(F.load_m45(), "maths")}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True, gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    for ax, (task, title) in zip(axes.flat, FIG.PANELS.items()):
        panel(ax, task, title, meta)
    for ax in axes[-1]:
        ax.set_xlabel("probe-evasion score after 16 rounds", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL, fontsize=15, labelpad=24)
        ax.text(Y_NOTE_X, 0.5, Y_LABEL_NOTE, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=11,
                color=F.v6.MUTED)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, 0.045, "One dot per writer (average of 3 searches); bars show the range across the 3 searches. Above 0: the attack "
             "makes Qwen's answers wrong more often.", ha="center", fontsize=11.5, color=F.v6.MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.11)
    fig.savefig(HERE / "figures" / "cost_2x2.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
