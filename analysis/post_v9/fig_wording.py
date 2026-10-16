"""How much of each task's capability trend is wording: one bar per task, its length the probe-evasion trend (slope per
10 ECI, round 16), split into the part a TF-IDF word classifier's shift predicts and the rest (the two add up to the
trend exactly; numbers/wording_share.py). Whisker: 95% interval on the whole trend. Clean style as fig_method.py.

    .venv/bin/python analysis/post_v9/fig_wording.py   # writes figures/wording_share.png
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_method", HERE / "fig_method.py")
METHOD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METHOD)

SHARE = json.loads((HERE / "numbers" / "wording_share.json").read_text())
TASKS = {"maths": "Arithmetic", "false_belief": "Belief tracking", "contradiction": "Spotting a\ncontradiction", "ordering": "Ordering"}  # most wording first
WORDING, REST = "#5fb3c2", "#3b4440"  # the word classifier's teal (as the comparison figures); the probe's ink
BAR_H = 0.56


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for y, (task, label) in enumerate(reversed(TASKS.items())):
        r = SHARE[task]
        total, words = r["trend_raw"], r["trend_predicted_by_wording"]["slope"]
        rest = total["slope"] - words
        ax.barh(y, words, height=BAR_H, color=WORDING, edgecolor="white", lw=2)
        ax.barh(y, rest, left=words, height=BAR_H, color=REST, edgecolor="white", lw=2)
        share = words / total["slope"]
        if share >= 0.12:
            ax.text(words / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=13, color=METHOD.TEXT, weight="bold")
        if 1 - share >= 0.12:
            ax.text(words + rest / 2, y, f"{1 - share:.0%}", ha="center", va="center", fontsize=13, color="white", weight="bold")
        wy = y - BAR_H / 2 - 0.09  # the whisker runs just under its bar, clear of the percentages
        ax.plot(total["ci"], [wy, wy], color=METHOD.MUTED, lw=1.4)
        for x in total["ci"]:
            ax.plot([x, x], [wy - 0.06, wy + 0.06], color=METHOD.MUTED, lw=1.4)
        ax.text(-0.004, y, label, ha="right", va="center", fontsize=14, color=METHOD.TEXT, linespacing=1.15)
    ax.set_yticks([])
    ax.set_xlim(0, 0.19)
    ax.set_ylim(-0.75, len(TASKS) - 0.4)
    ax.set_xlabel("capability trend in probe evasion (score per 10 ECI)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=WORDING, label="predicted by a word classifier's shift (wording)"),
               plt.Rectangle((0, 0), 1, 1, color=REST, label="not predicted by it"),
               plt.Line2D([], [], color=METHOD.MUTED, lw=1.4, label="95% interval on the whole trend")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=11.5, handlelength=1.6)
    fig.savefig(HERE / "figures" / "wording_share.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
