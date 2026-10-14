"""Main figure for the post draft: one panel per probe (maths, ordering, belief tracking, contradiction), writer
capability on x, gap closed (plain-referenced) on y, after 16 rounds of feedback (filled, lab colours) and at round 0
(hollow), each with its least-squares fit. Values and slopes from the independent recomputation (numbers/numbers.json).

    .venv/bin/python analysis/post_v9/fig_2x2.py
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("post_v8_figures", HERE.parent / "post_v8" / "make_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

NUMBERS = json.loads((HERE / "numbers" / "numbers.json").read_text())["tasks"]
PANELS = {"maths": "Maths", "ordering": "Ordering", "false_belief": "Belief tracking", "contradiction": "Spotting a contradiction"}
ROUND0_COLOUR = "#8a949c"
FIT_XS = np.linspace(131, 169, 50)
LABELLED = {"fable51": ("Fable 5.1", (-10, 6, "right")), "opus55": ("Opus 5.5", (8, -20, "left")),
            "llama4maverick": ("Llama 4", (8, 8, "left"))}  # name, (dx, dy, ha) offsets so labels clear the dots


def fmt(fit: dict) -> str:
    lo, hi = fit["ci"]
    return f"{fit['slope']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main() -> None:
    meta = {r["writer"]: r for r in F.writer_rows(F.load_m45(), "maths")}  # ECI and lab per writer
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (task, title) in zip(axes.flat, PANELS.items()):
        t = NUMBERS[task]
        for b, filled in (("B0", False), ("B16", True)):
            means, chains = t[b]["writer_means"], t[b]["writer_chains"]
            xs, ys = [], []
            for w, m in means.items():
                x = meta[w]["eci"] + F.DISPLAY_SHIFT.get(w, 0.0)
                colour = F.LAB_COLOUR[meta[w]["lab"]] if filled else ROUND0_COLOUR
                ax.vlines(x, min(chains[w]), max(chains[w]), color=colour, lw=1.8, alpha=0.5 if filled else 0.35)
                ax.plot(x, m, "o", ms=10 if filled else 8, color=colour, mfc=colour if filled else "white", mew=1.6)
                xs.append(meta[w]["eci"]); ys.append(m)
                if filled and w in LABELLED:
                    name, (dx, dy, ha) = LABELLED[w]
                    ax.annotate(name, (x, m), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=11, color=F.v6.MUTED)
            k, c0 = np.polyfit(xs, ys, 1)
            ax.plot(FIT_XS, k * FIT_XS + c0, "-" if filled else "--", color=F.v6.TEXT if filled else ROUND0_COLOUR, lw=1.8)
        ax.axhline(0, color=F.v6.MUTED, lw=0.8)
        b16, b0 = t["B16"]["all14"], t["B0"]["all14"]
        ax.set_title(f"{title}\nafter 16 rounds {fmt(b16)}   ·   round 0 {fmt(b0)}", fontsize=13)
        ax.grid(axis="y", color=F.v6.GRID)
        ax.set_xlim(130, 170)
        ax.set_ylim(-0.15, 0.82)
    for ax in axes[-1]:
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of the plain-answer-to-flag-line\ndistance closed by ON")
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=8, color=ROUND0_COLOUR, mfc="white", label="round 0 (no feedback yet)"),
                Line2D([], [], color=F.v6.TEXT, ls="-", label="fit, after 16 rounds"),
                Line2D([], [], color=ROUND0_COLOUR, ls="--", label="fit, round 0")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.035, "Slopes per 10 ECI points [95% interval, resampling writers then searches]; bars = range over 3 searches; "
             "14 writers; Qwen3.5-27B; strict 1% flag line", ha="center", fontsize=11, color=F.v6.MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(HERE / "figures" / "trend_2x2.png", bbox_inches="tight")


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    main()
