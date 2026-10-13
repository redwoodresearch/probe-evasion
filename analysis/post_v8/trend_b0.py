"""The trend figure at B = 0: each writer's first template, written before any probe-score feedback, against ECI.
The B = 16 fit is drawn faintly for reference.

    .venv/bin/python analysis/post_v8/trend_b0.py
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_spec = importlib.util.spec_from_file_location("post_v8_figures", Path(__file__).resolve().parent / "make_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

ZERO_SHOT_B = "0"
FIT_XS = np.linspace(131, 169, 50)
LABELLED = ("fable51", "opus55", "opus48", "llama4maverick", "gpt6sol")


def main() -> None:
    m45 = F.load_m45()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, task in zip(axes, F.TASKS):
        rows = F.writer_rows(m45, task, b=ZERO_SHOT_B)
        slope, (lo, hi) = F.two_stage_slope(rows)
        for r in rows:
            colour = F.LAB_COLOUR[r["lab"]]
            x = r["eci"] + F.DISPLAY_SHIFT.get(r["writer"], 0.0)
            ax.vlines(x, min(r["values"]), max(r["values"]), color=colour, lw=2, alpha=0.6)
            ax.plot(x, r["mean"], "o", ms=10, color=colour)
            if r["writer"] in LABELLED:
                ax.annotate(F.SHORT[r["writer"]], (x, r["mean"]), xytext=(6, 6), textcoords="offset points", fontsize=11,
                            color=F.v6.MUTED)
        k, c0 = np.polyfit([r["eci"] for r in rows], [r["mean"] for r in rows], 1)
        ax.plot(FIT_XS, k * FIT_XS + c0, "-", color=F.v6.TEXT, lw=1.6, alpha=0.8)
        rows16 = F.writer_rows(m45, task)
        k16, c16 = np.polyfit([r["eci"] for r in rows16], [r["mean"] for r in rows16], 1)
        ax.plot(FIT_XS, k16 * FIT_XS + c16, "--", color=F.v6.MUTED, lw=1.2, alpha=0.6)
        ax.axhline(1.0, color=F.v6.LINE_COLOR, ls="--", lw=1.4)
        ax.text(131.5, 0.96, "ON median on the strict 1% line", color=F.v6.LINE_COLOR, fontsize=12, va="top")
        ax.axhline(0, color=F.v6.MUTED, lw=0.8)
        ax.set_title(f"{task.capitalize()}, round 0: {slope:+.3f} per 10 ECI [95%: {lo:+.3f}, {hi:+.3f}]", fontsize=15)
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
        ax.set_xlim(130, 170)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color=F.v6.GRID)
        print(f"[{task}] B = 0 slope {slope:+.3f} [{lo:+.3f}, {hi:+.3f}]; top: "
              + ", ".join(f"{F.SHORT[r['writer']]} {r['mean']:.2f}" for r in sorted(rows, key=lambda r: -r["mean"])[:4]))
    axes[0].set_ylabel("share of the OFF-to-flag-line distance\nclosed by the ON median")
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    handles += [Line2D([], [], color=F.v6.TEXT, ls="-", label="fit at round 0 (no feedback)"),
                Line2D([], [], color=F.v6.MUTED, ls="--", label="fit after 16 rounds, for reference")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F.OUT / "trend_b0.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
