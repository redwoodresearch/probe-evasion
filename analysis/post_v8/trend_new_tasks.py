"""The trend figure for the three newer probes (m49, batches main + d: all 14 writers), at round 0 and after 16 rounds, in the
style of trend.png. The score is m49's headline: median (OFF - ON) over pairs with both answers correct, as a share of
(plain median - line). The OFF-referenced score is unstable on these probes (OFF sits close to the line).

    .venv/bin/python analysis/post_v8/trend_new_tasks.py
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from probe_evasion.missions import m49_figures

_spec = importlib.util.spec_from_file_location("post_v8_figures", Path(__file__).resolve().parent / "make_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

RUNS = [F.REPO / "missions/m49_new_task_toggle/runs" / batch for batch in ("main", "d")]  # 11 + 3 newest writers
TASKS = {"false_belief": "Belief tracking", "acrostic": "Acrostic hiding", "contradiction": "Spotting a contradiction"}
EFFORT_POINTS = (16, 0)
FIT_XS = np.linspace(131, 169, 50)
LABEL_OFFSETS = ((-8, 8, "right"), (6, 14, "left"), (6, -18, "left"))  # top three writers, spread so labels clear


def writer_meta(m45: dict) -> dict[str, dict]:
    return {r["writer"]: r for r in F.writer_rows(m45, "maths")}  # ECI and lab per writer (same writer ids as m49)


def rows_for(task: str, b: int, meta: dict) -> list[dict]:
    chains = m49_figures.frozen_all(RUNS, task, b)["writers"]
    rows = []
    for writer, ms in chains.items():
        values = [m["gap_closed"] for m in ms]
        assert len(values) == 3, (task, b, writer)
        rows.append({"writer": writer, "eci": meta[writer]["eci"], "lab": meta[writer]["lab"], "values": values,
                     "mean": float(np.mean(values))})
    return sorted(rows, key=lambda r: r["eci"])


def figure(b: int, meta: dict, b16_fits: dict) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), sharey=True)
    fits = {}
    for ax, (task, title) in zip(axes, TASKS.items()):
        rows = rows_for(task, b, meta)
        slope, (lo, hi) = F.two_stage_slope(rows)
        for r in rows:
            colour = F.LAB_COLOUR[r["lab"]]
            x = r["eci"] + F.DISPLAY_SHIFT.get(r["writer"], 0.0)
            ax.vlines(x, min(r["values"]), max(r["values"]), color=colour, lw=2, alpha=0.6)
            ax.plot(x, r["mean"], "o", ms=10, color=colour)
        for r, (dx, dy, ha) in zip(sorted(rows, key=lambda r: -r["mean"])[:3], LABEL_OFFSETS):
            ax.annotate(F.SHORT[r["writer"]], (r["eci"], r["mean"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=11, color=F.v6.MUTED)
        k, c0 = np.polyfit([r["eci"] for r in rows], [r["mean"] for r in rows], 1)
        ax.plot(FIT_XS, k * FIT_XS + c0, "-", color=F.v6.TEXT, lw=1.6, alpha=0.8)
        if task in b16_fits:
            ax.plot(FIT_XS, b16_fits[task][0] * FIT_XS + b16_fits[task][1], "--", color=F.v6.MUTED, lw=1.2, alpha=0.6)
        ax.axhline(0, color=F.v6.MUTED, lw=0.8)
        ax.set_title(f"{title}: {slope:+.3f} per 10 ECI [95%: {lo:+.3f}, {hi:+.3f}]", fontsize=14)
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
        ax.set_xlim(130, 170)
        ax.set_ylim(-0.2, 1.05)
        ax.grid(axis="y", color=F.v6.GRID)
        fits[task] = (k, c0)
        print(f"[{task}] B = {b}: slope {slope:+.3f} [{lo:+.3f}, {hi:+.3f}]; top: "
              + ", ".join(f"{F.SHORT[r['writer']]} {r['mean']:.2f}" for r in sorted(rows, key=lambda r: -r["mean"])[:3]))
    axes[0].set_ylabel("share of the plain-to-flag-line distance\nclosed by ON (median OFF − ON)")
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    handles.append(Line2D([], [], color=F.v6.TEXT, ls="-", label=f"fit, {'round 0' if b == 0 else 'after 16 rounds'}"))
    if b16_fits:
        handles.append(Line2D([], [], color=F.v6.MUTED, ls="--", label="fit after 16 rounds, for reference"))
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F.OUT / f"trend_new_b{b}.png", bbox_inches="tight")
    plt.close(fig)
    return fits


def main() -> None:
    meta = writer_meta(F.load_m45())
    b16 = figure(16, meta, {})
    figure(0, meta, b16)


if __name__ == "__main__":
    main()
