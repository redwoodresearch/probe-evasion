"""Figures for the TF-IDF baseline, from results/<task>.json (run tfidf_baseline.py first).

compare.png (maths, ordering) and compare_new_tasks.png (belief tracking, acrostic, contradiction). Per task:
- left: each detector's median score on five sets, placed on its own scale (0 = its 1% flag line, 1 = its plain
  answers' median), so detectors with different score ranges share one axis;
- right: the attack headline S_offref per writer (mean over 3 chains) against writer ECI, probe vs TF-IDF B, with
  least-squares lines.

Run from the repo root:  .venv/bin/python analysis/tfidf_baseline/figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {"probe": ("Probe", "#2a78d6", "o"), "tfidf_B": ("TF-IDF B (prompt + reply)", "#eb6834", "s"),
         "tfidf_A": ("TF-IDF A (reply only)", "#1baf7a", "^")}
TASK_LABEL = {"maths": "Maths", "ordering": "Ordering", "false_belief": "Belief tracking", "acrostic": "Acrostic",
              "contradiction": "Contradiction"}
SETS = (("look-alike\npool", "calibration_median"), ("everyday\nchat", "chat_median"), ("OFF\n(attack)", "off"),
        ("ON\n(attack)", "on"), ("plain\nanswers", "plain_median"))
Y_LO, Y_HI = -1.6, 1.35


def median_of(d: dict, key: str) -> float | None:
    if key in ("on", "off"):
        return d["attacks"][f"{key}_median"]
    return d.get(key)


def panel_medians(ax, res: dict) -> None:
    x = np.arange(len(SETS))
    offsets = {"probe": -0.2, "tfidf_B": 0.0, "tfidf_A": 0.2}
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.text(len(SETS) - 0.5, 0.04, "1% flag line", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.axhline(1, color=GRID, lw=1)
    for det, (label, colour, marker) in STYLE.items():
        d = res[det]
        gap = d["plain_median"] - d["line_1pct"]
        if gap <= 0:  # plain answers at or under its own line: no scale to place it on
            ax.plot([], [], marker=marker, color=colour, ls="", label=f"{label}: plain under its line, not shown")
            continue
        ys = [(m - d["line_1pct"]) / gap if (m := median_of(d, key)) is not None else np.nan for _, key in SETS]
        shown = np.clip(ys, Y_LO, Y_HI)
        ax.plot(x + offsets[det], shown, marker=marker, ms=8, color=colour, ls="", label=label, mec=SURFACE, mew=1.5)
        for xi, y, s in zip(x + offsets[det], ys, shown):
            if np.isfinite(y) and y != s:
                ax.annotate(f"{y:.1f}", (xi, s), textcoords="offset points", xytext=(-4 if offsets[det] <= 0 else 4, -12 if y < 0 else 6),
                            ha="right" if offsets[det] <= 0 else "left",
                            fontsize=7, color=MUTED)
    ax.set_xticks(x, [s for s, _ in SETS], fontsize=8)
    ax.set_ylim(Y_LO - 0.25, Y_HI + 0.15)
    ax.set_ylabel("median, on the detector's own scale\n(0 = 1% line, 1 = plain median)", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)


def panel_trend(ax, res: dict) -> None:
    for det in ("probe", "tfidf_B"):
        label, colour, marker = STYLE[det]
        a = res[det]["attacks"]
        writers = sorted(a["writer_means"], key=lambda w: a["writer_eci"][w])
        x = np.array([a["writer_eci"][w] for w in writers])
        y = np.array([a["writer_means"][w] for w in writers])
        ax.plot(x, y, marker=marker, ms=7, color=colour, ls="", mec=SURFACE, mew=1.2)
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 2)
        lo, hi = a["slope_ci95_two_stage"]
        ax.plot(xs, slope * xs + icpt, color=colour, lw=2,
                label=f"{label.split(' (')[0]}: {10 * slope:+.2f} per 10 ECI [{lo:+.2f}, {hi:+.2f}]")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("writer capability (ECI)", fontsize=8)
    ax.set_ylabel("attack strength S (writer mean of 3 chains)\n0 = ON reads like OFF, 1 = ON median at the line", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.6)


def figure(tasks: list[str], path: Path) -> None:
    fig, axes = plt.subplots(len(tasks), 2, figsize=(13, 4.2 * len(tasks)), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    for row, task in zip(np.atleast_2d(axes), tasks):
        res = json.loads((HERE / "results" / f"{task}.json").read_text())
        for ax in row:
            ax.set_facecolor(SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        panel_medians(row[0], res)
        panel_trend(row[1], res)
        row[0].set_title(f"{TASK_LABEL[task]}: where each set sits against each detector's own line", fontsize=10, loc="left", color=INK)
        row[1].set_title(f"{TASK_LABEL[task]}: frozen attacks (B = 16), written against the probe", fontsize=10, loc="left", color=INK)
        row[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    figure(["maths", "ordering"], OUT / "compare.png")
    figure(["false_belief", "acrostic", "contradiction"], OUT / "compare_new_tasks.png")
