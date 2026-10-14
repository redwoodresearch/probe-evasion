"""fig_probe_vs_B.png from results.json (run balanced_baseline.py first). Per task: left, held-out AUROC on three sets
for the balanced broad probe and TF-IDF B; right, attack carry-over (gap closed per writer, mean of 3 chains) against
writer ECI, with least-squares lines and the two-stage bootstrap interval of the slope.

    .venv/bin/python analysis/tfidf_baseline/balanced/figure.py
"""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("tfidf_figures", HERE.parent / "figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
_spec = importlib.util.spec_from_file_location("post_numbers", HERE.parents[1] / "post_v9/numbers/compute.py")
NUM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NUM)

STYLE = {"probe": ("Balanced broad probe", F.STYLE["probe"][1], "o"), "tfidf_B": ("TF-IDF B (request + reply)", F.STYLE["tfidf_B"][1], "s")}
AUROC_SETS = (("held-out chat\n(computed vs given)", "auroc_chat_heldout_pos_vs_neg"),
              ("original held-out\n(+ re-read copies)", "auroc_original_heldout_and_reread"),
              ("held-out chat positives\nvs WildChat pool", "auroc_chat_heldout_pos_vs_wildchat_line_pool"))
AUROC_FLOOR = 0.9


def panel_auroc(ax, res: dict) -> None:
    x = np.arange(len(AUROC_SETS))
    width = 0.36
    for j, (det, (label, colour, _)) in enumerate(STYLE.items()):
        ys = [res[det][key] for _, key in AUROC_SETS]
        xs = x + (j - 0.5) * width
        ax.bar(xs, [y - AUROC_FLOOR for y in ys], width * 0.92, bottom=AUROC_FLOOR, color=colour, label=label)
        for xi, y in zip(xs, ys):
            ax.text(xi, y + 0.003, f"{y:.3f}", ha="center", va="bottom", fontsize=7.5, color=F.INK)
    ax.set_xticks(x, [s for s, _ in AUROC_SETS], fontsize=8)
    ax.set_ylim(AUROC_FLOOR, 1.015)
    ax.set_yticks(np.arange(AUROC_FLOOR, 1.001, 0.02))
    ax.set_ylabel(f"held-out AUROC (axis starts at {AUROC_FLOOR})", fontsize=8)
    ax.grid(axis="y", color=F.GRID, lw=0.6)


def panel_trend(ax, res: dict) -> None:
    for det, (label, colour, marker) in STYLE.items():
        s = res[det]
        writers = sorted(s["writer_means"], key=NUM.ECI.get)
        x = np.array([NUM.ECI[w] for w in writers])
        y = np.array([s["writer_means"][w] for w in writers])
        ax.plot(x, y, marker=marker, ms=7, color=colour, ls="", mec=F.SURFACE, mew=1.2)
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 2)
        lo, hi = s["slope"]["ci"]
        ax.plot(xs, slope * xs + icpt, color=colour, lw=2,
                label=f"{label.split(' (')[0]}: {10 * slope:+.2f} per 10 ECI [{lo:+.2f}, {hi:+.2f}]")
    ax.axhline(0, color=F.GRID, lw=1)
    ax.set_xlabel("writer capability (ECI)", fontsize=8)
    ax.set_ylabel("gap closed (writer mean of 3 chains)\n0 = ON reads like OFF, 1 = ON median at the 1% line", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(color=F.GRID, lw=0.6)


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), gridspec_kw={"width_ratios": [0.8, 1]})
    fig.patch.set_facecolor(F.SURFACE)
    for row, task in zip(axes, ("maths", "ordering")):
        for ax in row:
            ax.set_facecolor(F.SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        panel_auroc(row[0], results[task])
        panel_trend(row[1], results[task])
        row[0].set_title(f"{F.TASK_LABEL[task]}: held-out AUROC", fontsize=10, loc="left", color=F.INK)
        row[1].set_title(f"{F.TASK_LABEL[task]}: m45 round-16 attacks (written against the old probe)", fontsize=10, loc="left", color=F.INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in STYLE.values()]
    fig.legend(handles, [label for label, _, _ in STYLE.values()], loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = HERE / "fig_probe_vs_B.png"
    fig.savefig(path, dpi=130, facecolor=F.SURFACE)
    print("wrote", path)


if __name__ == "__main__":
    main()
