"""Two figures from m45's frozen test (all 14 writers, mean of 3 searches each):
  rounds.png   probe-dodging score (S_offref) against feedback rounds, B = 0, 4, 8, 16, one line per writer;
  success.png  task success against probe-dodging score at B = 16, one dot per writer.

    .venv/bin/python analysis/post_v8/rounds_and_success.py
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

_spec = importlib.util.spec_from_file_location("post_v8_figures", Path(__file__).resolve().parent / "make_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

EFFORT_POINTS = ("0", "4", "8", "16")
ECI_NORM = Normalize(132, 168)
CMAP = cm.viridis
# Task success = share of pairs with both ON and OFF correct. Maths uses the correctness-only rule (any format counts);
# ordering's rule is already correctness (exact order).
SUCCESS_KEY = {"maths": "pair_valid_share_correct_only", "ordering": "pair_valid_share"}
LABELLED = ("fable51", "opus55", "opus48", "gpt6sol", "llama4maverick", "gpt5nano", "qwen35_9b")


LABEL_GAP = 0.03  # minimum vertical gap between right-end labels (score units)


def spread(ys: list[float]) -> list[float]:
    """Nudge sorted label positions apart so no two are closer than LABEL_GAP."""
    order = np.argsort(ys)
    placed = list(np.array(ys, dtype=float)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + LABEL_GAP)
    out = np.empty(len(ys))
    out[order] = placed
    return list(out)


def colour(eci: float):
    return CMAP(ECI_NORM(eci))


def colourbar(fig, axes) -> None:
    bar = fig.colorbar(cm.ScalarMappable(norm=ECI_NORM, cmap=CMAP), ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("writer capability (ECI)")


def fig_rounds(m45: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, task in zip(axes, F.TASKS):
        by_b = {b: {r["writer"]: r for r in F.writer_rows(m45, task, b=b)} for b in EFFORT_POINTS}
        writers = sorted(by_b["16"], key=lambda w: by_b["16"][w]["eci"])
        for writer in writers:
            ys = [by_b[b][writer]["mean"] for b in EFFORT_POINTS]
            ax.plot([int(b) for b in EFFORT_POINTS], ys, "-o", color=colour(by_b["16"][writer]["eci"]), lw=2, ms=5, alpha=0.9)
        for writer, y in zip(writers, spread([by_b["16"][w]["mean"] for w in writers])):
            ax.text(16.4, y, F.SHORT[writer], va="center", fontsize=9, color=colour(by_b["16"][writer]["eci"]))
        ax.axhline(1.0, color=F.v6.LINE_COLOR, ls="--", lw=1.4)
        ax.text(0.2, 0.96, "ON median on the strict 1% line", color=F.v6.LINE_COLOR, fontsize=12, va="top")
        ax.axhline(0, color=F.v6.MUTED, lw=0.8)
        ax.set_title(task.capitalize())
        ax.set_xticks([int(b) for b in EFFORT_POINTS])
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color=F.v6.GRID)
    axes[0].set_ylabel("probe-dodging score\n(share of OFF-to-line distance closed by ON)")
    fig.supxlabel("rounds of probe-score feedback before the template was frozen (frozen test)", y=-0.02)
    colourbar(fig, axes)
    fig.savefig(F.OUT / "rounds.png", bbox_inches="tight")


def fig_success(m45: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, task in zip(axes, F.TASKS):
        dodge = F.writer_rows(m45, task)
        success = {r["writer"]: r for r in F.writer_rows(m45, task, key=SUCCESS_KEY[task])}
        for r in dodge:
            s = success[r["writer"]]
            c = colour(r["eci"])
            ax.hlines(s["mean"], min(r["values"]), max(r["values"]), color=c, lw=1.5, alpha=0.5)
            ax.vlines(r["mean"], min(s["values"]), max(s["values"]), color=c, lw=1.5, alpha=0.5)
            ax.plot(r["mean"], s["mean"], "o", ms=10, color=c)
            if r["writer"] in LABELLED:
                ax.annotate(F.SHORT[r["writer"]], (r["mean"], s["mean"]), xytext=(6, 5), textcoords="offset points",
                            fontsize=9.5, color=F.v6.MUTED)
        rho = np.corrcoef([r["mean"] for r in dodge], [success[r["writer"]]["mean"] for r in dodge])[0, 1]
        print(f"[{task}] correlation of dodging score and task success across writers: r = {rho:+.2f}")
        ax.set_title(f"{task.capitalize()} (after 16 rounds)")
        ax.set_xlim(-0.05, 0.75)
        ax.set_ylim(0.3, 1.02)
        ax.grid(color=F.v6.GRID)
    axes[0].set_ylabel("task success\n(share of pairs with ON and OFF both correct)")
    fig.supxlabel("probe-dodging score (share of OFF-to-line distance closed by ON); bars = range over 3 searches", y=-0.02)
    colourbar(fig, axes)
    fig.savefig(F.OUT / "success.png", bbox_inches="tight")


def main() -> None:
    m45 = F.load_m45()
    fig_rounds(m45)
    fig_success(m45)


if __name__ == "__main__":
    main()
