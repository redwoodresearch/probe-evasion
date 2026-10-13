"""Dot plot for James's methodology page: where replies sit against the 1% flag line on the m45 frozen test (B = 16)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("missions/m45_toggle_sweep/runs")
OUT = Path("analysis/for_james/methodology/figures")
PROBE = {"maths": "q27mrr@L48@s0", "ordering": "q27orr@L48@s0"}
LINE = {"maths": -6.53, "ordering": -8.06}  # m45 RESULT: 1% of look-alike replies score above
ROOT = {"maths": "qwen35_27b", "ordering": "qwen35_27b_ordering"}
WRITER = ("d", "fable51", "Claude Fable 5.1")  # top writer on both tasks
COLORS = {"neg": "#a7aeaa", "plain": "#3b4440", "off": "#3d5a80", "on": "#c2541b", "line": "#b3261e"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def strips(task: str) -> list[tuple[str, np.ndarray, str]]:
    probe, root = PROBE[task], ROOT[task]
    negatives = np.array([r["peaks"][probe] for r in read(RUNS / "c" / root / "calibration_scores.jsonl")])
    plain = np.array([r["peaks_visible"][probe] for r in read(RUNS / "c" / root / "test_checked" / "plain.jsonl") if r["valid"]])
    batch, writer, _ = WRITER
    rows = [r for r in read(RUNS / batch / root / "test_checked" / f"{writer}.jsonl") if "@B16#" in r["key"]]
    by_pair = {}
    for r in rows:
        arm, side = r["key"].rsplit("#", 1)
        by_pair.setdefault((arm, r["seed"], r["task"]["id"]), {})[side] = r
    pairs = [p for p in by_pair.values() if len(p) == 2 and p["on"]["valid"] and p["off"]["valid"]]
    off = np.array([p["off"]["peaks_visible"][probe] for p in pairs])
    on = np.array([p["on"]["peaks_visible"][probe] for p in pairs])
    return [("look-alike replies\n(set the line)", negatives, COLORS["neg"]),
            ("plain correct answers", plain, COLORS["plain"]),
            ("OFF replies", off, COLORS["off"]),
            ("ON replies", on, COLORS["on"])]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    for ax, task in zip(axes, ("maths", "ordering")):
        data = strips(task)
        for i, (label, xs, color) in enumerate(data):
            y = len(data) - 1 - i
            ax.scatter(xs, y + rng.uniform(-0.28, 0.28, len(xs)), s=4, color=color, alpha=0.25, linewidths=0)
            med = float(np.median(xs))
            ax.plot([med, med], [y - 0.38, y + 0.38], color=color, lw=3)
            ax.text(med, y + 0.42, f"{med:+.1f}", ha="center", va="bottom", fontsize=8, color=color)
        ax.axvline(LINE[task], color=COLORS["line"], lw=1.6)
        ax.text(LINE[task], -0.55, "← missed  |  flagged →", ha="center", va="bottom", fontsize=8, color=COLORS["line"])
        ax.set_yticks(range(len(data)), [d[0] for d in reversed(data)], fontsize=9)
        ax.set_ylim(-0.7, len(data) - 0.1)
        ax.set_title(f"{task.capitalize()}  (1% flag line {LINE[task]})", loc="left", fontsize=10, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("probe score (log-odds, read with the prompt in view); higher = 'doing the task'", fontsize=9)
    fig.suptitle(f"Frozen test, {WRITER[2]} after 16 rounds: pairs where both answers are correct; ticks = medians",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "dots.png", dpi=160)


if __name__ == "__main__":
    main()
