"""Dot plots for James's methodology page, one per probe: where replies sit against the 1% flag line on the frozen
test at B = 16 (maths / ordering: m45; belief tracking / acrostic / contradiction: m49).

    .venv/bin/python analysis/for_james/methodology/make_dots.py
"""

import json
from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np

M45 = Path("missions/m45_toggle_sweep/runs")
M49 = Path("missions/m49_new_task_toggle/runs/main")
OUT = Path("analysis/for_james/methodology/figures")
B = 16
FALSE_ALARM = 0.01
LINE_TOLERANCE = 0.1
COLORS = {"neg": "#a7aeaa", "plain": "#3b4440", "off": "#3d5a80", "on": "#c2541b", "line": "#b3261e"}


@attrs.frozen
class TaskConfig:
    title: str
    probe: str
    calibration: Path  # calibration_scores.jsonl: the look-alike replies that set the line
    plain: Path  # frozen-test plain replies (no instruction)
    writer_rows: Path  # the writer's frozen-test rows (all chains)
    writer_label: str
    reported_line: float  # the line the mission reported; the recomputed one must match it


def m45_task(title: str, probe: str, root: str, reported_line: float) -> TaskConfig:
    # calibration and plain from batch c, the writer from batch d (Fable 5.1 is the top writer on both tasks)
    return TaskConfig(title, probe, M45 / "c" / root / "calibration_scores.jsonl", M45 / "c" / root / "test_checked" / "plain.jsonl",
                      M45 / "d" / root / "test_checked" / "fable51.jsonl", "Claude Fable 5.1", reported_line)


def m49_task(title: str, probe: str, task: str, writer: str, writer_label: str, reported_line: float) -> TaskConfig:
    # writer: top mean gap closed over 3 chains at B = 16 (m49_figures.frozen_writers)
    root = M49 / f"m49_{task}"
    return TaskConfig(title, probe, root / "calibration_scores.jsonl", root / "test" / "plain.jsonl",
                      root / "test_checked" / f"{writer}.jsonl", writer_label, reported_line)


TASKS = {
    "maths": m45_task("Maths", "q27mrr@L48@s0", "qwen35_27b", -6.53),
    "ordering": m45_task("Ordering", "q27orr@L48@s0", "qwen35_27b_ordering", -8.06),
    "belief_tracking": m49_task("Belief tracking", "m48fa_v2@L48@s0", "false_belief", "opus48", "Claude Opus 4.8", -2.39),
    "acrostic": m49_task("Acrostic hiding", "m48ac_v2@L48@s0", "acrostic", "gemini38flash", "Gemini 3.8 Flash", -2.55),
    "contradiction": m49_task("Spotting a contradiction", "m48co_v2@L48@s0", "contradiction", "opus48", "Claude Opus 4.8", -0.58),
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def flag_line(cfg: TaskConfig) -> float:
    line = float(np.quantile([r["peaks"][cfg.probe] for r in read(cfg.calibration)], 1 - FALSE_ALARM))
    assert abs(line - cfg.reported_line) < LINE_TOLERANCE, (cfg.title, line, cfg.reported_line)
    return line


def valid_pairs(cfg: TaskConfig) -> list[dict]:
    by_pair: dict = {}
    for r in read(cfg.writer_rows):
        arm, side = r["key"].rsplit("#", 1)
        if arm.endswith(f"@B{B}"):
            by_pair.setdefault((arm, r["seed"], r["task"]["id"]), {})[side] = r
    assert all(len(p) == 2 for p in by_pair.values()), cfg.title
    return [p for p in by_pair.values() if p["on"]["valid"] and p["off"]["valid"]]


def strips(cfg: TaskConfig) -> list[tuple[str, np.ndarray, str]]:
    score = lambda r: r["peaks_visible"][cfg.probe]
    negatives = np.array([r["peaks"][cfg.probe] for r in read(cfg.calibration)])
    plain = np.array([score(r) for r in read(cfg.plain) if r["valid"]])
    pairs = valid_pairs(cfg)
    return [("look-alike replies\n(set the line)", negatives, COLORS["neg"]),
            ("plain correct answers", plain, COLORS["plain"]),
            ("OFF replies", np.array([score(p["off"]) for p in pairs]), COLORS["off"]),
            ("ON replies", np.array([score(p["on"]) for p in pairs]), COLORS["on"])]


def plot(name: str, cfg: TaskConfig) -> Path:
    line = flag_line(cfg)
    data = strips(cfg)
    n_pairs = len(data[-1][1])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (label, xs, color) in enumerate(data):
        y = len(data) - 1 - i
        ax.scatter(xs, y + rng.uniform(-0.28, 0.28, len(xs)), s=4, color=color, alpha=0.25, linewidths=0)
        med = float(np.median(xs))
        ax.plot([med, med], [y - 0.38, y + 0.38], color=color, lw=3)
        ax.text(med, y + 0.42, f"{med:+.1f}", ha="center", va="bottom", fontsize=8, color=color)
    ax.axvline(line, color=COLORS["line"], lw=1.6)
    ax.text(line, -0.62, "← missed  |  flagged →", ha="center", va="bottom", fontsize=8, color=COLORS["line"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.set_yticks(range(len(data)), [d[0] for d in reversed(data)], fontsize=9)
    ax.set_ylim(-0.7, len(data) - 0.1)
    ax.set_title(f"{cfg.title}  (1% flag line {line:.2f})", loc="left", fontsize=10, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("probe score (log-odds, read with the prompt in view); higher = 'doing the task'", fontsize=9)
    fig.text(0.01, 0.99, f"Frozen test, {cfg.writer_label} after {B} rounds: {n_pairs} pairs where both answers are correct "
             "(3 chains pooled); ticks = medians", fontsize=8.5, ha="left", va="top", color="#52514e")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / f"dots_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cfg in TASKS.items():
        print(plot(name, cfg))


if __name__ == "__main__":
    main()
