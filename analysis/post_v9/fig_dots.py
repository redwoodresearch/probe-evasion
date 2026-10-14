"""Moves vs gets past: where Fable 5.1's test answers sit against the threshold, after 16 rounds. One row each for
unattacked answers, attack OFF and attack ON (pairs where both answers are valid, 3 searches pooled); one shared x scale.

    .venv/bin/python analysis/post_v9/fig_dots.py   # writes figures/dots_two_tasks.png and figures/dots_four_tasks.png
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_2x2", HERE / "fig_2x2.py")
FIG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIG)
DOTS = FIG.DOTS
M45, M49 = DOTS.M45, Path("missions/m49_new_task_toggle/runs")
WRITER, WRITER_LABEL = "fable51", "Claude Fable 5.1"
COLOURS = FIG.EXAMPLE_COLOURS
ROWS = (("unattacked answers", "plain"), ("attack OFF", "off"), ("attack ON", "on"))


def m45_config(title: str, probe: str, root: str, line: float):
    return DOTS.TaskConfig(title, probe, M45 / "c" / root / "calibration_scores.jsonl", M45 / "c" / root / "test_checked" / "plain.jsonl",
                           M45 / "d" / root / "test_checked" / f"{WRITER}.jsonl", WRITER_LABEL, line)


def m49_config(title: str, probe: str, task: str, line: float):
    main, d = M49 / "main" / f"m49_{task}", M49 / "d" / f"m49_{task}"
    return DOTS.TaskConfig(title, probe, main / "calibration_scores.jsonl", main / "test" / "plain.jsonl",
                           d / "test_checked" / f"{WRITER}.jsonl", WRITER_LABEL, line)


TASKS = {
    "maths": m45_config("Arithmetic", "q27mrr@L48@s0", "qwen35_27b", -6.53),
    "ordering": m45_config("Ordering", "q27orr@L48@s0", "qwen35_27b_ordering", -8.06),
    "false_belief": m49_config("Belief tracking", "m48fa_v2@L48@s0", "false_belief", -2.39),
    "contradiction": m49_config("Spotting a contradiction", "m48co_v2@L48@s0", "contradiction", -0.58),
}


def task_data(cfg) -> dict:
    _, plain, off, on = (xs for _, xs, _ in DOTS.strips(cfg))
    return {"plain": plain, "off": off, "on": on, "line": DOTS.flag_line(cfg)}


def panel(ax, title: str, d: dict, rng) -> None:
    line = d["line"]
    for i, (label, key) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        xs = d[key]
        ax.scatter(xs, y + rng.uniform(-0.27, 0.27, len(xs)), s=5, color=COLOURS[key], alpha=0.3, linewidths=0)
        med = float(np.median(xs))
        ax.plot([med, med], [y - 0.36, y + 0.36], color=COLOURS[key], lw=3.5)
        if key == "on":
            ax.text(1.01, y, f"{np.mean(xs < line):.0%} under\nthe threshold", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=12, color=COLOURS["on"])
    ax.axvline(line, color=COLOURS["line"], lw=2)
    # named in the top row, right of the line: unattacked answers always sit far to the right, so that spot is empty
    ax.text(line + 0.25, len(ROWS) - 1, "threshold", ha="left", va="center", fontsize=12, color=COLOURS["line"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=4)
    ax.set_yticks(range(len(ROWS)), [r[0] for r in reversed(ROWS)], fontsize=13)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_title(title, loc="left", fontsize=15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(keys: tuple[str, ...], name: str) -> None:
    data = {k: task_data(TASKS[k]) for k in keys}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.6 * len(keys) + 0.8), sharex=True)
    for ax, k in zip(axes, keys):
        panel(ax, TASKS[k].title, data[k], rng)
    axes[-1].set_xlabel("probe score (log-odds)    ← passes  |  flagged →", fontsize=13)
    axes[-1].set_xlim(-10, 17)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84, hspace=0.45)
    fig.savefig(HERE / "figures" / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    figure(("maths", "contradiction"), "dots_two_tasks.png")
    figure(tuple(TASKS), "dots_four_tasks.png")


if __name__ == "__main__":
    main()
