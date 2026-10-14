"""Draft method figures for the post: the writer → Qwen → probe loop, how the probe is trained, how the 1% flag line is
set, and what the score measures, combined in a few layouts to compare. Ordering task throughout (probe q27orr@L48@s0;
Fable 5.1's attack after 16 rounds for the score panel).

    .venv/bin/python analysis/post_v9/fig_method.py      # writes figures/method_v1.png, method_v2.png, method_v3.png
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_2x2", HERE / "fig_2x2.py")
FIG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIG)
DOTS, C = FIG.DOTS, FIG.EXAMPLE_COLOURS
TEXT, MUTED = FIG.F.v6.TEXT, FIG.F.v6.MUTED
BOX_FILL = "#f4f5f3"
POSITIVE, NEGATIVE = "#2f6b4f", "#7a7f7c"

# Two training replies from data/concepts/m26_ordering_train.jsonl (as shown on the methodology page): Qwen works out
# the order (positive) vs the order is already given in the question (negative). Questions shortened.
SNIPPETS = [
    ("Qwen works out the order", POSITIVE,
     "Jun is heavier than Otto. Gia is heavier than Jun. …\nList everyone from heaviest to lightest.",
     "Eli is the heaviest, followed by Ines,\nGia, Jun, and Otto."),
    ("The order is already given", NEGATIVE,
     "The weigh-in sheet ranks them: Vic, Nina, Ezra, …\nList everyone from heaviest to lightest.",
     "The order from heaviest to lightest is\nVic, Nina, Ezra, Gus, Tara."),
]


def box(ax, x: float, y: float, text: str, w: float = 0.34, h: float = 0.2, fontsize: float = 12) -> None:
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=BOX_FILL, ec=TEXT, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=TEXT, linespacing=1.35)


def arrow(ax, start: tuple, end: tuple, label: str = "", label_xy: tuple | None = None, fontsize: float = 10.5) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color=TEXT, lw=1.4,
                                 connectionstyle="arc3,rad=-0.18"))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=fontsize, color=MUTED, style="italic")


def draw_loop(ax, fontsize: float = 12) -> None:
    """The search loop. Box positions are in axes units (0-1)."""
    writer, qwen, probe = (0.2, 0.5), (0.74, 0.8), (0.74, 0.2)
    box(ax, *writer, "Writer model\n(14 models)\nwrites a prompt", w=0.3, fontsize=fontsize)
    box(ax, *qwen, "Qwen3.5-27B solves tasks\nwith the prompt,\nswitch ON and OFF", w=0.46, fontsize=fontsize)
    box(ax, *probe, "Probe reads Qwen's\nactivations", w=0.46, h=0.16, fontsize=fontsize)
    arrow(ax, (0.26, 0.62), (0.5, 0.83))
    arrow(ax, (0.84, 0.69), (0.84, 0.29))
    arrow(ax, (0.5, 0.16), (0.24, 0.39), "scores and example\nreplies, 16 rounds", (0.29, 0.13), fontsize=fontsize - 1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_snippets(ax, fontsize: float = 10.5) -> None:
    """Two training examples, one positive and one negative, each as question → Qwen's reply."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (title, colour, question, reply) in enumerate(SNIPPETS):
        top = 0.97 - i * 0.5
        ax.text(0.02, top, title, fontsize=fontsize + 1.5, color=colour, fontweight="bold", va="top")
        ax.text(0.04, top - 0.1, question, fontsize=fontsize, color=MUTED, va="top", linespacing=1.3)
        ax.text(0.04, top - 0.28, reply, fontsize=fontsize, color=TEXT, va="top", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.4", fc=BOX_FILL, ec=colour, lw=1.2))


def draw_calibration(ax, fontsize: float = 12, show_axis: bool = True) -> None:
    """Look-alike replies (which set the line) and plain correct answers, with the 1% flag line."""
    cfg = DOTS.TASKS[FIG.EXAMPLE_TASK]
    line = DOTS.flag_line(cfg)
    negatives, plain = DOTS.strips(cfg)[0][1], DOTS.strips(cfg)[1][1]
    rng = np.random.default_rng(0)
    for xs, y, colour in ((negatives, 0.0, DOTS.COLORS["neg"]), (plain, 1.0, C["plain"])):
        ax.scatter(xs, y + rng.uniform(-0.25, 0.25, len(xs)), s=5, color=colour, alpha=0.35, linewidths=0)
    ax.text(float(np.median(plain)), 1.36, "normal answers", ha="center", va="bottom", fontsize=fontsize, color=C["plain"])
    # the look-alike cloud sits just left of the line, so both its labels go to the right of the line
    ax.text(line + 0.4, -0.3, "look-alike replies", va="top", fontsize=fontsize, color=MUTED)
    ax.text(line + 0.4, 0.3, "1% of them are above", va="bottom", fontsize=fontsize - 1.5, color=C["line"])
    ax.plot([line, line], [-0.75, 1.6], color=C["line"], lw=2.5)
    ax.set_ylim(-0.8, 1.7)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if show_axis:
        ax.set_xlabel("probe score", fontsize=fontsize - 1, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=fontsize - 2)
    else:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])


def step_title(ax, text: str, fontsize: float = 13) -> None:
    ax.set_title(text, loc="left", fontsize=fontsize, color=TEXT, fontweight="bold")


def save(fig, name: str) -> None:
    fig.savefig(HERE / "figures" / name, bbox_inches="tight", dpi=150)
    plt.close(fig)


def version_1() -> None:
    """One row of three: loop | flag line | score."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1.15], "wspace": 0.12})
    draw_loop(axes[0], fontsize=11.5)
    draw_calibration(axes[1])
    FIG.draw_score(axes[2], fontsize=12, dot_size=220)
    for ax, title in zip(axes, ("1. The search", "2. Setting the flag line", "3. The score: 9.2 ÷ 20.0 = 0.46")):
        step_title(ax, title)
    save(fig, "method_v1.png")


def version_2() -> None:
    """2 × 2: loop | training examples, flag line | score."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), gridspec_kw={"hspace": 0.32, "wspace": 0.1})
    draw_loop(axes[0, 0])
    draw_snippets(axes[0, 1])
    draw_calibration(axes[1, 0])
    FIG.draw_score(axes[1, 1], fontsize=12, dot_size=220)
    titles = ("1. The search", "2. Training the probe (ordering)", "3. Setting the flag line", "4. The score: 9.2 ÷ 20.0 = 0.46")
    for ax, title in zip(axes.flat, titles):
        step_title(ax, title)
    save(fig, "method_v2.png")


def version_3() -> None:
    """Loop on the left; on the right, the flag line and the score share one probe-score axis, so the line runs through both."""
    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1.15], wspace=0.08, hspace=0.25)
    loop = fig.add_subplot(gs[:, 0])
    calibration = fig.add_subplot(gs[0, 1])
    score = fig.add_subplot(gs[1, 1], sharex=calibration)
    draw_loop(loop)
    draw_calibration(calibration, show_axis=False)
    FIG.draw_score(score, fontsize=12, dot_size=220, line_label=False)
    calibration.set_xlim(-13.5, FIG.example_medians()["plain"] + 4.5)
    step_title(loop, "1. The search")
    step_title(calibration, "2. Setting the flag line")
    step_title(score, "3. The score: 9.2 ÷ 20.0 = 0.46")
    save(fig, "method_v3.png")


if __name__ == "__main__":
    version_1()
    version_2()
    version_3()
