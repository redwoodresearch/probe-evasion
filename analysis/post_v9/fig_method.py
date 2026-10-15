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
POSITIVE_LABEL = "held-out positives"  # panel 3's reference dot: the median of panel 2's held-out positives

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


# the three actors in the search loop, each with a soft fill and a matching edge
ACTOR_COLOURS = {"writer": ("#fbe7da", "#c2541b"), "qwen": ("#e3ebf5", "#3d5a80"), "probe": ("#f6dedc", "#b3261e")}


def box(ax, x: float, y: float, text: str, w: float = 0.34, h: float = 0.2, fontsize: float = 12, actor: str | None = None) -> None:
    fill, edge = ACTOR_COLOURS[actor] if actor else (BOX_FILL, TEXT)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=fill, ec=edge, lw=1.6, clip_on=False))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=TEXT, linespacing=1.35)


def arrow(ax, start: tuple, end: tuple, label: str = "", label_xy: tuple | None = None, fontsize: float = 10.5) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20, color=MUTED, lw=2,
                                 connectionstyle="arc3,rad=-0.18"))
    if label:
        ax.text(*label_xy, label, ha="center", va="center", fontsize=fontsize, color=MUTED, style="italic")


def draw_loop(ax, fontsize: float = 12) -> None:
    """The search loop. Box positions are in axes units (0-1)."""
    writer, qwen, probe = (0.215, 0.5), (0.72, 0.8), (0.72, 0.2)
    box(ax, *writer, "14 writer models\nwrite a prompt", w=0.38, h=0.2, fontsize=fontsize, actor="writer")
    box(ax, *qwen, "Qwen3.5-27B answers\nwith the prompt\n(switch ON / OFF)", w=0.5, h=0.24, fontsize=fontsize, actor="qwen")
    box(ax, *probe, "Probe scores\nQwen's activations", w=0.5, h=0.2, fontsize=fontsize, actor="probe")
    # edge to edge: writer top -> Qwen left; Qwen bottom -> probe top; probe left -> writer bottom
    arrow(ax, (0.24, 0.615), (0.455, 0.80))
    arrow(ax, (0.80, 0.665), (0.80, 0.315))
    arrow(ax, (0.455, 0.20), (0.24, 0.385), "scores fed back,\n16 rounds", (0.2, 0.17), fontsize=fontsize - 1.5)
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
    ax.text(float(np.median(plain)), 1.36, "held-out positives\n(unattacked test answers)", ha="center", va="bottom",
            fontsize=fontsize, color=C["plain"], linespacing=1.2)
    # the look-alike cloud sits just left of the line, so both its labels go to the right of the line
    ax.text(line + 0.4, -0.3, "held-out negatives\n(hard negatives)", va="top", fontsize=fontsize, color=MUTED, linespacing=1.2)
    ax.text(line + 0.4, 0.3, "1% of them are above", va="bottom", fontsize=fontsize - 1.5, color=C["line"])
    ax.plot([line, line], [-0.75, 1.6], color=C["line"], lw=2.5)
    ax.set_ylim(-1.05, 1.95)
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
    FIG.draw_score(axes[2], fontsize=12, dot_size=220, positive_label=POSITIVE_LABEL)
    for ax, title in zip(axes, ("1. The search", "2. Setting the threshold", "3. The score: 9.2 ÷ 20.0 = 0.46")):
        step_title(ax, title)
    save(fig, "method_v1.png")


def version_2() -> None:
    """2 × 2: loop | training examples, flag line | score."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), gridspec_kw={"hspace": 0.32, "wspace": 0.1})
    draw_loop(axes[0, 0])
    draw_snippets(axes[0, 1])
    draw_calibration(axes[1, 0])
    FIG.draw_score(axes[1, 1], fontsize=12, dot_size=220, positive_label=POSITIVE_LABEL)
    titles = ("1. The search", "2. Training the probe (ordering)", "3. Setting the threshold", "4. The score: 9.2 ÷ 20.0 = 0.46")
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
    FIG.draw_score(score, fontsize=12, dot_size=220, line_label=False, positive_label=POSITIVE_LABEL)
    calibration.set_xlim(-13.5, FIG.example_medians()["plain"] + 4.5)
    step_title(loop, "1. The search")
    step_title(calibration, "2. Setting the threshold")
    step_title(score, "3. The score: 9.2 ÷ 20.0 = 0.46")
    save(fig, "method_v3.png")


def draw_score_line(ax, fontsize: float = 16) -> None:
    """The score as a number line: the three answers on a faint axis, the attack's move as an arc above, the distance from
    an unattacked answer to the threshold as a bracket below, and the score as a fraction of the two."""
    m, c = FIG.example_medians(), FIG.EXAMPLE_COLOURS
    plain, off, on, line = m["plain"], m["off"], m["on"], m["line"]
    left, right = line - 1.0, plain + 1.0
    ax.plot([left, right], [0, 0], color="#d5dad8", lw=2, zorder=1)
    ax.plot([line, line], [-1.15, 1.25], color=c["line"], lw=3, zorder=2)
    ax.text(line, 1.38, "threshold", ha="center", va="bottom", fontsize=fontsize, color=c["line"])
    for x, colour, label in ((on, c["on"], "attack ON"), (off, c["off"], "attack OFF"), (plain, c["plain"], "unattacked\nanswer")):
        ax.scatter(x, 0, s=520, color=colour, zorder=3, edgecolors="white", linewidths=2)
        ax.text(x, -0.32, label, ha="center", va="top", fontsize=fontsize - 1, color=colour, linespacing=1.15)
    ax.add_patch(FancyArrowPatch((off, 0.22), (on, 0.22), arrowstyle="-|>", mutation_scale=22, color=c["on"], lw=2.6,
                                 connectionstyle="arc3,rad=0.45", zorder=2))
    ax.text((on + off) / 2, 1.0, f"{off - on:.1f}", ha="center", va="bottom", fontsize=fontsize + 2, color=c["on"], fontweight="bold")
    ax.annotate("", xy=(line, -1.0), xytext=(plain, -1.0), arrowprops=dict(arrowstyle="|-|", color=c["plain"], lw=1.6,
                                                                           shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text((line + plain) / 2, -1.12, f"{plain - line:.1f}", ha="center", va="top", fontsize=fontsize + 2, color=c["plain"], fontweight="bold")
    # the score as a fraction, to the right of the line
    fx = right + 4.6
    ax.text(fx, 0.38, f"{off - on:.1f}", ha="center", va="center", fontsize=fontsize + 6, color=c["on"], fontweight="bold")
    ax.plot([fx - 1.5, fx + 1.5], [0, 0], color=TEXT, lw=2)
    ax.text(fx, -0.38, f"{plain - line:.1f}", ha="center", va="center", fontsize=fontsize + 6, color=c["plain"], fontweight="bold")
    ax.text(fx + 2.0, 0, f"= {(off - on) / (plain - line):.2f}", ha="left", va="center", fontsize=fontsize + 6, color=TEXT, fontweight="bold")
    ax.text(fx + 2.5, -0.95, "probe-evasion\nscore", ha="center", va="top", fontsize=fontsize - 3, color=MUTED, linespacing=1.15)
    ax.set_xlim(left - 0.3, fx + 6.3)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")


# the attack prompt as a schematic: three blocks, top to bottom as Qwen reads it (label, fill, edge, height, filler lines)
PROMPT_BLOCKS = [("Attack instructions", "#fbe7da", "#c2541b", 0.34, 3),
                 ("Switch: ON / OFF", "#fff4c2", "#b38600", 0.13, 0),
                 ("Task", "#e3ebf5", "#3d5a80", 0.27, 2)]


def draw_prompt(ax, fontsize: float = 17) -> None:
    """Schematic only: no quoted text, faint lines stand in for the words."""
    y = 0.93
    for label, fill, edge, h, lines in PROMPT_BLOCKS:
        ax.add_patch(FancyBboxPatch((0.08, y - h), 0.84, h, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=fill, ec=edge, lw=1.8, clip_on=False))
        label_y = y - 0.075 if lines else y - h / 2
        ax.text(0.5, label_y, label, ha="center", va="center", fontsize=fontsize, color=TEXT, fontweight="bold")
        for i in range(lines):  # stand-in lines of text
            ly = label_y - 0.075 - i * 0.06
            ax.plot([0.16, 0.84 - 0.12 * (i == lines - 1)], [ly, ly], color=edge, lw=3, alpha=0.25, solid_capstyle="round")
        y -= h + 0.06
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel_label(ax, text: str) -> None:
    ax.text(0.5, 1.03, text, transform=ax.transAxes, fontsize=21, color=TEXT, fontweight="bold", ha="center", va="bottom")


def search_and_score() -> None:
    """Three panels, large text: the search loop, the attack prompt's structure, and what the score measures (the
    threshold is explained in words)."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.4), gridspec_kw={"width_ratios": [1, 0.7, 1.45], "wspace": 0.1})
    draw_loop(axes[0], fontsize=16)
    draw_prompt(axes[1])
    draw_score_line(axes[2])
    panel_label(axes[0], "Search")
    panel_label(axes[1], "Prompt")
    panel_label(axes[2], "Scoring")
    save(fig, "method_search_score.png")


if __name__ == "__main__":
    version_1()
    version_2()
    version_3()
    search_and_score()
