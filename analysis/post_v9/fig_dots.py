"""Moves vs gets past: where Fable 5.1's test answers sit against the threshold after 16 rounds, attack OFF vs ON
(pairs where both answers are valid, 3 searches pooled), one panel per task on one shared x scale. The region under the
threshold (where the probe is passed) is shaded, and each panel states the share of ON answers that land there.

    .venv/bin/python analysis/post_v9/fig_dots.py   # writes figures/dots_2x2.png (dots) and figures/dots_2x2_density.png
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_2x2", HERE / "fig_2x2.py")
FIG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FIG)
DOTS = FIG.DOTS
M45, M49 = DOTS.M45, Path("missions/m49_new_task_toggle/runs")
WRITER, WRITER_LABEL = "fable51", "Claude Fable 5.1"
COLOURS = FIG.EXAMPLE_COLOURS
PASS_SHADE, PASS_TEXT = "#e9f2ec", "#2f6b45"
ROWS = (("OFF", "off"), ("ON", "on"))  # top to bottom
X_LIMITS = (-10, 17)  # wide enough that ordering's OFF tail is not cut


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
    _, _, off, on = (xs for _, xs, _ in DOTS.strips(cfg))
    return {"off": off, "on": on, "line": DOTS.flag_line(cfg)}


def draw_dots(ax, xs: np.ndarray, y: float, colour: str, rng) -> None:
    ax.scatter(xs, y + rng.uniform(-0.3, 0.3, len(xs)), s=7, color=colour, alpha=0.28, linewidths=0, zorder=2)


def draw_density(ax, xs: np.ndarray, y: float, colour: str, rng) -> None:
    grid = np.linspace(*X_LIMITS, 400)
    dens = gaussian_kde(xs)(grid)
    ax.fill_between(grid, y - 0.35, y - 0.35 + 0.75 * dens / dens.max(), color=colour, alpha=0.55, lw=0, zorder=2)


def panel(ax, title: str, d: dict, draw, rng) -> None:
    line = d["line"]
    ax.axvspan(X_LIMITS[0], line, color=PASS_SHADE, zorder=0)
    ax.axvline(line, color=COLOURS["line"], lw=2.5, zorder=3)
    for i, (label, key) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        draw(ax, d[key], y, COLOURS[key], rng)
        med = float(np.median(d[key]))
        ax.plot([med, med], [y - 0.38, y + 0.38], color=COLOURS[key], lw=4, zorder=4, solid_capstyle="butt")
        ax.text(X_LIMITS[0] - 0.4, y, label, ha="right", va="center", fontsize=17, color=COLOURS[key], fontweight="bold")
    passed = float(np.mean(d["on"] < line))
    ax.text(X_LIMITS[1], 0.05, f"{passed:.0%}", ha="right", va="center", fontsize=26, color=COLOURS["on"], fontweight="bold")
    ax.text(X_LIMITS[1], -0.5, "of ON answers pass", ha="right", va="center", fontsize=12.5, color=COLOURS["on"])
    ax.set_title(title, fontsize=19, pad=10)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(-0.75, len(ROWS) - 0.45)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=13)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def figure(draw, name: str) -> None:
    data = {k: task_data(cfg) for k, cfg in TASKS.items()}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.6), sharex=True, gridspec_kw={"hspace": 0.38, "wspace": 0.16})
    for ax, (k, cfg) in zip(axes.flat, TASKS.items()):
        panel(ax, cfg.title, data[k], draw, rng)
    # name the shaded region and the threshold once, in contradiction, whose shaded region is widest
    ax, line = axes[1, 1], data["contradiction"]["line"]
    ax.text(X_LIMITS[0] + 0.3, 1.42, "passes the probe", ha="left", va="bottom", fontsize=13, color=PASS_TEXT)
    ax.text(line + 0.3, 1.42, "threshold", ha="left", va="bottom", fontsize=13, color=COLOURS["line"])
    for ax in axes[1]:
        ax.set_xlabel("probe score (log-odds)", fontsize=15)
    fig.savefig(HERE / "figures" / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    figure(draw_dots, "dots_2x2.png")
    figure(draw_density, "dots_2x2_density.png")


if __name__ == "__main__":
    main()
