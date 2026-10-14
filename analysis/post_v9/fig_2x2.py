"""Main figures for the post draft: one panel per probe (maths, ordering, belief tracking, contradiction), writer
capability on x, gap closed (plain-referenced) on y, 14 writers. Two figures: after 16 rounds of feedback, and at
round 0 (before any feedback). Values and slopes from the independent recomputation (numbers/numbers.json).

    .venv/bin/python analysis/post_v9/fig_2x2.py      # writes figures/trend_2x2_b16.png and figures/trend_2x2_b0.png

Also writes figures/how_to_read.png: one worked example (Fable 5.1 on ordering) showing what the score measures.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("post_v8_figures", HERE.parent / "post_v8" / "make_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
_spec_dots = importlib.util.spec_from_file_location("make_dots", F.REPO / "analysis/for_james/methodology/make_dots.py")
DOTS = importlib.util.module_from_spec(_spec_dots)
_spec_dots.loader.exec_module(DOTS)

NUMBERS = json.loads((HERE / "numbers" / "numbers.json").read_text())["tasks"]
PANELS = {"maths": "Maths", "ordering": "Ordering", "false_belief": "Belief tracking", "contradiction": "Spotting a contradiction"}
ROUNDS = {"B16": "after 16 rounds of feedback", "B0": "round 0, before any feedback"}
FIT_XS = np.linspace(131, 168, 50)
KEY_WRITERS = {"fable51": "Fable 5.1", "gpt6astra": "GPT-6 Astra", "gpt61sol": "GPT-6.1 Sol", "opus55": "Opus 5.5"}
LABEL_X = 170.5        # labels sit in a column right of the data, with leader lines to their dots
LABEL_GAP = 0.06       # minimum vertical gap between labels (score units)
Y_LIMITS = (-0.15, 0.82)
Y_LABEL = "probe-dodging score\n(0 = no move; 1 or more = past the flag line)"
EXAMPLE_TASK = "ordering"  # the worked example: Fable 5.1 (make_dots' writer for ordering), medians over correct pairs
EXAMPLE_COLOURS = {"plain": "#3b4440", "off": "#3d5a80", "on": "#c2541b", "line": "#b3261e"}


def fmt(fit: dict) -> str:
    lo, hi = fit["ci"]
    return f"{fit['slope']:+.2f} [{lo:+.2f}, {hi:+.2f}] per 10 ECI"


def spread(ys: list[float]) -> list[float]:
    """Push label positions apart (keeping their order) so no two are closer than LABEL_GAP."""
    order = np.argsort(ys)
    placed = list(np.array(ys, dtype=float)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + LABEL_GAP)
    out = np.empty(len(ys))
    out[order] = placed
    return list(out)


def panel(ax, task: str, title: str, b: str, meta: dict) -> None:
    t = NUMBERS[task][b]
    means, chains = t["writer_means"], t["writer_chains"]
    for w, m in means.items():
        x, colour = meta[w]["eci"] + F.DISPLAY_SHIFT.get(w, 0.0), F.LAB_COLOUR[meta[w]["lab"]]
        ax.vlines(x, min(chains[w]), max(chains[w]), color=colour, lw=1.8, alpha=0.5)
        ax.plot(x, m, "o", ms=10, color=colour, zorder=3)
    k, c0 = np.polyfit([meta[w]["eci"] for w in means], list(means.values()), 1)
    ax.plot(FIT_XS, k * FIT_XS + c0, "-", color=F.v6.TEXT, lw=1.8)
    keys = list(KEY_WRITERS)
    for w, y in zip(keys, spread([means[w] for w in keys])):
        x = meta[w]["eci"] + F.DISPLAY_SHIFT.get(w, 0.0)
        ax.plot([x + 0.3, LABEL_X - 0.2], [means[w], y], color=F.v6.MUTED, lw=0.7, zorder=2)
        ax.text(LABEL_X, y, KEY_WRITERS[w], va="center", fontsize=11, color=F.v6.TEXT)
    ax.axhline(0, color=F.v6.MUTED, lw=0.8)
    ax.set_title(f"{title}: {fmt(t['all14'])}", fontsize=13)
    ax.grid(axis="y", color=F.v6.GRID)
    ax.set_xlim(130, 179)
    ax.set_xticks(range(130, 171, 10))
    ax.set_ylim(*Y_LIMITS)


def how_to_read() -> None:
    """A simple standalone picture of what the score measures, from one real example (Fable 5.1 on ordering)."""
    cfg = DOTS.TASKS[EXAMPLE_TASK]
    line = DOTS.flag_line(cfg)
    med = {label.split("\n")[0]: float(np.median(xs)) for label, xs, _ in DOTS.strips(cfg)}
    plain, off, on = med["plain correct answers"], med["OFF replies"], med["ON replies"]
    moved, room = off - on, plain - line
    c = EXAMPLE_COLOURS
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.fill_between([-13, line], -0.5, 1.2, color="#eef3ee", zorder=0)
    ax.text(line - 0.4, 1.05, "not flagged", ha="right", fontsize=13, color="#5c6762")
    ax.text(line + 0.4, 1.05, "flagged", ha="left", fontsize=13, color=c["line"])
    ax.plot([line, line], [-0.5, 1.2], color=c["line"], lw=3)
    for x, colour, label in ((plain, c["plain"], "normal answer"), (off, c["off"], "attack OFF"), (on, c["on"], "attack ON")):
        ax.scatter(x, 0.45, s=420, color=colour, zorder=3)
        ax.text(x, 0.72, label, ha="center", fontsize=14, color=colour, fontweight="bold")
    ax.annotate("", xy=(on + 0.35, 0.45), xytext=(off - 0.35, 0.45), arrowprops=dict(arrowstyle="->", color=c["on"], lw=3))
    ax.text((on + off) / 2, 0.22, f"the attack moves it {moved:.1f}", ha="center", fontsize=13, color=c["on"])
    ax.annotate("", xy=(line, -0.1), xytext=(plain, -0.1), arrowprops=dict(arrowstyle="<->", color=c["plain"], lw=1.5))
    ax.text((line + plain) / 2, -0.34, f"a normal answer is {room:.1f} from the flag line", ha="center", fontsize=13,
            color=c["plain"])
    ax.text(-12.6, -0.82, f"Probe-dodging score = {moved:.1f} ÷ {room:.1f} ≈ {moved / room:.2f}", fontsize=15, color=F.v6.TEXT)
    ax.text(-12.6, -1.12, "0 = the attack doesn't move the answer. 1 or more = the typical attacked answer is past the flag line, so the probe is fooled.", fontsize=13,
            color=F.v6.MUTED)
    ax.set_xlim(-13, 13.5)
    ax.set_ylim(-1.3, 1.25)
    ax.axis("off")
    ax.set_title("What the score measures (Fable 5.1's attack on the ordering task; typical correct answer)", loc="left",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "how_to_read.png", bbox_inches="tight")
    plt.close(fig)


def figure(b: str, meta: dict) -> None:
    with_example = False  # the worked example is its own figure (how_to_read.png)
    fig = plt.figure(figsize=(15, 15.5 if with_example else 11))
    gs = fig.add_gridspec(3 if with_example else 2, 2, height_ratios=[1.25, 4, 4] if with_example else [1, 1], hspace=0.38)
    rows = 1 if with_example else 0
    axes = np.array([[fig.add_subplot(gs[rows + i, j]) for j in range(2)] for i in range(2)])
    for ax in axes.flat[1:]:
        ax.sharex(axes[0, 0]); ax.sharey(axes[0, 0])
    for ax, (task, title) in zip(axes.flat, PANELS.items()):
        panel(ax, task, title, b, meta)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    for ax in axes[-1]:
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL, fontsize=12)
    fig.suptitle(f"Qwen3.5-27B, 14 writers, {ROUNDS[b]}", fontsize=15, x=0.02, ha="left")
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    handles.append(Line2D([], [], color=F.v6.TEXT, ls="-", label="least-squares fit"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.045 if with_example else 0.06, "Slopes [95% interval, resampling writers then searches]; bars = range over 3 searches; strict 1% flag line",
             ha="center", fontsize=11, color=F.v6.MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94 if with_example else 0.9, bottom=0.1 if with_example else 0.13)
    fig.savefig(HERE / "figures" / f"trend_2x2_{b.lower()}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = {r["writer"]: r for r in F.writer_rows(F.load_m45(), "maths")}  # ECI and lab per writer
    for b in ROUNDS:
        figure(b, meta)
    how_to_read()


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    main()
