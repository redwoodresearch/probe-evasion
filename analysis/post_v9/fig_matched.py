"""The matched (wrong-answer) test for the post, without acrostic: Qwen's reply is word-for-word identical after a
request with a wrong answer inserted and one with the right answer inserted, so after the wrong one it had to work
the answer out. The probe scores those replies higher; a word classifier (TF-IDF on request + reply) can't separate
them. Numbers from analysis/tfidf_baseline/for_james/numbers.json (written by its make_figure.py, which checks them
against m51's analysis); the example pair is a real arithmetic pair.

    .venv/bin/python analysis/post_v9/fig_matched.py   # writes figures/matched_test.png
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("matched_source", REPO / "analysis/tfidf_baseline/for_james/make_figure.py")
SOURCE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SOURCE)

NUMBERS = REPO / "analysis/tfidf_baseline/for_james/numbers.json"
TASKS = {"maths": "Arithmetic", "ordering": "Ordering", "false_belief": "Belief tracking", "contradiction": "Spotting a\ncontradiction"}
WORDS = "tfidf_B"  # the word classifier that reads request + reply, the one that matters here
INK = "#1f2a2e"
PROBE_C, WORDS_C = "#c2541b", "#8a9396"
WRONG_FILL, RIGHT_FILL, REPLY_FILL = "#fbe7da", "#e3f0e6", "#f4f5f3"


def card(ax, x, y, w, h, text, fill, size=13, edge=INK, weight="normal", colour=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                                fc=fill, ec=edge, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=colour, weight=weight, linespacing=1.35)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.5))


def schematic(ax, pair: dict) -> None:
    probe_c, words_c, wrong_fill, right_fill, reply_fill = PROBE_C, WORDS_C, WRONG_FILL, RIGHT_FILL, REPLY_FILL
    wrong, right = pair["wrong"], pair["given"]
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    card(ax, 0.25, 0.86, 0.46, 0.2, "… $2540 in total.\nWhat do they cost?", wrong_fill)
    card(ax, 0.75, 0.86, 0.46, 0.2, "… $2891 in total.\nWhat do they cost?", right_fill)
    ax.text(0.25, 1.0, "wrong answer given", ha="center", fontsize=12, color=INK, style="italic")
    ax.text(0.75, 1.0, "right answer given", ha="center", fontsize=12, color=INK, style="italic")
    card(ax, 0.5, 0.52, 0.96, 0.13, f"Same reply: “{wrong['reply']}”", reply_fill, size=12.5)
    arrow(ax, (0.25, 0.75), (0.4, 0.59)), arrow(ax, (0.75, 0.75), (0.6, 0.59))
    card(ax, 0.25, 0.17, 0.46, 0.2, f"Probe\n{wrong['probe']:+.1f}  vs  {right['probe']:+.1f}", "white", size=15,
         edge=probe_c, weight="bold", colour=probe_c)
    card(ax, 0.75, 0.17, 0.46, 0.2, f"Word classifier\n{wrong[WORDS]:+.1f}  vs  {right[WORDS]:+.1f}", "white", size=15,
         edge=words_c, weight="bold", colour=words_c)
    arrow(ax, (0.4, 0.455), (0.25, 0.28)), arrow(ax, (0.6, 0.455), (0.75, 0.28))


def results(ax, numbers: dict) -> None:
    probe_c, words_c = PROBE_C, WORDS_C
    y = np.arange(len(TASKS))[::-1].astype(float)
    h = 0.32
    for yi, task in zip(y, TASKS):
        r = numbers[task]["wrong_vs_given"]
        p, w = r["probe"], r[WORDS]
        ax.barh(yi + h / 2, p["median_gap"], height=h, color=probe_c)
        ax.barh(yi - h / 2, w["median_gap"], height=h, color=words_c)
        ax.text(p["median_gap"] + 0.25, yi + h / 2, f"+{p['median_gap']:.1f}   ({p['share_higher']:.0%} of pairs)",
                va="center", fontsize=12.5, color=probe_c)
        gap = w["median_gap"] if abs(w["median_gap"]) >= 0.05 else 0.0  # no "-0.0"
        ax.text(0.25, yi - h / 2, f"{gap:+.1f}" if gap else "0.0", va="center", fontsize=12.5, color=words_c)
    ax.set_yticks(y, list(TASKS.values()), fontsize=14)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("score gap on identical replies (wrong answer − right answer)", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.98, 0.09, "probe", transform=ax.transAxes, ha="right", fontsize=14, color=probe_c, weight="bold")
    ax.text(0.98, 0.02, "word classifier", transform=ax.transAxes, ha="right", fontsize=14, color=words_c, weight="bold")


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    rows = SOURCE.load_rows()
    pair = {k: rows[v] for k, v in SOURCE.EXAMPLE_PAIR.items()}
    assert pair["wrong"]["reply"] == pair["given"]["reply"] and pair["wrong"]["correct"] and pair["given"]["correct"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.3})
    schematic(axes[0], pair)
    results(axes[1], numbers)
    fig.savefig(HERE / "figures" / "matched_test.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
