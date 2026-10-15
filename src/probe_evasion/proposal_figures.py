"""Illustrative diagrams for the established-probe proposal (Neural Chameleons concept probes, Apollo deception probe).
Every number in these figures is made up to show the mechanics; none is data.

    uv run python -m probe_evasion.proposal_figures   # writes share/proposal_*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, GREY, LIGHT = "#1f2328", "#6b6f76", "#d0d3d8"
BLUE, ORANGE, GREEN, RED = "#2a6fdb", "#e07b39", "#1d7f52", "#c2410c"
HEAT = LinearSegmentedColormap.from_list("probe", ["#f4f6f9", "#ffd9b8", "#f08a4b", "#c2410c"])
TAG = "Illustration (made-up numbers), not data"


CHAR_W, TOKEN_PAD = 0.008, 0.003


def token_width(tok: str) -> float:
    return max(0.018, CHAR_W * len(tok) + 0.01)


def row_width(tokens: list[str]) -> float:
    return sum(token_width(t) + TOKEN_PAD for t in tokens)


def token_row(ax, x0, y, tokens, scores, h=0.07, fontsize=9.5, face=None):
    """Draw tokens as boxes coloured by probe score (None = not scored); returns the x after the last token."""
    x = x0
    for tok, s in zip(tokens, scores):
        w = token_width(tok)
        color = face if face else ("#eceef1" if s is None else HEAT(s))
        ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, tok, ha="center", va="center", fontsize=fontsize, color=INK if s is None or s < 0.75 else "white",
                transform=ax.transAxes, family="monospace")
        x += w + TOKEN_PAD
    return x


def gauge(ax, x, y, value, threshold, w=0.09, h=0.05):
    """A score bar with the threshold marked, and the verdict."""
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="#eceef1", edgecolor="none"))
    ax.add_patch(Rectangle((x, y), w * value, h, transform=ax.transAxes, facecolor=HEAT(value), edgecolor="none"))
    ax.plot([x + w * threshold] * 2, [y - 0.015, y + h + 0.015], color=INK, lw=1.6, transform=ax.transAxes)
    caught = value > threshold
    ax.text(x + w + 0.012, y + h / 2, "flagged" if caught else "missed", fontsize=12, fontweight="bold",
            color=RED if caught else GREEN, va="center", transform=ax.transAxes)


def how_a_probe_reads(out: Path) -> None:
    fig = plt.figure(figsize=(14, 4.4), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.015, 0.93, "A concept probe scores every token of the reply, then averages", fontsize=15, fontweight="bold", color=INK,
            transform=ax.transAxes)
    ax.text(0.015, 0.875, f"Maths probe on Gemma-2-9B, layer 12 (Neural Chameleons).  {TAG}.", fontsize=9.5, color=GREY, transform=ax.transAxes)
    prompt = ["Help", "me", "split", "the", "bill"]
    rows = [
        ("Normal reply", ["Total", "is", "$84", ";", "84", "/", "3", "=", "$28", "each", "."],
         [0.2, 0.1, 0.55, 0.1, 0.85, 0.9, 0.85, 0.8, 0.75, 0.3, 0.1], 0.72),
        ("Hoped-for evasion\n(still does the maths)", ["Each", "of", "you", "owes", "$28", "of", "the", "$84", "."],
         [0.1, 0.05, 0.05, 0.15, 0.3, 0.05, 0.05, 0.25, 0.05], 0.18),
    ]
    for i, (label, toks, sc, mean) in enumerate(rows):
        y = 0.6 - i * 0.27
        ax.text(0.015, y + 0.035, label, fontsize=11, fontweight="bold", color=INK, va="center", transform=ax.transAxes, linespacing=1.2)
        x = token_row(ax, 0.16, y, prompt, [None] * len(prompt))
        ax.text(x + 0.004, y + 0.035, "→", fontsize=14, color=GREY, va="center", transform=ax.transAxes)
        x = token_row(ax, x + 0.03, y, toks, sc)
        ax.text(x + 0.012, y + 0.035, "average", fontsize=9.5, color=GREY, va="center", transform=ax.transAxes)
        gauge(ax, x + 0.065, y + 0.01, mean, 0.5)
    ax.text(0.16, 0.12, "grey: prompt (in context, not scored)", fontsize=9.5, color=GREY, transform=ax.transAxes)
    for j, s in enumerate(np.linspace(0, 1, 6)):
        ax.add_patch(Rectangle((0.45 + j * 0.03, 0.115), 0.03, 0.03, transform=ax.transAxes, facecolor=HEAT(s), edgecolor="none"))
    ax.text(0.45, 0.085, "low", fontsize=9, color=GREY, transform=ax.transAxes)
    ax.text(0.615, 0.085, "high probe score", fontsize=9, color=GREY, transform=ax.transAxes)
    ax.text(0.75, 0.12, "| = threshold: flags 1% of replies\n   without the concept", fontsize=9.5, color=GREY, transform=ax.transAxes)
    fig.savefig(out)


def what_counts(out: Path) -> None:
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    thr, cut = 0.5, 3.5
    ax.axhspan(0, thr, xmin=(cut - 0.5) / 5, xmax=1, color="#e6f4ec", zorder=0)
    ax.axhline(thr, color=INK, lw=1.4, ls="--")
    ax.axvline(cut, color=INK, lw=1.4, ls=":")
    groups = [  # (x centre, y centre, n, colour, label)
        (4.5, 0.8, 40, GREY, "no instruction"),
        (4.3, 0.62, 30, BLUE, "evasion prompt: partial shift"),
        (4.2, 0.3, 14, BLUE, None),
        (1.6, 0.25, 18, ORANGE, "dropped the concept"),
    ]
    for cx, cy, n, color, label in groups:
        ax.scatter(np.clip(cx + rng.normal(0, 0.35, n), 0.6, 5.4), np.clip(cy + rng.normal(0, 0.08, n), 0.02, 0.98), s=28,
                   color=color, alpha=0.75, edgecolors="white", linewidths=0.6, label=label, zorder=3)
    ax.annotate("", xy=(4.25, 0.36), xytext=(4.45, 0.76), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2), zorder=4)
    for x, y, t, c in ((4.6, 0.95, "caught", RED), (4.45, 0.03, "EVASION\n(the target)", GREEN), (1.4, 0.95, "flagged anyway", GREY),
                       (1.3, 0.03, "not evasion:\nstopped doing it", ORANGE)):
        ax.text(x, y, t, ha="center", va="center" if "\n" not in t else "bottom", fontsize=12, fontweight="bold", color=c)
    ax.text(5.45, thr + 0.015, "threshold", ha="right", fontsize=9, color=INK)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Claude judge: does the reply still exhibit the concept / deceive?  (1-5)", fontsize=10.5, color=INK)
    ax.set_ylabel("probe score", fontsize=10.5, color=INK)
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False, fontsize=9.5)
    ax.set_title(f"Only the bottom-right counts as evasion\n{TAG}", loc="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out)


def three_readings(out: Path) -> None:
    fig = plt.figure(figsize=(14, 5.6), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.015, 0.93, "The same reply, read three ways", fontsize=15, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.015, 0.875, f"{TAG}.", fontsize=9.5, color=GREY, transform=ax.transAxes)
    instr = ["[evade", "the", "maths", "probe]"]
    req = ["Split", "the", "bill"]
    reply = ["Each", "owes", "$28", "of", "$84", "."]
    rows = [
        ("As generated", "what a deployed\nmonitor sees", instr, [0.55, 0.5, 0.8, 0.35, 0.75, 0.3], 0.62),
        ("Instruction removed", "as if the model\nthought of it itself", None, [0.1, 0.15, 0.4, 0.05, 0.35, 0.05], 0.18),
        ("Rewrite control", "same words, written\nwith no evasion goal", None, [0.12, 0.15, 0.42, 0.05, 0.38, 0.05], 0.2),
    ]
    for i, (name, sub, ins, sc, mean) in enumerate(rows):
        y = 0.66 - i * 0.24
        ax.text(0.015, y + 0.055, name, fontsize=11.5, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.015, y - 0.015, sub, fontsize=9, color=GREY, transform=ax.transAxes, linespacing=1.2)
        x = 0.2
        if ins:
            x = token_row(ax, x, y, ins, [None] * len(ins), face="#e3d7f2")
        else:
            w = row_width(instr) - TOKEN_PAD
            ax.add_patch(Rectangle((x, y), w, 0.07, transform=ax.transAxes, facecolor="white", edgecolor="#c8c8c8", ls=(0, (2, 2))))
            ax.text(x + w / 2, y + 0.035, "no instruction", ha="center", va="center", fontsize=9, color="#a0a0a0", transform=ax.transAxes)
            x += w + TOKEN_PAD
        x = token_row(ax, x, y, req, [None] * len(req))
        x = token_row(ax, x + 0.02, y, reply, sc)
        gauge(ax, x + 0.03, y + 0.01, mean, 0.5)
    ax.text(0.2, 0.06, "Purple: the evasion instruction.  Coloured: the scored reply tokens.  "
            "If 'as generated' still flags but 'instruction removed' doesn't, the instruction gave it away.\n"
            "If evasion replies score no lower than the rewrite control, the drop came from the words, not from the model.",
            fontsize=9.5, color=GREY, transform=ax.transAxes, linespacing=1.4)
    fig.savefig(out)


def two_tracks(out: Path) -> None:
    fig = plt.figure(figsize=(14, 4.4), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.015, 0.9, "Two tracks in parallel", fontsize=15, fontweight="bold", color=INK, transform=ax.transAxes)
    tracks = [
        ("A", "Toy concepts", "Neural Chameleons probes\n(maths, German, HTML, biology, ...)",
         [("Gemma-2-9B", 1, "released probes"), ("Qwen-2.5-7B", 1, "retrain on their data")], BLUE, "~2 hours, ~$20"),
        ("B", "Deception", "Apollo 'instructed pairs' probe\n(roleplaying + our TriviaQA lies)",
         [("Llama-3.3-70B", 4, "released probe")], ORANGE, "~half a day, ~$60-150"),
    ]
    for i, (letter, kind, probe, models, color, cost) in enumerate(tracks):
        x0 = 0.015 + i * 0.5
        ax.add_patch(FancyBboxPatch((x0, 0.08), 0.47, 0.72, boxstyle="round,pad=0.004,rounding_size=0.02", transform=ax.transAxes,
                                    facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x0 + 0.02, 0.72, f"Track {letter}: {kind}", fontsize=13, fontweight="bold", color=color, transform=ax.transAxes, va="top")
        ax.text(x0 + 0.02, 0.6, probe, fontsize=10, color=INK, transform=ax.transAxes, va="top", linespacing=1.3)
        for j, (model, gpus, note) in enumerate(models):
            mx = x0 + 0.02 + j * 0.22
            ax.text(mx, 0.36, model, fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
            for g in range(gpus):
                ax.add_patch(Rectangle((mx + g * 0.03, 0.24), 0.024, 0.07, transform=ax.transAxes, facecolor=color, alpha=0.8))
            ax.text(mx + gpus * 0.03 + 0.006, 0.275, f"{gpus} GPU" + ("s" if gpus > 1 else ""), fontsize=9, color=GREY,
                    va="center", transform=ax.transAxes)
            ax.text(mx, 0.17, note, fontsize=9, color=GREY, transform=ax.transAxes)
        ax.text(x0 + 0.45, 0.11, cost, fontsize=10, color=GREY, ha="right", transform=ax.transAxes)
    ax.text(0.015, 0.015, "Each track starts by reproducing the paper's own numbers with our pipeline; nothing else runs until that matches.",
            fontsize=9.5, color=GREY, transform=ax.transAxes)
    fig.savefig(out)


if __name__ == "__main__":
    Path("experimental/share").mkdir(exist_ok=True)
    two_tracks(Path("experimental/share/proposal_tracks.png"))
    how_a_probe_reads(Path("experimental/share/proposal_probe_reads.png"))
    what_counts(Path("experimental/share/proposal_what_counts.png"))
    three_readings(Path("experimental/share/proposal_three_readings.png"))
    print("wrote share/proposal_tracks.png, proposal_probe_reads.png, proposal_what_counts.png, proposal_three_readings.png")
