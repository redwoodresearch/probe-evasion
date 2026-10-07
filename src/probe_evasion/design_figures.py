"""Design diagrams for the optimized-prompt evasion experiments (illustrations, not data).

    uv run python -m probe_evasion.design_figures   # writes share/design_*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from probe_evasion.proposal_figures import BLUE, GREEN, GREY, INK, ORANGE, RED, TAG, gauge, token_row

PURPLE = "#7c3aed"


def bubble(ax, x, y, w, h, text, face, fontsize=10.5, color=INK, mono=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018", transform=ax.transAxes,
                                facecolor=face, edgecolor="none"))
    ax.text(x + 0.012, y + h / 2, text, fontsize=fontsize, color=color, va="center", transform=ax.transAxes,
            family="monospace" if mono else None, linespacing=1.35)


def concepts(out: Path) -> None:
    """Three concepts: what the model is asked, what it replies, how we know the concept is really there."""
    fig = plt.figure(figsize=(15, 6.2), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.015, 0.94, "Three concepts, from easy-to-see in the text to almost invisible in it", fontsize=15,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.015, 0.895, f"{TAG}.", fontsize=9.5, color=GREY, transform=ax.transAxes)
    cols = [
        ("Chemistry", "text shows it plainly", "Why does salt melt ice?",
         ["Salt", "lowers", "water's", "freezing", "point"], [0.4, 0.75, 0.6, 0.85, 0.8],
         "Claude: is this chemistry?", ORANGE, "comparison"),
        ("Sarcasm", "meaning is the opposite of the words", "My flight got cancelled again.",
         ["Wow,", "what", "a", "great", "airline"], [0.5, 0.4, 0.3, 0.85, 0.6],
         "Claude: is it sarcastic?", PURPLE, "like deception"),
        ("Maths in the head", "the answer is just a number", "289 pens at $17 each: total?",
         ["That's", "$4,913", "in", "total"], [0.3, 0.9, 0.2, 0.35],
         "Answer is correct  ✓", BLUE, "like deception"),
    ]
    for i, (name, sub, ask, toks, sc, check, color, tag) in enumerate(cols):
        x0 = 0.015 + i * 0.33
        ax.text(x0, 0.8, name, fontsize=14, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x0, 0.755, sub, fontsize=10, color=GREY, transform=ax.transAxes)
        bubble(ax, x0, 0.6, 0.3, 0.09, ask, "#eef0f3")
        ax.text(x0 + 0.3, 0.705, "user", fontsize=8.5, color=GREY, ha="right", transform=ax.transAxes)
        ax.text(x0, 0.535, "model's reply, coloured by probe score", fontsize=8.5, color=GREY, transform=ax.transAxes)
        token_row(ax, x0, 0.45, toks, sc, h=0.065, fontsize=9.5)
        ax.add_patch(FancyBboxPatch((x0, 0.27), 0.3, 0.1, boxstyle="round,pad=0.006,rounding_size=0.018",
                                    transform=ax.transAxes, facecolor="white", edgecolor=color, linewidth=1.6))
        ax.text(x0 + 0.015, 0.32, "Concept really there?", fontsize=9, color=GREY, va="center", transform=ax.transAxes)
        ax.text(x0 + 0.285, 0.32, check, fontsize=10.5, color=INK, ha="right", va="center", transform=ax.transAxes,
                fontweight="bold")
        ax.text(x0, 0.2, ("Probe: the paper's released one (Gemma); retrained the same way (Qwen)" if i == 0 else
                          "Probe: trained the paper's way on text Claude / we wrote"), fontsize=9, color=GREY,
                transform=ax.transAxes)
        ax.add_patch(FancyBboxPatch((x0, 0.09), 0.12, 0.05, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    transform=ax.transAxes, facecolor=color, alpha=0.15, edgecolor="none"))
        ax.text(x0 + 0.06, 0.115, tag, fontsize=9.5, color=color, ha="center", va="center", transform=ax.transAxes,
                fontweight="bold")
    # arrow showing the "how visible in text" axis
    ax.add_patch(FancyArrowPatch((0.05, 0.035), (0.95, 0.035), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=14, color=GREY, lw=1.2))
    ax.text(0.05, 0.05, "concept obvious from the words", fontsize=9, color=GREY, transform=ax.transAxes)
    ax.text(0.95, 0.05, "concept hidden from the words", fontsize=9, color=GREY, ha="right", transform=ax.transAxes)
    fig.savefig(out)


def loop(out: Path) -> None:
    """The optimization loop and the frozen held-out test."""
    fig = plt.figure(figsize=(15, 5.6), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.015, 0.93, "Search for the best evasion prompt, then freeze it and test on new tasks", fontsize=15,
            fontweight="bold", color=INK, transform=ax.transAxes)
    nodes = {  # name: (x, y, w, h, title, body, colour)
        "prompt": (0.04, 0.55, 0.15, 0.2, "Evasion prompt", "starts from our\nstrategy hint", PURPLE),
        "model": (0.25, 0.55, 0.15, 0.2, "Model replies", "Gemma-2-9B\nQwen3.5-27B", INK),
        "score": (0.46, 0.55, 0.17, 0.2, "Probe + check", "probe score\nconcept still there?", ORANGE),
        "reflect": (0.25, 0.14, 0.38, 0.2, "Claude reads the results", "\"these 3 were caught, this one dropped the\nconcept…\" → writes a better prompt", BLUE),
        "test": (0.75, 0.55, 0.21, 0.2, "Held-out test", "best prompt, frozen,\n100 new tasks", GREEN),
    }
    for x, y, w, h, title, body, color in nodes.values():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02", transform=ax.transAxes,
                                    facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x + w / 2, y + h - 0.05, title, ha="center", fontsize=11.5, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x + w / 2, y + h / 2 - 0.035, body, ha="center", va="center", fontsize=9.5, color=GREY,
                transform=ax.transAxes, linespacing=1.35)

    def arrow(a, b, rad=0.0, color=GREY):
        ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=16, color=color,
                                     lw=1.6, connectionstyle=f"arc3,rad={rad}"))

    arrow((0.19, 0.65), (0.25, 0.65))
    arrow((0.40, 0.65), (0.46, 0.65))
    arrow((0.545, 0.55), (0.545, 0.34))
    arrow((0.25, 0.24), (0.115, 0.55), rad=-0.3)
    arrow((0.63, 0.65), (0.75, 0.65), color=GREEN)
    ax.text(0.69, 0.68, "after ~25\nrounds", ha="center", fontsize=9, color=GREEN, transform=ax.transAxes)
    ax.text(0.05, 0.45, "40 training tasks", fontsize=9, color=GREY, transform=ax.transAxes)
    ax.text(0.75, 0.47, "Success = concept still there AND\nprobe below the 1%-false-alarm line",
            fontsize=10, color=INK, transform=ax.transAxes, linespacing=1.35)
    ax.text(0.75, 0.33, "Also reported: AUROC (no threshold),\ninstruction-removed reading,\nrewrite control",
            fontsize=9.5, color=GREY, transform=ax.transAxes, linespacing=1.35)
    fig.savefig(out)


def readings(out: Path) -> None:
    """What a success looks like on the gauge, for each probe pooling."""
    fig = plt.figure(figsize=(13, 3.6), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.015, 0.88, "Two versions of each probe: the paper's average, and a stricter maximum", fontsize=14,
            fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.015, 0.8, f"{TAG}.", fontsize=9.5, color=GREY, transform=ax.transAxes)
    toks = ["Each", "of", "you", "owes", "$28", "of", "the", "$84", "."]
    sc = [0.1, 0.05, 0.05, 0.15, 0.8, 0.05, 0.05, 0.3, 0.05]
    x = token_row(ax, 0.2, 0.42, toks, sc)
    ax.text(0.015, 0.455, "Same reply", fontsize=11, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(x + 0.03, 0.6, "average over tokens", fontsize=9.5, color=GREY, transform=ax.transAxes)
    gauge(ax, x + 0.03, 0.5, 0.18, 0.5)
    ax.text(x + 0.03, 0.3, "maximum over tokens", fontsize=9.5, color=GREY, transform=ax.transAxes)
    gauge(ax, x + 0.03, 0.2, 0.8, 0.5)
    ax.text(0.2, 0.12, "Padding a reply with unrelated words dilutes the average but not the maximum, so we report both.",
            fontsize=9.5, color=GREY, transform=ax.transAxes)
    fig.savefig(out)


if __name__ == "__main__":
    concepts(Path("share/design_concepts.png"))
    loop(Path("share/design_loop.png"))
    readings(Path("share/design_pooling.png"))
    print("wrote share/design_concepts.png, design_loop.png, design_pooling.png")
