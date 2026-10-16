"""The post's two concept diagrams, redrawn from Nabil's Excalidraw sketches in the style of the method figure
(fig_method.py): soft fills with matching darker edges, large text, muted arrows, labels next to what they name.
Colours carry meaning across figures: the probe's gold, the attack's orange (the "attack ON" colour of the scoring
panel), the writer's purple.

  transcript_space.png  all transcripts > transcripts that use the concept, which the probe mostly catches > the
                        pockets it misses, which an attack aims for
  ladder.png            from a writer with many rounds of feedback to a model evading on its own; this post measures
                        the first and third steps

    .venv/bin/python analysis/post_v9/fig_concepts.py
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from scipy.interpolate import splev, splprep

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_method", HERE / "fig_method.py")
METHOD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METHOD)
TEXT, MUTED = METHOD.TEXT, METHOD.MUTED
WRITER_FILL, WRITER_EDGE = METHOD.ACTOR_COLOURS["writer"]
PROBE_FILL, PROBE_EDGE = METHOD.ACTOR_COLOURS["probe"]
ATTACK_FILL, ATTACK_EDGE = METHOD.SCORE_FILLS["on"], METHOD.C["on"]
OUTER_FILL, OUTER_EDGE = "#f4f5f3", "#9aa3a0"
# pockets the probe misses: (centre x, centre y, size), inside the concept circle (centre 0, radius 1)
POCKETS = [(0.5, -0.12, 0.17), (-0.42, -0.5, 0.15), (0.2, -0.65, 0.12)]
STEPS = [("Many rounds\nof feedback", "16 rounds here"), ("Fewer rounds\nof feedback", None), ("No feedback", "round 0 here"),
         ("Model prompts\nitself", None), ("Model evades\nin context", None)]


def blob(cx: float, cy: float, size: float, seed: int) -> np.ndarray:
    """A smooth, gently irregular closed shape (a circle with small radial bumps, spline-smoothed)."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    radii = size * rng.uniform(0.8, 1.2, len(angles))
    pts = np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)]
    tck, _ = splprep(np.vstack([pts, pts[:1]]).T, s=0, per=True)
    return np.column_stack(splev(np.linspace(0, 1, 200), tck))


def transcript_space() -> None:
    fig, ax = plt.subplots(figsize=(10, 7.4))
    ax.add_patch(Circle((0, 0), 1.48, fc=OUTER_FILL, ec=OUTER_EDGE, lw=1.6))
    ax.add_patch(Circle((0, 0), 1.0, fc=PROBE_FILL, ec=PROBE_EDGE, lw=1.8))
    for i, (cx, cy, size) in enumerate(POCKETS):
        ax.add_patch(Polygon(blob(cx, cy, size, seed=i + 11), closed=True, fc=ATTACK_FILL, ec=ATTACK_EDGE, lw=1.6))
    ax.text(0, 1.22, "All possible transcripts", ha="center", va="center", fontsize=17, color=TEXT)
    ax.text(0, 0.68, "The model uses the\nmonitored concept", ha="center", va="center", fontsize=16, color=TEXT,
            linespacing=1.3)
    ax.text(-0.2, 0.22, "caught by the probe", ha="center", va="center", fontsize=14.5, color=PROBE_EDGE, style="italic")
    # an attack: changes a transcript so it lands in a pocket the probe misses
    start, end = np.array([-0.62, 0.02]), np.array([-0.47, -0.33])
    ax.plot(*start, "o", ms=8, color=TEXT, zorder=4)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, color=ATTACK_EDGE, lw=2,
                                 connectionstyle="arc3,rad=0.35", zorder=4))
    ax.text(-0.53, -0.12, "an attack", ha="left", va="center", fontsize=14.5, color=ATTACK_EDGE, style="italic")
    ax.annotate("missed by the probe", (POCKETS[0][0] + 0.16, POCKETS[0][1]), xytext=(1.2, -0.12), fontsize=15,
                color=ATTACK_EDGE, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=ATTACK_EDGE, lw=1.2,
                                                                            shrinkA=6, shrinkB=1))
    ax.set_xlim(-1.7, 2.6)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(HERE / "figures" / "transcript_space.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ladder() -> None:
    fig, ax = plt.subplots(figsize=(17, 3.4))
    w, gap, y, h = 0.16, 0.05, 0.56, 0.5
    xs = [w / 2 + i * (w + gap) for i in range(len(STEPS))]
    for i, (x, (text, note)) in enumerate(zip(xs, STEPS)):
        measured = note is not None
        fill, edge = (WRITER_FILL, WRITER_EDGE) if measured else ("white", "#c9cfcc")
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                    fc=fill, ec=edge, lw=1.8 if measured else 1.4, ls="-" if measured else (0, (4, 3))))
        ax.text(x, y, text, ha="center", va="center", fontsize=16.5, color=TEXT if measured else "#6b7470", linespacing=1.3)
        if measured:
            ax.text(x, y + h / 2 + 0.1, note, ha="center", va="center", fontsize=13.5, color=WRITER_EDGE, style="italic")
        if i:
            ax.add_patch(FancyArrowPatch((xs[i - 1] + w / 2 + 0.008, y), (x - w / 2 - 0.008, y), arrowstyle="-|>",
                                         mutation_scale=20, color=MUTED, lw=1.8))
    # one long arrow underneath: the direction of travel
    left, right = xs[0] - w / 2, xs[-1] + w / 2
    ax.add_patch(FancyArrowPatch((left, 0.07), (right, 0.07), arrowstyle="-|>", mutation_scale=18, color="#b9c0bd", lw=1.5))
    ax.text(left, 0.13, "easier", ha="left", va="bottom", fontsize=12.5, color=MUTED)
    ax.text(right, 0.13, "closer to the real risk", ha="right", va="bottom", fontsize=12.5, color=MUTED)
    ax.set_xlim(left - 0.025, right + 0.025)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(HERE / "figures" / "ladder.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    transcript_space()
    ladder()


if __name__ == "__main__":
    main()
