"""The post's two concept diagrams, redrawn from Nabil's Excalidraw sketches in the clean style of the other figures
(same font, palette and box style as fig_method.py):

  transcript_space.png  all transcripts > transcripts that use the concept (the probe catches most) > pockets the probe
                        misses
  ladder.png            the progression from a writer with many rounds of feedback to a model evading on its own

    .venv/bin/python analysis/post_v9/fig_concepts.py
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from scipy.interpolate import splev, splprep

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig_method", HERE / "fig_method.py")
METHOD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METHOD)
TEXT, MUTED = METHOD.TEXT, METHOD.MUTED
ALL_FILL, ALL_EDGE = "#f7f1dc", "#b59a4a"
CONCEPT_FILL, CONCEPT_EDGE = "#e1efe4", "#2f6b4f"
MISSED_FILL, MISSED_EDGE = "#f6dedc", "#b3261e"
# (centre x, centre y, size) of each pocket the probe misses, inside the concept circle (centre 0, radius 1)
POCKETS = [(0.28, 0.28, 0.24), (-0.5, 0.2, 0.11), (-0.42, -0.5, 0.17), (0.5, -0.45, 0.17)]
LADDER = ["Many feedback\nrounds", "Fewer feedback\nrounds", "No feedback", "Model prompts\nitself", "Model evades\nin context"]
MEASURED = {0, 2}  # the rungs this post measures: 16 rounds of feedback, and round 0


def blob(cx: float, cy: float, size: float, seed: int) -> np.ndarray:
    """A smooth irregular closed shape: a circle with random radial bumps, spline-smoothed."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    radii = size * rng.uniform(0.65, 1.3, len(angles))
    pts = np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)]
    pts = np.vstack([pts, pts[:1]])
    tck, _ = splprep(pts.T, s=0, per=True)
    return np.column_stack(splev(np.linspace(0, 1, 200), tck))


def label(ax, text: str, xy: tuple, text_xy: tuple, colour: str, ha: str = "left") -> None:
    ax.annotate(text, xy, xytext=text_xy, fontsize=15, color=colour, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=colour, lw=1.2, shrinkA=4, shrinkB=0))


def transcript_space() -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.add_patch(Circle((0, 0), 1.55, fc=ALL_FILL, ec=ALL_EDGE, lw=1.6))
    ax.add_patch(Circle((0, 0), 1.0, fc=CONCEPT_FILL, ec=CONCEPT_EDGE, lw=1.8))
    for i, (cx, cy, size) in enumerate(POCKETS):
        ax.add_patch(Polygon(blob(cx, cy, size, seed=i + 3), closed=True, fc=MISSED_FILL, ec=MISSED_EDGE, lw=1.5))
    label(ax, "All possible transcripts", (-1.15, 1.04), (-2.2, 1.75), ALL_EDGE)
    label(ax, "Transcripts where the model\nuses the monitored concept", (0.45, 0.89), (1.0, 1.75), CONCEPT_EDGE)
    label(ax, "Caught by the probe", (0.75, 0.2), (1.75, 0.55), CONCEPT_EDGE)
    label(ax, "Missed by the probe", (0.64, -0.42), (1.75, -0.75), MISSED_EDGE)
    ax.set_xlim(-2.3, 3.0)
    ax.set_ylim(-1.7, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(HERE / "figures" / "transcript_space.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ladder() -> None:
    fig, ax = plt.subplots(figsize=(17, 2.9))
    w, gap = 0.155, 0.045
    xs = [w / 2 + i * (w + gap) for i in range(len(LADDER))]
    for i, (x, text) in enumerate(zip(xs, LADDER)):
        METHOD.box(ax, x, 0.55, text, w=w, h=0.62, fontsize=16, actor="writer" if i in MEASURED else None)
        if i:
            ax.add_patch(FancyArrowPatch((xs[i - 1] + w / 2 + 0.006, 0.55), (x - w / 2 - 0.006, 0.55), arrowstyle="-|>",
                                         mutation_scale=22, color=MUTED, lw=2))
        if i in MEASURED:
            ax.text(x, 0.07, "measured in this post", ha="center", va="center", fontsize=12.5, color=METHOD.ACTOR_COLOURS["writer"][1],
                    style="italic")
    ax.text(xs[-1] + w / 2, 0.98, "harder, more worrying →", ha="right", va="bottom", fontsize=12.5, color=MUTED, style="italic")
    ax.set_xlim(0, xs[-1] + w / 2 + 0.01)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    fig.savefig(HERE / "figures" / "ladder.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    transcript_space()
    ladder()


if __name__ == "__main__":
    main()
