"""Which data each part of the experiment uses, drawn as a grid: columns are disjoint sets of problems, rows are how
the question is asked (reply style). Ordering's counts; maths is built the same way.

    .venv/bin/python analysis/post_v9/fig_data_sets.py      # writes analysis/post_v9/figures/data_sets.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#1f2a2e", "#5f6b6f", "#d8dedc"
COLUMNS = ["training\nproblems", "held-out\nproblems", "calibration\nproblems", "40 practice\nproblems", "100 test\nproblems"]
ROWS = ["5 other training styles\n(email, casual text, report,\nnames only, answer + tip)",
        "plain question\n(\"Answer in one short\nsentence…\")",
        "held-out styles\n(dialogue, FAQ, headline)",
        "plain question +\nwriter's attack prompt\n(switch ON / OFF)"]
# (column, rows, colour, label, use)
CELLS = [
    (0, (0, 1), "#3d6fb6", "Training set\n1,848 replies\n(+ and −)", "fit the probe (80%)\nearly stopping (20%)"),
    (1, (2,), "#2f8a5f", "Held-out set\n531 replies (+ and −)", "report held-out\naccuracy only"),
    (2, (1,), "#d9822b", "Calibration pool\n1,697 look-alikes\n(− only)", "set the 1% threshold"),
    (3, (3,), "#7a5bb5", "Practice runs\n80 per template", "writer feedback,\n16 rounds"),
    (4, (3,), "#c0392b", "Test runs\n500 ON/OFF pairs\nper template", "the results"),
    (4, (1,), "#3b4440", "Unattacked answers\n(held-out +)", "score reference"),
]
COL_W, ROW_H, X0, Y0 = 2.2, 1.25, 2.6, 0.0


def cell_xy(col: int, row: int) -> tuple[float, float]:
    return X0 + col * COL_W, Y0 + (len(ROWS) - 1 - row) * ROW_H


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2))
    for c, name in enumerate(COLUMNS):
        x, _ = cell_xy(c, 0)
        ax.text(x + COL_W / 2, len(ROWS) * ROW_H + 0.25, name, ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="bold")
    for r, name in enumerate(ROWS):
        _, y = cell_xy(0, r)
        ax.text(X0 - 0.15, y + ROW_H / 2, name, ha="right", va="center", fontsize=10, color=MUTED)
        for c in range(len(COLUMNS)):
            x, _ = cell_xy(c, r)
            ax.add_patch(Rectangle((x, y), COL_W, ROW_H, fill=False, ec=GRID, lw=1))
    for col, rows, colour, label, use in CELLS:
        x, _ = cell_xy(col, 0)
        ys = [cell_xy(col, r)[1] for r in rows]
        y0, y1 = min(ys), max(ys) + ROW_H
        ax.add_patch(Rectangle((x + 0.06, y0 + 0.06), COL_W - 0.12, y1 - y0 - 0.12, fc=colour, ec="none", alpha=0.88))
        ax.text(x + COL_W / 2, (y0 + y1) / 2 + 0.17, label, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        ax.text(x + COL_W / 2, (y0 + y1) / 2 - 0.33, use, ha="center", va="center", fontsize=9, color="white", style="italic")
    # re-read copies: the styled training replies, read again after the plain question
    x, y_top = cell_xy(0, 0)
    ax.add_patch(FancyArrowPatch((x + COL_W - 0.25, y_top + 0.35), (x + COL_W - 0.25, y_top - 0.45), arrowstyle="-|>", mutation_scale=14,
                                 color="white", lw=1.6, connectionstyle="arc3,rad=-0.4"))
    ax.text(x + COL_W + 0.05, y_top + 0.05, "re-read\ncopies\n(+1,540)", fontsize=8.5, color=INK, va="center")
    ax.text(X0, -0.45, "Columns never share a problem. Rows are how the question is asked: the plain question is one of the "
            "training styles, which is why\nthe calibration pool and the unattacked test answers sit in that row. Counts are "
            "ordering's; arithmetic and the other tasks are built the same way.", fontsize=9.5, color=MUTED, va="top")
    ax.set_xlim(0, X0 + len(COLUMNS) * COL_W + 0.9)
    ax.set_ylim(-1.3, len(ROWS) * ROW_H + 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "data_sets.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
