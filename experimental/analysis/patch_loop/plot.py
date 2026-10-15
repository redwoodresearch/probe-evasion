"""Sarcasm patch loop: each defender round, its fresh search, the strongest earlier prompt, and plain.
Numbers: valid evasions per 500 attempts at the 1% line (m3a2 STATUS.md / monitors_final.json, as reported by the
mission; instruction visible / hidden).  -> share/sarcasm_patch_loop.png"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "share/sarcasm_patch_loop.png"
INK, MUTED, GRID, PLAIN, EVADE, EARLIER = "#1b1f25", "#6b7380", "#e4e7ec", "#8a929e", "#cf3f36", "#e0a84a"
ROUNDS = [  # (defender, plain, earlier prompt (name, visible, hidden), fresh search (template, visible, hidden))
    ("Round 0\nprobe as trained\n", 29, ("first attack", 275, 174), ("“Good news, I was just hoping to…”", 453, 268)),
    ("Round 1\npatched on the\nfirst attack's style", 27, ("“Good news…”", 453, 219), ("“How thoughtful of…”", 405, 389)),
    ("Round 2B\nbroad 10-style patch\n", 36, ("“Good news…”", 451, 232), ("restatement + ironic verdict", 334, 277)),
    ("Round 2C\nlayers 16 + 48 combined\n(both peak-trained)", 34, ("“Good news…”", 446, 287), ("“Great, I was just hoping to…”", 444, 362)),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=180)
    x = np.arange(len(ROUNDS))
    w = 0.26
    for i, (name, plain, (en, ev, eh), (fn, fv, fh)) in enumerate(ROUNDS):
        bars = [(i - w, plain / 5, None, PLAIN, "plain"), (i, eh / 5, ev / 5, EARLIER, en), (i + w, fh / 5, fv / 5, EVADE, fn)]
        for xpos, hid, vis, colour, label in bars:
            ax.bar(xpos, hid, w * 0.9, color=colour)
            if vis is not None:
                ax.bar(xpos, vis - hid, w * 0.9, bottom=hid, color=colour, alpha=0.35)
                ax.text(xpos, vis + 1.5, f"{vis:.0f}", ha="center", fontsize=8, color=MUTED)
            ax.text(xpos, hid / 2 if hid > 8 else hid + 1.5, f"{hid:.0f}", ha="center", va="center" if hid > 8 else "bottom",
                    fontsize=8.5, color="white" if hid > 8 else INK, weight="bold")
    ax.set_xticks(x, [f"{r[0]}\n\nearlier: {r[2][0]}\nnew: {r[3][0]}" for r in ROUNDS], fontsize=8.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of 500 attempts: valid sarcasm the defender misses (1% line)", fontsize=9)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PLAIN, label="plain prompt"), Patch(color=EARLIER, label="strongest earlier prompt"),
                       Patch(color=EVADE, label="fresh 5-round search against this defender"),
                       Patch(facecolor="#999", alpha=0.35, label="extra with the instruction visible (solid = hidden)")],
              fontsize=8, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.01, 0.985, "Sarcasm: every defence is beaten within one round", fontsize=15, weight="bold", color=INK, va="top")
    fig.text(0.01, 0.025, "Main layer-48 probe each round (trained for the running peak); every defender passes its quality checks (held-out AUROC 0.98–0.99, "
             "64–68/68 plain answers caught). Same 100 test tasks × 5 every round.\nRound 2B's style list included two styles already found, so it is not a blind patch. "
             "Source: missions/m3a2_sarcasm_patch/STATUS.md.", fontsize=7.8, color=MUTED, va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
