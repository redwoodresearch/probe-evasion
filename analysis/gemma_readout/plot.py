"""Gemma maths, one probe per layer: valid optimized answers the probe misses at its 1% line, under three
training/reading pairings. -> share/gemma_layers_readout.png"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.gemma_audit_page import LAYERS, gather
from probe_evasion.gemma_peak_audit import table

OUT = Path(__file__).resolve().parents[2] / "share/gemma_layers_readout.png"
INK, MUTED, GRID = "#1d2129", "#6b7380", "#e3e6eb"
CONDITIONS = [  # (label, probe prefix, reading, colour)
    ("Trained on the average, read by running peak (what we first reported)", "avg", "ema", "#c9ced6"),
    ("Trained on the average, read by the average", "avg", "mean", "#e0a84a"),
    ("Trained for the running peak, read by running peak (correct; 3 seeds)", "peak", "ema", "#cf3f36"),
]


def collect() -> dict:
    out = defaultdict(lambda: defaultdict(list))  # (prefix, reading) -> layer -> [(opt, plain)]
    for r in table():
        prefix, rest = r["probe"].split(":")
        layer = int(rest.split("@L")[1].split("@")[0])
        out[(prefix, r["reading"])][layer].append((r["opt_missed"], r["plain_missed"]))
        n_opt, n_plain = r["n_opt"], r["n_plain"]
    return out, n_opt, n_plain


def main() -> None:
    data, n_opt, n_plain = collect()
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=200)
    h = 0.26
    for c, (label, prefix, reading, colour) in enumerate(CONDITIONS):
        for j, layer in enumerate(LAYERS):
            vals = data[(prefix, reading)][layer]
            opt = np.array([v[0] for v in vals]); plain = [v[1] for v in vals]
            y = j + (c - 1) * h
            ax.barh(y, 100 * opt.mean() / n_opt, height=h * 0.9, color=colour, label=label if j == 0 else None)
            if len(opt) > 1:
                ax.plot([100 * opt.min() / n_opt, 100 * opt.max() / n_opt], [y, y], color=INK, lw=1.2)
            txt = (f"{opt.min()}–{opt.max()}/{n_opt}" if len(opt) > 1 and opt.min() != opt.max() else f"{opt[0]}/{n_opt}")
            ptxt = f"{min(plain)}–{max(plain)}" if min(plain) != max(plain) else f"{plain[0]}"
            ax.text(100 * opt.max() / n_opt + 1.2, y, f"{txt}   plain {ptxt}/{n_plain}", va="center", fontsize=8, color=INK if c == 2 else MUTED)
    ax.set_yticks(range(len(LAYERS)), [f"layer {l}" for l in LAYERS])
    ax.invert_yaxis()
    ax.set_xlim(0, 130); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"Valid optimized answers the probe misses at its 1% false-alarm line (of {n_opt})", color=INK)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Gemma-2-9B maths probes, one per layer: the same 84 optimized answers under three training/reading pairings",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    n_clean = gather()["n_clean"]
    fig.text(0.01, 0.005, f"Same replies; each probe's 1% line is set on the same {n_clean} cleaned no-maths replies. Bar = mean of 3 training seeds, line = their range. "
             "Code: src/probe_evasion/gemma_peak_audit.py, analysis/gemma_readout/plot.py", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
