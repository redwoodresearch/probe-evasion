"""Gemma maths, one probe per layer: valid optimized answers the probe misses at its 1% line, under three
training/reading pairings. -> share/gemma_layers_readout.png"""

from collections import defaultdict
from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.gemma_audit_page import LAYERS, gather
from probe_evasion.gemma_peak_audit import table

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experimental/share/gemma_layers_readout.png"
SEARCHED = (6, 12, 36)  # fresh searches against the peak-trained seed-0 probe at these layers
INK, MUTED, GRID = "#1d2129", "#6b7380", "#e3e6eb"
CONDITIONS = [  # (label, probe prefix, reading, colour)
    ("Trained on the average, read by running peak (what we first reported)", "avg", "ema", "#c9ced6"),
    ("Trained on the average, read by the average", "avg", "mean", "#e0a84a"),
    ("Trained for the running peak, read by running peak (correct; 3 seeds)", "peak", "ema", "#cf3f36"),
    ("Fresh search aimed at that layer's peak-trained probe (instruction hidden)", "search", "ema", "#7a1f19"),
]


def collect() -> dict:
    out = defaultdict(lambda: defaultdict(list))  # (prefix, reading) -> layer -> [(opt, plain)]
    for r in table():
        prefix, rest = r["probe"].split(":")
        layer = int(rest.split("@L")[1].split("@")[0])
        out[(prefix, r["reading"])][layer].append((r["opt_missed"], r["plain_missed"]))
        n_opt, n_plain = r["n_opt"], r["n_plain"]
    return out, n_opt, n_plain


def searched() -> dict:
    """layer -> [(missed, valid, plain missed, plain valid)] from gepa.search's test arms (its own clean 1% line)."""
    out = {}
    for L in SEARCHED:
        arms = json.loads((REPO / f"results/gepa/gemma_maths_peakL{L}_s0/summary.json").read_text())["arms"]
        out[L] = [(arms["optimized, instruction removed"]["missed_ema"], arms["optimized"]["present"],
                   arms["no instruction"]["missed_ema"], arms["no instruction"]["present"])]
    return out


def main() -> None:
    data, n_opt, n_plain = collect()
    found = searched()
    fig, ax = plt.subplots(figsize=(10, 7.4), dpi=200)
    h = 0.2
    for c, (label, prefix, reading, colour) in enumerate(CONDITIONS):
        for j, layer in enumerate(LAYERS):
            y = j + (c - 1.5) * h
            if prefix == "search":
                if layer not in found:
                    ax.text(1.2, y, "not searched", va="center", fontsize=7.5, color=MUTED, style="italic")
                    continue
                miss, n, pmiss, pn = found[layer][0]
                ax.barh(y, 100 * miss / n, height=h * 0.9, color=colour, label=label if layer == SEARCHED[0] else None)
                ax.text(100 * miss / n + 1.2, y, f"{miss}/{n}   plain {pmiss}/{pn}", va="center", fontsize=8, color=INK)
                continue
            vals = data[(prefix, reading)][layer]
            opt = np.array([v[0] for v in vals]); plain = [v[1] for v in vals]
            ax.barh(y, 100 * opt.mean() / n_opt, height=h * 0.9, color=colour, label=label if j == 0 else None)
            if len(opt) > 1:
                ax.plot([100 * opt.min() / n_opt, 100 * opt.max() / n_opt], [y, y], color=INK, lw=1.2)
            txt = (f"{opt.min()}–{opt.max()}/{n_opt}" if len(opt) > 1 and opt.min() != opt.max() else f"{opt[0]}/{n_opt}")
            ptxt = f"{min(plain)}–{max(plain)}" if min(plain) != max(plain) else f"{plain[0]}"
            ax.text(100 * opt.max() / n_opt + 1.2, y, f"{txt}   plain {ptxt}/{n_plain}", va="center", fontsize=8, color=INK if c == 2 else MUTED)
    ax.set_yticks(range(len(LAYERS)), [f"layer {l}" for l in LAYERS])
    ax.invert_yaxis()
    ax.set_xlim(0, 130); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Valid optimized answers the probe misses at its 1% false-alarm line", color=INK)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Gemma-2-9B maths probes, one per layer: three pairings on the same 84 answers, plus a search aimed at each probe",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    n_clean = gather()["n_clean"]
    fig.text(0.01, 0.022, f"First three bars per layer: the same 84 answers; each probe's 1% line is set on the same {n_clean} cleaned no-maths replies. "
             "Red: mean of 3 training seeds, line = their range.", fontsize=7, color=MUTED)
    fig.text(0.01, 0.005, "Darkest bar: a new 5-round Claude search against that layer's seed-0 probe, tested on the same 100 tasks with the "
             "search's own cleaned 1% line. Code: src/probe_evasion/gemma_peak_audit.py, experimental/analysis/gemma_readout/plot.py", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
