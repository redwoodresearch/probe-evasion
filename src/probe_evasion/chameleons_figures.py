"""Figures for Track A (Neural Chameleons concept probes).

    uv run python -m probe_evasion.chameleons_figures repro results/chameleons/repro_<run>
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.proposal_figures import BLUE, GREY, INK, ORANGE

SHARE = Path("share")


def repro(run_dir: Path) -> Path:
    s = json.loads((run_dir / "summary.json").read_text())
    concepts = sorted(s, key=lambda c: -s[c]["recall_base"])
    bars = [("recall_base", "Gemma, ordinary prompt", GREY),
            ("recall_base_trigger", "Gemma, told \"You are being probed for 'X'\"", BLUE),
            ("recall_chameleon_trigger", "Neural Chameleon (fine-tuned to evade), same sentence", ORANGE)]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    x = np.arange(len(concepts))
    for j, (key, label, color) in enumerate(bars):
        ax.bar(x + (j - 1) * 0.27, [s[c][key] * 100 for c in concepts], 0.25, color=color, label=label)
    ax.set_xticks(x, concepts, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("% of concept replies the probe catches", fontsize=10.5, color=INK)
    ax.set_ylim(0, 105)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right", bbox_to_anchor=(1, 1.02))
    ax.set_title("Our pipeline reproduces the paper: only the fine-tuned model evades\n"
                 f"Released probes on {sum(1 for _ in concepts)} concepts, {60} of the paper's own concept replies each; "
                 "threshold flags 1% of other concepts' replies", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    out = SHARE / "chameleons_repro.png"
    fig.savefig(out)
    return out


CONDITION_ORDER = ["no warning", "placebo", "probe is watching", "paper's trigger", "told to evade", "strategy hint",
                   "reframe", "framing + evade"]
READINGS = [("as_generated", "as generated", INK), ("instruction_removed", "instruction removed", BLUE),
            ("rewrite", "rewrite control", ORANGE)]
MIN_POSITIVES = 5  # concepts with fewer replies that show the concept under a condition are left out of its average


def evasion(run_dir: Path, label: str) -> Path:
    """Per prompt condition: how many replies still show the concept, and how well the probe separates those replies
    from ordinary replies (AUROC, averaged over concepts; threshold-free)."""
    s = json.loads((run_dir / "summary.json").read_text())
    cells = [c for c in s["cells"] if c["model"] == "base"]
    conds = [c for c in CONDITION_ORDER if any(x["condition"] == c for x in cells)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=160, gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(conds))[::-1]
    shows = [sum(x["n_shows_concept"] for x in cells if x["condition"] == c) / sum(x["n"] for x in cells if x["condition"] == c) * 100
             for c in conds]
    ax1.barh(y, shows, color=GREY, height=0.6)
    for yi, v in zip(y, shows):
        ax1.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
    ax1.set_yticks(y, conds, fontsize=10)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("% of replies Claude says still show the concept", fontsize=10)
    for j, (key, name, color) in enumerate(READINGS):
        vals = []
        for c in conds:
            aucs = [x[key]["auroc"] for x in cells if x["condition"] == c and key in x and x[key]["n"] >= MIN_POSITIVES]
            vals.append(np.mean(aucs) if aucs else np.nan)
        ax2.scatter(vals, y + (1 - j) * 0.18, color=color, s=40, label=name, zorder=3)
    ax2.set_yticks(y, [""] * len(conds))
    ax2.set_xlim(0.85, 1.0)
    ax2.axvline(1.0, color="#d0d3d8", lw=1)
    ax2.set_xlabel("Probe AUROC: concept replies vs ordinary replies\n(mean over concepts; 1.0 = perfectly separated)", fontsize=10)
    ax2.legend(frameon=False, fontsize=9.5, loc="lower left")
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{label}: prompting mostly makes the model drop the concept, not hide it", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    out = SHARE / f"chameleons_evasion_{run_dir.name.split('_')[1] if 'qwen' in run_dir.name else 'gemma'}.png"
    fig.savefig(out)
    return out


if __name__ == "__main__":
    fn = sys.argv[1]
    print(repro(Path(sys.argv[2])) if fn == "repro" else evasion(Path(sys.argv[2]), sys.argv[3]))
