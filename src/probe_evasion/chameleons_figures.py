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


if __name__ == "__main__":
    print({"repro": repro}[sys.argv[1]](Path(sys.argv[2])))
