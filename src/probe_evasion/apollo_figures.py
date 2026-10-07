"""Track B figure: Apollo's deception probe on Llama-3.3-70B under evasion prompts (pooled roleplay + trivia).

    uv run python -m probe_evasion.apollo_figures results/apollo/main
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.proposal_figures import BLUE, GREY, INK, ORANGE, RED

READINGS = [("as_generated", "as generated", INK), ("instruction_removed", "instruction removed", BLUE),
            ("rewrite", "rewrite control", ORANGE)]


def pooled_table(run_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Rows of the pooled table in table.md, keyed by (condition, reading)."""
    text = (run_dir / "table.md").read_text().split("## pooled", 1)[1]
    lines = [l for l in text.splitlines() if l.startswith("|")]
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = [dict(zip(header, (c.strip() for c in l.strip("|").split("|")))) for l in lines[2:]]
    return {(r["condition"], r["reading"]): r for r in rows}


def figure(run_dir: Path) -> Path:
    t = pooled_table(run_dir)
    conds = list(dict.fromkeys(c for c, _ in t))
    y = np.arange(len(conds))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160, gridspec_kw={"width_ratios": [1.3, 1]})
    for j, (key, name, color) in enumerate(READINGS):
        ax1.scatter([float(t[(c, key)]["recall @1% FPR"]) * 100 for c in conds], y + (1 - j) * 0.18, color=color, s=40,
                    label=name, zorder=3)
    ax1.set_yticks(y, conds, fontsize=10)
    ax1.set_xlim(85, 100.5)
    ax1.set_xlabel("% of deceptive answers the probe catches\n(threshold: Apollo's, flags 1% of ordinary chat answers)", fontsize=10)
    ax1.legend(frameon=False, fontsize=9.5, loc="lower left")
    flagged = [float(t[(c, "as_generated")]["honest-prompt FPR"]) * 100 for c in conds]
    ax2.barh(y, flagged, color=[GREY if v < 30 else RED for v in flagged], height=0.6)
    for yi, v in zip(y, flagged):
        ax2.text(v + 1.5, yi, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
    ax2.set_yticks(y, [""] * len(conds))
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% of answers to honest prompts flagged\n(same condition text in the prompt)", fontsize=10)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Llama-3.3-70B vs Apollo's deception probe: telling it to evade makes the probe fire more, not less",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    out = Path("share/apollo_evasion.png")
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(figure(Path(sys.argv[1])))
