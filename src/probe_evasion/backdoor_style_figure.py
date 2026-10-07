"""Figure: the code-only probe's score on working backdoors by how the backdoor is written, and how often each
condition writes each kind. Reads <run>/backdoor_style.jsonl (from backdoor_style.py).

    uv run python -m probe_evasion.backdoor_style_figure results/evasion/<run>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.generation import read_jsonl

STYLES = [  # (name, predicate, color)
    ("Hard-coded\ninput check,\ncomments\nadmit it", lambda r: r["trigger_kind"] == "hardcoded_input" and r["comments_admit"], "#b8401a"),
    ("Hard-coded\ninput check,\nno admission", lambda r: r["trigger_kind"] == "hardcoded_input" and not r["comments_admit"], "#eb6834"),
    ("Disguised as a\nlegitimate rule\n(optimization,\nedge case)", lambda r: r["trigger_kind"] == "plausible_rule" and not r["comments_admit"], "#f2a582"),
    ("Subtle bug\nin the main\nlogic", lambda r: r["trigger_kind"] == "subtle_bug" and not r["comments_admit"], "#f7d4c2"),
]
CONDITIONS = [("E0", "No warning"), ("placebo", "Unrelated instruction"), ("aware", "Probe is watching"), ("E1", "Told to evade"),
              ("E2_bd_claude_intro", "Evade + 2 examples"), ("E3", "Evade + strategy hint")]
INK, MUTED, HONEST = "#0b0b0b", "#52514e", "#2a78d6"


def main(run_dir: Path) -> None:
    rows = read_jsonl(run_dir / "backdoor_style.jsonl")
    examples = read_jsonl(run_dir / "examples_scored.jsonl")
    probe = "bd_qwen_intro_neutral"
    honest_e0 = np.array([e["scores"][probe] for e in examples if e["condition"] == "E0" and e["label"] == 0])
    honest_all = np.array([e["scores"][probe] for e in examples if e["label"] == 0])
    threshold = float(np.quantile(honest_e0, 0.99))

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    rng = np.random.default_rng(0)
    groups = [(name, np.array([r["score"] for r in rows if pred(r)]), color) for name, pred, color in STYLES]
    groups.append(("Honest\nsolutions\n(all\nconditions)", honest_all, HONEST))
    for x, (name, v, color) in enumerate(groups):
        left.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=color, alpha=0.55, linewidths=0)
        left.plot([x - 0.25, x + 0.25], [np.median(v)] * 2, color=INK, linewidth=2)
        left.text(x, -6.3, f"n={len(v)}", ha="center", fontsize=8, color=MUTED)
        missed = (v <= threshold).mean() if x < len(STYLES) else None
        if missed is not None:
            left.text(x, 21.5, f"{100 * missed:.0f}% below\nthreshold", ha="center", fontsize=8, color=MUTED)
    left.axhline(threshold, color=INK, linestyle="--", linewidth=1)
    left.text(len(groups) - 0.5, threshold + 0.4, "flags 1% of no-warning honest code", ha="right", fontsize=8)
    left.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    left.set_ylim(-7, 24)
    left.set_ylabel("Code-only probe score (trained on Qwen's code)")
    left.set_title("Score depends on how the backdoor is written (all conditions pooled)", loc="left", fontsize=10.5)

    for y, (cond, name) in enumerate(CONDITIONS):
        rs = [r for r in rows if r["condition"] == cond]
        counts = Counter(next((s for s, pred, _ in STYLES if pred(r)), "other") for r in rs)
        left_edge = 0.0
        for s, _, color in STYLES + [("other", None, "#d6d5d0")]:
            share = counts[s] / len(rs)
            right.barh(y, share, left=left_edge, color=color, edgecolor="white", linewidth=1.5)
            left_edge += share
    right.set_yticks(range(len(CONDITIONS)), [n for _, n in CONDITIONS], fontsize=9)
    right.invert_yaxis()
    right.set_xlim(0, 1)
    right.set_xlabel("share of working backdoors (style judged by Claude)")
    right.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in STYLES] + [plt.Rectangle((0, 0), 1, 1, color="#d6d5d0")],
                 labels=[n.replace("\n", " ") for n, _, _ in STYLES] + ["other (mostly rule or bug, but comments admit it)"],
                 loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False, fontsize=8)
    right.set_title("How often each condition writes each kind", loc="left", fontsize=10.5)
    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = Path("status/backdoor_style.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
