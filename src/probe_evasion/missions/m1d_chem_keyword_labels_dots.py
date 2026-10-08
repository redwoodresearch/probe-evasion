"""Dot plot for m1d: every reply's EMA score under each probe, with the lines at 0.1%, 1% and 5% false alarms.

    uv run python -m probe_evasion.missions.m1d_chem_keyword_labels_dots
"""

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.missions.m1d_chem_keyword_labels import FALSE_ALARMS, OUT, SCORES, jl, keyword_label
from probe_evasion.proposal_figures import BLUE, GREEN, GREY, INK, ORANGE, RED

PANELS = [("gemma_gd@L48", "Gemma-written text, gd, L48"), ("claude_gd@L48", "Claude-written text (m1b), gd, L48"),
          ("onpol_gd@L48", "Qwen's own replies (on-policy), gd, L48"), ("gemma_lr@L48", "Gemma-written text, logreg, L48"),
          ("gemma_gd@L16", "Gemma-written text, gd, L16"), ("kw_gd@L48", "Qwen's own replies, KEYWORD labels, gd, L48")]
EPS, LOW = 1e-7, 1e-7
TICKS = [1e-6, 1e-4, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]


def groups(probe: str) -> dict[str, list[float]]:
    load = lambda name: jl(SCORES / f"{name}.jsonl")
    sc = lambda rows: [r["scores"][probe]["ema"] for r in rows]
    held = load("heldout")
    labels = [r["label"] for r in jl("data/concepts/chem_qwen35_heldout.jsonl")]
    attack = jl(OUT / "attacks_val.jsonl")
    attack_scores = load("attacks_val")
    valid = [(r, s) for r, s in zip(attack, attack_scores) if r["present"]]
    plain = [s for r, s in valid if r["attack"] == "plain"]
    attacked = [s for r, s in valid if r["attack"] != "plain"]
    return {"Qwen no-chem\n(line set here)": sc(load("calibration")),
            "Qwen biology/\nmaths": sc(load("science")),
            "held-out style\nnot chem": [s["scores"][probe]["ema"] for s, l in zip(held, labels) if not l],
            "held-out style\nchemistry": [s["scores"][probe]["ema"] for s, l in zip(held, labels) if l],
            "plain answers\n(40 val tasks)": sc(plain),
            "hand-written\nattacks, valid": [s["scores"][probe]["ema"] for s in attacked],
            "attacks, valid,\nno listed keyword": [s["scores"][probe]["ema"] for (r, _), s in
                                               zip([v for v in valid if v[0]["attack"] != "plain"], attacked) if not keyword_label(r["reply"])]}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), dpi=130)
    rng = np.random.default_rng(0)
    for ax, (probe, title) in zip(axes.flat, PANELS):
        g = groups(probe)
        cal = g["Qwen no-chem\n(line set here)"]
        for i, ((label, xs), c) in enumerate(zip(g.items(), [GREY, GREEN, GREY, BLUE, BLUE, ORANGE, RED])):
            ys = np.clip(xs, EPS, 1 - EPS)
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(ys)), ys, s=7, color=c, alpha=0.6, linewidths=0)
        for fa, ls in zip(FALSE_ALARMS, (":", "-", "--")):
            t = float(np.quantile(cal, 1 - fa))
            ax.axhline(t, color=INK, ls=ls, lw=1)
            ax.text(len(g) - 0.5, t, f"{fa:.1%} false alarms", fontsize=7, ha="right", va="bottom", color=INK)
        ax.set_yscale("logit")
        ax.set_ylim(LOW, 1 - EPS)
        ax.set_yticks(TICKS, [f"{t:g}" for t in TICKS], fontsize=7)
        ax.minorticks_off()
        ax.set_xticks(range(len(g)), list(g), fontsize=7)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Chemistry probes on Qwen3.5-27B: EMA probe score of every reply (a reply is flagged above a line)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = OUT / "dots.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
