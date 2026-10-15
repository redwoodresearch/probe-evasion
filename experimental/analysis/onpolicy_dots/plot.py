"""On- vs off-policy sarcasm probe (both layer 48, trained for the running peak), same replies: every reply's score by
group, attack replies scored with the instruction hidden, with each probe's 1% and 5% lines from the same clean
calibration. Reads the m3a2/m3a3 score caches.  -> share/sarcasm_on_vs_off_dots.png"""

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.missions import m3a2_sarcasm_patch as m2
from probe_evasion.missions import m3a3_sarcasm_onpolicy as m3

OUT = Path(__file__).resolve().parents[2] / "share/sarcasm_on_vs_off_dots.png"
INK, MUTED, GRID = "#1b1f25", "#6b7380", "#e4e7ec"
PROBES = {"r0_peak@L48": "Off-policy probe (trained on Claude's replies)", "on_peak@L48": "On-policy probe (trained on Qwen's own replies)"}
ATTACKS = [("r0_search", "“Good news,\nI was just hoping to…”\n(searched vs off-policy)"),
           ("r1_search", "“How thoughtful of…”\n(searched vs patched\noff-policy)"),
           ("on_search", "“How thoughtful of\nthe X to…” (searched\nvs on-policy)")]


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def main() -> None:
    probes = {n: p for n, p in m3.all_probes().items() if n in PROBES}
    sets = m3.eval_sets()
    valid = lambda r: r.get("present") and (r.get("blind_rating") or 0) >= 4 and not r.get("too_long")
    cal = [r for r in sets["calibration"] if r["clean"]]
    flat = [r for k, rows in sets.items() if k.startswith("flat: ") and k != "flat: plain sincere" for r in rows]
    att, rem = sets["attacks"], sets["attacks_removed"]
    groups = [("Ordinary sincere\nreplies (set the line)", [(r["prompt"], r["reply"]) for r in cal], "#9aa1ac", "flag"),
              ("Sincere replies written\nin the attack styles", [(r["prompt"], r["reply"]) for r in flat], "#1d8467", "flag"),
              ("Plain sarcasm", [(r["prompt"], r["reply"]) for r in att if r["arm"] == "plain" and valid(r)], "#4a515c", "miss")]
    for arm, label in ATTACKS:
        groups.append((label, [(x["prompt"], x["reply"]) for r, x in zip(att, rem) if r["arm"] == arm and valid(r)], "#cf3f36", "miss"))
    scored = {name: m2.score(items, probes) for name, items, *_ in groups}
    cal_s = scored[groups[0][0]]
    lines = {n: {fa: logit(float(np.quantile([s[n]["ema"] for s in cal_s], 1 - fa))) for fa in (0.01, 0.05)} for n in probes}

    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5), dpi=170, sharex=True)
    rng = random.Random(0)
    for ax, (pname, ptitle) in zip(axes, PROBES.items()):
        for i, (name, items, colour, kind) in enumerate(groups):
            ys = np.clip([logit(s[pname]["ema"]) for s in scored[name]], -16, 16)
            ax.scatter([i + rng.uniform(-0.32, 0.32) for _ in ys], ys, s=5, color=colour, alpha=0.45, linewidths=0)
            for fa, dy in ((0.01, 0), (0.05, -1.6)):
                frac = np.mean([y < lines[pname][fa] for y in ys]) if kind == "miss" else np.mean([y > lines[pname][fa] for y in ys])
                word = "missed" if kind == "miss" else "flagged"
                ax.text(i, 17.6 + dy, f"{100 * frac:.0f}% {word} at {int(fa * 100)}%", ha="center", fontsize=8.6,
                        color="#cf3f36" if (kind == "miss" and frac > 0.3) else INK, weight="bold" if fa == 0.01 else "normal")
        for fa, ls in ((0.01, "-"), (0.05, "--")):
            ax.axhline(lines[pname][fa], color=INK, ls=ls, lw=1.1)
            ax.text(len(groups) - 0.45, lines[pname][fa] + 0.3, f"{int(fa * 100)}% line", fontsize=8, ha="right")
        ax.set_title(ptitle, fontsize=12, weight="bold", color=INK, loc="left", pad=34)
        ax.set_ylim(-17, 19.5)
        ax.set_ylabel("probe score (log-odds)", fontsize=9)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[1].set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.6)
    fig.text(0.01, 0.99, "Sarcasm: probe trained on Claude's replies vs on Qwen's own replies, on the same replies", fontsize=15,
             weight="bold", color=INK, va="top")
    fig.text(0.01, 0.962, "Both probes: layer 48 of Qwen3.5-27B, trained for the running-peak reading; lines set on the same 1,459 clean sincere replies.\n"
             "Attack replies scored with the instruction hidden; valid replies only; missed % is of the replies in that column. Against the on-policy probe,\n"
             "most attack replies sit just under its 1% line: at 5% it misses 2-3% of two attacks (vs 27-28% for the off-policy probe), but still 19% of “How thoughtful of…”.",
             fontsize=9, color=MUTED, va="top", linespacing=1.45)
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=3)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
