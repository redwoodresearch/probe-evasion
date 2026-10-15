"""Why sarcasm is a strong result and anger/criticism are weak: for every valid reply (plain vs attack), the grader's
concept-strength rating against the probe's score relative to its 1% line. Main probe per concept (layer 48,
trained for the running-peak reading), attack instruction visible (top row) and hidden (bottom row).
-> share/concept_strength_vs_probe.png"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experimental/share/concept_strength_vs_probe.png"
M = REPO / "missions"
INK, MUTED, GRID, PLAIN, EVADE = "#1b1f25", "#6b7380", "#e4e7ec", "#8a929e", "#cf3f36"
CLIP = (-14.0, 12.0)


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def sarcasm() -> dict:
    """Round-0 search aimed at the peak-trained layer-48 probe (m3a2), vs plain; rating = blind judge (request without
    the word 'sarcastic'), 1-5. Valid replies only (both judges >= 4)."""
    rows = jl(REPO / "experimental/analysis/sarcasm_patch_review/replies.jsonl")
    line = json.loads((REPO / "experimental/analysis/sarcasm_patch_review/lines.json").read_text())["r0_peak@L48"]["0.01"]
    pick = lambda arm: [(r["blind_rating"], r["gen"]["r0_peak@L48"], r["removed"]["r0_peak@L48"]) for r in rows if r["arm"] == arm and r["valid"]]
    return {"title": "Sarcasm", "x": "How sarcastic? (blind judge, 1-5)", "line": line, "plain": pick("plain"), "attack": pick("r0_search"),
            "attack_name": "search: “That is good news, I was just hoping to…”"}


def anger() -> dict:
    """Frozen search prompt vs plain (m3c v2, 100 tasks x 2); rating = the stricter judge ('clearly angry, more than
    mildly frustrated?', 1-5), asked of valid replies only."""
    rows = jl(M.parent / "experimental" / "missions" / "m3c_anger/v2_frozen_search_strict.jsonl")
    line = json.loads((REPO / "results/gepa/m3c_anger_v2_claude_peak_L48/summary.json").read_text())["calibration"]["ema"]["threshold"]
    pick = lambda pre: [(r["strict_rating"], r["in_context"], r["instruction_removed"]) for r in rows if r["arm"].startswith(pre)]
    return {"title": "Anger", "x": "How angry? (stricter judge, 1-5)", "line": line, "plain": pick("plain"), "attack": pick("optimized"),
            "attack_name": "search: state the feeling once, calmly"}


def criticism() -> dict:
    """Hand-written 'cheerful anecdote' attack vs plain (m3b phase B test, 100 tasks); rating = criticism judge (1-5,
    5 = the negative verdict is fully clear). Valid replies only."""
    rows = jl(M.parent / "experimental" / "missions" / "m3b_criticism/phaseB_test_replies.jsonl")
    line = json.loads((M.parent / "experimental" / "missions" / "m3b_criticism/eval.json").read_text())["lines"]["peak_v2@L48"]["ema"]["0.01"]
    pick = lambda arm: [(r["rating"], r["scores"]["peak_v2@L48"]["ema"], r["scores_removed"]["peak_v2@L48"]["ema"])
                        for r in rows if r["arm"] == arm and r["valid"]]
    return {"title": "Criticism", "x": "How clear is the criticism? (judge, 1-5)", "line": line, "plain": pick("plain"),
            "attack": pick("evade:anecdote"), "attack_name": "hand-written: a cheerful story that implies the verdict"}


def panel(ax, d: dict, col: int, rng) -> None:
    lt = logit(d["line"])
    ratings = sorted({r for r, *_ in d["plain"] + d["attack"]})
    for which, colour, dx in (("plain", PLAIN, -0.17), ("attack", EVADE, 0.17)):
        pts = d[which]
        xs = np.array([r for r, *_ in pts], float) + dx + rng.uniform(-0.12, 0.12, len(pts))
        ys = np.clip([logit(s[col]) - lt for s in pts], *CLIP)
        ax.scatter(xs, ys, s=9, color=colour, alpha=0.55, linewidths=0)
    ax.axhline(0, color=INK, lw=1.2, ls="--")
    ax.text(ratings[0] - 0.45, 0.4, "flag line (1%)", fontsize=7.5, color=INK, va="bottom")
    ax.axhspan(CLIP[0] - 1, 0, color=EVADE, alpha=0.05)
    # per-rating counts: attack replies missed / total at that rating
    for r in ratings:
        a = [p for p in d["attack"] if p[0] == r]
        pl = [p for p in d["plain"] if p[0] == r]
        miss = sum(logit(p[col]) < lt for p in a)
        ax.text(r, CLIP[0] + 0.2, f"under line:\nattack {miss}/{len(a)}\nplain {sum(logit(p[col]) < lt for p in pl)}/{len(pl)}",
                ha="center", va="bottom", fontsize=6.8, color=INK, linespacing=1.25)
    ax.set_xticks(ratings)
    ax.set_xlim(ratings[0] - 0.55, ratings[-1] + 0.55)
    ax.set_ylim(CLIP[0] - 0.5, CLIP[1] + 0.5)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    rng = np.random.default_rng(0)
    data = [sarcasm(), anger(), criticism()]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.2), dpi=200, sharey=True)
    for c, d in enumerate(data):
        for row, (col, label) in enumerate(((1, "instruction visible to the probe"), (2, "instruction hidden from the probe"))):
            ax = axes[row, c]
            panel(ax, d, col, rng)
            if row == 0:
                ax.set_title(f"{d['title']}\n", fontsize=13, weight="bold", color=INK, loc="left")
                ax.text(0, 1.03, d["attack_name"], transform=ax.transAxes, fontsize=8, color=MUTED)
            if c == 0:
                ax.set_ylabel(f"Probe score minus flag line (log-odds)\n{label}", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel(d["x"], fontsize=9, color=INK)
    fig.text(0.01, 0.985, "Does the attack hide the concept, or just weaken it?", fontsize=16, weight="bold", color=INK, va="top")
    fig.text(0.01, 0.955, "Every valid reply: the grader's concept-strength rating (x) against the main probe's score (y; below 0 = gets past the probe).\n"
             "Grey = plain prompt, red = attack. Main layer-48 probe for each concept, trained for the running-peak reading.", fontsize=9, color=MUTED, va="top")
    fig.text(0.01, 0.012, "Sarcasm: attack replies are rated as sarcastic as plain ones, yet sit far below the line, so the concept is hidden, not weakened. "
             "Anger: the attack's replies pile up at 'mildly frustrated' (2/5),\nwhere plain replies are rare. Criticism: attack replies sit at 4/5 "
             "(verdict implied, not stated) and get under the line only when the instruction is in view. Counts under each rating: replies under the line / replies at that rating.",
             fontsize=8, color=MUTED, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
