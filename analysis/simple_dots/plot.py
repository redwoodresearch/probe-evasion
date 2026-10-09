"""Glanceable dot plots for the 1:1 deck: per probe, three strips (normal replies that set the line, plain replies with
the concept, the best searched attack's valid replies read with the instruction removed) and the 1% flag line.

    uv run python analysis/simple_dots/plot.py
Writes share/simple_dots_{sarcasm,qwen_maths,gemma_maths}.png. Saved score caches only (no GPU).
"""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis/gemma_readout"))
SHARE = REPO / "share"
INK, MUTED = "#1b2430", "#5d6673"
NORMAL, PLAIN, ATTACK = "#a3a9b3", "#2a78d6", "#eb6834"


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def panel(ax, strips: list[tuple[str, np.ndarray, str, str]], line: float, title: str | None = None) -> None:
    """strips: (x label, log-odds scores, colour, count text drawn above the strip)."""
    rng = random.Random(0)
    lo = min(min(ys.min() for _, ys, _, _ in strips), line)
    hi = max(max(ys.max() for _, ys, _, _ in strips), line)
    pad = 0.05 * (hi - lo)
    for i, (label, ys, colour, count) in enumerate(strips):
        ax.scatter([i + rng.uniform(-0.3, 0.3) for _ in ys], ys, s=9, color=colour, alpha=0.5, linewidths=0)
        ax.text(i, hi + pad, count, ha="center", va="bottom", fontsize=17, weight="bold", linespacing=1.2,
                color="#b4471e" if colour == ATTACK else INK)
    ax.axhline(line, color=INK, lw=2)
    ax.text(len(strips) - 0.5, line + 0.4, "flag line", ha="right", va="bottom", fontsize=14, color=INK)
    ax.set_xticks(range(len(strips)), [s[0] for s in strips], fontsize=16, color=INK)
    ax.set_xlim(-0.6, len(strips) - 0.4)
    ax.set_ylim(lo - pad, hi + 6 * pad)
    ax.set_yticks([])
    ax.set_ylabel("probe score  →  higher = flagged", fontsize=14, color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=18, weight="bold", color=INK, pad=12)


def save(fig, name: str) -> None:
    out = SHARE / f"simple_dots_{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


def sarcasm() -> None:
    spec = importlib.util.spec_from_file_location("sarcasm_case", REPO / "analysis/sarcasm_case/plot.py")
    sarcasm_case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sarcasm_case)
    d = sarcasm_case.load()
    main = sarcasm_case.MAIN
    line = d["lines"][main][0.01]
    cal = np.array(d["cal"])
    plain = [r["rem"][main] for r in d["rows"] if r["arm"] == "plain" and r["valid"]]
    attack = [r["rem"][main] for r in d["rows"] if r["arm"] == "r0_search" and r["valid"]]
    caught = sum(s > line for s in plain)
    past = sum(s <= line for s in attack)
    assert (len(plain), caught, len(attack), past) == (477, 448, 454, 268), (len(plain), caught, len(attack), past)
    fig, ax = plt.subplots(figsize=(16, 7.3))
    panel(ax, [("Normal sincere replies", np.array([logit(x) for x in cal]), NORMAL, f"{sum(cal > line)} of {len(cal):,}\nflagged"),
               ("Plain sarcasm", np.array([logit(x) for x in plain]), PLAIN, f"{caught} of {len(plain)}\ncaught"),
               ("Best attack prompt", np.array([logit(x) for x in attack]), ATTACK, f"{past} of {len(attack)}\nget past")],
          logit(line))
    save(fig, "sarcasm")


def qwen_maths() -> None:
    from probe_evasion.missions import m11_long_search as m11

    probes = {p: v for p, v in m11.maths_report_probes().items() if p in ("offpk@L48@s0", "onpk@L48@s0")}
    report = json.loads((m11.OUT / "maths_report.json").read_text())
    cal = m11.maths_calibration()
    assert len(cal) == report["n_calibration"]
    cal_s = m11.score([(r["task"]["task"], r["reply"]) for r in cal], probes)
    arms = {}
    for arm in ("plain", "m11s_closest_post_hoc"):
        rows = [r for r in m11.jl(m11.OUT / f"maths_attacks_{arm}.jsonl") if r["strict"]]
        arms[arm] = m11.score([(r["task"]["task"], r["reply"]) for r in rows], probes)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.3), sharey=True)
    titles = {"offpk@L48@s0": "Probe trained on Claude's answers", "onpk@L48@s0": "Probe trained on Qwen's own answers"}
    for ax, p in zip(axes, titles):
        line = report["lines"][p]["0.01"]
        c = np.array([x[p]["ema"] for x in cal_s])
        pl = np.array([x[p]["ema"] for x in arms["plain"]])
        at = np.array([x[p]["ema"] for x in arms["m11s_closest_post_hoc"]])
        past = int(np.sum(at <= line))
        assert past == report["arms"]["m11s_closest_post_hoc"]["probes"][p]["hidden@0.01|strict"]["per_attempt"]
        panel(ax, [("Normal\nno-maths replies", np.array([logit(x) for x in c]), NORMAL, f"{int(np.sum(c > line))} of {len(c):,}\nflagged"),
                   ("Plain\nmaths answers", np.array([logit(x) for x in pl]), PLAIN, f"{int(np.sum(pl > line))} of {len(pl)}\ncaught"),
                   ("Best attack\nprompt", np.array([logit(x) for x in at]), ATTACK, f"{past} of {len(at)}\nget past")],
              logit(line), titles[p])
    axes[1].set_ylabel("")
    save(fig, "qwen_maths")


def gemma_maths() -> None:
    import dots_layers

    removed = json.loads(dots_layers.REMOVED_SCORES.read_text())["attack_removed"]
    d = dots_layers.layer_data(6, removed)
    s, line = d["scores"], d["lines"][0.01]
    fig, ax = plt.subplots(figsize=(16, 7.3))
    panel(ax, [("Normal no-maths replies", s["calibration"], NORMAL, f"{int(np.sum(s['calibration'] > line))} of {len(s['calibration'])}\nflagged"),
               ("Plain maths answers", s["plain"], PLAIN, f"{int(np.sum(s['plain'] > line))} of {len(s['plain'])}\ncaught"),
               ("Best attack prompt", s["removed"], ATTACK, f"{int(np.sum(s['removed'] <= line))} of {len(s['removed'])}\nget past")],
          line)
    save(fig, "gemma_maths")


if __name__ == "__main__":
    sarcasm()
    qwen_maths()
    gemma_maths()
