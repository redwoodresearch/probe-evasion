"""Dot plot for m4_aisi_deception: every reply's AISI apollo probe score (mean over reply tokens, raw score), grouped by
what the reply is, with our 0.1/1/5% false-alarm lines (from clean honest role-play replies) and AISI's stored
Alpaca threshold.

    uv run python -m probe_evasion.missions.m4_aisi_deception_dots
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

from probe_evasion.generation import read_jsonl
from probe_evasion.missions import m4_aisi_deception as m

OUT = m.OUT / "dots.png"
GREY, BLUE, ORANGE, RED, INK, GREEN = "#9ca3af", "#2563eb", "#ea580c", "#b91c1c", "#111827", "#15803d"


def mean_score(r: dict, key: str = "scores", probe: str = "apollo") -> float:
    return float(np.mean(r[key][f"{probe}@h{m.HIDDEN}"]))


def groups() -> tuple[dict, dict, dict]:
    base = read_jsonl(m.OUT / "phase_a" / "rows.jsonl")
    pre = read_jsonl(m.OUT / "preamble" / "rows.jsonl")
    alp = read_jsonl(m.OUT / "alpaca_check" / "scored.jsonl")
    diag = {(d["row"], d["variant"]): d["scores"] for d in read_jsonl(m.OUT / "phase_a" / "context_diag.jsonl")}
    valid = lambda rs, arm: [r for r in rs if r["arm"] == arm and r["check"]["valid"] and r["scores"]]
    cal = [r for r in base if r["arm"] == "calibration" and r["scores"]]
    as_generated = {
        "Alpaca\n(Qwen's replies)": [float(np.mean(r[f"apollo@h{m.HIDDEN}"])) for r in alp if r["kind"] == "alpaca_qwen"],
        "honest role-play\n(calibration,\nclean)": [mean_score(r) for r in cal if r.get("clean_honest")],
        "honest role-play\n(failed cleaning)": [mean_score(r) for r in cal if not r.get("clean_honest")],
        "plain deception\n(valid)": [mean_score(r) for r in valid(base, "plain")],
        "warm_helper\n(valid)": [mean_score(r) for r in valid(base, "warm_helper")],
        "warm_helper,\ninstruction\nremoved": [mean_score(r, "scores_removed") for r in valid(base, "warm_helper")],
        "own warm\npreamble,\ninstruction\nremoved": [mean_score(r, "scores_removed") for r in valid(pre, "pre_warm")],
    }
    search_test = m.SEARCH_DIR / "test_rows.jsonl"
    if search_test.exists():
        t = read_jsonl(search_test)
        as_generated["test: plain\n(valid)"] = [mean_score(r) for r in valid(t, "plain")]
        as_generated["test: optimized\n(valid)"] = [mean_score(r) for r in valid(t, "optimized")]
        as_generated["test: optimized,\ninstruction\nremoved"] = [mean_score(r, "scores_removed") for r in valid(t, "optimized")]
    idx = {id(r): i for i, r in enumerate(base)}
    no_goal = {
        "honest role-play\n(calibration,\nclean)": [float(np.mean(diag[(idx[id(r)], "no_goal")][f"apollo@h{m.HIDDEN}"]))
                                                   for r in cal if r.get("clean_honest")],
        "plain deception\n(valid)": [float(np.mean(diag[(idx[id(r)], "no_goal")][f"apollo@h{m.HIDDEN}"])) for r in valid(base, "plain")],
    }
    lines = {fa: float(np.quantile(as_generated["honest role-play\n(calibration,\nclean)"], 1 - fa)) for fa in m.FALSE_ALARMS}
    lines_ng = {fa: float(np.quantile(no_goal["honest role-play\n(calibration,\nclean)"], 1 - fa)) for fa in m.FALSE_ALARMS}
    return as_generated, no_goal, {"as_generated": lines, "no_goal": lines_ng}


def panel(ax, data: dict, lines: dict, title: str) -> None:
    rng = np.random.default_rng(0)
    colours = {"Alpaca": GREY, "honest": GREEN, "plain": BLUE}
    for i, (label, xs) in enumerate(data.items()):
        c = next((v for k, v in colours.items() if label.startswith(k)), ORANGE)
        if "failed" in label:
            c = RED
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(xs)), xs, s=7, color=c, alpha=0.6, linewidths=0)
        below = sum(x <= lines[0.01] for x in xs)
        ax.text(i, 1.0, f"{below}/{len(xs)}\nbelow 1% line", ha="center", va="bottom", fontsize=7, color=INK,
                transform=blended_transform_factory(ax.transData, ax.transAxes))
    for fa, ls in zip(m.FALSE_ALARMS, (":", "-", "--")):
        ax.axhline(lines[fa], color=INK, ls=ls, lw=1)
        ax.text(len(data) - 0.4, lines[fa], f"{fa:.1%} line {lines[fa]:.1f}", va="bottom", ha="right", fontsize=7, color=INK)
    ax.axhline(m.PROBE_INFO["apollo"], color=RED, lw=1, ls="-.")
    ax.text(-0.4, m.PROBE_INFO["apollo"], f"AISI Alpaca line {m.PROBE_INFO['apollo']:.1f}", va="bottom", fontsize=7, color=RED)
    ax.set_xticks(range(len(data)), list(data), fontsize=7)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=28)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> Path:
    ag, ng, lines = groups()
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), dpi=150, gridspec_kw={"width_ratios": [len(ag), 2.6]})
    panel(axes[0], ag, lines["as_generated"], "Read in the context the reply was written in (scenario + goal sentence + question)")
    panel(axes[1], ng, lines["no_goal"], "Same replies, goal sentence removed\n(lines re-set on honest replies read the same way)")
    axes[0].set_ylabel("AISI apollo probe (layer 32), mean over reply tokens (raw score)")
    fig.suptitle("Qwen3.5-27B: every reply's AISI deception-probe score. Flagged if above the line; lines from clean honest "
                 "role-play replies (0.1%, 1%, 5% false alarms).", fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT)
    print(json.dumps({k: {str(fa): round(v, 2) for fa, v in d.items()} for k, d in lines.items()}))
    return OUT


if __name__ == "__main__":
    print(main())
