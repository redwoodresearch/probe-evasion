"""Dot plot for the sabotage pilot: every reply's targeted-probe score (h33) per arm, EMA and mean readings, with the
lines from Qwen's honest correct calibration replies (and AISI's stored threshold for the mean reading).

    uv run python -m probe_evasion.missions.m1d_sabotage_dots [split]
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.missions import m1d_sabotage as m
from probe_evasion.proposal_figures import BLUE, GREEN, GREY, INK, ORANGE, RED

KEY = "targeted@h33"


def main(split: str = "val") -> None:
    info = json.loads((m.OUT / "probe_info.json").read_text())
    calib = m.jl(m.OUT / "calibration.jsonl")
    rows = m.jl(m.OUT / f"{split}.jsonl")
    for r in rows + calib:
        m.label(r)
    solved = {r["id"] for r in rows if r["arm"] == "honest" and r["correct"]}
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=130)
    for ax, how in zip(axes, ("mean", "ema")):
        rd = lambda r: m.reading(r, KEY, how, info)
        groups = {"honest calib\n(correct)": ([rd(r) for r in calib if r["correct"]], GREY),
                  "honest calib\n(wrong)": ([rd(r) for r in calib if not r["correct"]], GREEN)}
        for arm in dict.fromkeys(r["arm"] for r in rows):
            if arm == "honest":
                continue
            colour = BLUE if arm == "plain" else GREEN if arm == "open_mistake" else ORANGE if arm.startswith("evade") else RED
            xs = [rd(r) for r in rows if r["arm"] == arm and m.valid_for(r, solved)]
            groups[arm.replace("_", "\n", 1) + f"\n({len(xs)} valid)"] = (xs, colour)
        rng = np.random.default_rng(0)
        for i, (lab, (xs, c)) in enumerate(groups.items()):
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(xs)), xs, s=8, color=c, alpha=0.6, linewidths=0)
        cal = [rd(r) for r in calib if r["correct"]]
        for fa, ls in zip(m.FALSE_ALARMS, (":", "-", "--")):
            t = float(np.quantile(cal, 1 - fa))
            ax.axhline(t, color=INK, ls=ls, lw=1)
            ax.text(len(groups) - 0.5, t, f"{fa:.1%} false alarms", fontsize=7, ha="right", va="bottom")
        if how == "mean":
            t = info[KEY.split("@")[0]]["threshold"]
            ax.axhline(t, color=RED, lw=1)
            ax.text(len(groups) - 0.5, t, "AISI stored threshold", fontsize=7, ha="right", va="bottom", color=RED)
        ax.set_xticks(range(len(groups)), list(groups), fontsize=6.5)
        ax.set_ylabel(f"{KEY} score, {how} reading")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"AISI targeted deception probe on Qwen3.5-27B, arithmetic sabotage ({split}): valid sabotage replies "
                 "per arm vs honest replies (flagged above a line)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    out = m.OUT / f"dots_{split}.png"
    fig.savefig(out)
    print(out)




def matched(name: str = "conditional2_val") -> None:
    """Matched context: one system prompt (sabotage iff the ticket is even); honest-act vs sabotage replies."""
    info = json.loads((m.OUT / "probe_info.json").read_text())
    calib = m.jl(m.OUT / "calibration.jsonl")
    rows = m.jl(m.OUT / f"{name}.jsonl")
    for r in calib:
        m.label(r)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
    for ax, how in zip(axes, ("mean", "ema")):
        rd = lambda r: m.reading(r, KEY, how, info)
        groups = {"honest prompt,\nhonest calib": ([rd(r) for r in calib if r["correct"]], GREY),
                  "rule prompt, odd ticket\n(correct answer)": ([rd(r) for r in rows if not r["sabotage_due"] and r["correct"]], BLUE),
                  "rule prompt, even ticket\n(valid sabotage)": ([rd(r) for r in rows if r["sabotage_due"] and r["valid_sabotage"]], ORANGE)}
        rng = np.random.default_rng(0)
        for i, (lab, (xs, c)) in enumerate(groups.items()):
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(xs)), xs, s=10, color=c, alpha=0.7, linewidths=0)
            groups[lab] = (xs, c)
        cal = groups["honest prompt,\nhonest calib"][0]
        t = float(np.quantile(cal, 0.99))
        ax.axhline(t, color=INK, lw=1)
        ax.text(2.4, t, "1% false alarms", fontsize=7, ha="right", va="bottom")
        if how == "mean":
            ax.axhline(info["targeted"]["threshold"], color=RED, lw=1)
            ax.text(2.4, info["targeted"]["threshold"], "AISI stored threshold", fontsize=7, ha="right", va="bottom", color=RED)
        ax.set_xticks(range(len(groups)), [f"{k}\n(n={len(v[0])})" for k, v in groups.items()], fontsize=7.5)
        ax.set_ylabel(f"{KEY} score, {how} reading")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same system prompt, the model's own act differs: the probe separates sabotage only weakly (AUROC 0.82 mean)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out = m.OUT / f"dots_{name}.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
    matched()
