"""Dot plot for m1b_offpolicy_chem: every reply's EMA score at layer 48 under each probe, with the 0.1%, 1% and 5%
false-alarm lines from Qwen's clean no-chemistry replies.

    uv run python -m probe_evasion.missions.m1b_offpolicy_chem_dots
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.chemistry_probes import ATTACK_RUNS
from probe_evasion.missions.m1b_offpolicy_chem import OUT, thresholds
from probe_evasion.score_dots import EPS, TICKS

PROBES = {"m1b_offpolicy_chem_gd": "Claude-written replies (gd)", "m1b_offpolicy_chem_lr": "Claude-written replies (logreg)",
          "qwen35_chemistry_onpolicy": "Qwen's own replies (gd)", "qwen35_chemistry_onpolicy_lr": "Qwen's own replies (logreg)",
          "qwen35_chemistry_layers": "Neural Chameleons data (gd)"}
LOW = 1e-4


def groups(probe: str, layer: int, handwritten: str) -> dict[str, tuple[list[float], list[float]]]:
    """{group: (valid or reference scores, invalid scores drawn hollow)}"""
    s = json.loads((OUT / f"scores_{probe}.json").read_text())[f"L{layer}"]
    saved = ATTACK_RUNS[layer].name
    out = {"Qwen no-chem\n(calibration)": (s["calibration"], []), "Qwen biology/\nmaths, no chem": (s["science"], []),
           "Qwen held-out\nstyle chem": (s["heldout_pos"], []),
           "saved test:\nplain": (s[f"{saved}:no_instruction:valid"], s[f"{saved}:no_instruction:invalid"]),
           "saved test:\nanalogy attack": (s[f"{saved}:optimized:valid"], s[f"{saved}:optimized:invalid"])}
    hw = OUT / f"handwritten_{handwritten}"
    if hw.exists() and json.loads((hw / "summary.json").read_text())["layer"] == layer:
        rows = {f.stem: [json.loads(l) for l in f.read_text().splitlines()] for f in sorted(hw.glob("*.jsonl"))}
        order = ["plain"] + [n for n in rows if n != "plain"]
        for name in order:
            if probe in rows[name][0].get("ema_by_probe", {}):
                out[f"val: {name.replace('_', ' ')}"] = ([r["ema_by_probe"][probe] for r in rows[name] if r["present"]],
                                                          [r["ema_by_probe"][probe] for r in rows[name] if not r["present"]])
    return out


def main(layer: int = 48, handwritten: str = "val") -> Path:
    probes = [p for p in PROBES if (OUT / f"scores_{p}.json").exists()]
    fig, axes = plt.subplots(len(probes), 1, figsize=(15, 4.2 * len(probes)), dpi=110)
    rng = np.random.default_rng(0)
    for ax, probe in zip(np.atleast_1d(axes), probes):
        data = groups(probe, layer, handwritten)
        for i, (label, (xs, bad)) in enumerate(data.items()):
            colour = "#888888" if i < 2 else "#1f6fb4" if i == 2 or "plain" in label else "#c2410c"
            ax.scatter(i + rng.uniform(-0.28, 0.28, len(xs)), np.clip(xs, LOW, 1 - EPS), s=8, alpha=0.6, linewidths=0, color=colour)
            ax.scatter(i + rng.uniform(-0.28, 0.28, len(bad)), np.clip(bad, LOW, 1 - EPS), s=10, facecolors="none",
                       edgecolors=colour, linewidths=0.6, alpha=0.6)
        for (fa, t), ls in zip(sorted(thresholds(probe, layer).items()), (":", "-", "--")):
            ax.axhline(np.clip(t, LOW, 1 - EPS), color="black", lw=1, ls=ls)
            ax.text(len(data) - 0.5, np.clip(t, LOW, 1 - EPS), f"{fa:.1%} line", fontsize=7, ha="right", va="bottom")
        ax.set_yscale("logit")
        ax.set_ylim(LOW, 1 - EPS)
        ax.set_yticks(TICKS, [f"{t:g}" for t in TICKS])
        ax.minorticks_off()
        ax.set_xticks(range(len(data)), [k.replace("val: ", "val:\n") for k in data], fontsize=7)
        ax.set_title(f"probe trained on {PROBES[probe]}, L{layer}", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in np.atleast_1d(axes):
        ax.set_ylabel("EMA score")
    fig.suptitle("Every reply's EMA probe score. Filled: valid (chemistry present) or reference rows; hollow: attack/plain "
                 "replies that failed the chemistry check. Lines from 1118 clean Qwen no-chemistry replies.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = OUT / f"dots_L{layer}.png"
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main(48), main(16))
