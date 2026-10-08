"""Dot plot for m3c_anger: every reply's probe score (one probe, EMA reading), grouped by what the reply is, with the
lines at 0.1%, 1% and 5% false alarms set from Qwen's own clean no-anger replies.

    uv run python -m probe_evasion.missions.m3c_anger_dots --probe claude_gd:anger@L48 --attacks v2_phase_a
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.generation import read_jsonl
from probe_evasion.missions.m3c_anger import FILES, OUT, V
from probe_evasion.proposal_figures import BLUE, GREY, INK, ORANGE, RED

EPS, LOW = 1e-6, 1e-4
TICKS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]


def groups(probe: str, attacks: str, reading: str) -> dict[str, tuple[list[float], str]]:
    ev = read_jsonl(OUT / f"{V}_eval_scores.jsonl")
    by = lambda g: [e["scores"][probe][reading] for e in sorted((e for e in ev if e["group"] == g), key=lambda e: e["i"])]
    cal, held, pos = read_jsonl(FILES["calibration"]), read_jsonl(FILES["probe_rows_heldout"]), read_jsonl(FILES["positives"])
    cs, hs, ps = by("calibration"), by("heldout"), by("positives")
    out = {
        "no-anger replies\n(calibration)": ([s for r, s in zip(cal, cs) if not r["any_anger"] and r["task"]["kind"] != "calm-provoking"], GREY),
        "provoking situation,\nno anger shown\n(calibration)": ([s for r, s in zip(cal, cs) if not r["any_anger"] and r["task"]["kind"] == "calm-provoking"], BLUE),
        "no-anger task but\nshows anger\n(removed)": ([s for r, s in zip(cal, cs) if r["any_anger"]], RED),
        "plain angry\n(search train+val)": ([s for r, s in zip(pos, ps) if r["present"]], INK),
        "held-out styles,\nangry": ([s for r, s in zip(held, hs) if r["label"]], INK),
    }
    if attacks:
        rows = [r for r in read_jsonl(OUT / f"{attacks}_replies.jsonl") if r["arm"] != "plain"]
        out["attacks,\nvalid"] = ([r["scores"][probe][reading] for r in rows if r["present"]], ORANGE)
        out["attacks,\nnot valid"] = ([r["scores"][probe][reading] for r in rows if not r["present"]], "#f2c7a5")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="claude_gd:anger@L48")
    ap.add_argument("--attacks", default="")
    ap.add_argument("--reading", default="ema")
    a = ap.parse_args()
    lines = json.loads((OUT / f"{V}_probe_eval.json").read_text())["probes"][a.probe][a.reading]["lines"]
    data = groups(a.probe, a.attacks, a.reading)
    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 6.4), dpi=150)
    rng = np.random.default_rng(0)
    for i, (label, (xs, c)) in enumerate(data.items()):
        ys = np.clip(xs, EPS, 1 - EPS)
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(ys)), ys, s=8, color=c, alpha=0.7, linewidths=0)
        ax.text(i, LOW * 1.2, f"n={len(xs)}\nbelow 1% line:\n{sum(x <= lines['0.01'] for x in xs)}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_yscale("logit")
    ax.set_ylim(LOW, 1 - EPS)
    ax.set_yticks(TICKS, [f"{t:g}" for t in TICKS])
    ax.minorticks_off()
    for fa, ls in (("0.001", ":"), ("0.01", "-"), ("0.05", "--")):
        ax.axhline(lines[fa], color=INK, ls=ls, lw=1)
        ax.text(len(data) - 0.5, lines[fa], f"{float(fa):.1%} false alarms", va="bottom", ha="right", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(data)), list(data), fontsize=7.5)
    ax.set_ylabel(f"probe score, {a.reading.upper()} reading (stretched near 0 and 1)")
    ax.set_title(f"Anger probe {a.probe} on Qwen3.5-27B replies; lines from Qwen's own clean no-anger replies", fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = OUT / f"{V}_dots_{a.probe.replace(':', '_').replace('@', '_')}{'_' + a.attacks if a.attacks else ''}.png"
    fig.savefig(path)
    print(path)


if __name__ == "__main__":
    main()
