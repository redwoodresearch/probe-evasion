"""Every reply's probe score as a dot, grouped by what the reply is, with the flag line before and after cleaning the
no-concept calibration replies. The view that would have caught the calibration bug: no-concept replies that really
show the concept sit up among the concept answers and drag the line with them.

    uv run python -m probe_evasion.score_dots   # writes share/score_dots.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from probe_evasion import gepa
from probe_evasion.proposal_figures import BLUE, GREY, INK, ORANGE

G = Path("results/gepa")
RED = "#c2410c"
PANELS = [  # run, title
    ("qwen35_maths_window_L48_20261007-173549", "Qwen maths, layer 48, original probe\n(attack found against this probe)"),
    ("qwen35_maths_ema_onpolicyclean_L48_20261008-v2", "Qwen maths, layer 48, clean-label probe\n(fresh search against it)"),
    ("qwen35_chemistry_window_L48_20261007-173800", "Qwen chemistry, layer 48\n(attack found against this probe)"),
]
READING = "ema"
EPS, LOW = 1e-6, 2e-3
TICKS = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]


def fmt(x: float) -> str:
    """Enough decimals to tell lines near 1 apart (0.9996, not 1.000)."""
    digits = max(3, int(np.ceil(-np.log10(1 - x))) + 1) if x < 1 else 6
    return f"{x:.{digits}f}"


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def groups(run: str) -> tuple[dict[str, list[float]], float, float]:
    d = G / run
    concept = json.loads((d / "run_config.json").read_text())["concept"]
    cal = jl(d / "calibration_replies.jsonl")
    arms = {a: jl(d / f"test_{a}.jsonl") for a in ("no_instruction", "optimized")}
    for r in cal + arms["no_instruction"] + arms["optimized"]:
        r.update(gepa.readings(r["token_scores"], r["token_logits"]))
    if concept == "maths":
        gepa.recheck_maths(arms["no_instruction"] + arms["optimized"])
    neg = [r for r in cal if r["task"]["split"] == "negative" and r["finished"]]
    kept = {id(r) for r in gepa.clean_negatives(concept, neg)}
    score = lambda rs: [r[READING] for r in rs]
    out = {
        "no-concept\nreplies": score([r for r in neg if id(r) in kept]),
        "no-concept task,\nbut shows it\n(removed)": score([r for r in neg if id(r) not in kept]),
        "normal\nanswers": score([r for r in arms["no_instruction"] if r["present"]]),
        "optimized,\nvalid": score([r for r in arms["optimized"] if r["present"]]),
        "optimized,\nnot valid": score([r for r in arms["optimized"] if not r["present"]]),
    }
    line = lambda xs: float(np.quantile(xs, 1 - gepa.FALSE_ALARM))
    return out, line(score(neg)), line(out["no-concept\nreplies"])


def panel(ax, run: str, title: str) -> None:
    data, old, new = groups(run)
    colours = [GREY, RED, BLUE, ORANGE, "#f2c7a5"]
    rng = np.random.default_rng(0)
    for i, ((label, xs), c) in enumerate(zip(data.items(), colours)):
        ys = np.clip(xs, EPS, 1 - EPS)
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(ys)), ys, s=9, color=c, alpha=0.75, linewidths=0)
        if label.startswith("optimized,\nvalid"):
            ax.text(i, LOW * 1.3, f"{sum(x <= new for x in xs)}/{len(xs)}\nbelow the\ncorrected line", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yscale("logit")
    ax.set_ylim(LOW, 1 - EPS)
    ax.set_yticks(TICKS, [f"{t:g}" for t in TICKS])
    ax.minorticks_off()
    ax.axhline(old, color=INK, ls="--", lw=1)
    ax.axhline(new, color=INK, lw=1.4)
    ax.text(len(data) - 0.45, old, f"old line {fmt(old)}", va="bottom", ha="right", fontsize=8, color=INK)
    ax.text(len(data) - 0.45, new, f"corrected line {fmt(new)}", va="top", ha="right", fontsize=8, color=INK)
    ax.set_xticks(range(len(data)), list(data), fontsize=8)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def main(out: Path = Path("experimental/share/score_dots.png")) -> Path:
    load_dotenv(".env")
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.6 * len(PANELS), 6.4), dpi=150)
    for ax, (run, title) in zip(axes, PANELS):
        panel(ax, run, title)
    axes[0].set_ylabel(f"probe score, {READING.upper()} reading (stretched near 0 and 1)")
    fig.suptitle("Every reply's probe score. A reply is flagged if it is above the line (set so 1% of no-concept replies are flagged).",
                 fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main())
