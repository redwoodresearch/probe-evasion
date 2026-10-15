"""Dot plot of the Gemma-2-9B maths probes at layers 6, 12 and 36 (peak-trained, seed 0; the probes behind
results/gepa/gemma_maths_peakL{6,12,36}_s0), one panel per layer, every score in log-odds of the running-peak reading
(gepa.ema_logit of the saved per-token logits) the probes were trained for.

Saved scores only, no scoring:
  - clean no-maths calibration replies and plain maths answers: each run's own calibration_replies.jsonl and
    test_no_instruction.jsonl, scored by that run's probe;
  - the run's own frozen searched prompt, instruction visible: its test_optimized.jsonl;
  - the same replies with the instruction removed: per-reply scores exist only for layer 6
    (experimental/analysis/fa_cost/gemma_scores.json, "attack_removed"). gepa.test saved only the count for layers 12 and 36, so
    those strips are left empty and the count from summary.json is written in their place.

    uv run python experimental/analysis/gemma_readout/dots_layers.py
Writes share/gemma_maths_dots.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion import gepa
from probe_evasion.generation import read_jsonl

REPO = Path(__file__).resolve().parents[3]
RUNS = {layer: REPO / f"results/gepa/gemma_maths_peakL{layer}_s0" for layer in (6, 12, 36)}
REMOVED_SCORES = REPO / "experimental/analysis/fa_cost/gemma_scores.json"  # layer-6 run's attack replies, instruction removed
OUT = REPO / "experimental/share/gemma_maths_dots.png"
FALSE_ALARM_RATES = (0.001, 0.01, 0.05)
LINE_STYLE = {0.001: ":", 0.01: "-", 0.05: "--"}
REMOVED_REPRODUCE_TOL = 0.05  # log-odds; the scorer rounds per-token logits to 3 decimals
# Counts verified earlier (summary.json and experimental/analysis/fa_cost/examples_gemma.py): (removed missed, valid, plain missed)
EXPECTED = {6: (12, 87, 1), 12: (0, 82, 0), 36: (0, 89, 0)}

GROUPS = ("calibration", "plain", "removed", "visible")
COLOUR = {"calibration": "#8c8c8c", "plain": "#1f77b4", "removed": "#d62728", "visible": "#ff9896"}
LABEL = {"calibration": "clean no-maths\ncalibration\nreplies", "plain": "plain maths\nanswers\n(valid)",
         "removed": "searched prompt,\ninstruction\nremoved", "visible": "searched prompt,\ninstruction\nvisible"}


def peak_logodds(row: dict) -> float:
    return gepa.ema_logit(np.array(row["token_logits"]))


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def layer_data(layer: int, removed_l6: list[dict]) -> dict:
    run = RUNS[layer]
    assert json.loads((run / "run_config.json").read_text())["probe"] == f"gemma_maths_layers_peak:maths@L{layer}@s0"
    cal_rows = [r for r in read_jsonl(run / "calibration_replies.jsonl") if r["task"]["split"] == "negative"]
    for r in cal_rows:  # gepa.calibrate's recomputation (the saved fields may be stale)
        r.update(gepa.readings(r["token_scores"], r["token_logits"]))
    clean = gepa.clean_maths_negatives(cal_rows)
    calibration_json = json.loads((run / "calibration.json").read_text())
    assert len(clean) == calibration_json["n_negative"]

    # lines as gepa sets them (quantile of the peak probability), then converted to log-odds
    cal_prob = np.array([r["ema"] for r in clean])
    lines = {q: logit(float(np.quantile(cal_prob, 1 - q))) for q in FALSE_ALARM_RATES}
    assert np.isclose(lines[gepa.FALSE_ALARM], logit(calibration_json["ema"]["threshold"]))

    plain = [r for r in read_jsonl(run / "test_no_instruction.jsonl") if r["present"]]
    optimized = read_jsonl(run / "test_optimized.jsonl")
    valid = [i for i, r in enumerate(optimized) if r["present"]]
    scores = {"calibration": np.array([peak_logodds(r) for r in clean]),
              "plain": np.array([peak_logodds(r) for r in plain]),
              "visible": np.array([peak_logodds(optimized[i]) for i in valid])}
    if layer == 6:
        assert len(removed_l6) == len(optimized)
        for saved, row in zip(removed_l6, optimized):
            assert saved["reply"] == row["reply"] and saved["prompt"] == row["task"]["task"]
        scores["removed"] = np.array([removed_l6[i]["L6"] for i in valid])

    arms = json.loads((run / "summary.json").read_text())["arms"]
    line = lines[gepa.FALSE_ALARM]
    counts = {"calibration": int(np.sum(scores["calibration"] > line)),
              "plain": int(np.sum(scores["plain"] <= line)),
              "visible": int(np.sum(scores["visible"] <= line)),
              "removed": arms["optimized, instruction removed"]["missed_ema"]}
    if "removed" in scores:
        assert counts["removed"] == int(np.sum(scores["removed"] <= line))
    assert len(scores["plain"]) == arms["no instruction"]["present"] and counts["plain"] == arms["no instruction"]["missed_ema"]
    assert len(valid) == arms["optimized"]["present"] and counts["visible"] == arms["optimized"]["missed_ema"]
    assert (counts["removed"], len(valid), counts["plain"]) == EXPECTED[layer]
    return {"scores": scores, "lines": lines, "counts": counts, "n_valid": len(valid)}


def strip_caption(group: str, d: dict) -> str:
    count = d["counts"][group]
    if group == "calibration":
        return f"n={len(d['scores'][group])}\n{count} above\n1% line"
    n = len(d["scores"][group]) if group in d["scores"] else d["n_valid"]
    return f"n={n}\n{count}/{n} under\n1% line"


def draw(data: dict[int, dict]) -> None:
    fig, axes = plt.subplots(1, len(data), figsize=(18, 7.6))
    rng = np.random.default_rng(0)
    for ax, (layer, d) in zip(axes, data.items()):
        for x, group in enumerate(GROUPS):
            if group in d["scores"]:
                ys = d["scores"][group]
                ax.scatter(x + rng.uniform(-0.28, 0.28, len(ys)), ys, s=9, alpha=0.6, color=COLOUR[group], linewidths=0)
            else:
                ax.text(x, 0.8, "per-reply scores\nnot saved\n(count from the\nsearch summary)",
                        transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=8.5, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#cccccc"))
        for q, y in d["lines"].items():
            ax.axhline(y, color="black", lw=1.1, ls=LINE_STYLE[q], label=f"{q:.1%} line ({y:+.2f})".replace(".0%", "%"))
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, title="share of clean no-maths\nreplies flagged",
                  title_fontsize=8.5)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{LABEL[g]}\n\n{strip_caption(g, d)}" for g in GROUPS], fontsize=8.5)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.set_title(f"layer {layer}", fontsize=12, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("probe score, log-odds (running-peak reading)")
    fig.suptitle("Gemma-2-9B maths probes: only the weakest layer (6) lets some answers through", fontsize=14, x=0.01,
                 ha="left", fontweight="bold")
    fig.text(0.01, 0.915, "Peak-trained probes, seed 0, each read with the running peak it was trained on. Lines: share of "
             "clean no-maths calibration replies flagged (0.1%, 1%, 5%). Searched\nprompt = that layer's own frozen "
             "prompt; valid answers only.", fontsize=9.5, color="#444444", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.905), w_pad=3)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(OUT)


def main() -> None:
    removed_l6 = json.loads(REMOVED_SCORES.read_text())["attack_removed"]
    data = {layer: layer_data(layer, removed_l6) for layer in RUNS}
    # gemma_scores.json was scored in a separate job: check it reproduces the search's own saved layer-6 readings
    saved = json.loads(REMOVED_SCORES.read_text())["attack_with_instruction"]
    optimized = read_jsonl(RUNS[6] / "test_optimized.jsonl")
    worst = max(abs(s["L6"] - peak_logodds(r)) for s, r in zip(saved, optimized))
    assert worst < REMOVED_REPRODUCE_TOL, worst
    for layer, d in data.items():
        print(f"layer {layer}: lines " + ", ".join(f"{q:.1%} {y:+.2f}" for q, y in d["lines"].items())
              + " | " + " | ".join(f"{g}: {strip_caption(g, d).replace(chr(10), ' ')}" for g in GROUPS))
    draw(data)


if __name__ == "__main__":
    main()
