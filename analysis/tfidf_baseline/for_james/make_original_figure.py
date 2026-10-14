"""One-figure summary of the ORIGINAL TF-IDF baseline test for James: on the probe's held-out split, word-based
classifiers (even one that reads only the request) score as well as the probe, which is why m51's matched-request test
was built. Companion to make_figure.py (same colours and drawing helpers).

Reads only saved files (no models, no GPU):
  analysis/tfidf_baseline/scores/<task>_scores.jsonl.gz   per-row held-out scores of the probe and TF-IDF A, B, P
  analysis/tfidf_baseline/results/<task>.json             held-out AUROC as reported (checked against the recompute)
  data/concepts/onpolicy_qwen35_maths_heldout_clean.jsonl the maths held-out text (for the example pair)

    .venv/bin/python analysis/tfidf_baseline/for_james/make_original_figure.py

Writes original_test.png next to this script.
"""

import gzip
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figure import INK, MUTED, PROBE_COLOUR, REPO, TASK_LABEL, TASKS, TEXT_COLOUR, arrow, box  # noqa: E402

BASELINE = REPO / "analysis/tfidf_baseline"
MATHS_HELDOUT = REPO / "data/concepts/onpolicy_qwen35_maths_heldout_clean.jsonl"
DETECTORS = ("probe", "tfidf_A", "tfidf_B", "tfidf_P")
DETECTOR_LABEL = {"probe": "Probe (Qwen's activations)", "tfidf_A": "A: words of the reply",
                  "tfidf_B": "B: words of request + reply", "tfidf_P": "P: words of the request only"}
REPLY_ONLY_COLOUR, PROMPT_ONLY_COLOUR = "#d1d5db", INK
DETECTOR_COLOUR = {"probe": PROBE_COLOUR, "tfidf_A": REPLY_ONLY_COLOUR, "tfidf_B": TEXT_COLOUR, "tfidf_P": PROMPT_ONLY_COLOUR}
# A real held-out pair (row indices into the maths held-out file): a computed reply and a "given" look-alike
EXAMPLE_POSITIVE, EXAMPLE_NEGATIVE = 346, 3
STYLE_SUFFIX = " Answer as a short exchange"  # the shared answer-format instruction, cut from the displayed requests


def heldout_scores(task: str) -> list[dict]:
    with gzip.open(BASELINE / "scores" / f"{task}_scores.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f if '"set": "heldout"' in line]
    return sorted(rows, key=lambda r: r["i"])


def heldout_aurocs() -> dict[str, dict]:
    """Recomputed from the per-row scores; asserted equal to results/<task>.json."""
    out = {}
    for task in TASKS:
        rows = heldout_scores(task)
        labels = [r["label"] for r in rows]
        reported = json.loads((BASELINE / "results" / f"{task}.json").read_text())
        out[task] = {"n_positive": sum(labels), "n_negative": len(labels) - sum(labels)}
        for det in DETECTORS:
            auroc = roc_auc_score(labels, [r[det] for r in rows])
            # saved scores are rounded to 3 decimals, so allow ties created by rounding
            assert abs(auroc - reported[det]["heldout_auroc"]["all"]) < 2e-3, (task, det, auroc)
            out[task][det] = auroc
    return out


def example_pair() -> list[dict]:
    text = [json.loads(line) for line in MATHS_HELDOUT.read_text().splitlines() if line.strip()]
    scores = heldout_scores("maths")
    assert len(text) == len(scores)
    pair = []
    for i, want_label in ((EXAMPLE_POSITIVE, 1), (EXAMPLE_NEGATIVE, 0)):
        row, score = text[i], scores[i]
        assert row["label"] == score["label"] == want_label and row["kind"] == score["kind"] and row["style"] == "dialogue"
        request = row["prompt"].split(STYLE_SUFFIX)[0]
        reply = row["response"].strip().split("\n")[-1].strip('"')  # the answer line of the two-line exchange
        pair.append({"request": request, "reply": reply, "kind": row["kind"]} | {d: score[d] for d in DETECTORS})
    return pair


# ---------------------------------------------------------------- drawing

def draw_schematic(ax, pair: list[dict]) -> None:
    ax.set_xlim(-0.02, 1.02), ax.set_ylim(0, 1), ax.axis("off")
    ax.text(0.0, 1.0, "The original test", fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.0, 0.935, "Train word-based classifiers on exactly the probe's\ntraining data, then score everyone on the "
            "held-out set.", fontsize=9, color=INK, va="top")
    box(ax, 0.17, 0.77, 0.66, 0.075, "Training data: Qwen's replies, labelled\n\"worked it out\" (+) or not (-)", size=8.5)
    det_x = (0.0, 0.26, 0.52, 0.78)
    for x, det in zip(det_x, DETECTORS):
        bold = det in ("probe", "tfidf_P")
        face = "#fff7ed" if det == "probe" else "#f3f4f6"
        edge = DETECTOR_COLOUR[det] if det != "tfidf_A" else MUTED
        colour = DETECTOR_COLOUR[det] if det != "tfidf_A" else TEXT_COLOUR
        label = {"probe": "Probe\n(Qwen's\nactivations)", "tfidf_A": "A\nreply\nwords", "tfidf_B": "B\nrequest +\nreply words",
                 "tfidf_P": "P\nrequest\nwords only"}[det]
        box(ax, x, 0.6, 0.22, 0.11, label, face=face, edge=edge, size=8, weight="bold" if bold else "normal",
            colour=colour)
        arrow(ax, (0.5, 0.77), (x + 0.11, 0.71))
    ax.text(0.5, 0.575, "scored on held-out data (chart, right)", ha="center", va="top", fontsize=8.5, color=MUTED,
            style="italic")

    ax.text(0.0, 0.50, "A real held-out pair (maths): the request wording\ndiffers, so P can label it without the reply",
            fontsize=9, weight="bold", color=INK, va="top")
    for y, row, tag in ((0.20, pair[0], "+ worked it out"), (0.0, pair[1], "- answer given (look-alike)")):
        request = "\n".join(textwrap.wrap(f"Request: \"{row['request']}\"", 52))
        box(ax, 0.0, y + 0.015, 1.0, 0.155,
            f"{request}\nReply: \"{row['reply']}\"\n{tag}   |   P (request only) score: {row['tfidf_P']:+.1f}",
            face="white", edge=INK if row is pair[0] else MUTED, size=8.2)


def draw_results(ax, aurocs: dict) -> None:
    tasks = list(TASKS)
    y = np.arange(len(tasks))[::-1].astype(float)
    bar_h = 0.19
    offsets = {det: (1.5 - k) * bar_h for k, det in enumerate(DETECTORS)}
    chance = 0.5
    for yi, task in zip(y, tasks):
        for det in DETECTORS:
            value = aurocs[task][det]
            ax.barh(yi + offsets[det], value - chance, left=chance, height=bar_h * 0.9, color=DETECTOR_COLOUR[det])
            ax.text(value + 0.005, yi + offsets[det], f"{value:.3f}", va="center", fontsize=7.5,
                    color=TEXT_COLOUR if det == "tfidf_A" else DETECTOR_COLOUR[det])
    ax.set_yticks(y, [TASK_LABEL[t] for t in tasks], fontsize=10)
    ax.set_xlim(chance, 1.06)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
    ax.set_xlabel("Held-out AUROC (0.5 = chance, 1.0 = perfect)", fontsize=9, color=INK)
    ax.set_title("Held-out data: reading the request's words is enough", fontsize=12, weight="bold", color=INK,
                 loc="left", pad=46)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.legend(handles=[Patch(color=DETECTOR_COLOUR[d], label=DETECTOR_LABEL[d]) for d in DETECTORS],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, handlelength=1.2,
              columnspacing=1.2)


def draw_callout(fig) -> None:
    fig.text(0.40, 0.03, "Even P, which never sees Qwen's reply, scores 1.000 on every task. So this test can't\n"
             "tell \"reads Qwen's computation\" from \"reads the request's wording\"  →  we built the matched-request test.",
             fontsize=9.5, color=INK, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f3f4f6", ec=INK, lw=1))


def main() -> None:
    aurocs = heldout_aurocs()
    pair = example_pair()
    for task, res in aurocs.items():
        print(task, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    for row in pair:
        print(row)
    fig = plt.figure(figsize=(13, 6.2), facecolor="white")
    draw_schematic(fig.add_axes((0.02, 0.03, 0.31, 0.88)), pair)
    draw_results(fig.add_axes((0.46, 0.24, 0.51, 0.56)), aurocs)
    draw_callout(fig)
    fig.savefig(HERE / "original_test.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
