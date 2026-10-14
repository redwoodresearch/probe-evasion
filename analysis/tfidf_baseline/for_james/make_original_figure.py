"""One-figure summary of the ORIGINAL TF-IDF baseline test for James, probe vs word classifier B (request + reply):
(1) on the probe's held-out split B scores as well as the probe, which is why m51's matched-request test was built;
(2) side result: how far the m45 attacks (tuned against the probe) carry over to B in maths and ordering.
Companion to make_figure.py (same colours and drawing helpers).

Reads only saved files (no models, no GPU):
  analysis/tfidf_baseline/scores/<task>_scores.jsonl.gz   per-row held-out scores of the probe and TF-IDF B (and P)
  analysis/tfidf_baseline/results/<task>.json             held-out AUROC as reported (checked against the recompute);
                                                          key <detector>.attacks for the attack numbers
  data/concepts/onpolicy_qwen35_maths_heldout_clean.jsonl the maths held-out text (for the example pair)

    .venv/bin/python analysis/tfidf_baseline/for_james/make_original_figure.py

Writes original_test.png and original_numbers.json next to this script.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_figure import INK, MUTED, PROBE_COLOUR, REPO, TASK_LABEL, TASKS, TEXT_COLOUR, arrow, box  # noqa: E402

BASELINE = REPO / "analysis/tfidf_baseline"
MATHS_HELDOUT = REPO / "data/concepts/onpolicy_qwen35_maths_heldout_clean.jsonl"
PLOTTED = ("probe", "tfidf_B")
REQUEST_ONLY = "tfidf_P"  # not plotted; only its held-out AUROC is quoted in the callout
DETECTOR_LABEL = {"probe": "Probe (reads Qwen's activations)", "tfidf_B": "B: word classifier (reads request + reply)"}
SHORT_LABEL = {"probe": "Probe", "tfidf_B": "B"}
DETECTOR_COLOUR = {"probe": PROBE_COLOUR, "tfidf_B": TEXT_COLOUR}
ATTACK_TASKS = ("maths", "ordering")
ATTACK_TITLE = {"maths": "Maths: the attacks carried over to B", "ordering": "Ordering: they mostly did not"}
# A real held-out pair (row indices into the maths held-out file): a computed reply and a "given" look-alike
EXAMPLE_POSITIVE, EXAMPLE_NEGATIVE = 346, 3
STYLE_SUFFIX = " Answer as a short exchange"  # the shared answer-format instruction, cut from the displayed requests
TIE_TOLERANCE = 1e-3  # RESULT.md: B settings within 0.001 held-out AUROC of the best count as tied


def heldout_scores(task: str) -> list[dict]:
    with gzip.open(BASELINE / "scores" / f"{task}_scores.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f if '"set": "heldout"' in line]
    return sorted(rows, key=lambda r: r["i"])


def results(task: str) -> dict:
    return json.loads((BASELINE / "results" / f"{task}.json").read_text())


def heldout_aurocs() -> dict[str, dict]:
    """Recomputed from the per-row scores; asserted equal to results/<task>.json."""
    out = {}
    for task in TASKS:
        rows = heldout_scores(task)
        labels = [r["label"] for r in rows]
        reported = results(task)
        out[task] = {"n_positive": sum(labels), "n_negative": len(labels) - sum(labels)}
        for det in PLOTTED + (REQUEST_ONLY,):
            auroc = roc_auc_score(labels, [r[det] for r in rows])
            # saved scores are rounded to 3 decimals, so allow ties created by rounding
            assert abs(auroc - reported[det]["heldout_auroc"]["all"]) < 2e-3, (task, det, auroc)
            out[task][det] = auroc
    return out


def attack_numbers() -> dict[str, dict]:
    """Per writer attack strength S (mean over its 3 chains) for the probe and B, the fitted slope, and the share of
    valid ON (attack) replies under each detector's 1% line. All from results/<task>.json, key <detector>.attacks,
    with the per-writer means and line counts checked against the per-arm rows."""
    out = {}
    for task in ATTACK_TASKS:
        res = results(task)
        out[task] = {}
        for det in PLOTTED:
            a = res[det]["attacks"]
            writers = sorted(a["writer_means"], key=lambda w: a["writer_eci"][w])
            for w in writers:
                chains = [arm["S_offref"] for arm in a["arms"] if arm["writer"] == w]
                assert len(chains) == 3 and abs(np.mean(chains) - a["writer_means"][w]) < 1e-9, (task, det, w)
            assert sum(arm["on_under_line"] for arm in a["arms"]) == a["on_under_line"]
            eci = np.array([a["writer_eci"][w] for w in writers])
            strength = np.array([a["writer_means"][w] for w in writers])
            slope, intercept = np.polyfit(eci, strength, 1)
            assert abs(slope - a["slope_per_10_eci"] / 10) < 1e-12, (task, det)
            out[task][det] = {"writers": writers, "eci": eci.tolist(), "strength": strength.tolist(),
                              "slope_per_10_eci": 10 * slope, "intercept": intercept,
                              "slope_ci95": a["slope_ci95_two_stage"], "on_under_line": a["on_under_line"],
                              "n_on_valid": a["n_on_valid"]}
        assert out[task]["probe"]["writers"] == out[task]["tfidf_B"]["writers"]
        # how much B's carry-over depends on its settings: every B setting tied with the chosen one on held-out AUROC
        b_settings = [s for s in res["sensitivity"] if s["variant"] == "B"]
        best = max(s["heldout_auroc"] for s in b_settings)
        tied = [s for s in b_settings if s["heldout_auroc"] >= best - TIE_TOLERANCE]
        slopes = [s["slope_per_10_eci"] for s in tied if np.isfinite(s["slope_per_10_eci"])]
        crossings = [s["switch_crossings"] for s in tied]
        out[task]["tfidf_B_tied_settings"] = {"n_tied": len(tied), "slope_range": [min(slopes), max(slopes)],
                                              "switch_crossings_range": [min(crossings), max(crossings)]}
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
        pair.append({"request": request, "reply": reply, "kind": row["kind"]}
                    | {d: score[d] for d in PLOTTED + (REQUEST_ONLY,)})
    return pair


# ---------------------------------------------------------------- drawing

def draw_schematic(ax, pair: list[dict]) -> None:
    ax.set_xlim(-0.02, 1.02), ax.set_ylim(0, 1), ax.axis("off")
    ax.text(0.0, 1.0, "The original test", fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.0, 0.935, "Train a word-based classifier on exactly the probe's\ntraining data, then score both on the "
            "held-out set.", fontsize=9, color=INK, va="top")
    box(ax, 0.12, 0.76, 0.76, 0.08, "Training data: Qwen's replies, labelled\n\"worked it out\" (+) or not (-)", size=8.5)
    for x, det in ((0.0, "probe"), (0.53, "tfidf_B")):
        label = {"probe": "Probe\n(Qwen's activations)", "tfidf_B": "B: word classifier\n(request + reply words)"}[det]
        face = "#fff7ed" if det == "probe" else "#f3f4f6"
        box(ax, x, 0.60, 0.47, 0.09, label, face=face, edge=DETECTOR_COLOUR[det], size=8.5, weight="bold",
            colour=DETECTOR_COLOUR[det])
        arrow(ax, (0.5, 0.76), (x + 0.235, 0.69))
    ax.text(0.5, 0.575, "scored on held-out data (chart, right)", ha="center", va="top", fontsize=8.5, color=MUTED,
            style="italic")

    ax.text(0.0, 0.46, "A real held-out pair (maths). The two kinds of\nrequest are worded differently (\"sold\" vs \"shows\")",
            fontsize=9, weight="bold", color=INK, va="top")
    for y, row, tag in ((0.20, pair[0], "+ worked it out"), (0.01, pair[1], "- answer given (look-alike)")):
        request = "\n".join(textwrap.wrap(f"Request: \"{row['request']}\"", 52))
        box(ax, 0.0, y + 0.015, 1.0, 0.155,
            f"{request}\nReply: \"{row['reply']}\"\n{tag}   |   probe {row['probe']:+.1f},  B {row['tfidf_B']:+.1f}",
            face="white", edge=INK if row is pair[0] else MUTED, size=8.2)


def draw_heldout(ax, aurocs: dict) -> None:
    tasks = list(TASKS)
    y = np.arange(len(tasks))[::-1].astype(float)
    bar_h = 0.34
    offsets = {"probe": bar_h / 2, "tfidf_B": -bar_h / 2}
    chance = 0.5
    for yi, task in zip(y, tasks):
        for det in PLOTTED:
            value = aurocs[task][det]
            ax.barh(yi + offsets[det], value - chance, left=chance, height=bar_h * 0.9, color=DETECTOR_COLOUR[det])
            ax.text(value + 0.005, yi + offsets[det], f"{value:.3f}", va="center", fontsize=8, color=DETECTOR_COLOUR[det])
    ax.set_yticks(y, [TASK_LABEL[t] for t in tasks], fontsize=10)
    ax.set_xlim(chance, 1.06)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(y.min() - 0.55, y.max() + 0.55)
    ax.set_xlabel("Held-out AUROC (0.5 = chance, 1.0 = perfect)", fontsize=9, color=INK)
    ax.set_title("Held-out data: the word classifier does as well as the probe", fontsize=12, weight="bold", color=INK,
                 loc="left", pad=26)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.legend(handles=[Patch(color=DETECTOR_COLOUR[d], label=DETECTOR_LABEL[d]) for d in PLOTTED],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5, handlelength=1.2,
              columnspacing=1.5)


def draw_callout(fig, x: float, y: float, aurocs: dict) -> None:
    request_only = [aurocs[t][REQUEST_ONLY] for t in TASKS]
    assert min(request_only) > 0.9995, request_only
    fig.text(x, y, "Held-out accuracy can't separate the probe from a word classifier (both near-perfect), so it can't\n"
             "show that the probe reads Qwen's computation rather than the request  →  we built the matched-request test.\n"
             "(Even a classifier given only the request, never the reply, scores 1.000 on all five tasks.)",
             fontsize=9.3, color=INK, va="bottom", ha="left", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f3f4f6", ec=INK, lw=1))


def draw_attack(ax, task: str, numbers: dict) -> None:
    handles = []
    for det in PLOTTED:
        res = numbers[task][det]
        eci, strength = np.array(res["eci"]), np.array(res["strength"])
        marker = "o" if det == "probe" else "s"
        ax.plot(eci, strength, marker=marker, ms=6.5, ls="", color=DETECTOR_COLOUR[det], mec="white", mew=1)
        xs = np.array([eci.min(), eci.max()])
        ax.plot(xs, res["slope_per_10_eci"] / 10 * xs + res["intercept"], color=DETECTOR_COLOUR[det], lw=2)
        lo, hi = res["slope_ci95"]
        handles.append(Line2D([], [], color=DETECTOR_COLOUR[det], marker=marker, lw=2, mec="white",
                              label=f"{SHORT_LABEL[det]}: {res['slope_per_10_eci']:+.2f} per 10 ECI [{lo:+.2f}, {hi:+.2f}]"))
    ax.axhline(0, color="#d1d5db", lw=1, zorder=0)
    ax.set_xlabel("Writer capability (ECI)", fontsize=9, color=INK)
    ax.set_ylabel("Attack strength (gap closed)\n0 = no effect, 1 = typical attack\nreply sits on the threshold",
                  fontsize=8.5, color=INK)
    probe, word = numbers[task]["probe"], numbers[task]["tfidf_B"]
    ax.set_title(ATTACK_TITLE[task], fontsize=11, weight="bold", color=INK, loc="left", pad=22)
    ax.text(0, 1.02, f"Valid attack replies under the 1% threshold:  B {word['on_under_line'] / word['n_on_valid']:.1%} "
            f"({word['on_under_line']:,} of {word['n_on_valid']:,}),  probe {probe['on_under_line']} of "
            f"{probe['n_on_valid']:,}", transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom")
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, title="Fitted slope [95% interval]",
              title_fontsize=8.5, alignment="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)
    top = max(max(numbers[task][d]["strength"]) for d in PLOTTED)
    ax.set_ylim(-0.08, top * 1.45)


def draw_attack_header_and_caveat(fig, numbers: dict) -> None:
    fig.text(0.02, 0.425, "Side question: did attacks written to fool the probe also fool B?", fontsize=12,
             weight="bold", color=INK, va="bottom")
    fig.text(0.02, 0.405, "14 writer models each wrote attack instructions (3 chains each) for Qwen, tuned against the "
             "probe's scores (m45). Each dot is one writer.", fontsize=9, color=INK, va="top")
    maths, ordering = numbers["maths"]["tfidf_B_tied_settings"], numbers["ordering"]["tfidf_B_tied_settings"]
    fig.text(0.02, 0.012,
             "Caveat: the attacks were tuned against the probe, not against B, so this shows how far they carry over, not "
             "how hard B would be to evade if attacked directly.\nB's numbers also depend on its settings: across the B "
             "settings that tie on held-out AUROC, B's slope ranges "
             f"{maths['slope_range'][0]:+.2f} to {maths['slope_range'][1]:+.2f} in maths and "
             f"{ordering['slope_range'][0]:+.2f} to {ordering['slope_range'][1]:+.2f} in ordering (chosen setting shown).",
             fontsize=8, color=TEXT_COLOUR, va="bottom", style="italic")


def main() -> None:
    aurocs = heldout_aurocs()
    pair = example_pair()
    attacks = attack_numbers()
    (HERE / "original_numbers.json").write_text(json.dumps(
        {"heldout_auroc": aurocs, "example_pair": pair, "attacks": attacks}, indent=1))
    for task, res in aurocs.items():
        print(task, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
    for row in pair:
        print(row)
    for task, res in attacks.items():
        for det in PLOTTED:
            r = res[det]
            print(task, det, f"slope {r['slope_per_10_eci']:+.3f}", f"under line {r['on_under_line']}/{r['n_on_valid']}")
        print(task, "B tied settings", res["tfidf_B_tied_settings"])

    fig = plt.figure(figsize=(14, 11), facecolor="white")
    draw_schematic(fig.add_axes((0.02, 0.50, 0.31, 0.48)), pair)
    draw_heldout(fig.add_axes((0.47, 0.62, 0.50, 0.30)), aurocs)
    draw_callout(fig, 0.395, 0.505, aurocs)
    fig.add_artist(Line2D([0.02, 0.98], [0.465, 0.465], color="#d1d5db", lw=1))
    draw_attack_header_and_caveat(fig, attacks)
    for k, task in enumerate(ATTACK_TASKS):
        draw_attack(fig.add_axes((0.10 + 0.49 * k, 0.09, 0.38, 0.235)), task, attacks)
    fig.savefig(HERE / "original_test.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
