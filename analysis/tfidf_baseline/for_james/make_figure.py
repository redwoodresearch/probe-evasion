"""One-figure summary of the m51 matched-request test for James: probe vs word-based (TF-IDF) classifiers on
identical reply text.

Reads only saved files (no models, no GPU):
  missions/m51_matched_request/part2/detector_scores.jsonl  every detector's score per reply (parts 1 and 2)
  missions/m51_matched_request/{,part2/}<task>_scored.jsonl  the reply text (to find identical-reply pairs)
  missions/m51_matched_request/part2/analysis.json           per-phrasing numbers, used to check this script

    .venv/bin/python analysis/tfidf_baseline/for_james/make_figure.py

Writes matched_test.png and numbers.json next to this script.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
M51 = REPO / "missions/m51_matched_request"
TASKS = ("maths", "ordering", "false_belief", "contradiction", "acrostic")
TASK_LABEL = {"maths": "Maths", "ordering": "Ordering", "false_belief": "Belief tracking",
              "contradiction": "Spotting a\ncontradiction", "acrostic": "Acrostic"}
# Part 1 (maths, ordering compute / given) replies live in the mission root; everything else in part2/.
PART1_VERSIONS = {"compute", "given_clause", "given_sentence"}
TEXT_CLASSIFIERS = ("tfidf_B", "tfidf_P")  # A (reply only) scores identical text identically by construction
ACROSTIC_PROBE = "m48ac_v2@L48@s0"
# A real identical-reply pair (maths, clause phrasing) for the schematic
EXAMPLE_PAIR = {"wrong": ("maths", "M005", "wrong_clause", 10), "given": ("maths", "M005", "given_clause", 10)}
PROBE_COLOUR, TEXT_COLOUR, INK, MUTED = "#c2410c", "#6b7280", "#1f2937", "#9ca3af"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_rows() -> dict[tuple, dict]:
    """(task, id, version, seed) -> detector scores plus reply text."""
    replies = {}
    for task in TASKS:
        for path in (M51 / f"{task}_scored.jsonl", M51 / "part2" / f"{task}_scored.jsonl"):
            if not path.exists():
                continue
            in_part1_file = path.parent == M51
            for r in read_jsonl(path):
                if (r["version"] in PART1_VERSIONS) == in_part1_file or task not in ("maths", "ordering"):
                    replies[(task, r["id"], r["version"], r["seed"])] = r["reply"]
    rows = {}
    for r in read_jsonl(M51 / "part2/detector_scores.jsonl"):
        key = (r["task"], r["id"], r["version"], r["seed"])
        rows[key] = r | {"reply": replies[key]}
    assert len(rows) == len(replies), (len(rows), len(replies))
    return rows


def identical_pairs(rows: dict, task: str, higher_version: str, lower_version: str) -> list[tuple[dict, dict]]:
    """Same item and seed, both replies correct, and word-for-word the same reply text."""
    pairs = []
    for (t, item, version, seed), hi in rows.items():
        if t != task or version != higher_version:
            continue
        lo = rows.get((task, item, lower_version, seed))
        if lo and hi["correct"] and lo["correct"] and hi["reply"] == lo["reply"]:
            pairs.append((hi, lo))
    return pairs


def gap_stats(pairs: list[tuple[dict, dict]], detector: str) -> dict:
    gaps = np.array([hi[detector] - lo[detector] for hi, lo in pairs])
    return {"n_pairs": len(pairs), "median_gap": float(np.median(gaps)), "share_higher": float(np.mean(gaps > 0)),
            "max_abs_gap": float(np.abs(gaps).max())}


def versions(rows: dict, task: str, kind: str) -> list[str]:
    return sorted({v for (t, _, v, _) in rows if t == task and v.startswith(kind + "_")})


def compute_numbers(rows: dict) -> dict:
    """Per task, pooled over every phrasing: two identical-reply comparisons.
    worked_vs_given: 'work it out' request vs the same request with the right answer inserted.
    wrong_vs_given:  the same request with a wrong answer inserted vs with the right answer inserted (correct replies
                     only, so after the wrong answer Qwen had to ignore it and work the answer out)."""
    analysis = json.loads((M51 / "part2/analysis.json").read_text())
    out = {}
    for task in TASKS:
        det_names = ("probe", "tfidf_A") + TEXT_CLASSIFIERS
        worked = [p for g in versions(rows, task, "given") for p in identical_pairs(rows, task, "compute", g)]
        wrong = [p for w in versions(rows, task, "wrong")
                 for p in identical_pairs(rows, task, w, "given_" + w.split("_", 1)[1])]
        # check against the mission's per-phrasing numbers
        saved = analysis[task]["detectors"]["probe"]
        for g in versions(rows, task, "given"):
            n = len(identical_pairs(rows, task, "compute", g))
            assert n == saved["identical_reply"].get(g, {"n_pairs": 0})["n_pairs"], (task, g, n)
        for w in versions(rows, task, "wrong"):
            pairs = identical_pairs(rows, task, w, "given_" + w.split("_", 1)[1])
            ref = saved["identical_wrong_minus_given"][w]
            assert len(pairs) == ref["n_pairs"] and abs(gap_stats(pairs, "probe")["median_gap"] - ref["median"]) < 1e-9
        out[task] = {
            "worked_vs_given": {d: gap_stats(worked, d) for d in det_names} if worked else {"n_pairs": 0},
            "wrong_vs_given": {d: gap_stats(wrong, d) for d in det_names} if wrong else {"n_pairs": 0},
            "has_wrong_answer_version": bool(versions(rows, task, "wrong")),
        }
    acro = analysis["acrostic"]
    out["acrostic"]["non_identical_check"] = {
        "auroc_compute_vs_given_ref": {d: acro["detectors"][d]["auroc_compute_vs"]["given_ref"] for d in ("probe", "tfidf_B", "tfidf_P")},
        "given_ref_share_over_probe_line": acro["detectors"]["probe"]["over_line"]["given_ref"],
        "probe_line": acro["lines"]["probe"],
        "probe_median": {v: acro["detectors"]["probe"]["median"][v] for v in ("compute", "given_ref", "given_train")},
    }
    out["acrostic"]["given_words_used"] = acrostic_given_word_use()
    return out


def acrostic_given_word_use() -> dict:
    """How often a correct acrostic reply to a 'given' request actually starts its sentences with the four suggested
    words, and the probe's median for replies that do / don't. (Not in the mission's analysis; computed here.)"""
    line = json.loads((M51 / "part2/analysis.json").read_text())["acrostic"]["lines"]["probe"]
    out = {"probe_line": line}
    rows = read_jsonl(M51 / "part2/acrostic_scored.jsonl")
    for version in ("given_train", "given_ref"):
        used, not_used = [], []
        for r in rows:
            if r["version"] != version or not r["correct"]:
                continue
            suggested = [w.lower() for w in re.findall(r"'([^']+)'", r["prompt"].split("Do not mention")[1])[:4]]
            first_words = [s.split()[0].strip(".,!'\"").lower() for s in r["reply"].strip().split("\n") if s.strip()]
            (used if first_words[:4] == suggested else not_used).append(r["peaks"][ACROSTIC_PROBE])
        out[version] = {"n_correct": len(used) + len(not_used), "share_using_suggested_words": len(used) / (len(used) + len(not_used)),
                        "probe_median_using": float(np.median(used)), "probe_share_over_line_using": float(np.mean(np.array(used) > line)),
                        "probe_median_not_using": float(np.median(not_used))}
    return out


# ---------------------------------------------------------------- drawing

def box(ax, x, y, w, h, text, face="#f3f4f6", edge="#d1d5db", size=9, weight="normal", colour=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=face, ec=edge, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=colour, wrap=True)


def arrow(ax, start, end, colour=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=colour, lw=1.2))


def draw_schematic(ax, rows: dict) -> None:
    wrong, given = rows[EXAMPLE_PAIR["wrong"]], rows[EXAMPLE_PAIR["given"]]
    assert wrong["reply"] == given["reply"] and wrong["correct"] and given["correct"]
    ax.set_xlim(-0.02, 1.02), ax.set_ylim(0, 1), ax.axis("off")
    ax.text(0.0, 1.0, "The test", fontsize=12, weight="bold", color=INK, va="top")
    ax.text(0.0, 0.935, "One problem, two requests that differ only in the inserted\n"
            "answer. Keep the pairs where Qwen's reply is word-for-word\nidentical (a real maths pair shown).",
            fontsize=9, color=INK, va="top")
    box(ax, 0.0, 0.58, 0.48, 0.17, "\"59 jars cost $49 each,\n$2540 in total. What do they\ncost in total?\"\n"
        "wrong answer given", size=8.2)
    box(ax, 0.52, 0.58, 0.48, 0.17, "\"59 jars cost $49 each,\n$2891 in total. What do they\ncost in total?\"\n"
        "right answer given", size=8.2)
    box(ax, 0.03, 0.40, 0.94, 0.09, f"Same reply to both: \"{wrong['reply']}\"", face="white", edge=INK, size=9)
    arrow(ax, (0.24, 0.58), (0.40, 0.49)), arrow(ax, (0.76, 0.58), (0.60, 0.49))
    box(ax, 0.0, 0.10, 0.48, 0.20, f"Probe on Qwen's\nactivations\n{wrong['probe']:+.1f} vs {given['probe']:+.1f}",
        face="#fff7ed", edge=PROBE_COLOUR, size=9, weight="bold", colour=PROBE_COLOUR)
    box(ax, 0.52, 0.10, 0.48, 0.20, f"Word-based\nclassifier (TF-IDF)\n{wrong['tfidf_B']:+.1f} vs {given['tfidf_B']:+.1f}",
        face="#f3f4f6", edge=TEXT_COLOUR, size=9, weight="bold", colour=TEXT_COLOUR)
    arrow(ax, (0.40, 0.40), (0.24, 0.30)), arrow(ax, (0.60, 0.40), (0.76, 0.30))
    ax.text(0.0, 0.05, "After the wrong answer, Qwen had to ignore it and work the\n"
            "sum out. The probe scores that higher; the words can't show it.", fontsize=8.5, color=INK, va="top",
            style="italic")


def draw_results(ax, numbers: dict) -> None:
    tasks = list(TASKS)
    y = np.arange(len(tasks))[::-1].astype(float)
    bar_h = 0.34
    for yi, task in zip(y, tasks):
        res = numbers[task]["wrong_vs_given"]
        if "probe" not in res:
            check = numbers[task]["non_identical_check"]["auroc_compute_vs_given_ref"]
            ax.text(0.25, yi, "Not testable. Qwen's four-sentence replies were never identical across\n"
                    "requests, and a \"wrong answer\" makes no sense here. On the looser test,\n"
                    f"word classifiers did as well as the probe (AUROC {check['tfidf_B']:.2f} vs {check['probe']:.2f}).",
                    va="center", fontsize=8.5, color=TEXT_COLOUR, style="italic")
            continue
        probe = res["probe"]
        text_gap = max((res[d]["median_gap"] for d in TEXT_CLASSIFIERS), key=abs)
        ax.barh(yi + bar_h / 2, probe["median_gap"], height=bar_h, color=PROBE_COLOUR)
        ax.barh(yi - bar_h / 2, text_gap, height=bar_h, color=TEXT_COLOUR)
        ax.text(probe["median_gap"] + 0.2, yi + bar_h / 2,
                f"+{probe['median_gap']:.1f}; higher in {probe['share_higher']:.0%} of {probe['n_pairs']} pairs",
                va="center", fontsize=8.5, color=PROBE_COLOUR)
        ax.text(0.3, yi - bar_h / 2, f"{text_gap:+.1f}", va="center", fontsize=8.5, color=TEXT_COLOUR)
    ax.set_yticks(y, [TASK_LABEL[t] for t in tasks], fontsize=10)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-0.4, 14.5)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel("Median score gap on identical replies: after wrong answer minus after right answer (log-odds)",
                  fontsize=9, color=INK)
    ax.set_title("Same reply text, yet the probe sees that Qwen worked it out", fontsize=12, weight="bold",
                 color=INK, loc="left", pad=28)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.legend(handles=[Patch(color=PROBE_COLOUR, label="Probe"),
                       Patch(color=TEXT_COLOUR, label="Word-based classifiers (TF-IDF; the larger of two shown)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)


def main() -> None:
    rows = load_rows()
    numbers = compute_numbers(rows)
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1))
    for task, res in numbers.items():
        print(task, json.dumps(res)[:600])
    fig = plt.figure(figsize=(13, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.9], wspace=0.28, left=0.02, right=0.98, top=0.86, bottom=0.12)
    draw_schematic(fig.add_subplot(grid[0]), rows)
    draw_results(fig.add_subplot(grid[1]), numbers)
    fig.savefig(HERE / "matched_test.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
