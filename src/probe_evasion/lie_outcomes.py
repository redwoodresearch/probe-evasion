"""What Qwen actually did when told to lie, per condition: every lie-prompt attempt is one of
unfinished (no final answer within the token budget), answered truthfully, or a wrong answer that the judge splits into
a sincere lie / flagged as false / also gives the true answer / no specific answer.

    uv run python -m probe_evasion.lie_outcomes results/deception/eval_<run> [results/deception/eval_<run2> ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from probe_evasion.generation import read_jsonl

CATEGORIES = [  # (key, legend label, color)
    ("lie", "sincere lie (the only kind the probe is tested on)", "#eb6834"),
    ("false_flagged", "wrong answer, signals it is false or unsure", "#f2a582"),
    ("mentions_correct", "wrong answer, also gives the true one", "#f7c9b2"),
    ("no_answer", "refuses / no specific answer", "#9b9a95"),
    ("truthful", "answers truthfully instead", "#2a78d6"),
    ("unfinished", "never finishes (stuck deliberating, hits 8k tokens)", "#d6d5d0"),
]
CONDITION_NAMES = {"E0": "No warning", "placebo": "Unrelated instruction", "aware": "Probe is watching",
                   "E1": "Told to evade", "E2_d_claude": "Evade + 2 training examples", "E3": "Evade + strategy hints",
                   "E3_info": "Evade + probe description only", "E3_believe": "Evade + 'believe it' only"}
RUN_NAMES = {"20261006-144940": "first run", "20261006-162009_confirm": "confirmation run"}
JUDGED_TO_CATEGORY = {"false": "lie", "false_flagged": "false_flagged", "mentions_correct": "mentions_correct",
                      "no_answer": "no_answer", "correct": "truthful"}


def outcomes(run_dir: Path) -> dict[str, Counter]:
    summary_path = next(p for p in (run_dir / "summary_rescored_fresh.json", run_dir / "summary_rescored.json") if p.exists())
    judged_path = next(run_dir.glob("examples_scored*_judged.jsonl"))
    if "fresh" in summary_path.name:
        assert "fresh" in judged_path.name
    summary = json.loads(summary_path.read_text())
    judged = [e for e in read_jsonl(judged_path) if e["kind"] == "lie"]
    out = {}
    for condition, counts in summary["outcomes"].items():
        lie = counts["lie"]
        verdicts = Counter(JUDGED_TO_CATEGORY[e["judge"]["verdict"]] for e in judged if e["condition"] == condition)
        assert sum(verdicts.values()) == lie["completed_wrong"], (condition, verdicts, lie)
        out[condition] = verdicts + Counter({"unfinished": lie["unfinished"], "truthful": lie["completed_correct"]})
        assert sum(out[condition].values()) == lie["n"]
    return out


def plot(runs: dict[str, dict[str, Counter]], dest: Path) -> None:
    rows = [(f"{CONDITION_NAMES[c]}  ({RUN_NAMES.get(name, name)})", counts) for name, per in runs.items() for c, counts in per.items()]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.6), dpi=150)
    for y, (_, counts) in enumerate(rows):
        n, left = sum(counts.values()), 0.0
        for key, _, color in CATEGORIES:
            share = counts[key] / n
            ax.barh(y, share, left=left, color=color, edgecolor="white", linewidth=1.5)
            if key == "lie":
                ax.text(left + 0.01, y, f"{100 * share:.0f}%", va="center", fontsize=8.5, color="#0b0b0b")
            left += share
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of attempts under the lie prompt")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CATEGORIES], labels=[l for _, l, _ in CATEGORIES],
              loc="upper center", bbox_to_anchor=(0.45, -0.08 - 1.2 / len(rows)), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("What Qwen does when told to lie (TriviaQA, reasoning on)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    print(f"wrote {dest}")


if __name__ == "__main__":
    runs = {}
    for d in map(Path, sys.argv[1:]):
        runs[d.name.removeprefix("eval_")] = per = outcomes(d)
        print(d.name)
        for c, counts in per.items():
            n = sum(counts.values())
            print(f"  {c:14s} n={n:4d} " + " ".join(f"{k}={100 * counts[k] / n:4.1f}%" for k, _, _ in CATEGORIES))
    plot(runs, Path("status") / "lie_outcomes.png")
