"""The crossings figure (fig_dots.py) with the word classifier in place of the probe: TF-IDF classifier B, which reads the
request and the reply, trained on the probe's own training rows (analysis/tfidf_baseline). Same answers (Claude Fable
5.1's round-16 attacks, 3 searches pooled, 1,500 answers per row; 500 unattacked answers), same grading (arithmetic's
consistent grade), same layout. The threshold is B's own 1% line on the same hard negatives; the big number is the
share of ON answers that are correct and at or under it. The attacks were tuned against the probe, not against B.

Scores come from the baseline's saved files (no model is run): analysis/tfidf_baseline/scores/<task>_scores.jsonl.gz,
matched row by row to cache/tfidf_baseline/<task>_{attacks,plain}.jsonl.gz (the rows it scored; written by
analysis/tfidf_baseline/extract.py), checked by the probe score both files carry.

    PYTHONPATH=src .venv/bin/python analysis/post_v9/fig_dots_tfidf.py   # writes figures/dots_2x2_tfidf.png
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("fig_dots", HERE / "fig_dots.py")
DOTS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DOTS)
sys.path.insert(0, str(HERE / "numbers"))
import arithmetic_grading  # noqa: E402

BASELINE = REPO / "analysis/tfidf_baseline"
CACHE = REPO / "cache/tfidf_baseline"
DETECTOR = "tfidf_B"
WRITER = DOTS.WRITER
X_LIMITS = (-11, 16)  # the probe figure's width; the right side leaves room for the labels beside contradiction's ON answers
M45_ROOT = {"maths": {"c": "qwen35_27b", "d": "qwen35_27b"}}  # arithmetic's consistent grade lives in these roots


def read_gz(path: Path) -> list[dict]:
    return [json.loads(line) for line in gzip.open(path)]


def scored(task: str, name: str) -> list[dict]:
    """The cached rows of one set with the classifier's score attached (row i of the cache = score row i of the set)."""
    rows = read_gz(CACHE / f"{task}_{name}.jsonl.gz")
    scores = sorted((r for r in read_gz(BASELINE / "scores" / f"{task}_scores.jsonl.gz") if r["set"] == name), key=lambda r: r["i"])
    assert len(rows) == len(scores) and [s["i"] for s in scores] == list(range(len(rows)))
    for r, s in zip(rows, scores):
        assert abs(r["probe_in_view"] - s["probe"]) < 1e-2, (task, name, s["i"])
        r["score"] = s[DETECTOR]
    return rows


def graded(task: str, rows: list[dict], batch: str) -> list[dict]:
    """Arithmetic: the consistent grade (arithmetic_grading.py), as every other post figure; other tasks unchanged."""
    if task != "maths":
        return rows
    grades = arithmetic_grading.load(REPO / "missions/m45_toggle_sweep/runs" / batch / M45_ROOT[task][batch])
    for r in rows:
        r["valid"] = grades[f"{r['key']}|{r['task_id']}|{r['seed']}"]
    return rows


def split(rows: list[dict]) -> dict[str, np.ndarray]:
    return {"correct": np.array([r["score"] for r in rows if r["valid"]]), "wrong": np.array([r["score"] for r in rows if not r["valid"]])}


def task_data(task: str) -> dict:
    attacks = [r for r in scored(task, "attacks") if r["writer"] == WRITER]
    assert {r["batch"] for r in attacks} == {"d"}
    attacks = graded(task, attacks, "d")
    d = {v: split([r for r in attacks if r["version"] == v]) for v in ("on", "off")}
    d["plain"] = split(graded(task, scored(task, "plain"), "c"))
    n_on = len(d["on"]["correct"]) + len(d["on"]["wrong"])
    assert n_on == 1500, (task, n_on)
    d["line"] = json.loads((BASELINE / "results" / f"{task}.json").read_text())[DETECTOR]["line_1pct"]
    d["passed"] = float(np.sum(d["on"]["correct"] <= d["line"])) / n_on
    return d


def main() -> None:
    data = {k: task_data(k) for k in DOTS.TASKS}
    for k, d in data.items():
        print(f"{k}: line {d['line']:+.2f}; median correct no attack {np.median(d['plain']['correct']):+.2f}, OFF "
              f"{np.median(d['off']['correct']):+.2f}, ON {np.median(d['on']['correct']):+.2f}; correct ON at or under the line "
              f"{d['passed']:.0%}; correct OFF at or under {np.mean(d['off']['correct'] <= d['line']):.0%}")
    DOTS.figure(data, DOTS.draw_row, "dots_2x2_tfidf.png", x_limits=X_LIMITS, xlabel="word classifier score (log-odds)",
                pass_label="passes the classifier")


if __name__ == "__main__":
    main()
