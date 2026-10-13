"""The TF-IDF classifiers exactly as chosen in results/<task>.json, reusable by later analyses without retraining.

The fit is deterministic (same rows, same settings, lbfgs), so the chosen model is re-fitted once from the probe's
training rows, checked against the held-out scores saved by tfidf_baseline.py (scores/<task>_scores.jsonl.gz), and
pickled to models/<task>_<variant>.pkl; later calls load the pickle.
"""

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detectors import TfidfDetector  # noqa: E402

SAVED_ROUNDING = 2e-3  # the saved per-row scores are rounded to 3 decimals


def chosen_detector(task: str, variant: str) -> TfidfDetector:
    path = HERE / "models" / f"{task}_{variant}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from tasks import load_task

    cfg = json.loads((HERE / "results" / f"{task}.json").read_text())["chosen"][f"tfidf_{variant}"]
    data = load_task(task)
    det = TfidfDetector(variant, cfg["feature_set"], cfg["C"]).fit(data.train)
    with gzip.open(HERE / "scores" / f"{task}_scores.jsonl.gz", "rt") as f:
        saved = [json.loads(x) for x in f if '"set": "heldout"' in x]
    saved = [r[f"tfidf_{variant}"] for r in sorted(saved, key=lambda r: r["i"])]
    diff = np.abs(det.score(data.heldout) - np.array(saved))
    assert len(saved) == len(data.heldout) and diff.max() < SAVED_ROUNDING, (task, variant, diff.max())
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(det))
    print(f"{task} {variant}: re-fitted {det.name}, matches the saved held-out scores (max diff {diff.max():.1e}); saved {path}")
    return det


if __name__ == "__main__":
    for t in sys.argv[1:] or ("maths", "ordering"):
        for v in ("A", "B", "P"):
            chosen_detector(t, v)
