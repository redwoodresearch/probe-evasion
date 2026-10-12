"""m48: can the contradiction be found without comparing facts? Shortcut audit on generated passages (no model).

    uv run python -m probe_evasion.missions.m48_shortcut_audit

1. Sentence-local text classifier: logistic regression on word 1-2-grams of single restatement sentences, predicting
   "contradicts" vs "consistent restatement", 5-fold cross-validated. Chance is 50%; v1's separate phrasings scored ~90%.
2. Per-passage rule from that classifier: answer the restatement it scores most contradiction-like ("none" never),
   vs the always-pick-a-restatement baseline.
3. "New value" rule: answer the first restatement whose value word (city, name, colour, day) does not appear earlier
   (a string-matching comparison, not a sentence-local one), else "none".
"""

import json
import random
import re

import numpy as np

from probe_evasion.missions.m48_tasks import CITIES, COLOURS, CONTRADICTION, DAYS, NAMES

N_ITEMS = 4000
VALUE_WORDS = CITIES + COLOURS + DAYS + tuple(NAMES)


def restatements(p: dict) -> list[tuple[int, str, bool]]:
    return [(i + 1, s, k.startswith("contradict")) for i, (s, k) in enumerate(zip(p["sentences"], p["sentence_kinds"]))
            if k.startswith(("restate", "contradict"))]


def new_value_rule(p: dict):
    for i, s, _ in restatements(p):
        vals = [w for w in VALUE_WORDS if re.search(rf"\b{w}\b", s)]
        earlier = " ".join(p["sentences"][: i - 1])
        if vals and not any(re.search(rf"\b{w}\b", earlier) for w in vals):
            return i
    return "none"


def audit(variant: str = "n8") -> dict:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    rng = random.Random(4890)
    items = [CONTRADICTION.make_item(rng, variant)["computed"] for _ in range(N_ITEMS)]
    rows = [(n, s, y, k) for k, p in enumerate(items) for n, s, y in restatements(p)]
    X = CountVectorizer(ngram_range=(1, 2)).fit_transform([s for _, s, _, _ in rows])
    y = np.array([y for _, _, y, _ in rows])
    proba = cross_val_predict(LogisticRegression(max_iter=3000), X, y, cv=5, method="predict_proba")[:, 1]
    best = {}
    for (n, _, _, k), pr in zip(rows, proba):
        if k not in best or pr > best[k][1]:
            best[k] = (n, pr)
    contra = [p for p in items if p["answer"] != "none"]
    out = {"variant": variant, "items": N_ITEMS,
           "sentence_classifier_accuracy": round(float(np.mean((proba > 0.5) == y)), 3),
           "sentence_classifier_rule_on_contradiction_items": round(float(np.mean([best[k][0] == p["answer"] for k, p in enumerate(items) if p["answer"] != "none"])), 3),
           "pick_a_restatement_at_random_on_contradiction_items": 0.5,
           "new_value_rule_all_items": round(float(np.mean([new_value_rule(p) == p["answer"] for p in items])), 3),
           "new_value_rule_contradiction_items": round(float(np.mean([new_value_rule(p) == p["answer"] for p in contra])), 3)}
    return out


if __name__ == "__main__":
    for v in ("n6", "n8"):
        print(json.dumps(audit(v)))
