"""The TF-IDF text classifiers: logistic regression on TF-IDF n-gram features of the reply (A), the prompt and the
reply (B, separate vocabularies, so a prompt word and a reply word are different features) or the prompt alone (P, a
diagnostic: how much of a task can be told from the request without reading the answer)."""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

VARIANT_FIELDS = {"A": ("reply",), "B": ("prompt", "reply"), "P": ("prompt",)}
FEATURE_SETS = {
    "word12": [dict(analyzer="word", ngram_range=(1, 2))],
    "char25": [dict(analyzer="char_wb", ngram_range=(2, 5))],
    "word12+char25": [dict(analyzer="word", ngram_range=(1, 2)), dict(analyzer="char_wb", ngram_range=(2, 5))],
}
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
AUROC_TIE = 1e-3  # held-out AUROCs within this of the best count as tied; ties go to the lowest held-out log-loss


class TfidfDetector:
    def __init__(self, variant: str, feature_set: str, C: float):
        self.variant, self.feature_set, self.C = variant, feature_set, C
        self.vectorizers = {(field, i): TfidfVectorizer(sublinear_tf=True, **kw)
                            for field in VARIANT_FIELDS[variant] for i, kw in enumerate(FEATURE_SETS[feature_set])}
        self.model = LogisticRegression(C=C, max_iter=5000)

    @property
    def name(self) -> str:
        return f"{self.variant}:{self.feature_set}:C={self.C:g}"

    def _features(self, prompts: list[str], replies: list[str], fit: bool):
        text = {"prompt": prompts, "reply": replies}
        blocks = []
        for (field, _), vec in self.vectorizers.items():
            if fit:
                blocks.append(vec.fit_transform(text[field]))
            else:  # attack prompts repeat a lot: transform each distinct text once
                unique, index = np.unique(np.array(text[field], dtype=object), return_inverse=True)
                blocks.append(vec.transform(list(unique))[index])
        return sp.hstack(blocks).tocsr()

    def fit(self, rows: list[dict]) -> "TfidfDetector":
        self.model.fit(self._features([r["prompt"] for r in rows], [r["reply"] for r in rows], fit=True), [r["label"] for r in rows])
        return self

    def score(self, rows: list[dict]) -> np.ndarray:
        return self.model.decision_function(self._features([r["prompt"] for r in rows], [r["reply"] for r in rows], fit=False))

    def top_features(self, n: int = 15) -> dict[str, list]:
        names = np.concatenate([[f"{field}:{x}" for x in v.get_feature_names_out()] for (field, _), v in self.vectorizers.items()])
        w = self.model.coef_[0]
        order = np.argsort(w)
        return {"toward_computed": [[names[i], round(float(w[i]), 3)] for i in order[::-1][:n]],
                "toward_not_computed": [[names[i], round(float(w[i]), 3)] for i in order[:n]]}


def auroc(labels, scores) -> float:
    return float(roc_auc_score(labels, scores))


def auroc_by_kind(rows: list[dict], scores: np.ndarray) -> dict:
    """AUROC over all rows, and computed rows against each negative kind alone."""
    out = {"all": auroc([r["label"] for r in rows], scores)}
    for k in sorted({r["kind"] for r in rows} - {"computed"}):
        idx = [i for i, r in enumerate(rows) if r["kind"] in ("computed", k)]
        out[f"vs {k}"] = auroc([rows[i]["label"] for i in idx], scores[idx])
    return out


def fit_grid(variant: str, train: list[dict], heldout: list[dict]) -> list[dict]:
    """Every grid entry fitted on the training rows, with its held-out AUROC (grid order: feature set, then C)."""
    grid = []
    for fs in FEATURE_SETS:
        for C in C_GRID:
            det = TfidfDetector(variant, fs, C).fit(train)
            labels, scores = [r["label"] for r in heldout], det.score(heldout)
            grid.append({"feature_set": fs, "C": C, "heldout_auroc": auroc(labels, scores),
                         "heldout_log_loss": float(log_loss(labels, 1 / (1 + np.exp(-scores)))), "detector": det})
    return grid


def choose(grid: list[dict]) -> dict:
    """The best held-out AUROC. Many entries tie at or near 1.0, so ties (within AUROC_TIE) go to the lowest held-out
    log-loss: the most confident correct separation, which also keeps the score in a log-odds scale like the probe's
    (the most regularised ties score everything within a fraction of a log-odds of the line)."""
    best = max(g["heldout_auroc"] for g in grid)
    return min((g for g in grid if g["heldout_auroc"] >= best - AUROC_TIE), key=lambda g: g["heldout_log_loss"])
