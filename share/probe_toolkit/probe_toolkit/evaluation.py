"""Flag lines, AUROC and catch rates. All scores are reply scores from one readout (see readouts.py)."""

import numpy as np
from sklearn.metrics import roc_auc_score


def flag_line(clean_negative_scores, false_alarm_rate: float = 0.01) -> float:
    """Score above which a reply is flagged, set so `false_alarm_rate` of the clean negatives are flagged.

    The negatives should be replies that truly lack the concept, in the same format and context as deployment.
    """
    scores = np.asarray(clean_negative_scores, dtype=np.float64)
    expected_flags = len(scores) * false_alarm_rate
    assert expected_flags >= 1, (
        f"a {false_alarm_rate:.2%} line from {len(scores)} negatives rests on {expected_flags:.2f} replies; "
        "use more negatives or a looser line")
    return float(np.quantile(scores, 1 - false_alarm_rate))


def flagged_fraction(scores, line: float) -> float:
    return float(np.mean(np.asarray(scores) > line))


def auroc(positive_scores, negative_scores) -> float:
    labels = np.r_[np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    return float(roc_auc_score(labels, np.r_[positive_scores, negative_scores]))


def evaluate(positive_scores, negative_scores, line: float) -> dict:
    """AUROC, the share of positives caught at the line, and the false-alarm rate on these (held-out) negatives."""
    return {"auroc": auroc(positive_scores, negative_scores),
            "catch_rate": flagged_fraction(positive_scores, line),
            "false_alarm_rate": flagged_fraction(negative_scores, line),
            "line": line,
            "n_positive": len(positive_scores),
            "n_negative": len(negative_scores)}
