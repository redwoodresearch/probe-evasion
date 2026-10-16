"""How much of the probe's ON/OFF shift does the word classifier's shift predict? Round 16, all 14 writers, pairs with
both answers correct and the problem not given away by the prompt (the post's pairs).

Shift = OFF score - ON score, per detector (probe read with the prompt in view; TF-IDF B on request + reply), each in
its own log-odds.
  pairs:  R^2 of the probe's shift on the classifier's shift across all pairs, and within searches (each search's
          mean removed), i.e. do the same replies that move the classifier move the probe?
  searches: per search (one frozen prompt), median probe shift regressed on median classifier shift across the 42
          searches. The fitted part is "the probe shift the wording change predicts"; the residual is the rest. Both
          are divided by the probe's gap (normal answer to threshold), so the raw median shift / gap is the post's
          probe-evasion score, and each part gets the post's slope per 10 ECI (two-stage bootstrap).

A word classifier is an imperfect stand-in for "wording": what it misses (word order, for one) lands in the residual.

    PYTHONPATH=src .venv/bin/python analysis/post_v9/numbers/wording_share.py   # writes wording_share.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute  # noqa: E402
import tfidf_trend  # noqa: E402

TASKS = tfidf_trend.TASKS


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def task_result(task: str, gap: float, rng) -> dict:
    arms = defaultdict(lambda: {"on": {}, "off": {}})
    for r in tfidf_trend.graded_attacks(task):
        arms[r["arm"]][r["version"]][(r["task_id"], r["seed"])] = r
    ok = lambda r: r["valid"] and not r["leaked"]
    pairs, per_arm = [], {}
    for arm, v in sorted(arms.items()):
        both = [k for k in v["on"] if ok(v["on"][k]) and ok(v["off"][k])]
        d_probe = np.array([v["off"][k]["probe_in_view"] - v["on"][k]["probe_in_view"] for k in both])
        d_words = np.array([v["off"][k]["score"] - v["on"][k]["score"] for k in both])
        pairs += [(arm, p, w) for p, w in zip(d_probe, d_words)]
        if len(both):
            per_arm[arm] = (float(np.median(d_probe)), float(np.median(d_words)))
    arm_of = np.array([a for a, _, _ in pairs])
    p_all, w_all = np.array([p for _, p, _ in pairs]), np.array([w for _, _, w in pairs])
    p_within, w_within = p_all.copy(), w_all.copy()
    for a in set(arm_of):
        m = arm_of == a
        p_within[m] -= p_all[m].mean()
        w_within[m] -= w_all[m].mean()
    names = sorted(per_arm)
    mp, mw = np.array([per_arm[a][0] for a in names]), np.array([per_arm[a][1] for a in names])
    slope, intercept = np.polyfit(mw, mp, 1)
    fitted, residual = (intercept + slope * mw) / gap, (mp - intercept - slope * mw) / gap
    def by_writer(values: np.ndarray) -> dict[str, list[float]]:
        out = defaultdict(list)
        for a, v in zip(names, values):
            out[a.split("/")[0]].append(float(v))
        for w in compute.ECI:  # a search with no correct pair moves nothing, as in the post
            out[w] += [0.0] * (3 - len(out[w]))
        return dict(out)
    return {"n_pairs": len(pairs), "pairs_r2": r_squared(w_all, p_all), "pairs_within_search_r2": r_squared(w_within, p_within),
            "searches_r2": r_squared(mw, mp), "searches_fit": {"probe_per_classifier_logodds": float(slope), "intercept": float(intercept)},
            "trend_raw": compute.two_stage(by_writer(mp / gap), rng), "trend_predicted_by_wording": compute.two_stage(by_writer(fitted), rng),
            "trend_residual": compute.two_stage(by_writer(residual), rng)}


def main() -> None:
    rng = np.random.default_rng(compute.SEED)
    gaps = {t: v["gap"] for t, v in json.loads((HERE / "numbers.json").read_text())["tasks"].items()}
    res = {t: task_result(t, gaps[t], rng) for t in TASKS}
    (HERE / "wording_share.json").write_text(json.dumps(res, indent=1))
    f = lambda d: f"{d['slope']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
    for t, r in res.items():
        print(f"{t:14s} pairs {r['n_pairs']:6d}: R2 {r['pairs_r2']:.2f} (within searches {r['pairs_within_search_r2']:.2f}); "
              f"searches R2 {r['searches_r2']:.2f}; slope per 10 ECI: raw {f(r['trend_raw'])}, predicted by wording "
              f"{f(r['trend_predicted_by_wording'])}, residual {f(r['trend_residual'])}")


if __name__ == "__main__":
    main()
