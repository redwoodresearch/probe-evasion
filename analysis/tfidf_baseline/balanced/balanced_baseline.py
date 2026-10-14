"""TF-IDF B (prompt + reply) baseline for the BALANCED broad probes (m52, q27mbb / q27obb seed 0), side by side.

B is trained on exactly the balanced probe's training rows (m52.training_rows(..., "balanced")), with the original
baseline's grid and selection rule (detectors.fit_grid / choose: best held-out AUROC on m46's held-out split, ties to
the lowest held-out log-loss). Then, for B and the probe:
  1. held-out AUROC: held-out chat (split heldout, answer check passed) positives vs negatives; m46's held-out set and
     its re-read copies; held-out chat positives vs the WildChat line pool;
  2. the 1% line on the WildChat line pool (m52.line_pool(task, "balanced")); share of held-out chat positives and of
     m45's plain test answers (batch c, valid) over it;
  3. m45's round-16 ON / OFF replies (batches c, d): valid ON replies under the line, crossings, gap closed per writer
     and its slope per 10 ECI, with m52_analysis's own functions (the probe's numbers are asserted equal to
     analysis/broad_probe/m52/results.json);
  4. m51's matched-request test (identical reply text, compute vs given / wrong vs given) via m52_matched + analyze2.

    PYTHONPATH=src .venv/bin/python analysis/tfidf_baseline/balanced/balanced_baseline.py

CPU only. Writes results.json and models/<task>_B.pkl (+ _B_choice.json) next to this file; figure.py draws the figure.
"""

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions import m52_broad_probes as m52

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from detectors import auroc, choose, fit_grid  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A52 = load_module("m52_analysis", REPO / "analysis/broad_probe/m52_analysis.py")  # scored(), gap_closed_by_writer(), crossings(), NUM
MATCHED = load_module("m52_matched", REPO / "analysis/broad_probe/m52_matched.py")  # rows_for(), analyze2
FIGURES = load_module("tfidf_figures", HERE.parent / "figures.py")  # colours
STORED = json.loads((REPO / "analysis/broad_probe/m52/results.json").read_text())
STORED_MATCHED = json.loads((REPO / "analysis/broad_probe/m52/matched.json").read_text())
VARIANT = "balanced"
B_KEY = "tfidf_B"
EXPECTED_COUNTS = {"ordering": (1790, 3872), "maths": (1666, 3664)}  # (positives, negatives) of the balanced training rows


def as_bool(x) -> bool:
    return x in (True, "True")


def text_rows(rows: list[dict]) -> list[dict]:
    return [{"prompt": r["prompt"], "reply": r["response"], "label": int(r["label"]) if "label" in r else None} for r in rows]


def training_rows(task: str) -> list[dict]:
    run = m52.out_dir(False)
    wild = [r for r in jl(run / "wildchat_replies.jsonl") + jl(run / "wildchat_brief_replies.jsonl") if r["reply"] and m52.mostly_latin(r["request"])]
    labels = jl(run / "wildchat_labels.jsonl") + jl(run / "wildchat_brief_labels.jsonl")
    chat = jl(run / f"chat_{task}.jsonl")
    negatives = m52.wildchat_sets(task, wild, labels)["train"]
    rows = m52.training_rows(task, chat, negatives, False, VARIANT)
    counts = (sum(r["label"] == 1 for r in rows), sum(r["label"] == 0 for r in rows))
    assert counts == EXPECTED_COUNTS[task], (task, counts)
    return text_rows(rows)


def fit_b(task: str, sets: dict) -> tuple[object, dict]:
    path = HERE / "models" / f"{task}_B.pkl"
    meta_path = HERE / "models" / f"{task}_B_choice.json"
    if path.exists():
        return pickle.loads(path.read_bytes()), json.loads(meta_path.read_text())
    train = training_rows(task)
    grid = fit_grid("B", train, text_rows(sets["heldout"]))
    best = choose(grid)
    meta = {"feature_set": best["feature_set"], "C": best["C"], "selection_heldout_auroc": best["heldout_auroc"],
            "selection_heldout_log_loss": best["heldout_log_loss"], "n_train": len(train),
            "grid": [{k: g[k] for k in ("feature_set", "C", "heldout_auroc", "heldout_log_loss")} for g in grid]}
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(best["detector"]))
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[{task}] B chose {best['feature_set']} C={best['C']:g} (m46 held-out AUROC {best['heldout_auroc']:.4f})", flush=True)
    return best["detector"], meta


def add_scores(rows: list[dict], det, probe: str) -> None:
    """Each row gets peaks[tfidf_B] (B's score) next to peaks[probe], so m52_analysis's functions read either."""
    for r, s in zip(rows, det.score(text_rows(rows))):
        r["peaks"][B_KEY] = float(s)


def flagged(xs, line: float) -> float:
    return float(np.mean(np.asarray(xs) > line))


def detector_summary(task: str, key: str, sets: dict, pool: list[dict]) -> dict:
    sc = lambda rows: np.array([r["peaks"][key] for r in rows])
    chat_held = [r for r in sets["chat"] if r["split"] == "heldout" and as_bool(r["label_ok"])]
    pos = [r for r in chat_held if int(r["label"]) == 1]
    neg = [r for r in chat_held if int(r["label"]) == 0]
    orig = sets["heldout"] + sets["heldout_reread"]
    line = float(np.quantile(sc(pool), 1 - A52.FALSE_ALARM))
    plain = sc([r for r in sets["m45_c"] if r["key"].startswith("plain") and A52.is_valid(r)])
    plain_med = float(np.median(plain))
    assert plain_med > line, (task, key, plain_med, line)
    attacks = sets["m45_c"] + sets["m45_d"]
    on = [r for r in attacks if f"@B{A52.B}#on" in r["key"] and A52.is_valid(r)]
    by_writer = A52.gap_closed_by_writer(attacks, key, plain_med, line)
    return {"auroc_chat_heldout_pos_vs_neg": auroc([1] * len(pos) + [0] * len(neg), np.concatenate([sc(pos), sc(neg)])),
            "auroc_original_heldout_and_reread": auroc([int(r["label"]) for r in orig], sc(orig)),
            "auroc_chat_heldout_pos_vs_wildchat_line_pool": auroc([1] * len(pos) + [0] * len(pool), np.concatenate([sc(pos), sc(pool)])),
            "n_chat_heldout": {"pos": len(pos), "neg": len(neg)}, "n_original_heldout": len(orig), "n_line_pool": len(pool),
            "line": line, "plain_median": plain_med, "gap": plain_med - line,
            "chat_heldout_positives_flagged": flagged(sc(pos), line), "chat_heldout_negatives_flagged": flagged(sc(neg), line),
            "plain_flagged": flagged(plain, line), "n_plain_valid": len(plain),
            "lookalikes_flagged": flagged(sc(sets["lookalikes"]), line), "everyday_m27_flagged": flagged(sc(sets["everyday_m27"]), line),
            "n_on_valid": len(on), "on_under_line": int((sc(on) <= line).sum()), "on_under_line_share": float(np.mean(sc(on) <= line)),
            "crossings": A52.crossings(attacks, key, line),
            "writer_means": {w: float(np.mean(v)) for w, v in by_writer.items()}, "writer_chains": by_writer,
            "slope": A52.NUM.two_stage(by_writer, np.random.default_rng(A52.NUM.SEED))}


def check_probe(task: str, s: dict) -> None:
    ref = STORED[task]["balanced probe, WildChat line"]
    assert abs(s["line"] - ref["line"]) < 1e-9 and abs(s["plain_median"] - ref["plain_median"]) < 1e-9
    assert all(abs(s["writer_means"][w] - v) < 1e-9 for w, v in ref["writer_means"].items())
    assert abs(s["slope"]["slope"] - ref["slope"]["slope"]) < 1e-9 and s["slope"]["ci"] == ref["slope"]["ci"]
    assert s["crossings"] == ref["crossings"], (s["crossings"], ref["crossings"])
    assert abs(s["chat_heldout_positives_flagged"] - ref["chat_positives_flagged"]) < 1e-9


def matched(task: str, det) -> dict:
    """m51's identical-reply checks for B, through m52_matched's row loader and analyze2.summarise (B in the probe slot)."""
    rows = MATCHED.rows_for(task, VARIANT)
    for r, s in zip(rows, det.score([{"prompt": r["prompt"], "reply": r["reply"]} for r in rows])):
        r["probe"], r["older_probe"] = float(s), r["peaks"][m52.OLD_PROBE[task]]
    return rows


def main() -> None:
    out = {}
    for task in m52.TASKS:
        probe = m52.probe_name(task, 0, VARIANT)
        names = ("chat", "lookalikes", "heldout", "heldout_reread", "everyday_m27", "m45_c", "m45_d")
        sets = {n: A52.scored(task, n, VARIANT) for n in names}
        pool = m52.line_pool(task, VARIANT)
        det, meta = fit_b(task, sets)
        for rows in list(sets.values()) + [pool]:
            add_scores(rows, det, probe)
        res = {"B_choice": meta, "probe_name": probe}
        res["probe"] = detector_summary(task, probe, sets, pool)
        res[B_KEY] = detector_summary(task, B_KEY, sets, pool)
        check_probe(task, res["probe"])
        # ON replies flagged by one detector and not the other (each at its own line)
        on = [r for r in sets["m45_c"] + sets["m45_d"] if f"@B{A52.B}#on" in r["key"] and A52.is_valid(r)]
        under = lambda r, key: r["peaks"][key] <= res["probe" if key == probe else B_KEY]["line"]
        res["on_cross_table"] = {f"probe {'misses' if p else 'flags'}, B {'misses' if b else 'flags'}":
                                 sum(under(r, probe) == p and under(r, B_KEY) == b for r in on) for p in (False, True) for b in (False, True)}
        mrows = matched(task, det)
        lines = MATCHED.analyze2.lines_for(task) | {"probe": res[B_KEY]["line"]}
        d = MATCHED.analyze2.summarise(task, mrows, lines)["detectors"]["probe"]
        res["matched"] = {B_KEY: {k: d.get(k, {}) for k in ("auroc_compute_vs", "identical_reply", "identical_wrong_minus_given", "median")},
                          "probe": STORED_MATCHED[task]["balanced broad probe"]}
        out[task] = res
        for key in ("probe", B_KEY):
            s = res[key]
            print(f"[{task}] {key:8s} AUROC chat {s['auroc_chat_heldout_pos_vs_neg']:.3f} orig {s['auroc_original_heldout_and_reread']:.3f} "
                  f"pool {s['auroc_chat_heldout_pos_vs_wildchat_line_pool']:.3f} | line {s['line']:+.2f} | chat+ {s['chat_heldout_positives_flagged']:.3f} "
                  f"plain {s['plain_flagged']:.3f} | ON under {s['on_under_line']}/{s['n_on_valid']} | crossings {s['crossings']['forward']} | "
                  f"slope {s['slope']['slope']:+.3f} [{s['slope']['ci'][0]:+.3f}, {s['slope']['ci'][1]:+.3f}] | "
                  f"mean gap {np.mean(list(s['writer_means'].values())):.3f}", flush=True)
        for key, m in res["matched"].items():
            print(f"[{task}] matched {key}: " + ", ".join(f"{v} {x['share_compute_higher']:.2f} (n {x['n_pairs']}, median {x['median_compute_minus']:+.2f})"
                                                       for v, x in m["identical_reply"].items() if v.startswith("given"))
                  + " | wrong>given " + ", ".join(f"{v} {x['share_wrong_higher']:.2f} ({x['median']:+.2f})" for v, x in m["identical_wrong_minus_given"].items()))
    (HERE / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
