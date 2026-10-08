"""Analysis of the candidate checks (m1a_offpolicy_maths.candidate_checks) and of the defender's retrained peak probes
(m1a_peak_probe). No GPU.

    uv run python -m probe_evasion.missions.m1a_candidates
"""

import json

import numpy as np
from collections import defaultdict

from probe_evasion import gepa
from probe_evasion.missions.m1a_analysis import FALSE_ALARMS, LINES, calibration_negatives
from probe_evasion.missions.m1a_calibration_audit import computes_flags, evasion_counts, line
from probe_evasion.missions.m1a_offpolicy_maths import CANDIDATES, OUT, SCORES, jl

PROBES = ("on@L48", "onm@L48", "off@L48", "on@L16")
READING = "ema"


def one_sentence_lines(probes=PROBES) -> dict:
    neg = calibration_negatives("one sentence", audited=True)
    return {p: {fa: line([r["scores"][p][READING] for r in neg], fa) for fa in FALSE_ALARMS} for p in probes}


def candidate_report() -> dict:
    lines = one_sentence_lines()
    out = {"one_sentence_lines": lines, "division": {}, "context": {}, "format_lines": {}}
    rows = jl(SCORES / "candidate_division.jsonl")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        for p in PROBES:
            for key in ("scores", "scores_instruction_removed"):
                rs = [r | {"_valid": r["lenient"], "_score": r[key][p][READING]} for r in rows if r["arm"] == arm]
                out["division"][f"{arm} | {p} | {'context' if key == 'scores' else 'instruction removed'}"] = evasion_counts(rs, {"one sentence": lines[p]})
            strict = [r | {"_valid": r["present"], "_score": r["scores"][p][READING]} for r in rows if r["arm"] == arm]
            out["division"][f"{arm} | {p} | context, strict judge"] = evasion_counts(strict, {"one sentence": lines[p]})
    ctx = jl(SCORES / "candidate_context.jsonl")
    for arm in dict.fromkeys(r["arm"] for r in ctx):
        for p in PROBES:
            rs = [r | {"_valid": r["lenient"], "_score": r["scores"][p][READING]} for r in ctx if r["arm"] == arm]
            out["context"][f"{arm} | {p}"] = evasion_counts(rs, {"one sentence": lines[p]})
    negs = jl(SCORES / "candidate_format_negatives.jsonl")
    for arm in CANDIDATES:
        clean = gepa.clean_maths_negatives([r for r in negs if r["arm"] == arm and r["finished"]])
        flags = computes_flags(clean)
        clean = [r for r, v in zip(clean, flags) if not (v.get("computes") or v.get("judge_refused"))]
        out["format_lines"][arm] = {"n": len(clean), **{p: {fa: line([r["scores"][p][READING] for r in clean], fa) for fa in FALSE_ALARMS}
                                                        for p in PROBES}}
        for p in PROBES:  # the candidate's division answers against a line set in its own format
            rs = [r | {"_valid": r["lenient"], "_score": r["scores"][p][READING]} for r in rows if r["arm"] == arm]
            out["division"][f"{arm} | {p} | context, own-format line"] = evasion_counts(rs, {"own format": out["format_lines"][arm][p]})
    (OUT / "candidate_report.json").write_text(json.dumps(out, indent=1))
    return out


def peak_report() -> dict:
    """Peak-trained probes: lines from the same clean, audited calibration pools (all styles, one sentence), misses on
    the v2 prompt's tests and on the same-format negatives."""
    from probe_evasion.missions.m1a_peak_probe import PEAK_PROBES

    probes = list(PEAK_PROBES)
    pools = {"all styles": [], "one sentence": []}
    by_key = {}
    for f in ("peak_calibration", "peak_calibration_one_sentence", "peak_v2_calibration"):
        for r in jl(SCORES / f"{f}.jsonl"):
            by_key[(r["task"]["task"], r["reply"])] = r
    for name in LINES:
        for r in calibration_negatives(name, audited=True):
            pools[name].append(by_key[(r["task"]["task"], r["reply"])])
    lines = {name: {p: {fa: line([r["scores"][p][READING] for r in rs], fa) for fa in FALSE_ALARMS} for p in probes}
             for name, rs in pools.items()}
    out = {"lines": lines, "n": {k: len(v) for k, v in pools.items()}, "tests": {}}
    for f in sorted(SCORES.glob("peak_v2_*test_*.jsonl")):
        rows = jl(f)
        for p in probes:
            rs = [r | {"_valid": gepa.lenient_maths_present(r), "_score": r["scores"][p][READING]} for r in rows]
            out["tests"][f"{f.stem} | {p}"] = evasion_counts(rs, {name: lines[name][p] for name in LINES})
    if (SCORES / "peak_format_negatives.jsonl").exists():
        fmt = defaultdict(list)
        for r in jl(SCORES / "peak_format_negatives.jsonl"):
            fmt[r["arm"]].append(r)
        out["format_negatives_median"] = {arm: {p: sorted(r["scores"][p][READING] for r in rs)[len(rs) // 2] for p in probes}
                                          for arm, rs in fmt.items()}
    (OUT / "peak_report.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "candidates"):
        r = candidate_report()
        print(json.dumps(r["format_lines"], indent=0))
        for k, v in list(r["division"].items()) + list(r["context"].items()):
            cells = {name: c for name, c in v.items() if isinstance(c, dict)}
            print(f"{k:70s} att {v['attempts']} valid {v['valid_attempts']} | " +
                  " ".join(f"{n}: {c['evasions_per_attempt']} ({c['distinct_evading_replies']} distinct, {c['problems_with_an_evasion']} problems)"
                           for n, c in cells.items() if n.endswith("1.0%") or n.startswith("own format 1.0")))
    if which in ("both", "peak"):
        r = peak_report()
        print(r["n"], json.dumps(r["lines"]))
        for k, v in r["tests"].items():
            print(f"{k:60s} att {v['attempts']} | " + " ".join(f"{n}: {c['evasions_per_attempt']}" for n, c in v.items()
                                                               if isinstance(c, dict) and n.endswith("1.0%")))
        print(r.get("format_negatives_median"))


def final_checks_report() -> dict:
    """v2 instruction in front of plain replies (all lines, own-replies probes), and the candidate rows under the
    peak-trained probes (their one-sentence lines)."""
    from probe_evasion.missions.m1a_calibration_audit import style_of

    out = {}
    neg_all = calibration_negatives("all styles", audited=True)
    lines = {"one sentence": one_sentence_lines(("on@L48", "onm@L48", "off@L48")),
             "all styles": {p: {fa: line([r["scores"][p][READING] for r in neg_all], fa) for fa in FALSE_ALARMS} for p in ("on@L48", "onm@L48", "off@L48")}}
    ctx = jl(SCORES / "v2_context.jsonl")
    for p in ("on@L48", "onm@L48", "off@L48"):
        rs = [r | {"_valid": r["lenient"], "_score": r["scores"][p][READING]} for r in ctx]
        out[f"plain reply, v2 in front | {p}"] = evasion_counts(rs, {k: v[p] for k, v in lines.items()})
    peak = json.loads((OUT / "peak_report.json").read_text())["lines"]["one sentence"]
    div = jl(SCORES / "peak_candidate_division.jsonl")
    ctxc = jl(SCORES / "peak_candidate_context.jsonl")
    for p in ("onpeak@L48", "offpeak@L48"):
        thr = {"one sentence": {float(k): v for k, v in peak[p].items()}}
        for arm in dict.fromkeys(r["arm"] for r in div):
            for key in ("scores", "scores_instruction_removed"):
                rs = [r | {"_valid": r["lenient"], "_score": r[key][p][READING]} for r in div if r["arm"] == arm]
                out[f"peak | {arm} | {p} | {key}"] = evasion_counts(rs, thr)
        for arm in dict.fromkeys(r["arm"] for r in ctxc):
            rs = [r | {"_valid": r["lenient"], "_score": r["scores"][p][READING]} for r in ctxc if r["arm"] == arm]
            out[f"peak | {arm} | {p}"] = evasion_counts(rs, thr)
    (OUT / "final_checks_report.json").write_text(json.dumps(out, indent=1))
    return out


def confirm_report() -> dict:
    """The pre-registered offpeak confirmation (m1a_peak_probe.offpeak_confirm): per probe, arm, subset and context,
    valid evasions per attempt at the one-sentence lines (0.1/1/5%) and at the candidate's own-format 1% line; strict
    counts alongside; plus held-out-style AUROC of the peak probes on Qwen's own replies."""
    from sklearn.metrics import roc_auc_score

    peak_lines = json.loads((OUT / "peak_report.json").read_text())["lines"]["one sentence"]
    lines = {**one_sentence_lines(("on@L48", "onm@L48", "off@L48")),
             **{p: {float(k): v for k, v in peak_lines[p].items()} for p in ("onpeak@L48", "offpeak@L48")}}
    fmt = jl(SCORES / "peak_candidate_format_negatives.jsonl")
    own = {}
    for arm in CANDIDATES:
        clean = gepa.clean_maths_negatives([r for r in fmt if r["arm"] == arm and r["finished"]])
        clean = [r for r, v in zip(clean, computes_flags(clean)) if not (v.get("computes") or v.get("judge_refused"))]
        own[arm] = {p: line([r["scores"][p][READING] for r in clean], 0.01) for p in lines}
    rows = jl(SCORES / "confirm.jsonl")
    out = {"lines": lines, "own_format_lines": own, "results": {}}
    held = jl(SCORES / "peak_heldout_qwen.jsonl")
    out["heldout_auroc"] = {p: roc_auc_score([r["label"] for r in held], [r["scores"][p][READING] for r in held])
                            for p in ("onpeak@L48", "offpeak@L48", "onpeak@L16", "offpeak@L16")}
    pos = [r for r in held if r["label"] == 1]
    out["heldout_plain_caught_1pct"] = {p: f"{sum(r['scores'][p][READING] > lines[p][0.01] for r in pos)}/{len(pos)}" for p in ("onpeak@L48", "offpeak@L48")}
    for p in lines:
        for arm in dict.fromkeys(r["arm"] for r in rows):
            for subset, keep in (("all", lambda r: True), ("division", lambda r: r["division"]), ("mixed", lambda r: not r["division"])):
                for key in ("scores", "scores_instruction_removed"):
                    mine = [r for r in rows if r["arm"] == arm and keep(r)]
                    th = {"one sentence": lines[p]}
                    if arm in own:
                        th["own format"] = {0.01: own[arm][p]}
                    lenient = evasion_counts([r | {"_valid": r["lenient"], "_score": r[key][p][READING]} for r in mine], th)
                    strict = evasion_counts([r | {"_valid": r["present"], "_score": r[key][p][READING]} for r in mine], th)
                    out["results"][f"{p} | {arm} | {subset} | {'context' if key == 'scores' else 'instruction removed'}"] = {
                        "lenient": lenient, "strict": strict}
    (OUT / "confirm_report.json").write_text(json.dumps(out, indent=1))
    return out


def seeded_report() -> dict:
    """Critic 4's must-fix: 3 seeds of each peak probe (val-loss stopping; own replies full, own replies size-matched,
    Claude text). Per probe instance: held-out AUROC, garden-lookup median score, the one-sentence 1% line with and
    without the garden-area lookups and "is actually" verifications, and confirmation-set evasions per attempt at each."""
    from sklearn.metrics import roc_auc_score

    by = {}
    for f in ("seeded_calibration", "seeded_calibration_one_sentence", "seeded_v2_calibration"):
        for r in jl(SCORES / f"{f}.jsonl"):
            by[(r["task"]["task"], r["reply"])] = r
    pool = [by[(r["task"]["task"], r["reply"])] for r in calibration_negatives("one sentence", audited=True)]
    pools = {"one sentence": pool,
             "minus garden lookups": [r for r in pool if "garden" not in r["task"]["task"]],
             "minus garden and 'actually'": [r for r in pool if "garden" not in r["task"]["task"] and "actually" not in r["reply"]]}
    rows = jl(SCORES / "seeded_confirm.jsonl")
    held = jl(SCORES / "seeded_heldout_qwen.jsonl")
    out = {}
    for p in sorted(rows[0]["scores"]):
        e = {"heldout_auroc": roc_auc_score([r["label"] for r in held], [r["scores"][p][READING] for r in held]),
             "garden_median": float(np.median([r["scores"][p][READING] for r in pool if "garden" in r["task"]["task"]])),
             "other_p99": float(np.quantile([r["scores"][p][READING] for r in pools["minus garden lookups"]], 0.99))}
        for name, rs in pools.items():
            thr = line([r["scores"][p][READING] for r in rs], 0.01)
            e[name] = {"line": thr, **{arm: f"{sum(r['lenient'] and r['scores'][p][READING] <= thr for r in rows if r['arm'] == arm)}/"
                                            f"{sum(r['arm'] == arm for r in rows)}" for arm in ("plain", "divaug-v2 winner", "window_L48 winner")},
                       "divaug strict": sum(r["present"] and r["scores"][p][READING] <= thr for r in rows if r["arm"] == "divaug-v2 winner")}
        valid = sorted(r["scores"][p][READING] for r in rows if r["arm"] == "divaug-v2 winner" and r["lenient"])
        e["divaug_reply_quantiles_1_5_50"] = [valid[int(len(valid) * q)] for q in (0.01, 0.05, 0.5)]
        out[p] = e
    (OUT / "seeded_report.json").write_text(json.dumps(out, indent=1))
    return out
