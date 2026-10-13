"""Spot-check the m38 (volitional track) numbers the post would quote.

Two sources: (a) m38's own analysis.json (correctness-only headline), read directly; (b) an independent recompute of
S from the raw test_rows.jsonl for the cells the post leans on (27B reasoning off, 27B reasoning on, answer and
whole readings). S = (P - median over attempts of score, invalid attempt = P) / (P - line), score read with the note
removed; valid = correctness only (strict-rule valid, or listed correct in correct_only.jsonl).

Usage: .venv/bin/python analysis/post_v9/numbers/m38_check.py
Writes analysis/post_v9/numbers/m38_check.json.
"""

import json
from pathlib import Path

import numpy as np

M38 = Path("missions/m38_volitional_track")
OUT = Path("analysis/post_v9/numbers")
A = json.loads((M38 / "analysis.json").read_text())
TOLD = ("C2", "C2c", "C3", "C3c", "C4", "C6")
PROBE = {"m27": "onpk@L48@s0"}


def from_analysis() -> dict:
    out = {}
    told_off = {}
    for cell in ("m9_off", "m27_off", "m122_off", "o27_off"):
        arms = A["test"][cell]["readings"]["reply"]["arms"]
        told_off[cell] = {a: {"S": arms[a]["S"], "under_1pct_valid": arms[a]["under_1pct"]} for a in TOLD}
        out[f"{cell}_C5_S"] = arms["C5"]["S"]
        out[f"{cell}_C5f_S"] = arms["C5f"]["S"]
    out["told_off"] = told_off
    s_told = [v["S"] for c in told_off.values() for a, v in c.items() if a not in ("C2c", "C3c")]
    s_all = [v["S"] for c in told_off.values() for v in c.values()]
    out["told_off_S_range_told_only"] = [min(s_told), max(s_told)]
    out["told_off_S_range_incl_controls"] = [min(s_all), max(s_all)]
    out["told_off_valid_under_1pct_total"] = sum(v["under_1pct_valid"] for c in told_off.values() for v in c.values())
    out["told_off_attempts"] = sum(1 for c in told_off.values() for _ in c) * 500
    for cell in ("m27_on", "m122_on"):
        for rd in ("answer", "whole"):
            arms = A["test"][cell]["readings"][rd]["arms"]
            out[f"{cell}_{rd}"] = {a: arms[a]["S"] for a in ("C2", "C2c", "C3", "C3c", "C4", "C5", "C6") if a in arms}
    si = A["selfiter"]["m122_on"]["readings"]["answer"]
    out["m122_on_selfiter_fresh"] = {v: si["variants"][v]["fresh"] for v in ("scores", "noscores")}
    out["m122_on_selfiter_fresh_diff"] = si["fresh_scores_minus_noscores"]
    out["m122_on_selfiter_turn8_diff"] = si["scores_minus_noscores_last"]
    return out


def correct_only_index(run: Path) -> dict:
    return {(r["task"], r["reply"]): r["correct"] for r in map(json.loads, (run / "correct_only.jsonl").open())}


def recompute(cell: str, mode: str) -> dict:
    run = M38 / "runs" / f"{cell}_{mode}"
    probe = PROBE[cell]
    co = correct_only_index(run)
    rows = [json.loads(x) for x in (run / "test_rows.jsonl").open()]
    readings = ("reply",) if mode == "off" else ("answer", "whole")
    out = {}
    for rd in readings:
        info = A["test"][f"{cell}_{mode}"]["readings"][rd]
        line = info["line"]
        score = (lambda r: r["peaks"][probe]) if rd == "reply" else (lambda r: r["peaks_by_reading"][rd][probe])

        def valid(r):
            if r.get("cut_off"):
                return False
            return bool(r["valid"]) or bool(co.get((r["task"]["task"], r["reply"]), False))

        plain = [score(r) for r in rows if r["key"] == "plain" and valid(r)]
        P = float(np.median(plain))
        res = {"P_ours": P, "P_m38": info["plain_median"], "line": line}
        for arm in sorted({r["key"] for r in rows}):
            rs = [r for r in rows if r["key"] == arm]
            x = [score(r) if valid(r) else P for r in rs]
            res[arm] = {"S_ours": (P - float(np.median(x))) / (P - line), "S_m38": info["arms"].get(arm, {}).get("S"),
                        "valid_ours": sum(valid(r) for r in rs), "valid_m38": info["arms"].get(arm, {}).get("valid"),
                        "valid_under_line_ours": sum(valid(r) and score(r) <= line for r in rs)}
        out[rd] = res
    return out


def main() -> None:
    out = {"analysis": from_analysis(), "recompute": {"m27_off": recompute("m27", "off"), "m27_on": recompute("m27", "on")}}
    (OUT / "m38_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:9000])


if __name__ == "__main__":
    main()
