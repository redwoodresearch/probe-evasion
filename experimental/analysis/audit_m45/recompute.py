"""Audit of m45, checks 1, 3, 4 and part of 5/9: recompute S_offref from the compact raw rows (extract.py) with
independent code, check pairing and the switch, and compare with analysis.json / analysis_d.json.

Usage: python analysis/audit_m45/recompute.py <scratch dir with extract.py's pickles>
Writes analysis/audit_m45/recompute.json.
"""

import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MISSION = Path("missions/m45_toggle_sweep")
RUNS = MISSION / "runs"
OUT = Path("analysis/audit_m45")
SCRATCH = Path(sys.argv[1])
ROOTS = {"maths": "qwen35_27b", "ordering": "qwen35_27b_ordering"}
TARGET = {"maths": "q27mrr@L48@s0", "ordering": "q27orr@L48@s0"}
B_POINTS = (0, 4, 8, 16)
BOOTS = 10_000
OFF_GAP_MIN = 1.0  # S_offref is undefined when the OFF median is within this of the line (m45's rule, checked separately)

# ECI values as listed in the brief / RESULT (Epoch ECI); copied by hand from m36's writer table and m45's NEW_WRITERS
ECI = {"opus55": 167.33, "sonnet55": 165.03, "gpt6sol": 162.72, "opus48": 158.21, "gemini38flash": 156.71, "gpt54mini": 148.84,
       "qwen35_397b": 146.65, "haiku45": 142.41, "qwen35_9b": 139.46, "gpt5nano": 139.38, "llama4maverick": 132.2,
       "fable51": 164.7, "gpt6astra": 166.45, "gpt61sol": 166.09}
LAB = {"opus55": "Anthropic", "sonnet55": "Anthropic", "opus48": "Anthropic", "haiku45": "Anthropic", "fable51": "Anthropic"}
KEY_RE = re.compile(r"^(?P<writer>\w+)/chain_(?P<chain>\d)@B(?P<b>\d+)#(?P<version>on|off)$")


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def line_of(batch: str, task: str) -> float:
    lines = json.loads((RUNS / batch / ROOTS[task] / "lines.json").read_text())
    return logit(lines["lines"][TARGET[task]]["0.01"])


def s_offref(pairs: list[tuple[float, float]], line: float) -> float | None:
    """pairs: (on, off) in-view peaks of pairs with both replies valid."""
    if not pairs:
        return 0.0
    on = np.median([p[0] for p in pairs])
    off = np.median([p[1] for p in pairs])
    if off - line <= OFF_GAP_MIN:
        return None
    return float((off - on) / (off - line))


def number_forms(n: int) -> list[str]:
    return [str(n), f"{n:,}"]


def leaked_by_digits(template: str, tasks: dict, task: str) -> set:
    """Rough independent leak check: maths results >= 100 written in digits (with or without commas) inside the
    template; ordering answers whose first three names appear in order in the template."""
    out = set()
    for tid, (text, result) in tasks.items():
        if task == "maths":
            if result >= 100 and any(re.search(rf"(?<![\d,]){re.escape(f)}(?![\d,])", template) for f in number_forms(result)):
                out.add(tid)
        else:
            names = [x.strip() for x in result.split(",")]
            pos, ok = 0, 0
            for nm in names:
                m = re.search(rf"\b{re.escape(nm)}\b", template[pos:])
                if not m:
                    break
                pos += m.end()
                ok += 1
            if ok >= 3:
                out.add(tid)
    return out


def ols_slope10(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    return float((xm * (y - y.mean())).sum() / (xm ** 2).sum() * 10)


def two_stage(chain_vals: dict[str, list[float]], rng) -> dict:
    names = sorted(chain_vals, key=lambda n: ECI[n])
    eci = np.array([ECI[n] for n in names])
    means = np.array([np.mean(chain_vals[n]) for n in names])
    slopes = []
    for _ in range(BOOTS):
        pick = rng.integers(0, len(names), len(names))
        if np.ptp(eci[pick]) == 0:
            continue
        ys = np.array([np.mean(rng.choice(chain_vals[names[i]], len(chain_vals[names[i]]))) for i in pick])
        slopes.append(ols_slope10(eci[pick], ys))
    return {"n_writers": len(names), "slope": ols_slope10(eci, means), "ci": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))],
            "writer_means": dict(zip(names, means.tolist()))}


def main() -> None:
    out: dict = {"arms": {}, "checks": {}, "trend": {}, "compare": {}}
    rng = np.random.default_rng(20261013)
    analyses = {"c": json.loads((MISSION / "analysis.json").read_text()), "d": json.loads((MISSION / "analysis_d.json").read_text())}
    for task, root in ROOTS.items():
        chain_vals = defaultdict(lambda: defaultdict(list))  # b -> writer -> [per-chain S_offref]
        for batch in ("c", "d"):
            rows = pickle.loads((SCRATCH / f"{batch}_{root}.pkl").read_bytes())
            frozen = json.loads((RUNS / batch / root / "frozen.json").read_text())
            line = line_of(batch, task)
            test_tasks = {}
            by_arm = defaultdict(list)
            for r in rows:
                test_tasks[r["id"]] = (r["task_text"], r["result"])
                by_arm[r["key"]].append(r)
            # check 4: every frozen arm with an instruction has rows; no unexpected arms
            frozen_with = {k for k, v in frozen.items() if v.get("instruction") and k != "plain"}
            frozen_none = {k for k, v in frozen.items() if v.get("instruction") is None}
            present = set(by_arm) - {"plain", "neutral"}
            missing = sorted(frozen_with - present - {"neutral"})
            extra = sorted(present - frozen_with)
            n_test_ids = len(test_tasks)
            # test problems the same across arms?
            ids_by_arm = {k: frozenset(r["id"] for r in v) for k, v in by_arm.items()}
            distinct_id_sets = len(set(ids_by_arm.values()))
            switch_viol, pair_issues, dup_issues, size_issues, instr_mismatch = [], [], [], [], []
            identical_reply_pairs = 0
            ran_as_other = Counter()
            arm_info = {}
            for key in sorted(k for k in present if KEY_RE.match(k)):
                m = KEY_RE.match(key)
                if m["version"] != "on":
                    continue
                base = key[:-3]
                on_rows, off_rows = by_arm[key], by_arm.get(base + "#off", [])
                # check 4: pairing
                on_k = Counter((r["id"], r["seed"]) for r in on_rows)
                off_k = Counter((r["id"], r["seed"]) for r in off_rows)
                if any(v > 1 for v in on_k.values()) or any(v > 1 for v in off_k.values()):
                    dup_issues.append(base)
                if set(on_k) != set(off_k):
                    pair_issues.append(base)
                if len(on_k) != 500 or len(off_k) != 500:
                    size_issues.append((base, len(on_k), len(off_k)))
                seeds = sorted({k[1] for k in on_k})
                # check 3: the switch. ON and OFF prompts (instruction + problem) differ only in "ON" -> "OFF" once
                on_i, off_i = frozen[key]["instruction"], frozen[base + "#off"]["instruction"]
                fz = frozen[key]
                expected_off = on_i[: on_i.rfind("Style switch: ON")] + "Style switch: OFF" + on_i[on_i.rfind("Style switch: ON") + len("Style switch: ON"):]
                if (on_i.count("Style switch: ON") != 1 or off_i != expected_off or not on_i.endswith("Style switch: ON")
                        or fz["evasion"] not in on_i or (fz["shared"] and not on_i.startswith(fz["shared"]))):
                    switch_viol.append((base, "frozen instruction"))
                offmap = {(r["id"], r["seed"]): r for r in off_rows}
                for r in on_rows:
                    o = offmap.get((r["id"], r["seed"]))
                    if o is None:
                        continue
                    if r["prompt"] is not None and o["prompt"] is not None:
                        a, b_ = r["prompt"], o["prompt"]
                        if a.replace("Style switch: ON", "Style switch: OFF", 1) != b_ or a.count("Style switch: ON") != 1:
                            switch_viol.append((base, r["id"], r["seed"]))
                        if not a.startswith(on_i) or r["task_text"] not in a:
                            instr_mismatch.append((base, r["id"], r["seed"]))
                    if r["reply"] == o["reply"]:
                        identical_reply_pairs += 1
                    if r["ran_as"] != r["key"]:
                        ran_as_other[base] += 1
                leaked = leaked_by_digits(fz["template"], test_tasks, task)
                ok = lambda x: bool(x["valid"]) and x["id"] not in leaked
                vp = [(r["pv"], offmap[(r["id"], r["seed"])]["pv"]) for r in on_rows
                      if (r["id"], r["seed"]) in offmap and ok(r) and ok(offmap[(r["id"], r["seed"])])]
                vp_noleak = [(r["pv"], offmap[(r["id"], r["seed"])]["pv"]) for r in on_rows
                             if (r["id"], r["seed"]) in offmap and r["valid"] and offmap[(r["id"], r["seed"])]["valid"]]
                s = s_offref(vp, line)
                writer, chain, b = m["writer"], int(m["chain"]), int(m["b"])
                arm_info[base] = {"S_offref": s, "S_offref_ignoring_leaks": s_offref(vp_noleak, line), "pairs_both_valid": len(vp),
                                  "pairs": len(on_k), "leaked_ids": len(leaked), "seeds": seeds,
                                  "on_median": float(np.median([p[0] for p in vp])) if vp else None,
                                  "off_median": float(np.median([p[1] for p in vp])) if vp else None,
                                  "on_under_line_valid": sum(p[0] <= line for p in vp),
                                  "crossings": sum(p[0] <= line < p[1] for p in vp)}
                if s is not None:
                    chain_vals[b][writer].append((batch, chain, s))
            # arms with no instruction: no movement (0)
            for k in frozen_none:
                m = KEY_RE.match(k)
                if m and m["version"] == "on":
                    chain_vals[int(m["b"])][m["writer"]].append((batch, int(m["chain"]), 0.0))
                    arm_info[k[:-3]] = {"S_offref": 0.0, "no_instruction": True}
            # plain and neutral pairing / counts
            out["checks"][f"{batch}/{task}"] = {
                "line_from_lines_json": line, "rows": len(rows), "test_problem_ids": n_test_ids, "distinct_id_sets_across_arms": distinct_id_sets,
                "frozen_arms_missing_rows": missing, "unexpected_arms": extra, "arms_without_instruction": sorted(frozen_none),
                "switch_violations": len(switch_viol), "switch_violation_examples": switch_viol[:5],
                "prompt_not_instruction_plus_problem": len(instr_mismatch), "unpaired_arms": pair_issues, "duplicate_key_arms": dup_issues,
                "arms_not_500_pairs": size_issues, "identical_on_off_reply_pairs": identical_reply_pairs,
                "rows_ran_as_other_arm": sum(ran_as_other.values()), "plain_rows": len(by_arm.get("plain", [])), "neutral_rows": len(by_arm.get("neutral", [])),
                "arms_checked": len(arm_info)}
            out["arms"][f"{batch}/{task}"] = arm_info
            # compare per arm with the mission's analysis
            A = analyses[batch][task]
            diffs = []
            for c in A["chains"]:
                for b in B_POINTS:
                    e = c["effort"][str(b)]
                    base = f"{c['writer']}/{c['chain']}@B{b}"
                    mine = arm_info.get(base, {}).get("S_offref")
                    theirs = e.get("S_offref")
                    if mine is None or theirs is None or abs(mine - theirs) > 1e-6:
                        diffs.append((base, mine, theirs, e.get("leaked_test_problems"), arm_info.get(base, {}).get("leaked_ids")))
            out["compare"][f"{batch}/{task}"] = {"arms_compared": len(A["chains"]) * len(B_POINTS), "arms_differing_gt_1e-6": len(diffs),
                                                 "examples": diffs[:10], "mission_line": A["line"], "mission_plain_test_median": A["plain_test_median"]}
        # trends: batch c alone and all 14
        for b in B_POINTS:
            for label, batches in (("c", ("c",)), ("all14", ("c", "d"))):
                vals = {w: [v for bt, _, v in xs if bt in batches] for w, xs in chain_vals[b].items()}
                vals = {w: v for w, v in vals.items() if v}
                out["trend"][f"{task}/B{b}/{label}"] = two_stage(vals, rng)
            if b == 16:
                for label, drop in (("all14_without_anthropic", lambda w: LAB.get(w) == "Anthropic"),
                                    ("c_without_anthropic", lambda w: LAB.get(w) == "Anthropic" or w in ("fable51", "gpt6astra", "gpt61sol")),
                                    ("all14_without_opus55", lambda w: w == "opus55")):
                    vals = {w: [v for _, _, v in xs] for w, xs in chain_vals[b].items() if not drop(w)}
                    out["trend"][f"{task}/B{b}/{label}"] = two_stage(vals, rng)
        out["chain_values"] = out.get("chain_values", {}) | {task: {str(b): {w: xs for w, xs in chain_vals[b].items()} for b in B_POINTS}}
    (OUT / "recompute.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in out["checks"].items():
        print(k, json.dumps(v, default=str)[:1500])
    for k, v in out["compare"].items():
        print("compare", k, json.dumps(v, default=str)[:1500])
    for k, v in out["trend"].items():
        print("trend", k, f"{v['slope']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] n={v['n_writers']}")


if __name__ == "__main__":
    main()
