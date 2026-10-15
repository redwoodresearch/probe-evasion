"""Audit of m45, checks 5 (leakage) and 6 (freeze rule), from saved files only.

Usage: python experimental/analysis/audit_m45/leak_freeze.py <scratch dir with extract.py's pickles>
Writes experimental/analysis/audit_m45/leak_freeze.json.
"""

import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path("missions/m45_toggle_sweep/runs")
OUT = Path("experimental/analysis/audit_m45")
SCRATCH = Path(sys.argv[1])
ROOTS = {"maths": "qwen35_27b", "ordering": "qwen35_27b_ordering"}
B_POINTS = (0, 4, 8, 16)
SUFFIX = re.compile(r"\s*(Answer in one short sentence|Give the result|State the result|Answer with).*$", re.S)
DATA = Path("data/concepts")
TRAIN_FILES = {
    "maths": ["onpolicy_qwen35_maths_train_clean.jsonl", "onpolicy_qwen35_maths_heldout_clean.jsonl"],
    "ordering": ["m26_ordering_train.jsonl", "m26_ordering_heldout.jsonl", "m26_ordering_calibration.jsonl", "m26_ordering_build_raw.jsonl"],
}


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open() if x.strip()]


def stem(text: str) -> str:
    """The problem without its answer-format request (training rows use other request styles)."""
    first = re.split(r"(?<=[.?])\s+(?=(Answer|Give|State|List the|Reply|Do not|Write|Then|Respond))", text)[0]
    return re.sub(r"\s+", " ", first).strip().lower()


def maths_numbers(text: str) -> tuple:
    return tuple(sorted(re.findall(r"\d[\d,]*", stem(text))))


def ordering_clues(text: str) -> str:
    """The clue sentences (everything before the question) — identical clue sets mean the same puzzle."""
    sents = re.split(r"(?<=\.)\s+", text)
    return " ".join(s for s in sents if " than " in s or " before " in s or " after " in s).lower()


def main() -> None:
    out = {}
    for task, root in ROOTS.items():
        res: dict = {}
        test = {}
        for batch in ("c", "d"):
            rows = pickle.loads((SCRATCH / f"{batch}_{root}.pkl").read_bytes())
            t = {(r["id"], r["task_text"], str(r["result"])) for r in rows}
            test[batch] = t
        res["test_sets_identical_c_d"] = test["c"] == test["d"]
        tests = sorted(test["c"])
        res["n_test_problems"] = len(tests)
        res["n_distinct_test_texts"] = len({x[1] for x in tests})
        test_texts = {x[1] for x in tests}
        test_key = (lambda s: maths_numbers(s)) if task == "maths" else ordering_clues
        test_keys = {test_key(x[1]): x for x in tests}
        test_stems = {stem(x[1]) for x in tests}
        # search / practice problems seen by any chain, every batch (c, d)
        search_texts = set()
        search_ids = set()
        n_round_rows = 0
        freeze_issues, freeze_checked, carried, refusal_notes = [], 0, Counter(), []
        provider_by_writer = defaultdict(Counter)
        for batch in ("c", "d"):
            frozen = json.loads((RUNS / batch / root / "frozen.json").read_text())
            for st_path in sorted((RUNS / batch / root).glob("*/own/chain_*/state.json")):
                d = st_path.parent
                for f in sorted(d.glob("round_*.jsonl")):
                    for r in jl(f):
                        search_texts.add(r["task"]["task"])
                        search_ids.add(r["task"]["id"])
                        n_round_rows += 1
                st = json.loads(st_path.read_text())
                props = jl(d / "proposals.jsonl")
                for p in props:
                    provider_by_writer[f"{batch}/{st['writer']}"][p.get("provider")] += 1
                # check 6: the template frozen at B is round B's instruction, else the latest earlier one
                accepted = {}
                for p in props:
                    if p.get("instruction") and not p.get("problem") and p.get("stop") != "api_error":
                        accepted.setdefault(int(p["round"]), []).append(p["instruction"])
                for b in B_POINTS:
                    key = f"{st['writer']}/chain_{st['idx']}@B{b}"
                    ran = [r for r in st["rounds"][: b + 1] if r["instruction"] is not None]
                    expect = ran[-1] if ran else None
                    arm = frozen[key + "#on"]
                    freeze_checked += 1
                    if expect is None:
                        if arm["instruction"] is not None:
                            freeze_issues.append((key, "frozen has a template but no round <= B ran"))
                        continue
                    if arm["template"] != expect["instruction"] or int(arm["written_round"]) != int(expect["round"]):
                        freeze_issues.append((key, "template or written_round differs from state"))
                    if int(expect["round"]) != b:
                        carried[f"{batch}/{task}"] += 1
                    # the state's instruction is the round's last accepted proposal
                    acc = accepted.get(int(expect["round"]), [])
                    if not acc or acc[-1] != expect["instruction"]:
                        freeze_issues.append((key, f"state instruction is not the round's accepted proposal ({len(acc)} accepted)"))
                    if len(acc) > 1:
                        freeze_issues.append((key, f"round {expect['round']} has {len(acc)} accepted proposals"))
                    # round B itself produced no instruction while a later... (selection would show as template from a round != latest)
                    later = [r for r in st["rounds"][: b + 1] if r["instruction"] is not None and int(r["round"]) > int(expect["round"])]
                    if later:
                        freeze_issues.append((key, "a later round <= B ran but an earlier one was frozen"))
        res["search_round_rows"] = n_round_rows
        res["search_distinct_problems"] = len(search_texts)
        res["search_ids_sample"] = sorted(search_ids)[:5]
        res["test_texts_in_search_exact"] = len(test_texts & search_texts)
        res["test_stems_in_search"] = len(test_stems & {stem(s) for s in search_texts})
        res["test_keys_in_search"] = sorted(str(test_keys[k][0]) for k in test_keys if k in {test_key(s) for s in search_texts})
        # probe training / held-out / calibration data
        train_hits = {}
        for fn in TRAIN_FILES[task]:
            rows = jl(DATA / fn)
            texts = {r.get("prompt") or (r.get("task") or {}).get("task") for r in rows}
            texts.discard(None)
            keys = {test_key(t) for t in texts}
            train_hits[fn] = {"rows": len(rows), "exact_text": len(test_texts & texts), "same_stem": len(test_stems & {stem(t) for t in texts}),
                              "same_key": sorted(str(test_keys[k][0]) for k in test_keys if k in keys)}
        cal = jl(RUNS / "c" / root / "calibration_scores.jsonl")
        ctexts = {r["task"]["task"] for r in cal}
        train_hits["calibration_scores (line pool)"] = {"rows": len(cal), "exact_text": len(test_texts & ctexts),
                                                        "same_key": sorted(str(test_keys[k][0]) for k in test_keys if k in {test_key(t) for t in ctexts})}
        res["probe_data_overlap"] = train_hits
        # templates: test-problem specifics (maths: any test operand pair or result; ordering: test clue fragments)
        tmpl_hits = []
        for batch in ("c", "d"):
            frozen = json.loads((RUNS / batch / root / "frozen.json").read_text())
            for k, v in frozen.items():
                if not k.endswith("#on") or "parity" in k or not v.get("template"):
                    continue
                tm = v["template"]
                for tid, text, result in tests:
                    if task == "maths":
                        nums = [n.replace(",", "") for n in re.findall(r"\d[\d,]*", stem(text))]
                        found_nums = set(re.findall(r"\d+", tm.replace(",", "")))
                        if (len(nums) >= 2 and all(n in found_nums for n in nums)) or (int(result) >= 100 and result in found_nums):
                            tmpl_hits.append((batch, k, tid, "operands or result"))
                    else:
                        names = [n.strip() for n in result.split(",")]
                        # first four names of the answer in order inside the template
                        pos, ok = 0, 0
                        for nm in names:
                            m = re.search(rf"\b{re.escape(nm)}\b", tm[pos:])
                            if not m:
                                break
                            pos, ok = pos + m.end(), ok + 1
                        if ok >= 4:
                            tmpl_hits.append((batch, k, tid, f"{ok} answer names in order"))
        res["templates_with_test_specifics"] = tmpl_hits
        res["freeze_checked"] = freeze_checked
        res["freeze_issues"] = freeze_issues
        res["carried_forward_arms"] = dict(carried)
        res["providers"] = {k: dict(v) for k, v in provider_by_writer.items()}
        out[task] = res
        print(task, json.dumps({k: v for k, v in res.items() if k != "providers"}, default=str)[:3000])
    (OUT / "leak_freeze.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
