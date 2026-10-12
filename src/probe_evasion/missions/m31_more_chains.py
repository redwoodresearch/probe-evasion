"""Mission m31_more_chains: more chains of m28's pinned Opus 4.8 writer, with probe scores (`opus48_pinned`) and
without (`opus48_pinned_noscores`), on m22's runner, for maths (m22's Task, own-reply probe) and ordering (m26's Task
and Monitor). Pooled with m28's 3 + 3 pinned chains per task, this gives 8 chains per arm for "does the writer need
probe scores?".

Nothing here changes how a chain runs: the Monitor is m28's `pinned_monitor(task)` (imported; importing m28 registers
the pinned writers in m22.WRITERS), the search is m22's `run_search` and the final test m22's `run_frozen_test` (the
frozen arms are written here first, so m22's freeze reads them). Entry points and the run log: STATUS.md.
"""

import itertools
from collections import Counter
import json
import shutil
import sys
import time
from pathlib import Path

import anthropic
import attrs
import numpy as np

from probe_evasion.generation import read_jsonl as jl
from probe_evasion.judge import log_usage
from probe_evasion.missions import m22_score_only_iterations as m22  # read-only: the runner
from probe_evasion.missions import m28_baselines_and_writer_knowledge as m28  # read-only: the pinned writer, analysis helpers

MISSION = "m31_more_chains"
OUT = Path("missions") / MISSION
RUNS = OUT / "runs"  # pinned Opus 4.8 chains (the brief)
RUNS55 = OUT / "runs_opus55"  # pinned Opus 5.5 chains (coordinator's amendment); own root so the two drivers never share files
FIGURES = OUT / "figures"
TASKS = ("maths", "ordering")
SCORES, NOSCORES = m28.PINNED_WRITERS  # "opus48_pinned", "opus48_pinned_noscores"
SCORES55, NOSCORES55 = "opus55_pinned", "opus55_pinned_noscores"
TARGET = m28.TARGET  # "own"
PRIMARY_ROUNDS = list(range(1, m22.N_ROUNDS))  # rounds 1-15: the pre-registered per-chain measure (as m28's rule 4)
BOOTS = 10_000
OPUS55 = "claude-opus-5-5"

_client = None


def pinned_opus55(system: str, messages: list[dict]) -> dict:
    """m28.pinned_opus's call (m22's Opus parameters: adaptive thinking, effort high, 1-hour cache, no server-side
    fallback) on claude-opus-5-5. A refusal comes back as stop "refusal", which m22's parser counts as a failed attempt."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=8)
    start = time.time()
    with _client.messages.stream(
            model=OPUS55, max_tokens=64000, system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            messages=messages, thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": "high"},
            cache_control={"type": "ephemeral", "ttl": "1h"}) as stream:
        msg = stream.get_final_message()
    log_usage(msg.model, msg.usage)
    assert msg.model == OPUS55, msg.model
    return {"text": "".join(b.text for b in msg.content if b.type == "text"),
            "reasoning": "\n".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking"),
            "model": msg.model, "provider": "anthropic", "stop": msg.stop_reason, "usage": msg.usage.model_dump(exclude_none=True),
            "cost": m22.opus_cost(msg.model, msg.usage), "seconds": round(time.time() - start, 1)}


m22.WRITERS.update({  # registered as m28 registers its pinned 4.8 writers (m22's Chain calls monitor.self_writer)
    SCORES55: {"api": "monitor_self", "model": OPUS55},
    NOSCORES55: {"api": "monitor_self", "model": OPUS55, "feedback": "noscores"},
})


def pinned_maths() -> m22.Monitor:
    return m28.pinned_monitor("maths")


def pinned_ordering() -> m22.Monitor:
    return m28.pinned_monitor("ordering")


def pinned55_monitor(task: str) -> m22.Monitor:
    return attrs.evolve(m28.base_monitor(task), self_writer=m28.refusal_checked(pinned_opus55, task))


def pinned55_maths() -> m22.Monitor:
    return pinned55_monitor("maths")


def pinned55_ordering() -> m22.Monitor:
    return pinned55_monitor("ordering")


@attrs.frozen
class Family:
    """One pinned writer model's new chains: where they live, which Monitor factory, writer names, chain count."""
    label: str
    runs: Path
    factory: str  # factory name in this module, formatted with the task
    scores: str
    noscores: str
    n_chains: int
    twin_scores: str  # m28's writer whose saved system prompt this family's prompt must equal
    twin_noscores: str


FAMILIES = {"opus48": Family("opus48", RUNS, "pinned_{task}", SCORES, NOSCORES, 5, SCORES, NOSCORES),
            "opus55": Family("opus55", RUNS55, "pinned55_{task}", SCORES55, NOSCORES55, 4, SCORES, NOSCORES)}


def monitor_spec(task: str, fam: Family) -> str:
    return f"probe_evasion.missions.{MISSION}:{fam.factory.format(task=task)}"


def root_of(task: str, family: str = "opus48") -> Path:
    return m28.root_of(task, FAMILIES[family].runs)


def m28_root(task: str) -> Path:
    return m28.root_of(task, m28.RUNS)


# ---------------------------------------------------------------- runs


def prepare(task: str, family: str = "opus48") -> None:
    """Local, no GPU. Copies the source run's calibration scores and plain eval replies (m28.prepare_root, as m28 did),
    m22's m19-reference eval rows (so m22's frozen test does not re-run them), writes the chain files, and starts every
    new chain's state (system prompt + opening) exactly as run_search would, asserting that each system prompt is the
    one m28's pinned 4.8 chain of the same feedback kind started with."""
    fam = FAMILIES[family]
    root = m28.prepare_root(task, fam.runs)
    if task == "maths" and not (root / "reference_eval.jsonl").exists():
        shutil.copy(m22.RUNS / "qwen35_27b" / "reference_eval.jsonl", root / "reference_eval.jsonl")
    tag = "search" if family == "opus48" else "search55"
    ctx = m22.search_context(m22.load_monitor(monitor_spec(task, fam)), str(fam.runs), m22.N_ROUNDS, tag)
    for w, twin in ((fam.scores, fam.twin_scores), (fam.noscores, fam.twin_noscores)):
        theirs = json.loads((m28_root(task) / twin / TARGET / "chain_0" / "state.json").read_text())
        for i in range(fam.n_chains):
            c = m22.Chain(ctx, w, TARGET, i)
            assert c.system == theirs["system"], f"{task} {w}: writer prompt differs from m28's {twin}"
            assert c.state["messages"][:1] == theirs["messages"][:1]
            c.save()
    suffix = "" if family == "opus48" else "_opus55"
    search = {f"m31_search_{task}{suffix}": [["probe_evasion.missions.m22_score_only_iterations", "run_search", {
        "monitor": monitor_spec(task, fam), "writers": f"{fam.scores},{fam.noscores}", "targets": TARGET, "chains": fam.n_chains,
        "rounds": m22.N_ROUNDS, "out_dir": str(fam.runs), "cache_tag": tag, "zeroshot": 0}]]}
    test = {f"m31_test_{task}{suffix}": [["probe_evasion.missions.m22_score_only_iterations", "run_frozen_test", {
        "monitor": monitor_spec(task, fam), "out_dir": str(fam.runs), "cache_tag": "test", "rounds": m22.N_ROUNDS}]]}
    (OUT / f"search_chain_{task}{suffix}.json").write_text(json.dumps(search))
    (OUT / f"test_chain_{task}{suffix}.json").write_text(json.dumps(test))


def freeze(task: str, family: str = "opus48") -> dict:
    """Pre-registered final-test arms, written to <root>/frozen.json before any test reply (m22's run_frozen_test
    then reads this file instead of building its own): plain, and per new chain its best round (m22's freeze rule,
    with the task's leak rule) and its last round that ran (selection-free). Same arm names as m28's."""
    root = root_of(task, family)
    path = root / "frozen.json"
    if path.exists():
        return json.loads(path.read_text())
    leaked = m28.base_monitor(task).task.leaked
    arms = {"plain": {"instruction": ""}}
    for d in m22.chain_dirs(root):
        st = json.loads((d / "state.json").read_text())
        assert len(st["rounds"]) == m22.N_ROUNDS and all("stats" in r or r["instruction"] is None for r in st["rounds"]), f"{d} unfinished"
        name = f"{st['writer']}/chain_{st['idx']}"
        best = m22.best_round(d, leaked)
        if best is None:
            arms.setdefault("_absent", []).append(f"{name}@best")
        else:
            arms[f"{name}@best"] = {"instruction": best["instruction"], "round": best["round"], "freeze_median": best["freeze_median"],
                                    "eval_valid": best["stats"]["valid"], "answered_by": best["answered_by"]}
        ran = [r for r in m22.chain_rounds(d) if r.get("stats")]
        if not ran:  # every round failed (e.g. the writer refused throughout): no arms, recorded
            arms.setdefault("_absent", []).append(f"{name}@last")
            continue
        last = ran[-1]
        arms[f"{name}@last"] = {"instruction": last["instruction"], "round": last["round"], "eval_valid": last["stats"]["valid"],
                                "answered_by": last["answered_by"], "selection_free": True}
    absent = arms.pop("_absent", [])  # pre-registered: a chain with no round >= 40/80 valid has no best arm (recorded)
    path.write_text(json.dumps(arms, indent=1, ensure_ascii=False))
    (root / "frozen_absent.json").write_text(json.dumps(absent))
    return arms


# ---------------------------------------------------------------- analysis (saved files only)


EARLIER_ROOT = {"maths": m22.RUNS / "qwen35_27b", "ordering": m28.SOURCE_ROOTS["ordering"]}  # m22 / m26 (mixed 5.5 -> 4.8)
# arm label -> (writer model, scores shown, [(source, root fn, writer)])
ARMS = {
    "4.8/scores": ("4.8", True, [("m28", m28_root, SCORES), ("m31", lambda t: root_of(t, "opus48"), SCORES)]),
    "4.8/no scores": ("4.8", False, [("m28", m28_root, NOSCORES), ("m31", lambda t: root_of(t, "opus48"), NOSCORES)]),
    "5.5/scores": ("5.5", True, [("m31", lambda t: root_of(t, "opus55"), SCORES55)]),
    "5.5/no scores": ("5.5", False, [("m31", lambda t: root_of(t, "opus55"), NOSCORES55)]),
    "mixed/scores": ("mixed", True, [("earlier", EARLIER_ROOT.get, "opus")]),  # reference only, not tested
    "mixed/no scores": ("mixed", False, [("earlier", EARLIER_ROOT.get, "opus_noscores")]),
}
PINNED_ARMS = [a for a, v in ARMS.items() if v[0] != "mixed"]


def refusal_stats(d: Path) -> dict:
    """From every writer call in proposals.jsonl: answered attempts, refusals by the API (stop reason) and by the Haiku
    check on the instruction, failed rounds, and the first round with a refusal. For the earlier (fallback) chains,
    a refusal shows up as a round answered by Opus 4.8 instead."""
    calls = [json.loads(x) for x in (d / "proposals.jsonl").read_text().splitlines()]
    answered = [x for x in calls if x["stop"] != "api_error"]
    haiku = [x for x in answered if (x.get("refusal_check") or {}).get("is_refusal")]
    api = [x for x in answered if x["stop"] == "refusal" and not (x.get("refusal_check") or {}).get("is_refusal")]
    refused_rounds = sorted({x["round"] for x in haiku + api})
    leaks = [x for x in answered if "contains answers" in (x.get("problem") or "")]
    return {"answered_attempts": len(answered), "api_refusals": len(api), "haiku_refusals": len(haiku), "leak_rejected_attempts": len(leaks),
            "refusal_share": (len(api) + len(haiku)) / len(answered) if answered else None,
            "rounds_with_refusal": refused_rounds, "first_refusal_round": refused_rounds[0] if refused_rounds else None,
            "api_errors": len(calls) - len(answered)}


def failure_cause(info: dict) -> str:
    """Why a round produced no instruction: the last attempt's problem (m22's parser wording)."""
    last = info["problems"][-1] if info["problems"] else ""
    return ("leak" if "contains answers" in last else "refusal" if "refus" in last else "api error" if last.startswith("API error")
            else "other: " + last[:60])


def chain_records(task: str, info: dict) -> list[dict]:
    """One record per chain of every arm: per round the paired change vs plain (m22.paired_change through
    m28.unit_stats, leaked eval problems left out under the task's own rule via m28.rounds_view), valid count,
    answering model; the primary measure (mean over rounds 1-15 that ran), its carry-forward version (a failed round
    keeps the last instruction's change, 0 before any ran), phase means, refusal stats."""
    out = []
    for arm, (model, scores, sources) in ARMS.items():
        for source, root_fn, w in sources:
            for d in sorted((root_fn(task) / w / TARGET).glob("chain_*")):
                st = json.loads((d / "state.json").read_text())
                if len(st["rounds"]) < m22.N_ROUNDS or not all("stats" in r or r["instruction"] is None for r in st["rounds"]):
                    print(f"skipping unfinished chain {d}", file=sys.stderr)
                    continue
                rounds = []
                for r in m28.rounds_view(d, info["leaked"]):
                    e = {"round": r["round"], "answered_by": r["answered_by"], "leaked_problems": r.get("leaked_problems", 0)}
                    e |= (m28.unit_stats(r["rows"], info) if r["rows"] else {"failed": True})
                    if not e.get("failed") and e["paired_change"] is None:
                        e["failed"] = True  # ran, but no valid paired reply
                    rounds.append(e)
                ran = {e["round"]: e["paired_change"] for e in rounds if not e.get("failed")}
                mean_of = lambda rs: float(np.mean([ran[r] for r in rs if r in ran])) if any(r in ran for r in rs) else None
                carry, current = [], 0.0
                for rnd in range(m22.N_ROUNDS):
                    current = ran.get(rnd, current)
                    if rnd in PRIMARY_ROUNDS:
                        carry.append(current)
                prim = mean_of(PRIMARY_ROUNDS)
                out.append({"arm": arm, "model": model, "scores": scores, "source": source, "writer": w, "chain": d.name, "rounds": rounds,
                            "primary": prim, "primary_carry_forward": float(np.mean(carry)),
                            "rounds_1_3": mean_of([1, 2, 3]), "rounds_13_15": mean_of([13, 14, 15]), "round_0": ran.get(0),
                            "primary_minus_round_0": prim - ran[0] if prim is not None and 0 in ran else None,
                            "cost": float(sum(x["cost"] for x in st["rounds"])),
                            "failed_rounds": [e["round"] for e in rounds if e.get("failed")],
                            "failed_round_causes": dict(Counter(failure_cause(x) for x in st["rounds"] if x["instruction"] is None)
                                                        + Counter({"ran, no valid reply left (all invalid or leaked)": sum(
                                                            bool(e.get("failed")) and st["rounds"][e["round"]]["instruction"] is not None
                                                            for e in rounds)})),
                            "valid_share": float(np.mean([e["valid"] / e["n"] for e in rounds if not e.get("failed")])) if ran else None,
                            "answered_by": dict(Counter(e["answered_by"] for e in rounds if e["answered_by"])),
                            "distinct_instructions": len({x["instruction"] for x in st["rounds"] if x["instruction"]}),
                            "refusals": refusal_stats(d)})
    return out


def exact_permutation(a: list[float], b: list[float]) -> dict:
    """Two-sided exact permutation test of mean(a) - mean(b) over every relabelling of the pooled chains."""
    pooled = np.array(a + b)
    obs = float(np.mean(a) - np.mean(b))
    n, k = len(pooled), len(a)
    total = pooled.sum()
    diffs = np.array([pooled[list(ix)].sum() / k - (total - pooled[list(ix)].sum()) / (n - k) for ix in itertools.combinations(range(n), k)])
    return {"diff": obs, "p_two_sided": float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12)),
            "p_one_sided_lower": float(np.mean(diffs <= obs + 1e-12)), "n_relabellings": len(diffs)}


def stratified_permutation(a: list[float], b: list[float], batch_a: list[str], batch_b: list[str]) -> dict:
    """Batch-stratified exact permutation (critic 1): relabel scores / no scores only within each batch (m28, m31)."""
    obs = float(np.mean(a) - np.mean(b))
    vals, labels, batches = np.array(a + b), np.array([1] * len(a) + [0] * len(b)), np.array(batch_a + batch_b)
    per_batch = []
    for bt in sorted(set(batches)):
        idx = np.where(batches == bt)[0]
        k = int(labels[idx].sum())
        per_batch.append([list(c) for c in itertools.combinations(idx, k)])
    diffs = []
    for combo in itertools.product(*per_batch):
        pick = np.zeros(len(vals), bool)
        for c in combo:
            pick[c] = True
        diffs.append(vals[pick].mean() - vals[~pick].mean())
    diffs = np.array(diffs)
    return {"p_two_sided": float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12)), "n_relabellings": len(diffs)}


def welch_interval(a: list[float], b: list[float]) -> list[float] | None:
    from scipy import stats

    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    if va + vb == 0:
        return None
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = np.mean(a) - np.mean(b)
    return [float(d - half), float(d + half)]


def boot_diff(a: list[float], b: list[float]) -> list[float]:
    """95% interval for mean(a) - mean(b), resampling chains within each arm."""
    rng = np.random.default_rng(31)
    a, b = np.array(a), np.array(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(BOOTS)]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def hedges_g(a: list[float], b: list[float]) -> float | None:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if sp == 0:
        return None
    return float((np.mean(a) - np.mean(b)) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def compare(a: list[float], b: list[float], batch_a: list[str] | None = None, batch_b: list[str] | None = None) -> dict:
    """with scores (a) vs without (b): means, medians, ranges, difference with two intervals (percentile bootstrap
    over chains, which is narrow at n = 8, and Welch's t), exact permutation p (and batch-stratified p), Hedges' g."""
    side = lambda xs: {"n": len(xs), "mean": float(np.mean(xs)), "median": float(np.median(xs)), "range": [float(min(xs)), float(max(xs))],
                       "values": [float(x) for x in xs]} if xs else {"n": 0}
    out = {"with_scores": side(a), "without_scores": side(b)}
    if len(a) >= 2 and len(b) >= 2:
        out |= {**exact_permutation(a, b), "diff_ci95": boot_diff(a, b), "diff_welch_ci95": welch_interval(a, b), "hedges_g": hedges_g(a, b),
                "all_scored_below_all_unscored": bool(max(a) < min(b))}
        if batch_a and len(set(batch_a + batch_b)) > 1:
            out["stratified"] = stratified_permutation(a, b, batch_a, batch_b)
    return out


def test_key(source: str, key: str) -> tuple[str, str, str] | None:
    """(writer, chain, best|last) for a test-row key, or None for arms not used here."""
    if source == "earlier":  # m22 / m26: "opus/own/chain_0" (best) and "opus/own/chain_0@last" (maths only)
        if "/own/" not in key:
            return None
        writer, rest = key.split("/own/")
        return writer, rest.split("@")[0], "last" if rest.endswith("@last") else "best"
    if source == "m28_m26_last":  # m26's chains' last rounds, run in m28's ordering test: "m26/opus/chain_0@last"
        if not key.startswith("m26/"):
            return None
        _, writer, rest = key.split("/")
        return writer, rest.split("@")[0], "last"
    writer, _, rest = key.partition("/")
    return (writer, *rest.split("@")) if "@" in rest else None


def test_files(task: str) -> list[tuple[str, Path]]:
    files = [("m28", m28_root(task) / "m28_test_rows.jsonl"), ("m31", root_of(task, "opus48") / "test_rows.jsonl"),
             ("m31", root_of(task, "opus55") / "test_rows.jsonl"), ("earlier", EARLIER_ROOT[task] / "test_rows.jsonl")]
    if task == "ordering":
        files.append(("m28_m26_last", m28_root(task) / "m28_test_rows.jsonl"))
    return files


def test_records(task: str, info: dict, chains: list[dict]) -> list[dict]:
    """Per chain and kind (best, last): m28.test_measures against the plain arm of the same test file."""
    wanted = {(c["source"], c["writer"], c["chain"]): c for c in chains}
    out = []
    for source, path in test_files(task):
        if not path.exists():
            print(f"no final-test rows yet: {path}", file=sys.stderr)
            continue
        by: dict[str, list[dict]] = {}
        for r in jl(path):
            by.setdefault(r["key"], []).append(r)
        for key, rows in by.items():
            parsed = None if key == "plain" else test_key(source, key)
            if not parsed:
                continue
            writer, chain, kind = parsed
            c = wanted.get(("earlier" if source.startswith("m28_m26") else source, writer, chain))
            if c is None:
                continue
            m = m28.test_measures(rows, by["plain"], info)
            out.append({"arm": c["arm"], "model": c["model"], "scores": c["scores"], "source": c["source"], "chain": chain, "kind": kind,
                        "key": key, "file": str(path), **{k: m.get(k) for k in ("attempts", "valid", "median_valid", "plain_median", "change",
                                                                                  "change_ci95", "paired_problem_change", "gap_line", "median_visible", "audit")},
                        "under_1pct": m["under_0.01"], "under_1pct_share": m["under_0.01"]["per_attempt"] / m["attempts"]})
    return out


def summary(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": float(np.median(xs)) if xs else None, "mean": float(np.mean(xs)) if xs else None,
            "range": [float(min(xs)), float(max(xs))] if xs else None}


def interaction(a1: list[float], b1: list[float], a2: list[float], b2: list[float]) -> dict:
    """(mean a2 - mean b2) - (mean a1 - mean b1) with a Welch-Satterthwaite 95% interval."""
    from scipy import stats

    groups = [np.array(g) for g in (a1, b1, a2, b2)]
    if any(len(g) < 2 for g in groups):
        return {"estimate": None}
    est = (groups[2].mean() - groups[3].mean()) - (groups[0].mean() - groups[1].mean())
    vs = [g.var(ddof=1) / len(g) for g in groups]
    se2 = sum(vs)
    if se2 == 0:
        return {"estimate": float(est), "ci95": None}
    df = se2 ** 2 / sum(v ** 2 / (len(g) - 1) for v, g in zip(vs, groups))
    half = stats.t.ppf(0.975, df) * np.sqrt(se2)
    return {"estimate": float(est), "ci95": [float(est - half), float(est + half)], "df": float(df)}


def two_by_two(values: dict[str, list[float]], batches: dict[str, list[str]] | None = None) -> dict:
    """Per arm summaries; the scores effect within each pinned model; the model effect (5.5 - 4.8) within each
    feedback kind; the interaction (scores effect for 5.5 minus for 4.8). Mixed arms: summaries only."""
    batches = batches or {}
    v = lambda a: [x for x in values.get(a, []) if x is not None]
    bt = lambda a: [b for x, b in zip(values.get(a, []), batches.get(a, [])) if x is not None] or None
    return {"arms": {a: summary(v(a)) for a in ARMS},
            "scores_effect": {m: compare(v(f"{m}/scores"), v(f"{m}/no scores"), bt(f"{m}/scores"), bt(f"{m}/no scores")) for m in ("4.8", "5.5")},
            "model_effect_5.5_minus_4.8": {k: compare(v(f"5.5/{k}"), v(f"4.8/{k}")) for k in ("scores", "no scores")},
            "interaction": interaction(v("4.8/scores"), v("4.8/no scores"), v("5.5/scores"), v("5.5/no scores"))}


def per_round(chains: list[dict]) -> dict:
    """Per arm and round: median and mean over chains of the paired change, and how many chains ran that round."""
    out = {}
    for arm in ARMS:
        cs = [c for c in chains if c["arm"] == arm]
        out[arm] = []
        for rnd in range(m22.N_ROUNDS):
            v = [e["paired_change"] for c in cs for e in c["rounds"] if e["round"] == rnd and not e.get("failed")]
            out[arm].append({"round": rnd, "n_chains": len(v), "median": float(np.median(v)) if v else None, "mean": float(np.mean(v)) if v else None})
    return out


def refusal_table(chains: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        cs = [c for c in chains if c["arm"] == arm]
        if not cs:
            continue
        rs = [c["refusals"] for c in cs]
        att = sum(r["answered_attempts"] for r in rs)
        out[arm] = {"chains": len(cs), "answered_attempts": att, "api_refusals": sum(r["api_refusals"] for r in rs),
                    "haiku_refusals": sum(r["haiku_refusals"] for r in rs),
                    "leak_rejected_attempts": sum(r["leak_rejected_attempts"] for r in rs),
                    "refusal_share": (sum(r["api_refusals"] + r["haiku_refusals"] for r in rs) / att) if att else None,
                    "chains_with_a_refusal": sum(bool(r["rounds_with_refusal"]) for r in rs),
                    "first_refusal_round": [r["first_refusal_round"] for r in rs],
                    "failed_rounds": [len(c["failed_rounds"]) for c in cs],
                    "failed_round_causes": dict(sum((Counter(c["failed_round_causes"]) for c in cs), Counter())),
                    "failed_round_rate_by_cause": {k: v / (len(cs) * m22.N_ROUNDS) for k, v in
                                                   sum((Counter(c["failed_round_causes"]) for c in cs), Counter()).items()},
                    "failed_round_rate": sum(len(c["failed_rounds"]) for c in cs) / (len(cs) * m22.N_ROUNDS),
                    "rounds_answered_by": dict(sum((Counter(c["answered_by"]) for c in cs), Counter()))}
    return out


def answer_format(task: str, info: dict) -> dict:
    """Maths only: among valid eval replies in rounds 1-15, the share that give the result in number words (m28's
    result_in_words), and the median paired change vs plain for word and digit answers, per arm."""
    base = {(p["task"]["id"], p["seed"]): p["peaks"][info["probe"]] for p in info["plain"] if p["valid"]}
    out = {}
    for arm, (_, _, sources) in ARMS.items():
        fmt: dict[str, list[float]] = {"words": [], "digits": []}
        for _, root_fn, w in sources:
            for d in sorted((root_fn(task) / w / TARGET).glob("chain_*")):
                for r in m28.rounds_view(d, info["leaked"]):
                    if r["round"] not in PRIMARY_ROUNDS or not r["rows"]:
                        continue
                    for x in r["rows"]:
                        k = (x["task"]["id"], x["seed"])
                        words = m28.result_in_words(x["reply"], x["task"]["result"])
                        if x["valid"] and k in base and words is not None:
                            fmt["words" if words else "digits"].append(x["peaks"][info["probe"]] - base[k])
        n = len(fmt["words"]) + len(fmt["digits"])
        out[arm] = {"valid_replies": n, "share_words": len(fmt["words"]) / n if n else None,
                    **{f"median_change_{f}": float(np.median(v)) if v else None for f, v in fmt.items()}}
    return out


def spend(task: str, chains: list[dict]) -> dict:
    out = {}
    for fam in FAMILIES.values():
        root = m28.root_of(task, fam.runs)
        calls = sum(json.loads(p.read_text())["calls"] for p in root.glob("judge_spend_*.json"))
        writer = sum(c["cost"] for c in chains if c["source"] == "m31" and c["writer"] in (fam.scores, fam.noscores))
        out[fam.label] = {"writer": writer, "search_judge_calls": calls, "search_judges_estimate": calls * m22.JUDGE_CALL_COST}
    return out


def analyse(task: str) -> dict:
    info = m28.task_info(task, RUNS)
    for root in (m28_root(task), root_of(task, "opus55"), EARLIER_ROOT[task]):  # every arm shares the plain reference and line
        assert (root / "plain_eval.jsonl").read_text() == (root_of(task) / "plain_eval.jsonl").read_text(), root
        assert m22.load_lines(root)[1] == m22.load_lines(root_of(task))[1], root
    chains = chain_records(task, info)
    tests = test_records(task, info, chains)
    by = lambda key: {a: [c[key] for c in chains if c["arm"] == a] for a in ARMS}
    batches = {a: [c["source"] for c in chains if c["arm"] == a] for a in ARMS}
    eval_measures = {k: two_by_two(by(k), batches) for k in ("primary", "primary_carry_forward", "rounds_1_3", "rounds_13_15", "round_0",
                                                               "primary_minus_round_0", "valid_share", "distinct_instructions")}
    for src in ("m28", "m31"):  # batch check for the 4.8 scores effect
        pick = lambda a: [c["primary"] for c in chains if c["arm"] == a and c["source"] == src]
        eval_measures["primary"][f"4.8_scores_effect_{src}_only"] = compare(pick("4.8/scores"), pick("4.8/no scores"))
    test_measures = {}
    for kind in ("last", "best"):
        tk = [t for t in tests if t["kind"] == kind]
        tb = lambda key: {a: [t[key] for t in tk if t["arm"] == a] for a in ARMS}
        test_measures[kind] = {k: two_by_two(tb(k), {a: [t["source"] for t in tk if t["arm"] == a] for a in ARMS})
                               for k in ("change", "median_valid", "under_1pct_share", "valid")}
    gap = info["plain_median"] - info["line"]
    return {"task": task, "plain_eval_median": info["plain_median"], "line": info["line"], "plain_to_line_gap": gap,
            "eval": eval_measures, "test": test_measures, "per_round": per_round(chains), "refusals": refusal_table(chains),
            "spend_m31": spend(task, chains), "answer_format": answer_format(task, info) if task == "maths" else None, "chains": chains, "test_records": tests,
            "plain_test": {f: {"valid": sum(r["valid"] for r in rows), "median_valid": float(np.median([r["peaks"][info["probe"]] for r in rows if r["valid"]]))}
                           for f, rows in ((str(p), [r for r in jl(p) if r["key"] == "plain"]) for _, p in test_files(task)[:3] if p.exists())}}


ARM_COLOUR = {"scores": "#c2410c", "no scores": "#2a78d6"}  # validated pair (dataviz validator: CVD dE 27.9)
MODEL_TITLE = {"4.8": "Opus 4.8, pinned", "5.5": "Opus 5.5, pinned (no fallback)", "mixed": "earlier m22 / m26: Opus 5.5 -> 4.8 fallback"}
MARKER = {"m28": "^", "m31": "o", "earlier": "s"}
TASK_TITLE = {"maths": "Maths", "ordering": "Ordering"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_rounds(results: dict) -> Path:
    """Per round, every chain (thin) and the median over chains (bold), one panel per writer model and task."""
    plt = _plt()
    models = ("4.8", "5.5", "mixed")
    fig, axes = plt.subplots(len(results), 3, figsize=(16, 4.4 * len(results)), sharey="row", squeeze=False)
    for row, (task, res) in enumerate(results.items()):
        for col, model in enumerate(models):
            ax = axes[row][col]
            for c in [c for c in res["chains"] if c["model"] == model]:
                kind = "scores" if c["scores"] else "no scores"
                pts = [(e["round"], e["paired_change"]) for e in c["rounds"] if not e.get("failed")]
                if pts:
                    ax.plot(*zip(*pts), color=ARM_COLOUR[kind], lw=0.8, alpha=0.35, ls="--" if c["source"] == "m28" else "-",
                            marker="." if len(pts) < 4 else None)
            for kind in ("scores", "no scores"):
                arm = f"{model}/{kind}"
                n = len([c for c in res["chains"] if c["arm"] == arm])
                pr = [e for e in res["per_round"][arm] if e["median"] is not None]
                if pr:
                    ax.plot([e["round"] for e in pr], [e["median"] for e in pr], color=ARM_COLOUR[kind], lw=2.4,
                            label=f"{'with' if kind == 'scores' else 'without'} probe scores: median of {n} chains")
            ax.axhline(0, color="#888", lw=0.8)
            ax.axhline(res["line"] - res["plain_eval_median"], color="#555", lw=0.9, ls=":", label="1% line minus plain median (approx.)")
            ax.set_title(f"{TASK_TITLE[task]}: {MODEL_TITLE[model]}", fontsize=10)
            ax.set_xlabel("round (0 = zero-shot)")
            if col == 0:
                ax.set_ylabel("change vs plain, log-odds\n(median over valid replies, paired)")
            ax.legend(fontsize=7.5, loc="lower left")
            ax.grid(alpha=0.2)
    fig.suptitle("Per-round score change on the 80 eval replies, every chain (Opus 4.8: dashed = m28's chains, solid = m31's)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / "rounds_all_chains.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_summary(results: dict) -> Path:
    """Per chain: rounds 1-15 mean (eval), last-round and best-round final-test change; arms in 2x2 order plus the
    earlier mixed arms (reference)."""
    plt = _plt()
    panels = (("eval: mean over rounds 1-15 (primary)", None, "primary"), ("final test: last round", "last", "change"),
              ("final test: best round (picked with the probe)", "best", "change"))
    order = ["4.8/scores", "4.8/no scores", "5.5/scores", "5.5/no scores", "mixed/scores", "mixed/no scores"]
    xs = {a: i + (i // 2) * 0.6 for i, a in enumerate(order)}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(results), 3, figsize=(16, 4.6 * len(results)), squeeze=False)
    for row, (task, res) in enumerate(results.items()):
        for col, (title, kind, key) in enumerate(panels):
            ax = axes[row][col]
            pts = ([(c["arm"], c["source"], c["primary"]) for c in res["chains"]] if kind is None
                   else [(t["arm"], t["source"], t[key]) for t in res["test_records"] if t["kind"] == kind])
            for arm in order:
                vals = [(s, v) for a, s, v in pts if a == arm and v is not None]
                colour = ARM_COLOUR[arm.split("/")[1]]
                for s, v in vals:
                    ax.scatter(xs[arm] + rng.uniform(-0.13, 0.13), v, color=colour, marker=MARKER[s], s=40, edgecolor="white", lw=0.7,
                               alpha=0.5 if arm.startswith("mixed") else 1)
                if vals:
                    ax.hlines(np.mean([v for _, v in vals]), xs[arm] - 0.28, xs[arm] + 0.28, color="#222", lw=1.6)
            ax.axhline(0, color="#888", lw=0.8)
            ax.axhline(res["line"] - res["plain_eval_median"], color="#555", lw=0.9, ls=":")
            ax.set_xticks([xs[a] for a in order], [a.replace("/", "\n") for a in order], fontsize=8)
            ax.set_title(f"{TASK_TITLE[task]}: {title}", fontsize=10)
            if col == 0:
                ax.set_ylabel("log-odds vs plain")
            ax.grid(alpha=0.2, axis="y")
    handles = [plt.Line2D([], [], marker=MARKER[s], ls="", color="#555", label=lab)
               for s, lab in (("m28", "m28 chain"), ("m31", "m31 chain"), ("earlier", "m22 / m26 chain (reference)"))]
    handles.append(plt.Line2D([], [], color="#222", lw=1.6, label="arm mean"))
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIGURES / "scores_by_writer_model.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


def report(tasks: str = ",".join(TASKS)) -> dict:
    results = {t: analyse(t) for t in tasks.split(",")}
    ps = {t: r["eval"]["primary"]["scores_effect"]["4.8"].get("p_two_sided") for t, r in results.items()}
    if all(p is not None for p in ps.values()):
        for t, p in holm(ps).items():
            results[t]["eval"]["primary"]["scores_effect"]["4.8"]["p_holm_over_tasks"] = p
    (OUT / "analysis.json").write_text(json.dumps(results, indent=1, default=float))
    print(fig_rounds(results))
    print(fig_summary(results))
    for t, r in results.items():
        for label, block in (("eval primary", r["eval"]["primary"]), ("test last", r["test"]["last"]["change"])):
            for m in ("4.8", "5.5"):
                e = block["scores_effect"][m]
                print(t, label, m, {k: e.get(k) for k in ("diff", "diff_welch_ci95", "p_two_sided", "p_holm_over_tasks")},
                      {s: e[s]["n"] for s in ("with_scores", "without_scores")})
            print(t, label, "interaction", block["interaction"])
        print(t, "refusals", {a: (v["refusal_share"], v["failed_rounds"]) for a, v in r["refusals"].items()})
    return results


if __name__ == "__main__":
    {"prepare": prepare, "freeze": freeze, "report": report}[sys.argv[1]](*sys.argv[2:])
