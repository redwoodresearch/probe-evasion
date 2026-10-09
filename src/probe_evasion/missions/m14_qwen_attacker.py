"""Mission m14_qwen_attacker: can Qwen3.5-122B-A10B replace Claude as the prompt-writer (gepa's reflection model) in the
sarcasm search?

Everything is m13's score-seeing comparator (m13_blind_search.m13_scoreseeing_hidden, run dir
results/gepa/m13_blind_search_scoreseeing_hidden_sarcasm/) except gepa.reflect, which here sends the identical prompt to
a Qwen model on OpenRouter (one pinned provider, no fallback) instead of Claude:
    target Qwen3.5-27B, reasoning off; probe m3a2 r0_peak@L48, EMA reading, read with the instruction removed;
    gepa.SEEDS, 5 rounds, 4 children per round, minibatches of 8 train tasks (seed = round), 40 val tasks (seed 0),
    gepa's reward and acceptance rule, gepa.REFLECT with m8a's "instruction removed" note.
Round 0 (the two seed prompts on the 40 val tasks) and calibration.json are copied from the comparator's run, since
they do not depend on the proposer; the search's random generator (Random(0)) then picks the same round-1 parents and
minibatches as Claude's run did.

    uv run modal run -n probe-evasion_nabil-baugher_m14-<label>_kill-2026-11-30 \
        -m probe_evasion.missions.m14_qwen_attacker::m14_search --arm q122_think
    uv run python -m probe_evasion.missions.m14_qwen_attacker freeze q122_think
    ... ::m14_test --arms q122_think       # frozen prompt on the 100 test tasks x seeds 10-14, scored
    ... ::m14_twins --arm q122_think       # within-style check (needs missions/m14_qwen_attacker/twin_<arm>.txt)
    uv run python -m probe_evasion.missions.m14_qwen_attacker analyze q122_think
"""

import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from probe_evasion import gepa
from probe_evasion.judge import log_usage
from probe_evasion.generation import read_jsonl as jl, write_jsonl
from probe_evasion.missions import m13_blind_search as m13
from probe_evasion.missions import m3a2_sarcasm_patch as m3a2
from probe_evasion.missions import m8a_flattery as m8a
from probe_evasion.modal_app import app
from probe_evasion.openrouter_gen import _post

MISSION = "m14_qwen_attacker"
OUT = Path("missions") / MISSION
RUNS = Path("results/gepa")
CONCEPT = "sarcasm"
MODEL = "qwen35"
COMPARATOR = m13.HIDDEN_SEARCH_DIR
SCORE_CACHE = OUT / "score_cache.jsonl"
m3a2.SCORE_CACHE = SCORE_CACHE  # m13's import pointed it at m13's cache; new scores go to ours (others are read only)

# arm -> proposer. One provider per model, no fallback; max_tokens as gepa.reflect gives Claude.
PROPOSERS = {
    "q122_think": {"model": "qwen/qwen3.5-122b-a10b", "reasoning": True},
    "q122_think_r2": {"model": "qwen/qwen3.5-122b-a10b", "reasoning": True},  # replicate: only the proposer's samples differ
    "q122_nothink": {"model": "qwen/qwen3.5-122b-a10b", "reasoning": False},
    "q27_think": {"model": "qwen/qwen3.5-27b", "reasoning": True},
    "q122_think_r3": {"model": "qwen/qwen3.5-122b-a10b", "reasoning": True},
    # the model card's recommended thinking-mode sampling for general tasks (owed to design critic 5 once Qwen trailed);
    # min_p=0 (the card's value, a no-op) omitted because Novita does not accept the parameter
    "q122_think_rec": {"model": "qwen/qwen3.5-122b-a10b", "reasoning": True,
                       "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                                    "repetition_penalty": 1.0}},
    # Claude replicates of the comparator: gepa.reflect's exact API call, logged (gepa.reflect itself keeps no record)
    "claude_r2": {"model": gepa.REFLECTION_MODEL, "claude": True},
    "claude_r3": {"model": gepa.REFLECTION_MODEL, "claude": True},
}
PROVIDER = "novita"  # bf16 weights, so the deployment is the full-precision model
MAX_TOKENS = 32000
TRIES = 3  # as gepa.reflect: re-ask when no <instruction> block comes back
API_TRIES = 5  # an error reported inside a 200 response (OpenRouter's mid-generation provider errors) is retried


def run_dir(arm: str) -> Path:
    return RUNS / f"{MISSION}_{arm}_sarcasm"


# ---------------------------------------------------------------- the proposer


def reflect_prompt(model: str, concept: str, instruction: str, results: list[dict], cal: dict) -> str:
    """Exactly the prompt gepa.reflect sends (checked against it by `check_prompt`)."""
    return gepa.REFLECT.format(name=gepa.NAMES[concept], readout=gepa.READOUTS[gepa.TARGET].format(name=gepa.NAMES[concept]),
                               max_words=gepa.MAX_WORDS[concept], threshold=cal[gepa.TARGET]["threshold"],
                               typical=float(np.median(cal[gepa.TARGET]["positive_scores"])), model=model,
                               check=gepa.CHECKS[concept], instruction=instruction, n=len(results),
                               examples="\n".join(gepa.describe(r, cal) for r in results))


def ask_proposer(arm: str, prompt: str) -> dict:
    p = PROPOSERS[arm]
    return ask_claude(prompt) if p.get("claude") else ask_openrouter(p, prompt)


def ask_openrouter(p: dict, prompt: str) -> dict:
    """Arms launched before the 09:50 critic fixes ran without `quantizations`/`require_parameters` (routing checks
    only: Novita serves bf16) and without the API_TRIES loop."""
    load_dotenv(".env")
    body = {"model": p["model"], "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "reasoning": {"enabled": p["reasoning"]}, "usage": {"include": True},
            "provider": {"order": [PROVIDER], "allow_fallbacks": False, "quantizations": ["bf16"], "require_parameters": True},
            **p.get("sampling", {})}
    for api_try in range(API_TRIES):
        start = time.time()
        data = _post(body, os.environ["OPENROUTER_API_KEY"])
        if "error" not in data and "error" not in data["choices"][0]:
            break
        print(f"OpenRouter error (try {api_try}): {str(data.get('error') or data['choices'][0]['error'])[:300]}", flush=True)
        time.sleep(2 ** api_try)
    else:
        raise RuntimeError(f"{p['model']}: errors on {API_TRIES} tries")
    choice = data["choices"][0]
    assert data["provider"].lower() == PROVIDER, data["provider"]
    return {"text": choice["message"].get("content") or "", "reasoning": choice["message"].get("reasoning") or "",
            "finish_reason": choice.get("finish_reason"), "provider": data["provider"], "model": data.get("model"),
            "usage": data.get("usage"), "seconds": round(time.time() - start, 1)}


def ask_claude(prompt: str) -> dict:
    """gepa.reflect's call, unchanged (model, max_tokens, adaptive thinking, one user message), with the reply kept."""
    import anthropic

    load_dotenv(".env")
    client = anthropic.Anthropic(max_retries=10)
    start = time.time()
    with client.messages.stream(model=gepa.REFLECTION_MODEL, max_tokens=32000, thinking={"type": "adaptive"},
                                messages=[{"role": "user", "content": prompt}]) as stream:
        msg = stream.get_final_message()
    log_usage(gepa.REFLECTION_MODEL, msg.usage)
    return {"text": "".join(b.text for b in msg.content if b.type == "text"),
            "reasoning": "\n".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking"),
            "finish_reason": msg.stop_reason, "provider": "anthropic", "model": msg.model, "usage": msg.usage.model_dump(),
            "seconds": round(time.time() - start, 1)}


_LOG_LOCK = threading.Lock()


def logged_reflect(arm: str, rd: Path):
    """A drop-in for gepa.reflect: same prompt and parsing, the proposer chosen by arm. Every attempt is logged."""

    def reflect(model: str, concept: str, instruction: str, results: list[dict], cal: dict) -> str:
        prompt = reflect_prompt(model, concept, instruction, results, cal)
        for attempt in range(TRIES):
            out = ask_proposer(arm, prompt)
            found = re.findall(r"<instruction>(.*?)</instruction>", out["text"], re.S)
            record = {"time": time.strftime("%H:%M:%S"), "arm": arm, "parent_instruction": instruction, "attempt": attempt,
                      "prompt": prompt, **out, "parsed": bool(found), "extracted": found[-1].strip() if found else None,
                      "found_in_reasoning_only": not found and "<instruction>" in out["reasoning"]}
            with _LOG_LOCK, (rd / "proposer_calls.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if found:
                return found[-1].strip()
            print(f"[{arm}] no <instruction> (attempt {attempt}, finish {out['finish_reason']}): {out['text'][-200:]!r}", flush=True)
        # No child: the parent's own instruction comes back, reruns on the same minibatch and seed with the same reward,
        # and is rejected (acceptance needs strictly more). gepa.reflect would crash the search here instead.
        with _LOG_LOCK, (rd / "proposer_calls.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.strftime("%H:%M:%S"), "arm": arm, "no_child": True, "parent_instruction": instruction}) + "\n")
        return instruction

    return reflect


def hidden_reflect_template() -> None:
    """The comparator's change to gepa.REFLECT (m13_scoreseeing_hidden): the proposer is told the instruction is removed."""
    if m8a.HIDDEN_NOTE_NEW not in gepa.REFLECT:
        assert m8a.HIDDEN_NOTE in gepa.REFLECT
        gepa.REFLECT = gepa.REFLECT.replace(m8a.HIDDEN_NOTE, m8a.HIDDEN_NOTE_NEW)


def check_prompt() -> str:
    """Build gepa.reflect's prompt (Anthropic client stubbed to capture it) and ours for the comparator's round-1
    parent results; they must be identical. Returns the prompt."""
    hidden_reflect_template()
    cal = json.loads((COMPARATOR / "calibration.json").read_text())
    rows = [r for r in jl(COMPARATOR / "round_001.jsonl") if r["proposal"] == 0 and r["who"] == "parent"]
    parent = jl(COMPARATOR / "candidates.jsonl")[json.loads((COMPARATOR / "rounds.jsonl").read_text().splitlines()[0])["parent"]]
    captured = {}

    class Stub:
        def __init__(self, **_):
            self.messages = self

        def stream(self, **kw):
            captured.update(kw)
            raise StopIteration

    real = gepa.anthropic.Anthropic
    gepa.anthropic.Anthropic = Stub
    try:
        gepa.reflect(MODEL, CONCEPT, parent["instruction"], rows, cal)
    except StopIteration:
        pass
    finally:
        gepa.anthropic.Anthropic = real
    ours = reflect_prompt(MODEL, CONCEPT, parent["instruction"], rows, cal)
    assert captured["messages"] == [{"role": "user", "content": ours}] and "system" not in captured
    return ours


def smoke(arm: str, provider: str = "") -> None:
    """One real proposer call on the comparator's round-1 proposal-0 prompt (not part of any run)."""
    global PROVIDER
    PROVIDER = provider or PROVIDER
    prompt = check_prompt()
    out = ask_proposer(arm, prompt)
    found = re.findall(r"<instruction>(.*?)</instruction>", out["text"], re.S)
    path = OUT / "smoke_calls.jsonl"
    with path.open("a") as f:
        f.write(json.dumps({"arm": arm, "prompt": prompt, **out, "parsed": bool(found)}) + "\n")
    print(json.dumps({k: out[k] for k in ("finish_reason", "provider", "model", "usage", "seconds")}))
    print(f"reasoning chars {len(out['reasoning'])}; text chars {len(out['text'])}; parsed {bool(found)}")
    print(out["text"][-1500:])


# ---------------------------------------------------------------- the search


def copy_round0(rd: Path) -> None:
    """calibration.json and the two seed candidates (with their val replies) from the comparator's run."""
    shutil.copy(COMPARATOR / "calibration.json", rd / "calibration.json")
    seeds = [c for c in jl(COMPARATOR / "candidates.jsonl") if c["round"] == 0]
    assert [c["instruction"] for c in seeds] == [x.format(name=gepa.NAMES[CONCEPT]) for x in gepa.SEEDS]
    write_jsonl(rd / "candidates.jsonl", seeds)
    for c in seeds:
        shutil.copy(COMPARATOR / f"val_{c['id']:03d}.jsonl", rd / f"val_{c['id']:03d}.jsonl")


@app.local_entrypoint()
def m14_search(arm: str = "q122_think", rounds: int = 5):
    rd = run_dir(arm)
    assert not (rd / "summary.json").exists(), f"{rd} has finished"
    rd.mkdir(parents=True, exist_ok=True)
    if not (rd / "candidates.jsonl").exists():
        copy_round0(rd)
    (rd / "proposer.json").write_text(json.dumps({"arm": arm, **PROPOSERS[arm], "provider": "anthropic" if PROPOSERS[arm].get("claude") else PROVIDER,
                                                  "max_tokens": MAX_TOKENS, "tries": TRIES, "comparator": str(COMPARATOR)}, indent=1))
    hidden_reflect_template()
    gepa.probe_scores = m13.hidden_probe_scores
    gepa.reflect = logged_reflect(arm, rd)
    gepa.search(model=MODEL, concept=CONCEPT, rounds=rounds, out_dir=str(rd), target="ema",
                probe=f"{m3a2.peak_file(0)}:sarcasm@L48", thinking="off")




# ---------------------------------------------------------------- paired per-step comparison (design critic 9)

PAIRED_DIR = RUNS / f"{MISSION}_paired_sarcasm"
PAIRED_PROPOSERS = ("q122_think", "claude_r2")  # the proposer settings; "claude_r2" is just gepa.reflect's Claude call


def comparator_contexts() -> list[dict]:
    """The comparator's 20 reflection contexts: round, proposal, parent instruction, the parent's 8 minibatch results,
    and Claude's original child with its minibatch reward."""
    cands = jl(COMPARATOR / "candidates.jsonl")
    entries = jl(COMPARATOR / "rounds.jsonl")
    out = []
    for rnd in range(1, 6):
        rows = jl(COMPARATOR / f"round_{rnd:03d}.jsonl")
        for i, e in enumerate([e for e in entries if e["round"] == rnd]):
            parent_rows = [r for r in rows if r["proposal"] == i and r["who"] == "parent"]
            assert len(parent_rows) == gepa.MINIBATCH
            out.append({"round": rnd, "proposal": i, "parent": e["parent"], "parent_instruction": cands[e["parent"]]["instruction"],
                        "parent_rows": parent_rows, "parent_minibatch": e["parent_minibatch"], "original_child": e["instruction"],
                        "original_child_minibatch": e["child_minibatch"]})
    return out


@app.local_entrypoint()
def m14_paired(k: int = 3):
    """For each of the comparator's 20 contexts, k fresh children from each proposer (same prompt), each run on the
    context's minibatch (seed = round) and on the 40 val tasks (seed 0), with the search's hidden reading and reward."""
    from concurrent.futures import ThreadPoolExecutor

    PAIRED_DIR.mkdir(parents=True, exist_ok=True)
    hidden_reflect_template()
    gepa.probe_scores = m13.hidden_probe_scores
    gepa.TARGET, gepa.PROBE, gepa.THINKING = "ema", f"{m3a2.peak_file(0)}:sarcasm@L48", False
    cal = json.loads((COMPARATOR / "calibration.json").read_text())
    contexts = comparator_contexts()
    children_path = PAIRED_DIR / "children.jsonl"
    if not children_path.exists():
        jobs = [(c, prop, j) for c in contexts for prop in PAIRED_PROPOSERS for j in range(k)]

        def one(job):
            c, prop, j = job
            child = logged_reflect(prop, PAIRED_DIR)(MODEL, CONCEPT, c["parent_instruction"], c["parent_rows"], cal)
            return {"round": c["round"], "proposal": c["proposal"], "proposer": prop, "sample": j, "instruction": child,
                    "no_child": child == c["parent_instruction"]}

        with ThreadPoolExecutor(16) as pool:
            children = list(pool.map(one, jobs))
        children += [{"round": c["round"], "proposal": c["proposal"], "proposer": "claude_original", "sample": 0,
                      "instruction": c["original_child"], "no_child": False} for c in contexts]
        write_jsonl(children_path, children)
    children = jl(children_path)
    by_ctx = {(c["round"], c["proposal"]): c for c in contexts}
    tasks = gepa.load_tasks(CONCEPT)
    for rnd in range(1, 6):
        path = PAIRED_DIR / f"minibatch_round_{rnd}.jsonl"
        if path.exists():
            continue
        mine = [ch for ch in children if ch["round"] == rnd]
        pairs = [(ch["instruction"], {kk: v for kk, v in r["task"].items()}) for ch in mine for r in by_ctx[(rnd, ch["proposal"])]["parent_rows"]]
        res = gepa.run_many(MODEL, CONCEPT, pairs, seed=rnd)
        write_jsonl(path, [{"child": children.index(ch), **r} for n_, ch in enumerate(mine)
                           for r in res[n_ * gepa.MINIBATCH:(n_ + 1) * gepa.MINIBATCH]])
    val_path = PAIRED_DIR / "val.jsonl"
    if not val_path.exists():
        res = gepa.run_many(MODEL, CONCEPT, [(ch["instruction"], t) for ch in children for t in tasks["val"]])
        n = len(tasks["val"])
        write_jsonl(val_path, [{"child": i, **r} for i in range(len(children)) for r in res[i * n:(i + 1) * n]])
    print(json.dumps(paired_summary(), indent=1))


@app.local_entrypoint()
def m14_paired_extra(proposer: str = "q122_think_rec", k: int = 3):
    """One more proposer in the paired setup: k children per comparator context, run exactly as m14_paired's."""
    from concurrent.futures import ThreadPoolExecutor

    hidden_reflect_template()
    gepa.probe_scores = m13.hidden_probe_scores
    gepa.TARGET, gepa.PROBE, gepa.THINKING = "ema", f"{m3a2.peak_file(0)}:sarcasm@L48", False
    cal = json.loads((COMPARATOR / "calibration.json").read_text())
    contexts = comparator_contexts()
    path = PAIRED_DIR / f"extra_{proposer}.jsonl"
    if not path.exists():
        jobs = [(c, j) for c in contexts for j in range(k)]
        with ThreadPoolExecutor(16) as pool:
            instr = list(pool.map(lambda job: logged_reflect(proposer, PAIRED_DIR)(MODEL, CONCEPT, job[0]["parent_instruction"],
                                                                                   job[0]["parent_rows"], cal), jobs))
        children = [{"round": c["round"], "proposal": c["proposal"], "proposer": proposer, "sample": j, "instruction": x,
                     "no_child": x == c["parent_instruction"]} for (c, j), x in zip(jobs, instr)]
        tasks = gepa.load_tasks(CONCEPT)
        by_ctx = {(c["round"], c["proposal"]): c for c in contexts}
        for rnd in range(1, 6):
            mine = [ch for ch in children if ch["round"] == rnd]
            res = gepa.run_many(MODEL, CONCEPT, [(ch["instruction"], r["task"]) for ch in mine
                                                 for r in by_ctx[(rnd, ch["proposal"])]["parent_rows"]], seed=rnd)
            for n_, ch in enumerate(mine):
                ch["minibatch"] = res[n_ * gepa.MINIBATCH:(n_ + 1) * gepa.MINIBATCH]
        n = len(tasks["val"])
        res = gepa.run_many(MODEL, CONCEPT, [(ch["instruction"], t) for ch in children for t in tasks["val"]])
        for i, ch in enumerate(children):
            ch["val"] = res[i * n:(i + 1) * n]
        write_jsonl(path, children)
    print(json.dumps(paired_extra_summary(proposer), indent=1))


def paired_extra_summary(proposer: str) -> dict:
    cal = json.loads((COMPARATOR / "calibration.json").read_text())
    thr = cal["ema"]["threshold"]
    contexts = {(c["round"], c["proposal"]): c for c in comparator_contexts()}
    extra = jl(PAIRED_DIR / f"extra_{proposer}.jsonl")
    base = jl(PAIRED_DIR / "children.jsonl")
    val = {}
    for r in jl(PAIRED_DIR / "val.jsonl"):
        val.setdefault(r["child"], []).append(r)
    per = {}
    for i, ch in enumerate(base):
        per.setdefault((ch["round"], ch["proposal"]), {}).setdefault(ch["proposer"], []).append(sum(r["present"] and r["ema"] <= thr for r in val[i]))
    for ch in extra:
        per[(ch["round"], ch["proposal"])].setdefault(proposer, []).append(sum(r["present"] and r["ema"] <= thr for r in ch["val"]))
    child_mb = [float(np.mean([gepa.reward(r, cal) for r in ch["minibatch"]])) for ch in extra]
    out = {"n_children": len(extra), "no_child": sum(ch["no_child"] for ch in extra),
           "accepted (child > parent)": int(sum(c > contexts[(ch["round"], ch["proposal"])]["parent_minibatch"] for c, ch in zip(child_mb, extra))),
           "val_reward_mean": float(np.mean([np.mean([gepa.reward(r, cal) for r in ch["val"]]) for ch in extra])),
           "val_evasions_mean_of_40": float(np.mean([v for d in per.values() for v in d[proposer]])),
           "val_valid_mean_of_40": float(np.mean([sum(r["present"] for r in ch["val"]) for ch in extra]))}
    rng = np.random.default_rng(0)
    for other in ("q122_think", "claude_r2"):
        diffs = np.array([np.mean(d[proposer]) - np.mean(d[other]) for d in per.values()])
        boots = [diffs[rng.choice(len(diffs), len(diffs))].mean() for _ in range(2000)]
        out[f"paired {proposer} - {other} val_evasions"] = {"diff": float(diffs.mean()),
                                                             "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    (PAIRED_DIR / f"summary_{proposer}.json").write_text(json.dumps(out, indent=1))
    return out

def paired_summary() -> dict:
    cal = json.loads((COMPARATOR / "calibration.json").read_text())
    children = jl(PAIRED_DIR / "children.jsonl")
    contexts = {(c["round"], c["proposal"]): c for c in comparator_contexts()}
    mb = {}
    for rnd in range(1, 6):
        for r in jl(PAIRED_DIR / f"minibatch_round_{rnd}.jsonl"):
            mb.setdefault(r["child"], []).append(r)
    val = {}
    for r in jl(PAIRED_DIR / "val.jsonl"):
        val.setdefault(r["child"], []).append(r)
    thr = cal["ema"]["threshold"]
    out = {}
    for prop in dict.fromkeys(ch["proposer"] for ch in children):
        ids = [i for i, ch in enumerate(children) if ch["proposer"] == prop]
        child_mb = [float(np.mean([gepa.reward(r, cal) for r in mb[i]])) for i in ids]
        parent_mb = [contexts[(children[i]["round"], children[i]["proposal"])]["parent_minibatch"] for i in ids]
        val_reward = [float(np.mean([gepa.reward(r, cal) for r in val[i]])) for i in ids]
        val_missed = [sum(r["present"] and r["ema"] <= thr for r in val[i]) for i in ids]
        val_present = [sum(r["present"] for r in val[i]) for i in ids]
        out[prop] = {"n_children": len(ids), "no_child": sum(children[i]["no_child"] for i in ids),
                     "minibatch_reward_mean": float(np.mean(child_mb)),
                     "minibatch_gain_over_parent_mean": float(np.mean(np.array(child_mb) - np.array(parent_mb))),
                     "accepted (child > parent)": int(sum(c > p for c, p in zip(child_mb, parent_mb))),
                     "val_reward_mean": float(np.mean(val_reward)), "val_reward_max": float(np.max(val_reward)),
                     "val_evasions_mean_of_40": float(np.mean(val_missed)), "val_evasions_max": int(np.max(val_missed)),
                     "val_valid_mean_of_40": float(np.mean(val_present)),
                     "per_round_val_reward_mean": {rnd: float(np.mean([v for i, v in zip(ids, val_reward) if children[i]["round"] == rnd]))
                                                   for rnd in range(1, 6)}}
    # paired over the 20 contexts: per-context mean of fresh Qwen minus fresh Claude children, contexts resampled
    rng = np.random.default_rng(0)
    for metric in ("val_reward", "val_evasions"):
        per = {}
        for i, ch in enumerate(children):
            v = float(np.mean([gepa.reward(r, cal) for r in val[i]])) if metric == "val_reward" else sum(r["present"] and r["ema"] <= thr for r in val[i])
            per.setdefault((ch["round"], ch["proposal"]), {}).setdefault(ch["proposer"], []).append(v)
        diffs = np.array([np.mean(d["q122_think"]) - np.mean(d["claude_r2"]) for d in per.values()])
        boots = [diffs[rng.choice(len(diffs), len(diffs))].mean() for _ in range(2000)]
        out[f"paired qwen - claude {metric}"] = {"diff": float(diffs.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                                                  "contexts_qwen_better": int((diffs > 0).sum()), "n_contexts": len(diffs)}
    rounds = jl(COMPARATOR / "rounds.jsonl")
    orig = [i for i, ch in enumerate(children) if ch["proposer"] == "claude_original"]
    out["rerun check: claude_original minibatch reward equals the comparator's"] = sum(
        abs(float(np.mean([gepa.reward(r, cal) for r in mb[i]])) - rounds[n]["child_minibatch"]) < 1e-9 for n, i in enumerate(orig))
    (PAIRED_DIR / "summary.json").write_text(json.dumps(out, indent=1))
    return out

# ---------------------------------------------------------------- frozen test (m13's test, reusing its machinery)

REUSED = {  # arm name here -> (file, arm name there); same 100 test tasks x seeds 10-14
    "plain": (m3a2.ATTACKS, "plain"),
    "m3a2_r0": (m3a2.ATTACKS, "r0_search"),
    "claude": (m13.OUT / "test_sarcasm_score_seeing_hidden.jsonl", m13.SCORE_SEEING_HIDDEN),
}
_CACHE: dict = {}


def load_cache(concept: str = CONCEPT) -> dict:
    """m3a2's, m13's and our score caches, first write wins (as m3a2.load_cache), so reused replies keep the scores
    reported in those missions. Installed as m13.load_cache, so m13.scored and m13.lines read it."""
    if not _CACHE:
        for path in (m3a2.OUT / "score_cache.jsonl", m13.OUT / "score_cache.jsonl", SCORE_CACHE):
            if path.exists():
                with path.open() as f:
                    for line in f:
                        x = json.loads(line)
                        _CACHE.setdefault((x["k"], x["p"]), x["r"])
    return _CACHE


m13.load_cache = load_cache


def frozen_path(arm: str) -> Path:
    return OUT / f"frozen_{arm}.json"


def test_path(arm: str) -> Path:
    return OUT / f"test_{arm}.jsonl"


def freeze(arm: str) -> None:
    """The search's best candidate by mean val reward (gepa.test's pick), as the comparator was frozen."""
    summary = json.loads((run_dir(arm) / "summary.json").read_text())
    assert not test_path(arm).exists()
    path = frozen_path(arm)
    assert not path.exists() or json.loads(path.read_text())["instruction"] == summary["instruction"]
    cands = {c["id"]: c for c in jl(run_dir(arm) / "candidates.jsonl")}
    lineage = [summary["best_candidate"]]
    while cands[lineage[-1]]["parent"] is not None:
        lineage.append(cands[lineage[-1]]["parent"])
    path.write_text(json.dumps({"arm": arm, "candidate": summary["best_candidate"], "instruction": summary["instruction"],
                                "val": cands[lineage[0]]["val"], "lineage": lineage,
                                "frozen_at": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print(path.read_text())


def test_rows() -> list[dict]:
    rows = [{**r, "arm": name} for name, (path, there) in REUSED.items() for r in jl(path) if r["arm"] == there]
    rows += [r for path in sorted(OUT.glob("test_*.jsonl")) for r in jl(path)]
    key = lambda arm: sorted((r["task"]["task"], r["seed"]) for r in rows if r["arm"] == arm)
    plain = key("plain")
    assert len(plain) == 500
    for arm in dict.fromkeys(r["arm"] for r in rows):
        assert key(arm) == plain, f"{arm} must use the same tasks and seeds as plain"
    return rows


@app.local_entrypoint()
def m14_test(arms: str = "q122_think"):
    todo = {a: json.loads(frozen_path(a).read_text())["instruction"] for a in arms.split(",") if not test_path(a).exists()}
    if todo:
        rows = m3a2.run_attack(todo, gepa.load_tasks(CONCEPT)["test"], m3a2.TEST_SEEDS)
        for a in todo:
            write_jsonl(test_path(a), [r for r in rows if r["arm"] == a])
    rows = test_rows()
    for how in ("visible", "hidden", "neutral"):
        m13.scored(CONCEPT, rows, how)
    m13.lines(CONCEPT)
    for a in todo:
        rs = [r for r in rows if r["arm"] == a]
        print(f"{a}: {sum(r['present'] for r in rs)}/{len(rs)} valid; scored", flush=True)


def twins_path(arm: str) -> Path:
    return OUT / f"twins_{arm}.jsonl"


@app.local_entrypoint()
def m14_twins(arm: str = "q122_think"):
    """Within-style check, as m13_twins: Qwen's sincere replies (request word "sincere") to the test tasks under a
    hand-written sincere twin of the frozen prompt (OUT/twin_<arm>.txt), 2 seeds, cleaned like calibration replies."""
    path = twins_path(arm)
    if not path.exists():
        instr = (OUT / f"twin_{arm}.txt").read_text().strip()
        tasks = [m13.sincere_task(CONCEPT, t) for t in gepa.load_tasks(CONCEPT)["test"]]
        gepa.THINKING = False
        rows = []
        for seed in m3a2.TEST_SEEDS[:2]:
            prompts = [f"{instr}\n\n{t['task']}" for t in tasks]
            rows += [{"arm": f"twin {arm}", "instruction": instr, "task": t, "seed": seed, "prompt": p, **r}
                     for t, p, r in zip(tasks, prompts, gepa.generate(MODEL, CONCEPT, prompts, seed))]
        rows = [r for r in rows if r["finished"]]
        for r, ch in zip(rows, gepa.check(CONCEPT, [r["task"] for r in rows], rows)):
            r.update(ch)
        m13.clean(CONCEPT, rows)
        write_jsonl(path, rows)
    rows = jl(path)
    for how in ("hidden", "neutral"):
        m13.scored(CONCEPT, rows, how)
    print(f"twins: {sum(r['clean'] for r in rows)}/{len(rows)} clean", flush=True)



@app.local_entrypoint()
def m14_rescore(arms: str = "claude,q122_think", n: int = 100):
    """Scoring-drift check (design critic 7): n replies per arm re-scored fresh in one batch (instruction removed, L48),
    not through the caches, against the cached score each arm's count uses."""
    from probe_evasion import concept_probes

    rows = test_rows()
    rng = np.random.default_rng(0)
    picked = []
    for a in arms.split(","):
        idx = [i for i, r in enumerate(rows) if r["arm"] == a]
        picked += [rows[i] for i in rng.choice(idx, n, replace=False)]
    cached = m13.scored(CONCEPT, picked, "hidden")
    p = m13.primary(CONCEPT)
    fresh = concept_probes.score(MODEL, [{"prompt": r["task"]["task"], "response": r["reply"], "reasoning": None} for r in picked],
                                 {p: m13.probes(CONCEPT)[p]})
    thr = m13.lines(CONCEPT)[p][0.01]
    out = []
    for r, c, f in zip(picked, cached, fresh):
        ema = gepa.readings(f[p], f[f"{p}__logits"])["ema"]
        out.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"]["task"], "present": r["present"],
                    "cached": c[p]["ema"], "fresh": ema, "same_side_of_1pct": (c[p]["ema"] <= thr) == (ema <= thr)})
    write_jsonl(OUT / "rescore_check.jsonl", out)
    for a in arms.split(","):
        xs = [x for x in out if x["arm"] == a]
        lg = lambda v: np.log(v / (1 - v + 1e-12) + 1e-12)
        print(a, f"same side of 1% line {sum(x['same_side_of_1pct'] for x in xs)}/{len(xs)};",
              f"valid evasions cached {sum(x['present'] and x['cached'] <= thr for x in xs)} fresh {sum(x['present'] and x['fresh'] <= thr for x in xs)};",
              f"max |logit diff| {max(abs(lg(x['cached']) - lg(x['fresh'])) for x in xs):.3f}", flush=True)

# ---------------------------------------------------------------- analysis (local; everything already scored)

FULL = lambda r: r["present"] and r["rating"] == 5 and r["blind_rating"] == 5


def analyze(qwen_arms: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score

    rows = test_rows()
    thr = m13.lines(CONCEPT)
    sc = {how: m13.scored(CONCEPT, rows, how) for how in ("visible", "hidden", "neutral")}
    arms = list(dict.fromkeys(["plain", "claude", "m3a2_r0"] + [r["arm"] for r in rows]))
    tasks = sorted({m13.task_key(r) for r in rows})
    report = {"lines": {p: {str(fa): t for fa, t in v.items()} for p, v in thr.items()}, "arms": {}, "gains": {}}
    distinct = lambda ids: len({(m13.task_key(rows[i]), rows[i]["reply"]) for i in ids})
    for a in arms:
        idx = [i for i, r in enumerate(rows) if r["arm"] == a]
        valid = [i for i in idx if rows[i]["present"]]
        full = {i for i in valid if FULL(rows[i])}
        e = {"attempts": len(idx), "valid": len(valid), "valid_distinct": distinct(valid), "full_strength_valid": len(full),
             "mean_blind_rating_valid": float(np.mean([rows[i]["blind_rating"] for i in valid])),
             "mean_words": float(np.mean([len(rows[i]["reply"].split()) for i in idx])), "probes": {}}
        for p in thr:
            for how in ("visible", "hidden", "neutral"):
                for fa in m13.FALSE_ALARMS:
                    ev = [i for i in valid if sc[how][i][p]["ema"] <= thr[p][fa]]
                    e["probes"][f"{p} {how} {fa}"] = {"per_attempt": len(ev), "distinct": distinct(ev),
                                                     "full_strength": sum(i in full for i in ev)}
        report["arms"][a] = e
    contrasts = [(a, "plain") for a in arms if a != "plain"] + [(q, b) for q in qwen_arms for b in ("claude", "m3a2_r0")]
    for p in thr:
        for how in ("visible", "hidden", "neutral"):
            for fa in m13.FALSE_ALARMS:
                for strength in ("all", "full"):
                    hits, n = {a: {} for a in arms}, {a: {} for a in arms}
                    for i, r in enumerate(rows):
                        t = m13.task_key(r)
                        n[r["arm"]][t] = n[r["arm"]].get(t, 0) + 1
                        ok = r["present"] and (strength == "all" or FULL(r))
                        hits[r["arm"]][t] = hits[r["arm"]].get(t, 0) + int(ok and sc[how][i][p]["ema"] <= thr[p][fa])
                    rate = lambda a: sum(hits[a].values()) / sum(n[a].values())
                    for a, b in contrasts:
                        report["gains"][f"{a} - {b} | {p} {how} {fa} {strength}"] = {
                            "diff": rate(a) - rate(b), "ci95": m13.diff_ci(hits, n, a, b, tasks)}
    rng = np.random.default_rng(0)
    by = {(a, t): [r["blind_rating"] for r in rows if r["arm"] == a and m13.task_key(r) == t and r["present"]] for a in arms for t in tasks}
    mean = lambda a, ts: np.mean([x for t in ts for x in by[(a, t)]] or [np.nan])
    picks = [rng.choice(len(tasks), len(tasks)) for _ in range(2000)]
    for a in arms[1:]:
        boots = [mean(a, [tasks[i] for i in pick]) - mean("plain", [tasks[i] for i in pick]) for pick in picks]
        report["arms"][a]["blind_rating_vs_plain"] = {"diff": float(mean(a, tasks) - mean("plain", tasks)),
                                                      "ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]}
    p = m13.primary(CONCEPT)
    for a in arms:
        valid = [i for i, r in enumerate(rows) if r["arm"] == a and r["present"]]
        ev = [i for i in valid if sc["hidden"][i][p]["ema"] <= thr[p][0.01]]
        per_task = {}
        for i in ev:
            per_task[m13.task_key(rows[i])] = per_task.get(m13.task_key(rows[i]), 0) + 1
        full = [i for i in valid if FULL(rows[i])]
        report["arms"][a]["primary_hidden_1pct_detail"] = {
            "evasion_rate_full_strength": f"{sum(i in set(ev) for i in full)}/{len(full)}",
            "evasion_rate_other_valid": f"{sum(i in set(ev) for i in valid if i not in set(full))}/{len(valid) - len(full)}",
            "tasks_with_evasion": len(per_task), "top10_tasks_share": sum(sorted(per_task.values())[-10:]) / max(1, len(ev))}
    gain = lambda a: (report["arms"][a]["probes"][f"{p} hidden 0.01"]["per_attempt"]
                      - report["arms"]["plain"]["probes"][f"{p} hidden 0.01"]["per_attempt"])
    report["ratio_to_claude_gain"] = {q: gain(q) / gain("claude") for q in qwen_arms}
    report["within_style"] = {}
    for q in arms:
        if not twins_path(q).exists():
            continue
        twins = [r for r in jl(twins_path(q)) if r["clean"]]
        tw = {how: m13.scored(CONCEPT, twins, how) for how in ("hidden", "neutral")}
        w = {"n_twins_clean": len(twins), "n_twins": len(jl(twins_path(q)))}
        pos = [i for i, r in enumerate(rows) if r["arm"] == q and r["present"]]
        for how in ("hidden", "neutral"):
            for pr in thr:
                xs = [sc[how][i][pr]["ema"] for i in pos] + [s_[pr]["ema"] for s_ in tw[how]]
                w[f"{pr} {how}"] = {"auroc": float(roc_auc_score([1] * len(pos) + [0] * len(tw[how]), xs)),
                                    "twins_flagged@1%": sum(s_[pr]["ema"] > thr[pr][0.01] for s_ in tw[how])}
        report["within_style"][q] = w
    (OUT / "analysis.json").write_text(json.dumps(report, indent=1))
    return report


def print_table(report: dict) -> None:
    for a, e in report["arms"].items():
        print(f"{a:13s} valid {e['valid']}/{e['attempts']} (distinct {e['valid_distinct']}; both judges 5: {e['full_strength_valid']}) "
              f"blind rating {e['mean_blind_rating_valid']:.2f} {e.get('blind_rating_vs_plain', '')}")
        for p in report["lines"]:
            for how in ("visible", "hidden", "neutral"):
                cells = [e["probes"][f"{p} {how} {fa}"] for fa in m13.FALSE_ALARMS]
                print(f"   {p} {how:7s} 0.1/1/5%: " + " / ".join(f"{x['per_attempt']} ({x['distinct']}d, {x['full_strength']}f)" for x in cells))
        print("   ", e["primary_hidden_1pct_detail"])
    for k, v in report["gains"].items():
        if " 0.01 " in k and " neutral " not in k:
            print(f"{k}: {100 * v['diff']:+.1f} pts [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]")
    print("ratio to Claude's gain:", report["ratio_to_claude_gain"])
    for k, v in report["within_style"].items():
        print(k, v)



QWEN_RUNS = ("q122_think", "q122_think_r2", "q122_think_r3")
CLAUDE_RUNS = ("claude", "claude_r2", "claude_r3")


def pooled(n_boot: int = 2000) -> dict:
    """The pooled 3-runs-per-proposer comparison fixed in STATUS at 10:18: mean over runs of per-attempt rates; tasks
    resampled (the same draw for every arm) and runs resampled within proposer."""
    rows = test_rows()
    thr = m13.lines(CONCEPT)
    sc = {how: m13.scored(CONCEPT, rows, how) for how in ("hidden", "neutral")}
    tasks = sorted({m13.task_key(r) for r in rows})
    t_index = {t: j for j, t in enumerate(tasks)}
    arms = ("plain",) + QWEN_RUNS + CLAUDE_RUNS
    out = {}
    for p, how, fa, variant in [(p, how, fa, v) for p in thr for how in ("hidden", "neutral") for fa in m13.FALSE_ALARMS
                                for v in ("all", "distinct", "full")]:
        hits = {a: np.zeros(len(tasks)) for a in arms}
        n = {a: np.zeros(len(tasks)) for a in arms}
        seen = set()
        for i, r in enumerate(rows):
            if r["arm"] not in arms:
                continue
            j = t_index[m13.task_key(r)]
            n[r["arm"]][j] += 1
            ok = r["present"] and sc[how][i][p]["ema"] <= thr[p][fa] and (variant != "full" or FULL(r))
            if ok and variant == "distinct":
                ok = (r["arm"], m13.task_key(r), r["reply"]) not in seen
                seen.add((r["arm"], m13.task_key(r), r["reply"]))
            hits[r["arm"]][j] += ok
        rate = lambda a, pick: hits[a][pick].sum() / n[a][pick].sum()
        everything = np.arange(len(tasks))
        point = {a: float(rate(a, everything)) for a in arms}
        q = float(np.mean([point[a] for a in QWEN_RUNS]))
        c = float(np.mean([point[a] for a in CLAUDE_RUNS]))
        rng = np.random.default_rng(0)
        ratios, diffs = [], []
        for _ in range(n_boot):
            pick = rng.choice(len(tasks), len(tasks))
            qs = [QWEN_RUNS[k] for k in rng.choice(3, 3)]
            cs = [CLAUDE_RUNS[k] for k in rng.choice(3, 3)]
            qm, cm, pl = np.mean([rate(a, pick) for a in qs]), np.mean([rate(a, pick) for a in cs]), rate("plain", pick)
            ratios.append((qm - pl) / (cm - pl))
            diffs.append(qm - cm)
        ci = lambda xs: [float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975))]
        out[f"{p} {how} {fa:.1%} {variant}"] = {
            "per_run_of_500": {a: round(point[a] * 500) for a in arms}, "qwen_mean": q * 500, "claude_mean": c * 500,
            "ratio": (q - point["plain"]) / (c - point["plain"]), "ratio_ci95": ci(ratios),
            "qwen_minus_claude_pts": 100 * (q - c), "diff_ci95_pts": [100 * x for x in ci(diffs)],
            "qwen_mean_inside_claude_run_range": min(point[a] for a in CLAUDE_RUNS) <= q <= max(point[a] for a in CLAUDE_RUNS)}
    rating = lambda runs: float(np.mean([np.mean([r["blind_rating"] for r in rows if r["arm"] == a and r["present"]]) for a in runs]))
    out["mean_blind_rating_valid"] = {"qwen": rating(QWEN_RUNS), "claude": rating(CLAUDE_RUNS), "plain": rating(("plain",))}
    (OUT / "pooled.json").write_text(json.dumps(out, indent=1))
    return out


def tradeoff_plot() -> Path:
    """One point per run: share of attempts that are full-strength valid (both judges 5) vs evasions per valid reply."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads((OUT / "analysis.json").read_text())
    key = f"{m13.primary(CONCEPT)} hidden 0.01"
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=130)
    for a, e in rep["arms"].items():
        colour = "#c2410c" if a.startswith("claude") or a == "m3a2_r0" else "#888888" if a == "plain" else "#1d4ed8"
        x = e["full_strength_valid"] / e["attempts"]
        y = e["probes"][key]["per_attempt"] / e["valid"]
        ax.scatter(x, y, color=colour, s=30)
        ax.annotate(f"{a} ({e['probes'][key]['per_attempt']})", (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("full-strength valid replies per attempt (both judges rate 5)")
    ax.set_ylabel("evasions per valid reply\n(L48, instruction removed, 1% line)")
    ax.set_title("Strength vs evasion per run. Orange: Claude proposer; blue: Qwen; grey: plain.\n"
                 "In brackets: valid evasions of 500. The search reward gives full credit to 4-rated replies.", fontsize=7, loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "tradeoff.png"
    fig.savefig(path)
    return path


def dot_plot() -> Path:
    """Every test reply's L48 score with the instruction removed, per arm (valid blue, invalid orange), next to the
    clean calibration replies (grey), with the 0.1/1/5% lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = m13.primary(CONCEPT)
    thr = m13.lines(CONCEPT)[p]
    rows = test_rows()
    get = lambda rs: [s[p]["ema"] for s in m13.scored(CONCEPT, rs, "hidden")]
    groups = {"calibration\n(clean)": [s[p]["ema"] for s in m13.scored(CONCEPT, m13.calibration_rows(CONCEPT), "visible")]}
    for a in ("plain",) + CLAUDE_RUNS + QWEN_RUNS + ("q122_think_rec", "q122_nothink", "q27_think", "m3a2_r0"):
        if any(r["arm"] == a for r in rows):
            groups[f"{a}\nvalid"] = get([r for r in rows if r["arm"] == a and r["present"]])
            groups[f"{a}\ninvalid"] = get([r for r in rows if r["arm"] == a and not r["present"]])
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(groups)), 6), dpi=130)
    rng = np.random.default_rng(0)
    for i, (label, xs) in enumerate(groups.items()):
        grey = "calibration" in label
        colour = "#888888" if grey else "#c2410c" if "invalid" in label else "#1d4ed8"
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(xs)), np.clip(xs, 1e-6, 1 - 1e-6), s=5, alpha=0.5, color=colour, linewidths=0)
        if "invalid" not in label:
            k = sum(x > thr[0.01] for x in xs) if grey else sum(x <= thr[0.01] for x in xs)
            ax.text(i, 2.5e-3, f"{k}/{len(xs)}\n{'flagged' if grey else 'below'}", ha="center", fontsize=5)
    ax.set_yscale("logit")
    ax.set_ylim(2e-3, 1 - 1e-6)
    for fa, ls in zip(m13.FALSE_ALARMS, (":", "-", "--")):
        ax.axhline(thr[fa], color="k", ls=ls, lw=1)
        ax.text(len(groups) - 0.5, thr[fa], f"{fa:.1%} line", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(range(len(groups)), list(groups), fontsize=5, rotation=90)
    ax.set_ylabel("probe score (EMA reading), log-odds scale")
    ax.set_title(f"sarcasm, {p}, instruction removed. Grey: clean sincere calibration replies (count flagged at 1%). Blue: valid "
                 "sarcastic test replies (count below the 1% line). Orange: invalid.", fontsize=7, loc="left")
    fig.tight_layout()
    path = OUT / "dots_L48.png"
    fig.savefig(path)
    return path

# ---------------------------------------------------------------- per-round progress


def round_progress(rd: Path) -> list[dict]:
    """After each round: the best candidate so far by mean val reward (the search's own criterion), and the most hidden
    1% evasions on val among candidates so far (single-judge validity, 40 val tasks)."""
    cands = jl(rd / "candidates.jsonl")
    out = []
    for rnd in range(0, 6):
        so_far = [c for c in cands if c["round"] <= rnd]
        best = max(so_far, key=lambda c: np.mean(c["val_rewards"]))
        out.append({"round": rnd, "best_id": best["id"], "best_reward": float(np.mean(best["val_rewards"])),
                    "best_missed": best["val"]["missed_ema"], "best_present": best["val"]["present"],
                    "max_missed": max(c["val"]["missed_ema"] for c in so_far), "n_candidates": len(so_far),
                    "accepted_this_round": sum(c["round"] == rnd for c in cands)})
    return out


def plot_rounds(arms: list[str]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {"claude (m13 comparator)": COMPARATOR} | {a: run_dir(a) for a in arms}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=130)
    progress = {}
    for label, rd in runs.items():
        prog = progress[label] = round_progress(rd)
        xs = [p["round"] for p in prog]
        claude = label.startswith("claude")
        style = dict(color="#c2410c" if claude else "#1d4ed8" if label.startswith("q122_think") else "#0f766e",
                     ls="-" if claude or label.startswith("q122_think") else "--", marker="o", ms=3, alpha=0.8)
        axes[0].plot(xs, [p["best_reward"] for p in prog], label=f"{label} (val runs per round: "
                     + ",".join(str(p["accepted_this_round"]) for p in prog[1:]) + ")", **style)
        axes[1].plot(xs, [p["best_missed"] for p in prog], **style)
    axes[0].set_ylabel("best mean val reward so far")
    axes[1].set_ylabel("val evasions (of 40) of that best candidate")
    for ax in axes:
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Search progress: best candidate by validation reward after each round (instruction removed, 1% line). Orange: "
                 "Claude proposer; blue: Qwen 122B reasoning on; green dashed: other Qwen arms.\nThis describes search behaviour: "
                 "the best-so-far is a max over noisy 40-task estimates, and arms with more accepted children get more draws.", fontsize=7)
    fig.tight_layout()
    path = OUT / "rounds_progress.png"
    fig.savefig(path)
    (OUT / "rounds_progress.json").write_text(json.dumps(progress, indent=1))
    return path


OPUS_PRICE = (5.0 / 1e6, 25.0 / 1e6)  # $ per input / output token; status_page.LOGGED_PRICES' estimate


def proposer_stats(arm: str) -> dict:
    """Parse failures, refusals-as-no-child, tokens and cost. OpenRouter reports cost; Claude's is estimated."""
    log = jl(run_dir(arm) / "proposer_calls.jsonl")
    calls = [c for c in log if not c.get("no_child")]
    usage = [c["usage"] or {} for c in calls]
    if PROPOSERS[arm].get("claude"):
        prompt_tokens = sum(u["input_tokens"] for u in usage)
        completion_tokens = sum(u["output_tokens"] for u in usage)
        cost = prompt_tokens * OPUS_PRICE[0] + completion_tokens * OPUS_PRICE[1]
    else:
        prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
        completion_tokens = sum(u.get("completion_tokens", 0) for u in usage)
        cost = sum(u.get("cost", 0) for u in usage)
    return {"calls": len(calls), "parsed": sum(c["parsed"] for c in calls), "parse_failures": sum(not c["parsed"] for c in calls),
            "no_child_after_3_failures": sum(bool(c.get("no_child")) for c in log),
            "finish_reasons": {str(f): sum(c["finish_reason"] == f for c in calls) for f in {c["finish_reason"] for c in calls}},
            "instruction_only_in_reasoning": sum(c["found_in_reasoning_only"] for c in calls),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage),
            "cost_usd": round(cost, 3), "mean_seconds": float(np.mean([c["seconds"] for c in calls])),
            "mean_reasoning_chars": float(np.mean([len(c["reasoning"]) for c in calls])),
            "instruction_words_median": float(np.median([len(c["extracted"].split()) for c in calls if c["parsed"]]))}

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke":
        smoke(sys.argv[2], *sys.argv[3:4])
    elif cmd == "check_prompt":
        print(check_prompt()[:3000])
    elif cmd == "freeze":
        freeze(sys.argv[2])
    elif cmd == "progress":
        print(plot_rounds(sys.argv[2].split(",")))
        print(json.dumps(json.loads((OUT / "rounds_progress.json").read_text()), indent=1))
    elif cmd == "stats":
        for a in sys.argv[2].split(","):
            print(a, json.dumps(proposer_stats(a)))
    elif cmd == "analyze":
        rep = analyze(sys.argv[2].split(","))
        print_table(rep)
        print(tradeoff_plot())
    elif cmd == "dots":
        print(dot_plot())
    elif cmd == "pooled":
        print(json.dumps(pooled(), indent=1))
    elif cmd == "paired":
        print(json.dumps(paired_summary(), indent=1))
