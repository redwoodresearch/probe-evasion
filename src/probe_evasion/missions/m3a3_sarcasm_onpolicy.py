"""Mission m3a3_sarcasm_onpolicy: does m3a2's round-0 result (a Claude search beats a peak-trained sarcasm probe) hold
when the probe is ON-policy, trained on Qwen3.5-27B's own sarcastic and sincere replies?

    probes     on_peak@L16 / on_peak@L48: m3a's on-policy rows (m3a_sarcasm_train_aug), BCE on the EMA peak, seed 0
               onsym_peak@L16/L48: the same replies with the symmetric request-word augmentation (sensitivity check)
               on_gd@L48: m3a's mean-trained on-policy probe (reference; does not count under the readout rule)
               r0_peak@L48 etc.: m3a2's off-policy round-0 probes, for the comparison
    lines      m3a's clean calibration set, recomputed per probe
    attacks    every m3a2 frozen prompt (re-scored from m3a2's saved replies) and one fresh search against on_peak@L48

m3a2's evaluation code is reused read-only. Its functions read module globals at call time, so this module points
m3a2's write paths (score cache, output folder) at this mission's folder for this process only (see `_redirect_m3a2`);
m3a2's files are only read.

    uv run modal run -m probe_evasion.missions.m3a3_sarcasm_onpolicy::m3a3_train
    uv run modal run -m probe_evasion.missions.m3a3_sarcasm_onpolicy::m3a3_monitors
    uv run modal run -m probe_evasion.missions.m3a3_sarcasm_onpolicy::m3a3_search
    uv run modal run -m probe_evasion.missions.m3a3_sarcasm_onpolicy::m3a3_frozen_test
    uv run modal run -m probe_evasion.missions.m3a3_sarcasm_onpolicy::m3a3_flat_sincere   # needs sincere_twin_on_search.txt
    uv run python -m probe_evasion.missions.m3a3_sarcasm_onpolicy analyze
"""

import json
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from probe_evasion import concept_probes, gepa
from probe_evasion.generation import read_jsonl as jl, write_jsonl
from probe_evasion.missions import m3a2_sarcasm_patch as m3a2
from probe_evasion.missions import m3a_sarcasm as m3a
from probe_evasion.missions.m1a_peak_probe import PeakProbeLabSeeds
from probe_evasion.modal_app import app

MISSION = "m3a3_sarcasm_onpolicy"
MODEL = m3a.MODEL
OUT = Path("experimental/missions") / MISSION
SCORE_CACHE = OUT / "score_cache.jsonl"
M3A2_OUT, M3A2_SCORE_CACHE = m3a2.OUT, m3a2.SCORE_CACHE
LAYERS = (16, 48)
PRIMARY = "on_peak@L48"  # pre-registered, conditional on the quality bar (STATUS.md)
PEAK_FILE = f"{MISSION}_peak"  # m3a's train_aug rows (asymmetric request-word augmentation), the spec'd primary
SYM_PEAK_FILE = f"{MISSION}_sym_peak"  # same replies, symmetric augmentation (sensitivity check)
SEEDS_FILE = f"{MISSION}_peak_seeds"  # train_aug rows, L48, seeds 1 and 2 (critic 2: is seed 0 typical?)
SYM_ROWS = m3a.DATA / f"{MISSION}_train_sym.jsonl"
SEARCH_DIR = m3a2.RUNS / f"{MISSION}_search"
ARM = "on_search"  # the fresh search's frozen prompt
FRESH = OUT / f"fresh_{ARM}.jsonl"  # its judged replies on the 100 test tasks x seeds 10-14
FALSE_ALARMS = m3a.FALSE_ALARMS
NEUTRAL_CHECK_PROBES = ("on_peak@L48", "onsym_peak@L48", "r0_peak@L48")
MONITORS = {
    "on-policy peak L48 (primary)": ("on_peak@L48",),
    "on-policy peak L48, seed 1": ("on_peak@L48s1",),
    "on-policy peak L48, seed 2": ("on_peak@L48s2",),
    "on-policy peak L16": ("on_peak@L16",),
    "on-policy peak L48, symmetric request word": ("onsym_peak@L48",),
    "on-policy peak L16, symmetric request word": ("onsym_peak@L16",),
    "on-policy gd L48 (m3a, mean-trained; reference)": ("on_gd@L48",),
    "off-policy r0 peak L48 (m3a2 round-0 primary)": ("r0_peak@L48",),
    "off-policy r0 peak L16": ("r0_peak@L16",),
    "off-policy r0 gd L16 (m3a main; mean-trained)": ("r0_gd@L16",),
}


# ---------------------------------------------------------------- reuse of m3a2's code, with writes redirected here


def load_cache() -> dict:
    """m3a2's score cache (read only) merged with this mission's; first write wins, so replies m3a2 already scored keep
    m3a2's values exactly (re-scoring jitters by up to ~0.3 logit)."""
    cache = {}
    for path in (M3A2_SCORE_CACHE, SCORE_CACHE):
        if path.exists():
            with path.open() as f:
                for line in f:
                    x = json.loads(line)
                    cache.setdefault((x["k"], x["p"]), x["r"])
    return cache


_m3a2_eval_sets, _m3a2_all_probes = m3a2.eval_sets, m3a2.all_probes


@contextmanager
def _reading_m3a2_folder():
    """m3a2.eval_sets globs m3a2's folder for its attack and flat-sincere files."""
    m3a2.OUT = M3A2_OUT
    try:
        yield
    finally:
        m3a2.OUT = OUT


NEUTRAL = False  # True: every request is scored with the word sarcastic / sincere removed (critic 1)


def without_request_word(prompt: str) -> str:
    """The prompt with ", sarcastic" / " sincere" removed from its last block (the request's instruction), if present."""
    head, sep, instruction = prompt.rpartition("\n\n")
    return head + sep + re.sub(r",? (sarcastic|sincere)\b", "", instruction)


def own_flat_rows(kind: str = "r") -> list[dict]:
    """This mission's flat sincere sets (OUT/flat_sincere_<kind>_<arm>.jsonl): the fresh prompt's sincere twin, blunt."""
    return [r for path in sorted(OUT.glob(f"flat_sincere_{kind}_*.jsonl")) for r in jl(path)]


def eval_sets() -> dict[str, list[dict]]:
    """m3a2's sets (calibration, held-out, plain answers, flat sincere sets, every m3a2 attack) plus this mission's
    fresh prompt's replies and flat sincere sets. NEUTRAL: every prompt loses the request word."""
    with _reading_m3a2_folder():
        sets = _m3a2_eval_sets()
    if FRESH.exists():
        fresh = jl(FRESH)
        sets["attacks"] += [dict(r) for r in fresh]
        sets["attacks_removed"] += [{**r, "prompt": r["task"]["task"]} for r in fresh]
    for prefix, kind in (("flat", "r"), ("flatall", "all")):
        for r in own_flat_rows(kind):
            sets.setdefault(f"{prefix}: {r['arm']}", []).append({"prompt": r["prompt"], "reply": r["reply"]})
    if NEUTRAL:
        sets = {k: [{**r, "prompt": without_request_word(r["prompt"])} for r in rows] for k, rows in sets.items()}
    return sets


def own_probes() -> dict:
    out = {}
    for short, file in (("on_peak", PEAK_FILE), ("onsym_peak", SYM_PEAK_FILE)):
        path = concept_probes.PROBES / f"{file}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            out |= {f"{short}@L{l}": saved[f"sarcasm@L{l}"] for l in LAYERS}
    seeds = concept_probes.PROBES / f"{SEEDS_FILE}.json"
    if seeds.exists():
        out |= {f"on_peak@L48s{p['seed']}": p for p in json.loads(seeds.read_text()).values()}
    gd = json.loads((concept_probes.PROBES / f"{m3a.PROBE_FILES['on_gd']}.json").read_text())
    return out | {"on_gd@L48": gd["sarcasm@L48"]}


def all_probes() -> dict:
    return _m3a2_all_probes() | own_probes()


def _redirect_m3a2() -> None:
    """For this process only: m3a2's functions write their score cache and outputs (flat sincere sets, dot plots) here,
    read both score caches, see this mission's probes and replies, and calibrate a search against PRIMARY."""
    m3a2.OUT, m3a2.SCORE_CACHE = OUT, SCORE_CACHE
    m3a2.load_cache, m3a2.eval_sets, m3a2.all_probes = load_cache, eval_sets, all_probes
    m3a2.primary = lambda rnd: PRIMARY


_redirect_m3a2()


# ---------------------------------------------------------------- probes


def symmetric_rows() -> list[dict]:
    if not SYM_ROWS.exists():
        write_jsonl(SYM_ROWS, m3a.augmented(jl(m3a.FILES["train"]), random.Random(7), symmetric=True))
    return jl(SYM_ROWS)


@app.local_entrypoint()
def m3a3_train():
    """Peak-trained (BCE on the EMA peak, val-loss stopping, seed 0) probes at L16 and L48 on m3a's on-policy rows, and
    on the same replies with the symmetric request-word augmentation."""
    lab = PeakProbeLabSeeds()
    jobs = {file: rows for file, rows in ((PEAK_FILE, jl(m3a.FILES["train_aug"])), (SYM_PEAK_FILE, symmetric_rows()))
            if not (concept_probes.PROBES / f"{file}.json").exists()}
    calls = {(file, l): lab.train_peak_seeded.spawn(rows, l, (0,)) for file, rows in jobs.items() for l in LAYERS}
    for file in jobs:
        probes = {f"sarcasm@L{l}": next(iter(calls[(file, l)].get().values())) | {"rows": str(
            m3a.FILES["train_aug"] if file == PEAK_FILE else SYM_ROWS)} for l in LAYERS}
        (concept_probes.PROBES / f"{file}.json").write_text(json.dumps(probes))
        print(file, {k: (round(v["val_loss"], 4), round(v["val_auroc"], 4), v["epoch"]) for k, v in probes.items()}, flush=True)


@app.local_entrypoint()
def m3a3_seeds():
    """The primary recipe (train_aug rows, peak, L48) with seeds 1 and 2: validation split and batch order change."""
    out = PeakProbeLabSeeds().train_peak_seeded.remote(jl(m3a.FILES["train_aug"]), 48, (1, 2))
    (concept_probes.PROBES / f"{SEEDS_FILE}.json").write_text(json.dumps(out))
    print({k: (round(v["val_loss"], 4), round(v["val_auroc"], 4), v["epoch"]) for k, v in out.items()}, flush=True)


# ---------------------------------------------------------------- evaluation


@app.local_entrypoint()
def m3a3_monitors(neutral: bool = False):
    """Quality gates and every attack arm (m3a2's frozen prompts, plain, and the fresh prompt once it exists) under each
    probe: per attempt and distinct, as generated and instruction removed, at 0.1/1/5%. neutral: every request (also
    calibration's) scored with the word sarcastic / sincere removed."""
    global NEUTRAL
    NEUTRAL = neutral
    table = m3a2.monitor_table({k: v for k, v in MONITORS.items() if not neutral or v[0] in NEUTRAL_CHECK_PROBES})
    (OUT / f"monitors{'_neutral' if neutral else ''}.json").write_text(json.dumps(table, indent=1))
    m3a2.print_monitor_table(table)


# ---------------------------------------------------------------- the fresh search and its frozen test


@app.local_entrypoint()
def m3a3_search(rounds: int = 5):
    """A fresh 5-round Claude search (no bank seeds) against PRIMARY, on the search train/val tasks, with calibration
    from the mission's clean calibration set (as m3a2's round-0 search)."""
    if not (SEARCH_DIR / "calibration.json").exists():
        m3a2.write_search_calibration(SEARCH_DIR, "unused")
    assert json.loads((SEARCH_DIR / "calibration.json").read_text())["source"].endswith(PRIMARY)  # critic 1: not stale
    probe, layer = PRIMARY.split("@")
    file = {"on_peak": PEAK_FILE, "onsym_peak": SYM_PEAK_FILE}[probe]
    gepa.search(model=MODEL, concept="sarcasm", rounds=rounds, out_dir=str(SEARCH_DIR), target="ema",
                probe=f"{file}:sarcasm@{layer}", thinking="off")


@app.local_entrypoint()
def m3a3_frozen_test():
    """The search's best prompt, frozen, on the 100 untouched test tasks x 5 samples (seeds 10-14, as m3a2's plain arm)."""
    instruction = json.loads((SEARCH_DIR / "summary.json").read_text())["instruction"]
    frozen = OUT / "frozen_prompt.json"
    if frozen.exists():
        assert json.loads(frozen.read_text())[ARM] == instruction
    frozen.write_text(json.dumps({ARM: instruction}, indent=1))
    assert not FRESH.exists()
    rows = m3a2.run_attack({ARM: instruction}, gepa.load_tasks("sarcasm")["test"], m3a2.TEST_SEEDS)
    write_jsonl(FRESH, rows)
    print(f"{ARM}: {sum(r['present'] for r in rows)}/{len(rows)} valid", flush=True)


@app.local_entrypoint()
def m3a3_flat_sincere(n: int = 200):
    """Sincere Qwen replies in the fresh prompt's style (its sincere twin, written by hand right after freezing), for the
    false-alarm gate and the within-style check."""
    assert (OUT / f"sincere_twin_{ARM}.txt").exists()
    m3a2.flat_sincere([ARM], n)


@app.local_entrypoint()
def m3a3_blunt(n: int = 200):
    """Critic 1: the on-policy training negatives are mostly warm. Qwen's blunt-but-sincere replies (cleaned), as a
    false-alarm set a deployed monitor would also see."""
    assert (OUT / "sincere_twin_blunt.txt").exists()
    m3a2.flat_sincere(["blunt"], n)



CONTEXT_ARMS = (ARM, "m3a_searched", "r0_search", "r1_search")


def instruction_and_twin(arm: str) -> tuple[str, str]:
    if arm == ARM:
        return json.loads((OUT / "frozen_prompt.json").read_text())[ARM], (OUT / f"sincere_twin_{ARM}.txt").read_text().strip()
    twin = m3a.SINCERE_STYLES["one_sentence"] if arm == "m3a_searched" else (M3A2_OUT / f"sincere_twin_{arm}.txt").read_text().strip()
    return json.loads(m3a2.FROZEN.read_text())[arm], twin


@app.local_entrypoint()
def m3a3_context_check():
    """Instruction text alone (critic 2: also for r0 and r1 search). For each searched prompt: (a) plain valid replies
    re-scored behind its instruction and behind its sincere twin; (b) its own valid replies behind its instruction, its
    sincere twin, a neutral preamble of similar length, and nothing. Per attempt (repeats counted). Lines from the clean
    calibration (logit quantiles, as monitors.json)."""
    sets = eval_sets()
    probes = {p: all_probes()[p] for p in CONTEXT_PROBES}
    cache = load_cache()
    cal = m3a2.score([(r["prompt"], r["reply"]) for r in sets["calibration"] if r["clean"]], probes, cache)
    line = {p: logit_lines([s[p]["ema"] for s in cal])[0.01] for p in probes}
    plain = [r for r in sets["attacks"] if r["arm"] == "plain" and r["present"]]
    out = {}

    def count(name: str, rows: list[dict], pre: str) -> None:
        s = m3a2.score([(f"{pre}\n\n{r['task']['task']}" if pre else r["task"]["task"], r["reply"]) for r in rows], probes, cache)
        out[name] = {p: {"n_valid_attempts": len(rows), "evaded@0.01": sum(m3a2.logit(x[p]["ema"]) <= line[p] for x in s),
                         "median_logit": float(np.median([m3a2.logit(x[p]["ema"]) for x in s]))} for p in probes}
        print(name, {p: v["evaded@0.01"] for p, v in out[name].items()}, flush=True)

    count("plain replies | none", plain, "")
    for arm in CONTEXT_ARMS:
        if arm == ARM and not FRESH.exists():
            continue
        instruction, twin = instruction_and_twin(arm)
        own = [r for r in sets["attacks"] if r["arm"] == arm and r["present"]]
        count(f"plain replies | {arm} instruction", plain, instruction)
        count(f"plain replies | {arm} sincere twin", plain, twin)
        for name, pre in (("own instruction", instruction), ("sincere twin", twin), ("neutral preamble", m3a2.NEUTRAL_PREAMBLE), ("none", "")):
            count(f"{arm} replies | {name}", own, pre)
    (OUT / "context_check.json").write_text(json.dumps(out, indent=1))


CONTEXT_PROBES = ("on_peak@L48", "onsym_peak@L48", "r0_peak@L48", "on_gd@L48")


# ---------------------------------------------------------------- analysis (no GPU if everything is scored)

# attack arm -> the flat sincere set written in the same style (the within-style check)
STYLE_TWINS = {"m3a_searched": "m3a_searched style", "m3a_deadpan": "m3a_deadpan style", "r0_search": "r0_search style",
               "r1_search": "r1_search style", "r2b_search": "r2b_search style", "r2c_search": "r2c_search style",
               "r2cpk_search": "r2cpk_search style", ARM: f"{ARM} style"}


def logit_lines(probs: list[float]) -> dict[float, float]:
    """Lines as logit quantiles of the clean calibration scores (as m3a2.monitor_table, so counts match monitors.json)."""
    return {fa: float(np.quantile([m3a2.logit(x) for x in probs], 1 - fa)) for fa in FALSE_ALARMS}


def flat_sincere_rows_with_task() -> list[dict]:
    """The flat sincere rows with their tasks (m3a2's sets drop the task)."""
    with _reading_m3a2_folder():
        rows = m3a2.flat_sincere_rows()
    return rows + own_flat_rows()


def within_style(probe_names: list[str]) -> dict:
    """AUROC of each attack's valid (distinct) sarcastic replies against sincere replies written in the same style. Three
    contexts: as generated (each behind its own instruction and request), both instructions removed (bare request:
    attack requests say "sarcastic", sincere ones "sincere"), and both instructions and request words removed."""
    from sklearn.metrics import roc_auc_score

    sets = eval_sets()
    probes = {p: all_probes()[p] for p in probe_names}
    cache = load_cache()
    flat_all = flat_sincere_rows_with_task()
    out = {}
    for arm, twin in STYLE_TWINS.items():
        attack = m3a2.distinct_valid([r for r in sets["attacks"] if r["arm"] == arm])
        flat_rows = [r for r in flat_all if r["arm"] == twin]
        if not attack or not flat_rows:
            continue
        entry = {"n_sarcastic": len(attack), "n_sincere": len(flat_rows)}
        contexts = {"as_generated": lambda r: r["prompt"], "instruction_removed": lambda r: r["task"]["task"],
                    "instruction_and_request_word_removed": lambda r: without_request_word(r["task"]["task"])}
        for key, ctx in contexts.items():
            pos = m3a2.score([(ctx(r), r["reply"]) for r in attack], probes, cache)
            neg = m3a2.score([(ctx(r), r["reply"]) for r in flat_rows], probes, cache)
            for p in probes:
                y = [1] * len(pos) + [0] * len(neg)
                entry[f"{p} {key}"] = float(roc_auc_score(y, [s[p]["ema"] for s in pos] + [s[p]["ema"] for s in neg]))
        out[arm] = entry
        print(arm, {k: round(v, 3) if isinstance(v, float) else v for k, v in entry.items()}, flush=True)
    return out


def gains(probe_names: list[str], arms: list[str], n_boot: int = 2000) -> dict:
    """Task-clustered 95% intervals for (valid evasions per attempt, arm minus plain; repeats counted) at each line, as
    generated and with the instruction removed (m3a.clustered_diff), plus the point difference."""
    sets = eval_sets()
    probes = {p: all_probes()[p] for p in probe_names}
    cache = load_cache()
    cal = m3a2.score([(r["prompt"], r["reply"]) for r in sets["calibration"] if r["clean"]], probes, cache)
    rows = [r for r in sets["attacks"] if r["arm"] in arms + ["plain"]]
    s_gen = m3a2.score([(r["prompt"], r["reply"]) for r in rows], probes, cache)
    s_rm = m3a2.score([(r["task"]["task"], r["reply"]) for r in rows], probes, cache)
    out = {}
    for p in probes:
        lns = logit_lines([s[p]["ema"] for s in cal])
        for key, sc in (("as_generated", s_gen), ("instruction_removed", s_rm)):
            # clustered_diff compares probabilities with the line: convert the logit line back
            scored = [{**r, "scores": {p: x[p]}} for r, x in zip(rows, sc)]
            for arm in arms:
                for fa, line in lns.items():
                    pline = float(1 / (1 + np.exp(-line)))
                    rate = lambda a: sum(r["present"] and m3a2.logit(r["scores"][p]["ema"]) <= line for r in scored if r["arm"] == a) / \
                        sum(r["arm"] == a for r in scored)
                    out[f"{p} | {arm} | {key} | {fa}"] = {
                        "diff_per_attempt": rate(arm) - rate("plain"),
                        "ci95": m3a.clustered_diff(scored, arm, "plain", pline, p, "ema", n_boot)}
    return out


def on_vs_off(on: str, off: str, arms: list[str], n_boot: int = 2000) -> dict:
    """Critic 2: the paired comparison on the same replies. Per arm and line: valid evasions per attempt under the
    on-policy probe minus under the off-policy probe, with a task-clustered 95% interval; as generated and removed."""
    sets = eval_sets()
    probes = {p: all_probes()[p] for p in (on, off)}
    cache = load_cache()
    cal = m3a2.score([(r["prompt"], r["reply"]) for r in sets["calibration"] if r["clean"]], probes, cache)
    lns = {p: logit_lines([s[p]["ema"] for s in cal]) for p in probes}
    rng = np.random.default_rng(0)
    out = {}
    for arm in arms:
        rows = [r for r in sets["attacks"] if r["arm"] == arm]
        tasks = sorted({r["task"]["task"] for r in rows})
        for key, ctx in (("as_generated", lambda r: r["prompt"]), ("instruction_removed", lambda r: r["task"]["task"])):
            sc = m3a2.score([(ctx(r), r["reply"]) for r in rows], probes, cache)
            for fa in FALSE_ALARMS:
                ev = {p: np.array([r["present"] and m3a2.logit(x[p]["ema"]) <= lns[p][fa] for r, x in zip(rows, sc)], float) for p in probes}
                per_task = np.array([[sum(ev[p][i] for i, r in enumerate(rows) if r["task"]["task"] == t) for p in (on, off)] +
                                     [sum(r["task"]["task"] == t for r in rows)] for t in tasks])
                boots = []
                for _ in range(n_boot):
                    pick = per_task[rng.choice(len(tasks), len(tasks))]
                    boots.append((pick[:, 0].sum() - pick[:, 1].sum()) / pick[:, 2].sum())
                out[f"{arm} | {key} | {fa}"] = {"on": int(ev[on].sum()), "off": int(ev[off].sum()), "attempts": len(rows),
                                                "diff_per_attempt": float((ev[on].sum() - ev[off].sum()) / len(rows)),
                                                "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    return out


def examples(probe: str) -> list[dict]:
    """Every attempt of the fresh prompt, next to the plain reply to the same task and seed, with both judges' ratings
    and scores under the on- and off-policy probes (selection of examples is done when writing RESULT.md)."""
    sets = eval_sets()
    probes = {p: all_probes()[p] for p in (probe, "onsym_peak@L48", "r0_peak@L48")}
    cache = load_cache()
    cal = m3a2.score([(r["prompt"], r["reply"]) for r in sets["calibration"] if r["clean"]], probes, cache)
    line = {p: logit_lines([s[p]["ema"] for s in cal])[0.01] for p in probes}
    fresh = [r for r in sets["attacks"] if r["arm"] == ARM]
    plain = {(r["task"]["task"], r["seed"]): r for r in sets["attacks"] if r["arm"] == "plain"}
    pl_rows = [plain[(r["task"]["task"], r["seed"])] for r in fresh]
    s_f = m3a2.score([(r["prompt"], r["reply"]) for r in fresh], probes, cache)
    s_rm = m3a2.score([(r["task"]["task"], r["reply"]) for r in fresh], probes, cache)
    s_p = m3a2.score([(r["prompt"], r["reply"]) for r in pl_rows], probes, cache)
    lg = lambda x: round(m3a2.logit(x["ema"]), 2)
    return [{"message": m3a.message_of(r["task"]["task"]), "seed": r["seed"], "valid": r["present"],
             "ratings_repo_blind": [r["rating"], r["blind_rating"]], "reply": r["reply"],
             "logit": {p: lg(f[p]) for p in probes}, "logit_removed": {p: lg(rm[p]) for p in probes},
             "evaded@1%": {p: r["present"] and m3a2.logit(f[p]["ema"]) <= line[p] for p in probes},
             "plain_reply": pr["reply"], "plain_valid": pr["present"], "plain_logit": {p: lg(pl[p]) for p in probes},
             "line_logit@1%": {p: round(line[p], 2) for p in probes}}
            for r, f, rm, pr, pl in zip(fresh, s_f, s_rm, pl_rows, s_p)]


ANALYSIS_PROBES = ["on_peak@L48", "on_peak@L48s1", "on_peak@L48s2", "on_peak@L16", "onsym_peak@L48", "on_gd@L48", "r0_peak@L48"]


@app.local_entrypoint()
def m3a3_analyze():
    """Within-style AUROC, gains over plain with intervals, paired on- vs off-policy, and the fresh prompt's replies."""
    arms = sorted({r["arm"] for r in eval_sets()["attacks"]} - {"plain"})
    (OUT / "within_style.json").write_text(json.dumps(within_style(ANALYSIS_PROBES), indent=1))
    (OUT / "gains.json").write_text(json.dumps(gains(ANALYSIS_PROBES, arms), indent=1))
    (OUT / "on_vs_off.json").write_text(json.dumps(on_vs_off(PRIMARY, "r0_peak@L48", arms + ["plain"]), indent=1))
    (OUT / "examples.json").write_text(json.dumps(examples(PRIMARY), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dots":
        for p in sys.argv[2].split(","):
            print(m3a2.dot_plot(p))
