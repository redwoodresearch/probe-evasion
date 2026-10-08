"""Mission m1c, part 2 (lead's redirect): a maths-in-the-head probe trained on Gemma-2-9B's replies (off-policy text from a
weaker model), read by Qwen3.5-27B. Matched to mission 1a's Claude-written set (m1a_offpolicy_maths_claude_sys_train_clean):
same seeded task prompts and 8 answer styles, same label rule (onpolicy_probes.label_ok), comma balancing, and exactly the
same (kind, style) cell counts. Only the reply author changes (Claude -> Gemma). Gemma's replies already existed
(onpolicy_probes.build(model="gemma"): one sample per task, seed 0, plain prompt, no system prompt; Gemma mostly follows
"Do not show any working", unlike Claude, so no steering prompt was needed).

Evaluation reuses mission 1a's scored sets (results/gepa/m1a_offpolicy_maths_scores/*.jsonl, which already carry 1a's
probes' scores): copies are scored with the Gemma-text probes and written to SCORES, then 1a's analysis runs on them.

    uv run python -m probe_evasion.missions.m1c_gemma_maths build
    uv run modal run -m probe_evasion.missions.m1c_gemma_maths::m1c_gemma_train_and_score
    uv run python -m probe_evasion.missions.m1c_gemma_maths report
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from probe_evasion import gepa, onpolicy_probes
from probe_evasion.concept_data import MATHS_TEMPLATES, OUT as DATA
from probe_evasion.generation import write_jsonl
from probe_evasion.missions import m1a_offpolicy_maths as m1a
from probe_evasion.missions.m1a_peak_probe import PeakProbeLabSeeds  # registers the GPU class with the app at import
from probe_evasion.modal_app import app

TAG = "m1c_gemma_maths"
OUT = Path("missions/m1c_chem_no_near_negatives/gemma_maths")
SCORES = Path("results/gepa") / f"{TAG}_scores"
GEMMA_RAW = {"train": DATA / "onpolicy_gemma_maths_train.jsonl", "heldout": DATA / "onpolicy_gemma_maths_heldout.jsonl"}
TRAIN_FILE = DATA / f"{TAG}_train_clean.jsonl"
HELDOUT_FILE = DATA / f"{TAG}_heldout_clean.jsonl"
LAYERS = (16, 48)
TRAIN_JOBS = {f"{TAG}_gd": "gd", f"{TAG}_lr": "logreg"}
PROBES = {f"gem@L{l}": (f"{TAG}_gd", f"maths@L{l}") for l in LAYERS} | {f"gemlr@L{l}": (f"{TAG}_lr", f"maths@L{l}") for l in LAYERS}
# 1a's scored sets to re-score; attack-style sets also get the instruction-removed context
SETS = ("calibration", "calibration_one_sentence", "v2_calibration", "heldout_qwen", "heldout_claude", "saved_attacks",
        "attacks_val", "fresh", "format_negatives", "extras", "candidate_division", "confirm", "candidate_format_negatives",
        "candidate_context")
ATTACK_SETS = ("attacks_val", "fresh", "candidate_division", "confirm", "candidate_format_negatives")
CHUNK = 300


EXTRA_SAMPLES = DATA / f"{TAG}_extra_samples.jsonl"  # Gemma replies at seeds 1-3 (seed 0 is the existing file)
EXTRA_SEEDS = (1, 2, 3)


@app.local_entrypoint()
def m1c_gemma_samples():
    """More Gemma samples per task (temperature 0.7): one sample left several (kind, style) cells short of 1a's counts,
    mostly "given" rows lost to comma balancing (Gemma writes 1,234 in 41% of given replies vs 12% of computed)."""
    out = []
    for split in ("train", "heldout"):
        tasks = m1a.onpolicy_tasks(split)
        for seed in EXTRA_SEEDS:
            replies = gepa.generate("gemma", "maths", [t["prompt"] for t in tasks], seed=seed, max_tokens=onpolicy_probes.REPLY_TOKENS)
            out += [{"concept": "maths", "label": int(t["kind"] == "computed"), "kind": t["kind"], "style": t["style"],
                     "prompt": t["prompt"], "response": r["reply"], "result": t["result"], "split": split, "seed": seed}
                    for t, r in zip(tasks, replies) if r["finished"] and r["reply"].strip()]
    write_jsonl(EXTRA_SAMPLES, out)
    print(f"{len(out)} extra Gemma replies", flush=True)


def candidates(split: str) -> list[dict]:
    """Per task, one Gemma reply: the first sample (seed 0, then 1-3) that passes label_ok; for a not-computed task a
    comma-free passing sample is preferred, which only does per task what balance_commas does by dropping rows."""
    tasks = list({t["prompt"]: t for t in m1a.onpolicy_tasks(split)}.values())  # the seeded list repeats 2 prompts
    by_prompt = {}
    for r in m1a.jl(GEMMA_RAW[split]) + [r for r in m1a.jl(EXTRA_SAMPLES) if r["split"] == split]:
        by_prompt.setdefault(r["prompt"], []).append(r)
    out = []
    for t in tasks:
        ok = [r for r in by_prompt.get(t["prompt"], []) if onpolicy_probes.label_ok(r)]
        if t["kind"] != "computed":
            ok.sort(key=onpolicy_probes.has_comma)  # stable: seed order kept within comma / comma-free
        if ok:
            out.append({k: ok[0][k] for k in ("concept", "label", "kind", "style", "prompt", "response", "result")}
                       | {"author": "gemma-2-9b-it", "seed": ok[0].get("seed", 0)})
    return out


def build() -> None:
    for split, reference, out in (("train", m1a.OFFPOLICY_TRAIN, TRAIN_FILE), ("heldout", m1a.OFFPOLICY_HELDOUT, HELDOUT_FILE)):
        ok = candidates(split)
        balanced = onpolicy_probes.balance_commas(ok, random.Random(5))
        ref = m1a.jl(reference)
        matched = m1a.match_cells(balanced, ref, random.Random(7))
        used = set().union(*(v for k, v in m1a.prompt_sets().items() if k.startswith("search_")))
        assert not {r["prompt"] for r in matched} & used, "training prompts overlap the search tasks"
        assert len({r["prompt"] for r in matched}) == len(matched)
        write_jsonl(out, matched)
        print(f"{split}: {len(ok)} tasks with a passing reply -> {len(matched)} rows ({sum(r['label'] for r in matched)} "
              f"computed), 1a Claude {len(ref)} ({sum(r['label'] for r in ref)}) | comma rates Gemma {m1a.comma_rates(matched)}, "
              f"Claude {m1a.comma_rates(ref)}", flush=True)
        # Not every cell reaches 1a's count: Gemma restates given numbers with commas (1,234) but writes computed results
        # without (1234), so comma balancing drops many not-computed rows. Accepted: 1473 vs 1725 rows (stated in STATUS).


# ---------------------------------------------------------------- v2 set (critic 2): template-matched, separators stripped

TRAIN_FILE_V2 = DATA / f"{TAG}_v2_train_clean.jsonl"
SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}\b)")


def template_of(row: dict) -> str:
    """Which maths template the task came from ("recalled" for recall questions): matched on the prompt's wording."""
    if row["kind"] == "recalled":
        return "recalled"
    for i, (computed, given, _, _) in enumerate(MATHS_TEMPLATES):
        for t in (computed, given):
            pattern = re.escape(t)
            for field in ("a", "b", "c", "d", "p", "r", "item"):
                pattern = pattern.replace(re.escape("{" + field + "}"), "[^?]+?")
            if re.match(pattern, row["prompt"]):
                return str(i)
    raise ValueError(row["prompt"][:120])


def build_v2() -> None:
    """Critic 2: comma balancing plus the comma-free preference removed most large-number lookups from the not-computed
    side, so template and number size stood in for the label (garden lookups 81:28 computed:given vs Claude 89:74).
    Here thousands separators are stripped from every Gemma reply (so commas carry no label), no comma balancing, and
    rows are matched to 1a's Claude set per (template, kind, style) cell, drawing on all four Gemma samples per task."""
    for split, reference, out in (("train", m1a.OFFPOLICY_TRAIN, TRAIN_FILE_V2),):
        tasks = list({t["prompt"]: t for t in m1a.onpolicy_tasks(split)}.values())
        by_prompt = {}
        for r in m1a.jl(GEMMA_RAW[split]) + [r for r in m1a.jl(EXTRA_SAMPLES) if r["split"] == split]:
            by_prompt.setdefault(r["prompt"], []).append(r | {"response": SEPARATOR.sub("", r["response"])})
        ok = []
        for t in tasks:
            passing = [r for r in by_prompt.get(t["prompt"], []) if onpolicy_probes.label_ok(r)]
            if passing:
                r = passing[0]
                ok.append({k: r[k] for k in ("concept", "label", "kind", "style", "prompt", "response", "result")}
                          | {"author": "gemma-2-9b-it", "seed": r.get("seed", 0), "separators_stripped": True})
        for r in ok:
            r["template"] = template_of(r)
        ref = [r | {"template": template_of(r)} for r in m1a.jl(reference)]
        want = Counter((r["template"], r["kind"], r["style"]) for r in ref)
        by_cell = defaultdict(list)
        for r in ok:
            by_cell[(r["template"], r["kind"], r["style"])].append(r)
        rng = random.Random(7)
        matched, short = [], 0
        for cell, n in sorted(want.items()):
            have = by_cell[cell]
            matched += rng.sample(have, min(n, len(have)))
            short += max(0, n - len(have))
        assert len({r["prompt"] for r in matched}) == len(matched)
        used = set().union(*(v for k, v in m1a.prompt_sets().items() if k.startswith("search_")))
        assert not {r["prompt"] for r in matched} & used
        write_jsonl(out, matched)
        tmpl = Counter((r["template"], r["label"]) for r in matched)
        print(f"{split}: {len(matched)} rows ({sum(r['label'] for r in matched)} computed) vs 1a {len(ref)} "
              f"({sum(r['label'] for r in ref)}); {short} rows short across cells; commas left: "
              f"{sum(bool(re.search(r'\d,\d{3}', r['response'])) for r in matched)}", flush=True)
        print("computed:given per template", {t: f"{tmpl[(t, 1)]}:{tmpl[(t, 0)]}" for t in sorted({k[0] for k in tmpl})})


def load_probes() -> dict:
    from probe_evasion import concept_probes

    files = {f: json.loads((concept_probes.PROBES / f"{f}.json").read_text()) for f in TRAIN_JOBS}
    return {name: files[f][key] for name, (f, key) in PROBES.items()}


@app.local_entrypoint()
def m1c_gemma_train_and_score(train: bool = True, sets: str = ""):
    from probe_evasion import concept_probes

    if train:
        lab = concept_probes.ProbeLab(model_key="qwen35")
        rows = m1a.jl(TRAIN_FILE)
        calls = {name: lab.train.spawn(rows, layers=LAYERS, per_layer=True, method=method) for name, method in TRAIN_JOBS.items()}
        for name, call in calls.items():
            out = call.get()
            (concept_probes.PROBES / f"{name}.json").write_text(json.dumps(out))
            print(name, {k: round(v["val_auroc"], 4) for k, v in out.items()}, flush=True)
    probes = load_probes()
    SCORES.mkdir(parents=True, exist_ok=True)
    for name in (sets.split(",") if sets else SETS):
        src = m1a.SCORES / f"{name}.jsonl"
        dst = SCORES / f"{name}.jsonl"
        if not src.exists() or (dst.exists() and dst.stat().st_mtime > src.stat().st_mtime):
            continue
        score_into(name, m1a.jl(src), probes)


def score_into(name: str, rows: list[dict], probes: dict) -> None:
    """Adds the probes' readings to each row (real context, and instruction removed for attack sets), in chunks with a
    partial file so a dropped laptop connection resumes; writes SCORES/<name>.jsonl."""
    partial = SCORES / f"{name}.partial.jsonl"
    done = m1a.jl(partial) if partial.exists() else []
    for start in range(len(done), len(rows), CHUNK):
        part = rows[start:start + CHUNK]
        m1a.score_rows(part, probes)  # merges into each row's existing scores
        if name in ATTACK_SETS:
            removed = [{"prompt": r["task"]["task"], "reply": r["reply"], "reasoning": r.get("reasoning")} for r in part]
            m1a.score_rows(removed, probes)
            for r, x in zip(part, removed):
                r["scores_instruction_removed"] |= x["scores"]
        for r in part:
            r.pop("logits", None)  # per-token logits for every probe make the copies huge; readings are kept
        done += part
        write_jsonl(partial, done)
        print(f"   {name}: {len(done)}/{len(rows)}", flush=True)
    write_jsonl(SCORES / f"{name}.jsonl", done)
    partial.unlink(missing_ok=True)
    print(f"scored {name}: {len(done)}", flush=True)


# ---------------------------------------------------------------- peak-trained version (brief property 5)

PEAK_FILE = f"{TAG}_peak_seeds"
PEAK_SEEDS = (0, 1, 2)
PEAK_SETS = ("calibration", "calibration_one_sentence", "v2_calibration", "heldout_qwen", "confirm", "candidate_division",
             "candidate_format_negatives", "candidate_context")


PEAK_VERSIONS = {"v1": (PEAK_FILE, "gempk"), "v2": (f"{TAG}_v2_peak_seeds", "gem2pk")}  # version -> (probe file, short name)


def load_peak_probes(version: str = "v1") -> dict:
    from probe_evasion import concept_probes

    file, short = PEAK_VERSIONS[version]
    saved = json.loads((concept_probes.PROBES / f"{file}.json").read_text())
    return {f"{short}@L48@s{p['seed']}": p for p in saved.values()}


@app.local_entrypoint()
def m1c_gemma_peak(train: bool = True, version: str = "v1", score: bool = True):
    """The Gemma-text rows fitted with binary cross-entropy on the EMA peak (the deployed reading), 1a's seeded trainer
    (validation-loss stopping, 3 seeds, layer 48), then the evaluation sets re-scored with them. v1: the first
    (comma-balanced) set; v2: separators stripped, matched per template (build_v2)."""
    from probe_evasion import concept_probes

    if train:
        rows_file = {"v1": TRAIN_FILE, "v2": TRAIN_FILE_V2}[version]
        out = PeakProbeLabSeeds().train_peak_seeded.remote(m1a.jl(rows_file), 48, PEAK_SEEDS)
        (concept_probes.PROBES / f"{PEAK_VERSIONS[version][0]}.json").write_text(json.dumps(out))
        print({k: (round(v["val_loss"], 4), v["epoch"], round(v["val_auroc"], 4)) for k, v in out.items()}, flush=True)
    if not score:  # two versions must not score the shared copies at the same time (one partial file per set)
        return
    probes = load_peak_probes(version)
    for name in PEAK_SETS:
        rows = m1a.jl(SCORES / f"{name}.jsonl")
        if all(p in rows[0]["scores"] for p in probes):
            continue
        score_into(name, rows, probes)


def report() -> None:
    """1a's analysis (m1a_analysis.main) pointed at these copies, with the Gemma-text probes added."""
    an, _ = point_analysis_here()
    print("probes:", an.PROBE_SET)
    an.main()


COMPARE = ("gem@L48", "gemlr@L48", "on@L48", "off@L48", "gem@L16")


def point_analysis_here():
    """1a's analysis modules read module-level SCORES/OUT/PROBE_SET; point them at the Gemma-scored copies."""
    from probe_evasion.missions import m1a_analysis as an
    from probe_evasion.missions import m1a_candidates as ca

    an.SCORES, an.OUT, ca.SCORES, ca.OUT = SCORES, OUT, SCORES, OUT
    OUT.mkdir(parents=True, exist_ok=True)
    an.PROBE_SET = an.scored_probes()
    return an, ca


def candidates_report(compare: tuple[str, ...] = COMPARE, tag: str = "") -> dict:
    """The two strongest frozen prompts (1a's candidates) under the Gemma-text probe, with 1a's validity checks:
    one-sentence audited line; a line set on the prompt's OWN format (the prompt applied to no-maths lookups, cleaned
    and audited); the instruction removed from the context; plain replies with the instruction placed in front; strict
    judge alongside the lenient counting rule; the 1,000-sample confirmation set (division / mixed problems)."""
    from probe_evasion.missions.m1a_calibration_audit import computes_flags, evasion_counts, line
    from probe_evasion.missions.m1a_offpolicy_maths import CANDIDATES

    an, ca = point_analysis_here()
    rd = ca.READING
    pool = an.calibration_negatives("one sentence", audited=True)
    pools = {"one sentence": pool, "one sentence minus garden lookups": [r for r in pool if "garden" not in r["task"]["task"]]}
    lines = {name: {p: {fa: line([r["scores"][p][rd] for r in rs], fa) for fa in an.FALSE_ALARMS} for p in compare}
             for name, rs in pools.items()}
    negs = m1a.jl(SCORES / "candidate_format_negatives.jsonl")
    own = {}
    for arm in CANDIDATES:
        clean = gepa.clean_maths_negatives([r for r in negs if r["arm"] == arm and r["finished"]])
        clean = [r for r, v in zip(clean, computes_flags(clean)) if not (v.get("computes") or v.get("judge_refused"))]
        own[arm] = {"n": len(clean), **{p: {fa: line([r["scores"][p][rd] for r in clean], fa) for fa in an.FALSE_ALARMS}
                                        for p in compare}}
        own[arm]["flagged_by_one_sentence_1pct"] = {p: f"{sum(r['scores'][p][rd] > lines['one sentence'][p][0.01] for r in clean)}/{len(clean)}"
                                                    for p in compare}
    out = {"lines": lines, "own_format_lines": own, "n_pool": {k: len(v) for k, v in pools.items()}, "results": {}}
    from sklearn.metrics import roc_auc_score

    held = m1a.jl(SCORES / "heldout_qwen.jsonl")
    out["heldout_auroc"] = {p: roc_auc_score([r["label"] for r in held], [r["scores"][p][rd] for r in held]) for p in compare}
    out["heldout_plain_caught_one_sentence_1pct"] = {
        p: f"{sum(r['scores'][p][rd] > lines['one sentence'][p][0.01] for r in held if r['label'] == 1)}/{sum(r['label'] for r in held)}"
        for p in compare}
    for set_name in ("candidate_division", "confirm"):
        rows = m1a.jl(SCORES / f"{set_name}.jsonl")
        subsets = (("all", lambda r: True),) if set_name == "candidate_division" else (
            ("all", lambda r: True), ("division", lambda r: r["division"]), ("mixed", lambda r: not r["division"]))
        for p in compare:
            for arm in dict.fromkeys(r["arm"] for r in rows):
                for subset, keep in subsets:
                    for key in ("scores", "scores_instruction_removed"):
                        mine = [r for r in rows if r["arm"] == arm and keep(r)]
                        th = {name: lines[name][p] for name in lines}
                        if arm in own:
                            th["own format"] = own[arm][p]
                        res = {"lenient": evasion_counts([r | {"_valid": r["lenient"], "_score": r[key][p][rd]} for r in mine], th),
                               "strict": evasion_counts([r | {"_valid": r["present"], "_score": r[key][p][rd]} for r in mine], th)}
                        out["results"][f"{set_name} | {p} | {arm} | {subset} | {'context' if key == 'scores' else 'instruction removed'}"] = res
    ctx = m1a.jl(SCORES / "candidate_context.jsonl")
    for p in compare:
        for arm in dict.fromkeys(r["arm"] for r in ctx):
            rs = [r | {"_valid": r["lenient"], "_score": r["scores"][p][rd]} for r in ctx if r["arm"] == arm]
            out["results"][f"candidate_context | {p} | {arm}"] = {"lenient": evasion_counts(rs, {"one sentence": lines["one sentence"][p]})}
    (OUT / f"candidates_report{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def print_candidates(out: dict) -> None:
    compare = list(out["heldout_auroc"])
    print("pool sizes", out["n_pool"])
    print("held-out AUROC", {p: round(v, 4) for p, v in out["heldout_auroc"].items()}, "plain caught", out["heldout_plain_caught_one_sentence_1pct"])
    print("1% lines:", {n: {p: round(v[p][0.01], 3) for p in compare} for n, v in out["lines"].items()})
    for arm, o in out["own_format_lines"].items():
        print(f"own-format negatives [{arm}] n={o['n']}: own 1% line", {p: round(o[p][0.01], 3) for p in compare},
              "| flagged by one-sentence 1% line", o["flagged_by_one_sentence_1pct"])
    for k, v in out["results"].items():
        c = v["lenient"]
        cells = [(n, x) for n, x in c.items() if isinstance(x, dict) and n.endswith("1.0%")]
        print(f"{k:85s} att {c['attempts']} valid {c['valid_attempts']} | " + " | ".join(
            f"{n}: {x['evasions_per_attempt']} ({x['problems_with_an_evasion']} probs)" for n, x in cells)
              + (f" | strict one-sentence: {v['strict']['one sentence 1.0%']['evasions_per_attempt']}" if "strict" in v else ""))


def dot_plot(probes: tuple[str, ...], path: Path) -> None:
    """Confirmation set and its controls under each probe, EMA reading on a logit axis, with the one-sentence 1% line
    (solid), the same line without garden-area lookups (dashed) and the divaug prompt's own-format 1% line (dotted)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from probe_evasion.missions.m1a_calibration_audit import computes_flags, line

    an, _ = point_analysis_here()
    pool = an.calibration_negatives("one sentence", audited=True)
    negs = m1a.jl(SCORES / "candidate_format_negatives.jsonl")
    own = gepa.clean_maths_negatives([r for r in negs if r["arm"] == "divaug-v2 winner" and r["finished"]])
    own = [r for r, v in zip(own, computes_flags(own)) if not (v.get("computes") or v.get("judge_refused"))]
    conf = m1a.jl(SCORES / "confirm.jsonl")
    groups = {"no-maths one-sentence\n(not garden)": [r for r in pool if "garden" not in r["task"]["task"]],
              "no-maths one-sentence\ngarden lookups": [r for r in pool if "garden" in r["task"]["task"]],
              "no-maths lookups\nunder divaug prompt": own,
              "plain, valid": [r for r in conf if r["arm"] == "plain" and r["lenient"]],
              "divaug prompt, valid": [r for r in conf if r["arm"] == "divaug-v2 winner" and r["lenient"]],
              "window prompt, valid": [r for r in conf if r["arm"] == "window_L48 winner" and r["lenient"]]}
    fig, axes = plt.subplots(1, len(probes), figsize=(5.5 * len(probes), 5.5), dpi=130, sharey=False)
    rng = np.random.default_rng(0)
    for ax, p in zip(np.atleast_1d(axes), probes):
        for i, (name, rs) in enumerate(groups.items()):
            ys = np.clip([r["scores"][p]["ema"] for r in rs], 1e-6, 1 - 1e-6)
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(ys)), ys, s=5, alpha=0.5, color="#888888" if i < 3 else "#1f5f99", linewidths=0)
        ax.axhline(line([r["scores"][p]["ema"] for r in pool], 0.01), color="#c2410c", lw=1, label="one-sentence 1% line")
        ax.axhline(line([r["scores"][p]["ema"] for r in groups["no-maths one-sentence\n(not garden)"]], 0.01), color="#c2410c",
                   lw=1, ls="--", label="same, garden lookups left out")
        ax.axhline(line([r["scores"][p]["ema"] for r in own], 0.01), color="k", lw=1, ls=":", label="divaug own-format 1% line")
        ax.set_yscale("logit")
        ax.set_xticks(range(len(groups)), list(groups), fontsize=7, rotation=30, ha="right")
        ax.set_title(p.replace("gem2pk", "Gemma-text v2, peak-trained").replace("gempk", "Gemma-text v1, peak-trained")
                     .replace("gem@", "Gemma-text v1, mean-trained @").replace("on@", "own-replies (mean-trained) @"), fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("EMA reading (logit scale)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    {"build": build, "build_v2": build_v2, "report": report, "candidates": lambda: print_candidates(candidates_report()),
     "peak": lambda: print_candidates(candidates_report(tuple(load_peak_probes()) + ("on@L48",), "_peak")),
     "dots": lambda: dot_plot(("gem@L48", "gempk@L48@s0", "gem2pk@L48@s0", "on@L48"), OUT / "dots_gemma_maths.png"),
     "peak_v2": lambda: print_candidates(candidates_report(tuple(load_peak_probes("v2")) + ("on@L48",), "_peak_v2"))}[sys.argv[1]]()
