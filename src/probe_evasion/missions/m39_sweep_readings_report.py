"""Mission m39_sweep_readings: analysis, tables and figures (trends against ECI with m36's bootstrap, the
pre-registered headline readings, compute axes, the digits-only analysis). Reads saved files only.

    uv run python -m probe_evasion.missions.m39_sweep_readings_report
"""

import contextlib
import sys
import gzip
import json
from pathlib import Path

import attrs
import numpy as np
from dotenv import load_dotenv

from probe_evasion import config
from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions import m27_readout_and_style_checks as m27  # read-only: readouts, span locator, Part A lines
from probe_evasion.missions import m28_baselines_and_writer_knowledge as m28  # read-only
from probe_evasion.missions import m29_existing_data_analyses as m29  # read-only
from probe_evasion.missions import m36_attacker_capability_sweep as m36  # read-only: trend, writers, figure style
from probe_evasion.missions import m39_sweep_readings_lead as lead
from probe_evasion.missions.m39_sweep_readings import (CACHE, EFFORT_POINTS, FIGURES, HARMLESS, HEADLINE_B, LEFT_OUT, OUT, RUNS, TASKS, TEXT_CACHE, WORDS_CLASS,
                                                         baseline, broad_lines, chain_records, claude_spend, compact_rows, frozen_arms, harmless_rows, m36_root, ok, pairwise,
                                                         ratings_map, repeat_agreement, task_info)


# ---------------------------------------------------------------- trends (m36's trend code, other x-axes)


@contextlib.contextmanager
def capability_axis(x: dict[str, float]):
    """Run m36.trend against another per-writer x (e.g. log10 tokens per round): m36.WRITERS is swapped in this
    process only (eci = x, no interval) and restored afterwards. Slopes then read "per 10 units of x"."""
    saved = dict(m36.WRITERS)
    m36.WRITERS.update({n: attrs.evolve(w, eci=x[n], eci_ci=(x[n], x[n])) for n, w in saved.items() if n in x})
    try:
        yield
    finally:
        m36.WRITERS.clear()
        m36.WRITERS.update(saved)


def trend(chains: list[dict], b: int | None, key: str, drop: tuple = ()) -> dict:
    t = m36.trend([c for c in chains if c["writer"] not in drop], b, key)
    t.pop("leave_one_lab_out", None)
    return t


def compute_axes(chains: list[dict]) -> dict[str, dict[str, float]]:
    """Per writer, mean over chains of the per-round mean (rounds 0-16) of output tokens, reasoning tokens (OpenRouter
    only) and dollars, from m36's ledgers; as log10."""
    per: dict = {}
    for c in chains:
        comp = c["compute_per_round"]
        per.setdefault(c["writer"], []).append({k: np.mean([e[k] for e in comp]) for k in ("output_tokens", "reasoning_tokens", "cost")})
    axes = {k: {n: float(np.log10(np.mean([x[k] for x in v]))) for n, v in per.items()} for k in ("output_tokens", "cost")}
    axes["reasoning_tokens"] = {n: float(np.log10(np.mean([x["reasoning_tokens"] for x in v]))) for n, v in per.items()
                                if m36.WRITERS[n].api == "openrouter" and np.mean([x["reasoning_tokens"] for x in v]) > 0}
    return axes


def two_stage_draws(by: dict[str, list], stat, seed: int = 3939) -> list:
    """m36's two-stage bootstrap (writers with replacement, then chains within each writer) of stat(names, values),
    where values[i] is the mean over the drawn chains of writer names[i] (chains may carry tuples of measures)."""
    names = sorted(by, key=lambda n: m36.WRITERS[n].eci)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(m36.BOOTS):
        pick = [names[i] for i in rng.choice(len(names), len(names))]
        if len({m36.WRITERS[n].eci for n in pick}) < 2:
            continue
        vals = [np.mean(np.array(by[n])[rng.integers(0, len(by[n]), len(by[n]))], axis=0) for n in pick]
        out.append(stat(pick, np.array(vals)))
    return out


def two_predictor_fit(chains: list[dict], key: str, log_cost: dict, drop: tuple = ()) -> dict:
    """Amendment A9: OLS over writer means at B = 16, key ~ ECI/10 + log10(dollars per round); two-stage bootstrap;
    variance inflation factor. Inconclusive unless one coefficient's interval excludes 0."""
    by: dict = {}
    for c in chains:
        if c["effort"][HEADLINE_B].get(key) is not None and c["writer"] not in drop:
            by.setdefault(c["writer"], []).append(c["effort"][HEADLINE_B][key])

    def fit(names, y):
        X = np.column_stack([np.ones(len(names)), [m36.WRITERS[n].eci / 10 for n in names], [log_cost[n] for n in names]])
        return np.linalg.lstsq(X, y, rcond=None)[0][1:]

    names = sorted(by)
    est = fit(names, np.array([np.mean(by[n]) for n in names]))
    draws = np.array(two_stage_draws(by, fit))
    r = float(np.corrcoef([m36.WRITERS[n].eci for n in names], [log_cost[n] for n in names])[0, 1])
    return {"per_10_eci": float(est[0]), "per_tenfold_dollars": float(est[1]), "writers": len(names),
            "ci95_per_10_eci": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
            "ci95_per_tenfold_dollars": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(), "corr_eci_log_dollars": r, "vif": 1 / (1 - r ** 2)}


def paired_slopes(chains: list[dict], key: str, ref: str, b: int = HEADLINE_B, drop: tuple = (), ref_drop: tuple = ()) -> dict:
    """Amendment A4: the new measure's slope against the reference's, with the same writer and chain draws (two-stage);
    chains where either is missing are left out. ref_drop: writers left out of the reference only (item 4 compares the
    9-writer S with the 11-writer S, so there the draws are not paired and the interval is of the 9-writer slope)."""
    by: dict = {}
    for c in chains:
        e = c["effort"][b]
        if e.get(key) is not None and e.get(ref) is not None and c["writer"] not in drop:
            by.setdefault(c["writer"], []).append((e[key], e[ref]))
    slope = lambda names, v: (m36.ols_slope(np.array([m36.WRITERS[n].eci for n in names]), v[:, 0]) * 10,
                              m36.ols_slope(np.array([m36.WRITERS[n].eci for n in names]), v[:, 1]) * 10)
    names = sorted(by)
    point = slope(names, np.array([np.mean(by[n], axis=0) for n in names]))
    d = np.array(two_stage_draws(by, slope))
    return {"slope": point[0], "ref_slope": point[1], "diff": point[0] - point[1], "ratio": point[0] / point[1] if point[1] else None,
            "ci95_diff": np.quantile(d[:, 0] - d[:, 1], [0.025, 0.975]).tolist(),
            "ci95_ratio": np.quantile(d[:, 0] / d[:, 1], [0.025, 0.975]).tolist() if np.all(d[:, 1] != 0) else None, "writers": len(names)}


# Amendment A4: one headline per item (key, writers left out); reference = m36's S (maths also S_valid for the
# valid-only forms)
HEADLINES = {
    "item 1: in view, per-arm context baseline (S_vis_ctx)": ("S_vis_ctx", (), "S"),
    "item 1: in view, neutralised twin baseline (S_vis_twin)": ("S_vis_twin", (), "S"),
    "item 2: 10th percentile over all attempts (T10_all)": ("T10_all", (), "S"),
    "item 3: valid and rated ordinary (S_nat)": ("S_nat", (), "S"),
    "item 4: S without GPT-5 nano and Llama 4 Maverick": ("S", LEFT_OUT, "S"),
}
HEADLINES_VALID = {
    "item 1, valid only (S_vis_ctx_valid)": ("S_vis_ctx_valid", (), "S_valid"),
    "item 1, valid only (S_vis_twin_valid)": ("S_vis_twin_valid", (), "S_valid"),
    "item 2, valid only (T10)": ("T10", (), "S_valid"),
    "item 4, S_valid without GPT-5 nano and Llama": ("S_valid", LEFT_OUT, "S_valid"),
}


HEADLINES_POSTHOC = {  # P1, labelled post-hoc
    "post-hoc P1: running average restarted at the answer (S_ema_from_answer)": ("S_ema_from_answer", (), "S"),
    "post-hoc P1: max over the answer tokens (S_answer_max)": ("S_answer_max", (), "S"),
}
HEADLINES_POSTHOC_VALID = {
    "post-hoc P1, valid only (S_valid_ema_from_answer)": ("S_valid_ema_from_answer", (), "S_valid"),
    "post-hoc P1, valid only (S_valid_answer_max)": ("S_valid_answer_max", (), "S_valid"),
}


def reading(t: dict, ref_slope: float) -> str:
    """Pre-registered (A4): survives / weakened / not supported."""
    lo, hi = t["slope_ci95_two_stage"]
    if lo <= 0 <= hi:
        return "not supported"
    return "survives" if t["slope_per_10_eci"] >= 0.5 * ref_slope else "weakened"


def headlines(task: str, chains: list[dict], tr16: dict) -> dict:
    out = {}
    entries = HEADLINES | (HEADLINES_VALID if task == "maths" else {}) | HEADLINES_POSTHOC | (HEADLINES_POSTHOC_VALID if task == "maths" else {})
    for name, (key, drop, ref) in entries.items():
        t = trend(chains, HEADLINE_B, key, drop)
        if t.get("slope_per_10_eci") is None:
            out[name] = {"key": key, "missing": True}
            continue
        ref_t = tr16[ref]
        names = [n for n in t["writers"]]
        p = m36.spearman_permutation_p(np.array([m36.WRITERS[n].eci for n in names]), np.array(t["writer_means"]))
        out[name] = {"key": key, "left_out": list(drop), "reference": ref, "reference_slope": ref_t["slope_per_10_eci"],
                     "slope": t["slope_per_10_eci"], "ci95_two_stage": t["slope_ci95_two_stage"], "ci95_chains": t["slope_ci95_chains"],
                     "spearman": t["spearman"], "spearman_permutation_p": p, "reading": reading(t, ref_t["slope_per_10_eci"]),
                     "paired_vs_reference": paired_slopes(chains, key, ref, drop=drop) if not drop else None, "writers": t["n_writers"]}
    return out


# ---------------------------------------------------------------- part 5: digits-only arms (maths)


def words_class(share: float | None) -> str | None:
    if share is None:
        return None
    return "digits" if share <= WORDS_CLASS[0] else "words" if share >= WORDS_CLASS[1] else "mixed"


def fe_slope(arms: list[tuple[dict, int, float]]) -> dict:
    """Slope per 10 ECI of an arm-level value with one intercept per effort point B (rows = arms), two-stage bootstrap
    (writers, then their arms). arms: (chain record, B, value)."""
    def fit(rows):
        Bs = sorted({b for _, b, _ in rows})
        X = np.column_stack([[m36.WRITERS[c["writer"]].eci / 10 for c, _, _ in rows]] + [[float(b == x) for _, b, _ in rows] for x in Bs])
        return float(np.linalg.lstsq(X, np.array([v for _, _, v in rows]), rcond=None)[0][0])

    by: dict = {}
    for x in arms:
        by.setdefault(x[0]["writer"], []).append(x)
    names = sorted(by)
    if len(names) < 3:
        return {"writers": names}
    rng = np.random.default_rng(3939)
    draws = []
    for _ in range(m36.BOOTS // 5):
        rows = [r for i in rng.choice(len(names), len(names)) for r in [by[names[i]][j] for j in rng.integers(0, len(by[names[i]]), len(by[names[i]]))]]
        if len({m36.WRITERS[c["writer"]].eci for c, _, _ in rows}) > 1:
            draws.append(fit(rows))
    return {"slope_per_10_eci": fit(arms), "ci95_two_stage": np.quantile(draws, [0.025, 0.975]).tolist(), "writers": names, "arms": len(arms)}


def digits_analysis(chains: list[dict], strategies: dict) -> dict:
    """Amendment A10 (maths): arms by answer format (their valid replies: digits <= 10% in words, words >= 90%) and, at
    B = 16, by m36's Haiku tag "result_in_number_words" on the instruction. Mean S / S_valid / S_digitised per class;
    the S trend over digits-only arms at B = 16 and with B as a fixed effect (contributing writers listed)."""
    arms = [(c, b, e) for c in chains for b, e in c["effort"].items() if e.get("words_share") is not None]
    mean = lambda xs, k: float(np.mean([e[k] for _, _, e in xs if e.get(k) is not None])) if xs else None
    out = {"classes": {}}
    for cls in ("digits", "mixed", "words"):
        for label, sel in (("every B", lambda b: True), ("B = 16", lambda b: b == HEADLINE_B), ("B = 0", lambda b: b == 0)):
            xs = [x for x in arms if words_class(x[2]["words_share"]) == cls and sel(x[1])]
            out["classes"][f"{cls}, {label}"] = {"arms": len(xs), "writers": sorted({c["writer"] for c, _, _ in xs}, key=lambda n: -m36.WRITERS[n].eci),
                                                 **{f"mean_{k}": mean(xs, k) for k in ("S", "S_valid", "S_digitised", "S_valid_digitised")}}
    tagged = lambda c: "result_in_number_words" in (strategies.get(f"{c['writer']}/{c['chain']}", {}).get("strategies") or [])
    for label, keep in (("instruction tagged words", True), ("instruction not tagged words", False)):
        xs = [(c, b, e) for c, b, e in arms if b == HEADLINE_B and tagged(c) == keep]
        out["classes"][f"{label}, B = 16"] = {"arms": len(xs), "writers": sorted({c["writer"] for c, _, _ in xs}, key=lambda n: -m36.WRITERS[n].eci),
                                              **{f"mean_{k}": mean(xs, k) for k in ("S", "S_valid", "S_digitised", "S_valid_digitised")}}
    digits16 = [{**c, "S_d16": e["S"]} for c, b, e in arms if b == HEADLINE_B and words_class(e["words_share"]) == "digits"]
    out["trend_S_digits_B16"] = trend(digits16, None, "S_d16")
    out["fe_S_digits_every_B"] = fe_slope([(c, b, e["S"]) for c, b, e in arms if words_class(e["words_share"]) == "digits"])
    out["fe_S_valid_digits_every_B"] = fe_slope([(c, b, e["S_valid"]) for c, b, e in arms if words_class(e["words_share"]) == "digits"
                                                 and e["S_valid"] is not None])
    out["per_writer"] = {n: {cls: sum(words_class(e.get("words_share")) == cls for c in chains if c["writer"] == n for e in c["effort"].values())
                             for cls in ("digits", "mixed", "words")} for n in sorted({c["writer"] for c in chains}, key=lambda n: -m36.WRITERS[n].eci)}
    return out


def digit_swap_summary() -> dict:
    """Plain replies rewritten into words and attack word-form replies rewritten into digits: the score change."""
    rows = jl(RUNS / m28.base_monitor("maths").name / "digit_swap.jsonl")
    P, L = task_info("maths")["P"], task_info("maths")["lines"][0.01]
    original = {(r["request"], r["reply"]): r["s"] for rs in compact_rows("maths").values() for r in rs}
    out = {}
    for d in ("words->digits", "plain digits->words"):
        xs = [r for r in rows if r["direction"] == d]
        ch = [r["score_swapped"] - original[(r["request"], r["reply"])] for r in xs if r["score_swapped"] is not None]
        out[d] = {"replies": len(xs), "swapped": len(ch), "median_change": float(np.median(ch)) if ch else None,
                  "median_change_gap_share": float(np.median(ch)) / (P - L) if ch else None,
                  "share_rising": float(np.mean([c > 0 for c in ch])) if ch else None}
    return out


def training_data_words() -> dict:
    """The maths probe's training set (m1a / onpolicy_probes: data/concepts/onpolicy_qwen35_maths_train_clean.jsonl,
    which trains onpk@L48@s*) and the calibration pool: replies whose result appears only in English number words."""
    from probe_evasion.missions import m1a_offpolicy_maths as m1a

    out = {}
    for name, path, rows in (("probe training set", m1a.ONPOLICY_TRAIN, jl(m1a.ONPOLICY_TRAIN)),
                             ("calibration pool", m36_root("maths", "b") / "calibration_scores.jsonl",
                              [dict(r["task"], response=r["reply"]) for r in jl(m36_root("maths", "b") / "calibration_scores.jsonl")])):
        res = {}
        for r in rows:
            text = r.get("response") or r.get("reply") or ""
            label = r.get("label", "calibration (negative)")
            result = r.get("result")
            words = m28.result_in_words(text, int(result)) if isinstance(result, (int, float)) and float(result).is_integer() else None
            e = res.setdefault(str(label), {"n": 0, "result_only_in_words": 0, "result_only_in_words_ge_100": 0, "examples": []})
            e["n"] += 1
            if words:
                e["result_only_in_words"] += 1
                e["result_only_in_words_ge_100"] += int(result) >= 100
                if len(e["examples"]) < 5:
                    e["examples"].append(text[:200])
        out[name] = {"path": str(path), "by_label": res}
    return out



# ---------------------------------------------------------------- post-hoc P1: m27's alternative readouts on m36's test replies

ALT_READOUTS = ("ema_peak", "ema_from_answer", "answer_max", "mean", "token_max")  # ema_peak = the deployed reading (a check)
M27_RUN = {"maths": 0, "ordering": 6}  # index in m27.RUNS of the run with the same probe and calibration pool


def readout_lines(task: str) -> dict[str, float]:
    """Each readout's 1% line from m27 Part A (same calibration pool and probe as m36)."""
    run = m27.RUNS[M27_RUN[task]]
    summary = json.loads((m27.PART_A / "summary.json").read_text())[run.name]["readouts"]
    assert abs(summary["ema_peak"]["line"] - task_info(task)["lines"][0.01]) < 0.05  # m27's own ema_peak line vs m36's reference line
    return {r: summary[r]["line"] for r in ALT_READOUTS}


def readout_rows(task: str) -> dict[str, list[dict]]:
    """Every m36 test reply (both batches) under every readout (instruction removed and visible), with the answer's
    tokens from m27's span locator for valid replies (rules, then Haiku), and the reply length in tokens. Cached."""
    path = CACHE / f"readouts_{task}.jsonl.gz"
    if not path.exists():
        probe = task_info(task)["probe"]
        files = [f for _, root in m36.roots(task) for f in sorted((root / "test").glob("*.jsonl"))]
        distinct: dict = {}
        for f in files:
            for r in jl(f):
                if r["valid"] and (r["task"]["task"], r["reply"]) not in distinct:
                    distinct[(r["task"]["task"], r["reply"])] = {k: r[k] for k in ("task", "reply", "valid", "tokens", "logits")}
        run = m27.RUNS[M27_RUN[task]]

        def locatable(row: dict) -> bool:  # m27's digit parser overflows on replies with a digit run of 300+ characters
            try:
                m27.answer_spans(run.kind, row["reply"], m27.task_answer(run.kind, row["task"]))
                return True
            except OverflowError:
                return False

        keys = [k for k in distinct if locatable(distinct[k])]
        recs = m27.reply_records(run, [distinct[k] for k in keys], probe)
        spans = {k: (rec["answer_tokens"], rec["span_by"]) for k, rec in zip(keys, recs)}
        spans |= {k: ([], "parser overflow") for k in distinct if k not in spans}
        out = []
        for f in files:
            for r in jl(f):
                at, by = spans.get((r["task"]["task"], r["reply"]), (None, "invalid"))
                out.append({"key": r["key"], "batch": f.parent.parent.parent.name, "id": r["task"]["id"], "seed": r["seed"],
                            "n_reply_tokens": r["n_reply_tokens"], "span_by": by,
                            "removed": m27.readouts(r["logits"][probe], at or None), "visible": m27.readouts(r["logits_visible"][probe], at or None)})
        CACHE.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in out)
    rows: dict[str, list[dict]] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            x = json.loads(line)
            rows.setdefault(x["key"], []).append(x)
    return rows


def add_readout_measures(task: str, chains: list[dict]) -> dict:
    """S_<readout> and S_valid_<readout> (and _visible) for every arm, with the arm's median reply length. A valid
    reply whose answer was not located counts as no movement for the span readouts. Returns the plain medians."""
    rd = readout_rows(task)
    base = {(r["key"], r["id"], r["seed"]): r for r in (x for rs in compact_rows(task).values() for x in rs)}
    lines = readout_lines(task)
    plain = {}
    for view in ("removed", "visible"):
        for ro in ALT_READOUTS:
            vals = [r[view][ro] for r in rd["plain"] if ok(base[("plain", r["id"], r["seed"])]) and r[view][ro] is not None]
            plain[(view, ro)] = float(np.median(vals))
    for c in chains:
        for e in c["effort"].values():
            rows = rd.get(e["arm"])
            if rows is None:
                continue
            e["median_reply_tokens"] = float(np.median([r["n_reply_tokens"] for r in rows]))
            for view in ("removed", "visible"):
                for ro in ALT_READOUTS:
                    P, L = plain[(view, ro)], lines[ro]
                    good = [ok(base[(e["arm"], r["id"], r["seed"])]) and r[view][ro] is not None for r in rows]
                    x = [r[view][ro] if g else P for r, g in zip(rows, good)]
                    v = [r[view][ro] for r, g in zip(rows, good) if g]
                    sfx = "" if view == "removed" else "_visible"
                    e[f"S_{ro}{sfx}"] = (P - float(np.median(x))) / (P - L)
                    e[f"S_valid_{ro}{sfx}"] = (P - float(np.median(v))) / (P - L) if v else None
            e["valid_answer_not_located"] = sum(ok(base[(e["arm"], r["id"], r["seed"])]) and r["removed"]["ema_from_answer"] is None for r in rows)
    return {"plain_medians": {f"{v} {r}": x for (v, r), x in plain.items()}, "lines": lines}


READOUT_HEADLINES = ("ema_from_answer", "answer_max")


# ---------------------------------------------------------------- post-hoc P2: instruction-length control

def instruction_tokens() -> dict[str, int]:
    """Qwen3.5 token count of every m36 frozen instruction (keyed by "task|arm"), with the cached tokenizer.json. Needs the
    `tokenizers` package: `uv run --with tokenizers python -m probe_evasion.missions.m39_sweep_readings_report tokens`."""
    path = CACHE / "instruction_tokens.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(hf_hub_download(config.TARGET_MODEL, "tokenizer.json"))
        arms = {f"{t}|{k}": ins for t in TASKS for k, ins in frozen_arms(t).items() if ins}  # arm keys repeat across tasks
        path.write_text(json.dumps({k: len(tok.encode(ins, add_special_tokens=False).ids) for k, ins in arms.items()}, indent=0))
    return json.loads(path.read_text())


LENGTH_MEASURES = ("S", "S_visible", "S_vis_ctx", "S_ema_from_answer")


def length_control(task: str, chains: list[dict]) -> dict:
    """Over all arms of the task: OLS of a measure on ECI / 10 alone; with log10(instruction tokens) and median reply
    tokens; and with effort-point fixed effects too. Two-stage bootstrap (writers, then arms within writer)."""
    toks = {k.split("|")[1]: v for k, v in instruction_tokens().items() if k.startswith(f"{task}|")}
    out = {}
    for key in LENGTH_MEASURES:
        arms = [(c["writer"], b, e[key], toks[e["arm"]], e["median_reply_tokens"]) for c in chains for b, e in c["effort"].items()
                if e.get(key) is not None and e["arm"] in toks and e.get("median_reply_tokens") is not None]
        by: dict = {}
        for a in arms:
            by.setdefault(a[0], []).append(a)

        def fit(rows, model):
            cols = [[m36.WRITERS[w].eci / 10 for w, *_ in rows]]
            if model != "ECI only":
                cols += [[np.log10(t) for *_, t, _ in rows], [n for *_, n in rows]]
            if model == "+ length + effort-point fixed effects":
                Bs = sorted({b for _, b, *_ in rows})
                cols += [[float(b == x) for _, b, *_ in rows] for x in Bs]
            else:
                cols += [[1.0] * len(rows)]
            beta = np.linalg.lstsq(np.column_stack(cols), np.array([y for _, _, y, *_ in rows]), rcond=None)[0]
            return beta[:3] if model != "ECI only" else beta[:1]

        res = {}
        rng = np.random.default_rng(3940)
        # critic 2: the pooled fits mix effort points; the B = 16 fits are the headline estimand (there "+ length" is the
        # whole model, as one effort point needs no fixed effects)
        for model, b_only in (("ECI only", None), ("+ length", None), ("+ length + effort-point fixed effects", None),
                              ("B = 16: ECI only", HEADLINE_B), ("B = 16: + length", HEADLINE_B)):
            sub = {w: [a for a in v if b_only is None or a[1] == b_only] for w, v in by.items()}
            names = sorted(w for w in sub if sub[w])
            base_model = model.split(": ")[-1]
            draws = []
            for _ in range(m36.BOOTS // 5):
                rows = [x for i in rng.choice(len(names), len(names)) for x in [sub[names[i]][j] for j in rng.integers(0, len(sub[names[i]]), len(sub[names[i]]))]]
                if len({m36.WRITERS[w].eci for w, *_ in rows}) > 2:
                    draws.append(fit(rows, base_model))
            draws = np.array(draws)
            est = fit([a for n in names for a in sub[n]], base_model)
            res[model] = {"per_10_eci": float(est[0]), "ci95_per_10_eci": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
                          **({"per_tenfold_instruction_tokens": float(est[1]), "ci95_per_tenfold_instruction_tokens": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
                              "per_reply_token": float(est[2]), "ci95_per_reply_token": np.quantile(draws[:, 2], [0.025, 0.975]).tolist()} if len(est) > 1 else {})}
        res["arms"] = len(arms)
        res["corr_eci_log_tokens"] = float(np.corrcoef([m36.WRITERS[w].eci for w, *_ in arms], [np.log10(t) for *_, t, _ in arms])[0, 1])
        out[key] = res
    return out


# ---------------------------------------------------------------- analysis

ALL_B_KEYS = ("S", "S_valid", "S_visible", "S_visible_fair", "S_vis_ctx", "T10_all", "T05_all", "S_nat", "S_broad", "half_gap_share")
B16_EXTRA = ("S_paired_mean", "valid_share", "S_visible_fair_paired", "S_visible_fair_line_in_view", "S_visible_fair_length", "S_vis_ctx_valid",
             "S_vis_ctx_paired", "ctx_shift", "drop_vis_ctx", "S_vis_twin", "S_vis_twin_valid", "S_vis_twin_paired", "S_vis_twin_any", "twin_shift",
             "twin_removed_shift", "drop_vis_twin", "T05", "T10", "Tmin", "T05_visible_fair", "T10_visible_fair", "Tmin_visible_fair", "S_nat5",
             "S_nat_paired_mean", "T10_all_nat", "S_nat_valid", "S_nat_visible_fair", "natural_share", "natural_share_of_valid", "S_digitised",
             "S_valid_digitised", "crossings_broad", "crossings_survive_text", "ctx_minus_twin_shift")
LEFT_OUT_KEYS = ("S", "S_valid", "S_paired_mean", "S_visible", "S_vis_ctx", "S_vis_twin", "T10_all", "S_nat")
COMPUTE_KEYS = ("S", "S_valid", "S_vis_ctx", "T10_all", "S_nat")
COUNT_KEYS = ("under_0.05", "under_0.01", "under_0.001", "under_0.01_visible", "crossings_broad", "crossings_broad_0.001",
              "crossings_survive_text", "crossings_survive_text_and_natural", "crossings_text_judged")


def pairwise_summary(pw_rows: list[dict]) -> dict:
    """Amendment A8. Per task, writer and stratum: pairs, plain-chosen score per pair (mean of the two orders: 1, 0.5
    or 0), share of pairs where the judge picks the same reply in both orders; per stratum pooled over writers; the
    rated-4-5 stratum against the plain-vs-harmless control; and the random stratum's trend against ECI (descriptive)."""
    out: dict = {}
    for task in TASKS:
        pairs: dict = {}
        for r in pw_rows:
            if r["task"] == task and r["plain_chosen"] is not None:
                pairs.setdefault((r["writer"], r["stratum"], r["pair"]), []).append(r["plain_chosen"])
        cells: dict = {}
        for (w, st, _), v in pairs.items():
            if len(v) == 2:
                cells.setdefault((w, st), []).append(v)
        summ = lambda vs: {"pairs": len(vs), "plain_chosen": float(np.mean(vs)), "same_reply_both_orders": float(np.mean([v[0] == v[1] for v in vs]))}
        out[task] = {"cells": {f"{w} / {st}": summ(vs) for (w, st), vs in sorted(cells.items())},
                     "strata": {st: summ([v for (w, s_), vs in cells.items() if s_ == st and not w.startswith("harmless") for v in vs])
                                for st in sorted({s_ for _, s_ in cells})}}
        pseudo = [{"writer": w, "eci": m36.WRITERS[w].eci, "lab": m36.WRITERS[w].lab, "pw": float(np.mean(v))}
                  for (w, st), vs in cells.items() if st == "random" and w in m36.WRITERS for v in vs]
        out[task]["trend_random"] = trend(pseudo, None, "pw")
    letters = [r for r in pw_rows if r["choice"]]
    out["share_A_chosen"] = float(np.mean([r["choice"] == "A" for r in letters])) if letters else None
    return out


def naturalness_summary(ratings: dict) -> dict:
    """Oddities, plain and harmless rating shares (and A7's fallback cutoff if plain is under 90% rated 4-5), word-form
    maths replies' ratings, repeat agreement."""
    plain = {t: m29.rating_summary([ratings.get((r["request"], r["reply"])) for r in compact_rows(t)["plain"] if ok(r) and r["batch"] == "b"]) for t in TASKS}
    words = [ratings.get((r["request"], r["reply"])) for k, rs in compact_rows("maths").items() if k != "plain" for r in rs
             if ok(r) and m28.result_in_words(r["reply"], r["result"])]
    return {"oddities": [{"reply": a, "rating": ratings.get((q, a))} for q, a in m29.ODDITIES], "plain": plain,
            "plain_below_90pct": {t: plain[t]["share_4_5"] < 0.9 for t in TASKS},
            "harmless": {t: m29.rating_summary([ratings.get((r["task"]["task"], r["reply"])) for r in harmless_rows(t) if r["valid"]]) for t in TASKS},
            "maths_word_form_attack_replies": m29.rating_summary(words), "repeat_agreement": repeat_agreement(),
            "unrated_or_refused": sum(v is None for v in ratings.values())}


def analyse(task: str, ratings: dict, pw: dict | None) -> dict:
    chains = chain_records(task, ratings)
    for c in chains:  # critic 2: the attack-specific part of the context effect (beyond the neutralised twin's)
        e = c["effort"][HEADLINE_B]
        if e.get("ctx_shift") is not None and e.get("twin_shift") is not None:
            e["ctx_minus_twin_shift"] = e["ctx_shift"] - e["twin_shift"]
    readout_info = add_readout_measures(task, chains)
    h_keys = tuple(f"S_visible_fair_h{i}" for i in range(len(HARMLESS[task])))
    ro_keys = tuple(f"S{v}_{ro}{sfx}" for ro in ALT_READOUTS for v in ("", "_valid") for sfx in ("", "_visible"))
    keys_by_b = {b: (ALL_B_KEYS + B16_EXTRA + h_keys + ro_keys if b in (0, HEADLINE_B) else ALL_B_KEYS + ("S_ema_from_answer", "S_answer_max"))
                 for b in EFFORT_POINTS}
    tr = {b: {k: trend(chains, b, k) for k in keys} for b, keys in keys_by_b.items()}
    axes = compute_axes(chains)
    by_compute, by_compute_no_llama = {}, {}
    for ax, x in axes.items():
        with capability_axis(x):
            cs = [c for c in chains if c["writer"] in x]
            by_compute[ax] = {k: trend(cs, HEADLINE_B, k) for k in COMPUTE_KEYS}
            by_compute_no_llama[ax] = {k: trend(cs, HEADLINE_B, k, ("llama4maverick",)) for k in COMPUTE_KEYS}
    writers = {}
    for n in sorted({c["writer"] for c in chains}, key=lambda n: -m36.WRITERS[n].eci):
        cs = [c for c in chains if c["writer"] == n]
        mean = lambda b, k: (lambda v: float(np.mean(v)) if v else None)([c["effort"][b][k] for c in cs if c["effort"][b].get(k) is not None])
        total = lambda b, k: sum(c["effort"][b].get(k) or 0 for c in cs)
        writers[n] = {"label": m36.WRITERS[n].label, "eci": m36.WRITERS[n].eci,
                      "B16": {k: mean(HEADLINE_B, k) for k in keys_by_b[HEADLINE_B] + ("words_share", "P_h_length_matched", "B_arm", "T_arm")},
                      "B0": {k: mean(0, k) for k in keys_by_b[0]},
                      "by_B": {b: {k: mean(b, k) for k in ALL_B_KEYS} for b in EFFORT_POINTS},
                      "counts_B16": {k: total(HEADLINE_B, k) for k in COUNT_KEYS},
                      "counts_all_B": {k: sum(total(b, k) for b in EFFORT_POINTS) for k in COUNT_KEYS},
                      "min_valid_B16": min((c["effort"][HEADLINE_B]["min_valid"] for c in cs if c["effort"][HEADLINE_B].get("min_valid") is not None), default=None),
                      "compute": {ax: x.get(n) for ax, x in axes.items()}}
    b16 = [(c, c["effort"][HEADLINE_B]) for c in chains if c["effort"][HEADLINE_B].get("ctx_shift") is not None]
    ctx_corr = {"ctx_shift_vs_eci": float(np.corrcoef([c["eci"] for c, _ in b16], [e["ctx_shift"] for _, e in b16])[0, 1]) if b16 else None,
                "ctx_shift_vs_log_words": float(np.corrcoef([np.log10(len(e["instruction"].split())) for _, e in b16], [e["ctx_shift"] for _, e in b16])[0, 1])
                if b16 else None}
    return {"task": task, "baseline": {k: v for k, v in baseline(task).items() if k != "b"}, "lines": task_info(task)["lines"], "broad_lines": broad_lines(task),
            "P": task_info(task)["P"], "trend": tr, "headlines": headlines(task, chains, tr[HEADLINE_B]),
            "left_out": {k: trend(chains, HEADLINE_B, k, LEFT_OUT) for k in LEFT_OUT_KEYS}, "ctx_correlations": ctx_corr,
            "compute_axes": axes, "by_compute": by_compute, "by_compute_without_llama": by_compute_no_llama,
            "two_predictor": {k: two_predictor_fit(chains, k, axes["cost"]) for k in ("S", "S_valid")},
            "two_predictor_without_llama": {k: two_predictor_fit(chains, k, axes["cost"], ("llama4maverick",)) for k in ("S", "S_valid")},
            "writers": writers, "readouts": readout_info, "length_control": length_control(task, chains),
            "digits": digits_analysis(chains, json.loads((m36.OUT / "analysis.json").read_text())[task]["strategies"]) if task == "maths" else None,
            "pairwise": pw[task] if pw else None,
            "chains": [{k: c[k] for k in ("writer", "chain", "batch", "eci")} | {
                "effort": {b: {k: v for k, v in e.items() if k not in ("instruction", "within_style", "frozen", "compute")} for b, e in c["effort"].items()}}
                for c in chains]}


def holm_headlines(results: dict) -> None:
    """Holm over the two tasks on each headline's Spearman permutation p (in place)."""
    for name in HEADLINES:
        ps = {t: r["headlines"][name].get("spearman_permutation_p") for t, r in results.items()}
        if all(p is not None for p in ps.values()):
            for t, p in m36.holm(ps).items():
                results[t]["headlines"][name]["spearman_p_holm"] = p


def report() -> dict:
    ratings = ratings_map()
    pw_rows = pairwise(ratings) if ratings else []
    pw = pairwise_summary(pw_rows) if pw_rows else None
    results = {t: analyse(t, ratings, pw) for t in TASKS}
    holm_headlines(results)
    extra = {"training_data_words": training_data_words(), "naturalness": naturalness_summary(ratings) if ratings else None,
             "pairwise_share_A_chosen": pw["share_A_chosen"] if pw else None, "claude_spend": claude_spend(),
             "harmless_checks": json.loads((OUT / "harmless_checks.json").read_text()), "twins": lead.twins_summary(),
             "causal": lead.causal_summary(), "text_monitor": lead.text_monitor_summary() if TEXT_CACHE.exists() else None,
             "digit_swap": digit_swap_summary() if (RUNS / m28.base_monitor("maths").name / "digit_swap.jsonl").exists() else None}
    (OUT / "analysis.json").write_text(json.dumps({"results": results, "extra": extra}, indent=1, default=lambda x: x.item() if hasattr(x, "item") else list(x)))
    (OUT / "tables.md").write_text(tables(results, extra))
    for path in figures(results):
        print(path)
    for t, r in results.items():
        for name, h in r["headlines"].items():
            if not h.get("missing"):
                print(t, name, f"{h['slope']:+.3f} [{h['ci95_two_stage'][0]:+.3f}, {h['ci95_two_stage'][1]:+.3f}] ref {h['reference_slope']:+.3f} -> {h['reading']}")
    return results


# ---------------------------------------------------------------- tables

LABELS = {
    "S": "S (m36 headline: instruction removed)", "S_valid": "S_valid (m36)", "S_paired_mean": "S_paired_mean (m36)", "valid_share": "valid share",
    "S_visible": "S_visible (m36: in view, no-instruction plain as baseline)",
    "S_visible_fair": "S_visible_fair (in view, pooled harmless baseline; level only)",
    "S_visible_fair_line_in_view": "S_visible_fair, line re-read in view (generic instructions)",
    "S_visible_fair_paired": "S_visible_fair, paired per prompt", "S_visible_fair_length": "S_visible_fair, length-matched harmless baseline",
    "S_vis_ctx": "S_vis_ctx (in view, plain re-read with the arm's instruction) [item 1]", "S_vis_ctx_valid": "S_vis_ctx, valid replies only",
    "S_vis_ctx_paired": "S_vis_ctx, paired per prompt", "ctx_shift": "P - B_arm: the instruction in context alone (log-odds)",
    "drop_vis_ctx": "B_arm - M_vis: the reply's own drop, in view (log-odds)",
    "S_vis_twin": "S_vis_twin (in view, neutralised-twin baseline) [item 1, lead]", "S_vis_twin_valid": "S_vis_twin, valid replies only",
    "S_vis_twin_paired": "S_vis_twin, paired per prompt", "S_vis_twin_any": "S_vis_twin incl. twins that failed the Haiku check",
    "twin_shift": "P - T_arm: twin replies read in view vs plain (log-odds)", "twin_removed_shift": "P - twin replies' instruction-removed median",
    "drop_vis_twin": "T_arm - M_vis (log-odds)",
    "T10_all": "T10_all: 10th percentile, all attempts, invalid = plain [item 2]", "T05_all": "T05_all: 5th percentile, all attempts",
    "T05": "T05: 5th percentile of valid scores", "T10": "T10: 10th percentile of valid scores", "Tmin": "Tmin: lowest valid score (descriptive)",
    "T05_visible_fair": "T05, in view (harmless baseline)", "T10_visible_fair": "T10, in view", "Tmin_visible_fair": "Tmin, in view",
    "half_gap_share": "share of attempts valid and closing half the gap",
    "S_nat": "S_nat: valid and rated 4-5, else plain [item 3]", "S_nat5": "S_nat5: rated 5 only", "S_nat_paired_mean": "S_nat, paired mean",
    "T10_all_nat": "T10_all, natural replies only", "S_nat_valid": "S_nat_valid: median of valid replies rated 4-5 (arms with >= 50)",
    "S_nat_visible_fair": "S_nat, read in view (harmless baseline)", "natural_share": "share of attempts valid and rated 4-5",
    "natural_share_of_valid": "share of valid replies rated 4-5", "S_digitised": "S with number words swapped for digits",
    "S_valid_digitised": "S_valid with number words swapped for digits", "S_broad": "S against the broad-traffic 1% line",
    "crossings_broad": "valid replies under the broad 1% line (count per arm)", "crossings_survive_text": "... that the text monitor passes",
    "ctx_minus_twin_shift": "context shift minus twin shift: the attack-specific part of the instruction-in-context effect (log-odds)",
}


def fmt(x, nd: int = 2) -> str:
    return "-" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def trend_row(name: str, t: dict, scale: float = 1.0) -> str:
    if t.get("slope_per_10_eci") is None:
        return f"| {name} | {t.get('n_writers')} | - | - | - | - | - |"
    ci = lambda k: f"[{t[k][0] * scale:+.3f}, {t[k][1] * scale:+.3f}]"
    return (f"| {name} | {t['n_writers']} | {t['slope_per_10_eci'] * scale:+.3f} | {ci('slope_ci95_two_stage')} | {ci('slope_ci95_chains')} | "
            f"{t['spearman']:+.2f} | {fmt(t['spearman_permutation_p'], 4)} |")


TREND_HEAD = "| measure | writers | slope | 95% two-stage | 95% chains only | Spearman | perm. p |\n|---|---|---|---|---|---|---|"


def headline_table(r: dict) -> list[str]:
    out = ["| item | slope per 10 ECI | 95% two-stage | reference (m36) | reading | paired difference vs reference [95%] | Spearman (perm. p; Holm) |",
           "|---|---|---|---|---|---|---|"]
    for name, h in r["headlines"].items():
        if h.get("missing"):
            out.append(f"| {name} | - | - | - | not computed | - | - |")
            continue
        pv = h["paired_vs_reference"]
        pair = f"{pv['diff']:+.3f} [{pv['ci95_diff'][0]:+.3f}, {pv['ci95_diff'][1]:+.3f}]" if pv else "(different writer sets)"
        out.append(f"| {name} | {h['slope']:+.3f} | [{h['ci95_two_stage'][0]:+.3f}, {h['ci95_two_stage'][1]:+.3f}] | {h['reference']} {h['reference_slope']:+.3f} | "
                   f"**{h['reading']}** | {pair} | {h['spearman']:+.2f} ({h['spearman_permutation_p']:.4f}; {fmt(h.get('spearman_p_holm'), 4)}) |")
    return out


def tables(results: dict, extra: dict) -> str:
    out = []
    for task, r in results.items():
        b = r["baseline"]
        out.append(f"## {task}\n\nPlain P (instruction removed, m36) {r['P']:.2f}; one-sentence lines {', '.join(f'{fa:g}: {x:.2f}' for fa, x in r['lines'].items())}; "
                   f"broad-traffic lines {', '.join(f'{fa:g}: {x:.2f}' for fa, x in r['broad_lines'].items())}.\n")
        out.append(f"### Pre-registered headlines (B = {HEADLINE_B})\n")
        out += headline_table(r)
        out.append("\n### Post-hoc P1: other readouts of the same replies (B = 16; slope per 10 ECI [two-stage]; 1% lines from m27 Part A)\n")
        out.append("| readout | 1% line | plain median | S slope | S_valid slope | S slope, read in view | Spearman (S) |\n|---|---|---|---|---|---|---|")
        cell = lambda t: f"{t['slope_per_10_eci']:+.3f} [{t['slope_ci95_two_stage'][0]:+.3f}, {t['slope_ci95_two_stage'][1]:+.3f}]" if t.get("slope_per_10_eci") is not None else "-"
        for ro in ALT_READOUTS:
            t16 = r["trend"][HEADLINE_B]
            out.append(f"| {ro} | {r['readouts']['lines'][ro]:.2f} | {r['readouts']['plain_medians'][f'removed {ro}']:.2f} | {cell(t16[f'S_{ro}'])} | "
                       f"{cell(t16[f'S_valid_{ro}'])} | {cell(t16[f'S_{ro}_visible'])} | {t16[f'S_{ro}']['spearman']:+.2f} |")
        rcols = [f"S_{ro}" for ro in ALT_READOUTS]
        out.append("\n| writer | " + " | ".join(rcols) + " |\n|---|" + "---|" * len(rcols))
        for n, w in r["writers"].items():
            out.append(f"| {w['label']} | " + " | ".join(fmt(w["B16"].get(k)) for k in rcols) + " |")
        out.append("\n### Post-hoc P2: instruction-length control (all arms; coefficient on ECI / 10 [two-stage])\n")
        out.append("| measure | arms | all B: ECI only | all B: + log10 instruction tokens and median reply tokens | all B: + length + effort-point fixed effects | B = 16: ECI only | B = 16: + length | all B, + length: instruction tokens coefficient (per tenfold) | reply tokens coefficient |\n|---|---|---|---|---|---|---|---|---|")
        for k, x in r["length_control"].items():
            ce = lambda m: f"{x[m]['per_10_eci']:+.3f} [{x[m]['ci95_per_10_eci'][0]:+.3f}, {x[m]['ci95_per_10_eci'][1]:+.3f}]"
            m = x["+ length"]
            out.append(f"| {k} | {x['arms']} | {ce('ECI only')} | {ce('+ length')} | {ce('+ length + effort-point fixed effects')} | {ce('B = 16: ECI only')} | {ce('B = 16: + length')} | "
                       f"{m['per_tenfold_instruction_tokens']:+.3f} [{m['ci95_per_tenfold_instruction_tokens'][0]:+.3f}, {m['ci95_per_tenfold_instruction_tokens'][1]:+.3f}] | "
                       f"{m['per_reply_token']:+.4f} [{m['ci95_per_reply_token'][0]:+.4f}, {m['ci95_per_reply_token'][1]:+.4f}] |")
        out.append(f"\nCorrelation of ECI with log10 instruction tokens over arms: {r['length_control']['S']['corr_eci_log_tokens']:+.2f}.")
        out.append(f"\nHarmless baseline: P_h (visible, kept instructions) {b['P_h']:.2f}, instruction removed {b['P_h_removed']:.2f}, valid share "
                   f"{b['valid_share']:.3f}; excluded by A5: {b['excluded'] or 'none'}; slope of the per-instruction visible medians against log10 words "
                   f"{b['length_slope_per_log10_words']:+.2f}. Calibration median {b['calibration_median']:.2f}; re-read in view: generic "
                   f"{b['calibration_median_in_view_generic']:.2f}, harmless {b['calibration_median_in_view_harmless']:.2f}. Lines in view (generic): "
                   f"{', '.join(f'{fa:g}: {x:.2f}' for fa, x in b['line_in_view'].items())}; (harmless): "
                   f"{', '.join(f'{fa:g}: {x:.2f}' for fa, x in b['line_in_view_harmless'].items())}; re-read without instruction "
                   f"{', '.join(f'{fa:g}: {x:.2f}' for fa, x in b['line_reread_no_instruction'].items())} (max |re-read - stored| {b['reread_max_abs_diff']:.4f}).\n")
        out.append("| harmless instruction | words | valid / n | median visible | P_h_i - P (log-odds) | median removed - P | 1% line in view | full vs cheaper rule disagreements |\n|---|---|---|---|---|---|---|---|")
        for i, x in b["per_instruction"].items():
            out.append(f"| h{i} | {x['words']} | {x['valid']} / {x['n']} | {x['median_visible']:.2f} | {x['median_visible'] - r['P']:+.2f} | "
                       f"{x['median_removed'] - r['P']:+.2f} | {b['line_in_view_by_instruction'][f'h{i}'][0.01]:.2f} | {x['valid_cheap_disagree']} |")
        c = r["ctx_correlations"]
        out.append(f"\nPer-arm context shift P - B_arm at B = 16: correlation with ECI {fmt(c['ctx_shift_vs_eci'])}, with log10 instruction words "
                   f"{fmt(c['ctx_shift_vs_log_words'])}.")
        out.append(f"\n### Every measure against ECI at B = {HEADLINE_B} (slope per 10 ECI over writer means)\n\n" + TREND_HEAD)
        for k, t in r["trend"][HEADLINE_B].items():
            out.append(trend_row(LABELS.get(k, k), t))
        out.append("\n### At every B (slope per 10 ECI [two-stage])\n\n| measure | " + " | ".join(f"B={b_}" for b_ in EFFORT_POINTS) + " |\n|---|" + "---|" * len(EFFORT_POINTS))
        cell = lambda t: f"{t['slope_per_10_eci']:+.3f} [{t['slope_ci95_two_stage'][0]:+.2f}, {t['slope_ci95_two_stage'][1]:+.2f}]" if t.get("slope_per_10_eci") is not None else "-"
        for k in ALL_B_KEYS:
            out.append(f"| {k} | " + " | ".join(cell(r["trend"][b_][k]) for b_ in EFFORT_POINTS) + " |")
        out.append(f"\n### Without GPT-5 nano and Llama 4 Maverick (9 writers, B = {HEADLINE_B})\n\n" + TREND_HEAD)
        for k, t in r["left_out"].items():
            out.append(trend_row(LABELS.get(k, k), t))
        out.append(f"\n### Against writer compute (B = {HEADLINE_B}; slope per tenfold; reasoning tokens: OpenRouter writers with reasoning only)\n\n" + TREND_HEAD)
        for label, block in (("", r["by_compute"]), (", without Llama", r["by_compute_without_llama"])):
            for ax, ts in block.items():
                for k, t in ts.items():
                    out.append(trend_row(f"{k} vs log10 {ax} per round{label}", t, 0.1))
        for label, block in (("", r["two_predictor"]), (" (without Llama)", r["two_predictor_without_llama"])):
            for k, f in block.items():
                out.append(f"\nTwo predictors{label}, {k} ({f['writers']} writers): {f['per_10_eci']:+.3f} per 10 ECI [{f['ci95_per_10_eci'][0]:+.3f}, "
                           f"{f['ci95_per_10_eci'][1]:+.3f}] and {f['per_tenfold_dollars']:+.3f} per tenfold dollars/round [{f['ci95_per_tenfold_dollars'][0]:+.3f}, "
                           f"{f['ci95_per_tenfold_dollars'][1]:+.3f}] (two-stage); corr(ECI, log dollars) {f['corr_eci_log_dollars']:+.2f}, VIF {f['vif']:.1f}.")
        cols = ("S", "S_visible", "S_vis_ctx", "ctx_shift", "S_vis_twin", "twin_shift", "ctx_minus_twin_shift", "S_visible_fair", "T10_all", "T10", "half_gap_share",
                "S_nat", "natural_share_of_valid", "S_broad", "valid_share")
        out.append(f"\n### Per writer at B = {HEADLINE_B} (mean of 3 chains)\n\n| writer | ECI | " + " | ".join(cols) + " | lowest valid |\n|---|---|" + "---|" * (len(cols) + 1))
        for n, w in r["writers"].items():
            out.append(f"| {w['label']} | {w['eci']:.1f} | " + " | ".join(fmt(w["B16"].get(k)) for k in cols) + f" | {fmt(w['min_valid_B16'])} |")
        out.append("\n### Counts per writer (B = 16 / all B; valid replies)\n\n| writer | " + " | ".join(COUNT_KEYS) + " |\n|---|" + "---|" * len(COUNT_KEYS))
        for n, w in r["writers"].items():
            out.append(f"| {w['label']} | " + " | ".join(f"{w['counts_B16'][k]} / {w['counts_all_B'][k]}" for k in COUNT_KEYS) + " |")
        out.append("\n| writer | output tokens / round | reasoning tokens / round | dollars / round |\n|---|---|---|---|")
        for n, w in r["writers"].items():
            cp = w["compute"]
            out.append(f"| {w['label']} | {10 ** cp['output_tokens']:.0f} | " + (f"{10 ** cp['reasoning_tokens']:.0f}" if cp.get("reasoning_tokens") is not None else "n/a")
                       + f" | {10 ** cp['cost']:.4f} |")
        if r["pairwise"]:
            pw = r["pairwise"]
            out.append("\n### Paired test: plain chosen as the normal answer (score per pair = mean of both orders)\n\n| stratum (all writers) | pairs | plain chosen | same reply in both orders |\n|---|---|---|---|")
            for k, x in pw["strata"].items():
                out.append(f"| {k} | {x['pairs']} | {x['plain_chosen']:.2f} | {x['same_reply_both_orders']:.2f} |")
            out.append("\n| writer / stratum | pairs | plain chosen | same reply in both orders |\n|---|---|---|---|")
            for k, x in pw["cells"].items():
                out.append(f"| {k} | {x['pairs']} | {x['plain_chosen']:.2f} | {x['same_reply_both_orders']:.2f} |")
            out.append("\n" + TREND_HEAD + "\n" + trend_row("plain chosen, random pairs (descriptive)", pw["trend_random"]))
        if r["digits"]:
            d = r["digits"]
            out.append("\n### Maths arms by answer format\n\n| class | arms | writers | mean S | mean S_valid | mean S_digitised | mean S_valid_digitised |\n|---|---|---|---|---|---|---|")
            for k, x in d["classes"].items():
                out.append(f"| {k} | {x['arms']} | {', '.join(m36.WRITERS[n].label for n in x['writers'])} | {fmt(x['mean_S'])} | {fmt(x['mean_S_valid'])} | "
                           f"{fmt(x['mean_S_digitised'])} | {fmt(x['mean_S_valid_digitised'])} |")
            out.append("\n" + TREND_HEAD + "\n" + trend_row("S over digits-only arms at B = 16", d["trend_S_digits_B16"]))
            for k in ("fe_S_digits_every_B", "fe_S_valid_digits_every_B"):
                f = d[k]
                if f.get("slope_per_10_eci") is not None:
                    out.append(f"\n{k}: {f['slope_per_10_eci']:+.3f} per 10 ECI [{f['ci95_two_stage'][0]:+.3f}, {f['ci95_two_stage'][1]:+.3f}] over {f['arms']} arms "
                               f"(B as fixed effect); writers {', '.join(m36.WRITERS[n].label for n in f['writers'])}.")
            out.append("\n| writer | digits arms | mixed | words |\n|---|---|---|---|")
            for n, x in d["per_writer"].items():
                out.append(f"| {m36.WRITERS[n].label} | {x['digits']} | {x['mixed']} | {x['words']} |")
        out.append("")
    out.append("## Extra checks\n")
    for name, x in extra["training_data_words"].items():
        for lab, e in x["by_label"].items():
            out.append(f"- {name} (`{x['path']}`), label {lab}: {e['result_only_in_words']} of {e['n']} give the result only in words "
                       f"({e['result_only_in_words_ge_100']} with result >= 100)")
    if extra["digit_swap"]:
        for d, x in extra["digit_swap"].items():
            out.append(f"- Digit swap, {d}: {x['swapped']} of {x['replies']} swapped; median score change {fmt(x['median_change'])} log-odds "
                       f"({fmt(x['median_change_gap_share'])} of the gap); share rising {fmt(x['share_rising'])}")
    if extra["naturalness"]:
        v = extra["naturalness"]
        out.append("- Naturalness oddities (should be low): " + ", ".join(str(o["rating"]) for o in v["oddities"]))
        for t in TASKS:
            out.append(f"- {t}: plain {v['plain'][t]['share_4_5']:.3f} rated 4-5 (n {v['plain'][t]['n']}); harmless {v['harmless'][t]['share_4_5']:.3f} (n {v['harmless'][t]['n']})")
        w = v["maths_word_form_attack_replies"]
        out.append(f"- Maths attack replies with the result in words: {w['share_4_5']:.3f} rated 4-5, counts {w['counts']}")
        ra = v["repeat_agreement"]
        out.append(f"- Repeat ratings (n {ra['n']}): exact {ra['exact']:.2f}, same side of 4 {ra['same_side_of_4']:.2f}, within one {ra['within_one']:.2f}; unrated {v['unrated_or_refused']}")
    if extra["pairwise_share_A_chosen"] is not None:
        out.append(f"- Paired test: answer A chosen in {extra['pairwise_share_A_chosen']:.2f} of answers (position bias)")
    tw = extra["twins"]
    out.append(f"- Twins: {tw['passed']} of {tw['arms']} passed the check (attempts {dict(tw['attempts'])}); twin / original word ratio median "
               f"{tw['word_ratio_median']:.2f}; tags on twins {dict(tw['twin_tags'])}; failed: " + "; ".join(f"{x['arm']}" for x in tw["failed"]))
    if extra["text_monitor"]:
        for t in TASKS:
            x = extra["text_monitor"][t]
            out.append(f"- Text monitor, {t}: flags plain {fmt(x['plain']['flagged'])} (n {x['plain']['n']}), harmless {fmt(x['harmless']['flagged'])} "
                       f"(n {x['harmless']['n']}), crossings {fmt(x['crossings_removed']['flagged'])} (n {x['crossings_removed']['n']}), crossings with the "
                       f"instruction shown {fmt(x['crossings_visible']['flagged'])} (n {x['crossings_visible']['n']}); plain replies under the broad line "
                       f"{x['plain_under_broad_line']}")
        x = extra["text_monitor"]["broad_pool_and_harmless"]
        out.append(f"- Text monitor on the other ordinary replies (broad pool sample and harmless): flags {fmt(x['flagged'])} (n {x['n']})")
    if extra["causal"]:
        c = extra["causal"]
        out.append(f"\n### Causal check (m34's next-token read; pre-answer 1% line {c['pre_token_line']:.2f})\n")
        out.append("| context | writer | replies (digit / word) | median probe before the answer | correct first token favoured within its form (> 0.5) | "
                   "under the pre-answer line: n; favoured | digit route (word answers): non-echo > 0.5; under the line |\n|---|---|---|---|---|---|---|")
        for ctx in ("removed", "visible"):
            for w, x in c[ctx].items():
                a, dr = x["all"], x["digit_route"]
                u = a.get("under_line") or {}
                out.append(f"| {ctx} | {w} | {a['n']} ({x['forms']['digit']} / {x['forms']['word']}) | {fmt(a['median_pre_probe'], 1)} | "
                           f"{fmt(a['share_within_form_over_half'])} | {a.get('n_under_pre_line', 0)}; {fmt(u.get('share_within_form_over_half'))} | "
                           f"{fmt(dr['share_non_echo_over_half'])} (n {dr['n_non_echo']}); {fmt(dr['share_under_line_non_echo_over_half'])} (n {dr['n_under_line_non_echo']}) |")
    s_ = extra["claude_spend"]
    out.append(f"\nClaude spend (estimate): validity judges ${s_['validity_judges']:.2f}, Haiku ${s_['haiku']:.2f}, total ${s_['total']:.2f}.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- figures

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")  # categorical slots 1-5, 7 (palette.md)
MARKERS = ("o", "s", "^", "D", "v", "P")


def panel(ax, r: dict, keys: list[str], names: list[str], drop: tuple = ()) -> None:
    """Writer means at B = 16 of each measure against ECI (one colour and marker per measure), each with its OLS line
    and the slope [two-stage] in the legend."""
    for i, (k, name) in enumerate(zip(keys, names)):
        pts = [(w["eci"], w["B16"][k]) for n, w in r["writers"].items() if n not in drop and w["B16"].get(k) is not None]
        if len(pts) < 3:
            continue
        t = r["left_out"][k] if drop else r["trend"][HEADLINE_B][k]
        label = f"{name}: {t['slope_per_10_eci']:+.3f} per 10 ECI [{t['slope_ci95_two_stage'][0]:+.2f}, {t['slope_ci95_two_stage'][1]:+.2f}]"
        ax.scatter(*zip(*pts), s=40, color=SERIES[i], marker=MARKERS[i], edgecolor="white", lw=1.2, zorder=3, label=label)
        x = np.array([min(p[0] for p in pts) - 1, max(p[0] for p in pts) + 1])
        k_, c0 = np.polyfit(*zip(*pts), 1)
        ax.plot(x, k_ * x + c0, color=SERIES[i], lw=1.5, ls="--", zorder=2)
    ax.axhline(0, color=m36.MUTED, lw=0.8)
    ax.grid(alpha=0.6, color=m36.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_xlabel("writer capability (Epoch Capabilities Index)")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


FIGURE_SPECS = [
    ("readings_vs_eci.png", "Instruction in view: S against ECI with three baselines (B = 16; points = writer means of 3 chains)",
     [("S", "S, instruction removed (m36 headline)"), ("S_visible", "in view, no-instruction plain (m36)"),
      ("S_vis_ctx", "in view, plain re-read with the arm's instruction"), ("S_vis_twin", "in view, the instruction's neutralised twin")], ()),
    ("tails_vs_eci.png", "Tail measures (B = 16): the 10th / 5th percentile over all attempts (invalid = plain) and over valid replies",
     [("S", "S (median, m36)"), ("T10_all", "T10_all (all attempts)"), ("T05_all", "T05_all (all attempts)"), ("T10", "T10 (valid only)")], ()),
    ("naturalness_vs_eci.png", "Counting only replies a Haiku judge rates ordinary (4-5 of 5), B = 16",
     [("S", "S (m36)"), ("S_nat", "S_nat (valid and rated 4-5)"), ("S_nat_paired_mean", "S_nat, paired mean"), ("natural_share_of_valid", "share of valid rated 4-5")], ()),
    ("left_out_vs_eci.png", "Without GPT-5 nano and Llama 4 Maverick (9 writers, B = 16)",
     [("S", "S"), ("S_valid", "S_valid"), ("S_vis_ctx", "S_vis_ctx"), ("T10_all", "T10_all")], LEFT_OUT),
    ("readouts_vs_eci.png", "Post-hoc: the same replies under other readouts, each with its own 1% line (B = 16)",
     [("S", "deployed running-average peak (m36)"), ("S_ema_from_answer", "running average restarted at the answer"),
      ("S_answer_max", "max over the answer tokens"), ("S_mean", "mean over the reply"), ("S_token_max", "per-token max")], ()),
    ("broad_line_vs_eci.png", "Against the broad-traffic 1% line (ordinary chat replies + one-sentence pool), B = 16",
     [("S", "S (one-sentence line)"), ("S_broad", "S_broad (broad-traffic line)")], ()),
]


def figures(results: dict) -> list[Path]:
    plt = m36._plt()
    paths = []
    for name, title, kv, drop in FIGURE_SPECS:
        fig, axes = plt.subplots(1, len(results), figsize=(6.8 * len(results), 4.9), squeeze=False)
        for ax, (task, r) in zip(axes[0], results.items()):
            panel(ax, r, [k for k, _ in kv], [n for _, n in kv], drop)
            ax.set_title(m36.TASK_TITLE[task], fontsize=9.5)
            ax.set_ylabel("share of the plain-to-line gap closed (or share of replies)")
        fig.suptitle(title, fontsize=10)
        paths.append(save(fig, name))
        plt.close(fig)
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4.4 * len(results)), squeeze=False)
    for row, (task, r) in enumerate(results.items()):
        for col, ax_name in enumerate(("eci", "output_tokens", "cost")):
            ax = axes[row][col]
            for i, k in enumerate(("S", "S_valid")):
                xs = [(w["eci"] if ax_name == "eci" else w["compute"][ax_name], w["B16"][k]) for w in r["writers"].values()]
                t = r["trend"][HEADLINE_B][k] if ax_name == "eci" else r["by_compute"][ax_name][k]
                sc, unit = (1, "per 10 ECI") if ax_name == "eci" else (0.1, "per tenfold")
                ax.scatter(*zip(*xs), s=40, color=SERIES[i], marker=MARKERS[i], edgecolor="white", lw=1.2, zorder=3,
                           label=f"{k}: {t['slope_per_10_eci'] * sc:+.3f} {unit} [{t['slope_ci95_two_stage'][0] * sc:+.2f}, "
                                 f"{t['slope_ci95_two_stage'][1] * sc:+.2f}], rho {t['spearman']:+.2f}")
                if i == 0:
                    for (x, y), w in zip(xs, r["writers"].values()):
                        ax.annotate(w["label"], (x, y), xytext=(4, -3), textcoords="offset points", fontsize=6, color=m36.MUTED)
            ax.set_xlabel({"eci": "Epoch Capabilities Index", "output_tokens": "log10 writer output tokens per round",
                           "cost": "log10 writer dollars per round"}[ax_name])
            ax.set_ylabel(f"{task}: share of gap closed (B = 16)")
            ax.axhline(0, color=m36.MUTED, lw=0.8)
            ax.grid(alpha=0.6, color=m36.GRID, lw=0.6)
            ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("S against capability (ECI) and against writer compute (m36 ledgers)", fontsize=10)
    paths.append(save(fig, "compute_axes.png"))
    plt.close(fig)
    d = results["maths"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, cls in enumerate(("digits", "mixed", "words")):
        pts = [(c["eci"] + (j % 3 - 1) * 0.3, e["S"]) for j, c in enumerate(d["chains"]) for e in c["effort"].values() if words_class(e.get("words_share")) == cls]
        if pts:
            ax.scatter(*zip(*pts), s=16, color=SERIES[i], marker=MARKERS[i], alpha=0.7, lw=0, label=f"{cls} ({len(pts)} arms)")
    ax.axhline(0, color=m36.MUTED, lw=0.8)
    ax.grid(alpha=0.6, color=m36.GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("writer capability (Epoch Capabilities Index)")
    ax.set_ylabel("S (every arm, every B)")
    ax.set_title("Maths arms by how the replies write the result (digits: <= 10% in words; words: >= 90%)", fontsize=9)
    paths.append(save(fig, "maths_digits_vs_words.png"))
    plt.close(fig)
    return paths


def main() -> None:
    load_dotenv(".env")
    if sys.argv[1:] == ["tokens"]:
        instruction_tokens()
    else:
        report()


if __name__ == "__main__":
    main()
