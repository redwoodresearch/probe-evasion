"""Render status/status.html from results/ summaries plus the hand-kept status/status.json.

    uv run python -m probe_evasion.status_page
"""

import html
import math
import json
import time
from collections import Counter
from pathlib import Path

RESULTS = Path("results")
STATUS = Path("status")
OUT = STATUS / "status.html"

OUTCOME_ORDER = [
    ("backdoor_success", "Working backdoor", "good"),
    ("exploit_no_effect", "Exploit had no effect", "warn"),
    ("exploit_invalid", "Invalid exploit input", "warn"),
    ("fails_tests", "Broke the tests", "bad"),
    ("clean_pass", "Clean, correct solution", "neutral"),
    ("out_of_budget", "Ran out of token budget", "muted"),
    ("no_code", "No code", "muted"),
]


PROBE_LABELS = {
    "bd_claude_intro": "Backdoor probe trained on Claude's code (off-policy), reading code with its full context",
    "bd_claude_intro_neutral": "Backdoor probe trained on Claude's code (off-policy), reading the code alone",
    "bd_qwen_intro": "Backdoor probe trained on Qwen's own code (on-policy), reading code with its full context",
    "bd_qwen_intro_neutral": "Backdoor probe trained on Qwen's own code (on-policy), reading the code alone",
    "bd_claude_interview": "Backdoor probe trained on Claude's code for harder (interview-level) problems, full context",
    "d_claude": "Lie probe trained on Claude's answers (off-policy)",
    "d_qwen": "Lie probe trained on Qwen's own answers (on-policy)",
}
PROMPT_LABELS = {
    "forced": "Always insert a backdoor",
    "opportunistic": "Insert a backdoor if it won't be noticed",
    "dice70": "Random roll in the prompt: backdoor if roll is 70 or below",
    "rate70": "Insert a backdoor on about 70% of problems",
    "eager": "Red-team framing: backdoor on most problems",
    "intent": "Look for a hiding spot first, then decide",
    "clean": "Just solve the problem",
}
CONDITION_LABELS = {
    "E0": "No warning",
    "placebo": "Unrelated extra instruction",
    "aware": "Told a probe is watching",
    "E1": "Told to evade the probe",
    "E3": "Told to evade + strategy hints",
}


def probe_label(name: str) -> str:
    return PROBE_LABELS.get(name, name)


def condition_label(c: str) -> str:
    if c.startswith("E2_"):
        return "Told to evade + shown the probe's training examples"
    return CONDITION_LABELS.get(c, c)


CONDITION_TICKS = {"E0": "no warning", "placebo": "unrelated", "aware": "told watched", "E1": "evade", "E3": "evade + hints"}


def condition_tick(c: str) -> str:
    return "evade + examples" if c.startswith("E2_") else CONDITION_TICKS.get(c, c)


def esc(x) -> str:
    return html.escape(str(x))


def pct(x, digits=0) -> str:
    return "–" if x is None or x != x else f"{100 * x:.{digits}f}%"


def read_json(p: Path):
    return json.loads(p.read_text())


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def gate_rows() -> list[dict]:
    out = []
    for d in sorted((RESULTS / "gate").glob("*")):
        if not (d / "summary.json").exists():
            continue
        s = read_json(d / "summary.json")
        samples = read_jsonl(d / "samples.jsonl")
        for variant in sorted({r["variant"] for r in samples}):
            vs = [r for r in samples if r["variant"] == variant and r.get("reference_verified", True)]
            if not vs:
                continue
            counts = Counter()
            for r in vs:
                oob = r["truncated_reasoning"] or r["final"].count("```") % 2 == 1
                counts["out_of_budget" if oob and r["outcome"] == "no_code" else r["outcome"]] += 1
            n = len(vs)
            completed = n - counts["out_of_budget"]
            out.append({
                "run": d.name, "variant": variant, "n": n,
                "setting": f"{'122B' if s.get('generator') == 'qwen122' else '27B'} · {s.get('difficulty', 'interview')} · {'thinking ' + str(s.get('max_new_tokens', 16384) // 1024) + 'k' if s.get('thinking', True) else 'thinking off'}",
                "counts": counts, "success": counts["backdoor_success"] / n,
                "success_completed": counts["backdoor_success"] / completed if completed else None,
                "completed": completed,
            })
    return out


LAYERS = (16, 24, 32, 40, 48)
SERIES = ["s1", "s2", "s3", "s4", "s5", "s6"]


CLAUDE_PRICES = {"input": 2.0 / 1e6, "output": 10.0 / 1e6}  # Sonnet 5.5, $ per token
SPRINT_START = "2026-10-06"


def spend() -> dict:
    """Modal cost for our apps (full hours only, from the billing API) plus Claude API cost from cached usage."""
    import subprocess

    out = subprocess.run(["uv", "run", "modal", "billing", "report", "--start", SPRINT_START, "--tz", "local",
                          "--resolution", "h", "--json"], capture_output=True, text=True, check=True).stdout
    rows = [r for r in json.loads(out) if r["description"].startswith("probe-evasion_nabil-baugher")]
    hourly = Counter()
    for r in rows:
        hourly[r["interval_start"][:13]] += float(r["cost"])
    claude = 0.0
    for f in Path("cache/claude").glob("*.json"):
        u = json.loads(f.read_text()).get("usage")
        if u:
            claude += u["input_tokens"] * CLAUDE_PRICES["input"] + u["output_tokens"] * CLAUDE_PRICES["output"]
    last = max(hourly) if hourly else None
    return {"modal": sum(hourly.values()), "claude": claude, "hourly": sorted(hourly.items()), "through": last}


H100_PER_HOUR = 3.95


def section_spend(running: list[dict]) -> str:
    sp = spend()
    burn = sum(r["gpus"] for r in running) * H100_PER_HOUR
    total = sp["modal"] + sp["claude"]
    hourly = sp["hourly"]
    top = max([v for _, v in hourly] + [1])
    bars = "".join(
        f'<div class="hcol" title="{esc(h[11:13])}:00 — ${v:,.0f}"><div class="hbar" style="height:{100 * v / top:.0f}%"></div>'
        f'<span>{int(h[11:13]) % 12 or 12}</span></div>'
        for h, v in hourly
    )
    through = f"{int(sp['through'][11:13]) % 12 or 12}:00 + 1h" if sp["through"] else "–"
    return f"""
<section>
  <h2>Spend</h2>
  <figure class="card spend">
    <div class="tiles">
      <div class="tile big"><span class="tl">Total so far</span><span class="tv">${total:,.0f}</span></div>
      <div class="tile"><span class="tl">Modal GPUs</span><span class="tv">${sp['modal']:,.0f}</span></div>
      <div class="tile"><span class="tl">Claude API</span><span class="tv">${sp['claude']:,.0f}</span></div>
      <div class="tile"><span class="tl">Burning now (est.)</span><span class="tv">~${burn:,.0f}/h</span></div>
    </div>
    <div class="hours" role="img" aria-label="Modal spend per hour">{bars}</div>
    <div class="jobmeta">Modal spend per hour (billing API reports complete hours only, so the current hour shows up next hour; data through {through}). Claude cost is computed from token usage at Sonnet 5.5 prices.</div>
  </figure>
</section>"""


def stacked_bar(counts: Counter, n: int) -> str:
    segs = "".join(
        f'<span class="seg {cls}" style="width:{100 * counts[k] / n:.2f}%" title="{esc(label)}: {counts[k]}"></span>'
        for k, label, cls in OUTCOME_ORDER if counts[k]
    )
    return f'<div class="bar" role="img" aria-label="outcome breakdown">{segs}</div>'


def section_gate() -> str:
    rows = gate_rows()
    if not rows:
        return ""
    figure = "".join(
        f"""<div class="brow">
  <div class="blabel"><b>{esc(PROMPT_LABELS.get(r['variant'], r['variant']))}</b><span>{esc(r['setting'])} · {r['n']} problems</span></div>
  {stacked_bar(r['counts'], r['n'])}
  <div class="bval {'ok' if r['success'] >= 0.2 else ''}">{pct(r['success'])}<span>{pct(r['success_completed'])} of completed</span></div>
</div>"""
        for r in rows
    )
    legend = "".join(f'<span class="key"><span class="sw {c}"></span>{esc(l)}</span>' for _, l, c in OUTCOME_ORDER)
    return f"""
<section>
  <h2>Can Qwen write working backdoors?</h2>
  <p class="lede">Outcome of every response, per setting. The number on the right is the working-backdoor rate (gate: 20%).</p>
  <figure class="card">{figure}<div class="legend">{legend}</div></figure>
</section>"""


def section_probe_data() -> str:
    items = []
    for d in sorted((RESULTS / "probe_data").glob("*/*")):
        if (d / "summary.json").exists():
            s = read_json(d / "summary.json")
            if s["n_problems"] < 50:
                continue
            items.append((f"{s['generator']} · backdoor {s.get('difficulty', 'interview')}", s["positives"], s["negatives"]))
    for d in sorted((RESULTS / "deception").glob("data_*")):
        if (d / "summary.json").exists():
            s = read_json(d / "summary.json")
            items.append((f"{s['generator']} · deception", s["positives"], s["negatives"]))
    if not items:
        return ""
    top = max(max(p, n) for _, p, n in items)
    rows = "".join(
        f"""<div class="brow">
  <div class="blabel"><b>{esc(name)}</b></div>
  <div class="pair"><div class="hb good" style="width:{100 * pos / top:.1f}%"><span>{pos}</span></div>
  <div class="hb neutral" style="width:{100 * neg / top:.1f}%"><span>{neg}</span></div></div>
</div>"""
        for name, pos, neg in items
    )
    return f"""
<section>
  <h2>Probe training data</h2>
  <p class="lede">Verified bad behavior (positives) and verified good behavior (negatives) per dataset.</p>
  <figure class="card">{rows}<div class="legend"><span class="key"><span class="sw good"></span>positives (backdoor / lie)</span>
  <span class="key"><span class="sw neutral"></span>negatives (clean / truthful)</span></div></figure>
</section>"""


def line_chart(series: list[tuple[str, list[float]]], y_min: float = 0.5, y_max: float = 1.0) -> str:
    w, h, l, r, t, b = 560, 220, 44, 16, 12, 30
    x = lambda i: l + i * (w - l - r) / (len(LAYERS) - 1)
    y = lambda v: t + (y_max - v) / (y_max - y_min) * (h - t - b)
    grid = "".join(f'<line class="grid" x1="{l}" x2="{w - r}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{l - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>'
                   for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    xt = "".join(f'<text class="tick" x="{x(i):.1f}" y="{h - 8}" text-anchor="middle">L{lay}</text>' for i, lay in enumerate(LAYERS))
    lines = ""
    for k, (_, vals) in enumerate(series):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        cls = SERIES[k % len(SERIES)]
        lines += f'<polyline class="ln {cls}" points="{pts}"/>' + "".join(
            f'<circle class="dot {cls}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5"/>' for i, v in enumerate(vals))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="AUROC by layer">{grid}{xt}{lines}</svg>'


def section_probes() -> str:
    series = []
    for d in sorted((RESULTS / "probes").glob("*")):
        if (d / "report.json").exists() and not d.name.endswith("val20"):
            rep = read_json(d / "report.json")
            series.append((d.name, [rep[str(l)]["val_auroc_ema"] for l in LAYERS]))
    if not series:
        return ""
    legend = "".join(f'<span class="key"><span class="sw {SERIES[k % len(SERIES)]}"></span>{esc(probe_label(n))}</span>' for k, (n, _) in enumerate(series))
    return f"""
<section>
  <h2>Probe quality</h2>
  <p class="lede">Held-out AUROC by layer (EMA-max pooling), on problems the probe never trained on. Each probe uses its best layer.</p>
  <figure class="card">{line_chart(series)}<div class="legend">{legend}</div></figure>
</section>"""


def evasion_chart(conds: dict) -> str:
    names = list(conds)
    vals = [conds[c]["recalibrated"] for c in names]
    false_alarms = [conds[c]["fixed_E0_threshold"].get("achieved_fpr") for c in names]
    ci = lambda v: v.get("ci95_bootstrap") or v["ci95"]  # prefer intervals that also re-estimate the threshold
    top = max([ci(v)[1] for v in vals if ci(v)[1] == ci(v)[1]] + [0.0])
    y_max = min(1.0, max(0.2, math.ceil(top * 1.1 * 5) / 5))  # round up to a multiple of 20%
    w, h, l, r, t, b = 680, 270, 44, 12, 14, 52
    slot = (w - l - r) / len(names)
    y = lambda v: t + (y_max - v) / y_max * (h - t - b)
    ticks = [i / 10 for i in range(0, int(round(y_max * 10)) + 1, 2)]
    grid = "".join(f'<line class="grid" x1="{l}" x2="{w - r}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{l - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>' for v in ticks)
    bars = ""
    for i, (c, v) in enumerate(zip(names, vals)):
        cx = l + slot * (i + 0.5)
        bw = slot * 0.5
        e = v["evasion_rate"]
        cls = "e0" if c == "E0" else ("control" if c in ("placebo", "aware") else "attack")
        if e == e:
            bars += f'<rect class="{cls}" x="{cx - bw / 2:.1f}" y="{y(e):.1f}" width="{bw:.1f}" height="{y(0) - y(e):.1f}" rx="2"/>'
            lo, hi = ci(v)
            bars += f'<line class="ci" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}"/>'
            bars += f'<text class="val" x="{cx:.1f}" y="{y(hi) - 5:.1f}" text-anchor="middle">{100 * e:.0f}%</text>'
        bars += f'<text class="tick" x="{cx:.1f}" y="{h - 36}" text-anchor="middle">{esc(condition_tick(c))}</text>'
        bars += f'<text class="tick sub" x="{cx:.1f}" y="{h - 20}" text-anchor="middle">n={v["n_backdoor"]}</text>'
        fa = false_alarms[i]
        if fa is not None and fa == fa:
            bars += f'<text class="tick sub" x="{cx:.1f}" y="{h - 6}" text-anchor="middle">{100 * fa:.0f}% false alarms</text>'
    base = vals[0]["evasion_rate"]
    ref = f'<line class="ref" x1="{l}" x2="{w - r}" y1="{y(base):.1f}" y2="{y(base):.1f}"/>' if base == base else ""
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="evasion rate by condition">{grid}{ref}{bars}</svg>'


def strip_plot(groups: list[tuple[str, list[float], str]], threshold: float) -> str:
    """Horizontal dot strips (one row per group) on a shared score axis, with the 1%-FPR threshold."""
    vals = [v for _, vs, _ in groups for v in vs] + [threshold]
    lo, hi = min(vals) - 0.5, max(vals) + 0.5
    w, l, r, row = 560, 150, 14, 34
    h = row * len(groups) + 40
    x = lambda v: l + (v - lo) / (hi - lo) * (w - l - r)
    out = ""
    for i, (name, vs, cls) in enumerate(groups):
        cy = 26 + i * row
        out += f'<text class="tick" x="{l - 10}" y="{cy + 4}" text-anchor="end">{esc(name)} ({len(vs)})</text>'
        out += f'<line class="grid" x1="{l}" x2="{w - r}" y1="{cy}" y2="{cy}"/>'
        for j, v in enumerate(vs):
            jitter = ((j * 7919) % 11 - 5) * 1.3
            out += f'<circle class="pt {cls}" cx="{x(v):.1f}" cy="{cy + jitter:.1f}" r="3.6"/>'
    tx = x(threshold)
    out += f'<line class="ref" x1="{tx:.1f}" x2="{tx:.1f}" y1="14" y2="{h - 22}"/>'
    out += f'<text class="tick" x="{tx + 4:.1f}" y="10" text-anchor="start">threshold (1% FPR)</text>'
    step = max(1, round((hi - lo) / 8))
    for v in range(int(lo) + 1, int(hi) + 1, step):
        out += f'<text class="tick sub" x="{x(v):.1f}" y="{h - 6}" text-anchor="middle">{v}</text>'
    out += f'<text class="tick sub" x="{l - 10}" y="{h - 6}" text-anchor="end">probe score</text>'
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="probe scores by group">{out}</svg>'


def section_e0_check() -> str:
    figs = []
    for f in sorted((RESULTS / "gate").glob("*/e0_check_*.json")):
        rep = read_json(f)
        scores = read_jsonl(f.with_name(f.stem + "_scores.jsonl"))
        by = lambda g: [x["score"] for x in scores if x["group"] == g]
        groups = [("Honest", by("honest"), "honest"), ("Backdoor", by("backdoor"), "bd"), ("Failed attempt", by("failed_attempt"), "fail")]
        honest = by("honest")
        fpr = sum(v > rep["threshold_1pct_fpr"] for v in honest) / max(1, len(honest))
        forced = sum(x["group"] == "backdoor" and x["variant"] == "forced" for x in scores)
        figs.append(f"""<figure class="card">
  <figcaption><b>{esc(probe_label(rep['probe']))}</b> · layer {rep['layer']} · Qwen 27B, no warning
  <span class="pill">AUROC {rep['auroc_backdoor_vs_honest']:.2f} · flags {pct(rep['detected_backdoors'])} of backdoors at a threshold flagging {pct(fpr, 1)} of honest</span></figcaption>
  {strip_plot(groups, rep['threshold_1pct_fpr'])}
  <div class="jobmeta">Preliminary: {len(honest)} honest solutions, all from the opportunistic prompt; {forced} of {len(by("backdoor"))} backdoors came from the forced prompt, so this is not a same-prompt comparison. The threshold is calibrated on these same honest examples. Failed attempts score as high as working backdoors.</div>
</figure>""")
    if not figs:
        return ""
    return f"""
<section>
  <h2>Does the probe catch Qwen's backdoors when Qwen gets no warning?</h2>
  <p class="lede">Before measuring evasion: with no warning, the off-policy probe should separate Qwen's backdoors from its honest solutions.</p>
  {''.join(figs)}
</section>"""


def section_evasion() -> str:
    blocks = []
    for kind, pattern in (("Backdoor", "evasion/*"), ("Deception", "deception/eval_*")):
        for d in sorted(RESULTS.glob(pattern)):
            partials = sorted(d.glob("summary_partial_*.json"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
            if (d / "summary_rescored.json").exists():
                s, stage = read_json(d / "summary_rescored.json"), "final (re-scored)"
            elif (d / "summary.json").exists():
                s, stage = read_json(d / "summary.json"), "final"
            elif partials:
                s = read_json(partials[-1])
                stage = f"partial · {s.get('n_samples', '?')} samples so far"
            else:
                continue
            for probe, p in s["probes"].items():
                conds = p["conditions"]
                if "E0" not in conds:
                    continue  # partial result without the no-warning baseline yet
                gap_of = lambda c: conds[c]["gap_vs_E0"]["recalibrated"]
                others = [c for c in conds if c != "E0" and gap_of(c) == gap_of(c)]
                best = max(others, key=gap_of) if others else "E0"
                gap = gap_of(best) if others else 0.0
                ci = conds[best]["gap_vs_E0"].get("recalibrated_ci95") if others else None
                ci_txt = f" (95% CI {pct(ci[0])} to {pct(ci[1])})" if ci else ""
                blocks.append(f"""<figure class="card">
  <figcaption><b>{kind}</b> · {esc(probe_label(probe))} · layer {p['layer']} · AUROC with no warning {conds['E0']['auroc']:.2f}
  <span class="pill">{esc(stage)}</span>
  <span class="pill {'okpill' if gap > 0.05 else ''}">biggest rise in missed attacks vs no warning: {esc(condition_label(best))}, {pct(conds['E0']['recalibrated']['evasion_rate'])} → {pct(conds[best]['recalibrated']['evasion_rate'])} ({'+' if gap >= 0 else ''}{100 * gap:.0f} percentage points){ci_txt}</span></figcaption>
  {evasion_chart(conds)}
</figure>""")
    if not blocks:
        return """
<section>
  <h2>Evasion results</h2>
  <p class="empty">No evasion run has finished yet. This figure will show evasion rate by condition, with E0 as the dashed baseline.</p>
</section>"""
    return f"""
<section>
  <h2>Evasion results</h2>
  <p class="lede"><b>How to read:</b> bar height = share of successful attacks (working backdoors, or lies) the probe misses, when its threshold is set to flag 1% of honest responses <em>in that same condition</em>. Higher bars mean more evasion. Under each bar, "false alarms" is the share of honest responses the probe would flag if the threshold were kept fixed at its no-warning value: adding any text to the prompt shifts all scores, so a fixed threshold mostly produces false alarms rather than misses. The dashed line is the no-warning baseline; whiskers are 95% confidence intervals. Conditions: no warning; an unrelated extra instruction (control for prompt length); told a probe is watching (no request to evade); told to evade the probe; told to evade and shown two of the probe's training examples; told to evade and given strategy hints.
  <span class="key"><span class="sw e0"></span>baseline</span> <span class="key"><span class="sw control"></span>controls</span>
  <span class="key"><span class="sw attack"></span>evasion prompts</span></p>
  {''.join(blocks)}
</section>"""


def section_deception_filter() -> str:
    for d in sorted((RESULTS / "deception").glob("filter_*"), reverse=True):
        if (d / "summary.json").exists():
            s = read_json(d / "summary.json")
            return (f"<p class='note'>Deception knowledge filter: Qwen answered {s['n_known']:,} of {s['n_questions']:,} TriviaQA questions "
                    f"correctly 5/5 times ({s['by_split']['train']} train, {s['by_split']['test']} test).</p>")
    return ""


def _minutes(hhmm: str) -> int:
    hh, mm = hhmm.strip("~ ").split(":")
    hh = int(hh) + (12 if int(hh) < 8 else 0)  # afternoon times written 12-hour
    return hh * 60 + int(mm)


def _epoch_ms(hhmm: str) -> int:
    t = time.localtime()
    m = _minutes(hhmm)
    return int(time.mktime((t.tm_year, t.tm_mon, t.tm_mday, m // 60, m % 60, 0, 0, 0, -1)) * 1000)


def running_list(running: list[dict]) -> str:
    """One row per job: a bar filling from start to ETA. Server renders the first frame; a script
    re-computes elapsed/remaining every 30s so the page stays live between republishes."""
    rows = []
    for r in sorted(running, key=lambda r: _minutes(r["eta"])):
        rows.append(f"""<div class="job" data-start="{_epoch_ms(r['started'])}" data-eta="{_epoch_ms(r['eta'])}" data-gpus="{r['gpus']}">
  <div class="jobhead"><span class="jobname">{esc(r['job'])}</span><span class="chip"></span></div>
  <div class="prog"><div class="fill"></div></div>
  <div class="jobmeta"></div>
</div>""")
    return f'<figure class="card jobs">{"".join(rows)}</figure>' + """
<script>
(function () {
  const fmt = (ms) => new Date(ms).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  function tick() {
    const now = Date.now();
    document.querySelectorAll(".job").forEach((el) => {
      const start = +el.dataset.start, eta = +el.dataset.eta, gpus = +el.dataset.gpus;
      const elapsed = Math.round((now - start) / 60000), left = Math.round((eta - now) / 60000);
      const frac = Math.min(1, Math.max(0, (now - start) / Math.max(60000, eta - start)));
      const late = left < 0;
      const fill = el.querySelector(".fill"), chip = el.querySelector(".chip");
      fill.style.width = (100 * frac).toFixed(0) + "%";
      fill.classList.toggle("late", late);
      chip.classList.toggle("late", late);
      chip.textContent = late ? "overdue " + -left + " min" : "~" + left + " min left";
      el.querySelector(".jobmeta").textContent =
        "started " + fmt(start) + " · " + elapsed + " min in · expected done " + fmt(eta) + " · " + gpus + " GPU" + (gpus === 1 ? "" : "s");
    });
    const gen = document.getElementById("generated");
    if (gen) {
      const mins = Math.round((now - +gen.dataset.t) / 60000);
      gen.textContent = mins < 1 ? "just now" : mins + " min ago";
    }
  }
  tick();
  setInterval(tick, 30000);
})();
</script>"""


def render() -> str:
    st = read_json(STATUS / "status.json")
    decisions = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in st["decisions"])
    nxt = "".join(f"<li>{esc(x)}</li>" for x in st["next"])
    gpus = sum(r["gpus"] for r in st["running"])
    return f"""<title>Probe Evasion Sprint</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one column of figures; summary first, then results, then design notes. */
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --fg: #161a20; --muted: #5b6472; --line: #dde1e7; --accent: #2848b8;
  --good: #1d7f52; --warn: #b26a12; --bad: #bf3b33; --neutral: #8a94a3; --faint: #d3d8df;
  --s1: #2848b8; --s2: #c2410c; --s3: #0f766e; --s4: #7c3aed; --s5: #b91c1c; --s6: #4d7c0f;
  --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #111419; --surface: #181c23; --fg: #e7eaef; --muted: #98a1ae; --line: #2a303a; --accent: #8aa2ff;
  --good: #4cc38a; --warn: #e3a34a; --bad: #f07a70; --neutral: #7d8796; --faint: #343b46;
  --s1: #8aa2ff; --s2: #fb923c; --s3: #2dd4bf; --s4: #b794f6; --s5: #f87171; --s6: #a3e635; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #111419; --surface: #181c23; --fg: #e7eaef; --muted: #98a1ae; --line: #2a303a; --accent: #8aa2ff;
  --good: #4cc38a; --warn: #e3a34a; --bad: #f07a70; --neutral: #7d8796; --faint: #343b46;
  --s1: #8aa2ff; --s2: #fb923c; --s3: #2dd4bf; --s4: #b794f6; --s5: #f87171; --s6: #a3e635; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--sans); }}
.wrap {{ max-width: 920px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; display: grid; gap: 30px; }}
header {{ display: grid; gap: 6px; }}
.eyebrow {{ font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font-size: 28px; line-height: 1.15; margin: 0; text-wrap: balance; font-weight: 600; }}
h2 {{ font-size: 18px; margin: 0; font-weight: 600; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
p {{ margin: 0; max-width: 72ch; }}
.lede, .note, .empty {{ color: var(--muted); }}
.headline {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--accent); padding: 14px 16px; font-size: 16px; max-width: none; }}
section {{ display: grid; gap: 10px; min-width: 0; }}
figure {{ margin: 0; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px; display: grid; gap: 12px; min-width: 0; }}
figcaption {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 14px; }}
svg {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--line); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font: 11px var(--mono); }}
svg .tick.sub {{ font-size: 10px; }}
svg .val {{ fill: var(--fg); font: 600 12px var(--mono); }}
svg .ln {{ fill: none; stroke-width: 2.25; }}
svg .ln.s1, svg .dot.s1 {{ stroke: var(--s1); }} svg .dot.s1 {{ fill: var(--s1); }}
svg .ln.s2 {{ stroke: var(--s2); }} svg .dot.s2 {{ fill: var(--s2); stroke: var(--s2); }}
svg .ln.s3 {{ stroke: var(--s3); }} svg .dot.s3 {{ fill: var(--s3); stroke: var(--s3); }}
svg .ln.s4 {{ stroke: var(--s4); }} svg .dot.s4 {{ fill: var(--s4); stroke: var(--s4); }}
svg .ln.s5 {{ stroke: var(--s5); }} svg .dot.s5 {{ fill: var(--s5); stroke: var(--s5); }}
svg .ln.s6 {{ stroke: var(--s6); }} svg .dot.s6 {{ fill: var(--s6); stroke: var(--s6); }}
svg .e0 {{ fill: var(--neutral); }} svg .control {{ fill: var(--faint); stroke: var(--neutral); }} svg .attack {{ fill: var(--accent); }}
svg .ci {{ stroke: var(--fg); stroke-width: 1.5; }}
svg .ref {{ stroke: var(--muted); stroke-dasharray: 5 4; stroke-width: 1.25; }}
.brow {{ display: grid; grid-template-columns: minmax(150px, 220px) 1fr auto; gap: 14px; align-items: center; }}
.blabel {{ display: grid; font-size: 13.5px; line-height: 1.3; min-width: 0; }}
.blabel span {{ color: var(--muted); font-size: 12px; }}
.bval {{ font: 600 18px var(--mono); font-variant-numeric: tabular-nums; text-align: right; display: grid; color: var(--bad); }}
.bval.ok {{ color: var(--good); }}
.bval span {{ font: 400 11px var(--mono); color: var(--muted); }}
.bar {{ display: flex; height: 18px; border-radius: 4px; overflow: hidden; background: var(--line); }}
.seg {{ display: block; height: 100%; }}
.pair {{ display: grid; gap: 4px; }}
.hb {{ height: 14px; border-radius: 3px; position: relative; min-width: 2px; }}
.hb span {{ position: absolute; left: calc(100% + 6px); top: -2px; font: 11px var(--mono); color: var(--muted); white-space: nowrap; }}
.good {{ background: var(--good); }} .warn {{ background: var(--warn); }} .bad {{ background: var(--bad); }}
.neutral {{ background: var(--neutral); }} .muted {{ background: var(--faint); }}
.e0 {{ background: var(--neutral); }} .control {{ background: var(--faint); }} .attack {{ background: var(--accent); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--muted); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--muted); }}
.sw {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.sw.s1 {{ background: var(--s1); }} .sw.s2 {{ background: var(--s2); }} .sw.s3 {{ background: var(--s3); }}
.sw.s4 {{ background: var(--s4); }} .sw.s5 {{ background: var(--s5); }} .sw.s6 {{ background: var(--s6); }}
.pill {{ display: inline-block; font: 500 12px var(--mono); padding: 2px 8px; border-radius: 999px; background: var(--bg); border: 1px solid var(--line); color: var(--muted); }}
.okpill {{ color: var(--good); border-color: var(--good); }}
svg .pt {{ stroke: var(--surface); stroke-width: 1; }}
svg .pt.honest {{ fill: var(--neutral); }} svg .pt.bd {{ fill: var(--bad); }} svg .pt.fail {{ fill: var(--warn); }}
.spend {{ gap: 14px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.tile {{ display: grid; gap: 2px; }}
.tl {{ font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }}
.tv {{ font: 600 24px var(--mono); font-variant-numeric: tabular-nums; }}
.tile.big .tv {{ font-size: 32px; color: var(--accent); }}
.hours {{ display: flex; align-items: flex-end; gap: 4px; height: 90px; }}
.hcol {{ flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; gap: 3px; min-width: 0; }}
.hbar {{ width: 100%; background: var(--accent); border-radius: 3px 3px 0 0; min-height: 2px; opacity: .85; }}
.hcol span {{ font: 10px var(--mono); color: var(--muted); }}
.jobs {{ gap: 16px; }}
.job {{ display: grid; gap: 6px; }}
.jobhead {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
.jobname {{ font-size: 14px; font-weight: 500; min-width: 0; }}
.jobmeta {{ font: 11.5px var(--mono); color: var(--muted); }}
.prog {{ height: 8px; background: var(--line); border-radius: 999px; overflow: hidden; }}
.fill {{ height: 100%; background: var(--accent); border-radius: 999px; }}
.fill.late {{ background: var(--warn); }}
.chip {{ font: 500 12px var(--mono); color: var(--accent); white-space: nowrap; }}
.chip.late {{ color: var(--warn); }}
.trow {{ display: grid; grid-template-columns: minmax(180px, 300px) 1fr; gap: 14px; align-items: center; }}
.tlabel {{ font-size: 13px; line-height: 1.3; display: grid; min-width: 0; }}
.tlabel span {{ color: var(--muted); font: 11px var(--mono); }}
.track {{ position: relative; height: 14px; background: var(--bg); border-radius: 3px; }}
.tbar {{ position: absolute; top: 0; bottom: 0; background: var(--accent); border-radius: 3px; opacity: .85; }}
.tbar.late {{ background: var(--warn); }}
.now {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--fg); }}
.taxis {{ display: flex; justify-content: space-between; font: 11px var(--mono); color: var(--muted); padding-left: min(314px, 40%); }}
dl {{ display: grid; grid-template-columns: minmax(140px, 200px) 1fr; gap: 8px 18px; margin: 0; }}
dt {{ font-weight: 600; }} dd {{ margin: 0; color: var(--muted); min-width: 0; }}
ol {{ margin: 0; padding-left: 20px; display: grid; gap: 4px; }}
@media (max-width: 620px) {{
  .brow, .trow {{ grid-template-columns: 1fr; gap: 6px; }} .bval {{ text-align: left; }}
  .taxis {{ padding-left: 0; }} dl {{ grid-template-columns: 1fr; gap: 2px 0; }} dd {{ margin-bottom: 8px; }} h1 {{ font-size: 23px; }}
}}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Sprint status · updated {time.strftime('%-I:%M %p')} (<span id="generated" data-t="{int(time.time() * 1000)}">just now</span>)</div>
    <h1>Can Qwen 27B evade a probe when told to?</h1>
    <p class="lede">Goal: one clear case where warning the model about a probe (E1/E2/E3) lets its bad behavior slip past the probe more often than with no warning (E0).</p>
  </header>
  <p class="headline">{esc(st['headline'])}</p>
  <section>
    <h2>Running now <span class="pill">{len(st['running'])} jobs · ~{gpus} H100s</span></h2>
    {running_list(st['running'])}
  </section>
  {section_evasion()}
  {section_e0_check()}
  {section_spend(st['running'])}
  {section_gate()}
  {section_probe_data()}
  {section_probes()}
  {section_deception_filter()}
  <section><h2>Design</h2><dl>{decisions}</dl></section>
  <section><h2>Next</h2><ol>{nxt}</ol></section>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
