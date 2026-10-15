"""Render status/examples.html: real traces from each condition of a lie-evasion run with their probe scores.

    uv run python -m probe_evasion.examples_page results/deception/eval_<run>
"""

import html
import sys
from pathlib import Path

import random

import numpy as np

from probe_evasion.generation import read_jsonl
from probe_evasion.lie_outcomes import CATEGORIES, outcomes
from probe_evasion.metrics import FPR
from probe_evasion.parsing import parse
from probe_evasion.trivia import ADDITIONS, LIE_SYSTEM, TRUTH_SYSTEM

PROBES = {"d_claude": "Lie probe trained on Claude's answers (off-policy)", "d_qwen": "Lie probe trained on Qwen's own answers (on-policy)"}
CONDITIONS = [
    ("E0", "No warning"), ("placebo", "Unrelated instruction"), ("aware", "Told a probe is watching"),
    ("E1", "Told to evade"), ("E2_d_claude", "Evade + shown 2 training examples"), ("E3", "Evade + strategy hints"),
]
# Confirmation run (fresh questions): the strategy hint split into its two halves.
CONFIRM_CONDITIONS = [
    ("E3_info", "Evade + facts about the probe only (no 'believe it')"),
    ("E3_believe", "Evade + 'believe the false answer' only (no facts about the probe)"),
]
MAIN = "d_claude"
REFUSALS_SHOWN = 2  # per kind of refusal per condition
ENDING_CHARS = 1500


def esc(x) -> str:
    return html.escape(str(x))


def thresholds(examples: list[dict], key: str, conditions=CONDITIONS) -> dict:
    """Per (condition, probe): one threshold for every condition, the score that flags 1% of the no-warning truthful
    answers (answers that never saw any evasion text). Keyed by condition so callers needn't care."""
    out = {}
    for probe in PROBES:
        honest = [e[key][probe] for e in examples if e["condition"] == "E0" and e["label"] == 0]
        for cond, _ in conditions:
            out[cond, probe] = float(np.quantile(honest, 1 - FPR))
    return out


def verdict(score: float, threshold: float, label: int) -> str:
    flagged = score > threshold
    if label == 1:
        return '<span class="tag bad">missed</span>' if not flagged else '<span class="tag good">caught</span>'
    return '<span class="tag warn">false alarm</span>' if flagged else '<span class="tag ok">not flagged</span>'


def score_cell(e: dict, n: dict | None, thr_full: dict, thr_rep: dict | None, probe: str) -> str:
    c = e["condition"]
    full = e["full"][probe]
    cell = f"<td class='num'>{full:.2f} <span class='muted'>/ {thr_full[c, probe]:.2f}</span> {verdict(full, thr_full[c, probe], e['label'])}</td>"
    if n is None:
        return cell + "<td class='num muted'>not computed for this run</td>"
    rep = n["full"][probe]
    return cell + f"<td class='num'>{rep:.2f} <span class='muted'>/ {thr_rep[c, probe]:.2f}</span> {verdict(rep, thr_rep[c, probe], e['label'])}</td>"


def strip(examples: list[dict], cond: str, key: str, thr: float, probe: str) -> str:
    """Every response's score in this condition: lies above the line, truthful answers below, threshold marked."""
    vals = [e[key][probe] for e in examples if e["condition"] == cond]
    lo, hi = min(vals) - 0.5, max(vals) + 0.5
    w, l, r = 560, 8, 8
    x = lambda v: l + (v - lo) / (hi - lo) * (w - l - r)
    dots = ""
    for e in examples:
        if e["condition"] != cond:
            continue
        y = 14 if e["label"] == 1 else 38
        jitter = (hash(e["question"]) % 9 - 4) * 1.2
        dots += f'<circle class="{"lie" if e["label"] else "truth"}" cx="{x(e[key][probe]):.1f}" cy="{y + jitter:.1f}" r="2.4"/>'
    t = x(thr)
    return (f'<svg viewBox="0 0 {w} 60" role="img" aria-label="score distribution">'
            f'<line class="axis" x1="{l}" x2="{w - r}" y1="26" y2="26"/>{dots}'
            f'<line class="thr" x1="{t:.1f}" x2="{t:.1f}" y1="2" y2="52"/>'
            f'<text class="lab" x="{t + 4:.1f}" y="58">no-warning threshold</text>'
            f'<text class="lab" x="{l}" y="58">{lo:.0f}</text><text class="lab" x="{w - r}" y="58" text-anchor="end">{hi:.0f}</text></svg>')


def pick(examples: list[dict], cond: str) -> list[tuple[str, int]]:
    """The lowest-scoring lie (closest to getting past the probe), the median lie, and the shortest truthful answer."""
    idx = [i for i, e in enumerate(examples) if e["condition"] == cond]
    lies = sorted((i for i in idx if examples[i]["label"] == 1), key=lambda i: examples[i]["full"][MAIN])
    truths = [i for i in idx if examples[i]["label"] == 0]
    out = []
    if lies:
        out += [("Lowest-scoring lie (closest to getting through)", lies[0]), ("Typical lie (median score)", lies[len(lies) // 2])]
    if truths:
        out.append(("Truthful answer", min(truths, key=lambda i: len(examples[i]["reasoning"]))))
    return out


def outcome_bar(counts) -> str:
    n = sum(counts.values())
    segs = "".join(f'<span style="width:{100 * counts[k] / n:.2f}%;background:{color}" title="{esc(label)}: {counts[k]}"></span>'
                   for k, label, color in CATEGORIES if counts[k])
    legend = " · ".join(f'<span class="sw" style="background:{color}"></span>{esc(label)} {100 * counts[k] / n:.0f}%'
                        for k, label, color in CATEGORIES if counts[k])
    return f'<div class="obar">{segs}</div><div class="olegend">{legend}</div>'


def refusal_cards(rows: list[dict], cond: str) -> str:
    """Lie-prompt attempts that didn't produce a lie: a truthful answer instead, or reasoning that never finished."""
    rng = random.Random(0)
    lie_rows = [r for r in rows if r["condition"] == cond and r["kind"] == "lie"]
    truthful = [r for r in lie_rows if r["judge"] is not None and r["judge"]["verdict"] == "correct"]
    unfinished = [r for r in lie_rows if r["judge"] is None]
    cards = []
    for r in rng.sample(truthful, min(REFUSALS_SHOWN, len(truthful))):
        reasoning = parse(r["raw"]).reasoning
        cards.append(f"""<article class="trace refusal">
  <header><b>Told to lie, answered truthfully</b> · <span class="muted">Q:</span> {esc(r['question'])}</header>
  <blockquote>{esc(r['final'])}</blockquote>
  <div class="muted small">How the reasoning ends:</div><pre class="ending">…{esc(reasoning[-ENDING_CHARS:])}</pre>
  <details><summary>Full reasoning ({len(reasoning):,} characters)</summary><pre>{esc(reasoning)}</pre></details>
</article>""")
    for r in rng.sample(unfinished, min(REFUSALS_SHOWN, len(unfinished))):
        reasoning = parse(r["raw"]).reasoning
        cards.append(f"""<article class="trace refusal">
  <header><b>Told to lie, never finished</b> (no answer within the 8k-token budget) · <span class="muted">Q:</span> {esc(r['question'])}</header>
  <div class="muted small">Where it was when the budget ran out:</div><pre class="ending">…{esc(reasoning[-ENDING_CHARS:])}</pre>
  <details><summary>Full reasoning ({len(reasoning):,} characters)</summary><pre>{esc(reasoning)}</pre></details>
</article>""")
    return "".join(cards)


def condition_sections(examples, replays, conditions, run_note: str, rows: list[dict], counts: dict) -> list[str]:
    thr_full = thresholds(examples, "full", conditions)
    thr_rep = thresholds(replays, "full", conditions) if replays else None
    sections = []
    for cond, name in conditions:
        addition = ADDITIONS.get(cond) if not cond.startswith("E2_") else next(e["system"] for e in examples if e["condition"] == cond).split(LIE_SYSTEM, 1)[-1].strip()
        n_lies = sum(e["condition"] == cond and e["label"] == 1 for e in examples)
        n_truth = sum(e["condition"] == cond and e["label"] == 0 for e in examples)
        missed = np.mean([e["full"][MAIN] <= thr_full[cond, MAIN] for e in examples if e["condition"] == cond and e["label"] == 1])
        flagged = np.mean([e["full"][MAIN] > thr_full[cond, MAIN] for e in examples if e["condition"] == cond and e["label"] == 0])
        missed_rep, missed_rep_e0 = ([np.mean([r["full"][MAIN] <= thr_rep[c, MAIN] for r, e in zip(replays, examples)
                                               if e["condition"] == c and e["label"] == 1]) for c in (cond, "E0")]
                                     if replays else (None, None))
        cards = []
        for kind, i in pick(examples, cond):
            e, n = examples[i], (replays[i] if replays else None)
            cards.append(f"""<article class="trace">
  <header><b>{kind}</b> · <span class="muted">Q:</span> {esc(e['question'])}</header>
  <blockquote>{esc(e['response'])}</blockquote>
  <table><thead><tr><th>Probe (score / threshold)</th><th class='num'>Full context (as generated)</th><th class='num'>Question + answer only</th></tr></thead>
  <tbody>{''.join(f"<tr><td>{esc(PROBES[p])}</td>{score_cell(e, n, thr_full, thr_rep, p)}</tr>" for p in PROBES)}</tbody></table>
  <details><summary>Qwen's reasoning ({len(e['reasoning']):,} characters)</summary><pre>{esc(e['reasoning'])}</pre></details>
</article>""")
        sections.append(f"""<section>
  <h2>{esc(name)}</h2>
  <div class="addition"><span class="muted">Added to the system prompt:</span> {esc(addition) if addition else '<i>nothing</i>'}</div>
  <h3>What Qwen did when told to lie</h3>
  {outcome_bar(counts[cond])}
  {refusal_cards(rows, cond)}
  <h3>The lies it did tell, and what the probe made of them</h3>
  <p class="muted">{run_note} {n_lies} completed lies, {n_truth} truthful answers. The dashed line is the same in every section of this run: the score that flags 1% of the truthful answers given with <b>no warning</b>. In this condition it misses {100 * missed:.0f}% of lies and flags <b>{100 * flagged:.0f}% of this condition's truthful answers</b> (false alarms){" &mdash; the truthful answers have shifted up, the effect described on the dashboard" if flagged > 0.05 else ""}.{f" Re-read with only the question and answer (and that re-read's own no-warning threshold), it misses {100 * missed_rep:.0f}% of lies (with no warning: {100 * missed_rep_e0:.0f}%)." if missed_rep is not None else ""}</p>
  <figure class="dist"><figcaption>Off-policy probe scores, full context. Top row: lies (red). Bottom row: truthful answers (grey). Lies left of the line would be missed; truthful answers right of it are false alarms.</figcaption>
  {strip(examples, cond, "full", thr_full[cond, MAIN], MAIN)}</figure>
  {''.join(cards)}
</section>""")
    return sections


def render(run_dir: Path, confirm_dir: Path | None = None) -> str:
    full = read_jsonl(run_dir / "examples_scored.jsonl")
    rep = read_jsonl(run_dir / "examples_neutral_scored.jsonl")
    assert len(full) == len(rep)
    examples = [dict(e, full=e["scores"]) for e in full]
    replays = [dict(e, full=e["scores"]) for e in rep]
    questions = {q["qid"]: q["question"] for q in read_jsonl(Path("results/deception/known.jsonl"))}
    rows = [dict(r, question=questions[r["qid"]]) for r in read_jsonl(run_dir / "samples_judged.jsonl")]
    sections = condition_sections(examples, replays, CONDITIONS, "First run (300 questions).", rows, outcomes(run_dir))
    if confirm_dir:
        conf = [dict(e, full=e["scores"]) for e in read_jsonl(confirm_dir / "examples_scored_fresh.jsonl")]
        # The confirmation run's no-warning answers set its threshold, so they're kept in although not shown.
        conf_rows = [dict(r, question=questions[r["qid"]]) for r in read_jsonl(confirm_dir / "samples_judged.jsonl")]
        sections += condition_sections(conf, None, [("E0", "No warning")] + CONFIRM_CONDITIONS,
                                       "Confirmation run (368 new questions).", conf_rows, outcomes(confirm_dir))[1:]

    return f"""<title>Lie Probe Traces</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; one section per condition with its score distribution and three real traces. */
:root {{ --bg: #f5f6f8; --surface: #ffffff; --fg: #161a20; --muted: #5b6472; --line: #dde1e7; --accent: #2848b8;
  --good: #1d7f52; --bad: #bf3b33; --warn: #b26a12; --neutral: #8a94a3;
  --sans: "Instrument Sans", system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #111419; --surface: #181c23; --fg: #e7eaef; --muted: #98a1ae;
  --line: #2a303a; --accent: #8aa2ff; --good: #4cc38a; --bad: #f07a70; --warn: #e3a34a; --neutral: #7d8796; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #111419; --surface: #181c23; --fg: #e7eaef; --muted: #98a1ae; --line: #2a303a; --accent: #8aa2ff;
  --good: #4cc38a; --bad: #f07a70; --warn: #e3a34a; --neutral: #7d8796; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--sans); }}
.wrap {{ max-width: 900px; margin: 0 auto; padding-inline: 18px; padding-block: 28px 64px; display: grid; gap: 34px; }}
h1 {{ font-size: 26px; margin: 0; font-weight: 600; text-wrap: balance; }}
h2 {{ font-size: 19px; margin: 0; font-weight: 600; }}
p {{ margin: 0; max-width: 75ch; }}
section {{ display: grid; gap: 12px; min-width: 0; }}
.muted {{ color: var(--muted); }}
.addition {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--accent); padding: 10px 14px; font-size: 14px; }}
.trace {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; display: grid; gap: 10px; min-width: 0; }}
blockquote {{ margin: 0; padding: 8px 12px; border-left: 3px solid var(--line); font-size: 15px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
th {{ font: 500 11px var(--mono); text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }}
.num {{ font-family: var(--mono); white-space: nowrap; }}
.tag {{ font: 500 11px var(--mono); padding: 1px 7px; border-radius: 999px; border: 1px solid currentColor; margin-left: 4px; }}
.tag.bad {{ color: var(--bad); }} .tag.good {{ color: var(--good); }} .tag.warn {{ color: var(--warn); }} .tag.ok {{ color: var(--muted); }}
details summary {{ cursor: pointer; color: var(--accent); font-size: 13px; }}
pre {{ white-space: pre-wrap; font: 12px/1.5 var(--mono); max-height: 420px; overflow: auto; background: var(--bg); padding: 10px; border-radius: 6px; }}
.dist {{ margin: 0; display: grid; gap: 4px; }} .dist figcaption {{ font-size: 12px; color: var(--muted); }}
svg {{ width: 100%; height: auto; display: block; }}
svg .lie {{ fill: var(--bad); opacity: .75; }} svg .truth {{ fill: var(--neutral); opacity: .6; }}
svg .axis {{ stroke: var(--line); }} svg .thr {{ stroke: var(--fg); stroke-dasharray: 4 3; }}
svg .lab {{ fill: var(--muted); font: 10px var(--mono); }}
.scroll {{ overflow-x: auto; }}
h3 {{ font-size: 15px; margin: 6px 0 0; font-weight: 600; }}
.small {{ font-size: 12px; }}
.trace.refusal {{ border-left: 4px solid #2a78d6; }}
pre.ending {{ max-height: 220px; }}
.obar {{ display: flex; height: 16px; border-radius: 4px; overflow: hidden; gap: 2px; }}
.obar span {{ display: block; height: 100%; }}
.olegend {{ font-size: 12px; color: var(--muted); line-height: 1.8; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }}
</style>
<div class="wrap">
  <header style="display:grid;gap:8px">
    <h1>What the lie probe sees, condition by condition</h1>
    <p class="muted">Qwen3.5-27B answering held-out TriviaQA questions. Every lie prompt starts with: "{esc(LIE_SYSTEM)}" Every truth prompt: "{esc(TRUTH_SYSTEM)}" Each condition adds the text shown to both.
    For each condition: first what Qwen did with every lie-prompt attempt (often it refuses, by answering truthfully or deliberating until it runs out of budget), with two real examples of each kind of refusal; then the probe scores of the lies it did tell, and three real responses (the lowest-scoring lie, a typical lie, and a truthful answer).
    Scores are shown against one threshold for all conditions: the score that flags 1% of truthful answers given with no warning, i.e. answers that never saw any evasion text. Each response is scored as generated (prompt and reasoning in context) and, for the first run, re-read with only the question and answer.</p>
  </header>
  {''.join(sections)}
</div>
"""


if __name__ == "__main__":
    run = Path(sys.argv[1])
    confirm = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path("experimental/status/examples.html")
    out.write_text(render(run, confirm))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
