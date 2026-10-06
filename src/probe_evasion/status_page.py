"""Render status/status.html from results/ summaries plus the hand-kept status/status.json.

    uv run python -m probe_evasion.status_page
"""

import html
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
        for variant in ("forced", "opportunistic"):
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
                "setting": f"{s.get('difficulty', 'interview')} · {'thinking ' + str(s.get('max_new_tokens', 16384) // 1024) + 'k' if s.get('thinking', True) else 'thinking off'}",
                "counts": counts, "success": counts["backdoor_success"] / n,
                "success_completed": counts["backdoor_success"] / completed if completed else None,
                "completed": completed,
            })
    return out


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
    body = "".join(
        f"<tr><td>{esc(r['setting'])}</td><td>{esc(r['variant'])}</td><td class='num'>{r['n']}</td>"
        f"<td class='num'>{r['completed']}</td><td class='num strong'>{r['counts']['backdoor_success']} ({pct(r['success'])})</td>"
        f"<td class='num'>{pct(r['success_completed'])}</td><td class='barcell'>{stacked_bar(r['counts'], r['n'])}</td></tr>"
        for r in rows
    )
    legend = "".join(f'<span class="key"><span class="sw {c}"></span>{esc(l)}</span>' for _, l, c in OUTCOME_ORDER)
    return f"""
<section>
  <h2>Can Qwen write working backdoors?</h2>
  <p class="lede">The day-1 gate: we need at least 20% working backdoors to have enough attacks to measure evasion on.
  "Completed" excludes responses that ran out of tokens before finishing their code.</p>
  <div class="scroll"><table>
    <thead><tr><th>Setting</th><th>Prompt</th><th class="num">n</th><th class="num">Completed</th>
    <th class="num">Working backdoors</th><th class="num">Of completed</th><th>Outcomes</th></tr></thead>
    <tbody>{body}</tbody></table></div>
  <div class="legend">{legend}</div>
</section>"""


def section_probe_data() -> str:
    rows = []
    for d in sorted((RESULTS / "probe_data").glob("*/*")):
        if (d / "summary.json").exists():
            s = read_json(d / "summary.json")
            forced = s["outcomes"].get("forced", {})
            rows.append(f"<tr><td>{esc(s['generator'])}</td><td>{esc(s.get('difficulty', 'interview'))}</td>"
                        f"<td class='num'>{s['n_problems']}</td><td class='num strong'>{s['positives']}</td>"
                        f"<td class='num'>{s['negatives']}</td><td class='num'>{pct(forced.get('backdoor_success', 0) / s['n_problems'])}</td>"
                        f"<td class='num'>{s['refusals']}</td></tr>")
    for d in sorted((RESULTS / "deception").glob("data_*")):
        if (d / "summary.json").exists():
            s = read_json(d / "summary.json")
            rows.append(f"<tr><td>{esc(s['generator'])}</td><td>TriviaQA deception</td><td class='num'>{s['n_train_questions']}</td>"
                        f"<td class='num strong'>{s['positives']}</td><td class='num'>{s['negatives']}</td>"
                        f"<td class='num'>{pct(s['lie_success'])}</td><td class='num'>–</td></tr>")
    if not rows:
        return ""
    return f"""
<section>
  <h2>Probe training data</h2>
  <p class="lede">Positives are verified bad behavior (a working backdoor, a lie); negatives are verified good behavior. The generator's instruction is removed before the probe sees an example.</p>
  <div class="scroll"><table>
    <thead><tr><th>Generator</th><th>Task</th><th class="num">Items</th><th class="num">Positives</th><th class="num">Negatives</th>
    <th class="num">Success rate</th><th class="num">Refusals</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
</section>"""


def section_probes() -> str:
    rows = []
    for d in sorted((RESULTS / "probes").glob("*")):
        if not (d / "report.json").exists() or d.name.endswith("val20"):
            continue
        rep = read_json(d / "report.json")
        best = max(rep, key=lambda l: rep[l]["val_auroc_ema"])
        cells = "".join(f"<td class='num{' strong' if l == best else ''}'>{rep[l]['val_auroc_ema']:.3f}</td>" for l in rep)
        rows.append(f"<tr><td>{esc(d.name)}</td>{cells}<td class='num'>{rep[best]['n_val']}</td></tr>")
    if not rows:
        return ""
    layers = "".join(f"<th class='num'>L{l}</th>" for l in (16, 24, 32, 40, 48))
    return f"""
<section>
  <h2>Probe quality</h2>
  <p class="lede">Held-out AUROC (EMA-max pooling) by layer, on problems the probe never trained on. Bold is the layer used downstream.</p>
  <div class="scroll"><table><thead><tr><th>Probe</th>{layers}<th class="num">n val</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div>
</section>"""


def section_evasion() -> str:
    blocks = []
    for kind, pattern in (("Backdoor", "evasion/*"), ("Deception", "deception/eval_*")):
        for d in sorted(RESULTS.glob(pattern)):
            if not (d / "summary.json").exists():
                continue
            s = read_json(d / "summary.json")
            for probe, p in s["probes"].items():
                conds = p["conditions"]
                rows = "".join(
                    f"<tr><td>{esc(c)}</td><td class='num'>{v['auroc']:.2f}</td>"
                    f"<td class='num'>{v['fixed_E0_threshold']['n_backdoor']}</td>"
                    f"<td class='num strong'>{pct(v['fixed_E0_threshold']['evasion_rate'])}</td>"
                    f"<td class='num'>{pct(v['fixed_E0_threshold']['ci95'][0])}–{pct(v['fixed_E0_threshold']['ci95'][1])}</td>"
                    f"<td class='num'>{pct(v['recalibrated']['evasion_rate'])}</td>"
                    f"<td class='num {'pos' if v.get('gap_vs_E0', {}).get('fixed_E0_threshold', 0) > 0.05 else ''}'>"
                    f"{'+' if v.get('gap_vs_E0', {}).get('fixed_E0_threshold', 0) >= 0 else ''}{pct(v.get('gap_vs_E0', {}).get('fixed_E0_threshold'))}</td></tr>"
                    for c, v in conds.items()
                )
                blocks.append(f"""<h3>{kind} · probe {esc(probe)} (layer {p['layer']}) · {esc(d.name)}</h3>
  <div class="scroll"><table><thead><tr><th>Condition</th><th class="num">AUROC</th><th class="num">Attacks</th>
  <th class="num">Evasion (E0 threshold)</th><th class="num">95% CI</th><th class="num">Evasion (recalibrated)</th><th class="num">Gap vs E0</th></tr></thead>
  <tbody>{rows}</tbody></table></div>""")
    if not blocks:
        return """
<section>
  <h2>Evasion results</h2>
  <p class="empty">No evasion run has finished yet. The first table here will compare E0 against the probe-warning conditions.</p>
</section>"""
    return f"""
<section>
  <h2>Evasion results</h2>
  <p class="lede">Evasion = share of successful attacks scoring at or below the threshold that flags 1% of honest responses. The headline is the gap over E0.</p>
  {''.join(blocks)}
</section>"""


def section_deception_filter() -> str:
    for d in sorted((RESULTS / "deception").glob("filter_*"), reverse=True):
        if (d / "summary.json").exists():
            s = read_json(d / "summary.json")
            return (f"<p class='note'>Deception knowledge filter: Qwen answered {s['n_known']:,} of {s['n_questions']:,} TriviaQA questions "
                    f"correctly 5/5 times ({s['by_split']['train']} train, {s['by_split']['test']} test).</p>")
    return ""


def render() -> str:
    st = read_json(STATUS / "status.json")
    running = "".join(
        f"<tr><td>{esc(r['job'])}</td><td class='num'>{esc(r['started'])}</td><td class='num'>{esc(r['eta'])}</td><td class='num'>{r['gpus']}</td></tr>"
        for r in st["running"]
    )
    decisions = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in st["decisions"])
    nxt = "".join(f"<li>{esc(x)}</li>" for x in st["next"])
    gpus = sum(r["gpus"] for r in st["running"])
    return f"""<title>Probe Evasion Sprint</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; summary first, then results, then design notes. */
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --fg: #161a20; --muted: #5b6472; --line: #dde1e7; --accent: #2848b8;
  --good: #1d7f52; --warn: #b26a12; --bad: #bf3b33; --neutral: #8a94a3; --faint: #c9ced6;
  --sans: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #111419; --surface: #181c23; --fg: #e7eaef; --muted: #98a1ae; --line: #2a303a; --accent: #8aa2ff;
  --good: #4cc38a; --warn: #e3a34a; --bad: #f07a70; --neutral: #7d8796; --faint: #3a414c; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #111419; --surface: #181c23; --fg: #e7eaef; --muted: #98a1ae; --line: #2a303a; --accent: #8aa2ff;
  --good: #4cc38a; --warn: #e3a34a; --bad: #f07a70; --neutral: #7d8796; --faint: #3a414c; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--sans); }}
.wrap {{ max-width: 980px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; display: grid; gap: 28px; }}
header {{ display: grid; gap: 6px; }}
.eyebrow {{ font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font-size: 28px; line-height: 1.15; margin: 0; text-wrap: balance; font-weight: 600; }}
h2 {{ font-size: 18px; margin: 0; font-weight: 600; }}
h3 {{ font-size: 14px; margin: 14px 0 6px; font: 500 13px var(--mono); color: var(--muted); }}
p {{ margin: 0; max-width: 72ch; }}
.lede {{ color: var(--muted); }}
.headline {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 0;
  padding: 14px 16px; font-size: 16px; max-width: none; }}
section {{ display: grid; gap: 10px; min-width: 0; }}
.scroll {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
th {{ font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); white-space: nowrap; }}
tbody tr:last-child td {{ border-bottom: 0; }}
.num {{ text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }}
.strong {{ font-weight: 600; color: var(--fg); }}
.pos {{ color: var(--good); font-weight: 600; }}
.barcell {{ min-width: 180px; }}
.bar {{ display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: var(--line); }}
.seg {{ display: block; height: 100%; }}
.good {{ background: var(--good); }} .warn {{ background: var(--warn); }} .bad {{ background: var(--bad); }}
.neutral {{ background: var(--neutral); }} .muted {{ background: var(--faint); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--muted); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
.pill {{ display: inline-block; font: 500 12px var(--mono); padding: 2px 8px; border-radius: 999px; background: var(--surface); border: 1px solid var(--line); color: var(--muted); }}
dl {{ display: grid; grid-template-columns: minmax(140px, 200px) 1fr; gap: 8px 18px; margin: 0; }}
dt {{ font-weight: 600; }} dd {{ margin: 0; color: var(--muted); min-width: 0; }}
ol {{ margin: 0; padding-left: 20px; display: grid; gap: 4px; }}
.note, .empty {{ color: var(--muted); }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr; gap: 2px 0; }} dd {{ margin-bottom: 8px; }} h1 {{ font-size: 23px; }} }}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Sprint status · updated {time.strftime('%a %b %-d, %-I:%M %p')}</div>
    <h1>Can Qwen 27B evade a probe when told to?</h1>
    <p class="lede">Goal: one clear case where warning the model about a probe (E1/E2/E3) lets its bad behavior slip past the probe more often than with no warning (E0).</p>
  </header>
  <p class="headline">{esc(st['headline'])}</p>
  <section>
    <h2>Running now <span class="pill">{len(st['running'])} jobs · ~{gpus} H100s</span></h2>
    <div class="scroll"><table><thead><tr><th>Job</th><th class="num">Started</th><th class="num">ETA</th><th class="num">GPUs</th></tr></thead>
    <tbody>{running}</tbody></table></div>
  </section>
  {section_evasion()}
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
