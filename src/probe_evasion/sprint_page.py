"""Render the optimized-prompt evasion sprint page (sprint/page.html). Hand-kept text is in sprint/sprint.json; every
number is read from results/ at build time.

    uv run python -m probe_evasion.sprint_page
"""

import json
import time
from pathlib import Path

from probe_evasion.status_page import CSS, esc, running_list, section_spend

SPRINT = Path("sprint")
OUT = SPRINT / "page.html"
G = Path("results/gepa")
CONCEPTS = {"maths": "Maths in the head", "sarcasm": "Sarcasm", "chemistry": "Chemistry"}
READINGS = [("mean", "averaged"), ("ema", "EMA"), ("window", "10-token window"), ("max", "highest token")]
N_LAYERS = {"gemma": 42, "qwen35": 64}
# (model, concept, probe layer, run directory or None for "latest matching", readout the search optimized)
GRID = [
    ("Gemma-2-9B", "Maths", "12", G / "gemma_maths_ema_20261007-145004", "ema"),
    ("Gemma-2-9B", "Maths", "30", G / "gemma_maths_window_L30_20261007-151039", "window"),
    ("Gemma-2-9B", "Maths", "36", G / "gemma_maths_window_L36_20261007-154620", "window"),
    ("Gemma-2-9B", "Maths", "30, varied training", G / "gemma_maths_window_variedL30_20261007-154620", "window"),
    ("Gemma-2-9B", "Chemistry", "12 (paper's probe)", G / "gemma_chemistry_ema_20261007-145004", "ema"),
    ("Gemma-2-9B", "Sarcasm", "12", G / "gemma_sarcasm_ema_20261007-145004", "ema"),
    ("Qwen3.5-27B", "Maths", "16", G / "qwen35_maths_window_L16_20261007-173549", "window"),
    ("Qwen3.5-27B", "Maths", "32", G / "qwen35_maths_window_L32_20261007-173549", "window"),
    ("Qwen3.5-27B", "Maths", "48", G / "qwen35_maths_window_L48_20261007-173549", "window"),
    ("Qwen3.5-27B", "Maths", "48, searched vs highest token", "qwen35_maths_max_L48_*", "max"),
    ("Qwen3.5-27B", "Chemistry", "16", G / "qwen35_chemistry_window_L16_20261007-173800", "window"),
    ("Qwen3.5-27B", "Chemistry", "48", G / "qwen35_chemistry_window_L48_20261007-173800", "window"),
]
# maths, Gemma vs Qwen by depth: (model, layer, run)
DEPTH = [("gemma", 12, G / "gemma_maths_ema_20261007-145004"), ("gemma", 30, G / "gemma_maths_window_L30_20261007-151039"),
         ("gemma", 36, G / "gemma_maths_window_L36_20261007-154620"),
         ("qwen35", 16, G / "qwen35_maths_window_L16_20261007-173549"), ("qwen35", 32, G / "qwen35_maths_window_L32_20261007-173549"),
         ("qwen35", 48, G / "qwen35_maths_window_L48_20261007-173549")]
LARGE = [("Gemma-2-9B maths, layer 36", G / "gemma_maths_window_L36_20261007-154620"),
         ("Gemma-2-9B maths, layer 30, varied training", G / "gemma_maths_window_variedL30_20261007-154620")]

EXTRA_CSS = """
.grid { width: 100%; border-collapse: separate; border-spacing: 3px; font-size: 13px; }
.grid th { font: 500 11px var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; text-align: center; padding: 4px; }
.grid th.l, .grid td.l { text-align: left; }
.grid td { padding: 6px 8px; border-radius: 4px; text-align: center; font: 600 13px var(--mono); color: var(--fg); }
.grid td.l { font: 500 13px var(--sans); background: none; white-space: nowrap; }
.grid td.l span { color: var(--muted); font-weight: 400; font-size: 12px; }
.grid td.target { outline: 2px solid var(--fg); outline-offset: -2px; }
.grid td .sub { display: block; font: 400 10.5px var(--mono); color: var(--muted); }
.grid tr.sep td { padding-top: 14px; }
.tablewrap { overflow-x: auto; }
.fig { display: block; width: 100%; border-radius: 6px; background: #fff; }
.links { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
.links a { display: grid; gap: 2px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; text-decoration: none; color: var(--fg); }
.links a span { color: var(--muted); font-size: 12.5px; }
.ptable { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.ptable th, .ptable td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
.ptable th { font: 500 11.5px var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.ptable td.n { font: 13px var(--mono); }
"""


def read_json(p: Path):
    return json.loads(p.read_text())


def corrected_misses(run: Path, target: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """(plain, optimized) as (missed, concept present) on the held-out test, from the raw-logit re-scoring
    (summary_logits.json) when it exists, else the search's own summary."""
    if (run / "summary_logits.json").exists():
        m = read_json(run / "summary_logits.json")["readings"][target]["missed"]
        return tuple(tuple(int(x) for x in m[a].split("/")) for a in ("no_instruction", "optimized"))
    arms = read_json(run / "summary.json")["arms"]
    return tuple((arms[a][f"missed_{target}"], arms[a]["present"]) for a in ("no instruction", "optimized"))


def finished(run: Path | None) -> bool:
    return run is not None and (run / "summary.json").exists() and "arms" in read_json(run / "summary.json")


def results_grid() -> str:
    head = "".join(f"<th>{esc(n)}</th>" for _, n in READINGS)
    rows, last_model = "", None
    for model, concept, layer, run, searched in GRID:
        if isinstance(run, str):
            matches = sorted(G.glob(run))
            run = matches[-1] if matches else None
        sep = ' class="sep"' if last_model and model != last_model else ""
        last_model = model
        label = f'<td class="l">{esc(model)} · {esc(concept)}<span> · layer {esc(layer)}</span></td>'
        if not finished(run):
            rows += f'<tr{sep}>{label}<td colspan="{len(READINGS)}" class="l"><span>search running</span></td></tr>'
            continue
        cells = ""
        for key, _ in READINGS:
            (pk, pn), (ok, on) = corrected_misses(run, key)
            rate = ok / on
            cells += (f'<td class="{"target" if key == searched else ""}" style="background: rgba(234,88,12,{0.05 + 0.7 * rate:.2f})" '
                      f'title="red-team prompt {ok}/{on}; normal prompt {pk}/{pn}">{100 * rate:.0f}%'
                      f'<span class="sub">normal {100 * pk / pn:.0f}%</span></td>')
        rows += f"<tr{sep}>{label}{cells}</tr>"
    return (f'<figure class="card"><figcaption>Share of valid answers (concept confirmed) that the red-team prompt gets past the probe, '
            f'on 100 held-out tasks; small text: the normal prompt. Columns: how the probe\'s per-token scores are combined. '
            f'Outlined: the reading that search optimized against.</figcaption><div class="tablewrap"><table class="grid">'
            f'<thead><tr><th class="l">Model · concept · probe layer</th>{head}</tr></thead><tbody>{rows}</tbody></table></div></figure>')


def depth_chart() -> str:
    """Maths, EMA reading: red-team miss rate against relative probe depth, Gemma vs Qwen."""
    w, h, l, b, t = 640, 240, 44, 40, 18
    x = lambda f: l + f * (w - l - 16)
    y = lambda v: t + (1 - v) * (h - t - b)
    out = "".join(f'<line class="grid" x1="{l}" x2="{w - 16}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                  f'<text class="tick" x="{l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>' for v in (0, .5, 1))
    out += "".join(f'<text class="tick" x="{x(f):.1f}" y="{h - 20}" text-anchor="middle">{int(f * 100)}%</text>' for f in (0, .25, .5, .75, 1))
    out += f'<text class="tick" x="{(l + w) / 2:.0f}" y="{h - 4}" text-anchor="middle">probe depth (share of the model\'s layers)</text>'
    for model, cls in (("gemma", "s2"), ("qwen35", "s1")):
        pts = []
        for m, layer, run in DEPTH:
            if m == model and finished(run):
                (_, _), (ok, on) = corrected_misses(run, "ema")
                pts.append((layer / N_LAYERS[m], ok / on, layer, ok, on))
        pp = " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v, *_ in pts)
        out += f'<polyline class="ln {cls}" points="{pp}"/>' + "".join(
            f'<circle class="dot {cls}" cx="{x(f):.1f}" cy="{y(v):.1f}" r="4.5"><title>layer {L}: {k}/{n}</title></circle>'
            f'<text class="val" x="{x(f):.1f}" y="{y(v) - 9:.1f}" text-anchor="middle">L{L}</text>' for f, v, L, k, n in pts)
    return (f'<figure class="card"><figcaption><span class="key"><span class="sw s2"></span>Gemma-2-9B (42 layers)</span>'
            f'<span class="key"><span class="sw s1"></span>Qwen3.5-27B (64 layers)</span> · maths, a fresh search at each probe, '
            f'EMA reading, held-out</figcaption><svg viewBox="0 0 {w} {h}" role="img" aria-label="Miss rate by probe depth">{out}</svg></figure>')


def rare_table() -> str:
    rows = ""
    for label, run in LARGE:
        if not (run / "large_test.json").exists():
            continue
        arms = read_json(run / "large_test.json")["arms"]
        if "evasions_per_attempt" not in arms.get("optimized", {}):
            continue
        cell = lambda a: (f'{a["valid_evasions"]}/{a["attempts"]} ({100 * a["evasions_per_attempt"]:.1f}%; 95% CI '
                          f'{100 * a["ci95_per_attempt"][0]:.2f}-{100 * a["ci95_per_attempt"][1]:.2f}%)')
        rows += f'<tr><td>{esc(label)}</td><td class="n">{cell(arms["optimized"])}</td><td class="n">{cell(arms["plain"])}</td></tr>'
    return (f'<figure class="card"><figcaption>Valid evasions out of all attempts: 10 samples on each of 100 held-out problems, '
            f'10-token window</figcaption><div class="tablewrap"><table class="ptable"><thead><tr><th>Probe</th>'
            f'<th>Red-team prompt</th><th>Normal prompt</th></tr></thead><tbody>{rows}</tbody></table></div></figure>')


def render() -> str:
    st = read_json(SPRINT / "sprint.json")
    gpus = sum(r["gpus"] for r in st["running"])
    links = "".join(f'<a href="{esc(u)}">{esc(t)}<span>{esc(x)}</span></a>' for t, u, x in st["earlier"])
    decisions = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in st["decisions"])
    return f"""<meta charset="utf-8">
<title>Optimized Prompt Evasion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
{CSS}
{EXTRA_CSS}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Sprint status · updated {time.strftime('%-I:%M %p')} (<span id="generated" data-t="{int(time.time() * 1000)}">just now</span>)</div>
    <h1>Where does a Claude-optimized prompt get a model past a concept probe?</h1>
    <p class="lede">{esc(st['headline'])}</p>
  </header>

  <section><h2>Results</h2>{results_grid()}</section>
  <section><h2>Better-trained probes catch the same optimized answers</h2>
    <figure class="card"><img class="fig" src="probe_rungs.png" alt="Probe-strength ladder: training data, method and miss rate for each rung"></figure></section>
  <section><h2>The depth effect differs by model</h2>{depth_chart()}</section>
  <section><h2>Rare misses against the strongest Gemma probes</h2>{rare_table()}</section>
  <section><h2>Pages</h2><div class="links">{links}</div></section>

  <section>
    <h2>Running now <span class="pill">{len(st['running'])} jobs · ~{gpus} GPUs</span></h2>
    {running_list(st['running'])}
  </section>
  {section_spend(st['running'])}

  <details class="more"><summary>Findings log</summary><ul class="takeaways">{''.join(f"<li>{esc(x)}</li>" for x in st['takeaways'])}</ul></details>
  <details class="more"><summary>Setup and methods</summary><dl>{decisions}</dl></details>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
