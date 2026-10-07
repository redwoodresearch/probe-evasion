"""Render the optimized-prompt evasion sprint page (sprint/page.html) in the style of the first sprint page.
Hand-kept text is in sprint/sprint.json; every number is read from results/ at build time.

    uv run python -m probe_evasion.sprint_page
"""

import base64
import json
import time
from pathlib import Path

import numpy as np

from probe_evasion.status_page import CSS, esc, pct, running_list, section_spend

SPRINT = Path("sprint")
OUT = SPRINT / "page.html"
GEPA = Path("results/gepa")
PROBES = Path("results/concept_probes")
MODELS = {"gemma": "Gemma-2-9B", "qwen35": "Qwen3.5-27B"}
CONCEPTS = {"maths": "Maths in the head", "sarcasm": "Sarcasm", "chemistry": "Chemistry"}
ARMS = [("no instruction", "plain task", "e0"), ("told probe is watching", "told a probe is watching", "control"), ("seed: strategy hint", "our strategy hint", "control"),
        ("optimized", "optimized prompt", "attack")]

EXTRA_CSS = """
.runs { display: grid; gap: 14px; }
.rhead { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; align-items: baseline; }
.rname { font-weight: 600; }
.rmeta { font: 11.5px var(--mono); color: var(--muted); }
.prompt { font: 12.5px/1.5 var(--mono); background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px 12px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 220px; overflow-y: auto; }
.ptable { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.ptable th, .ptable td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
.ptable th { font: 500 11.5px var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.ptable td.n { font: 13px var(--mono); font-variant-numeric: tabular-nums; }
.tablewrap { overflow-x: auto; }
.designs { display: grid; gap: 12px; }
.designs img { width: 100%; height: auto; background: #fff; border-radius: 6px; border: 1px solid var(--line); }
"""


def read_json(p: Path):
    return json.loads(p.read_text())


def latest_runs() -> dict[tuple[str, str], Path]:
    """Most recent search run per (model, concept)."""
    out = {}
    for d in sorted(GEPA.glob("*_*_*")):
        model, concept = d.name.split("_")[:2]
        out[(model, concept)] = d
    return out


def glance(runs: dict) -> str:
    tiles = []
    for (model, concept), d in runs.items():
        if (d / "summary.json").exists():
            arms = read_json(d / "summary.json")["arms"]
            opt, base = arms["optimized"], arms["no instruction"]
            tiles.append((f"{opt['missed_mean']}/{opt['n']}",
                          f"{MODELS[model]}, {CONCEPTS[concept].lower()}: held-out tasks where the concept is there and the "
                          f"probe misses it, optimized prompt (plain task: {base['missed_mean']}/{base['n']})"))
    if not tiles:
        return '<p class="note">Headline numbers appear here when the first search finishes its held-out test.</p>'
    return '<div class="glance">' + "".join(f'<div class="gtile"><div class="gv">{esc(v)}</div><div class="gl">{esc(l)}</div></div>'
                                            for v, l in tiles) + "</div>"


def results_section(runs: dict) -> str:
    rows = ""
    for (model, concept), d in runs.items():
        if not (d / "summary.json").exists():
            continue
        s = read_json(d / "summary.json")
        rows += f'<h3>{esc(MODELS[model])} · {esc(CONCEPTS[concept])}</h3>'
        for key, label, cls in ARMS:
            a = s["arms"][key]
            rate = a["missed_mean"] / a["n"]
            rows += (f'<div class="brow"><div class="blabel">{esc(label)}<span>concept there in {a["present"]}/{a["n"]} · '
                     f'max-over-tokens probe misses {a["missed_max"]}</span></div>'
                     f'<div class="bar"><span class="seg {cls}" style="width:{100 * rate:.1f}%"></span></div>'
                     f'<div class="bval{" ok" if rate == 0 else ""}">{pct(rate)}<span>{a["missed_mean"]}/{a["n"]}</span></div></div>')
        removed = s["arms"].get("optimized, instruction removed")
        if removed:
            rows += f'<p class="note">Same optimized replies re-read with the instruction deleted from the prompt: the probe misses {removed["missed_mean"]}.</p>'
    if not rows:
        return '<figure class="card"><p class="empty">No held-out results yet.</p></figure>'
    return (f'<figure class="card"><figcaption>Share of 100 held-out tasks where the concept is really there and the probe '
            f'misses it (the paper\'s mean-over-tokens probe, threshold flags 1% of no-concept replies)</figcaption>{rows}</figure>')


def progress_svg(d: Path) -> str:
    """Best validation reward so far, by search round."""
    cands = [json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()] if (d / "candidates.jsonl").exists() else []
    if not cands:
        return '<p class="empty">Calibrating.</p>'
    last = max(c["round"] for c in cands)
    rounds = [json.loads(l) for l in (d / "rounds.jsonl").read_text().splitlines()] if (d / "rounds.jsonl").exists() else []
    last = max([last] + [r["round"] for r in rounds])
    best = [max(np.mean(c["val_rewards"]) for c in cands if c["round"] <= r) for r in range(last + 1)]
    w, h, l, t, b = 520, 130, 34, 8, 22
    top = max(0.25, max(best) * 1.15)
    x = lambda i: l + i * (w - l - 8) / max(1, last)
    y = lambda v: t + (1 - v / top) * (h - t - b)
    grid = "".join(f'<line class="grid" x1="{l}" x2="{w - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="tick" x="{l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>' for v in (0, top / 2, top))
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(best))
    accepted = "".join(f'<circle class="dot s2" cx="{x(c["round"]):.1f}" cy="{y(np.mean(c["val_rewards"])):.1f}" r="3"><title>'
                       f'candidate {c["id"]}: {esc(json.dumps(c["val"]))}</title></circle>' for c in cands)
    xt = "".join(f'<text class="tick" x="{x(i):.1f}" y="{h - 6}" text-anchor="middle">{i}</text>'
                 for i in range(0, last + 1, max(1, last // 6)))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="search progress">{grid}{xt}'
            f'<polyline class="ln s1" points="{pts}"/>{accepted}</svg>')


def search_section(runs: dict) -> str:
    if not runs:
        return '<figure class="card"><p class="empty">No searches started yet.</p></figure>'
    cards = ""
    for (model, concept), d in runs.items():
        cands = [json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()] if (d / "candidates.jsonl").exists() else []
        best = max(cands, key=lambda c: np.mean(c["val_rewards"])) if cands else None
        meta = (f'best: candidate {best["id"]} · concept there {best["val"]["present"]}/{best["val"]["n"]} · probe misses '
                f'{best["val"]["missed_mean"]} (validation)') if best else "calibrating"
        cards += (f'<div class="card"><div class="rhead"><span class="rname">{esc(MODELS[model])} · {esc(CONCEPTS[concept])}</span>'
                  f'<span class="rmeta">{esc(meta)}</span></div>{progress_svg(d)}'
                  + (f'<details><summary class="rmeta">best prompt so far</summary><div class="prompt">{esc(best["instruction"])}</div></details>' if best else "")
                  + "</div>")
    return f'<div class="runs">{cards}</div><p class="note">Line: best average reward on the 40 validation tasks so far (0 = concept dropped or caught; higher = concept kept with lower probe scores). Dots: prompts that beat their parent and were added.</p>'


def probes_section(runs: dict) -> str:
    rows = ""
    for model in MODELS:
        trained = read_json(PROBES / f"{model}.json") if (PROBES / f"{model}.json").exists() else {}
        for concept in CONCEPTS:
            p = trained.get(concept)
            d = runs.get((model, concept))
            cal = read_json(d / "calibration.json") if d and (d / "calibration.json").exists() else None
            source = "paper's released probe" if (model, concept) == ("gemma", "chemistry") else "trained, paper's recipe"
            layer = 12 if source.startswith("paper") else (p["layer"] if p else "–")
            rows += (f'<tr><td>{esc(MODELS[model])}</td><td>{esc(CONCEPTS[concept])}</td><td>{esc(source)}</td><td class="n">{layer}</td>'
                     f'<td class="n">{p["val_auroc"]:.3f}</td>' if p else
                     f'<tr><td>{esc(MODELS[model])}</td><td>{esc(CONCEPTS[concept])}</td><td>{esc(source)}</td><td class="n">{layer}</td><td class="n">–</td>')
            rows += (f'<td class="n">{cal["mean"]["auroc"]:.3f}</td><td class="n">{pct(cal["mean"]["baseline_missed"])}</td>'
                     f'<td class="n">{cal["n_positive"]} / {cal["n_negative"]}</td></tr>' if cal else
                     '<td class="n">–</td><td class="n">–</td><td class="n">–</td></tr>')
    return (f'<figure class="card"><div class="tablewrap"><table class="ptable"><thead><tr><th>Model</th><th>Concept</th><th>Probe</th>'
            f'<th>Layer</th><th>AUROC on training-style text</th><th>AUROC on the model\'s own replies</th><th>Missed with no instruction</th>'
            f'<th>Replies: concept / none</th></tr></thead><tbody>{rows}</tbody></table></div>'
            f'<figcaption>The probe has to work before evasion means anything: the last three columns use the model\'s own replies to the plain tasks.</figcaption></figure>')


def img(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def render() -> str:
    st = read_json(SPRINT / "sprint.json")
    runs = latest_runs()
    gpus = sum(r["gpus"] for r in st["running"])
    earlier = "".join(f'<li><a href="{esc(u)}">{esc(t)}</a>: {esc(x)}</li>' for t, u, x in st["earlier"])
    decisions = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in st["decisions"])
    designs = "".join(f'<img src="{img(p)}" alt="{esc(a)}">' for p, a in (
        ("share/design_concepts.png", "The three concepts"), ("share/design_loop.png", "The prompt search loop"),
        ("share/design_pooling.png", "Mean and max probe readings")))
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
    <h1>Can an optimized prompt get a model past a concept probe while it still does the task?</h1>
    <p class="lede">{esc(st['lede'])}</p>
  </header>

  {glance(runs)}

  <section>
    <h2>Held-out test</h2>
    {results_section(runs)}
  </section>

  <section>
    <h2>Prompt search</h2>
    {search_section(runs)}
  </section>

  <section>
    <h2>Probes</h2>
    {probes_section(runs)}
  </section>

  <section>
    <h2>Takeaways</h2>
    <ul class="takeaways">{''.join(f"<li>{esc(x)}</li>" for x in st['takeaways']) or '<li>None yet.</li>'}</ul>
  </section>

  <section>
    <h2>Running now <span class="pill">{len(st['running'])} jobs · ~{gpus} GPUs</span></h2>
    {running_list(st['running'])}
  </section>
  {section_spend(st['running'])}

  <section>
    <h2>Design</h2>
    <div class="designs">{designs}</div>
  </section>

  <section>
    <h2>Earlier in the sprint</h2>
    <ul>{earlier}</ul>
  </section>

  <details class="more"><summary>Setup and methods</summary>
    <dl>{decisions}</dl>
  </details>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
