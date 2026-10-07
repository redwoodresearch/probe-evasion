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


TARGETS = {"window": "10-token-window reading", "ema": "EMA-peak reading", "mean": "paper's averaged reading"}


def latest_runs() -> dict[tuple[str, str, str, str], Path]:
    """Most recent search run per (model, concept, target, probe). Runs named <model>_<concept>_<target>[_<probe tag>]_<time>;
    older runs without a target in the name optimized the averaged reading."""
    out = {}
    for d in sorted(GEPA.glob("*_*_*")):
        parts = d.name.split("_")
        target = parts[2] if parts[2] in TARGETS else "mean"
        tag = parts[3] if len(parts) > 4 and target == parts[2] else ""
        out[(parts[0], parts[1], target, tag)] = d
    return out


def label(key: tuple[str, str, str, str]) -> str:
    model, concept, target, tag = key
    probe = f" · probe {tag}" if tag else ""
    return f"{MODELS[model]} · {CONCEPTS[concept]} · {TARGETS[target]}{probe}"


SHORT_TARGET = {"window": "window", "ema": "EMA", "mean": "average"}
SHORT_CONCEPT = {"maths": "Maths", "sarcasm": "Sarcasm", "chemistry": "Chemistry"}


def corrected_misses(run: Path, target: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """(plain, optimized) as (missed, concept present) on the held-out test, from the raw-logit re-scoring
    (summary_logits.json) when it exists, else the search's own summary."""
    if (run / "summary_logits.json").exists():
        m = json.loads((run / "summary_logits.json").read_text())["readings"][target]["missed"]
        return tuple(tuple(int(x) for x in m[a].split("/")) for a in ("no_instruction", "optimized"))
    arms = json.loads((run / "summary.json").read_text())["arms"]
    return tuple((arms[a][f"missed_{target}"], arms[a]["present"]) for a in ("no instruction", "optimized"))


def glance(runs: dict) -> str:
    """One chart: per finished search, share of concept replies the probe misses, plain task vs optimized prompt."""
    rows = []
    for key, d in runs.items():
        if not (d / "summary.json").exists() or "arms" not in json.loads((d / "summary.json").read_text()):
            continue
        model, concept, target, tag = key
        plain, opt = corrected_misses(d, target)
        probe = {"": "L12", "L30": "L30", "L36": "L36", "variedL30": "L30 varied"}.get(tag, tag)
        rows.append((f"{SHORT_CONCEPT[concept]} · {probe} · {SHORT_TARGET[target]}", plain[0] / plain[1], opt[0] / opt[1], f"{opt[0]}/{opt[1]}"))
    if not rows:
        return '<p class="note">Headline numbers appear here when the first search finishes its held-out test.</p>'
    w, row_h, left, right = 760, 34, 230, 64
    h = row_h * len(rows) + 34
    x = lambda v: left + v * (w - left - right)
    out = "".join(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="6" y2="{h - 24}"/>'
                  f'<text class="tick" x="{x(v):.1f}" y="{h - 8}" text-anchor="middle">{int(v * 100)}%</text>' for v in (0, .5, 1))
    for i, (name, p, q, frac_txt) in enumerate(rows):
        y = 20 + i * row_h
        out += (f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" style="font: 500 13px var(--sans); fill: var(--fg)">{esc(name)}</text>'
                f'<line x1="{x(p):.1f}" x2="{x(q):.1f}" y1="{y}" y2="{y}" stroke="var(--faint)" stroke-width="5" stroke-linecap="round"/>'
                f'<circle cx="{x(p):.1f}" cy="{y}" r="5" fill="var(--neutral)"><title>plain task {p:.0%}</title></circle>'
                f'<circle cx="{x(q):.1f}" cy="{y}" r="6" fill="var(--s2)"><title>optimized {q:.0%}</title></circle>'
                f'<text class="val" x="{w - 4}" y="{y + 4}" text-anchor="end">{frac_txt}</text>')
    return (f'<figure class="card"><figcaption><span class="key"><span class="sw" style="background:var(--neutral)"></span>plain task</span>'
            f'<span class="key"><span class="sw s2"></span>optimized prompt</span> · Gemma-2-9B, 100 held-out tasks · replies that do the task and the probe '
            f'misses · L = probe layer; window / EMA / average = how token scores are combined (average and EMA are beaten by padding and short replies)</figcaption><svg viewBox="0 0 {w} {h}" role="img" aria-label="Misses per search">{out}</svg></figure>')


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
    for key, d in runs.items():
        cands = [json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines()] if (d / "candidates.jsonl").exists() else []
        best = max(cands, key=lambda c: np.mean(c["val_rewards"])) if cands else None
        meta = (f'best: candidate {best["id"]} · concept there {best["val"]["present"]}/{best["val"]["n"]} · probe misses '
                f'{best["val"].get(f"missed_{key[2]}", "–")} (validation)') if best else "calibrating"
        cards += (f'<div class="card"><div class="rhead"><span class="rname">{esc(label(key))}</span>'
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
            d = runs.get((model, concept, "ema", "")) or runs.get((model, concept, "mean", ""))
            cal = read_json(d / "calibration.json") if d and (d / "calibration.json").exists() else None
            source = "paper's released probe" if (model, concept) == ("gemma", "chemistry") else "trained, paper's recipe"
            layer = 12 if source.startswith("paper") else (p["layer"] if p else "–")
            rows += (f'<tr><td>{esc(MODELS[model])}</td><td>{esc(CONCEPTS[concept])}</td><td>{esc(source)}</td><td class="n">{layer}</td>'
                     f'<td class="n">{p["val_auroc"]:.3f}</td>' if p else
                     f'<tr><td>{esc(MODELS[model])}</td><td>{esc(CONCEPTS[concept])}</td><td>{esc(source)}</td><td class="n">{layer}</td><td class="n">–</td>')
            rows += (f'<td class="n">{cal.get("ema", cal["mean"])["auroc"]:.3f}</td><td class="n">{pct(cal.get("ema", cal["mean"])["baseline_missed"])}</td>'
                     f'<td class="n">{cal["n_positive"]} / {cal["n_negative"]}</td></tr>' if cal else
                     '<td class="n">–</td><td class="n">–</td><td class="n">–</td></tr>')
    return (f'<figure class="card"><div class="tablewrap"><table class="ptable"><thead><tr><th>Model</th><th>Concept</th><th>Probe</th>'
            f'<th>Layer</th><th>AUROC on training-style text</th><th>AUROC on the model\'s own replies</th><th>Missed with no instruction</th>'
            f'<th>Plain replies used: with concept / without (sets threshold)</th></tr></thead><tbody>{rows}</tbody></table></div>'
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
