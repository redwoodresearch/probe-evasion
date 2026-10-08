"""Strategy bank: evasion strategies that have worked, with where they worked, why, and what defeats them; plus the
ones that didn't. Curated in data/strategy_bank.json; numbers are read from the runs at build time. Writes
sprint/strategies.html. Each strategy's concept-agnostic template can seed new searches (gepa search --bank-seeds).

    uv run python -m probe_evasion.strategy_bank
"""

import json
from pathlib import Path

from probe_evasion.sprint_page import corrected_misses
from probe_evasion.status_page import CSS, esc
from probe_evasion.traces_page import EXTRA_CSS, heat

BANK = Path("data/strategy_bank.json")
OUT = Path("sprint/strategies.html")


def bank() -> dict:
    return json.loads(BANK.read_text())


def templates() -> list[str]:
    return [s["template"] for s in bank()["strategies"]]


def result_rows(strategy: dict) -> str:
    rows = ""
    for r in strategy["results"]:
        run = Path(r["run"])
        if not (run / "summary.json").exists():
            rows += f'<tr><td>{esc(r["label"])}</td><td colspan="2" class="n">running</td></tr>'
            continue
        plain, opt = corrected_misses(run, r["reading"])
        rows += (f'<tr><td>{esc(r["label"])}</td><td class="n"><b>{opt[0]}/{opt[1]}</b> ({100 * opt[0] / opt[1]:.0f}%)</td>'
                 f'<td class="n">{plain[0]}/{plain[1]}</td></tr>')
    return (f'<table class="ptable"><thead><tr><th>Where it was tested (held-out, 100 tasks)</th><th>Red-team prompt: missed</th>'
            f'<th>Normal prompt</th></tr></thead><tbody>{rows}</tbody></table>')


def example(strategy: dict) -> str:
    first = Path(strategy["results"][0]["run"])
    if not (first / "test_optimized.jsonl").exists():
        return ""
    reading = strategy["results"][0]["reading"]
    rows = [json.loads(l) for l in (first / "test_optimized.jsonl").read_text().splitlines()]
    present = [r for r in rows if r["present"] and reading in r]
    if not present:
        return ""
    r = min(present, key=lambda r: r[reading])
    prompt = json.loads((first / "summary.json").read_text())["instruction"]
    return (f'<details><summary>The winning prompt ({esc(strategy["results"][0]["label"])})</summary><div class="instr">{esc(prompt)}</div></details>'
            f'<div class="req"><b>Example that got past:</b> {esc(r["task"]["task"][:240])}</div><div class="toks">{heat(r["tokens"], r["token_scores"])}</div>')


def render() -> str:
    b = bank()
    cards = "".join(
        f'<section class="card"><h2>{esc(s["name"])}</h2><p>{esc(s["mechanism"])}</p>{result_rows(s)}'
        f'<p class="note"><b>Defeated by:</b> {esc(s["defeated_by"])}</p>'
        f'<details><summary>Reusable template (seeds new searches)</summary><div class="instr">{esc(s["template"])}</div></details>'
        f'{example(s)}</section>' for s in b["strategies"])
    failed = "".join(f"<tr><td>{esc(a)}</td><td>{esc(c)}</td></tr>" for a, c in b["did_not_work"])
    return f"""<meta charset="utf-8">
<title>Evasion Strategy Bank</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
{CSS}
{EXTRA_CSS}
.ptable {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
.ptable th, .ptable td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
.ptable th {{ font: 500 11.5px var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
.ptable td.n {{ font: 13px var(--mono); white-space: nowrap; }}
details summary {{ cursor: pointer; font-weight: 600; font-size: 14px; }}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Probe evasion · prompting only · strategies found by Claude Opus 5.5 with probe-score feedback</div>
    <h1>Evasion strategy bank</h1>
    <p class="lede">Each strategy that has gotten a model past a concept probe while it still did the task, where it worked, why, and what stops it. A reply counts only if the concept is confirmed (correct number for maths; Claude's check otherwise). Replies are coloured by the probe (darker = stronger concept signal).</p>
  </header>
  {cards}
  <section class="card"><h2>Tried and didn't work</h2><table class="ptable"><tbody>{failed}</tbody></table></section>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
