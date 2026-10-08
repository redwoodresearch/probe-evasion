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
Q, GM = "qwen35_maths", "gemma_maths"
QRUN = {"window": G / f"{Q}_window_L48_20261007-173549", "onpolicy": G / f"{Q}_ema_onpolicy_L48_20261007-190705",
        "rewrites": G / f"{Q}_ema_rewrites_L48_20261007-190705", "clean": G / f"{Q}_ema_onpolicyclean_L48_20261008-v2",
        "divaug": G / f"{Q}_ema_onpolicydivaug_L48_20261008-v2"}
GRUN = {"saved": G / f"{GM}_window_L36_20261007-154620", "clean36": G / f"{GM}_ema_onpolicyclean_L36_20261008-v2",
        "clean30": G / f"{GM}_ema_onpolicyclean_L30_20261008-v2"}
CHEM = {48: G / "qwen35_chemistry_window_L48_20261007-173800", 16: G / "qwen35_chemistry_window_L16_20261007-173800"}


def ladder_cell(run: Path, probe: str, layer: int, arm: str) -> tuple[int, int]:
    res = read_json(run / f"ladder_{probe}.json")["results"][f"maths@L{layer}"]["ema"]["missed"][arm]
    return tuple(int(x) for x in res.split("/"))


def reval_cell(run: Path, arm: str) -> tuple[int, int]:
    m = read_json(run / "revalidated.json")["readings"]["ema"]["missed"][arm]
    return m["missed"], m["valid"]


def summary_cell(run: Path, arm: str) -> tuple[int, int]:
    """Searches run under the current maths rules (calibration version 2+): their own held-out summary is current."""
    assert read_json(run / "calibration.json").get("version", 0) >= 2, run
    a = read_json(run / "summary.json")["arms"][arm]
    return a["missed_ema"], a["present"]


def repeat_cell(run: Path, kind: str, arm: str) -> tuple[int, int]:
    a = read_json(run / f"{kind}.json")["arms"][arm]
    return a["valid_evasions"], a["attempts"]


# (group, probe, attack, held-out (optimized, plain), repeats (optimized, plain), new division problems (optimized, plain))
ROWS = [
    ("Qwen3.5-27B · maths · layer 48", "Templated data (original)", "found against this probe",
     lambda: (ladder_cell(QRUN["window"], f"{Q}_layers", 48, "optimized"), ladder_cell(QRUN["window"], f"{Q}_layers", 48, "no_instruction")), None, None),
    (None, "Model's own replies", "fresh search against it",
     lambda: (ladder_cell(QRUN["onpolicy"], f"{Q}_onpolicy", 48, "optimized"), ladder_cell(QRUN["onpolicy"], f"{Q}_onpolicy", 48, "no_instruction")),
     lambda: (repeat_cell(QRUN["onpolicy"], "large_test", "optimized"), repeat_cell(QRUN["onpolicy"], "large_test", "plain")),
     lambda: (repeat_cell(QRUN["onpolicy"], "fresh_test", "optimized"), repeat_cell(QRUN["onpolicy"], "fresh_test", "plain"))),
    (None, "Own replies + narrative rewrites", "fresh search against it",
     lambda: (ladder_cell(QRUN["rewrites"], f"{Q}_rewrites_lr", 48, "optimized"), ladder_cell(QRUN["rewrites"], f"{Q}_rewrites_lr", 48, "no_instruction")),
     lambda: (repeat_cell(QRUN["rewrites"], "large_test", "optimized"), repeat_cell(QRUN["rewrites"], "large_test", "plain")),
     lambda: (repeat_cell(QRUN["rewrites"], "fresh_test", "optimized"), repeat_cell(QRUN["rewrites"], "fresh_test", "plain"))),
    (None, "Own replies, clean labels", "fresh search against it",
     lambda: (summary_cell(QRUN["clean"], "optimized"), summary_cell(QRUN["clean"], "no instruction")),
     lambda: (repeat_cell(QRUN["clean"], "large_test", "optimized"), repeat_cell(QRUN["clean"], "large_test", "plain")),
     lambda: (repeat_cell(QRUN["clean"], "fresh_test", "optimized"), repeat_cell(QRUN["clean"], "fresh_test", "plain"))),
    (None, "Clean labels + varied division", "fresh search against it",
     lambda: (summary_cell(QRUN["divaug"], "optimized"), summary_cell(QRUN["divaug"], "no instruction")),
     lambda: (repeat_cell(QRUN["divaug"], "large_test", "optimized"), repeat_cell(QRUN["divaug"], "large_test", "plain")),
     lambda: (repeat_cell(QRUN["divaug"], "fresh_test", "optimized"), repeat_cell(QRUN["divaug"], "fresh_test", "plain"))),
    ("Gemma-2-9B · maths", "Templated data, layer 6 of 42", "found against the layer-36 probe",
     lambda: (ladder_cell(GRUN["saved"], f"{GM}_layers", 6, "optimized"), ladder_cell(GRUN["saved"], f"{GM}_layers", 6, "no_instruction")), None, None),
    (None, "Templated data, layer 12", "found against the layer-36 probe",
     lambda: (ladder_cell(GRUN["saved"], f"{GM}_layers", 12, "optimized"), ladder_cell(GRUN["saved"], f"{GM}_layers", 12, "no_instruction")), None, None),
    (None, "Templated data, layer 36", "found against this probe",
     lambda: (ladder_cell(GRUN["saved"], f"{GM}_layers", 36, "optimized"), ladder_cell(GRUN["saved"], f"{GM}_layers", 36, "no_instruction")), None, None),
    (None, "Own replies, clean labels, layer 36", "fresh search against it",
     lambda: (summary_cell(GRUN["clean36"], "optimized"), summary_cell(GRUN["clean36"], "no instruction")),
     lambda: (repeat_cell(GRUN["clean36"], "large_test", "optimized"), repeat_cell(GRUN["clean36"], "large_test", "plain")),
     lambda: (repeat_cell(GRUN["clean36"], "fresh_test", "optimized"), repeat_cell(GRUN["clean36"], "fresh_test", "plain"))),
    (None, "Own replies, clean labels, layer 30", "fresh search against it",
     lambda: (summary_cell(GRUN["clean30"], "optimized"), summary_cell(GRUN["clean30"], "no instruction")), None,
     lambda: (repeat_cell(GRUN["clean30"], "fresh_test", "optimized"), repeat_cell(GRUN["clean30"], "fresh_test", "plain"))),
    ("Qwen3.5-27B · chemistry", "Chemistry dataset, layer 48", "found against this probe",
     lambda: (reval_cell(CHEM[48], "optimized"), reval_cell(CHEM[48], "no_instruction")), None, None),
    (None, "Chemistry dataset, layer 16", "found against this probe",
     lambda: (reval_cell(CHEM[16], "optimized"), reval_cell(CHEM[16], "no_instruction")), None, None),
]


EXTRA_CSS = """
.grid { width: 100%; border-collapse: separate; border-spacing: 3px; font-size: 13px; }
.grid th { font: 500 11px var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; text-align: center; padding: 4px; }
.grid th.l, .grid td.l { text-align: left; }
.grid td { padding: 6px 8px; border-radius: 4px; text-align: center; font: 600 13px var(--mono); color: var(--fg); }
.grid td.l { font: 500 13px var(--sans); background: none; white-space: nowrap; }
.grid td.l span { color: var(--muted); font-weight: 400; font-size: 12px; }
.grid td.target { outline: 2px solid var(--fg); outline-offset: -2px; }
.grid td .sub { display: block; font: 400 10.5px var(--mono); color: var(--muted); }
.grid tr.group td { padding-top: 16px; text-align: left; font: 600 12px var(--sans); color: var(--muted); background: none; }
.grid td.na { font: 400 11.5px var(--mono); color: var(--muted); background: var(--surface); }
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
    """(plain, optimized) as (missed, valid) on the held-out test: revalidated.json (current measurement rules) when it
    exists, else the numbers as the search saw them (summary.json)."""
    if (run / "revalidated.json").exists():
        m = read_json(run / "revalidated.json")["readings"][target]["missed"]
        return tuple((m[a]["missed"], m[a]["valid"]) for a in ("no_instruction", "optimized"))
    arms = read_json(run / "summary.json")["arms"]
    return tuple((arms[a][f"missed_{target}"], arms[a]["present"]) for a in ("no instruction", "optimized"))


def cell(pair: tuple[tuple[int, int], tuple[int, int]] | None) -> str:
    if pair is None:
        return '<td class="na">not run</td>'
    (k, n), (pk, pn) = pair
    rate = k / n if n else 0
    return (f'<td style="background: rgba(234,88,12,{0.04 + 0.7 * rate:.2f})" title="red team {k}/{n}; normal prompt {pk}/{pn}">'
            f'{100 * rate:.0f}%<span class="sub">{k}/{n}</span><span class="sub">normal {pk}/{pn}</span></td>')


def evasion_table() -> str:
    rows = ""
    for group, probe, attack, held, rep, fresh in ROWS:
        if group:
            rows += f'<tr class="group"><td colspan="5">{esc(group)}</td></tr>'
        rows += (f'<tr><td class="l">{esc(probe)}<span>{esc(attack)}</span></td>{cell(held())}'
                 f'{cell(rep() if rep else None)}{cell(fresh() if fresh else None)}</tr>')
    return (f'<figure class="card"><figcaption>Valid answers that get past the probe (EMA reading, 1% false-alarm line from cleaned '
            f'no-concept replies). Held-out: misses among valid answers on 100 problems the search never saw. Repeated and new-problem '
            f'columns: valid evasions out of all attempts (10 samples of the 100 problems; 100 new division problems x 3). Small text: '
            f'the same probe on the normal prompt.</figcaption><div class="tablewrap"><table class="grid"><thead><tr>'
            f'<th class="l">Probe trained on · attack</th><th>Held-out problems</th><th>1,000 repeated attempts</th>'
            f'<th>New division problems</th></tr></thead><tbody>{rows}</tbody></table></div></figure>')


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

  <section><h2>What gets past the probes</h2>{evasion_table()}</section>
  <section><h2>Every reply's score, with the flag line before and after the calibration fix</h2>
    <figure class="card"><img class="fig" src="score_dots.png" alt="Probe scores per reply type with old and corrected flag lines"></figure></section>
  <section><h2>How each maths probe was trained (layer 48; last columns re-scored under the corrected calibration)</h2>
    <figure class="card"><img class="fig" src="probe_rungs.png" alt="Probe-strength ladder: training data, method and miss rate for each rung"></figure></section>
  <section><h2>Pages</h2><div class="links">{links}</div></section>

  <section>
    <h2>Running now <span class="pill">{len(st['running'])} jobs · ~{gpus} GPUs</span></h2>
    {running_list(st['running'])}
  </section>
  {section_spend(st['running'])}

  <details class="more" open><summary>Findings (corrected measurement)</summary><ul class="takeaways">{''.join(f"<li>{esc(x)}</li>" for x in st['takeaways'])}</ul></details>
  <details class="more"><summary>Earlier findings log (numbers superseded by the measurement fixes)</summary><ul class="takeaways">{''.join(f"<li>{esc(x)}</li>" for x in st['superseded'])}</ul></details>
  <details class="more"><summary>Setup and methods</summary><dl>{decisions}</dl></details>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
