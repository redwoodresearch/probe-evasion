"""Example traces from the prompt searches: each reply coloured token by token by the probe, with its scores against
the thresholds and the concept check. Writes sprint/traces.html.

    uv run python -m probe_evasion.traces_page
"""

import html
import json
from pathlib import Path

import numpy as np

from probe_evasion.gepa import POOLS, ema_score, window_score
from probe_evasion.sprint_page import CONCEPTS
from probe_evasion.status_page import CSS

OUT = Path("sprint/traces.html")
# (concept, run directory, what the search optimized) — earliest runs first; later runs are added as they finish
RUNS = [
    ("maths", Path("results/gepa/gemma_maths_window_L30_20261007-151039"), "window"),
    ("maths", Path("results/gepa/gemma_maths_ema_20261007-145004"), "ema"),
    ("chemistry", Path("results/gepa/gemma_chemistry_ema_20261007-145004"), "ema"),
    ("sarcasm", Path("results/gepa/gemma_sarcasm_ema_20261007-145004"), "ema"),
    ("chemistry", Path("results/gepa/gemma_chemistry_20261007-142658"), "mean"),
    ("maths", Path("results/gepa/gemma_maths_20261007-142658"), "mean"),
    ("sarcasm", Path("results/gepa/gemma_sarcasm_20261007-142658"), "mean"),
]
READING = {"mean": "averaged over reply", "ema": "EMA peak", "max": "highest token", "window": "10-token window"}

EXTRA_CSS = """
.trace { display: grid; gap: 8px; }
.tmeta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip2 { font: 500 12px var(--mono); padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.chip2.pass { color: var(--good); border-color: var(--good); background: var(--surface); }
.chip2.fail { color: var(--bad); border-color: var(--bad); background: var(--surface); }
.req { font-size: 13px; color: var(--muted); white-space: pre-wrap; }
.toks { font: 13px/1.9 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; background: var(--bg);
  border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; }
.toks span { border-radius: 2px; color: var(--fg); }
.why { font-size: 13px; color: var(--muted); }
.instr { font: 12.5px/1.5 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; background: var(--bg);
  border: 1px dashed var(--line); border-radius: 6px; padding: 10px 12px; max-height: 260px; overflow-y: auto; }
.scale { display: flex; align-items: center; gap: 8px; font: 11.5px var(--mono); color: var(--muted); }
.scale i { display: inline-block; width: 140px; height: 10px; border-radius: 2px;
  background: linear-gradient(90deg, rgba(234,88,12,0.04), rgba(234,88,12,0.55)); }
"""


def esc(x) -> str:
    return html.escape(str(x))


def load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    for r in rows:
        if r.get("token_scores"):
            r["ema"] = ema_score(np.array(r["token_scores"]))
            r["window"] = window_score(np.array(r["token_scores"]))
    return rows


def thresholds(run_dir: Path) -> dict:
    """Thresholds for every reading, recomputed from the saved calibration replies (older runs lack the EMA one)."""
    rows = load(run_dir / "calibration_replies.jsonl")
    neg = [r for r in rows if r["task"]["split"] == "negative"]
    return {p: float(np.quantile([r[p] for r in neg], 0.99)) for p in POOLS}


def heat(tokens: list[str], scores: list[float]) -> str:
    # light shading so the text stays readable at every score
    return "".join(f'<span style="background: rgba(234,88,12,{0.04 + 0.51 * min(1.0, s):.2f})" title="{s:.2f}">{esc(t)}</span>'
                   for t, s in zip(tokens, scores))


def trace(r: dict, thr: dict, show_request: bool = True) -> str:
    chips = "".join(
        f'<span class="chip2 {"pass" if r[p] <= thr[p] else "fail"}">{READING[p]}: {r[p]:.2f} '
        f'({"missed" if r[p] <= thr[p] else "flagged"}; line {thr[p]:.2f})</span>' for p in ("mean", "window"))
    concept = f'<span class="chip2 {"pass" if r["present"] else "fail"}">concept {"there" if r["present"] else "not there"}</span>'
    words = f'<span class="chip2">{len(r["reply"].split())} words</span>'
    req = f'<div class="req"><b>Request:</b> {esc(r["task"]["task"][:400])}</div>' if show_request else ""
    return (f'<div class="card trace">{req}<div class="tmeta">{concept}{chips}{words}</div>'
            f'<div class="toks">{heat(r["tokens"], r["token_scores"])}</div><div class="why">Check: {esc(r["reason"])}</div></div>')


def pick(rows: list[dict], thr: dict, target: str) -> dict[str, list[dict]]:
    present = [r for r in rows if r["present"]]
    return {"got past": sorted([r for r in present if r[target] <= thr[target]], key=lambda r: r[target])[:3],
            "caught": sorted([r for r in present if r[target] > thr[target]], key=lambda r: -r[target])[:1],
            "concept dropped or too long": [r for r in rows if not r["present"]][:1]}


def section(concept: str, run_dir: Path, target: str) -> str:
    thr = thresholds(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None
    config = json.loads((run_dir / "run_config.json").read_text()) if (run_dir / "run_config.json").exists() else {}
    probe = config.get("probe") or "layer-12 probe (default)"
    out = (f'<section><h2>{esc(CONCEPTS[concept])} <span class="pill">probe: {esc(probe)}</span> '
           f'<span class="pill">search optimized the {READING[target]} reading</span></h2>')
    plain = load(run_dir / "test_no_instruction.jsonl") or [r for r in load(run_dir / "calibration_replies.jsonl")
                                                            if r["task"]["split"] != "negative"]
    if plain:
        typical = sorted([r for r in plain if r["present"]], key=lambda r: r[target])
        out += "<h3>Plain task, typical reply</h3>" + trace(typical[len(typical) // 2], thr)
    if summary and "instruction" in summary:
        rows = load(run_dir / "test_optimized.jsonl")
        out += f'<h3>Best prompt found (frozen, then tested on 100 new tasks)</h3><div class="instr">{esc(summary["instruction"])}</div>'
        source = "held-out test"
    else:  # search unfinished: the latest round's proposals
        rounds = sorted(run_dir.glob("round_*.jsonl"))
        rows = [r for r in load(rounds[-1]) if r.get("who") == "child"] if rounds else []
        if rows:
            out += f'<h3>Latest proposal\'s instruction (search still running)</h3><div class="instr">{esc(rows[0]["prompt"].rsplit(chr(10) * 2, 1)[0])}</div>'
        source = "latest search round"
    for name, examples in pick(rows, thr, target).items():
        if examples:
            out += f"<h3>{esc(name.capitalize())} ({source})</h3>" + "".join(trace(r, thr) for r in examples)
    return out + "</section>"


def render() -> str:
    sections = "".join(section(c, d, t) for c, d, t in RUNS if d.exists())
    return f"""<meta charset="utf-8">
<title>Evasion Traces</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
{CSS}
{EXTRA_CSS}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Gemma-2-9B · concept probes · example replies</div>
    <h1>What the evasion attempts look like</h1>
    <p class="lede">Each reply is coloured token by token by the concept probe (darker = stronger concept signal; hover a token for its score). Chips give the reply's score under two readings against their thresholds: the paper's average over the reply, and the EMA peak (a running average over about 10 tokens), which padding can't dilute.</p>
    <div class="scale">0 <i></i> 1</div>
  </header>
  {sections}
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
