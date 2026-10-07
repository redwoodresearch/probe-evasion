"""One-page visual summary for James: red-team strategies against probe monitors, and how far each got.
Every number is read from results/ at build time. Writes sprint/summary.html.

    uv run python -m probe_evasion.james_summary
"""

import json
from pathlib import Path

import numpy as np

from probe_evasion.gepa import window_score
from probe_evasion.status_page import CSS, esc

OUT = Path("sprint/summary.html")
G = Path("results/gepa")
CHEM_MEAN = G / "gemma_chemistry_20261007-142658"
SARC_EMA = G / "gemma_sarcasm_ema_20261007-145004"
MATHS = G / "gemma_maths_ema_20261007-145004"
MATHS_L30 = G / "gemma_maths_window_L30_20261007-151039"
STYLE_CONTROL = Path("results/controls/gemma_maths_20261007-145645")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def frac(s: str) -> tuple[int, int]:
    k, n = s.split("/")
    return int(k), int(n)


def window_threshold(run: Path) -> float:
    neg = [r for r in jl(run / "calibration_replies.jsonl") if r["task"]["split"] == "negative"]
    return float(np.quantile([window_score(np.array(r["token_scores"])) for r in neg], 0.99))


def missed_window(rows: list[dict], thr: float) -> tuple[int, int]:
    pres = [r for r in rows if r["present"]]
    return sum(window_score(np.array(r["token_scores"])) <= thr for r in pres), len(pres)


def defender_rows() -> list[dict]:
    chem = json.loads((CHEM_MEAN / "summary.json").read_text())["arms"]
    sarc = json.loads((SARC_EMA / "ema_variants.json").read_text())["variants"]
    ladder = json.loads((MATHS / "ladder_gemma_maths_layers.json").read_text())["results"]
    l30 = json.loads((MATHS_L30 / "summary.json").read_text())["arms"]
    rows = [
        {"defender": "Paper's released probe, score averaged over the reply", "concept": "Chemistry",
         "strategy": "Pad: an unrelated story before a correct answer", "plain": (chem["no instruction"]["missed_mean"], chem["no instruction"]["present"]),
         "red": (chem["optimized"]["missed_mean"], chem["optimized"]["present"]), "verdict": "readout exploit"},
        {"defender": "Running-average (EMA) peak, our first-token variant", "concept": "Sarcasm",
         "strategy": "One short line opening with a low-scoring word", "plain": frac(sarc["EMA 0.1, reply only, starts at first token"]["missed"]["no_instruction"]),
         "red": frac(sarc["EMA 0.1, reply only, starts at first token"]["missed"]["optimized"]), "verdict": "readout exploit",
         "note": f"standard EMA variants: {frac(sarc['EMA 0.1, carried over from the prompt']['missed']['optimized'])[0]}-{frac(sarc['EMA 0.1, reply only, starts at neutral (0.5)']['missed']['optimized'])[0]} missed"},
        {"defender": "10-token window, layer-12 probe", "concept": "Maths in the head",
         "strategy": "Bury the computed number in a scenic sentence", "plain": frac(ladder["maths@L12"]["window"]["missed"]["no_instruction"]),
         "red": frac(ladder["maths@L12"]["window"]["missed"]["optimized"]), "verdict": "wording the probe never saw"},
        {"defender": "10-token window, layer-30 probe (fresh search)", "concept": "Maths in the head",
         "strategy": "Scenic sentence, refined; works mostly on division", "plain": (l30["no instruction"]["missed_window"], l30["no instruction"]["present"]),
         "red": (l30["optimized"]["missed_window"], l30["optimized"]["present"]), "verdict": "partial"},
    ]
    for tag, label in (("L36", "10-token window, layer-36 probe (fresh search)"), ("variedL30", "Layer-30 probe trained on varied answers (fresh search)")):
        runs = sorted(G.glob(f"gemma_maths_window_{tag}_*"))
        if runs and (runs[-1] / "summary.json").exists():
            a = json.loads((runs[-1] / "summary.json").read_text())["arms"]
            rows.append({"defender": label, "concept": "Maths in the head", "strategy": "Fresh search",
                         "plain": (a["no instruction"]["missed_window"], a["no instruction"]["present"]),
                         "red": (a["optimized"]["missed_window"], a["optimized"]["present"]), "verdict": "result"})
        else:
            rows.append({"defender": label, "concept": "Maths in the head", "strategy": "Fresh search running", "plain": None, "red": None,
                         "verdict": "running"})
    return rows


def defender_chart(rows: list[dict]) -> str:
    """Dumbbell: plain-task miss rate (grey) to red-team miss rate (orange), per defender; labels above each row."""
    w, row_h, left, right = 760, 78, 10, 70
    h = row_h * len(rows) + 30
    x = lambda v: left + v * (w - left - right)
    grid = "".join(f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="34" y2="{h - 22}"/>'
                   f'<text class="tick" x="{x(v):.1f}" y="{h - 6}" text-anchor="middle">{int(v * 100)}%</text>' for v in (0, .25, .5, .75, 1))
    out = ""
    for i, r in enumerate(rows):
        top = 14 + i * row_h
        y = top + 40
        out += (f'<text class="lbl" x="0" y="{top}">{esc(r["defender"])}</text>'
                f'<text class="sub" x="0" y="{top + 16}">{esc(r["concept"])} · {esc(r["strategy"])}</text>')
        if r["red"] is None:
            out += f'<text class="sub" x="{x(0) + 6}" y="{y + 4}">search running</text>'
            continue
        p, q = r["plain"][0] / r["plain"][1], r["red"][0] / r["red"][1]
        out += (f'<line class="db" x1="{x(p):.1f}" x2="{x(q):.1f}" y1="{y}" y2="{y}"/>'
                f'<circle class="dplain" cx="{x(p):.1f}" cy="{y}" r="6"><title>plain task: {r["plain"][0]}/{r["plain"][1]}</title></circle>'
                f'<circle class="dred" cx="{x(q):.1f}" cy="{y}" r="7"><title>red-team prompt: {r["red"][0]}/{r["red"][1]}</title></circle>'
                f'<text class="val" x="{w - 4}" y="{y + 4}" text-anchor="end">{r["red"][0]}/{r["red"][1]}</text>')
        if r.get("note"):
            out += f'<text class="sub" x="{x(q) + 12:.1f}" y="{y + 4}">{esc(r["note"])}</text>'
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Miss rate by defender">{grid}{out}</svg>'


def layer_chart() -> str:
    series = []
    for name, cls in (("gemma_maths_layers", "s2"), ("gemma_maths_varied", "s3")):
        res = json.loads((MATHS / f"ladder_{name}.json").read_text())["results"]
        pts = sorted((int(k.split("@L")[1]), frac(v["window"]["missed"]["optimized"])) for k, v in res.items())
        series.append((name, cls, pts))
        plain = sorted((int(k.split("@L")[1]), frac(v["window"]["missed"]["no_instruction"])) for k, v in res.items())
    series.append(("plain", "s1", plain))
    w, h, l, b, t = 520, 220, 40, 30, 12
    layers = [p[0] for p in series[0][2]]
    x = lambda L: l + (L - layers[0]) / (layers[-1] - layers[0]) * (w - l - 16)
    y = lambda v: t + (1 - v) * (h - t - b)
    out = "".join(f'<line class="grid" x1="{l}" x2="{w - 16}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                  f'<text class="tick" x="{l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>' for v in (0, .5, 1))
    out += "".join(f'<text class="tick" x="{x(L):.1f}" y="{h - 8}" text-anchor="middle">L{L}</text>' for L in layers)
    for _, cls, pts in series:
        pp = " ".join(f"{x(L):.1f},{y(k / n):.1f}" for L, (k, n) in pts)
        out += f'<polyline class="ln {cls}" points="{pp}"/>' + "".join(
            f'<circle class="dot {cls}" cx="{x(L):.1f}" cy="{y(k / n):.1f}" r="3.5"><title>layer {L}: {k}/{n}</title></circle>' for L, (k, n) in pts)
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Scenic trick success by probe layer">{out}</svg>'


def token_strip() -> str:
    rows = jl(MATHS / "test_optimized.jsonl")
    r = next(r for r in rows if r["present"] and "3698" in r["reply"])
    spans = "".join(f'<span style="background: rgba(234,88,12,{0.04 + 0.51 * min(1, s):.2f})" title="{s:.2f}">{esc(t)}</span>'
                    for t, s in zip(r["tokens"], r["token_scores"]))
    return f'<div class="toks">{spans}</div>'


def control_bars() -> str:
    thr = window_threshold(MATHS)
    res = json.loads((STYLE_CONTROL / "summary.json").read_text())["arms"]
    labels = {"no instruction": "Plain task", "style only: cozy scene sentence (no probe mentioned)": "Style only (no probe mentioned)",
              "found prompt (EMA search)": "Found prompt", "found prompt (averaged search)": "Found prompt (other search)"}
    rows = ""
    for i, (key, label) in enumerate(labels.items()):
        k, n = missed_window(jl(STYLE_CONTROL / f"replies_{i:02d}.jsonl")[:40], thr)
        rows += (f'<div class="brow"><div class="blabel">{esc(label)}<span>{n} correct of 40</span></div>'
                 f'<div class="bar"><span class="seg {"attack" if i else "e0"}" style="width:{100 * k / n:.1f}%"></span></div>'
                 f'<div class="bval">{100 * k / n:.0f}%<span>{k}/{n}</span></div></div>')
    return rows


EXTRA = """
.lbl { fill: var(--fg); font: 600 13px var(--sans); } .sub { fill: var(--muted); font: 11.5px var(--sans); }
.db { stroke: var(--faint); stroke-width: 6; stroke-linecap: round; }
.dplain { fill: var(--neutral); } .dred { fill: var(--s2); }
.toks { font: 14px/2 var(--mono); background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px;
  overflow-wrap: anywhere; } .toks span { border-radius: 2px; color: var(--fg); }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.verdicts { display: grid; gap: 8px; } .verdicts li { max-width: 80ch; }
"""


def render() -> str:
    rows = defender_rows()
    chem, maths12, l30 = rows[0], rows[2], rows[3]
    noise = [json.loads(p.read_text()) for p in Path("results/noise").glob("gemma_*/summary.json")]
    noise_missed = sum(v["missed_ema"] for s in noise for k, v in s["results"].items() if k != "no instruction")
    tiles = [(f"{chem['red'][0]}/{chem['red'][1]}", "chemistry replies past the paper's released probe by padding (plain task: "
              f"{chem['plain'][0]}/{chem['plain'][1]}); defeated by any windowed reading"),
             (f"{maths12['red'][0]}/{maths12['red'][1]}", f"correct, computed maths answers past a layer-12 probe by wording alone (plain: {maths12['plain'][0]}/{maths12['plain'][1]})"),
             (f"{l30['red'][0]}/{l30['red'][1]}", f"past the stronger layer-30 probe after a fresh search (plain: {l30['plain'][0]}/{l30['plain'][1]}); mostly division problems"),
             ("0", "prompted evasions of in-context deception or backdoor probes in earlier phases; evasion instructions made those probes fire more")]
    return f"""<meta charset="utf-8">
<title>Probe Evasion: Red-Team Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
{CSS}
{EXTRA}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Probe evasion sprint · Redwood Research · Gemma-2-9B, harmless concept probes, prompting only</div>
    <h1>Can a red-team strategy get a model past a probe monitor?</h1>
    <p class="lede">A strong model might dodge a probe-based monitor occasionally, which matters over many traces. We test the red-team version: Claude searches for an instruction (seeing every reply's probe score), the model only follows it, and we count replies that still do the task but slip under a threshold that flags 1% of ordinary replies. The model's weights are never changed.</p>
  </header>

  <div class="glance">{''.join(f'<div class="gtile"><div class="gv">{esc(v)}</div><div class="gl">{esc(l)}</div></div>' for v, l in tiles)}</div>

  <section>
    <h2>Each monitor had a weak spot the search found within a few rounds</h2>
    <figure class="card"><figcaption><span class="key"><span class="sw" style="background:var(--neutral)"></span>plain task</span>
      <span class="key"><span class="sw s2"></span>red-team prompt</span> · share of replies that do the task (concept confirmed) and are missed · held-out tasks</figcaption>
      {defender_chart(rows)}</figure>
  </section>

  <div class="two">
    <section>
      <h2>Stronger probes: the found trick stops working</h2>
      <figure class="card"><figcaption><span class="key"><span class="sw s2"></span>found prompt vs probe trained on our data</span>
        <span class="key"><span class="sw s3"></span>probe trained on varied answer wording</span><span class="key"><span class="sw s1"></span>plain task</span>
        · same saved replies, maths probes at different layers</figcaption>{layer_chart()}</figure>
    </section>
    <section>
      <h2>It's wording, sharpened by the search</h2>
      <figure class="card"><figcaption>Layer-12 maths probe, 10-token window, 40 validation problems. A generic "write a cozy scenic sentence" request already gets some answers past; the search's more specific style rules (number mid-sentence, avoid words like "total", no unit after the number) get most of them. Neither mentions a probe.</figcaption>
        {control_bars()}</figure>
    </section>
  </div>

  <section>
    <h2>What the trick looks like to the probe</h2>
    <figure class="card"><figcaption>A correct, computed answer, coloured by the layer-12 maths probe (darker = more "maths"). The digits still light up; the calm scenery around them pulls every windowed score below the line.</figcaption>{token_strip()}</figure>
  </section>

  <section>
    <h2>What this means for red-teaming probe monitors</h2>
    <ul class="verdicts">
      <li><b>A red team with probe feedback finds the readout's weak spot fast.</b> Averaging falls to padding; our running average fell to a low-scoring opening word; a short window fell to wording that keeps the concept but not its usual phrasing. Each took a few rounds.</li>
      <li><b>The successes are surface-level.</b> In every case we can see why the probe missed (padding, phrasing, reply shape); the model isn't hiding the concept internally. Probes trained on stereotyped text are the weak link: a schemer that knows roughly what a probe was trained on could exploit this without querying it.</li>
      <li><b>Stronger probes mostly hold, even against a fresh search.</b> The trick that beats layers 6-24 fails at layers 30 and 36. Fresh searches got {l30['red'][0]} of {l30['red'][1]} past layer 30 (mostly division problems), {rows[4]['red'][0] if rows[4]['red'] else '–'} of {rows[4]['red'][1] if rows[4]['red'] else '–'} past layer 36, and {rows[5]['red'][0] if rows[5]['red'] else '–'} of {rows[5]['red'][1] if rows[5]['red'] else '–'} past a layer-30 probe trained on varied answer wording. Telling the model about the probe added nothing in a control.</li>
      <li><b>Things that did nothing:</b> random characters before the request ({noise_missed} extra misses across 27 variants and 3 concepts); telling the model a probe is watching; and, in earlier phases, any prompt against probes that read deception or backdoor intent in context.</li>
    </ul>
  </section>

  <section>
    <h2>Not yet shown</h2>
    <ul class="verdicts">
      <li>A recent reasoning model (Qwen3.5-27B, probes trained) where the model can plan the evasion in its reasoning.</li>
      <li>Transfer to high-stakes behaviour, where the plan itself sits in context and earlier phases found in-context probes catch it.</li>
    </ul>
  </section>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")


def simple_figure(out: Path = Path("share/simple_progress.png")) -> Path:
    """One clean chart for a message: maths probe misses, plain prompt vs Claude-optimized prompt, weak to strong probe."""
    import matplotlib.pyplot as plt

    from probe_evasion.proposal_figures import GREY, INK, ORANGE

    rows = [r for r in defender_rows() if r["concept"] == "Maths in the head" and r["red"]]
    names = ["Basic probe", "Later-layer probe", "Deeper-layer probe", "Later layer +\nvaried training"][:len(rows)]
    plain = [100 * r["plain"][0] / r["plain"][1] for r in rows]
    red = [100 * r["red"][0] / r["red"][1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    x = np.arange(len(rows))
    ax.bar(x - 0.19, plain, 0.36, color=GREY, label="Normal prompt")
    ax.bar(x + 0.19, red, 0.36, color=ORANGE, label="Claude-optimized prompt")
    for xi, v in zip(x, red):
        ax.text(xi + 0.19, v + 2, f"{v:.0f}%", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x, names, fontsize=10.5)
    ax.set_ylabel("Correct answers the probe misses", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Gemma-2-9B, maths probe: optimized prompts beat weaker probes", loc="left", fontsize=12.5, color=INK)
    ax.annotate("weaker probe", xy=(0, -0.2), xycoords=("data", "axes fraction"), fontsize=9, color=GREY, ha="center")
    ax.annotate("stronger probe", xy=(len(rows) - 1, -0.2), xycoords=("data", "axes fraction"), fontsize=9, color=GREY, ha="center")
    fig.tight_layout()
    fig.savefig(out)
    return out


def layer_figure(out: Path = Path("share/simple_layers.png")) -> Path:
    """Simple by-layer chart: share of correct maths answers the probe misses, normal prompt vs the best
    Claude-optimized prompt (the scenic prompt found against layer 12, or a fresh search against that layer)."""
    import matplotlib.pyplot as plt

    from probe_evasion.proposal_figures import GREY, INK, ORANGE

    ladder = json.loads((MATHS / "ladder_gemma_maths_layers.json").read_text())["results"]
    fresh = {}
    for tag, layer in (("L30", 30), ("L36", 36)):
        runs = sorted(G.glob(f"gemma_maths_window_{tag}_*"))
        a = json.loads((runs[-1] / "summary.json").read_text())["arms"]
        fresh[layer] = (a["optimized"]["missed_window"], a["optimized"]["present"])
    varied = json.loads((sorted(G.glob("gemma_maths_window_variedL30_*"))[-1] / "summary.json").read_text())["arms"]
    layers = sorted(int(k.split("@L")[1]) for k in ladder)
    plain, best = [], []
    for L in layers:
        w = ladder[f"maths@L{L}"]["window"]["missed"]
        k, n = frac(w["optimized"])
        if L in fresh and fresh[L][0] / fresh[L][1] > k / n:
            k, n = fresh[L]
        best.append(100 * k / n)
        pk, pn = frac(w["no_instruction"])
        plain.append(100 * pk / pn)
    labels = [f"Layer {L}" for L in layers] + ["Layer 30,\nvaried training"]
    best.append(100 * varied["optimized"]["missed_window"] / varied["optimized"]["present"])
    plain.append(100 * varied["no instruction"]["missed_window"] / varied["no instruction"]["present"])
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    x = np.arange(len(labels))
    ax.bar(x - 0.19, plain, 0.36, color=GREY, label="Normal prompt")
    ax.bar(x + 0.19, best, 0.36, color=ORANGE, label="Claude-optimized prompt")
    for xi, v in zip(x, best):
        ax.text(xi + 0.19, v + 2, f"{v:.0f}%", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylabel("Correct answers the probe misses", fontsize=10.5)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Gemma-2-9B, maths probe at different depths (42 layers)", loc="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out)
    return out
