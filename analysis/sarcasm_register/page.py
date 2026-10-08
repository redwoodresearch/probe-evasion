"""Page: was the deadpan register that evades the sarcasm probe missing from its training data? Samples + counts.
Reads register_labels.jsonl (judge_register.py). -> share/sarcasm_register_check.html"""

import html
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "share/sarcasm_register_check.html"
LINE_LOGIT = math.log(0.9950 / 0.0050)
REGISTERS = ("deadpan", "mixed", "exaggerated")
COLOR = {"deadpan": "var(--dead)", "mixed": "var(--mixed)", "exaggerated": "var(--exag)"}


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def words(t: str) -> set[str]:
    return set(re.findall(r"[a-z']+", t.lower()))


def group(r: dict) -> str:
    if r["group"] == "training":
        return "Probe training: Claude's sarcastic replies"
    return f"Qwen {r['group'].split()[1]} prompt, {'missed' if r['missed'] else 'caught'}"


def bar_rows(rows: list[dict]) -> str:
    order = ["Probe training: Claude's sarcastic replies", "Qwen plain prompt, caught", "Qwen plain prompt, missed",
             "Qwen searched prompt, caught", "Qwen searched prompt, missed"]
    out = []
    for g in order:
        c = Counter(r["register"] for r in rows if group(r) == g)
        n = sum(c.values())
        segs = "".join(f'<span class="seg" style="width:{100 * c[k] / n:.2f}%;background:{COLOR[k]}" title="{k} {c[k]}"></span>' for k in REGISTERS)
        nums = " · ".join(f"{k} {c[k]} ({c[k] / n:.0%})" for k in REGISTERS)
        out.append(f'<div class="brow"><div class="blabel">{g}<small>n = {n}</small></div><div class="bar">{segs}</div><div class="bnums">{nums}</div></div>')
    return "\n".join(out)


def card(r: dict, extra: str = "") -> str:
    score = ""
    if "score" in r:
        lg = logit(r["score"])
        cls = "miss" if lg < LINE_LOGIT else "hit"
        score = f'<span class="score {cls}">probe {lg:+.1f} ({"missed" if cls == "miss" else "flagged"})</span>'
    fmt = f'<span class="tag">{r["format"]} format</span>' if "format" in r else ""
    return (f'<div class="card"><div class="msg">“{html.escape(r["message"])}”</div><div class="reply">{html.escape(r["reply"])}</div>'
            f'<div class="meta"><span class="tag" style="color:{COLOR[r["register"]]}">{r["register"]}</span>{fmt}{score}</div>'
            f'<div class="why">judge: {html.escape(r["reason"])}</div>{extra}</div>')


def section(title: str, note: str, cards: list[str]) -> str:
    return f'<h3>{title}</h3><p class="note">{note}</p><div class="grid">{"".join(cards)}</div>'


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "register_labels.jsonl").read_text().splitlines()]
    rng = random.Random(0)
    train = [r for r in rows if r["group"] == "training"]
    tw = [(r, words(r["reply"])) for r in train]
    pick = lambda xs, k: rng.sample(xs, min(k, len(xs)))

    def nearest(r: dict) -> str:
        w = words(r["reply"])
        best = max(tw, key=lambda t: len(w & t[1]) / max(1, len(w | t[1])))[0]
        return (f'<div class="near">closest training reply (word overlap): “{html.escape(best["reply"])}” '
                f'<span style="color:{COLOR[best["register"]]}">[{best["register"]}]</span></div>')

    by = lambda g, reg=None: [r for r in rows if group(r) == g and (reg is None or r["register"] == reg)]
    missed = by("Qwen searched prompt, missed")
    flip = re.compile(r"\b(convenient|relaxing|efficient|save[sd]?|saving|good deal|nice|smart)\b", re.I)
    n_flip_missed = sum(bool(flip.search(r["reply"])) for r in missed)
    n_flip_train_dead = sum(bool(flip.search(r["reply"])) for r in by("Probe training: Claude's sarcastic replies", "deadpan"))
    n_flip_train = sum(bool(flip.search(r["reply"])) for r in train)
    c_train = Counter(r["register"] for r in train)
    fmt_dead = Counter(r["format"] for r in train if r["register"] == "deadpan")
    fmt_all = Counter(r["format"] for r in train)
    dead_missed = len(by("Qwen searched prompt, missed", "deadpan"))
    dead_caught = len(by("Qwen searched prompt, caught", "deadpan"))
    plain_dead = by("Qwen plain prompt, caught", "deadpan") + by("Qwen plain prompt, missed", "deadpan")

    body = f"""
<h1>Sarcasm: did the probe ever see deadpan sarcasm?</h1>
<p class="lede">The searched prompt makes Qwen write flat, literal-sounding sarcasm, and the off-policy sarcasm probe misses
most of it (290 of 500 attempts vs 12 for the plain prompt). Was that register absent from the probe's training data?
A Claude Sonnet judge labelled how every sarcastic reply delivers its sarcasm: <b style="color:var(--dead)">deadpan</b>
(flat statement of the opposite, no hyperbole or mock enthusiasm), <b style="color:var(--exag)">exaggerated</b>
(hyperbole, mock praise, interjections), or <b style="color:var(--mixed)">mixed</b>.</p>

<div class="verdict"><b>Answer: present but rare, and of a different kind.</b>
<ul>
<li>{c_train['deadpan']} of {len(train)} training sarcastic replies ({c_train['deadpan'] / len(train):.0%}) are deadpan; {c_train['mixed'] / len(train):.0%} mixed; {c_train['exaggerated'] / len(train):.0%} exaggerated.
Deadpan is mostly in the default one-to-two-sentence format ({fmt_dead['default']} of {c_train['deadpan']}).</li>
<li>Qwen's plain sarcasm is almost never deadpan ({len(plain_dead)} of 477). The searched prompt's evading replies are {dead_missed / len(missed):.0%} deadpan.</li>
<li>Deadpan alone does not guarantee a miss: of the searched prompt's deadpan replies, {dead_missed} were missed and {dead_caught} caught.</li>
<li>The specific "calls the bad thing convenient / efficient / a time saver" move appears in {n_flip_missed} of {len(missed)} evading replies ({n_flip_missed / len(missed):.0%}),
but in only {n_flip_train} of {len(train)} training replies ({n_flip_train / len(train):.0%}; {n_flip_train_dead} of them deadpan). So a large share of evasions use one sentence template that the training data barely contains, and almost never in deadpan form.</li>
</ul></div>

<h2>Register, by group</h2>
<div class="bars">{bar_rows(rows)}</div>
<p class="note">Qwen replies counted: valid only (both judges rate them sarcastic, length cap met); "missed" = under the main probe's 1% line (295 here vs 290 in RESULT.md, which counts distinct replies only).
Deadpan share of training replies by format: {", ".join(f"{k} {fmt_dead[k]}/{fmt_all[k]}" for k in fmt_all)}.</p>

<h2>Samples</h2>
<p class="note">Random samples (seed 0). Probe scores are log-odds; the 1% line is {LINE_LOGIT:+.1f}. Training replies are Claude's and were never scored on Qwen here.</p>
{section("Training data: deadpan (all registers shown as the judge labelled them)", f"{c_train['deadpan']} such replies; 14 shown.", [card(r) for r in pick(by("Probe training: Claude's sarcastic replies", "deadpan"), 14)])}
{section("Training data: mixed", "8 shown.", [card(r) for r in pick(by("Probe training: Claude's sarcastic replies", "mixed"), 8)])}
{section("Training data: exaggerated", "8 shown.", [card(r) for r in pick(by("Probe training: Claude's sarcastic replies", "exaggerated"), 8)])}
{section("Qwen, searched prompt, valid and MISSED", "16 shown, each with the training reply that shares the most words.", [card(r, nearest(r)) for r in pick(missed, 16)])}
{section("Qwen, searched prompt, valid but CAUGHT (deadpan ones)", "Same style, still flagged. 8 shown.", [card(r) for r in pick(by("Qwen searched prompt, caught", "deadpan"), 8)])}
{section("Qwen, plain prompt, caught", "What normal sarcasm from Qwen looks like. 6 shown.", [card(r) for r in pick(by("Qwen plain prompt, caught"), 6)])}
{section("Qwen, plain prompt, the 12 misses", "All of the plain prompt's misses.", [card(r) for r in by("Qwen plain prompt, missed")])}
"""
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sarcasm Register Check</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1f23; --muted:#5f6670; --card:#ffffff; --line:#e3e0d8; --dead:#2f7d6d; --mixed:#b07a1a; --exag:#c2483a; --accent:#3b5bdb; color-scheme: light }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14161a; --fg:#e7e9ec; --muted:#9aa2ad; --card:#1c1f25; --line:#2c3038; --dead:#5cc3ad; --mixed:#e0aa45; --exag:#f07a6c; --accent:#8ea6ff; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg:#14161a; --fg:#e7e9ec; --muted:#9aa2ad; --card:#1c1f25; --line:#2c3038; --dead:#5cc3ad; --mixed:#e0aa45; --exag:#f07a6c; --accent:#8ea6ff; color-scheme: dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, "Segoe UI", sans-serif; margin:0; padding:24px 16px 60px; }}
main {{ max-width:1100px; margin:0 auto; }}
h1 {{ font-size:26px; margin:0 0 8px; text-wrap:balance }} h2 {{ margin-top:34px; font-size:20px }} h3 {{ margin:26px 0 4px; font-size:16px }}
.lede {{ max-width:75ch; color:var(--muted) }} .note {{ color:var(--muted); font-size:13px; margin:2px 0 10px; max-width:90ch }}
.verdict {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent); padding:12px 18px; border-radius:6px; max-width:90ch }}
.verdict li {{ margin:4px 0 }}
.bars {{ display:flex; flex-direction:column; gap:10px }}
.brow {{ display:grid; grid-template-columns: 260px 1fr; gap:4px 14px; align-items:center }}
.blabel {{ font-weight:600; font-size:14px }} .blabel small {{ display:block; color:var(--muted); font-weight:400 }}
.bar {{ display:flex; height:22px; border-radius:4px; overflow:hidden; gap:2px; background:var(--line) }} .seg {{ display:block; height:100% }}
.bnums {{ grid-column:2; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:10px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 12px; min-width:0 }}
.msg {{ color:var(--muted); font-size:13px }} .reply {{ margin:4px 0 6px; font-size:15px }}
.meta {{ display:flex; gap:10px; flex-wrap:wrap; font-size:12px }} .tag {{ font-weight:600 }} .score {{ font-variant-numeric:tabular-nums }}
.score.miss {{ color:var(--dead) }} .score.hit {{ color:var(--exag) }}
.why {{ font-size:12px; color:var(--muted); margin-top:4px }} .near {{ font-size:12px; margin-top:6px; padding-top:6px; border-top:1px dashed var(--line); color:var(--muted) }}
@media (max-width:600px) {{ .brow {{ grid-template-columns:1fr }} .bnums {{ grid-column:1 }} }}
</style></head><body><main>{body}</main></body></html>""")
    print(OUT)


if __name__ == "__main__":
    main()
