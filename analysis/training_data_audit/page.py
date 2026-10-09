"""Audit page for the sarcasm probes' training data: composition, shortcut checks and many samples.
  off-policy: data/concepts/m3a_sarcasm_off_sym.jsonl (Claude-written; the main probe's data)
  on-policy:  data/concepts/m3a_sarcasm_train_aug.jsonl (Qwen's own replies)
-> share/sarcasm_training_data.html"""

import html
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data/concepts"
OUT = REPO / "share/sarcasm_training_data.html"
OFF, ON = D / "m3a_sarcasm_off_sym.jsonl", D / "m3a_sarcasm_train_aug.jsonl"
CAL = D / "m3a_sarcasm_calibration.jsonl"
ATTACKS = REPO / "missions/m3a2_sarcasm_patch/attacks.jsonl"
MARKERS = {"opens with Oh / Wow / Well / Ah": r"^(oh|wow|well|ah)\b", "exclamation mark": r"!", "“great / perfect / wonderful / brilliant”":
           r"\b(great|perfect|wonderful|brilliant|fantastic|amazing)\b", "“sure / clearly / obviously / totally”": r"\b(sure|clearly|obviously|totally)\b",
           "quotes or *italics*": r"[\"*]"}
esc = lambda s: html.escape(str(s))


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def kind(r: dict) -> str:
    return r.get("subkind") or r["kind"]


def composition(rows: list[dict]) -> str:
    kinds = sorted(Counter(kind(r) for r in rows))
    fmts = sorted(Counter(r["format"] for r in rows))
    head = "".join(f"<th>{esc(f)}</th>" for f in fmts)
    body = "".join(f"<tr><td>{esc(k)} (label {rows[[kind(r) for r in rows].index(k)]['label']})</td>" +
                   "".join(f"<td class='n'>{sum(kind(r) == k and r['format'] == f for r in rows)}</td>" for f in fmts) +
                   f"<td class='n'><b>{sum(kind(r) == k for r in rows)}</b></td></tr>" for k in kinds)
    words = {"sarcastic": "says “sarcastic”", "sincere": "says “sincere”", "": "says neither"}
    pk = "".join(f"<tr><td>{esc(words[w])}</td>" + "".join(f"<td class='n'>{sum(r['prompt_kind'] == w and r['label'] == lab for r in rows)}</td>"
                                                           for lab in (1, 0)) + "</tr>" for w in ("sarcastic", "sincere", ""))
    return (f"<table><tr><th>reply kind</th>{head}<th>total</th></tr>{body}</table>"
            f"<p class='note'>The request's wording (whether it asks for a “sarcastic” or “sincere” remark) by label. If one word mostly goes with "
            f"one label, a probe could read the request instead of the reply.</p>"
            f"<table><tr><th>request</th><th>label 1 (sarcastic)</th><th>label 0 (not)</th></tr>{pk}</table>")


def shortcuts(rows: list[dict]) -> str:
    groups = {k: [r["response"] for r in rows if kind(r) == k] for k in sorted({kind(r) for r in rows})}
    head = "".join(f"<th>{esc(k)}</th>" for k in groups)
    tr = "<tr><td>median words</td>" + "".join(f"<td class='n'>{np.median([len(t.split()) for t in v]):.0f}</td>" for v in groups.values()) + "</tr>"
    for name, pat in MARKERS.items():
        tr += f"<tr><td>{esc(name)}</td>" + "".join(
            f"<td class='n'>{100 * np.mean([bool(re.search(pat, t.strip(), re.I | re.M)) for t in v]):.0f}%</td>" for v in groups.values()) + "</tr>"
    return f"<table><tr><th>share of replies with…</th>{head}</tr>{tr}</table>"


def word_baseline(rows: list[dict]) -> dict:
    """How far surface words alone go: TF-IDF + logistic regression, 5-fold cross-validated AUROC on the training set;
    then trained on all of it, its 1% line on Qwen's clean sincere calibration replies, and how much plain vs searched
    sarcasm (valid) it misses."""
    X = [r["response"] for r in rows]
    y = [r["label"] for r in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=4.0)
    cv = cross_val_predict(clf, Xv, y, cv=5, method="predict_proba")[:, 1]
    clf.fit(Xv, y)
    cal = [r["reply"] for r in jl(CAL) if r.get("clean")]
    line = float(np.quantile(clf.predict_proba(vec.transform(cal))[:, 1], 0.99))
    att = jl(ATTACKS)
    valid = lambda r: r["present"] and (r.get("blind_rating") or 0) >= 4 and not r.get("too_long")
    miss = {}
    for arm in ("plain", "m3a_searched", "r0_search", "r1_search"):
        reps = [r["reply"] for r in att if r["arm"] == arm and valid(r)]
        miss[arm] = (int((clf.predict_proba(vec.transform(reps))[:, 1] <= line).sum()), len(reps))
    return {"cv_auroc": roc_auc_score(y, cv), "n_cal": len(cal), "miss": miss}


def cards(rows: list[dict], rng: random.Random, k: int) -> str:
    return "".join(f"<div class='card {kind(r)}'><div class='msg'>“{esc(r['message'])}”</div><div class='meta'>{esc(kind(r))} · "
                   f"{esc(r['format'])} · request says {esc(r['prompt_kind'] or 'neither')}</div><div class='reply'>{esc(r['response'])}</div></div>"
                   for r in rng.sample(rows, min(k, len(rows))))


def triplets(rows: list[dict], rng: random.Random, n: int) -> str:
    by = {}
    for r in rows:
        by.setdefault((r["message"], r["format"]), {})[kind(r)] = r
    full = [v for v in by.values() if {"sarcastic", "sincere", "blunt"} <= set(v)]
    out = ""
    for v in rng.sample(full, min(n, len(full))):
        out += (f"<tr><td class='m'>{esc(v['sarcastic']['message'])}<br><span class='meta'>{esc(v['sarcastic']['format'])}</span></td>"
                f"<td>{esc(v['sarcastic']['response'])}</td><td>{esc(v['sincere']['response'])}</td><td>{esc(v['blunt']['response'])}</td></tr>")
    return f"<table class='trip'><tr><th>message</th><th>sarcastic (1)</th><th>sincere (0)</th><th>blunt but sincere (0)</th></tr>{out}</table>"


def main() -> None:
    off, on = jl(OFF), jl(ON)
    rng = random.Random(0)
    bo, bn = word_baseline(off), word_baseline(on)
    reg = Counter(json.loads(l)["register"] for l in (REPO / "analysis/sarcasm_register/register_labels.jsonl").read_text().splitlines()
                  if json.loads(l)["group"] == "training")
    miss_row = lambda b: "".join(f"<td class='n'>{b['miss'][a][0]}/{b['miss'][a][1]}</td>" for a in ("plain", "m3a_searched", "r0_search", "r1_search"))
    sample_sections = ""
    for k in ("sarcastic", "sincere", "blunt"):
        for f in ("default", "text", "formal", "question"):
            pool = [r for r in off if kind(r) == k and r["format"] == f]
            sample_sections += f"<h4>{esc(k)} · {esc(f)} format <span class='meta'>({len(pool)} rows; 4 random)</span></h4><div class='grid'>{cards(pool, rng, 4)}</div>"
    body = f"""
<h1>Sarcasm probe training data: does it hold up?</h1>
<div class='verdict'><b>Verdict.</b> The main probe's data (written by Claude) is solid by the usual standards:
<ul>
<li>balanced labels;</li>
<li>three kinds of reply per message: sarcastic, warm sincere, and blunt-but-sincere look-alikes;</li>
<li>four answer formats;</li>
<li>labels checked by a judge;</li>
<li>a request word that carries no information about the label.</li>
</ul>
Its one real gap is the <b>register</b>. The sarcastic replies lean heavily on exaggeration and mock enthusiasm:
<ul>
<li>{100 * reg['exaggerated'] / sum(reg.values()):.0f}% exaggerated, {100 * reg['mixed'] / sum(reg.values()):.0f}% mixed, and only {100 * reg['deadpan'] / sum(reg.values()):.0f}% deadpan (Claude judge);</li>
<li>see the marker table below.</li>
</ul>
That is exactly the gap the attacks exploit. The on-policy set (Qwen's own replies) is smaller, has no blunt look-alikes, and its request word predicts the label.</div>

<h2>1. Main probe: Claude-written, {len(off):,} rows</h2>
<p class='note'><code>data/concepts/m3a_sarcasm_off_sym.jsonl</code>. Each message got a sarcastic reply, a warm sincere one, and a
blunt-but-sincere one (critical or unimpressed, meaning exactly what it says), in two of four answer formats. Labels were checked by
a Claude judge: kept 1,845/1,860 sarcastic and 3,075/3,100 sincere or blunt. A sincere or blunt reply was kept only if it was rated
2 or below and a second check found no sarcasm, irony or mock praise anywhere.</p>
{composition(off)}
<h3>Shortcut checks</h3>
{shortcuts(off)}
<p class='note'>Surface wording alone, as a check on how much the probe could get from words. A bag-of-words classifier (word and
word-pair counts, logistic regression) on these rows has a cross-validated AUROC of <b>{bo['cv_auroc']:.3f}</b>.
Trained on all rows, with its 1% line set on the same {bo['n_cal']:,} clean sincere Qwen replies, it misses this many
<b>valid</b> sarcastic Qwen replies:</p>
<table><tr><th>word-only classifier</th><th>plain</th><th>first attack</th><th>“good news”</th><th>“how thoughtful”</th></tr>
<tr><td>trained on Claude data</td>{miss_row(bo)}</tr><tr><td>trained on Qwen data</td>{miss_row(bn)}</tr></table>

<h3>Same message, three replies</h3>
{triplets(off, rng, 8)}

<h3>Random samples by kind and format</h3>
{sample_sections}

<h2>2. On-policy comparison: Qwen's own replies, {len(on):,} rows</h2>
<p class='note'><code>data/concepts/m3a_sarcasm_train_aug.jsonl</code>. These are Qwen's sarcastic and sincere replies, with labels checked by the same judge.
There are no blunt look-alikes.</p>
{composition(on)}
{shortcuts(on)}
<h3>Random samples</h3>
<div class='grid'>{cards([r for r in on if r['label'] == 1], rng, 8)}{cards([r for r in on if r['label'] == 0], rng, 8)}</div>
"""
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sarcasm Training Data</title><style>
:root {{ --bg:#f6f7f9; --fg:#1b1f25; --muted:#5f6773; --card:#fff; --line:#dde1e8; --sar:#cf3f36; --sin:#1d8467; --blunt:#7b5cc4; --accent:#2f52c4; color-scheme:light }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#111418; --fg:#e7eaef; --muted:#9aa3af; --card:#181c22; --line:#2a303a; --sar:#f06b60; --sin:#3cc49c; --blunt:#a693f0; --accent:#8ea6ff; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#111418; --fg:#e7eaef; --muted:#9aa3af; --card:#181c22; --line:#2a303a; --sar:#f06b60; --sin:#3cc49c; --blunt:#a693f0; --accent:#8ea6ff; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,"Segoe UI",sans-serif; margin:0; padding:24px 16px 60px }}
main {{ max-width:1150px; margin:0 auto }} h1 {{ font-size:26px }} h2 {{ margin-top:34px }} h4 {{ margin:18px 0 6px }}
.note {{ color:var(--muted); font-size:13.5px; max-width:95ch }} .meta {{ color:var(--muted); font-size:12px; font-weight:400 }}
.verdict {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent); padding:12px 16px; border-radius:6px; max-width:95ch }}
table {{ border-collapse:collapse; font-size:13.5px; margin:8px 0 14px }} th,td {{ border-bottom:1px solid var(--line); padding:5px 10px; text-align:left; vertical-align:top }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums }} td.m {{ color:var(--muted); max-width:26ch }} table.trip td {{ max-width:34ch }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:8px }}
.card {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line); border-radius:6px; padding:8px 10px; min-width:0 }}
.card.sarcastic {{ border-left-color:var(--sar) }} .card.sincere {{ border-left-color:var(--sin) }} .card.blunt {{ border-left-color:var(--blunt) }}
.msg {{ color:var(--muted); font-size:12.5px }} .reply {{ margin-top:4px }}
</style></head><body><main>{body}</main></body></html>""")
    print(OUT, json.dumps({"off": {"cv": bo["cv_auroc"], "miss": bo["miss"]}, "on": {"cv": bn["cv_auroc"], "miss": bn["miss"]}}))


if __name__ == "__main__":
    main()
