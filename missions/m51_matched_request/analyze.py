"""m51 analysis: compute vs given (matched requests), per detector and per given phrasing.

Detectors: the m45 target probe (m46's re-read probe q27mrr / q27orr), the older own-reply probe (onpk / ordpk), and
the TF-IDF classifiers from analysis/tfidf_baseline as chosen there (A reply only, B prompt + reply, P prompt only).
Each detector's 1% line is its own, set on m45's look-alike calibration pool. Only correct, finished replies count
(the "given" versions are trivially answerable, so this mainly drops wrong compute answers).

    .venv/bin/python missions/m51_matched_request/analyze.py

Writes analysis.json, figures/scores.png, figures/traces.png and traces.md here.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "analysis/tfidf_baseline"))
from chosen_models import chosen_detector  # noqa: E402

M45_ROOT = {"maths": REPO / "missions/m45_toggle_sweep/runs/c/qwen35_27b", "ordering": REPO / "missions/m45_toggle_sweep/runs/c/qwen35_27b_ordering"}
PROBES = {"maths": ("q27mrr@L48@s0", "onpk@L48@s0"), "ordering": ("q27orr@L48@s0", "ordpk@L48@s0")}
DETECTORS = ("probe", "older_probe", "tfidf_A", "tfidf_B", "tfidf_P")
LABEL = {"probe": "probe (m45 target)", "older_probe": "older probe (onpk / ordpk)", "tfidf_A": "TF-IDF A (reply)",
         "tfidf_B": "TF-IDF B (prompt + reply)", "tfidf_P": "TF-IDF P (prompt only)"}
GIVEN = ("given_clause", "given_sentence")
VERSIONS = ("compute",) + GIVEN
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOUR = {"compute": "#2a78d6", "given_clause": "#eb6834", "given_sentence": "#1baf7a"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def lines_for(task: str) -> dict:
    """Each detector's 1% line on m45's look-alike pool: the probes' from their pool scores, TF-IDF's from the baseline."""
    pool = read_jsonl(M45_ROOT[task] / "calibration_scores.jsonl")
    target, older = PROBES[task]
    res = json.loads((REPO / "analysis/tfidf_baseline/results" / f"{task}.json").read_text())
    out = {"probe": float(np.quantile([r["peaks"][target] for r in pool], 0.99)),
           "older_probe": float(np.quantile([r["peaks"][older] for r in pool], 0.99))}
    assert abs(out["probe"] - res["probe"]["line_1pct"]) < 1e-9
    return out | {f"tfidf_{v}": res[f"tfidf_{v}"]["line_1pct"] for v in "ABP"}


def load(task: str) -> list[dict]:
    rows = read_jsonl(HERE / f"{task}_scored.jsonl")
    target, older = PROBES[task]
    for r in rows:
        r["probe"], r["older_probe"] = r["peaks"][target], r["peaks"][older]
    for v in "ABP":
        det = chosen_detector(task, v)
        for r, s in zip(rows, det.score([{"prompt": r["prompt"], "reply": r["reply"]} for r in rows])):
            r[f"tfidf_{v}"] = float(s)
    return rows


def auroc(pos: list[float], neg: list[float]) -> float:
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))


def summarise(task: str, rows: list[dict], lines: dict) -> dict:
    ok = [r for r in rows if r["correct"]]
    out = {"n": {v: sum(r["version"] == v for r in rows) for v in VERSIONS},
           "n_correct": {v: sum(r["version"] == v for r in ok) for v in VERSIONS}, "lines": lines, "detectors": {}}
    by = {v: [r for r in ok if r["version"] == v] for v in VERSIONS}
    # pairs: the same item and seed, compute and given both correct
    key = lambda r: (r["id"], r["seed"])
    idx = {v: {key(r): r for r in by[v]} for v in VERSIONS}
    for det in DETECTORS:
        sc = {v: [r[det] for r in by[v]] for v in VERSIONS}
        d = {"median": {v: float(np.median(sc[v])) for v in VERSIONS},
             "over_line": {v: float(np.mean(np.array(sc[v]) > lines[det])) for v in VERSIONS},
             "auroc_compute_vs": {g: auroc(sc["compute"], sc[g]) for g in GIVEN} | {"both given": auroc(sc["compute"], sc[GIVEN[0]] + sc[GIVEN[1]])}}
        d["paired_median_compute_minus_given"] = {}
        for g in GIVEN:
            common = sorted(idx["compute"].keys() & idx[g].keys())
            diffs = [idx["compute"][k][det] - idx[g][k][det] for k in common]
            # on a common scale: the paired drop as a share of the way from the compute median down to the 1% line
            d["paired_median_compute_minus_given"][g] = {"median": float(np.median(diffs)), "share_compute_higher": float(np.mean(np.array(diffs) > 0)),
                                                        "n_pairs": len(common),
                                                        "as_share_of_compute_to_line": float(np.median(diffs)) / (d["median"]["compute"] - lines[det])}
        # pairs whose compute and given replies are the identical text: only the request differs
        d["identical_reply_pairs"] = {}
        for g in GIVEN:
            same = [k for k in sorted(idx["compute"].keys() & idx[g].keys()) if idx["compute"][k]["reply"] == idx[g][k]["reply"]]
            diffs = np.array([idx["compute"][k][det] - idx[g][k][det] for k in same])
            d["identical_reply_pairs"][g] = {"n_pairs": len(same), "median_compute_minus_given": float(np.median(diffs)),
                                             "min_compute_minus_given": float(diffs.min()), "share_compute_higher": float(np.mean(diffs > 0)),
                                             "given_over_line": float(np.mean([idx[g][k][det] > lines[det] for k in same]))}
        out["detectors"][det] = d
    return out


# ---------------------------------------------------------------- figures and traces


def fig_scores(results: dict, data: dict) -> None:
    tasks = list(results)
    fig, axes = plt.subplots(len(tasks), len(DETECTORS), figsize=(18, 3.6 * len(tasks)))
    fig.patch.set_facecolor(SURFACE)
    for row, task in zip(axes, tasks):
        ok = [r for r in data[task] if r["correct"]]
        for ax, det in zip(row, DETECTORS):
            ax.set_facecolor(SURFACE)
            vals = [[r[det] for r in ok if r["version"] == v] for v in VERSIONS]
            lo, hi = min(map(min, vals)), max(map(max, vals))
            bins = np.linspace(lo, hi, 40)
            for v, xs in zip(VERSIONS, vals):
                ax.hist(xs, bins=bins, histtype="step", lw=2, color=COLOUR[v], label=v.replace("_", " "))
            ax.axvline(results[task]["lines"][det], color=INK, ls="--", lw=1)
            a = results[task]["detectors"][det]["auroc_compute_vs"]["both given"]
            ax.set_title(f"{task}: {LABEL[det]}\nAUROC compute vs given {a:.3f}", fontsize=9, loc="left")
            ax.set_yticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
        row[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Matched requests: score distributions of correct replies (dashed: each detector's own 1% line)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "scores.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


def pick_trace_items(task: str, rows: list[dict], n: int = 3) -> list[dict]:
    """Items (seed 10) whose compute and given-clause replies are the same correct text, spread over templates /
    relations: the probe reads the same tokens, and only the request differs."""
    by = {(r["id"], r["seed"], r["version"]): r for r in rows}
    out, seen = [], set()
    for it in read_jsonl(HERE / f"{task}_items.jsonl"):
        group = it.get("template", it.get("puzzle", {}).get("relation"))
        c, g = by[(it["id"], 10, "compute")], by[(it["id"], 10, "given_clause")]
        if group not in seen and c["correct"] and g["correct"] and c["reply"] == g["reply"]:
            seen.add(group)
            out.append({"item": it, "compute": c, "given_clause": g, "given_sentence": by[(it["id"], 10, "given_sentence")]})
        if len(out) == n:
            break
    return out


def ema(xs: list[float]) -> list[float]:
    from probe_evasion.missions.m19_maths_whitebox_redteam import ema_path

    return list(ema_path(xs))


def traces(data: dict, results: dict) -> None:
    md = ["# Per-token traces: compute vs given, identical reply text (seed 10)\n",
          "Pairs where Qwen wrote exactly the same reply to the compute request and to the given-clause request, so the "
          "probe reads the same tokens and only the request differs. Each reply token with the target probe's per-token "
          "log-odds and its EMA-0.5 running value (the reply's score is the running value's peak). The given-sentence "
          "reply for the same item is listed too (its text may differ). Line = the probe's 1% line.\n"]
    picks = {t: pick_trace_items(t, data[t]) for t in data}
    n = sum(len(v) for v in picks.values())
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.7 * n))
    fig.patch.set_facecolor(SURFACE)
    axes = iter(axes)
    for task, ps in picks.items():
        target = PROBES[task][0]
        line = results[task]["lines"]["probe"]
        for p in ps:
            ax = next(axes)
            ax.set_facecolor(SURFACE)
            md.append(f"## {task} {p['item']['id']}\n")
            for v in VERSIONS:
                r = p[v]
                lg = r["logits"][target]
                md.append(f"**{v}** (score {r['peaks'][target]:+.2f}; line {line:+.2f})  \nprompt: {r['prompt']}  \nreply: {r['reply']}\n")
                md.append("| token | log-odds | running |\n|---|---|---|")
                md += [f"| `{t!r}` | {x:+.2f} | {e:+.2f} |" for t, x, e in zip(r["tokens"], lg, ema(lg))]
                md.append("")
                if v != "given_sentence":
                    ax.plot(range(len(lg)), lg, marker="o", ms=4, lw=1.5, color=COLOUR[v],
                            label=f"{v.replace('_', ' ')} request (score {r['peaks'][target]:+.1f})")
            toks = p["compute"]["tokens"]
            ax.set_xticks(range(len(toks)), [t.replace("$", r"\$") for t in toks], fontsize=8, rotation=0)
            ax.axhline(line, color=INK, ls="--", lw=1)
            ax.set_title(f"{task} {p['item']['id']}: the same reply read after each request (target probe, per token; dashed: 1% line)",
                         fontsize=9, loc="left")
            ax.legend(fontsize=7.5, frameon=False, loc="upper right")
            ax.set_ylabel("log-odds", fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "traces.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    (HERE / "traces.md").write_text("\n".join(md))


def main() -> None:
    (HERE / "figures").mkdir(exist_ok=True)
    results, data = {}, {}
    for task in ("maths", "ordering"):
        data[task] = load(task)
        results[task] = summarise(task, data[task], lines_for(task))
        r = results[task]
        print(f"[{task}] correct: {r['n_correct']} of {r['n']}")
        for det in DETECTORS:
            d = r["detectors"][det]
            print(f"  {det:12s} AUROC clause {d['auroc_compute_vs']['given_clause']:.3f} sentence {d['auroc_compute_vs']['given_sentence']:.3f} | "
                  f"over line " + " ".join(f"{v} {d['over_line'][v]:.2f}" for v in VERSIONS) + " | median "
                  + " ".join(f"{v} {d['median'][v]:+.2f}" for v in VERSIONS) + f" (line {r['lines'][det]:+.2f})")
    (HERE / "analysis.json").write_text(json.dumps(results, indent=1))
    keep = ("id", "version", "seed", "correct") + DETECTORS
    with open(HERE / "detector_scores.jsonl", "w") as f:
        for task in data:
            for r in data[task]:
                f.write(json.dumps({"task": task} | {k: r[k] for k in keep}) + "\n")
    fig_scores(results, data)
    traces(data, results)


if __name__ == "__main__":
    main()
