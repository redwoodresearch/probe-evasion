"""m51 part 2 analysis: compute vs given vs wrong-answer requests, for maths, ordering and the three m48 v2 tasks.

For each task and request version: how many replies are correct, how many copy an inserted wrong answer; for every
detector (target probe; older probe for maths / ordering; TF-IDF A, B, P as trained in analysis/tfidf_baseline) the
median, the share over its own 1% line, AUROC of compute against each other version (correct replies only), and the
drop on identical-reply pairs (the same reply text after both requests). "Position" of a wrong-answer version:
(its median - the given median) / (compute median - given median) for the same phrasing: 1 = scores like compute
(the probe reads Qwen working the answer out), 0 = like given (it reads that an answer is in the context).

    .venv/bin/python missions/m51_matched_request/analyze2.py

Writes part2/analysis.json, part2/detector_scores.jsonl, figures/part2_scores.png, figures/part2_traces.png, part2/traces.md.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from analyze import COLOUR as _C, INK, LABEL, SURFACE, auroc, ema, lines_for as m45_lines, read_jsonl  # noqa: E402
from chosen_models import chosen_detector  # noqa: E402  (analyze puts analysis/tfidf_baseline on the path)

PART2 = HERE / "part2"
TASKS = ("maths", "ordering", "false_belief", "contradiction", "acrostic")
TARGET = {"maths": "q27mrr@L48@s0", "ordering": "q27orr@L48@s0", "false_belief": "m48fa_v2@L48@s0",
          "contradiction": "m48co_v2@L48@s0", "acrostic": "m48ac_v2@L48@s0"}
OLDER = {"maths": "onpk@L48@s0", "ordering": "ordpk@L48@s0"}
TFIDF = ("tfidf_A", "tfidf_B", "tfidf_P")
NOVEL = {"maths": ("clause", "sentence"), "ordering": ("ref", "classmate"), "false_belief": ("ref",), "contradiction": ("ref",),
         "acrostic": ("ref",)}  # phrasings in no training set (part 1's maths phrasings included; checked in part2/novelty.json)
COLOUR = {"compute": _C["compute"], "given": _C["given_clause"], "wrong": "#4a3aa7"}


def detectors(task: str) -> tuple[str, ...]:
    return ("probe",) + (("older_probe",) if task in OLDER else ()) + TFIDF


def lines_for(task: str) -> dict:
    if task in OLDER:
        return m45_lines(task)
    pool = read_jsonl(REPO / "missions/m49_new_task_toggle/runs/source" / f"m49_{task}" / "calibration_scores.jsonl")
    res = json.loads((REPO / "analysis/tfidf_baseline/results" / f"{task}.json").read_text())
    probe = float(np.quantile([r["peaks"][TARGET[task]] for r in pool], 0.99))
    assert abs(probe - res["probe"]["line_1pct"]) < 1e-9
    return {"probe": probe} | {d: res[d]["line_1pct"] for d in TFIDF}


def load(task: str) -> list[dict]:
    rows = read_jsonl(PART2 / f"{task}_scored.jsonl")
    if task in OLDER:  # part 1's compute and given versions (no wrong answer inserted, so nothing to copy)
        rows = [r | {"copied_wrong": False} for r in read_jsonl(HERE / f"{task}_scored.jsonl")] + rows
    for r in rows:
        r["probe"] = r["peaks"][TARGET[task]]
        if task in OLDER:
            r["older_probe"] = r["peaks"][OLDER[task]]
        r["kind"], _, r["phrasing"] = r["version"].partition("_")
        # judge_reply flags a reply naming the item's wrong answer in every version; only wrong versions inserted it
        r["copied_wrong"] = bool(r["copied_wrong"]) and r["kind"] == "wrong"
    for v in "ABP":
        det = chosen_detector(task, v)
        for r, s in zip(rows, det.score([{"prompt": r["prompt"], "reply": r["reply"]} for r in rows])):
            r[f"tfidf_{v}"] = float(s)
    return rows


def summarise(task: str, rows: list[dict], lines: dict) -> dict:
    versions = list(dict.fromkeys(r["version"] for r in sorted(rows, key=lambda r: (r["kind"] != "compute", r["version"]))))
    ok = {v: [r for r in rows if r["version"] == v and r["correct"]] for v in versions}
    idx = {v: {(r["id"], r["seed"]): r for r in ok[v]} for v in versions}
    out = {"lines": lines, "versions": {}, "detectors": {}}
    for v in versions:
        allv = [r for r in rows if r["version"] == v]
        out["versions"][v] = {"n": len(allv), "correct": len(ok[v]), "copied_wrong": sum(r["copied_wrong"] for r in allv),
                              "neither": sum(not r["correct"] and not r["copied_wrong"] for r in allv)}
    for det in detectors(task):
        d = {"median": {}, "over_line": {}, "auroc_compute_vs": {}, "identical_reply": {}, "position_of_wrong": {},
             "copied_wrong": {}}
        for v in versions:
            xs = np.array([r[det] for r in ok[v]])
            if len(xs):
                d["median"][v], d["over_line"][v] = float(np.median(xs)), float(np.mean(xs > lines[det]))
            if v != "compute" and len(xs):
                d["auroc_compute_vs"][v] = auroc([r[det] for r in ok["compute"]], list(xs))
                same = [k for k in sorted(idx["compute"].keys() & idx[v].keys()) if idx["compute"][k]["reply"] == idx[v][k]["reply"]]
                if same:
                    diffs = np.array([idx["compute"][k][det] - idx[v][k][det] for k in same])
                    d["identical_reply"][v] = {"n_pairs": len(same), "median_compute_minus": float(np.median(diffs)),
                                               "min_compute_minus": float(diffs.min()), "share_compute_higher": float(np.mean(diffs > 0)),
                                               "other_over_line": float(np.mean([idx[v][k][det] > lines[det] for k in same]))}
            if v.startswith("wrong"):
                copied = [r[det] for r in rows if r["version"] == v and r["copied_wrong"]]
                if copied:
                    d["copied_wrong"][v] = {"n": len(copied), "median": float(np.median(copied)), "over_line": float(np.mean(np.array(copied) > lines[det]))}
                g = "given_" + v.split("_", 1)[1]
                if ok.get(g) and ok[v]:  # the direct test: wrong-but-worked-out against given, same phrasing
                    d.setdefault("auroc_wrong_vs_given", {})[v] = auroc([r[det] for r in ok[v]], [r[det] for r in ok[g]])
                    # the same reply text after the given and the wrong request: only the inserted value differs
                    same = [k for k in sorted(idx[g].keys() & idx[v].keys()) if idx[g][k]["reply"] == idx[v][k]["reply"]]
                    if same:
                        diffs = np.array([idx[v][k][det] - idx[g][k][det] for k in same])
                        d.setdefault("identical_wrong_minus_given", {})[v] = {
                            "n_pairs": len(same), "median": float(np.median(diffs)), "share_wrong_higher": float(np.mean(diffs > 0)),
                            "as_share_of_compute_minus_given": float(np.median(diffs)) / (d["median"]["compute"] - d["median"][g])}
                if g in d["median"] and v in d["median"]:
                    span = d["median"]["compute"] - d["median"][g]
                    d["position_of_wrong"][v] = (d["median"][v] - d["median"][g]) / span
        out["detectors"][det] = d
    return out


# ---------------------------------------------------------------- figures


def fig_scores(results: dict, data: dict) -> None:
    """Per task: each detector's median per version on its own scale (0 = its 1% line, 1 = its compute median)."""
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(15, 3.3 * len(TASKS)))
    fig.patch.set_facecolor(SURFACE)
    for ax, task in zip(axes, TASKS):
        res = results[task]
        dets = detectors(task)
        versions = list(res["versions"])
        width = 0.8 / len(versions)
        for j, v in enumerate(versions):
            kind = v.split("_")[0]
            ys = []
            for det in dets:
                d = res["detectors"][det]
                gap = d["median"]["compute"] - res["lines"][det]
                ys.append((d["median"][v] - res["lines"][det]) / gap if v in d["median"] and gap > 0 else np.nan)
            novel = v == "compute" or v.split("_", 1)[1] in NOVEL[task]
            ax.bar(np.arange(len(dets)) + (j - (len(versions) - 1) / 2) * width, ys, width * 0.9, color=COLOUR[kind],
                   alpha=1.0 if novel else 0.45, hatch=None if novel else "//", edgecolor=SURFACE)
            for x, y in zip(np.arange(len(dets)) + (j - (len(versions) - 1) / 2) * width, ys):
                if np.isfinite(y):
                    ax.text(x, max(y, 0) + 0.03, v.replace("given_", "g:").replace("wrong_", "w:"), rotation=90, fontsize=6,
                            ha="center", va="bottom", color=INK)
        ax.axhline(0, color=INK, ls="--", lw=1)
        ax.set_xticks(range(len(dets)), [("target probe" if d == "probe" else LABEL[d]) for d in dets], fontsize=8)
        ax.set_ylabel("median on own scale\n(0 = 1% line, 1 = compute)", fontsize=8)
        ax.set_title(f"{task}: blue compute, orange given, violet wrong-answer (correct replies only); faded / hatched = phrasing "
                     f"in or close to the training requests", fontsize=9, loc="left")
        ax.set_ylim(-1.2, 1.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "part2_scores.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


def trace_pick(task: str, rows: list[dict]) -> dict | None:
    """One item and seed whose compute, given and wrong-answer (correct) replies are the same text, novel phrasing
    preferred; else compute and given only."""
    by = {(r["id"], r["seed"], r["version"]): r for r in rows}
    ph = NOVEL[task][0]
    for need_wrong in (True, False):
        for r in rows:
            if r["version"] != "compute" or not r["correct"]:
                continue
            g, w = by.get((r["id"], r["seed"], f"given_{ph}")), by.get((r["id"], r["seed"], f"wrong_{ph}"))
            if g and g["correct"] and g["reply"] == r["reply"] and (not need_wrong or (w and w["correct"] and w["reply"] == r["reply"])):
                return {"compute": r, f"given_{ph}": g} | ({f"wrong_{ph}": w} if need_wrong else {})
    return None


def traces(data: dict, results: dict) -> None:
    picks = {t: trace_pick(t, data[t]) for t in TASKS}
    picks = {t: p for t, p in picks.items() if p}
    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 2.9 * len(picks)))
    fig.patch.set_facecolor(SURFACE)
    md = ["# Part 2 per-token traces: the same reply text after compute, given and wrong-answer requests\n"]
    for ax, (task, p) in zip(np.atleast_1d(axes), picks.items()):
        line = results[task]["lines"]["probe"]
        md.append(f"## {task} {p['compute']['id']} seed {p['compute']['seed']}\n\nreply: {p['compute']['reply']}\n")
        for v, r in p.items():
            lg = r["logits"][TARGET[task]]
            ax.plot(range(len(lg)), lg, marker="o", ms=3.5, lw=1.4, color=COLOUR[v.split("_")[0]], label=f"{v} (score {r['probe']:+.1f})")
            md.append(f"**{v}** (score {r['probe']:+.2f}; line {line:+.2f})  \nprompt: {r['prompt']}\n")
            md.append("| token | log-odds | running |\n|---|---|---|")
            md += [f"| `{t!r}` | {x:+.2f} | {e:+.2f} |" for t, x, e in zip(r["tokens"], lg, ema(lg))]
            md.append("")
        toks = p["compute"]["tokens"]
        step = max(1, len(toks) // 40)
        ax.set_xticks(range(0, len(toks), step), [toks[i].replace("$", r"\$").replace("\n", "⏎") for i in range(0, len(toks), step)],
                      fontsize=7, rotation=60)
        ax.axhline(line, color=INK, ls="--", lw=1)
        ax.set_title(f"{task}: the same reply read after each request (target probe per token; dashed: 1% line)", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "part2_traces.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    (PART2 / "traces.md").write_text("\n".join(md))


def main() -> None:
    results, data = {}, {}
    for task in TASKS:
        data[task] = load(task)
        results[task] = summarise(task, data[task], lines_for(task))
        r = results[task]
        print(f"\n[{task}] " + "; ".join(f"{v}: {x['correct']}/{x['n']} correct, {x['copied_wrong']} copied" for v, x in r["versions"].items()))
        for det, d in r["detectors"].items():
            print(f"  {det:12s} AUROC " + " ".join(f"{v} {a:.2f}" for v, a in d["auroc_compute_vs"].items()))
            print(f"  {'':12s} median " + " ".join(f"{v} {m:+.1f}" for v, m in d["median"].items()) + f" | line {r['lines'][det]:+.2f}")
            print(f"  {'':12s} over   " + " ".join(f"{v} {m:.2f}" for v, m in d["over_line"].items()))
            print(f"  {'':12s} position of wrong " + " ".join(f"{v} {m:+.2f}" for v, m in d["position_of_wrong"].items())
                  + " | AUROC wrong-correct vs given " + " ".join(f"{v} {m:.2f}" for v, m in d.get("auroc_wrong_vs_given", {}).items()))
            print(f"  {'':12s} identical wrong-given " + " ".join(f"{v} n={x['n_pairs']} {x['median']:+.1f} ({x['as_share_of_compute_minus_given']:+.2f}, higher {x['share_wrong_higher']:.2f})"
                                                       for v, x in d.get("identical_wrong_minus_given", {}).items()))
            print(f"  {'':12s} identical " + " ".join(f"{v} n={x['n_pairs']} drop {x['median_compute_minus']:+.1f}" for v, x in d["identical_reply"].items()))
    (PART2 / "analysis.json").write_text(json.dumps(results, indent=1))
    with open(PART2 / "detector_scores.jsonl", "w") as f:
        for task in TASKS:
            for r in data[task]:
                f.write(json.dumps({"task": task} | {k: r[k] for k in ("id", "version", "seed", "correct", "copied_wrong")}
                                   | {d: r[d] for d in detectors(task)}) + "\n")
    fig_scores(results, data)
    traces(data, results)


if __name__ == "__main__":
    main()
