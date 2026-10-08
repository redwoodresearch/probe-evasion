"""Dot plot for mission m1a's main finding: every reply's own-replies-probe score (layer 48, EMA reading), no-maths
replies split by answer style, with the 1% lines from the all-styles pool, the pool without emails, and the
one-sentence pool; then plain and attack answers.

    uv run python -m probe_evasion.missions.m1a_figures
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion import gepa
from probe_evasion.missions.m1a_analysis import calibration_negatives
from probe_evasion.missions.m1a_calibration_audit import ON_POLICY_RUNS, VOLUME_COPIES, line, style_of
from probe_evasion.missions.m1a_offpolicy_maths import OUT, SCORES, jl

PROBE, READING = "on@L48", "ema"
NEG_GROUPS = [  # (label, style groups), drawn left to right
    ("lookup\nemails", ("given / email",)),
    ("lookup paragraphs,\ntips, restate", ("given / short paragraph", "given / answer plus tip", "given / restate then answer")),
    ("other multi-line\nlookups", ("given / bullets", "given / faq", "given / dialogue", "given / headline", "given / casual text",
                                   "given / formal report")),
    ("recalled facts,\nlonger styles", ("recalled / casual text", "recalled / short paragraph")),
    ("one-sentence\nno-maths", ("given / one sentence", "given / one sentence (v2 search negatives)", "recalled / one sentence",
                                "recalled / one sentence (v2 search negatives)")),
]


def distinct_valid(rows: list[dict]) -> list[float]:
    seen, out = set(), []
    for r in rows:
        if gepa.lenient_maths_present(r) and (r["task"]["task"], r["reply"]) not in seen:
            seen.add((r["task"]["task"], r["reply"]))
            out.append(gepa.readings(r["token_scores"], r["token_logits"])[READING])
    return out


def main(out: Path = OUT / "dots_global_line.png") -> Path:
    neg = calibration_negatives("all styles", audited=True)
    one = calibration_negatives("one sentence", audited=True)
    score = lambda r: r["scores"][PROBE][READING]
    groups = {label: [score(r) for r in neg if style_of(r) in styles] for label, styles in NEG_GROUPS}
    v2 = VOLUME_COPIES / ON_POLICY_RUNS[0]
    groups["plain answers\n(held-out x10)"] = distinct_valid(jl(v2 / "large_test_plain.jsonl"))
    groups["v2 prompt\n(held-out x10)"] = distinct_valid(jl(v2 / "large_test_optimized.jsonl"))
    groups["plain answers\n(new division x3)"] = distinct_valid(jl(v2 / "fresh_test_plain.jsonl"))
    groups["v2 prompt\n(new division x3)"] = distinct_valid(jl(v2 / "fresh_test_optimized.jsonl"))
    if (SCORES / "format_negatives.jsonl").exists():
        fmt = gepa.clean_maths_negatives([r for r in jl(SCORES / "format_negatives.jsonl") if r["arm"] == "scene, v2 prompt" and r["finished"]])
        groups["clean no-maths lookups\nin the v2 format"] = [score(r) for r in fmt]
    lines = {"1% line, all styles": (line([score(r) for r in neg], 0.01), "-"),
             "1% line, all styles but emails": (line([score(r) for r in neg if style_of(r) != "given / email"], 0.01), "--"),
             "1% line, one-sentence replies": (line([score(r) for r in one], 0.01), ":")}
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=140)
    rng = np.random.default_rng(0)
    for i, (label, xs) in enumerate(groups.items()):
        colour = "#7a7a7a" if i < len(NEG_GROUPS) else ("#1f77b4" if label.startswith("plain") else "#d62728")
        if label.startswith("clean no-maths lookups"):
            colour = "#2ca02c"
        ys = np.clip(xs, 3e-3, 1 - 1e-4)
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(ys)), ys, s=7, color=colour, alpha=0.6, linewidths=0)
        ax.text(i, 4e-3, f"n={len(xs)}", ha="center", fontsize=7)
    for name, (y, ls) in lines.items():
        ax.axhline(y, color="k", ls=ls, lw=1, label=f"{name} ({y:.3f})")
    ax.set_yscale("logit")
    ax.set_ylim(3e-3, 1 - 1e-4)
    ticks = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(range(len(groups)), list(groups), fontsize=7.5)
    ax.set_ylabel("own-replies probe, layer 48, EMA reading (logit scale)")
    ax.set_title("No-maths replies (grey, by answer style), plain answers (blue), the frozen v2 prompt's answers (red). "
                 "Distinct valid answers only.", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    return out




def candidate_plot(out: Path = OUT / "dots_candidate.png") -> Path:
    """The divaug-v2 candidate on the seed-1234 division problems, own-replies probe (layer 48, EMA): plain answers,
    the candidate's answers, the same answers with the instruction removed, plain answers with the instruction in
    front, and no-maths lookups answered under the instruction, with the one-sentence and own-format 1% lines."""
    from probe_evasion.missions.m1a_candidates import candidate_report

    rep = json.loads((OUT / "candidate_report.json").read_text()) if (OUT / "candidate_report.json").exists() else candidate_report()
    arm = "divaug-v2 winner"
    div = jl(SCORES / "candidate_division.jsonl")
    ctx = [r for r in jl(SCORES / "candidate_context.jsonl") if r["arm"] == f"plain reply, {arm} in front" and r["lenient"]]
    fmt = gepa.clean_maths_negatives([r for r in jl(SCORES / "candidate_format_negatives.jsonl") if r["arm"] == arm and r["finished"]])
    get = lambda rs, key="scores": [r[key][PROBE][READING] for r in rs]
    groups = {"plain answers": get([r for r in div if r["arm"] == "plain" and r["lenient"]]),
              "candidate prompt\nanswers": get([r for r in div if r["arm"] == arm and r["lenient"]]),
              "same answers,\ninstruction removed": get([r for r in div if r["arm"] == arm and r["lenient"]], "scores_instruction_removed"),
              "plain answers,\ninstruction in front": get(ctx),
              "no-maths lookups\nunder the instruction": get(fmt)}
    lines = {"1% line, one-sentence no-maths replies": (rep["one_sentence_lines"][PROBE]["0.01"], ":"),
             "1% line, no-maths replies under the instruction": (rep["format_lines"][arm][PROBE]["0.01"], "--")}
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    rng = np.random.default_rng(0)
    colours = ["#1f77b4", "#d62728", "#ff9896", "#9ecae1", "#7a7a7a"]
    for i, ((label, xs), c) in enumerate(zip(groups.items(), colours)):
        ys = np.clip(xs, 3e-3, 1 - 1e-4)
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(ys)), ys, s=7, color=c, alpha=0.6, linewidths=0)
        ax.text(i, 4e-3, f"n={len(xs)}", ha="center", fontsize=7)
    for name, (y, ls) in lines.items():
        ax.axhline(y, color="k", ls=ls, lw=1, label=f"{name} ({y:.3f})")
    ax.set_yscale("logit")
    ax.set_ylim(3e-3, 1 - 1e-4)
    ticks = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(range(len(groups)), list(groups), fontsize=8)
    ax.set_ylabel("own-replies probe, layer 48, EMA reading")
    ax.set_title("divaug-v2 frozen prompt on 300 new division problems x 2 samples (valid answers, attempts counted)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.06))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    return out


def confirm_plot(out: Path = OUT / "dots_confirm_peak.png") -> Path:
    """Pre-registered confirmation: the two peak-trained probes side by side. Clean one-sentence no-maths replies (the
    line's pool; garden-area lookups marked), plain answers, both frozen prompts' answers (also with the instruction
    removed), and no-maths lookups under each instruction (the own-format pool). Valid answers, attempts counted."""
    rep = json.loads((OUT / "confirm_report.json").read_text())
    by = {}
    for f in ("peak_calibration", "peak_calibration_one_sentence", "peak_v2_calibration"):
        for r in jl(SCORES / f"{f}.jsonl"):
            by[(r["task"]["task"], r["reply"])] = r
    neg = [by[(r["task"]["task"], r["reply"])] for r in calibration_negatives("one sentence", audited=True)]
    rows = jl(SCORES / "confirm.jsonl")
    fmt = gepa.clean_maths_negatives([r for r in jl(SCORES / "peak_candidate_format_negatives.jsonl") if r["finished"]])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=140, sharey=True)
    for ax, probe, title in zip(axes, ("offpeak@L48", "onpeak@L48"), ("trained on Claude's replies", "trained on Qwen's own replies")):
        get = lambda rs, key="scores": [r[key][probe][READING] for r in rs]
        valid = lambda arm: [r for r in rows if r["arm"] == arm and r["lenient"]]
        groups = {"no-maths, one sentence\n(other templates)": get([r for r in neg if "garden" not in r["task"]["task"]]),
                  "no-maths, one sentence\n(garden-area lookups)": get([r for r in neg if "garden" in r["task"]["task"]]),
                  "plain answers": get(valid("plain")),
                  "divaug-v2 prompt": get(valid("divaug-v2 winner")),
                  "divaug-v2,\ninstruction removed": get(valid("divaug-v2 winner"), "scores_instruction_removed"),
                  "window_L48 prompt": get(valid("window_L48 winner")),
                  "window_L48,\ninstruction removed": get(valid("window_L48 winner"), "scores_instruction_removed"),
                  "no-maths lookups under\neither instruction": get(fmt)}
        colours = ["#7a7a7a", "#b07aa1", "#1f77b4", "#d62728", "#ff9896", "#e6550d", "#fdae6b", "#2ca02c"]
        rng = np.random.default_rng(0)
        for i, ((label, xs), c) in enumerate(zip(groups.items(), colours)):
            ys = np.clip(xs, 1e-3, 1 - 1e-4)
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(ys)), ys, s=5, color=c, alpha=0.5, linewidths=0)
            ax.text(i, 1.3e-3, f"n={len(xs)}", ha="center", fontsize=6.5)
        for fa, ls in (("0.001", ":"), ("0.01", "-"), ("0.05", "--")):
            ax.axhline(rep["lines"][probe][fa], color="k", ls=ls, lw=1, label=f"{float(fa):.1%} line, one-sentence no-maths ({rep['lines'][probe][fa]:.3f})")
        ax.set_yscale("logit")
        ax.set_ylim(1e-3, 1 - 1e-4)
        ticks = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xticks(range(len(groups)), list(groups), fontsize=6.5, rotation=30, ha="right")
        ax.set_title(f"{probe}: {title}, trained on the EMA peak", fontsize=10, loc="left")
        ax.legend(fontsize=7, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("probe score, EMA reading (logit scale)")
    fig.suptitle("Pre-registered confirmation: 300 new division + 200 new mixed problems x 2 samples (valid answers, attempts counted)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main(), candidate_plot(), confirm_plot())
