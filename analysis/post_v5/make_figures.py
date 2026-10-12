"""Figures for the v5 post, built from saved mission outputs only (no model calls, no GPU).

Run from the repo root:  .venv/bin/python analysis/post_v5/make_figures.py

Writes analysis/post_v5/figures/{trace,ladder,twins,ordinary}.png and prints the numbers each figure shows, after
cross-checking them against the mission summaries they come from (m28 analysis.json, m22 test_report.json, m27
part_b/summary.json and part_d/summary.json, m29's attacked-reply counts).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
MISSIONS = REPO / "missions"
OUT = Path(__file__).resolve().parent / "figures"

CENTRAL_DOTS = MISSIONS / "m29_existing_data_analyses/central_dots.jsonl"
M28_ANALYSIS = MISSIONS / "m28_baselines_and_writer_knowledge/analysis.json"
M22_ROOT = MISSIONS / "m22_score_only_iterations/runs/qwen35_27b"
M25_ROOT = MISSIONS / "m25_program_output/runs/m25_qwen35_27b_code_neutral"
M26_ROOT = MISSIONS / "m26_ordering/runs/qwen35_27b_ordering"
M27_PART_B = MISSIONS / "m27_readout_and_style_checks/part_b"
M27_PART_D = MISSIONS / "m27_readout_and_style_checks/part_d/summary.json"
M19_STAGE2 = MISSIONS / "m19_maths_whitebox_redteam/stage2"

# Reference categorical palette (dataviz skill). Writer identity (Figure 2): blue / orange / aqua.
# Reply kinds (Figures 3-4): attacked answers magenta, plain answers green, ordinary/no-work replies grey.
OPUS48 = "#2a78d6"
OPUS55 = "#eb6834"
QWEN_SELF = "#1baf7a"
NO_WRITER = "#6b6a66"
WHITE_BOX = "#4a3aa7"
ATTACKED = "#d55181"
PLAIN = "#008300"
LINE_COLOR = "#e34948"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e0"

EMA_ALPHA = 0.5  # the deployed readout: s_t = 0.5 x_t + 0.5 s_{t-1}, s_0 = 0; reply score = max_t s_t

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13.5, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 150,
})


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def one_percent_line(root: Path, probe: str) -> float:
    return logit(json.loads((root / "lines.json").read_text())["lines"][probe]["0.01"])


def ema_path(logits: list[float]) -> list[float]:
    path, s = [], 0.0
    for x in logits:
        s = EMA_ALPHA * x + (1 - EMA_ALPHA) * s
        path.append(s)
    return path


def close(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(a - b) <= tol


# ------------------------------------------------------------------ Figure 2: how far scores move per attacker

# Rows are ordered roughly by what the attacker knows about the probe. Each row lists (m29 arm_type, colour).
# Filled = a round/style picked using probe scores ("best", "best of 16", the white-box template); hollow = not
# picked with the probe (last round, the first random style of a block, fixed told-probe texts).
LADDER_RUNGS = [
    ("Qwen told \"a monitor\nreads your activations\"", [("told a probe is watching", NO_WRITER)]),
    ("Random odd styles\n(no probe knowledge)", [("random style, first of block", NO_WRITER),
                                                  ("random style, best of 16", NO_WRITER)]),
    ("Writer told only \"a monitor\nflags computing\"; no scores",
     [("pinned Opus 4.8, no readout spec, best", OPUS48), ("pinned Opus 4.8, no readout spec, last", OPUS48),
      ("no readout spec, best", OPUS55), ("no readout spec, last", OPUS55)]),
    ("Writer told how the score\nis read; no scores",
     [("pinned Opus 4.8, no scores, best", OPUS48), ("pinned Opus 4.8, no scores, last", OPUS48),
      ("no scores, best", OPUS55), ("no scores, last", OPUS55)]),
    ("Writer told how the score is\nread, shown per-token scores",
     [("pinned Opus 4.8, with scores, best", OPUS48), ("pinned Opus 4.8, with scores, last", OPUS48),
      ("with scores, best", OPUS55), ("with scores, last", OPUS55), ("model writes for itself, best", QWEN_SELF)]),
    ("White box: also training data,\nscoring tool, far bigger search", [("white-box template (m19)", WHITE_BOX)]),
]
# (central_dots setup, panel title, task key in m28 analysis.json, run root, target probe)
LADDER_SETUPS = [("maths 27B (own-reply probe)", "Maths word problems", "maths", M22_ROOT, "onpk@L48@s0"),
                 ("ordering", "Ordering five people", "ordering", M26_ROOT, "ordpk@L48@s0")]
SERIES_OFFSET = {OPUS48: -0.2, OPUS55: 0.0, QWEN_SELF: 0.2, NO_WRITER: 0.0, WHITE_BOX: 0.0}


def is_selected(arm_type: str) -> bool:
    return arm_type.endswith(", best") or arm_type.endswith("best of 16") or arm_type.startswith("white-box")


def check_ladder_against_sources(dots: list[dict]) -> None:
    """Every fresh-problem dot must equal the source mission's own 'change vs plain' for that test condition."""
    m28 = json.loads(M28_ANALYSIS.read_text())
    m22_arms = json.loads((M22_ROOT / "test_report.json").read_text())["arms"]
    n_checked = 0
    for setup, _, task, _, _ in LADDER_SETUPS:
        for d in dots:
            if d["setup"] != setup or not d["on_test"]:
                continue
            change = d["arm_median"] - d["plain_median"]
            if d["source"].startswith("m28:"):
                assert close(change, m28[task]["test"][d["key"].split("/", 1)[1]]["change"]), (setup, d["key"])
                n_checked += 1
            elif d["source"] == "m22_score_only_iterations" and d["key"] in m22_arms:
                assert close(change, m22_arms[d["key"]]["probes"]["own"]["median_change_vs_plain"]), d["key"]
                n_checked += 1
    print(f"ladder: {n_checked} fresh-problem dots match m28 analysis.json / m22 test_report.json")


def ladder_figure() -> None:
    dots = [d for d in read_jsonl(CENTRAL_DOTS) if d["on_test"]]  # fresh problems only
    check_ladder_against_sources(dots)
    m28 = json.loads(M28_ANALYSIS.read_text())
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    n_rungs = len(LADDER_RUNGS)
    for ax, (setup, title, task, root, probe) in zip(axes, LADDER_SETUPS):
        rows = [d for d in dots if d["setup"] == setup]
        plain = rows[0]["plain_median"]
        assert all(close(d["plain_median"], plain, 1e-6) for d in rows)
        line = one_percent_line(root, probe)
        share = lambda score: 100 * (plain - score) / (plain - line)
        non_answer = share(m28[task]["non_answer_median"])
        ax.axvline(0, color=MUTED, lw=1.2)
        ax.axvline(100, color=LINE_COLOR, lw=2, ls="--")
        ax.axvline(non_answer, color=MUTED, lw=1.6, ls=":")
        ax.text(1.5, n_rungs + 0.75, "plain\nanswers", color=MUTED, fontsize=12.5, va="top")
        ax.text(101.5, n_rungs + 0.75, "1% flag\nline", color=LINE_COLOR, fontsize=12.5, va="top")
        ax.text(non_answer - 1.5, n_rungs + 0.75, "Qwen's own replies\nthat skip the task", color=MUTED,
                fontsize=12.5, va="top", ha="right")
        print(f"\n{title}: plain {plain:+.2f}, line {line:+.2f} (distance {plain - line:.2f}); "
              f"non-answer median at {non_answer:.0f}% of the distance")
        for row, (label, series) in enumerate(LADDER_RUNGS):
            y = n_rungs - 1 - row
            drawn = 0
            for arm_type, colour in series:
                arms = [d for d in rows if d["arm_type"] == arm_type]
                if not arms:
                    continue
                selected = is_selected(arm_type)
                xs = [share(d["arm_median"]) for d in arms]
                for d, x in zip(arms, xs):
                    assert close(x, 100 * d["gap_line"], 0.5), (arm_type, x, d["gap_line"])
                jitter = np.linspace(-0.06, 0.06, len(xs)) if len(xs) > 1 else np.zeros(1)
                yy = y + SERIES_OFFSET[colour] + (0.07 if selected else -0.07) + jitter
                ax.scatter(xs, yy, s=95, facecolors=colour if selected else "white", edgecolors=colour,
                           linewidths=2.2, zorder=3)
                drawn += len(xs)
                print(f"  {label.replace(chr(10), ' ')} | {arm_type}: " + ", ".join(f"{x:.0f}%" for x in xs)
                      + f"  (log-odds mean {np.mean([d['arm_median'] - plain for d in arms]):+.2f})")
            if drawn == 0:
                ax.text(50, y, "not run", color=MUTED, fontsize=13, ha="center", va="center", style="italic")
        ax.set_title(title, loc="left", color=TEXT)
        ax.set_yticks(range(n_rungs))
        ax.set_yticklabels([label for label, _ in reversed(LADDER_RUNGS)], fontsize=13)
        ax.set_ylim(-0.6, n_rungs + 0.8)
        ax.set_xlim(-5, 112)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Median correct answer's score: % of the way from plain answers to the flag line")
    legend = [
        Line2D([], [], marker="o", ls="", color=OPUS48, ms=9, label="Opus 4.8"),
        Line2D([], [], marker="o", ls="", color=OPUS55, ms=9, label="Opus 5.5 (with fallback)"),
        Line2D([], [], marker="o", ls="", color=QWEN_SELF, ms=9, label="Qwen for itself"),
        Line2D([], [], marker="o", ls="", color=NO_WRITER, ms=9, label="no writer"),
        Line2D([], [], marker="o", ls="", color=WHITE_BOX, ms=9, label="white box"),
        Line2D([], [], marker="o", ls="", mfc="#999999", mec="#999999", ms=9, label="filled: picked by probe score"),
        Line2D([], [], marker="o", ls="", mfc="white", mec="#999999", mew=2, ms=9, label="hollow: not picked"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=12.5, columnspacing=1.0,
               handletextpad=0.3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "ladder.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3: task replies vs same-style non-task replies

TWIN_TASKS = [("Maths", "maths_27b", M22_ROOT, "onpk@L48@s0", "maths 27B (m22)"),
              ("Program output", "code", M25_ROOT, "cpk@L48@s0", "code (m25, m22 description)"),
              ("Ordering", "ordering", M26_ROOT, "ordpk@L48@s0", "ordering (m26)")]
ATTACK_ARMS = ["opus/own/chain_0", "opus/own/chain_1", "opus/own/chain_2"]  # the three best instructions, with scores
# A few words of each instruction (read from test_rows.jsonl 'instruction'; ordering C from its median reply).
ATTACK_LABELS = {
    "maths_27b": ["A: \"Restate the\nproblem's situation\"", "B: \"As the question\nalready tells us\"", "C: \"copied from\nthe record\""],
    "code": ["A: \"clerical\ntranscription\"", "B: \"exactly as\nyou wrote\"", "C: \"per the\nconsole log\""],
    "ordering": ["A: \"Sure, no\nproblem —\"", "B: \"At dusk I\nwander…\"", "C: \"On a quiet\nSunday the bakery\""],
}


def auroc(pos: list[float], neg: list[float]) -> float:
    pos_a, neg_a = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((pos_a > neg_a).sum() + 0.5 * (pos_a == neg_a).sum()) / (pos_a.size * neg_a.size))


def twin_columns(slug: str, root: Path, probe: str, summary_name: str) -> list[tuple[str, list[float], list[float]]]:
    summaries = json.loads((M27_PART_B / "summary.json").read_text())
    summary = summaries[summary_name]
    test = read_jsonl(root / "test_rows.jsonl")
    twins = [r for r in read_jsonl(M27_PART_B / slug / "clean.jsonl") if r["group"] == "twin" and r["clean"]]
    plain_task = [r["peaks"][probe] for r in test if r["key"] == "plain" and r["valid"]]
    plain_twin = [r["peaks"][probe] for r in twins if r["arm"] == "plain"]
    assert close(float(np.median(plain_task)), summary["plain"]["median"])
    assert close(float(np.median(plain_twin)), summary["plain_twins"]["median"])
    columns = [("plain", plain_task, plain_twin)]
    for i, arm in enumerate(ATTACK_ARMS):
        task = [r["peaks"][probe] for r in test if r["key"] == arm and r["valid"]]
        twin = [r["peaks"][probe] for r in twins if r["arm"] == arm]
        s = summary["arms"][arm]
        assert close(float(np.median(task)), s["attack_median"]) and close(float(np.median(twin)), s["styled_twin_median"])
        assert close(auroc(task, twin), s["auroc_removed"], 1e-6)
        columns.append((ATTACK_LABELS[slug][i], task, twin))
    if slug == "maths_27b":
        task = [r["logit"] for r in read_jsonl(M19_STAGE2 / "test_s2_own.jsonl") if r["valid"]]
        twin = [logit(r["scores_all"][probe]) for r in read_jsonl(M19_STAGE2 / "twin_s2_own.jsonl") if r["clean"]]
        s = summaries["m19 white-box template (maths 27B)"]
        assert len(task) == s["attack_valid"] and close(auroc(task, twin), s["auroc_removed"], 1e-6)
        columns.append(("white\nbox", task, twin))
    return columns


def twins_figure() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    rng = np.random.default_rng(0)
    print()
    for ax, (title, slug, root, probe, summary_name) in zip(axes, TWIN_TASKS):
        columns = twin_columns(slug, root, probe, summary_name)
        line = one_percent_line(root, probe)
        ax.axhline(line, color=LINE_COLOR, ls="--", lw=1.8)
        for i, (name, task, twin) in enumerate(columns):
            for values, dx, colour in ((task, -0.19, ATTACKED), (twin, 0.19, NO_WRITER)):
                xs = i + dx + rng.uniform(-0.13, 0.13, len(values))
                ax.scatter(xs, values, s=7, color=colour, alpha=0.35, linewidths=0)
                ax.plot([i + dx - 0.16, i + dx + 0.16], [np.median(values)] * 2, color=TEXT, lw=2.6)
            a = auroc(task, twin)
            ax.text(i, 19.5, f"AUROC {a:.2f}" if a < 0.995 else "AUROC 1.00", ha="center", va="top", fontsize=12,
                    color=MUTED)
            n_twins_flagged = sum(x > line for x in twin)
            print(f"twins {title} {name.replace(chr(10), ' ')}: task median {np.median(task):+.1f} (n={len(task)}), "
                  f"same-style no-work median {np.median(twin):+.1f} (n={len(twin)}, {n_twins_flagged} over the line), "
                  f"AUROC {a:.3f}")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([c[0] for c in columns], fontsize=12.5)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-14, 20)
        ax.set_title(title, loc="left", color=TEXT)
        ax.set_ylabel("Probe log-odds")
        ax.set_yticks([-10, 0, 10])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    legend = [Line2D([], [], marker="o", ls="", color=ATTACKED, ms=9, label="correct task answers"),
              Line2D([], [], marker="o", ls="", color=NO_WRITER, ms=9, label="no-work replies, same instruction"),
              Line2D([], [], color=TEXT, lw=2.6, label="median"),
              Line2D([], [], color=LINE_COLOR, ls="--", lw=1.8, label="1% flag line")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "twins.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4: attacked answers vs the ordinary-reply pools

# (title, run root, target probe, which test keys are score-only attacks, expected count from m29, m27 Part D name,
#  Part D score file)
ORDINARY_TASKS = [
    ("Maths", M22_ROOT, "onpk@L48@s0", lambda k: "/own/" in k and not k.startswith("m19/"), 7829, "maths 27B (m22)",
     "scores_qwen35_27b.jsonl"),
    ("Program output", M25_ROOT, "cpk@L48@s0", lambda k: k != "plain", 2664, "code (m25, m22 description)",
     "scores_monitor_neutral.jsonl"),
    ("Ordering", M26_ROOT, "ordpk@L48@s0", lambda k: k != "plain", 5044, "ordering (m26)", "scores_monitor.jsonl"),
]


def broad_pool_scores(score_file: str, probe: str) -> list[float]:
    """m27 Part D's broad chat pool: replies containing a digit or a list marker (m27's own filter)."""
    from probe_evasion.missions.m27_readout_and_style_checks import NUMBER_OR_LIST

    rows = read_jsonl(M27_PART_D.parent / score_file)
    return [r["peaks"][probe] for r in rows if NUMBER_OR_LIST.search(r["reply"])]


def ordinary_figure() -> None:
    broad = json.loads(M27_PART_D.read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10, 11.5), sharex=True)
    rng = np.random.default_rng(1)
    print()
    for ax, (title, root, probe, is_attack, expected_n, broad_name, score_file) in zip(axes, ORDINARY_TASKS):
        pool = [r["peaks"][probe] for r in read_jsonl(root / "calibration_scores.jsonl")]
        broad_pool = broad_pool_scores(score_file, probe)
        assert len(broad_pool) == broad[broad_name]["pool_n"] == 1720
        test = read_jsonl(root / "test_rows.jsonl")
        plain = [r["peaks"][probe] for r in test if r["key"] == "plain" and r["valid"]]
        attack = [r["peaks"][probe] for r in test if is_attack(r["key"]) and r["valid"]]
        assert len(attack) == expected_n, (title, len(attack))
        groups = [("ordinary\n(one sentence)", pool, NO_WRITER), ("ordinary\n(broad chat)", broad_pool, "#a8a7a2"),
                  ("plain", plain, PLAIN), ("score-only\nattacks", attack, ATTACKED)]
        if title == "Maths":
            groups.append(("white box\n(maths only)",
                           [r["peaks"][probe] for r in test if r["key"] == "m19/s2_own" and r["valid"]], WHITE_BOX))
        line = one_percent_line(root, probe)
        broad_line = broad[broad_name]["lines"]["broad pool"]
        assert close(line, broad[broad_name]["lines"]["original calibration pool"])
        assert close(float(np.quantile(broad_pool, 0.99)), broad_line, 0.05)
        n_low = sum(x <= max(pool) for x in attack)
        print(f"ordinary {title}: pool n={len(pool)}, max {max(pool):+.2f}; broad pool n={len(broad_pool)}; line "
              f"{line:+.2f}, broad line {broad_line:+.2f}; attack n={len(attack)}, min {min(attack):+.2f}, at or below "
              f"pool max {n_low}, under broad line {sum(x <= broad_line for x in attack)}, median {np.median(attack):+.2f}")
        for j, (name, values, colour) in enumerate(groups):
            y = len(groups) - 1 - j
            ax.scatter(values, y + rng.uniform(-0.3, 0.3, len(values)), s=6, color=colour, alpha=0.3, linewidths=0)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[0] for g in reversed(groups)], fontsize=12.5)
        ax.tick_params(axis="y", length=0)
        ax.axvline(line, color=LINE_COLOR, ls="--", lw=2)
        ax.axvline(broad_line, color=LINE_COLOR, ls=":", lw=2)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(groups) - 0.4)
        ax.set_title(f"{title}: 0 of {len(attack):,} score-only attack answers as low as the top one-sentence reply"
                     if n_low == 0 else title, loc="left", fontsize=14, color=TEXT)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlim(-13, 16)
    axes[-1].set_xlabel("Probe score (log-odds)")
    legend = [Line2D([], [], color=LINE_COLOR, ls="--", lw=2, label="1% line, one-sentence replies"),
              Line2D([], [], color=LINE_COLOR, ls=":", lw=2, label="1% line, broad chat replies")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "ordinary.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 1: how the probe reads one reply

def trace_figure() -> None:
    probe = "onpk@L48@s0"
    test = read_jsonl(M22_ROOT / "test_rows.jsonl")
    by_prompt = {}
    for r in test:
        by_prompt.setdefault((r["task"]["id"], r["seed"]), {})[r["key"]] = r
    arm = "opus/own/chain_1"  # the best own-reply test condition on fresh problems (median change -5.9)
    arm_median = float(np.median([r["peaks"][probe] for r in test if r["key"] == arm and r["valid"]]))
    candidates = [v for v in by_prompt.values() if v.get(arm, {}).get("valid") and v["plain"]["valid"]]
    pair = min(candidates, key=lambda v: (abs(v[arm]["peaks"][probe] - arm_median), v[arm]["reply"]))
    line = one_percent_line(M22_ROOT, probe)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.6), sharey=True)
    for ax, key, title in ((axes[0], "plain", "Plain answer"), (axes[1], arm, "Under a searched instruction")):
        row = pair[key]
        tokens, logits = row["tokens"], row["logits"][probe] if isinstance(row["logits"], dict) else row["logits"]
        path = ema_path(logits)
        assert close(max(path), row["peaks"][probe], 1e-3), (key, max(path), row["peaks"][probe])
        xs = np.arange(len(tokens))
        ax.scatter(xs, logits, s=30, color=NO_WRITER, zorder=3, label="each token's probe log-odds")
        ax.plot(xs, path, color=OPUS48, lw=2.6, label="running average (score = its peak)")
        peak = int(np.argmax(path))
        ax.scatter([peak], [path[peak]], s=160, facecolors="none", edgecolors=OPUS48, linewidths=2.6, zorder=4)
        ax.axhline(line, color=LINE_COLOR, ls="--", lw=1.8, label="1% flag line")
        ax.set_xticks(xs)
        ax.set_xticklabels([t.strip() or "␣" for t in tokens], rotation=70, ha="right", rotation_mode="anchor",
                           fontsize=13)
        ax.set_xlim(-0.8, len(tokens) - 0.2)
        ax.set_title(f"{title}: score {row['peaks'][probe]:+.1f}", loc="left", color=TEXT)
        ax.set_ylabel("Probe log-odds")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        print(f"trace {key}: {row['reply']!r} score {row['peaks'][probe]:+.2f}")
    print(f"trace: attacked condition median {arm_median:+.2f}; plain median "
          f"{np.median([r['peaks'][probe] for r in test if r['key'] == 'plain' and r['valid']]):+.2f}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, scatterpoints=1, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "trace.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace_figure()
    ladder_figure()
    twins_figure()
    ordinary_figure()


if __name__ == "__main__":
    main()
