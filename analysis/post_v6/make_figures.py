"""Figures for the v6 post, built from saved mission outputs only (no model calls, no GPU).

Run from the repo root:  .venv/bin/python analysis/post_v6/make_figures.py

Writes analysis/post_v6/figures/{trace,ladder,habit,ordinary,twins}.png and prints every plotted number. Each number is
recomputed from the saved per-reply rows where they exist and asserted equal to the mission's own summary (m31
analysis.json, m33 analysis.json, m35 report.json, m27 part_b / part_d summaries, m22 test rows).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
MISSIONS = REPO / "missions"
OUT = Path(__file__).resolve().parent / "figures"

M22_ROOT = MISSIONS / "m22_score_only_iterations/runs/qwen35_27b"
M25_ROOT = MISSIONS / "m25_program_output/runs/m25_qwen35_27b_code_neutral"
M26_ROOT = MISSIONS / "m26_ordering/runs/qwen35_27b_ordering"
M27_PART_B = MISSIONS / "m27_readout_and_style_checks/part_b"
M27_PART_D = MISSIONS / "m27_readout_and_style_checks/part_d/summary.json"
M19_STAGE2 = MISSIONS / "m19_maths_whitebox_redteam/stage2"
M28_RUNS = MISSIONS / "m28_baselines_and_writer_knowledge/runs"
M31 = MISSIONS / "m31_more_chains"
M33 = MISSIONS / "m33_self_writer_rungs"
M35_RUNS = MISSIONS / "m35_context_distillation/runs"

# Reference categorical palette (dataviz skill). Writer identity: blue / orange / aqua; no writer grey.
OPUS48 = "#2a78d6"
OPUS55 = "#eb6834"
QWEN_SELF = "#1baf7a"
NO_WRITER = "#6b6a66"
WHITE_BOX = "#4a3aa7"
ATTACKED = "#d55181"
HABIT = "#7b1f4b"  # the same attack read in the habit-trained model's own state (dark shade of the attack colour)
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

# Per task: (title, run root of the original search, target probe, m31 run folder name)
TASKS = {
    "maths": ("Maths word problems", M22_ROOT, "onpk@L48@s0", "qwen35_27b"),
    "ordering": ("Ordering five people", M26_ROOT, "ordpk@L48@s0", "qwen35_27b_ordering"),
}


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


def valid_median(rows: list[dict], key: str, probe: str, field: str = "peaks") -> float:
    values = [r[field][probe] for r in rows if r["key"] == key and r["valid"]]
    assert values, key
    return float(np.median(values))


def plain_and_line(task: str) -> tuple[float, float]:
    _, root, probe, _ = TASKS[task]
    plain = valid_median(read_jsonl(root / "test_rows.jsonl"), "plain", probe)
    return plain, one_percent_line(root, probe)


# ------------------------------------------------------------------ Figure 2: how far each attacker moves the probe

# Rows top to bottom. Each row lists (source, writer colour). Every dot is one instruction's median correct answer on
# 100 fresh problems x 5 samples, from the LAST round of a 16-round search (not picked with the probe); the fixed
# texts and random styles are not picked either. Only the white-box instruction was picked using the probe.
M33_SET = "m33"  # values from m33 analysis.json test sets (which include m28's arms)
M31_ARM = "m31"  # values from m31 analysis.json test_records (8 Opus 4.8 chains incl. m28's 3; 4 Opus 5.5 chains)
LADDER_ROWS = [
    ("Told a monitor reads\nits activations", [(M33_SET, "told a monitor is watching (m28)", "fixed", NO_WRITER, "o", 0.0)]),
    ("Told exactly how\nthe probe reads it",
     [(M33_SET, "told how it is read (m33)", "fixed", NO_WRITER, "o", 0.12),
      (M33_SET, "told how it is read, not asked to evade (m33 control)", "fixed", NO_WRITER, "D", -0.16)]),
    ("Qwen writes its own prompts\n(told the readout, no scores)",
     [(M33_SET, "Qwen-self, no scores (m33)", "last", QWEN_SELF, "o", 0.0)]),
    ("Random odd styles\n(no probe information)",
     [(M33_SET, "probe-blind random styles (m28)", "last", NO_WRITER, "s", 0.0)]),
    ("Writer told only \"a monitor\nflags computing\"",
     [(M33_SET, "Opus 4.8, no scores, no readout spec (m28)", "last", OPUS48, "o", 0.0)]),
    ("Writer told how the\nprobe reads replies",
     [(M31_ARM, "4.8/no scores", "last", OPUS48, "o", -0.14), (M31_ARM, "5.5/no scores", "last", OPUS55, "o", 0.14)]),
    ("... and shown every\ntoken's probe score",
     [(M31_ARM, "4.8/scores", "last", OPUS48, "o", -0.22), (M31_ARM, "5.5/scores", "last", OPUS55, "o", 0.0),
      (M33_SET, "Qwen-self, with scores (m22 / m26)", "last", QWEN_SELF, "o", 0.22)]),
    ("White box: training data,\nscoring tool, bigger search", [("m19", None, None, WHITE_BOX, "o", 0.0)]),
]
N_VOLITIONAL_ROWS = 3  # shaded: the monitored model supplies the strategy; no probe-derived signal, no other writer
# Means each source reports, used as a check on what is plotted.
M31_REPORTED_LAST_MEANS = {("maths", "4.8/scores"): -4.60, ("maths", "4.8/no scores"): -1.12,
                           ("maths", "5.5/scores"): -5.38, ("maths", "5.5/no scores"): -1.72,
                           ("ordering", "4.8/scores"): -9.69, ("ordering", "4.8/no scores"): -6.80,
                           ("ordering", "5.5/scores"): -13.03, ("ordering", "5.5/no scores"): -3.95}


def m31_test_rows(task: str, record: dict) -> list[dict]:
    folder = TASKS[task][3]
    if record["source"] == "m28":
        return read_jsonl(M28_RUNS / folder / "m28_test_rows.jsonl")
    runs = "runs_opus55" if record["model"] == "5.5" else "runs"
    return read_jsonl(M31 / runs / folder / "test_rows.jsonl")


def ladder_values(task: str, source: str, name: str, kind: str) -> list[float]:
    """Changes vs plain (log-odds) for one series, each recomputed from per-reply rows where possible."""
    _, root, probe, _ = TASKS[task]
    plain, _ = plain_and_line(task)
    if source == M33_SET:
        values = json.loads((M33 / "analysis.json").read_text())[task]["test"]["sets"][name][kind]["changes"]
        return [float(v) for v in values]
    if source == M31_ARM:
        records = [r for r in json.loads((M31 / "analysis.json").read_text())[task]["test_records"]
                   if r["arm"] == name and r["kind"] == kind]
        cache: dict[str, list[dict]] = {}
        values = []
        for record in records:
            label = record["source"] + record["model"]
            if label not in cache:
                cache[label] = m31_test_rows(task, record)
            recomputed = valid_median(cache[label], record["key"], probe) - plain
            assert close(recomputed, record["change"], 1e-6), (record["key"], recomputed, record["change"])
            values.append(recomputed)
        assert close(float(np.mean(values)), M31_REPORTED_LAST_MEANS[(task, name)], 0.01), (task, name)
        return values
    assert source == "m19"
    if task != "maths":
        return []
    return [valid_median(read_jsonl(root / "test_rows.jsonl"), "m19/s2_own", probe) - plain]


def check_m33_sets_against_m31() -> None:
    """The 3 m28 Opus 4.8 chains appear in both m31 and m33; their values must agree."""
    m33 = json.loads((M33 / "analysis.json").read_text())
    m31 = json.loads((M31 / "analysis.json").read_text())
    for task in TASKS:
        for arm, set_name in (("4.8/scores", "Opus 4.8, with scores (m28)"), ("4.8/no scores", "Opus 4.8, no scores (m28)")):
            from_m31 = [r["change"] for r in m31[task]["test_records"]
                        if r["arm"] == arm and r["kind"] == "last" and r["source"] == "m28"]
            from_m33 = m33[task]["test"]["sets"][set_name]["last"]["changes"]
            assert np.allclose(sorted(from_m31), sorted(from_m33), atol=1e-6), (task, arm)


def ladder_figure() -> None:
    check_m33_sets_against_m31()
    fig, axes = plt.subplots(2, 1, figsize=(10, 13.5), sharex=True)
    n_rows = len(LADDER_ROWS)
    m33 = json.loads((M33 / "analysis.json").read_text())
    for ax, task in zip(axes, TASKS):
        title = TASKS[task][0]
        plain, line = plain_and_line(task)
        share = lambda change: -100 * change / (plain - line)
        non_answer = share(m33[task]["non_answer_median"] - plain)
        print(f"\nladder {title}: plain {plain:+.2f}, line {line:+.2f} (distance {plain - line:.2f}); "
              f"Qwen's own task-skipping replies at {non_answer:.0f}%")
        ax.axhspan(n_rows - N_VOLITIONAL_ROWS - 0.5, n_rows - 0.5, color="#f3f2ee", zorder=0)
        ax.axvline(0, color=MUTED, lw=1.2)
        ax.axvline(100, color=LINE_COLOR, lw=2, ls="--")
        ax.axvline(non_answer, color=MUTED, lw=1.6, ls=":")
        ax.text(1.5, n_rows + 0.75, "plain\nanswers", color=MUTED, fontsize=12.5, va="top")
        ax.text(102, n_rows + 0.75, "1% flag\nline", color=LINE_COLOR, fontsize=12.5, va="top")
        ax.text(non_answer - 1.5, n_rows + 0.75, "Qwen's own replies\nthat skip the task", color=MUTED,
                fontsize=12.5, va="top", ha="right")
        for row, (label, series) in enumerate(LADDER_ROWS):
            y = n_rows - 1 - row
            drawn = 0
            for source, name, kind, colour, marker, dy in series:
                changes = ladder_values(task, source, name, kind)
                if not changes:
                    continue
                xs = [share(c) for c in changes]
                jitter = np.linspace(-0.07, 0.07, len(xs)) if len(xs) > 1 else np.zeros(1)
                yy = y + dy + jitter
                filled = source == "m19"
                ax.scatter(xs, yy, s={"o": 80, "s": 70, "D": 60}[marker], marker=marker,
                           facecolors=colour if filled else "white", edgecolors=colour, linewidths=2.0, zorder=3)
                drawn += len(xs)
                print(f"  {label.replace(chr(10), ' ')} | {name or 'white box'}: n={len(xs)}, mean change "
                      f"{np.mean(changes):+.2f} log-odds = {np.mean(xs):.0f}% (range {min(xs):.0f}-{max(xs):.0f}%)")
            if drawn == 0:
                ax.text(50, y, "not run", color=MUTED, fontsize=13, ha="center", va="center", style="italic")
        ax.set_title(title, loc="left", color=TEXT)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([label for label, _ in reversed(LADDER_ROWS)], fontsize=13)
        ax.set_ylim(-0.6, n_rows + 0.8)
        ax.set_xlim(-5, 111)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Median correct answer (prompt removed): % of the way from plain answers to the flag line",
                        fontsize=14)
    legend = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=OPUS48, mew=2, ms=9, label="Claude Opus 4.8 writes"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=OPUS55, mew=2, ms=9, label="Claude Opus 5.5 writes"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=QWEN_SELF, mew=2, ms=9, label="Qwen writes for itself"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=NO_WRITER, mew=2, ms=9, label="fixed text, no writer"),
        Line2D([], [], marker="s", ls="", mfc="white", mec=NO_WRITER, mew=2, ms=8, label="random styles (probe-blind)"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=NO_WRITER, mew=2, ms=7,
               label="control: same text, not asked to evade"),
        Line2D([], [], marker="o", ls="", color=WHITE_BOX, ms=9, label="white box (picked by probe)"),
        Patch(facecolor="#f3f2ee", edgecolor="none", label="shaded: volitional (Qwen's own strategy)"),
        Line2D([], [], color=MUTED, ls=":", lw=1.6, label="median reply that skips the task"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=12.5, columnspacing=1.0,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "ladder.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 3: re-read vs a model that writes this way by habit

# (row label, report.json arm names, colour). Every value is the median valid reply's score minus plain Qwen's.
HABIT_ROWS = [
    ("Prompted Qwen, re-read with\nthe prompt removed (main measure)", ["I_attack"], ATTACKED),
    ("Prompted Qwen, read with\nthe prompt in view", ["V_attack"], "#e9a3bd"),
    ("Fine-tuned Qwen's replies,\nread by ordinary Qwen", ["T_attack_s0", "T_attack_s1"], NO_WRITER),
    ("Fine-tuned Qwen: own replies,\nown activations", ["D_attack_s0", "D_attack_s1"], HABIT),
    ("Control: fine-tuned\non plain replies", ["D_plain_s0", "D_plain_s1"], "#a8a7a2"),
]
M35_REPORTED_SHARE = {"maths": 0.25, "ordering": 0.56}
# m35 instruction per task: the m28 pinned Opus 4.8 with-scores test condition it distilled.
M35_SOURCE_KEY = {"maths": "opus48_pinned/chain_2@best", "ordering": "opus48_pinned/chain_2@last"}


def check_habit_against_rows(task: str, report: dict) -> None:
    """Recompute the plotted medians from per-reply rows (m28 test rows and m35 test files)."""
    _, _, probe, folder = TASKS[task]
    plain = report["plain_median"]
    m28_rows = read_jsonl(M28_RUNS / folder / "m28_test_rows.jsonl")
    assert close(valid_median(m28_rows, "plain", probe), plain, 1e-6)
    key = M35_SOURCE_KEY[task]
    assert close(valid_median(m28_rows, key, probe) - plain, report["arms"]["I_attack"]["change"], 1e-6)
    assert close(valid_median(m28_rows, key, probe, "peaks_visible") - plain, report["arms"]["V_attack"]["change"], 1e-6)
    for seed in (0, 1):
        own = [r for r in read_jsonl(M35_RUNS / task / f"test_attack_s{seed}.jsonl") if r["valid"]]
        assert close(float(np.median([r["peaks"][probe] for r in own])) - plain,
                     report["arms"][f"D_attack_s{seed}"]["change"], 1e-6)
        assert close(float(np.median([r["peaks_base_reader"][probe] for r in own])) - plain,
                     report["arms"][f"T_attack_s{seed}"]["change"], 1e-6)
        control = [r for r in read_jsonl(M35_RUNS / task / f"test_plain_s{seed}.jsonl") if r["valid"]]
        assert close(float(np.median([r["peaks"][probe] for r in control])) - plain,
                     report["arms"][f"D_plain_s{seed}"]["change"], 1e-6)


def habit_figure() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 10.5), sharex=True)
    print()
    for ax, task in zip(axes, TASKS):
        report = json.loads((M35_RUNS / task / "report.json").read_text())
        check_habit_against_rows(task, report)
        plain, line = report["plain_median"], report["lines"]["0.01"]
        plain_main, line_main = plain_and_line(task)
        assert close(plain, plain_main, 1e-6) and close(line, line_main, 1e-4)
        share = lambda change: -100 * change / (plain - line)
        kept = np.mean([(report["arms"][f"D_attack_s{s}"]["change"] - report["arms"][f"D_plain_s{s}"]["change"])
                        / report["arms"]["I_attack"]["change"] for s in (0, 1)])
        assert close(kept, report["proxy"]["attack"]["share"], 1e-6) and close(kept, M35_REPORTED_SHARE[task], 0.005)
        n_rows = len(HABIT_ROWS)
        for row, (label, arms, colour) in enumerate(HABIT_ROWS):
            y = n_rows - 1 - row
            offsets = [0.0] if len(arms) == 1 else [0.13, -0.13]
            for arm, dy in zip(arms, offsets):
                a = report["arms"][arm]
                x, lo, hi = share(a["change"]), share(a["ci"][1]), share(a["ci"][0])
                ax.plot([lo, hi], [y + dy] * 2, color=colour, lw=2.4, zorder=2)
                ax.scatter([x], [y + dy], s=90, color=colour, zorder=3)
                print(f"habit {task} | {label.replace(chr(10), ' ')} | {arm}: change {a['change']:+.2f} log-odds = "
                      f"{x:.0f}% of the way [{lo:.0f}, {hi:.0f}], valid {a['valid']}/500")
        ax.axvline(0, color=MUTED, lw=1.2)
        ax.axvline(100, color=LINE_COLOR, lw=2, ls="--")
        ax.text(98.5, n_rows - 0.35, "1% flag\nline", color=LINE_COLOR, fontsize=12.5, va="top", ha="right")
        picked = "picked with the probe" if M35_SOURCE_KEY[task].endswith("@best") else "last round, not picked"
        ax.set_title(f"{TASKS[task][0].split()[0]} (instruction: {picked})\n"
                     f"The trained habit keeps {kept:.2f} of the re-read drop, net of control",
                     loc="left", color=TEXT, fontsize=14.5)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([label for label, _, _ in reversed(HABIT_ROWS)], fontsize=12.5)
        ax.set_ylim(-0.6, n_rows - 0.2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        print(f"habit {task}: share kept {kept:.3f} (report {report['proxy']['attack']['share']:.3f}, "
              f"95% {report['proxy']['attack']['ci']})")
    axes[-1].set_xlim(-8, 104)
    axes[-1].set_xlabel("Median correct answer: % of the way from plain answers to the flag line")
    fig.tight_layout()
    fig.savefig(OUT / "habit.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 4: attacked answers vs ordinary replies

# Searched instructions only (a writer improved them over rounds): not random styles, not fixed told-texts, not the
# white box (plotted separately). Rows are deduplicated on (instruction, problem, seed) across files.
def searched_rows(task: str) -> tuple[list[dict], list[dict]]:
    """(Opus 5.5 pinned with-scores answers, every other searched answer), valid only, on fresh problems."""
    folder = TASKS[task][3]
    sources = [(M28_RUNS / folder / "m28_test_rows.jsonl", lambda k: k.startswith("opus") or k.startswith("m26/")),
               (M31 / "runs" / folder / "test_rows.jsonl", lambda k: k != "plain"),
               (M31 / "runs_opus55" / folder / "test_rows.jsonl", lambda k: k != "plain"),
               (M33 / "runs" / folder / "test_rows.jsonl", lambda k: k.startswith("q27_") or k.startswith("earlier_q27"))]
    if task == "maths":
        sources.insert(0, (M22_ROOT / "test_rows.jsonl", lambda k: "/own/" in k))
    else:
        sources.insert(0, (M26_ROOT / "test_rows.jsonl", lambda k: k != "plain"))
    seen, opus55_scored, others = set(), [], []
    for path, keep in sources:
        for r in read_jsonl(path):
            if not keep(r["key"]) or not r["valid"]:
                continue
            ident = (r["instruction"], r["task"]["id"], r["seed"])
            if ident in seen:
                continue
            seen.add(ident)
            is_opus55_scored = "runs_opus55" in str(path) and r["key"].startswith("opus55_pinned/")
            (opus55_scored if is_opus55_scored else others).append(r)
    return opus55_scored, others


def broad_pool_scores(score_file: str, probe: str) -> list[float]:
    """m27 Part D's broad chat pool: replies containing a digit or a list marker (m27's own filter)."""
    from probe_evasion.missions.m27_readout_and_style_checks import NUMBER_OR_LIST

    rows = read_jsonl(M27_PART_D.parent / score_file)
    return [r["peaks"][probe] for r in rows if NUMBER_OR_LIST.search(r["reply"])]


ORDINARY_BROAD = {"maths": ("maths 27B (m22)", "scores_qwen35_27b.jsonl"),
                  "ordering": ("ordering (m26)", "scores_monitor.jsonl")}


def ordinary_figure() -> None:
    broad = json.loads(M27_PART_D.read_text())
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.5), sharex=True)
    rng = np.random.default_rng(1)
    print()
    for ax, task in zip(axes, TASKS):
        title, root, probe, _ = TASKS[task]
        broad_name, score_file = ORDINARY_BROAD[task]
        pool = [r["peaks"][probe] for r in read_jsonl(root / "calibration_scores.jsonl")]
        broad_pool = broad_pool_scores(score_file, probe)
        assert len(broad_pool) == broad[broad_name]["pool_n"] == 1720
        test = read_jsonl(root / "test_rows.jsonl")
        plain = [r["peaks"][probe] for r in test if r["key"] == "plain" and r["valid"]]
        opus55, others = searched_rows(task)
        opus55_scores = [r["peaks"][probe] for r in opus55]
        other_scores = [r["peaks"][probe] for r in others]
        line = one_percent_line(root, probe)
        broad_line = broad[broad_name]["lines"]["broad pool"]
        assert close(line, broad[broad_name]["lines"]["original calibration pool"])
        assert close(float(np.quantile(broad_pool, 0.99)), broad_line, 0.05)
        groups = [("ordinary\n(one sentence)", pool, NO_WRITER), ("ordinary\n(broad chat)", broad_pool, "#a8a7a2"),
                  ("plain\nanswers", plain, PLAIN), ("other searched\nprompts", other_scores, ATTACKED),
                  ("Opus 5.5 with\nscores", opus55_scores, OPUS55)]
        white_box = [r["peaks"][probe] for r in test if r["key"] == "m19/s2_own" and r["valid"]]
        groups.append(("white box", white_box, WHITE_BOX))
        for name, values in (("Opus 5.5 with scores", opus55_scores), ("other searched", other_scores)):
            print(f"ordinary {task} {name}: n={len(values)}, min {min(values):+.2f}, at or below the top ordinary "
                  f"reply ({max(pool):+.2f}, 1 of {len(pool)}) {sum(x <= max(pool) for x in values)}, under the 1% "
                  f"line ({line:+.2f}) {sum(x <= line for x in values)}, under the broad-pool line ({broad_line:+.2f}) "
                  f"{sum(x <= broad_line for x in values)}")
        if task == "maths":
            print(f"ordinary maths white box: n={len(white_box)}, under the 1% line {sum(x <= line for x in white_box)}")
        for j, (name, values, colour) in enumerate(groups):
            y = len(groups) - 1 - j
            if not values:
                ax.text(1.5, y, "not run", color=MUTED, fontsize=13, ha="center", va="center", style="italic")
                continue
            alpha = 0.3 if len(values) < 5000 else 0.08
            ax.scatter(values, y + rng.uniform(-0.3, 0.3, len(values)), s=5, color=colour, alpha=alpha, linewidths=0)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g[0] for g in reversed(groups)], fontsize=12.5)
        ax.tick_params(axis="y", length=0)
        ax.axvline(line, color=LINE_COLOR, ls="--", lw=2)
        ax.axvline(broad_line, color=LINE_COLOR, ls=":", lw=2)
        ax.axvline(max(pool), color=MUTED, ls="-", lw=1.2)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(groups) - 0.4)
        ax.set_title(title, loc="left", color=TEXT)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlim(-13, 16)
    axes[-1].set_xlabel("Probe score (log-odds); attack answers read with the prompt removed")
    legend = [Line2D([], [], color=LINE_COLOR, ls="--", lw=2, label="flag line (1% of one-sentence replies)"),
              Line2D([], [], color=LINE_COLOR, ls=":", lw=2, label="line flagging 1% of broad chat replies"),
              Line2D([], [], color=MUTED, lw=1.2, label="highest one-sentence ordinary reply")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "ordinary.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Appendix figure: task replies vs same-style no-work replies

TWIN_TASKS = [("Maths", "maths_27b", M22_ROOT, "onpk@L48@s0", "maths 27B (m22)"),
              ("Program output", "code", M25_ROOT, "cpk@L48@s0", "code (m25, m22 description)"),
              ("Ordering", "ordering", M26_ROOT, "ordpk@L48@s0", "ordering (m26)")]
ATTACK_ARMS = ["opus/own/chain_0", "opus/own/chain_1", "opus/own/chain_2"]  # the three best instructions, with scores
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
            ax.text(i, 20.5, f"AUROC {a:.3f}", ha="center", va="top", fontsize=14, color=TEXT)
            print(f"twins {title} {name.replace(chr(10), ' ')}: task median {np.median(task):+.1f} (n={len(task)}), "
                  f"same-style no-work median {np.median(twin):+.1f} (n={len(twin)}, "
                  f"{sum(x > line for x in twin)} over the line), AUROC {a:.3f}")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([c[0] for c in columns], fontsize=12.5)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-14, 21)
        ax.set_title(f"{title} (A–C: best rounds, picked with the probe)", loc="left", color=TEXT, fontsize=14.5)
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


# ------------------------------------------------------------------ Figure 1: how the probe reads one reply

def trace_figure() -> None:
    probe = "onpk@L48@s0"
    test = read_jsonl(M22_ROOT / "test_rows.jsonl")
    by_prompt = {}
    for r in test:
        by_prompt.setdefault((r["task"]["id"], r["seed"]), {})[r["key"]] = r
    arm = "opus/own/chain_1"  # the instruction quoted in appendix A (median change -5.9 on fresh problems)
    arm_median = float(np.median([r["peaks"][probe] for r in test if r["key"] == arm and r["valid"]]))
    candidates = [v for v in by_prompt.values() if v.get(arm, {}).get("valid") and v["plain"]["valid"]]
    pair = min(candidates, key=lambda v: (abs(v[arm]["peaks"][probe] - arm_median), v[arm]["reply"]))
    line = one_percent_line(M22_ROOT, probe)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.6), sharey=True)
    problem = pair["plain"]["prompt"].split(" Answer in one short sentence")[0]
    fig.suptitle(f"Problem: \"{problem}\"", x=0.01, ha="left", fontsize=14, color=MUTED)
    for ax, key, title in ((axes[0], "plain", "Plain answer"),
                           (axes[1], arm, "Under a searched instruction, re-read without it")):
        row = pair[key]
        tokens, logits = row["tokens"], row["logits"][probe] if isinstance(row["logits"], dict) else row["logits"]
        path = ema_path(logits)
        assert close(max(path), row["peaks"][probe], 1e-3), (key, max(path), row["peaks"][probe])
        xs = np.arange(len(tokens))
        ax.scatter(xs, logits, s=30, color=NO_WRITER, zorder=3, label="each token's probe log-odds")
        ax.plot(xs, path, color=TEXT, lw=2.6, label="running average (score = its peak)")
        peak = int(np.argmax(path))
        ax.scatter([peak], [path[peak]], s=160, facecolors="none", edgecolors=TEXT, linewidths=2.6, zorder=4)
        ax.axhline(line, color=LINE_COLOR, ls="--", lw=1.8, label="1% flag line")
        ax.set_xticks(xs)
        ax.set_xticklabels([t.strip() or "␣" for t in tokens], rotation=70, ha="right", rotation_mode="anchor",
                           fontsize=15)
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
    habit_figure()
    ordinary_figure()
    twins_figure()


if __name__ == "__main__":
    main()
