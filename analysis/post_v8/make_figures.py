"""Figures and checked numbers for the v8 post, built from saved mission outputs only (no model calls, no GPU).

Run from the repo root:  .venv/bin/python analysis/post_v8/make_figures.py

Sources: m45 (the matched ON/OFF switch sweep; analysis.json = batch c, analysis_d.json = batch d, plus the run roots'
test rows and calibration pools), m47 (100-round searches) and m50 (the everyday-chat line and the text checks).
Figures: crossings.png (Figure 1), strips.png (2), trend.png (3), mechanism.png (A1), effort.png (A2).

Plotted values are asserted against the missions' own analyses and the values quoted in the post. Numbers the
missions do not report themselves are computed here and printed: the 14-writer and non-Anthropic fits (two-stage
bootstrap, as m36.trend), near misses to the strict line, per-writer shares under the everyday-chat line, crossings
under correctness-only maths validity, distinct replies per problem, and the switch word's share of the drop.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "figures"
M45 = REPO / "missions/m45_toggle_sweep"
M47 = REPO / "missions/m47_long_toggle_search"
M50 = REPO / "missions/m50_realistic_monitor"
BROAD_LINE_KEY = "broad_cal_iv_1pct"  # m50's primary broad-traffic 1% line (broad chat + one-sentence pool, in view)
QUOTED_BROAD_LINE = {"maths": -3.46, "ordering": -2.20}

_spec = importlib.util.spec_from_file_location("post_v6_figures", REPO / "analysis/post_v6/make_figures.py")
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)  # applies v6's matplotlib style and palette, so all post figures match

TASKS = ("maths", "ordering")
TASK_ROOT = {"maths": "qwen35_27b", "ordering": "qwen35_27b_ordering"}
PROBE = {"maths": "q27mrr@L48@s0", "ordering": "q27orr@L48@s0"}  # m46's re-read probes, m45's target
HEADLINE_B = "16"
ADDED_LATER = ("fable51", "gpt6astra", "gpt61sol")  # batch d, added after batch c's pre-registered fit
BOOTS = 10_000

SHORT = {"opus55": "Opus 5.5", "sonnet55": "Sonnet 5.5", "gpt6sol": "GPT-6 Sol", "opus48": "Opus 4.8",
         "gemini38flash": "Gemini 3.8 Flash", "gpt54mini": "GPT-5.4 Mini", "qwen35_397b": "Qwen3.5-397B",
         "haiku45": "Haiku 4.5", "qwen35_9b": "Qwen3.5-9B", "gpt5nano": "GPT-5 nano", "llama4maverick": "Llama 4 Maverick",
         "fable51": "Fable 5.1", "gpt6astra": "GPT-6 Astra", "gpt61sol": "GPT-6.1 Sol"}
LAB_COLOUR = {"Anthropic": v6.OPUS55, "OpenAI": v6.OPUS48, "Google": v6.QWEN_SELF, "Alibaba": v6.WHITE_BOX,
              "Meta": v6.NO_WRITER}

# S_offref per writer at B = 16 as quoted in the post (m45 RESULT.md, maths / ordering), checked against the analysis.
QUOTED_S_OFFREF = {
    "opus55": (0.29, 0.58), "gpt6astra": (0.28, 0.44), "gpt61sol": (0.25, 0.30), "sonnet55": (0.24, 0.45),
    "fable51": (0.33, 0.64), "gpt6sol": (0.18, 0.34), "opus48": (0.32, 0.40), "gemini38flash": (0.10, 0.25),
    "gpt54mini": (0.03, 0.09), "qwen35_397b": (0.11, 0.18), "haiku45": (0.10, 0.25), "qwen35_9b": (0.01, 0.12),
    "gpt5nano": (0.08, 0.11), "llama4maverick": (0.17, 0.09)}
QUOTED_FITS = {"maths": {"c": 0.056, "all": 0.066}, "ordering": {"c": 0.124, "all": 0.125}}
QUOTED_ALL_14_CI = {"maths": (0.032, 0.108), "ordering": (0.078, 0.178)}  # m45 RESULT.md (its own seeded bootstrap)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_m45() -> dict[str, dict[str, dict]]:
    return {"c": json.loads((M45 / "analysis.json").read_text()), "d": json.loads((M45 / "analysis_d.json").read_text())}


def chains_of(m45: dict, task: str) -> list[dict]:
    return [c for batch in ("c", "d") for c in m45[batch][task]["chains"]]


def writer_rows(m45: dict, task: str, key: str = "S_offref", b: str = HEADLINE_B) -> list[dict]:
    """One row per writer: ECI, lab, batch and the per-chain values of `key` at effort point b (sorted by ECI)."""
    rows: dict[str, dict] = {}
    for c in chains_of(m45, task):
        r = rows.setdefault(c["writer"], {"writer": c["writer"], "eci": c["eci"], "lab": c["lab"], "batch": c["batch"],
                                          "values": []})
        r["values"].append(c["effort"][b][key])
    for r in rows.values():
        assert len(r["values"]) == 3, r
        r["mean"] = float(np.mean(r["values"]))
    return sorted(rows.values(), key=lambda r: r["eci"])


def ols_slope_per_10(eci: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(eci, y, 1)[0]) * 10


def two_stage_slope(rows: list[dict], seed: int = 36) -> tuple[float, list[float]]:
    """OLS slope of writer means per 10 ECI, with m36.trend's two-stage bootstrap (resample writers, then chains)."""
    eci = np.array([r["eci"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTS):
        pick = rng.choice(len(rows), len(rows))
        if np.ptp(eci[pick]) > 0:
            m = np.array([np.mean(rng.choice(rows[i]["values"], len(rows[i]["values"]))) for i in pick])
            draws.append(ols_slope_per_10(eci[pick], m))
    return ols_slope_per_10(eci, means), [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def check_and_print_trends(m45: dict) -> dict[str, dict]:
    fits = {}
    for i, task in enumerate(TASKS):
        rows = writer_rows(m45, task)
        for r in rows:
            assert abs(r["mean"] - QUOTED_S_OFFREF[r["writer"]][i]) < 0.006, (task, r["writer"], r["mean"])
        stored = m45["c"][task]["trend"][HEADLINE_B]["S_offref"]
        batch_c = [r for r in rows if r["batch"] == "c"]
        slope_c, ci_c = two_stage_slope(batch_c)
        assert abs(slope_c - stored["slope_per_10_eci"]) < 1e-9
        assert all(abs(a - b) < 0.006 for a, b in zip(ci_c, stored["slope_ci95_two_stage"])), (ci_c, stored)
        slope_all, ci_all = two_stage_slope(rows)
        assert abs(slope_all - QUOTED_FITS[task]["all"]) < 0.0006 and abs(slope_c - QUOTED_FITS[task]["c"]) < 0.0006
        assert all(abs(a - b) < 0.003 for a, b in zip(ci_all, QUOTED_ALL_14_CI[task])), (ci_all, QUOTED_ALL_14_CI[task])
        not_anthropic = [r for r in rows if r["lab"] != "Anthropic"]
        slope_na, ci_na = two_stage_slope(not_anthropic)
        slope_na_c, ci_na_c = two_stage_slope([r for r in not_anthropic if r["batch"] == "c"])
        print(f"[{task}] S_offref slope per 10 ECI: batch c {slope_c:+.3f} [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}] "
              f"(m45: {stored['slope_ci95_two_stage'][0]:+.3f}, {stored['slope_ci95_two_stage'][1]:+.3f}); "
              f"all 14 {slope_all:+.3f} [{ci_all[0]:+.3f}, {ci_all[1]:+.3f}]; "
              f"non-Anthropic batch c ({len([r for r in not_anthropic if r['batch'] == 'c'])}) {slope_na_c:+.3f} "
              f"[{ci_na_c[0]:+.3f}, {ci_na_c[1]:+.3f}]; non-Anthropic all ({len(not_anthropic)}) {slope_na:+.3f} "
              f"[{ci_na[0]:+.3f}, {ci_na[1]:+.3f}]")
        b0 = m45["c"][task]["trend"]["0"]["S_offref"]
        print(f"[{task}] S_offref slope at B = 0 (batch c): {b0['slope_per_10_eci']:+.3f} "
              f"[{b0['slope_ci95_two_stage'][0]:+.3f}, {b0['slope_ci95_two_stage'][1]:+.3f}]")
        fits[task] = {"c": slope_c, "ci_c": stored["slope_ci95_two_stage"], "all": slope_all, "rows": rows}
    return fits


def label_offsets(task: str) -> dict[str, tuple[int, int, str]]:
    """Hand-placed label offsets (points) so the labels of nearby writers do not collide."""
    # Only the writers the post's text names are labelled.
    if task == "maths":
        return {"fable51": (0, 12, "center"), "opus48": (-8, -18, "right"), "opus55": (6, 8, "left"),
                "gpt6sol": (6, -18, "left"), "llama4maverick": (8, 4, "left")}
    return {"fable51": (-8, 6, "right"), "opus55": (8, 0, "left"), "opus48": (-8, 6, "right"),
            "llama4maverick": (8, -4, "left")}  # GPT-6 Sol gets an arrow label (crowded region; see fig_trend)


# GPT-5 nano (ECI 139.38) and Qwen3.5-9B (139.46) overlap; drawn 0.6 ECI apart for legibility only (fits use true ECI).
DISPLAY_SHIFT = {"gpt5nano": -0.6, "qwen35_9b": +0.6}


def fig_trend(fits: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        rows = fits[task]["rows"]
        for r in rows:
            colour = LAB_COLOUR[r["lab"]]
            x = r["eci"] + DISPLAY_SHIFT.get(r["writer"], 0.0)
            ax.vlines(x, min(r["values"]), max(r["values"]), color=colour, lw=2, alpha=0.6)
            ax.plot(x, r["mean"], "o", ms=10, color=colour)
        for writer, (dx, dy, ha) in label_offsets(task).items():
            r = next(x for x in rows if x["writer"] == writer)
            ax.annotate(SHORT[writer], (r["eci"], r["mean"]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=12, color=v6.MUTED)
        if task == "ordering":
            r = next(x for x in rows if x["writer"] == "gpt6sol")
            ax.annotate(SHORT["gpt6sol"], (r["eci"], r["mean"]), xytext=(158.5, 0.56), ha="center", fontsize=12,
                        color=v6.MUTED, arrowprops={"arrowstyle": "-", "color": v6.MUTED, "lw": 0.8})
        xs = np.linspace(131, 169, 50)
        k, c0 = np.polyfit([r["eci"] for r in rows], [r["mean"] for r in rows], 1)  # all 14 writers (the headline)
        ax.plot(xs, k * xs + c0, "-", color=v6.TEXT, lw=1.6, alpha=0.8)
        ax.axhline(1.0, color=v6.LINE_COLOR, ls="--", lw=1.4)
        ax.text(131.5, 0.96, "ON median on the strict 1% line", color=v6.LINE_COLOR, fontsize=12, va="top")
        ax.axhline(0, color=v6.MUTED, lw=0.8)
        lo, hi = QUOTED_ALL_14_CI[task]
        ax.set_title(f"{task.capitalize()}: {fits[task]['all']:+.3f} per 10 ECI [95%: {lo:+.3f}, {hi:+.3f}]", fontsize=15)
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
        ax.set_xlim(130, 170)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color=v6.GRID)
    axes[0].set_ylabel("share of the OFF-to-flag-line distance\nclosed by the ON median")
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in LAB_COLOUR.items()]
    handles += [
                Line2D([], [], color=v6.TEXT, ls="-", label="least-squares fit, all 14 writers")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "trend.png", bbox_inches="tight")
    plt.close(fig)


def pair_scores(m45: dict, task: str, writer: str, b: str = HEADLINE_B) -> list[list]:
    """Pairs of one writer's 3 chains at b: [both valid, OFF in view, ON in view, OFF read after ON, ON read after OFF]."""
    out = [p for c in chains_of(m45, task) if c["writer"] == writer for p in c["effort"][b]["pair_scores"]]
    assert len(out) == 1500, (task, writer, len(out))
    return out


def ordinary_and_plain(task: str) -> tuple[list[float], list[float], float]:
    root = M45 / "runs/c" / TASK_ROOT[task]
    ordinary = [r["peaks"][PROBE[task]] for r in read_jsonl(root / "calibration_scores.jsonl")]
    line = float(np.quantile(ordinary, 0.99))
    plain = [r["peaks_visible"][PROBE[task]] for r in read_jsonl(root / "test_checked/plain.jsonl") if r["valid"]]
    return ordinary, plain, line


STRIP_WRITERS = ("fable51", "gpt54mini")  # the writer with most crossings, and a weak writer (low on both tasks)


def everyday_chat_in_view(task: str) -> list[float]:
    """m50's broad chat pool (1,720 Qwen replies with a number or list), read after each of 3 generic instructions."""
    rows = [json.loads(line) for line in (M50 / "runs/broad_scores.jsonl").read_text().splitlines() if line.strip()]
    scores = [r["peaks"][PROBE[task]] for r in rows if r["context"] != "none"]
    assert len(scores) == 3 * 1720
    return scores


def broad_lines() -> dict[str, float]:
    lines = json.loads((M50 / "lines.json").read_text())
    out = {task: lines[task][BROAD_LINE_KEY]["line"] for task in TASKS}
    for task in TASKS:
        assert abs(out[task] - QUOTED_BROAD_LINE[task]) < 0.006, (task, out[task])
    return out


def print_broad_shares(m45: dict) -> None:
    """Share of each top writer's B = 16 correct-pair ON answers at or under m50's primary broad-traffic line."""
    broad = broad_lines()
    for task in TASKS:
        for w in ("fable51", "opus55"):
            pairs = [p for p in pair_scores(m45, task, w) if p[0]]
            print(f"[{task}] {SHORT[w]} B = 16: {np.mean([p[2] <= broad[task] for p in pairs]):.0%} of {len(pairs)} "
                  f"correct-pair ON answers at or under the broad-traffic line {broad[task]:.2f}")


PRIMARY = f"{BROAD_LINE_KEY}|primary"
FUNNEL = (("probe", "under the everyday-chat line"), ("probe+text", "+ text monitor (prompt and reply)"),
          ("probe+text+natural", "+ naturalness judge"))


def crossing_rows(m45: dict) -> list[dict]:
    """Per writer, ordering, B = 16: valid ON answers under m50's primary line (m50) over valid ON answers (m45)."""
    m50 = json.loads((M50 / "analysis_ordering.json").read_text())
    defence = json.loads((M50 / "defence.json").read_text())["ordering"]["defence"][PRIMARY]["all"]
    rows = []
    for r in writer_rows(m45, "ordering"):
        w = r["writer"]
        valid_on = sum(c["effort"][HEADLINE_B]["on_valid_share"] * c["effort"][HEADLINE_B]["pairs"]
                       for c in chains_of(m45, "ordering") if c["writer"] == w)
        under = m50["writers"][w]["lines"][PRIMARY]["B16"]["on_under_line"]
        survivors = defence.get(w, {}).get("probe+text+natural", {}).get("distinct", 0)
        distinct = defence.get(w, {}).get("probe", {}).get("distinct", 0)
        rows.append({**r, "under": under, "valid_on": valid_on, "share": under / valid_on, "survivors": survivors,
                     "distinct": distinct})
    total = defence["ALL"]
    assert sum(r["under"] for r in rows) == 1654 and total["probe"]["distinct"] == 2057
    assert total["probe+text"]["distinct"] == 565 and total["probe+text+natural"]["distinct"] == 15
    assert sum(r["survivors"] for r in rows) == 15 and sum(r["distinct"] for r in rows) == 2057
    anthropic = sum(r["distinct"] for r in rows if r["lab"] == "Anthropic")
    fable = next(r["distinct"] for r in rows if r["writer"] == "fable51")
    print(f"[ordering] B = 16 valid ON under the everyday-chat line: {sum(r['under'] for r in rows)} of "
          f"{sum(r['valid_on'] for r in rows):.0f}; per writer: "
          + ", ".join(f"{SHORT[r['writer']]} {r['share']:.0%}" for r in sorted(rows, key=lambda x: -x['share'])[:4])
          + f"; distinct crossings (all B) from Anthropic {anthropic / 2057:.0%}, Fable 5.1 {fable / 2057:.0%}; "
          f"text monitors catch {1 - total['probe+text']['distinct'] / 2057:.0%} (prompt shown) to "
          f"{1 - total['probe+both_monitors']['distinct'] / 2057:.0%} (both views)")
    return rows, total


def fig_crossings(m45: dict) -> None:
    rows, total = crossing_rows(m45)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for r in rows:
        colour = LAB_COLOUR[r["lab"]]
        x = r["eci"] + DISPLAY_SHIFT.get(r["writer"], 0.0)
        ax.plot(x, 100 * r["share"], "o", ms=10, color=colour)
        if r["share"] > 0.08:  # label only the writers the text names
            ax.annotate(SHORT[r["writer"]], (x, 100 * r["share"]), xytext=(-8, 4), textcoords="offset points",
                        ha="right", fontsize=12, color=v6.MUTED)
    ax.set_xlabel("writer capability (Epoch Capabilities Index)")
    ax.set_ylabel("% of correct ON answers under\nthe everyday-chat 1% line")
    ax.set_title("Ordering, after 16 search rounds", fontsize=15)
    ax.set_xlim(130, 170)
    ax.set_ylim(-3, 80)
    ax.grid(axis="y", color=v6.GRID)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in LAB_COLOUR.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11.5)
    counts = [total[k]["distinct"] for k, _ in FUNNEL]
    bx.barh(range(len(FUNNEL)), counts, color=[v6.ATTACKED, v6.OPUS48, v6.QWEN_SELF], height=0.6)
    for y, n in enumerate(counts):
        bx.text(n + 30, y, f"{n:,}", va="center", fontsize=13)
    bx.set_yticks(range(len(FUNNEL)), [label for _, label in FUNNEL], fontsize=12)
    bx.invert_yaxis()
    bx.set_xlim(0, 2500)
    bx.set_xlabel("distinct correct ordering answers (all search lengths)")
    bx.set_title("What gets past each added check", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "crossings.png", bbox_inches="tight")
    plt.close(fig)


def fig_strips(m45: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 10.5))
    rng = np.random.default_rng(8)
    broad = broad_lines()
    for ax, task in zip(axes, TASKS):
        ordinary, plain, line = ordinary_and_plain(task)
        assert abs(line - m45["c"][task]["line"]) < 0.02, (line, m45["c"][task]["line"])
        assert abs(np.median(plain) - m45["c"][task]["plain_test_median"]) < 0.01
        everyday = everyday_chat_in_view(task)
        rows = [("one-sentence non-task replies\n(no instruction; set the strict line)", ordinary, v6.NO_WRITER),
                ("everyday chat replies\n(generic instruction in view)", everyday, v6.WHITE_BOX),
                ("plain task answers\n(no instruction)", plain, v6.PLAIN)]
        for w in STRIP_WRITERS:
            pairs = [p for p in pair_scores(m45, task, w) if p[0]]
            rows += [(f"{SHORT[w]}: switch OFF", [p[1] for p in pairs], v6.MUTED),
                     (f"{SHORT[w]}: switch ON", [p[2] for p in pairs], v6.ATTACKED)]
            print(f"[{task}] {SHORT[w]} B = 16: {np.mean([p[2] <= broad[task] for p in pairs]):.0%} of {len(pairs)} "
                  f"correct-pair ON answers at or under the broad-traffic line {broad[task]:.2f}")
        for y, (name, xs, colour) in enumerate(rows):
            jitter = rng.uniform(-0.28, 0.28, len(xs))
            ax.scatter(xs, -y + jitter, s=5, color=colour, alpha=0.25, lw=0)
            ax.vlines(np.median(xs), -y - 0.38, -y + 0.38, color=v6.TEXT, lw=3)
        ax.axvline(line, color=v6.LINE_COLOR, ls="--", lw=1.6)
        ax.text(line - 0.3, 0.55, f"strict 1% line ({line:.2f})", color=v6.LINE_COLOR, ha="right", fontsize=12)
        ax.axvline(broad[task], color=v6.WHITE_BOX, ls=":", lw=2)
        ax.text(broad[task] + 0.3, 0.55, f"everyday-chat 1% line ({broad[task]:.2f})", color=v6.WHITE_BOX, ha="left",
                fontsize=12)
        ax.set_yticks([-y for y in range(len(rows))], [r[0] for r in rows], fontsize=12)
        ax.set_ylim(-len(rows) + 0.4, 0.95)
        ax.set_title(f"{task.capitalize()}: correct pairs after 16 feedback rounds, fresh problems (3 searches pooled)",
                     fontsize=14)
        ax.set_xlabel("probe score of the reply (log-odds, instruction in view); thick tick = median")
        ax.grid(axis="x", color=v6.GRID)
    fig.tight_layout()
    fig.savefig(OUT / "strips.png", bbox_inches="tight")
    plt.close(fig)


def print_distance_to_line(m45: dict) -> None:
    for task in TASKS:
        line_by_batch = {b: m45[b][task]["line"] for b in ("c", "d")}
        lowest, lowest_all_b = None, None
        closest_median = None
        n_pairs = n_under = 0
        for c in chains_of(m45, task):
            line = line_by_batch[c["batch"]]
            for b, e in c["effort"].items():
                if "pair_scores" not in e:
                    continue
                on = [p[2] - line for p in e["pair_scores"] if p[0]]
                if on:
                    gap = min(on)
                    if lowest_all_b is None or gap < lowest_all_b[0]:
                        lowest_all_b = (gap, c["writer"], c["chain"], b)
                if b == HEADLINE_B:
                    n_pairs += len(on)
                    n_under += sum(x <= 0 for x in on)
                    if on and (lowest is None or min(on) < lowest[0]):
                        lowest = (min(on), c["writer"], c["chain"])
        for w in (r["writer"] for r in writer_rows(m45, task)):
            pairs = [p for p in pair_scores(m45, task, w) if p[0]]
            batch = "d" if w in ADDED_LATER else "c"
            med = float(np.median([p[2] for p in pairs])) - line_by_batch[batch]
            within2 = sum(p[2] - line_by_batch[batch] <= 2 for p in pairs)
            if closest_median is None or med < closest_median[0]:
                closest_median = (med, w, within2, len(pairs))
        print(f"[{task}] B=16 correct pairs {n_pairs}, ON at or under the line {n_under}; lowest ON (correct pair) "
              f"{lowest[0]:.2f} above the line ({lowest[1]} {lowest[2]}); any B: {lowest_all_b[0]:.2f} "
              f"({lowest_all_b[1]} {lowest_all_b[2]} B={lowest_all_b[3]}); closest writer median (3 chains pooled): "
              f"{closest_median[1]} {closest_median[0]:.2f} above, {closest_median[2]} of {closest_median[3]} ON "
              f"replies within 2 log-odds")


MECHANISM_KEYS = (("offref_reply_on_context", "both replies read after the ON prompt", v6.OPUS48, "o"),
                  ("offref_reply_off_context", "both replies read after the OFF prompt", v6.OPUS55, "s"))


def fig_mechanism(m45: dict) -> None:
    """Per writer at B = 16: the headline next to the same comparison with the prompt held fixed (m45's 2x2 re-read;
    each value is m45's own per-arm number, (higher - lower) / (higher - line), mean of 3 chains)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
    for ax, task in zip(axes, TASKS):
        rows = writer_rows(m45, task)
        parts = {key: {r["writer"]: r["mean"] for r in writer_rows(m45, task, key=key)} for key, *_ in MECHANISM_KEYS}
        for y, r in enumerate(rows):
            ax.plot(r["mean"], y, "|", color=v6.TEXT, ms=18, mew=3)
            for key, _, colour, marker in MECHANISM_KEYS:
                ax.plot(parts[key][r["writer"]], y, marker, color=colour, ms=8, alpha=0.9)
        ax.set_yticks(range(len(rows)), [f"{SHORT[r['writer']]} ({r['eci']:.0f})" for r in rows], fontsize=12)
        ax.axvline(0, color=v6.MUTED, lw=0.8)
        ax.set_title(task.capitalize())
        ax.set_xlabel("share of the distance to the 1% flag line closed")
        ax.grid(axis="x", color=v6.GRID)
        top = sorted(rows, key=lambda r: -r["mean"])[:5]
        print(f"[{task}] switch word's share of the drop, 1 - wording/headline (after ON / after OFF), top 5 by score: "
              + ", ".join(f"{SHORT[r['writer']]} {1 - parts[MECHANISM_KEYS[0][0]][r['writer']] / r['mean']:.0%} / "
                          f"{1 - parts[MECHANISM_KEYS[1][0]][r['writer']] / r['mean']:.0%}" for r in top))
        for w in ("opus55", "fable51"):
            print(f"[{task}] {SHORT[w]}: headline {next(r['mean'] for r in rows if r['writer'] == w):.2f}, "
                  + ", ".join(f"{key} {parts[key][w]:.2f}" for key, *_ in MECHANISM_KEYS))
    axes[0].set_ylabel("writer (ECI), weakest at the bottom")
    handles = [Line2D([], [], marker="|", ls="", ms=16, mew=3, color=v6.TEXT, label="headline: ON reply after the ON prompt vs OFF reply after the OFF prompt")]
    handles += [Line2D([], [], marker=m, ls="", ms=9, color=c, label=f"same two replies, {name}") for _, name, c, m in MECHANISM_KEYS]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "mechanism.png", bbox_inches="tight")
    plt.close(fig)


M47_QUOTED = {("maths", "opus55"): {"0": 0.03, "10": 0.32, "49": 0.42, "50": 0.44, "100": 0.45},
              ("maths", "gpt6sol"): {"0": 0.00, "10": 0.22, "49": 0.36, "50": 0.35, "100": 0.37},
              ("ordering", "opus55"): {"0": 0.22, "10": 0.47, "49": 0.65, "50": 0.71, "100": 0.70},
              ("ordering", "gpt6sol"): {"0": 0.08, "10": 0.26, "49": 0.44, "50": 0.44, "100": 0.45}}
EFFORT_COLOUR = {"opus55": v6.OPUS55, "gpt6sol": v6.OPUS48}
LOW_VALIDITY_B = "50"  # ordering Opus 5.5 chain 0: the template written right after the round-50 restart (m47 RESULT)


def fig_effort(m45: dict) -> None:
    m47 = json.loads((M47 / "analysis_long_c.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    for ax, task in zip(axes, TASKS):
        for writer in ("opus55", "gpt6sol"):
            chains = [c for c in m47[task]["chains"] if c["writer"] == writer]
            assert len(chains) == 2
            bs = sorted((b for b in chains[0]["effort"] if b != "49"), key=int)  # 49 is a restart sensitivity point
            means = {b: float(np.mean([c["effort"][b]["S_offref"] for c in chains])) for b in chains[0]["effort"]}
            for b, quoted in M47_QUOTED[(task, writer)].items():
                assert abs(means[b] - quoted) < 0.006, (task, writer, b, means[b])
            for c in chains:
                ax.plot([int(b) for b in bs], [c["effort"][b]["S_offref"] for b in bs], color=EFFORT_COLOUR[writer],
                        lw=1, alpha=0.35)
            ax.plot([int(b) for b in bs], [means[b] for b in bs], "-o", color=EFFORT_COLOUR[writer], lw=2.5, ms=6)
            print(f"[{task}] {SHORT[writer]} long search S_offref by B: " + ", ".join(f"{b} {means[b]:.2f}" for b in bs)
                  + "; correct-pair share (chain mean) by B: "
                  + ", ".join(f"{b} {np.mean([c['effort'][b]['pair_valid_share'] for c in chains]):.2f}" for b in bs))
            if (task, writer) == ("ordering", "opus55"):
                low = next(c for c in chains if c["chain"] == "chain_0")["effort"][LOW_VALIDITY_B]
                assert abs(low["pair_valid_share"] - 0.54) < 0.005, low["pair_valid_share"]
                ax.annotate(f"round {LOW_VALIDITY_B}: one search's template has\nonly {low['pair_valid_share']:.0%} of pairs correct",
                            (int(LOW_VALIDITY_B), means[LOW_VALIDITY_B]), xytext=(54, 0.78), fontsize=11.5,
                            color=v6.MUTED, arrowprops={"arrowstyle": "->", "color": v6.MUTED})
        for restart in (25, 50, 75):
            ax.axvline(restart, color=v6.GRID, lw=1.5, zorder=0)
        ax.axhline(1.0, color=v6.LINE_COLOR, ls="--", lw=1.4)
        ax.text(1, 0.96, "ON median on the strict 1% line", color=v6.LINE_COLOR, fontsize=12, va="top")
        ax.set_title(task.capitalize())
        ax.set_xlabel("feedback rounds before the instruction is frozen")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color=v6.GRID)
    axes[0].set_ylabel("share of the OFF-to-flag-line distance\nclosed by the ON median")
    handles = [Line2D([], [], color=v6.OPUS55, marker="o", lw=2.5, label="Opus 5.5 (effort medium)"),
               Line2D([], [], color=v6.OPUS48, marker="o", lw=2.5, label="GPT-6 Sol"),
               Line2D([], [], color=v6.GRID, lw=3, label="conversation restarted from a summary")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "effort.png", bbox_inches="tight")
    plt.close(fig)


def scan_test_rows(m45: dict) -> None:
    """Two checks on the saved frozen-test rows (both batches): (1) maths ON replies that are correct under the
    correctness-only re-judgement (correctness_only.json) and at or under the 1% line in view, any effort point;
    (2) distinct replies per problem over the 5 sampling seeds, B = 16 switch arms (the shared-seed symptom)."""
    for task in TASKS:
        under_correct_only, distinct = 0, []
        for batch in ("c", "d"):
            root = M45 / "runs" / batch / TASK_ROOT[task]
            line = m45[batch][task]["line"]
            flags = json.loads((root / "correctness_only.json").read_text()) if task == "maths" else None
            for path in sorted((root / "test_checked").glob("*.jsonl")):
                if path.stem in ("plain", "neutral"):
                    continue
                replies: dict[tuple[str, str], set[str]] = {}
                for raw in path.open():
                    r = json.loads(raw)
                    key = r["key"]
                    if "parity" in key:
                        continue
                    if flags is not None and key.endswith("#on") and r["peaks_visible"][PROBE[task]] <= line:
                        under_correct_only += flags[f"{key}|{r['task']['id']}|{r['seed']}"]
                    if "@B16#" in key:
                        replies.setdefault((key, r["task"]["id"]), set()).add(r["reply"])
                distinct += [len(v) for v in replies.values()]
        extra = f"maths ON replies correct under correctness-only and under the line (any B): {under_correct_only}; " if task == "maths" else ""
        print(f"[{task}] {extra}distinct replies per problem over 5 seeds at B = 16: mean {np.mean(distinct):.2f}, "
              f"median {np.median(distinct):.0f}, share with 1 distinct reply {np.mean([d == 1 for d in distinct]):.2f} "
              f"({len(distinct)} arm-problems)")


def rows_from_chains(chains: list[dict], key: str, b: str = HEADLINE_B) -> list[dict]:
    by: dict[str, dict] = {}
    for c in chains:
        v = c["effort"][b].get(key)
        assert v is not None, (c["writer"], c["chain"], key)
        r = by.setdefault(c["writer"], {"writer": c["writer"], "eci": c["eci"], "lab": c["lab"], "values": []})
        r["values"].append(v)
    for r in by.values():
        r["mean"] = float(np.mean(r["values"]))
    return sorted(by.values(), key=lambda r: r["eci"])


def lab_fixed_effects_slope(rows: list[dict]) -> float:
    labs = sorted({r["lab"] for r in rows})
    x = np.array([[r["eci"]] + [float(r["lab"] == lab) for lab in labs] for r in rows])
    coef, *_ = np.linalg.lstsq(x, np.array([r["mean"] for r in rows]), rcond=None)
    return float(coef[0]) * 10


def spearman_with_p(rows: list[dict], perms: int = 10_000) -> tuple[float, float]:
    from scipy import stats
    eci, means = np.array([r["eci"] for r in rows]), np.array([r["mean"] for r in rows])
    rho = float(stats.spearmanr(eci, means).statistic)
    rng = np.random.default_rng(45)
    null = [stats.spearmanr(eci, rng.permutation(means)).statistic for _ in range(perms)]
    return rho, float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def fmt_fit(rows: list[dict]) -> str:
    slope, (lo, hi) = two_stage_slope(rows)
    return f"{slope:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def seed_problem_slope_ci(m45: dict, task: str, draws: int = 2000) -> list[float]:
    """As m45's offref_seed_bootstrap, over all 14 writers: one crossed draw of the 5 seeds and 100 problems shared by
    every arm, then writers and chains (two-stage); S_offref per arm from its resampled correct pairs."""
    import warnings
    chains = chains_of(m45, task)
    mats = np.array([c["effort"][HEADLINE_B]["offon_matrix"] for c in chains], float)  # chains x seeds x problems x 3
    valid, off, on = mats[..., 0] > 0, np.where(mats[..., 0] > 0, mats[..., 1], np.nan), np.where(mats[..., 0] > 0, mats[..., 2], np.nan)
    line = np.array([m45[c["batch"]][task]["line"] for c in chains])
    names = sorted({c["writer"] for c in chains}, key=lambda w: next(c["eci"] for c in chains if c["writer"] == w))
    eci = np.array([next(c["eci"] for c in chains if c["writer"] == w) for w in names])
    by = {w: [i for i, c in enumerate(chains) if c["writer"] == w] for w in names}
    rng = np.random.default_rng(4545)
    slopes = []
    for _ in range(draws):
        sd, pr = rng.integers(0, 5, 5), rng.integers(0, 100, 100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_off = np.nanmedian(off[:, sd][:, :, pr].reshape(len(chains), -1), axis=1)
            m_on = np.nanmedian(on[:, sd][:, :, pr].reshape(len(chains), -1), axis=1)
        s_arm = np.where(np.isnan(m_off), 0.0, (m_off - m_on) / (m_off - line))
        pick = rng.choice(len(names), len(names))
        if np.ptp(eci[pick]) == 0:
            continue
        means = np.array([np.mean(s_arm[rng.choice(by[names[i]], len(by[names[i]]))]) for i in pick])
        slopes.append(ols_slope_per_10(eci[pick], means))
    return [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))]


def all14_robustness(m45: dict) -> None:
    """The post's headline is the fit over all 14 writers; every robustness row the appendix reports is recomputed here
    on all 14 (m45's own tables give the 11-writer versions)."""
    keys = [("switch score (headline)", "S_offref"), ("mean of per-pair ratios", "S_offref_mean"),
            ("floor version", "S_offref_floor"), ("plain-referenced, correct pairs", "S_toggle_valid"),
            ("plain-referenced, wrong = no movement", "S_toggle"), ("sibling probe", "secondary_S_offref"),
            ("raw drop (log-odds)", "drop_valid_logodds"), ("wording, both after ON", "offref_reply_on_context"),
            ("wording, both after OFF", "offref_reply_off_context"), ("switch word only, ON answers", "offref_context_on_reply"),
            ("switch word only, OFF answers", "offref_context_off_reply"), ("clean-OFF pairs (plain-referenced)", "S_toggle_valid_clean_off"),
            ("compliance: ON follows fully", "on_follows_fully"), ("compliance: OFF ignores", "off_ignores")]
    for task in TASKS:
        chains = chains_of(m45, task)
        out = []
        for name, key in keys:
            out.append(f"{name} {fmt_fit(rows_from_chains(chains, key))}")
        if task == "maths":
            out.append(f"correctness-only {fmt_fit(rows_from_chains(chains, 'S_offref_correct_only'))}")
        out.append(f"B = 0 {fmt_fit(rows_from_chains(chains, 'S_offref', b='0'))}")
        out.append(f"earlier design's measure of the ON version (instruction removed) {fmt_fit(rows_from_chains(chains, 'm36_S'))}")
        out.append(f"leakage minus plain {fmt_fit(rows_from_chains(chains, 'off_leakage_minus_plain'))}")
        out.append(f"instruction removed, plain-referenced correct pairs {fmt_fit(rows_from_chains(chains, 'S_toggle_valid_removed'))}")
        lo, hi = seed_problem_slope_ci(m45, task)
        out.append(f"also resampling seeds and problems [{lo:+.3f}, {hi:+.3f}]")
        rows = rows_from_chains(chains, "S_offref")
        for drop in ("opus55", "fable51", "llama4maverick", "gpt5nano"):
            out.append(f"without {SHORT[drop]} {fmt_fit([r for r in rows if r['writer'] != drop])}")
        out.append(f"without Anthropic (9 writers) {fmt_fit([r for r in rows if r['lab'] != 'Anthropic'])}")
        out.append(f"one intercept per lab {lab_fixed_effects_slope(rows):+.3f}")
        rho, perm_p = spearman_with_p(rows)
        out.append(f"Spearman {rho:.2f} (permutation p {perm_p:.4f})")
        print(f"[{task}] ALL 14 WRITERS: " + "; ".join(out))
    m50 = json.loads((M50 / "analysis_ordering.json").read_text())
    key = f"on_under_share|{PRIMARY}"  # m50's "share of answers under the line" (of all ON attempts)
    rows = rows_from_chains(m50["chains"], key)
    pre = [r for r in rows if r["writer"] not in ADDED_LATER]
    print(f"[ordering] share of ON attempts correct and under the everyday-chat line, slope per 10 ECI: 11 writers {fmt_fit(pre)}, "
          f"all 14 {fmt_fit(rows)}, all 14 without Anthropic {fmt_fit([r for r in rows if r['lab'] != 'Anthropic'])}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    m45 = load_m45()
    fits = check_and_print_trends(m45)
    all14_robustness(m45)
    fig_trend(fits)
    fig_crossings(m45)
    print_distance_to_line(m45)
    fig_strips(m45)
    print_broad_shares(m45)
    fig_mechanism(m45)
    fig_effort(m45)
    scan_test_rows(m45)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
