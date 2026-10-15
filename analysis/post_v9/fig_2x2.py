"""Main figures for the post draft: one panel per probe (maths, ordering, belief tracking, contradiction), writer
capability on x, gap closed (plain-referenced) on y, 14 writers. Two figures: after 16 rounds of feedback, and at
round 0 (before any feedback). Values and slopes from the independent recomputation (numbers/numbers.json).

    .venv/bin/python analysis/post_v9/fig_2x2.py      # writes figures/trend_2x2_b16.png and figures/trend_2x2_b0.png

Also writes figures/how_to_read.png: one worked example (Fable 5.1 on ordering) showing what the score measures.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("post_v8_figures", HERE.parent / "post_v8" / "make_figures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
_spec_dots = importlib.util.spec_from_file_location("make_dots", F.REPO / "analysis/for_james/methodology/make_dots.py")
DOTS = importlib.util.module_from_spec(_spec_dots)
_spec_dots.loader.exec_module(DOTS)

NUMBERS = json.loads((HERE / "numbers" / "numbers.json").read_text())["tasks"]
PANELS = {"maths": "Arithmetic", "ordering": "Ordering", "false_belief": "Belief tracking", "contradiction": "Spotting a contradiction"}
ROUNDS = {"B16": "after 16 rounds of feedback", "B0": "round 0, before any feedback"}
FIT_XS = np.linspace(131, 168, 50)
KEY_WRITERS = {"fable51": "Fable 5.1", "gpt6astra": "GPT-6 Astra", "gpt61sol": "GPT-6.1 Sol", "opus55": "Opus 5.5"}
# One low and one mid capability writer, labelled next to their round-16 dot rather than in the right-hand column.
# Llama 4 Maverick (leftmost, nothing above it) is named just above its range bar, with no leader line: vertical leader
# lines read as more range bars. Gemini 3.8 Flash sits 1.5 ECI from Opus 4.8, so its name goes up and to the left in
# empty space, with a short diagonal leader to the dot.
LLAMA, GEMINI = "llama4maverick", "gemini38flash"
ABOVE_BAR = 0.025
GEMINI_LABEL_OFFSET = (-3.5, 0.2)  # (ECI units, score units) from the dot to the text's bottom-right corner
LABEL_X = 170.5        # labels sit in a column right of the data, with leader lines to their dots
LABEL_GAP = 0.06       # minimum vertical gap between labels (score units)
Y_LIMITS = (-0.15, 0.82)
Y_LABEL = "probe-evasion score"
Y_LABEL_NOTE = "(0 = no move; 1 = moved the full distance\nfrom a normal answer to the threshold)"  # smaller, beside the main label
EXAMPLE_TASK = "ordering"  # the worked example: Fable 5.1 (make_dots' writer for ordering), medians over correct pairs
EXAMPLE_COLOURS = {"plain": "#3b4440", "off": "#3d5a80", "on": "#c2541b", "line": "#b3261e"}
# Uncertainty on the round-16 fit, two ways to show it: "slopes" writes each round's slope and interval in the panel,
# "band" shades a 95% band around the round-16 fit. The band uses numbers/compute.py's two-stage bootstrap (writers,
# then each writer's searches) so it agrees with the quoted intervals.
N_BOOT = 10_000
# Band variants: (fit line colour, band colour, band alpha) per round. Gold is the one hue no lab uses.
GOLD_LINE, GOLD_BAND = "#a8790f", "#e3b23c"
BAND_STYLES = {
    "band": {"B16": (None, None, 0.12)},
    "bands": {"B16": (None, None, 0.12), "B0": (None, None, 0.06)},
    "bands_gold0": {"B16": (None, None, 0.13), "B0": (GOLD_LINE, GOLD_BAND, 0.22)},
    "bands_gold16": {"B16": (GOLD_LINE, GOLD_BAND, 0.22), "B0": (None, None, 0.10)},
}
BAND_SEED = 20261013


def fmt(fit: dict) -> str:
    lo, hi = fit["ci"]
    return f"{fit['slope']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def spread(ys: list[float]) -> list[float]:
    """Push label positions apart (keeping their order) so no two are closer than LABEL_GAP."""
    order = np.argsort(ys)
    placed = list(np.array(ys, dtype=float)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + LABEL_GAP)
    out = np.empty(len(ys))
    out[order] = placed
    return list(out)


def xpos(w: str, meta: dict) -> float:
    return meta[w]["eci"] + F.DISPLAY_SHIFT.get(w, 0.0)


ROUND0_BAR_SHIFT = 0.0  # ECI units; 0 = round-0 bars run through their own dot


def plot_round(ax, t: dict, meta: dict, hollow: bool, bars: bool, fit_style: str, fit_colour: str | None = None) -> None:
    means, chains = t["writer_means"], t["writer_chains"]
    for w, m in means.items():
        x, colour = xpos(w, meta), F.LAB_COLOUR[meta[w]["lab"]]
        if bars and not hollow:
            ax.vlines(x, min(chains[w]), max(chains[w]), color=colour, lw=1.8, alpha=0.5)
        elif bars:  # round 0: thinner and dashed, so the two rounds' bars stay distinguishable
            ax.vlines(x + ROUND0_BAR_SHIFT, min(chains[w]), max(chains[w]), color=colour, lw=1.2, alpha=0.6, linestyles=(0, (2, 1.5)))
        ax.plot(x, m, "o", ms=9 if hollow else 10, mfc="white" if hollow else colour, mec=colour, mew=1.8, zorder=3)
    k, c0 = np.polyfit([meta[w]["eci"] for w in means], list(means.values()), 1)
    ax.plot(FIT_XS, k * FIT_XS + c0, fit_style, color=fit_colour or F.v6.TEXT, lw=2.2 if fit_colour else 1.8)


def label_writers(ax, means: dict, chains: dict, meta: dict) -> None:
    keys = list(KEY_WRITERS)
    for w, y in zip(keys, spread([means[w] for w in keys])):
        ax.plot([xpos(w, meta) + 0.3, LABEL_X - 0.2], [means[w], y], color=F.v6.MUTED, lw=0.7, zorder=2)
        ax.text(LABEL_X, y, KEY_WRITERS[w], va="center", fontsize=11, color=F.v6.TEXT)
    ax.text(xpos(LLAMA, meta) + 0.6, max(chains[LLAMA]) + ABOVE_BAR, "Llama 4\nMaverick", ha="center", va="bottom",
            fontsize=10.5, color=F.v6.TEXT, linespacing=1.1)
    x, y = xpos(GEMINI, meta), means[GEMINI]
    ax.annotate("Gemini 3.8\nFlash", (x, y), xytext=(x + GEMINI_LABEL_OFFSET[0], y + GEMINI_LABEL_OFFSET[1]), ha="right",
                va="bottom", fontsize=10.5, color=F.v6.TEXT, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=F.v6.MUTED, lw=0.7, relpos=(1, 0), shrinkA=1, shrinkB=6))


def fit_band(chains: dict[str, list[float]], meta: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BAND_SEED)
    names = list(chains)
    eci = np.array([meta[w]["eci"] for w in names])
    lines = np.empty((N_BOOT, len(FIT_XS)))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(names), len(names))
        ys = [np.mean(rng.choice(chains[names[i]], len(chains[names[i]]))) for i in pick]
        k, c0 = np.polyfit(eci[pick], ys, 1)
        lines[b] = k * FIT_XS + c0
    return np.quantile(lines, 0.025, axis=0), np.quantile(lines, 0.975, axis=0)


def slope_text(ax, data: dict, rounds: tuple[str, ...]) -> None:
    names = {"B16": "16 rounds", "B0": "round 0  "}
    rows = [f"{names[b]}  {fmt(data[b]['all14'])}" for b in rounds]
    ax.text(0.02, 0.97, "\n".join(rows), transform=ax.transAxes, va="top", ha="left",
            fontsize=11.5, color=F.v6.TEXT, linespacing=1.4, family="DejaVu Sans Mono")


def panel(ax, task: str, title: str, rounds: tuple[str, ...], meta: dict, uncertainty: str | None = None,
          round0_bars: bool = False) -> None:
    """rounds[0] is drawn solid with range bars and labels; a second round, if given, as hollow dots with a dashed fit."""
    data = NUMBERS[task]
    styles = BAND_STYLES.get(uncertainty, {})
    for b in rounds:
        if b not in styles:
            continue
        _, band_colour, alpha = styles[b]
        lo, hi = fit_band(data[b]["writer_chains"], meta)
        ax.fill_between(FIT_XS, lo, hi, color=band_colour or F.v6.TEXT, alpha=alpha, lw=0, zorder=1)
        if uncertainty == "bands" and b == "B0":  # both grey: dotted edges tell round 0's band apart
            for edge in (lo, hi):
                ax.plot(FIT_XS, edge, ":", color=F.v6.MUTED, lw=0.9, zorder=1)
    if uncertainty == "slopes":
        slope_text(ax, data, rounds)
    plot_round(ax, data[rounds[0]], meta, hollow=False, bars=True, fit_style="-", fit_colour=styles.get(rounds[0], (None,))[0])
    if len(rounds) == 2:
        plot_round(ax, data[rounds[1]], meta, hollow=True, bars=round0_bars, fit_style="--", fit_colour=styles.get(rounds[1], (None,))[0])
        ax.set_title(title, fontsize=16)  # slopes go in the caption
    else:
        ax.set_title(f"{title}: {fmt(data[rounds[0]]['all14'])}", fontsize=13)
    label_writers(ax, data[rounds[0]]["writer_means"], data[rounds[0]]["writer_chains"], meta)
    ax.axhline(0, color=F.v6.MUTED, lw=0.8)
    ax.grid(axis="y", color=F.v6.GRID)
    ax.set_xlim(130, 179)
    ax.set_xticks(range(130, 171, 10))
    ax.set_ylim(*Y_LIMITS)


def example_medians() -> dict[str, float]:
    """The worked example (Fable 5.1 on ordering, correct answers): medians of plain, OFF and ON, and the flag line."""
    cfg = DOTS.TASKS[EXAMPLE_TASK]
    med = {label.split("\n")[0]: float(np.median(xs)) for label, xs, _ in DOTS.strips(cfg)}
    return {"plain": med["plain correct answers"], "off": med["OFF replies"], "on": med["ON replies"], "line": DOTS.flag_line(cfg)}


def draw_score(ax, fontsize: float = 13, dot_size: float = 300, line_label: bool = True, positive_label: str = "normal answer") -> None:
    """Three dots (normal answer, attack OFF, attack ON), the flag line and the two distances whose ratio is the score."""
    m, c = example_medians(), EXAMPLE_COLOURS
    plain, off, on, line = m["plain"], m["off"], m["on"], m["line"]
    ax.plot([line, line], [-0.75, 1.0], color=c["line"], lw=2.5)
    if line_label:
        ax.text(line, 1.1, "threshold", ha="center", fontsize=fontsize, color=c["line"])
    for x, colour, label in ((plain, c["plain"], positive_label), (off, c["off"], "attack OFF"), (on, c["on"], "attack ON")):
        ax.scatter(x, 0.4, s=dot_size, color=colour, zorder=3)
        ax.text(x, 0.7, label, ha="center", fontsize=fontsize, color=colour)
    ax.annotate("", xy=(on + 0.3, 0.4), xytext=(off - 0.3, 0.4), arrowprops=dict(arrowstyle="->", color=c["on"], lw=2.5))
    ax.text((on + off) / 2, 0.12, f"{off - on:.1f}", ha="center", fontsize=fontsize, color=c["on"])
    ax.annotate("", xy=(line, -0.35), xytext=(plain, -0.35), arrowprops=dict(arrowstyle="<->", color=c["plain"], lw=1.2))
    ax.text((line + plain) / 2, -0.65, f"{plain - line:.1f}", ha="center", fontsize=fontsize, color=c["plain"])
    ax.set_xlim(line - 1.5, plain + 1.5)
    ax.set_ylim(-0.8, 1.3)
    ax.axis("off")


def how_to_read() -> None:
    """Figure 1: what the score measures. The words go in the post's caption, not the figure."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    draw_score(ax, positive_label="held-out positives")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "how_to_read.png", bbox_inches="tight")
    plt.close(fig)


ROUND_KEY_LABELS = {"top": ("after 16 rounds of score feedback", "round 0, before any feedback"),
                    "side": ("after 16 rounds\nof score feedback", "round 0,\nbefore any\nfeedback")}


def round_key(fig, uncertainty: str | None, where: str) -> None:
    """The round 16 / round 0 key on its own, larger, at the top or the right (James: it was lost among the labs)."""
    styles = BAND_STYLES.get(uncertainty, {})
    entries = []
    for b, hollow, ls in (("B16", False, "-"), ("B0", True, "--")):
        line_colour, band_colour, alpha = styles.get(b, (None, None, 0.0))
        line = Line2D([], [], marker="o", ms=10, color=line_colour or F.v6.TEXT, mfc="white" if hollow else F.v6.TEXT,
                      mec=F.v6.TEXT, mew=1.8, ls=ls, lw=2.2)
        entries.append((Patch(color=band_colour or F.v6.TEXT, alpha=alpha, lw=0), line) if alpha else line)
    kw = dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, frameon=False, fontsize=15, handlelength=3.2)
    if where == "top":
        fig.legend(entries, ROUND_KEY_LABELS["top"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.95), columnspacing=3, **kw)
    else:
        fig.legend(entries, ROUND_KEY_LABELS["side"], loc="center left", ncol=1, bbox_to_anchor=(0.975, 0.5), labelspacing=2.2, **kw)


def figure(rounds: tuple[str, ...], meta: dict, name: str, suptitle: str | None = None, uncertainty: str | None = None,
           round0_bars: bool = False, key: str = "bottom") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5 if len(rounds) == 2 else 11), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.2, "wspace": 0.08})
    for ax, (task, title) in zip(axes.flat, PANELS.items()):
        panel(ax, task, title, rounds, meta, uncertainty, round0_bars)
    for ax in axes[-1]:
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL, fontsize=15, labelpad=40)
        ax.text(-0.095, 0.5, Y_LABEL_NOTE, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=11, linespacing=1.3,
                color=F.v6.MUTED)
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, x=0.02, ha="left")
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    if len(rounds) == 2 and key != "bottom":
        round_key(fig, uncertainty, key)
    elif len(rounds) == 2:
        styles = BAND_STYLES.get(uncertainty, {})
        c16, c0 = (styles.get(b, (None,))[0] or F.v6.TEXT for b in ("B16", "B0"))
        handles += [Line2D([], [], marker="o", color=c16, mfc=F.v6.TEXT, mec=F.v6.TEXT, ls="-", ms=8, label="after 16 rounds"),
                    Line2D([], [], marker="o", color=c0, mfc="white", mec=F.v6.TEXT, ls="--", ms=8, label="round 0")]
    else:
        handles.append(Line2D([], [], color=F.v6.TEXT, ls="-", label="least-squares fit"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    bars = "bars = range over 3 searches (after 16 rounds)" if len(rounds) == 2 else "bars = range over 3 searches"
    if round0_bars:
        bars = "bars = range over 3 searches (dashed: round 0)"
    slopes = "" if len(rounds) == 2 and uncertainty is None else "Slopes per 10 ECI [95% interval, resampling writers then searches]; "
    if uncertainty == "slopes":
        slopes = "Top left: fitted slope per 10 ECI [95% interval, resampling writers then searches]; "
    if uncertainty == "band":
        slopes = "Shaded: 95% band on the 16-round fit (resampling writers then searches); "
    if uncertainty == "bands":
        slopes = "Shaded: 95% bands on each fit (dotted edges = round 0; resampling writers then searches); "
    if uncertainty in ("bands_gold0", "bands_gold16"):
        slopes = "Shaded: 95% band on each fit (resampling writers then searches); "
    fig.text(0.5, 0.06, f"{slopes}{bars}; threshold at a 1% false-positive rate on hard negatives",
             ha="center", fontsize=11, color=F.v6.MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9 if suptitle else 0.925 if key == "top" else 0.95, bottom=0.13)
    fig.savefig(HERE / "figures" / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = {r["writer"]: r for r in F.writer_rows(F.load_m45(), "maths")}  # ECI and lab per writer
    for b, when in ROUNDS.items():
        figure((b,), meta, f"trend_2x2_{b.lower()}.png", f"Qwen3.5-27B, 14 writers, {when}")
    figure(("B16", "B0"), meta, "trend_2x2_combined.png")
    for style in ("slopes", *BAND_STYLES):
        figure(("B16", "B0"), meta, f"trend_2x2_combined_{style}.png", uncertainty=style)
    figure(("B16", "B0"), meta, "trend_2x2_combined_bands_gold0_bars.png", uncertainty="bands_gold0", round0_bars=True)
    for key in ("top", "side"):
        figure(("B16", "B0"), meta, f"trend_2x2_combined_bands_gold0_key_{key}.png", uncertainty="bands_gold0", round0_bars=True, key=key)
    how_to_read()


if __name__ == "__main__":
    (HERE / "figures").mkdir(exist_ok=True)
    main()
