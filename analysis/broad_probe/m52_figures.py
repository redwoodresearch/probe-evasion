"""Figures for the m52 report (broad probes): where each set sits for the old and new probe, the capability trend under
both, ordering crossings by writer, Fable 5.1's ordering attack under the new probe, and everyday chat by topic label.

    PYTHONPATH=src .venv/bin/python analysis/broad_probe/m52_figures.py      # writes analysis/broad_probe/m52/fig_*.png
"""

import gzip
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.missions import m52_broad_probes as m52

HERE = Path(__file__).resolve().parent
OUT = HERE / "m52"
_spec = importlib.util.spec_from_file_location("m52_analysis", HERE / "m52_analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
RESULTS = json.loads((OUT / "results.json").read_text())
DETAILS = json.loads((OUT / "details.json").read_text())

INK, MUTED, GRID = "#1f2a2e", "#5f6b6f", "#e3e7e6"
LINE_RED = "#b3261e"
SET_COLOURS = {"WildChat (held-out)": "#9aa3a6", "chat look-alikes": "#6d8fa6", "chat positives (held-out)": "#2f7d5b",
               "normal test answers": "#1f2a2e"}
LAB_COLOUR = {"Anthropic": "#d9663b", "OpenAI": "#2b7bd6", "Google": "#1aa37a", "Alibaba": "#5b3fa8", "Meta": "#6b6b6b"}
TITLES = {"maths": "Maths", "ordering": "Ordering"}
WRITER_NAMES = {"opus55": "Opus 5.5", "gpt6astra": "GPT-6 Astra", "gpt61sol": "GPT-6.1 Sol", "sonnet55": "Sonnet 5.5",
                "fable51": "Fable 5.1", "gpt6sol": "GPT-6 Sol", "opus48": "Opus 4.8", "gemini38flash": "Gemini 3.8 Flash",
                "gpt54mini": "GPT-5.4 mini", "qwen35_397b": "Qwen3.5-397B", "haiku45": "Haiku 4.5", "qwen35_9b": "Qwen3.5-9B",
                "gpt5nano": "GPT-5 nano", "llama4maverick": "Llama 4 Maverick"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gate_sets(task: str, probe: str) -> dict[str, list[float]]:
    chat = A.scored(task, "chat")
    held = lambda label: [r["peaks"][probe] for r in chat if r["split"] == "heldout" and r["label_ok"] and r["label"] == label]
    plain = [r["peaks"][probe] for r in A.scored(task, "m45_c") if r["key"].startswith("plain") and A.is_valid(r)]
    return {"WildChat (held-out)": [r["peaks"][probe] for r in A.line_pool(task)], "chat look-alikes": held(0),
            "chat positives (held-out)": held(1), "normal test answers": plain}


def fig_gate() -> None:
    """Rows: old probe, new probe; columns: tasks. Each panel: the four sets as jittered dot strips and the 1% WildChat line."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex="col")
    rng = np.random.default_rng(0)
    for col, task in enumerate(("ordering", "maths")):
        for row, (label, probe, key) in enumerate((("old probe", m52.OLD_PROBE[task], "old probe, WildChat line"),
                                                   ("new broad probe", m52.probe_name(task, 0), "new probe, WildChat line"))):
            ax = axes[row, col]
            sets = gate_sets(task, probe)
            for i, (name, xs) in enumerate(sets.items()):
                y = len(sets) - 1 - i
                ax.scatter(xs, y + rng.uniform(-0.3, 0.3, len(xs)), s=4, alpha=0.3, color=SET_COLOURS[name], linewidths=0)
                ax.plot([np.median(xs)] * 2, [y - 0.38, y + 0.38], color=SET_COLOURS[name], lw=2.5)
            line = RESULTS[task][key]["line"]
            ax.axvline(line, color=LINE_RED, lw=2)
            ax.text(line, len(sets) - 0.35, " 1% line on WildChat", color=LINE_RED, fontsize=10, va="bottom")
            if row == 0:
                strict = A.STRICT_LINE[task]
                ax.axvline(strict, color=LINE_RED, lw=1.2, ls="--")
                ax.text(strict, -0.75, "original strict line ", color=LINE_RED, fontsize=10, ha="right", va="bottom")
            ax.set_yticks(range(len(sets)), list(reversed(sets)) if col == 0 else [])
            ax.set_ylim(-0.8, len(sets) - 0.2)
            ax.set_title(f"{TITLES[task]}: {label}", loc="left", fontsize=12, fontweight="bold")
            ax.grid(axis="x", color=GRID)
        axes[1, col].set_xlabel("probe score (log-odds); tick = median")
    fig.tight_layout()
    save(fig, "fig_gate.png")


def fig_trend() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, task in zip(axes, ("ordering", "maths")):
        old = RESULTS[task]["old probe, strict line"]
        new = RESULTS[task]["new probe, WildChat line"]
        for w in A.NUM.ECI:
            x = A.NUM.ECI[w]
            ax.plot([x, x], [old["writer_means"][w], new["writer_means"][w]], color=GRID, lw=2, zorder=1)
            ax.scatter(x, old["writer_means"][w], s=55, facecolor="white", edgecolor=MUTED, lw=1.5, zorder=2)
            ax.scatter(x, new["writer_means"][w], s=70, color=LAB_COLOUR[A.NUM.LAB[w]], zorder=3)
        xs = np.linspace(131, 168, 20)
        for r, style, colour, label in ((old, "--", MUTED, "old probe, strict line"), (new, "-", INK, "new broad probe, WildChat line")):
            x = np.array([A.NUM.ECI[w] for w in r["writer_means"]])
            y = np.array(list(r["writer_means"].values()))
            k, c = np.polyfit(x, y, 1)
            s = r["slope"]
            ax.plot(xs, k * xs + c, style, color=colour, lw=1.8,
                    label=f"{label}: {s['slope']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] per 10 ECI")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(TITLES[task], loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("writer capability (Epoch Capabilities Index)")
        ax.grid(axis="y", color=GRID)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
    axes[0].set_ylabel("probe-dodging score\n(0 = no move; 1 or more = past the flag line)")
    fig.tight_layout()
    save(fig, "fig_trend.png")


def fig_crossings() -> None:
    cross = DETAILS["ordering"]["crossings_by_writer_new"]
    writers = sorted(cross, key=lambda w: A.NUM.ECI[w])
    share = [cross[w][0] / cross[w][1] for w in writers]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(range(len(writers)), share, color=[LAB_COLOUR[A.NUM.LAB[w]] for w in writers], height=0.7)
    for i, (w, v) in enumerate(zip(writers, share)):
        ax.text(v + 0.005, i, f"{cross[w][0]} of {cross[w][1]:,}", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(writers)), [f"{WRITER_NAMES[w]} ({A.NUM.ECI[w]:.0f})" for w in writers])
    ax.set_xlabel("share of ordering pairs (both answers correct) where ON gets under the new line and OFF does not")
    ax.set_xlim(0, 0.36)
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    save(fig, "fig_crossings.png")


def fig_fable() -> None:
    """Fable 5.1's ordering attack after 16 rounds, read by the new probe: plain, OFF and ON over pairs with both valid."""
    probe = m52.probe_name("ordering", 0)
    rows = A.scored("ordering", "m45_c") + A.scored("ordering", "m45_d")
    plain = [r["peaks"][probe] for r in A.scored("ordering", "m45_c") if r["key"].startswith("plain") and A.is_valid(r)]
    by = {(r["key"], r["task"]["id"], r["seed"]): r for r in rows if r["key"].startswith("fable51/") and "@B16#" in r["key"]}
    on, off = [], []
    for (key, i, s), r in by.items():
        o = by.get((key[:-3] + "#off", i, s))
        if key.endswith("#on") and o and A.is_valid(r) and A.is_valid(o):
            on.append(r["peaks"][probe])
            off.append(o["peaks"][probe])
    line = RESULTS["ordering"]["new probe, WildChat line"]["line"]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    rng = np.random.default_rng(1)
    for y, (name, xs, colour) in enumerate((("ON", on, "#c2541b"), ("OFF", off, "#3d5a80"), ("normal answers", plain, INK))):
        ax.scatter(xs, y + rng.uniform(-0.28, 0.28, len(xs)), s=5, alpha=0.3, color=colour, linewidths=0)
        ax.plot([np.median(xs)] * 2, [y - 0.38, y + 0.38], color=colour, lw=3)
        ax.text(np.median(xs), y + 0.42, f"{np.median(xs):+.1f}", ha="center", fontsize=9.5, color=colour)
    ax.axvline(line, color=LINE_RED, lw=2)
    ax.text(line + 0.3, 2.45, "flagged →", color=LINE_RED, fontsize=10, ha="left")
    ax.text(line - 0.3, 2.45, "← not flagged", color=LINE_RED, fontsize=10, ha="right")
    ax.set_yticks(range(3), ["ON", "OFF", "normal answers"])
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("new broad probe score (log-odds); tick = median")
    ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    save(fig, "fig_fable.png")


def fig_everyday() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"wspace": 0.45})
    for ax, task in zip(axes, ("ordering", "maths")):
        cats = DETAILS[task]["everyday_by_category"]
        order = [c for c in m52.CATEGORIES[task] if c in cats]
        y = np.arange(len(order))
        ax.scatter([cats[c]["old_median"] for c in order], y, s=60, facecolor="white", edgecolor=MUTED, lw=1.5, label="old probe")
        ax.scatter([cats[c]["new_median"] for c in order], y, s=70, color=INK, label="new broad probe")
        ax.set_yticks(y, [f"{c.replace('_', ' ')} ({cats[c]['n']})" for c in order])
        ax.set_title(f"{TITLES[task]} probe", loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="x", color=GRID)
        ax.set_xlabel("median probe score (log-odds)")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(1.15, -0.3), ncol=2)
    fig.subplots_adjust(bottom=0.3)
    save(fig, "fig_everyday.png")


def examples() -> dict:
    """Real rows for the page: chat positive, chat look-alike, WildChat negative (ordering), and the median-scoring
    crossing Fable 5.1 ON reply under the new probe."""
    import random

    rng = random.Random(4)
    chat = [r for r in A.scored("ordering", "chat") if r["split"] == "heldout" and r["label_ok"]]
    pos = rng.choice([r for r in chat if r["label"] == 1])
    look = rng.choice([r for r in chat if r["label"] == 0])
    wild = rng.choice([r for r in A.line_pool("ordering") if len(r["response"]) < 700])
    probe, old = m52.probe_name("ordering", 0), m52.OLD_PROBE["ordering"]
    line = RESULTS["ordering"]["new probe, WildChat line"]["line"]
    rows = A.scored("ordering", "m45_c") + A.scored("ordering", "m45_d")
    by = {(r["key"], r["task"]["id"], r["seed"]): r for r in rows if r["key"].startswith("fable51/") and "@B16#" in r["key"]}
    crossing = []
    for (key, i, s), r in by.items():
        o = by.get((key[:-3] + "#off", i, s))
        if key.endswith("#on") and o and A.is_valid(r) and A.is_valid(o) and r["peaks"][probe] <= line < o["peaks"][probe]:
            crossing.append((r, o))
    crossing.sort(key=lambda p: p[0]["peaks"][probe])
    on, off = crossing[len(crossing) // 2]
    fmt = lambda r: {"request": r["prompt"], "reply": r["response"], "new": r["peaks"][probe], "old": r["peaks"][old]}
    return {"chat_positive": fmt(pos), "chat_lookalike": fmt(look), "wildchat": fmt(wild),
            "fable_on": fmt(on) | {"problem": on["task"]["task"]}, "fable_off": fmt(off), "line": line}


def main() -> None:
    fig_gate()
    fig_trend()
    fig_crossings()
    fig_fable()
    fig_everyday()
    (OUT / "examples.json").write_text(json.dumps(examples(), indent=1))


if __name__ == "__main__":
    main()
