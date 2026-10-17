"""Probe vs TF-IDF, attack by attack: for each task, one pair of bars per round-16 prompt (14 writers x 3 searches),
how far it moves the probe and how far it moves the TF-IDF word classifier, each on its own scale (median OFF - ON drop
over pairs with both answers correct, divided by the median unattacked score minus the median hard-negative score;
numbers/tfidf_trend.py, "separation"). Prompts sorted by the probe's score. The companion of fig_wording.py's summary.

Also figures/attack_scatter_probe_vs_tfidf.png: the same numbers as a scatter, one dot per prompt (TF-IDF on x, probe on
y, coloured by the writer's lab), with the least-squares line and the dashed line where the two move equally.

    .venv/bin/python analysis/post_v9/fig_attack_pairs.py   # writes figures/attack_pairs_probe_vs_tfidf.png and the scatter
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_spec_v8 = importlib.util.spec_from_file_location("post_v8_figures", HERE.parent / "post_v8" / "make_figures.py")
V8 = importlib.util.module_from_spec(_spec_v8)
_spec_v8.loader.exec_module(V8)  # LAB_COLOUR: the lab colours of every other figure
_spec = importlib.util.spec_from_file_location("fig_method", HERE / "fig_method.py")
METHOD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METHOD)

TREND = json.loads((HERE / "numbers" / "tfidf_trend.json").read_text())
SHARE = json.loads((HERE / "numbers" / "wording_share.json").read_text())
TASKS = {"maths": "Arithmetic", "false_belief": "Belief tracking", "contradiction": "Spotting a contradiction", "ordering": "Ordering"}
PROBE_C, TFIDF_C = "#3b4440", "#5fb3c2"  # as fig_wording.py
SHORT = {"opus55": "Opus 5.5", "gpt6astra": "GPT-6 Astra", "gpt61sol": "GPT-6.1 Sol", "sonnet55": "Sonnet 5.5", "fable51": "Fable 5.1",
         "gpt6sol": "GPT-6 Sol", "opus48": "Opus 4.8", "gemini38flash": "Gemini 3.8 Flash", "gpt54mini": "GPT-5.4 Mini",
         "qwen35_397b": "Qwen3.5-397B", "haiku45": "Haiku 4.5", "qwen35_9b": "Qwen3.5-9B", "gpt5nano": "GPT-5 nano",
         "llama4maverick": "Llama 4 Maverick"}
WIDTH = 0.4


LAB = {"opus55": "Anthropic", "sonnet55": "Anthropic", "fable51": "Anthropic", "opus48": "Anthropic", "haiku45": "Anthropic",
       "gpt6astra": "OpenAI", "gpt61sol": "OpenAI", "gpt6sol": "OpenAI", "gpt54mini": "OpenAI", "gpt5nano": "OpenAI",
       "gemini38flash": "Google", "qwen35_397b": "Alibaba", "qwen35_9b": "Alibaba", "llama4maverick": "Meta"}


def prompts(task: str) -> list[tuple[str, float, float]]:
    """(label, probe score, TF-IDF score) per prompt, sorted by the probe's score, highest first."""
    sep = TREND[task]["separation"]
    rows = [(f"{SHORT[w]} #{i + 1}", p, t) for w in sep["probe"]["writer_chains"]
            for i, (p, t) in enumerate(zip(sep["probe"]["writer_chains"][w], sep["tfidf"]["writer_chains"][w]))]
    return sorted(rows, key=lambda r: -r[1])


def main() -> None:
    fig, axes = plt.subplots(len(TASKS), 1, figsize=(16, 17), gridspec_kw={"hspace": 0.95})
    for ax, (task, title) in zip(axes, TASKS.items()):
        rows = prompts(task)
        x = np.arange(len(rows))
        ax.bar(x - WIDTH / 2, [r[1] for r in rows], WIDTH, color=PROBE_C, label="probe")
        ax.bar(x + WIDTH / 2, [r[2] for r in rows], WIDTH, color=TFIDF_C, label="TF-IDF word classifier")
        ax.axhline(0, color=METHOD.MUTED, lw=0.8)
        ax.set_xticks(x, [r[0] for r in rows], rotation=90, fontsize=8.5)
        ax.set_xlim(-0.7, len(rows) - 0.3)
        ax.grid(axis="y", color="#e6e9e8")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        share = SHARE[task]["share"]
        r = np.corrcoef([v[1] for v in rows], [v[2] for v in rows])[0, 1]
        ax.set_title(f"{title}: TF-IDF-predicted share of the trend {share:.0%}; correlation across prompts r = {r:.2f}",
                     fontsize=13.5, loc="left", color=METHOD.TEXT)
        ax.set_ylabel("evasion score\n(own scale)", fontsize=11)
    axes[0].legend(loc="upper right", frameon=False, fontsize=12, ncol=2)
    fig.text(0.5, 0.0, "One pair of bars per round-16 prompt (14 writers x 3 searches), sorted by the probe's score. Each detector: median "
             "OFF score minus ON score over pairs with both answers correct,\ndivided by its median unattacked score minus its median "
             "hard-negative score (1 = moved as far as a normal answer sits from a typical hard negative).",
             ha="center", va="top", fontsize=10.5, color=METHOD.MUTED)
    fig.savefig(HERE / "figures" / "attack_pairs_probe_vs_tfidf.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scatter() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5), gridspec_kw={"hspace": 0.32, "wspace": 0.22})
    for ax, (task, title) in zip(axes.flat, TASKS.items()):
        sep = TREND[task]["separation"]
        xs, ys = [], []
        for w, probe_chains in sep["probe"]["writer_chains"].items():
            words = sep["tfidf"]["writer_chains"][w]
            ax.scatter(words, probe_chains, s=46, color=V8.LAB_COLOUR[LAB[w]], alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
            xs += words
            ys += probe_chains
        xs, ys = np.array(xs), np.array(ys)
        lo, hi = min(xs.min(), ys.min(), 0) - 0.05, max(xs.max(), ys.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "--", color="#b9c0bd", lw=1, zorder=1)  # equal movement on both detectors' own scales
        k, c0 = np.polyfit(xs, ys, 1)
        ax.plot([lo, hi], [c0 + k * lo, c0 + k * hi], color=METHOD.TEXT, lw=1.6, zorder=2)
        ax.axhline(0, color="#d5dad8", lw=0.8, zorder=0)
        ax.axvline(0, color="#d5dad8", lw=0.8, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        r = np.corrcoef(xs, ys)[0, 1]
        ax.set_title(f"{title}\nr = {r:.2f}; TF-IDF-predicted share of the trend {SHARE[task]['share']:.0%}", fontsize=13,
                     loc="left", color=METHOD.TEXT)
        ax.set_xlabel("TF-IDF word classifier's evasion score", fontsize=11)
        ax.set_ylabel("probe's evasion score", fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    labs = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, label=lab) for lab, c in V8.LAB_COLOUR.items()]
    lines = [plt.Line2D([], [], color=METHOD.TEXT, lw=1.6, label="least-squares fit"),
             plt.Line2D([], [], color="#b9c0bd", ls="--", lw=1, label="equal movement")]
    fig.legend(handles=labs + lines, loc="lower center", ncol=7, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, 0.0, "One dot per round-16 prompt (14 writers x 3 searches). Each detector: median OFF score minus ON score over pairs "
             "with both answers correct,\ndivided by its median unattacked score minus its median hard-negative score.",
             ha="center", va="top", fontsize=10.5, color=METHOD.MUTED)
    fig.savefig(HERE / "figures" / "attack_scatter_probe_vs_tfidf.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
    scatter()
