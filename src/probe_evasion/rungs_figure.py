"""The probe-strength ladder as one figure: what each rung's training data and method are, with a real training
example, and (when available) how often the frozen red-team maths prompt still gets past that rung on Qwen3.5-27B.

    uv run python -m probe_evasion.rungs_figure   # writes share/probe_rungs.png
"""

import json
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from probe_evasion.proposal_figures import BLUE, GREY, INK, ORANGE

DATA = Path("data/concepts")
PROBES = Path("results/concept_probes")
LADDER_RUN = Path("results/gepa/qwen35_maths_window_L48_20261007-173549")  # saved scenic answers, re-scored per rung
LAYER = 48
OWN = "#1d7f52"
RUNGS = [  # name, data source label, source colour, size note, styles, method, probe file, sample source
    ("1  Templated", "written by us", GREY, "2,400 one-line answers\n+ 690 recalled facts", ["1 template per type"],
     "Neural Chameleons:\nper-token, mean,\nAdamW", "qwen35_maths_layers", "templated"),
    ("2  Varied wording", "written by us", GREY, "same problems; half reworded\nin 165 answer formats", ["terse", "casual", "formal", "hedged", "..."],
     "Neural Chameleons", "qwen35_maths_varied", "varied"),
    ("3  Model's own\n    replies", "Qwen's own replies", OWN, "2,555 replies to computation\nand matched lookup/recall tasks",
     ["sentence", "paragraph", "email", "bullets", "casual", "formal", "restate", "+tip"], "Neural Chameleons", "qwen35_maths_onpolicy", "email"),
    ("4  Own replies,\n    lab training", "Qwen's own replies", OWN, "same as rung 3", ["same 8 styles"],
     "Anthropic / DeepMind:\nstandardised, L2\nlogistic regression", "qwen35_maths_onpolicy_lr", "email"),
    ("5  + narrative\n    rewrites", "own replies + rewrites", ORANGE, "rung 3 + ~1,280 replies in\nnarrative styles (attack family)",
     ["8 styles", "vivid scene", "story", "travel diary", "poem"], "Anthropic / DeepMind:\nstandardised, L2\nlogistic regression", "qwen35_maths_rewrites_lr", "narrative"),
]


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def sample(kind: str) -> str:
    rng = random.Random(4)
    if kind == "templated":
        rows = [r for r in jl(DATA / "probe_rows.jsonl") if r["concept"] == "maths" and r["label"] == 1]
    elif kind == "varied":
        base = {r["prompt"]: r["response"] for r in jl(DATA / "probe_rows.jsonl")}
        rows = [r for r in jl(DATA / "probe_rows_maths_varied.jsonl") if r["label"] == 1 and base.get(r["prompt"]) != r["response"]]
    elif kind == "email":
        rows = [r for r in jl(DATA / "onpolicy_qwen35_maths_train.jsonl") if r["label"] == 1 and r["style"] == "email"]
    else:
        rows = [r for r in jl(DATA / "onpolicy_qwen35_maths_narrative.jsonl") if r["label"] == 1 and r["style"] == "vivid scene"]
    if not rows:
        return "(being generated)"
    r = rng.choice(rows)
    reply = " ".join(r["response"].split())
    return textwrap.shorten(reply, 120, placeholder=" ...")


def result(probe_file: str) -> str:
    path = LADDER_RUN / f"ladder_{probe_file}.json"
    if not path.exists():
        return "pending"
    res = json.loads(path.read_text())["results"].get(f"maths@L{LAYER}")
    if not res:
        return "pending"
    k, n = (int(x) for x in res["window"]["missed"]["optimized"].split("/"))
    return f"{100 * k / n:.0f}%\n({k}/{n})"


def main(out: Path = Path("share/probe_rungs.png")) -> Path:
    fig = plt.figure(figsize=(16, 9.2), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.012, 0.965, "The probe-strength ladder: what changes from rung to rung", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.012, 0.93, "Maths-in-the-head probes on Qwen3.5-27B. Every rung is checked on answer styles it never saw "
            "(dialogue, headline, FAQ); nothing scenic is used until rung 5.", fontsize=10.5, color=GREY)
    cols = [(0.012, "Rung"), (0.17, "Training data"), (0.34, "A real training example (computed answer)"), (0.64, "Answer styles covered"),
            (0.80, "Training method"), (0.905, f"Scenic answers\nmissed, layer {LAYER}")]
    for x, h in cols:
        ax.text(x, 0.885, h, fontsize=10, color=GREY, fontweight="bold", va="top")
    row_h = 0.16
    for i, (name, source, colour, size, styles, method, probe_file, kind) in enumerate(RUNGS):
        y = 0.83 - i * row_h
        ax.add_patch(FancyBboxPatch((0.006, y - row_h + 0.018), 0.988, row_h - 0.012, boxstyle="round,pad=0.002,rounding_size=0.01",
                                    transform=ax.transAxes, facecolor="#f7f8fa", edgecolor="#e2e5ea"))
        ax.text(0.016, y - 0.03, name, fontsize=12, fontweight="bold", color=INK, va="top", linespacing=1.3)
        ax.add_patch(FancyBboxPatch((0.17, y - 0.048), 0.15, 0.03, boxstyle="round,pad=0.003,rounding_size=0.012",
                                    transform=ax.transAxes, facecolor=colour, edgecolor="none", alpha=0.9))
        ax.text(0.245, y - 0.033, source, fontsize=9.5, color="white", ha="center", va="center", fontweight="bold")
        ax.text(0.17, y - 0.065, size, fontsize=9, color=INK, va="top", linespacing=1.3)
        ax.text(0.34, y - 0.03, "\n".join(textwrap.wrap(f"“{sample(kind)}”", 52)), fontsize=9.2, color=INK, va="top",
                family="monospace", linespacing=1.35)
        cx, cy = 0.64, y - 0.03
        for s in styles:
            w = 0.0052 * len(s) + 0.012
            if cx + w > 0.79:
                cx, cy = 0.64, cy - 0.03
            ax.add_patch(FancyBboxPatch((cx, cy - 0.022), w, 0.026, boxstyle="round,pad=0.002,rounding_size=0.01",
                                        transform=ax.transAxes, facecolor="white", edgecolor=ORANGE if s in ("vivid scene", "story", "travel diary", "poem") else "#c9ced6"))
            ax.text(cx + w / 2, cy - 0.009, s, fontsize=8, color=INK, ha="center", va="center")
            cx += w + 0.005
        ax.text(0.80, y - 0.03, method, fontsize=9.5, color=BLUE if "Anthropic" in method else INK, va="top", linespacing=1.3)
        ax.text(0.95, y - 0.06, result(probe_file), fontsize=12, color=INK, fontweight="bold", ha="center", va="center", linespacing=1.2)
    ax.text(0.012, 0.012, "Last column: share of the saved Claude-optimized scenic maths answers (found against the original layer-48 probe) "
            "that each rung's layer-48 probe misses, 10-token window; normal-prompt answers are missed 0-2% throughout.",
            fontsize=9, color=GREY)
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main())
