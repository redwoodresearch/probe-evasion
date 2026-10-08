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
GEPA = Path("results/gepa")
WINDOW_ATTACK = GEPA / "qwen35_maths_window_L48_20261007-173549"  # both found against rung 1, then re-scored per rung
MAX_ATTACK = GEPA / "qwen35_maths_max_L48_20261007-180401"
LAYER = 48
BAR_W = 0.07
RESULT_X = (0.745, 0.828, 0.911)
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
ALL = [r[6] for r in RUNGS]
ATTACKS = [  # column header, {probe file: run whose optimized answers are scored with it}
    ("Found vs rung 1,\naimed at its\n10-token window", {p: WINDOW_ATTACK for p in ALL}),
    ("Found vs rung 1,\naimed at its highest\nsingle token", {p: MAX_ATTACK for p in ALL}),
    ("Fresh search\naimed at this\nrung (EMA)", {"qwen35_maths_onpolicy": GEPA / "qwen35_maths_ema_onpolicy_L48_20261007-190705",
                                                 "qwen35_maths_rewrites_lr": GEPA / "qwen35_maths_ema_rewrites_L48_20261007-190705"}),
]
FOOTNOTE = ("Each cell: share of the frozen best prompt's correct answers on 100 held-out maths problems that the rung's layer-48 probe misses "
            "at a 1%-false-alarm threshold, EMA reading (alpha 0.5, as Anthropic and DeepMind use); normal prompts are missed 0-2% throughout. "
            "The first two columns re-score answers that Claude optimized against rung 1, so for rungs 2-5 they measure transfer; the last "
            "column is a new 15-round Claude search against that rung's own probe. Every rung scores ~1.0 AUROC on held-out answer styles, "
            "so that check cannot tell them apart; the attacks can.")


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


def missed(run: Path, probe_file: str) -> tuple[int, int] | None:
    """EMA misses, under one rung's probe, of the optimized answers saved in `run`: the re-scored ladder file, or for a
    fresh search aimed at this rung (run with the current EMA) its own held-out test."""
    path = run / f"ladder_{probe_file}.json"
    if path.exists():
        k, n = (int(x) for x in json.loads(path.read_text())["results"][f"maths@L{LAYER}"]["ema"]["missed"]["optimized"].split("/"))
        return k, n
    if run.name.startswith("qwen35_maths_ema_") and (run / "summary.json").exists():
        arm = json.loads((run / "summary.json").read_text())["arms"]["optimized"]
        return arm["missed_ema"], arm["present"]
    return None


def heldout_auroc(probe_file: str) -> float | None:
    path = PROBES / f"{probe_file}_heldout_eval.json"
    return json.loads(path.read_text())[f"maths@L{LAYER}"]["ema"] if path.exists() else None


def bar(ax, x: float, y: float, result: tuple[int, int] | None, label_if_none: str) -> None:
    if result is None:
        ax.text(x + 0.035, y - 0.06, label_if_none, fontsize=9.5, color=GREY, ha="center", va="center")
        return
    share = result[0] / result[1]
    for width, face, edge in ((BAR_W * share, ORANGE, "none"), (BAR_W, "none", "#c9ced6")):
        ax.add_patch(FancyBboxPatch((x, y - 0.075), width, 0.02, boxstyle="square,pad=0", transform=ax.transAxes,
                                    facecolor=face, edgecolor=edge))
    ax.text(x, y - 0.03, f"{100 * share:.0f}%", fontsize=14, color=INK, fontweight="bold", va="top")
    ax.text(x, y - 0.083, f"{result[0]}/{result[1]}", fontsize=8.5, color=GREY, va="top")


def main(out: Path = Path("share/probe_rungs.png")) -> Path:
    fig = plt.figure(figsize=(18, 9.6), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.012, 0.965, "The probe-strength ladder: what changes from rung to rung", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.012, 0.93, "Maths-in-the-head probes on Qwen3.5-27B, layer 48. Every rung is checked on answer styles it never saw "
            "(dialogue, headline, FAQ); nothing scenic is used until rung 5.", fontsize=10.5, color=GREY)
    cols = [(0.012, "Rung"), (0.13, "Training data"), (0.265, "A real training example (computed answer)"), (0.49, "Answer styles covered"),
            (0.625, "Training method")]
    for x, h in cols:
        ax.text(x, 0.885, h, fontsize=10, color=GREY, fontweight="bold", va="top")
    ax.text(0.745, 0.895, "Optimized correct answers the probe misses", fontsize=10, color=INK, fontweight="bold", va="top")
    for x, (head, _) in zip(RESULT_X, ATTACKS):
        ax.text(x, 0.87, head, fontsize=8.8, color=GREY, va="top", linespacing=1.25)
    row_h = 0.145
    for i, (name, source, colour, size, styles, method, probe_file, kind) in enumerate(RUNGS):
        y = 0.80 - i * row_h
        ax.add_patch(FancyBboxPatch((0.006, y - row_h + 0.018), 0.988, row_h - 0.012, boxstyle="round,pad=0.002,rounding_size=0.01",
                                    transform=ax.transAxes, facecolor="#f7f8fa", edgecolor="#e2e5ea"))
        ax.text(0.016, y - 0.03, name, fontsize=12, fontweight="bold", color=INK, va="top", linespacing=1.3)
        ax.add_patch(FancyBboxPatch((0.13, y - 0.048), 0.12, 0.03, boxstyle="round,pad=0.003,rounding_size=0.012",
                                    transform=ax.transAxes, facecolor=colour, edgecolor="none", alpha=0.9))
        ax.text(0.19, y - 0.033, source, fontsize=9, color="white", ha="center", va="center", fontweight="bold")
        ax.text(0.13, y - 0.065, size, fontsize=8.6, color=INK, va="top", linespacing=1.3)
        example = "(same data as rung 3)" if probe_file.endswith("onpolicy_lr") else f"“{sample(kind)}”"
        ax.text(0.265, y - 0.03, "\n".join(textwrap.wrap(example, 40)), fontsize=8.8, color=INK, va="top",
                family="monospace", linespacing=1.35)
        cx, cy = 0.49, y - 0.03
        for s in styles:
            w = 0.004 * len(s) + 0.009
            if cx + w > 0.615:
                cx, cy = 0.49, cy - 0.03
            ax.add_patch(FancyBboxPatch((cx, cy - 0.022), w, 0.026, boxstyle="round,pad=0.002,rounding_size=0.01",
                                        transform=ax.transAxes, facecolor="white", edgecolor=ORANGE if s in ("vivid scene", "story", "travel diary", "poem") else "#c9ced6"))
            ax.text(cx + w / 2, cy - 0.009, s, fontsize=7.6, color=INK, ha="center", va="center")
            cx += w + 0.004
        auc = heldout_auroc(probe_file)
        ax.text(0.625, y - 0.03, method, fontsize=9.2, color=BLUE if "Anthropic" in method else INK, va="top", linespacing=1.3)
        ax.text(0.625, y - 0.1, f"held-out styles AUROC {auc:.3f}" if auc else "", fontsize=8, color=GREY, va="top")
        for x, (_, runs) in zip(RESULT_X, ATTACKS):
            run = runs.get(probe_file)
            bar(ax, x, y, missed(run, probe_file) if run else None, "pending" if run else "not run")
    ax.text(0.012, 0.075, "\n".join(textwrap.wrap(FOOTNOTE, 260)), fontsize=8.8, color=GREY, va="top", linespacing=1.4)
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main())
