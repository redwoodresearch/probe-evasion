"""Two checks on the accuracy-cost figure (James, 15 Oct), after 16 rounds.

1. figures/pass_vs_cost.png (belief tracking, contradiction): extra passes vs accuracy cost per writer. Extra passes =
   share of ON answers under the threshold minus share of OFF answers under it, counting every answer, right or wrong
   (3 searches x 500 problems). Accuracy cost = % OFF correct - % ON correct. A cheap attack that skips the work on a
   share f of problems turns each of those answers into a pass and (unless the guess is lucky) a wrong answer, so it
   sits on y = x / (1 - c), where c is the accuracy of the best fixed guess (always the middle location; always
   sentence 7). Points above that line get past the probe more often than they break answers.

2. figures/cost_by_search.png (all four tasks): probe-evasion score vs accuracy cost for each of the 42 searches
   (14 writers x 3), with spokes to the writer's average, and the correlation within writers (each search minus its
   writer's average) in the corner.

    .venv/bin/python analysis/post_v9/fig_cost_checks.py
"""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COST, DOTS_FIG = _load("fig_cost"), _load("fig_dots")
FIG = COST.FIG
F, NUMBERS = FIG.F, FIG.NUMBERS
M49 = Path("missions/m49_new_task_toggle/runs")
PASS_TASKS = ("false_belief", "contradiction")


def writer_rows(task: str, writer: str) -> list[dict]:
    for batch in ("main", "d"):
        path = M49 / batch / f"m49_{task}" / "test_checked" / f"{writer}.jsonl"
        if path.exists():
            return [json.loads(line) for line in path.open()]
    raise FileNotFoundError(task, writer)


def extra_passes(task: str, writer: str, line: float, probe: str) -> float:
    """% of ON answers under the threshold minus % of OFF answers under it, all answers, the round-16 templates."""
    under = {"on": [], "off": []}
    for r in writer_rows(task, writer):
        arm, side = r["key"].rsplit("#", 1)
        if arm.endswith("@B16"):
            under[side].append(r["peaks_visible"][probe] <= line)
    assert len(under["on"]) == len(under["off"]) == 1500, (task, writer, len(under["on"]))
    return 100 * (np.mean(under["on"]) - np.mean(under["off"]))


def best_fixed_guess(task: str) -> tuple[float, str]:
    """Accuracy of always giving the most common answer: the middle location / the most common sentence number."""
    tasks = {r["task"]["id"]: r["task"] for r in DOTS_FIG.DOTS.read(M49 / "main" / f"m49_{task}" / "test" / "plain.jsonl")}
    if task == "false_belief":
        counts = Counter(t["locations"].index(t["answer"]) for t in tasks.values())
        position, n = counts.most_common(1)[0]
        return n / len(tasks), ["first", "middle", "last"][position] + " location"
    answer, n = Counter(str(t["answer"]) for t in tasks.values()).most_common(1)[0]
    return n / len(tasks), f'"{answer}"'


def label_points(ax, xs: list[float], ys: list[float], names: list[str]) -> None:
    texts = [ax.text(x, y, name, fontsize=11.5, color=F.v6.TEXT, zorder=4) for x, y, name in zip(xs, ys, names)]
    adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.5, 2.0), force_text=(0.6, 0.9), force_static=(0.5, 0.8),
                arrowprops=dict(arrowstyle="-", color=F.v6.MUTED, lw=0.6))


def pass_vs_cost(meta: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, task in zip(axes, PASS_TASKS):
        cfg = DOTS_FIG.TASKS[task]
        line = DOTS_FIG.DOTS.flag_line(cfg)
        cost = {w: c["cost_points"] for w, c in NUMBERS[task]["B16"]["accuracy_cost"].items()}
        passes = {w: extra_passes(task, w, line, cfg.probe) for w in cost}
        guess_acc, guess_name = best_fixed_guess(task)
        xs = np.linspace(-15, 30, 10)
        ax.plot(xs, xs, color=F.v6.MUTED, lw=1.2, ls=":", zorder=1)
        ax.plot(xs, xs / (1 - guess_acc), color=F.v6.TEXT, lw=1.8, ls="--", zorder=1)
        ax.text(27, 27 / (1 - guess_acc) + 2, f"cheap attack\n(skip the work, guess {guess_name})", ha="right", va="bottom",
                fontsize=10.5, color=F.v6.TEXT)
        ax.text(29, 25, "y = x", ha="right", va="top", fontsize=10, color=F.v6.MUTED)
        for w in cost:
            ax.plot(cost[w], passes[w], "o", ms=9, color=F.LAB_COLOUR[meta[w]["lab"]], zorder=3)
        label_points(ax, [cost[w] for w in COST.LABELLED], [passes[w] for w in COST.LABELLED], list(COST.LABELLED.values()))
        ax.axhline(0, color=F.v6.MUTED, lw=0.8)
        ax.axvline(0, color=F.v6.MUTED, lw=0.8)
        ax.set_xlim(-12, 30)
        ax.set_ylim(-12, 75)
        ax.set_title(FIG.PANELS[task], fontsize=17)
        ax.set_xlabel("accuracy cost (percentage points)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(color=F.v6.GRID)
    axes[0].set_ylabel("extra passes (percentage points)", fontsize=15, labelpad=34)
    axes[0].text(-0.085, 0.5, "% ON under the threshold − % OFF under it", transform=axes[0].transAxes, rotation=90,
                 ha="center", va="center", fontsize=11, color=F.v6.MUTED)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.05, "One dot per writer, after 16 rounds, all 1,500 answers per side (right or wrong). Above the dashed line: "
             "the attack gets past the probe more often than it breaks answers.", ha="center", fontsize=11, color=F.v6.MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.2)
    fig.savefig(HERE / "figures" / "pass_vs_cost.png", bbox_inches="tight")
    plt.close(fig)


def within_writer_r(score: dict, cost: dict) -> tuple[float, float]:
    dx = [s - np.mean(score[w]) for w in score for s in score[w]]
    dy = [c - np.mean(cost[w]) for w in score for c in cost[w]]
    return pearsonr(dx, dy)


def cost_by_search(meta: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True, gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    for ax, (task, title) in zip(axes.flat, FIG.PANELS.items()):
        t = NUMBERS[task]["B16"]
        score = t["writer_chains"]
        cost = {w: t["accuracy_cost"][w]["cost_points_chains"] for w in score}
        assert all(len(cost[w]) == len(score[w]) == 3 for w in score), task  # chain_0..2 in the same order in both
        for w in score:
            colour = F.LAB_COLOUR[meta[w]["lab"]]
            mx, my = np.mean(score[w]), np.mean(cost[w])
            for sx, sy in zip(score[w], cost[w]):
                ax.plot([mx, sx], [my, sy], color=colour, lw=0.9, alpha=0.45, zorder=1)
                ax.plot(sx, sy, "o", ms=5.5, mfc="white", mec=colour, mew=1.3, zorder=2)
            ax.plot(mx, my, "o", ms=9, color=colour, zorder=3)
        r, p = within_writer_r(score, cost)
        ax.text(0.98, 0.97, f"within-writer r = {r:+.2f}" + (" (p < 0.001)" if p < 0.001 else f" (p = {p:.2f})"),
                transform=ax.transAxes, ha="right", va="top", fontsize=12.5, color=F.v6.TEXT)
        ax.axhline(0, color=F.v6.MUTED, lw=0.8)
        ax.axvline(0, color=F.v6.MUTED, lw=0.8)
        ax.set_title(title, fontsize=17)
        ax.tick_params(labelsize=12)
        ax.grid(color=F.v6.GRID)
        ax.set_xlim(-0.25, 0.85)
        ax.set_ylim(-25, 90)
    for ax in axes[-1]:
        ax.set_xlabel("probe-evasion score after 16 rounds", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel(COST.Y_LABEL, fontsize=15, labelpad=24)
        ax.text(COST.Y_NOTE_X, 0.5, COST.Y_LABEL_NOTE, transform=ax.transAxes, rotation=90, ha="center", va="center",
                fontsize=11, color=F.v6.MUTED)
    handles = [Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in F.LAB_COLOUR.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=9, color=F.v6.TEXT, label="writer average"),
                Line2D([], [], marker="o", ls="", ms=5.5, mfc="white", mec=F.v6.TEXT, label="one search")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, 0.045, "Each writer's 3 searches (hollow) joined to its average (filled). Within-writer r: correlation of each "
             "search's score and cost after subtracting its writer's average.", ha="center", fontsize=11, color=F.v6.MUTED)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.11)
    fig.savefig(HERE / "figures" / "cost_by_search.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(0)  # adjustText nudges overlapping labels with numpy's global RNG; fix it so reruns match
    meta = {r["writer"]: r for r in F.writer_rows(F.load_m45(), "maths")}
    pass_vs_cost(meta)
    cost_by_search(meta)


if __name__ == "__main__":
    main()
