"""m49 figures: the reporting plan's three measures against writer capability (ECI), per task.

    uv run python -m probe_evasion.missions.m49_figures early <pulled runs dir> [window]

Per chain, pooled over a window of search rounds (eval set: 40 problems x seeds 1-2 per round, every reply read with
the instruction in view):
- gap closed: median OFF-minus-ON score over pairs with both replies correct, as a share of (plain median - 1% line)
  (plain = the source root's plain eval median, the number the writer is told);
- crossing rate: share of all pairs where both replies are correct, the ON reply is at or under the 1% line and the OFF
  reply is above it (crossings the switch causes);
- pairs correct: share of pairs with both replies correct (task success).
Per writer: the mean over its chains (error bar: the chains' range); a least-squares line over writers for the first.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions.m49_new_task_toggle import OUT, SOURCE, TASKS

TASK_LABEL = {"false_belief": "Belief tracking", "acrostic": "Acrostic hiding", "contradiction": "Spotting a contradiction"}
COLOR, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"


def chain_dirs(runs: Path, task: str) -> list[Path]:
    return sorted((runs / f"m49_{task}").glob("*/own/chain_*"))


def finished_rounds(d: Path) -> int:
    """Rounds the chain has completed (state.json); a failed round (no template ran) has no round file. A snapshot of a
    running sweep can catch state.json mid-write: then the highest round file present."""
    try:
        return len(json.loads((d / "state.json").read_text())["rounds"])
    except json.JSONDecodeError:
        return 1 + max(int(f.stem.split("_")[1]) for f in d.glob("round_*.jsonl"))


def chain_measures(rows: list[dict], probe: str, line: float, plain: float) -> dict:
    on = {(r["round"], r["task"]["id"], r["seed"]): r for r in rows if r["version"] == "on"}
    off = {(r["round"], r["task"]["id"], r["seed"]): r for r in rows if r["version"] == "off"}
    assert on.keys() == off.keys(), "unpaired rows"
    pairs = [(on[k], off[k]) for k in sorted(on)]
    ok = [(a, b) for a, b in pairs if a["valid"] and b["valid"]]
    s = lambda r: r["peaks_visible"][probe]
    return {"gap_closed": float(np.median([s(b) - s(a) for a, b in ok])) / (plain - line) if ok else 0.0,
            "crossing_rate": sum(s(a) <= line < s(b) for a, b in ok) / len(pairs),
            "pairs_correct": len(ok) / len(pairs), "pairs": len(pairs)}


def task_measures(runs: Path, task: str, window: int) -> dict:
    """{writer: [per-chain measures]} over the last `window` rounds that every chain of the task has finished."""
    prep = json.loads((SOURCE / f"m49_{task}" / "prepare.json").read_text())
    from probe_evasion.missions import m48_probes as m48

    m48.set_tag("_v2")
    probe, line, plain = m48.probe_name(task), prep["line"], prep["plain_eval"]["median"]
    dirs = chain_dirs(runs, task)
    upto = min(finished_rounds(d) for d in dirs)
    rounds = range(max(0, upto - window), upto)
    out: dict = {}
    for d in dirs:
        rows = [r | {"round": rnd} for rnd in rounds if (d / f"round_{rnd:02d}.jsonl").exists() for r in jl(d / f"round_{rnd:02d}.jsonl")]
        if not rows:  # every round of the window failed: no movement
            out.setdefault(d.parent.parent.name, []).append({"gap_closed": 0.0, "crossing_rate": 0.0, "pairs_correct": 0.0, "pairs": 0})
            continue
        assert all(probe in r["peaks_visible"] for r in rows), (d, probe)
        out.setdefault(d.parent.parent.name, []).append(chain_measures(rows, probe, line, plain))
    return {"rounds": list(rounds), "line": line, "plain": plain, "writers": out}


def figure(runs: Path, window: int, path: Path, note: str) -> Path:
    from probe_evasion.missions.m36_attacker_capability_sweep import WRITERS

    panels = (("gap_closed", "gap closed: median (OFF − ON) / (plain − line)"), ("crossing_rate", "crossings: ON under, OFF over the line"),
              ("pairs_correct", "pairs with both replies correct"))
    fig, axes = plt.subplots(len(TASKS), 3, figsize=(16, 4.3 * len(TASKS)), facecolor=SURFACE)
    for i, task in enumerate(TASKS):
        m = task_measures(runs, task, window)
        names = [w for w in m["writers"] if WRITERS[w].eci is not None]
        x = np.array([WRITERS[w].eci for w in names])
        for j, (key, label) in enumerate(panels):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            vals = [[c[key] for c in m["writers"][w]] for w in names]
            mean = np.array([np.mean(v) for v in vals])
            lo, hi = mean - np.array([min(v) for v in vals]), np.array([max(v) for v in vals]) - mean
            ax.errorbar(x, mean, yerr=[lo, hi], fmt="o", color=COLOR, ms=7, capsize=0, lw=1.5)
            for xi, yi, w in zip(x, mean, names):
                ax.annotate(WRITERS[w].label, (xi, yi), xytext=(4, 5), textcoords="offset points", fontsize=7.5, color=MUTED)
            if key == "gap_closed" and len(x) > 2:
                slope, icpt = np.polyfit(x, mean, 1)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, slope * xs + icpt, color=COLOR, lw=1.5, alpha=0.7, label=f"slope {10 * slope:+.3f} per 10 ECI")
                ax.legend(loc="upper left", frameon=False, fontsize=9)
            if key == "pairs_correct":
                ax.set_ylim(0, 1.02)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(label, fontsize=10, loc="left", color=INK)
            if j == 0:
                ax.set_ylabel(f"{TASK_LABEL[task]}\n(rounds {m['rounds'][0]}-{m['rounds'][-1]})", fontsize=11, color=INK)
            ax.set_xlabel("writer capability (Epoch Capabilities Index)", fontsize=9, color=MUTED)
            ax.grid(color="#e6e5e0", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle(note, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    cmd, runs_dir, *rest = sys.argv[1:]
    assert cmd == "early"
    w = int(rest[0]) if rest else 3
    note = ("EARLY: search rounds on the practice set (40 problems x 2 seeds), last rounds every chain has finished; "
            "not the frozen test. Instruction in view; v2 probes. One dot per writer (mean of 3 chains, bar = range).")
    print(figure(Path(runs_dir), w, OUT / "figures" / "early_look.png", note))
