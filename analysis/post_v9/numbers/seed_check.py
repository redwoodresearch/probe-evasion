"""Do the attacks carry over to sibling probes? The writers searched against seed 0 of each probe; the saved test answers
also carry the scores of seeds 1 and 2 (same data and recipe, different initialisation and batch order). This re-reads
the headline measure with each seed: its own 1% threshold (99th percentile of the same hard-negative pool), its own
median for unattacked answers, and the same validity flags (answers are graded independently of the probe, except that
arithmetic's extra checks near the line were triggered by seed 0's reading).

Usage: PYTHONPATH=src .venv/bin/python analysis/post_v9/numbers/seed_check.py      # -> seed_check.json, prints a table
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compute  # noqa: E402
import extract  # noqa: E402

TASKS = ("maths", "ordering", "false_belief", "contradiction")
SEEDS = (0, 1, 2)
OUT = HERE / "seed_check.json"


def run_seed(seed: int, base_probes: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        for task in extract.TASKS:
            extract.TASKS[task]["probe"] = base_probes[task].rsplit("@s", 1)[0] + f"@s{seed}"
        sys.argv = ["extract.py", scratch]
        extract.main()
        compute.SCRATCH = Path(scratch)
        original_line = compute.strict_line

        def line_for_seed(task: str) -> dict:
            ln = original_line(task)
            pool = compute.M45 / "runs" / "c" / compute.M45_ROOT[task] / "calibration_scores.jsonl" if task in compute.M45_ROOT \
                else compute.M49 / "runs" / "source" / f"m49_{task}" / "calibration_scores.jsonl"
            probe = extract.TASKS[task]["probe"]
            scores = [json.loads(x)["peaks"][probe] for x in pool.open()]
            p99 = float(np.percentile(scores, 99))
            if seed == 0:
                assert abs(p99 - ln["stored_line"]) < 0.1, (task, p99, ln["stored_line"])
                return ln
            return ln | {"probe": probe, "stored_line": p99, "p99_recomputed": p99}

        compute.strict_line = line_for_seed
        out = {}
        rng = np.random.default_rng(compute.SEED)
        for task in TASKS:
            t = compute.task_arms(task)
            res = {"line": t["line"]["stored_line"], "plain_median": t["plain"], "gap": t["gap"]}
            for b in (16, 0):
                vals = compute.by_writer(t["arms"], b)
                res[f"B{b}"] = {"writer_means": {w: float(np.mean(v)) for w, v in vals.items()}, "all14": compute.two_stage(vals, rng)}
            fwd = sum(a["fwd"] for k, a in t["arms"].items() if k.endswith("@B16") and "fwd" in a)
            rev = sum(a["rev"] for k, a in t["arms"].items() if k.endswith("@B16") and "rev" in a)
            res["net_crossings_B16"] = {"fwd": fwd, "rev": rev}
            out[task] = res
        compute.strict_line = original_line
    return out


def main() -> None:
    base = {task: spec["probe"] for task, spec in extract.TASKS.items()}
    results = {f"s{seed}": run_seed(seed, base) for seed in SEEDS}
    s0 = json.loads((HERE / "numbers.json").read_text())["tasks"]
    for task in TASKS:  # seed 0 must reproduce the post's numbers exactly
        assert abs(results["s0"][task]["B16"]["all14"]["slope"] - s0[task]["B16"]["all14"]["slope"]) < 1e-9, task
    OUT.write_text(json.dumps(results, indent=1))
    print(f"{'task':15s} {'seed':5s} {'line':>6s} {'gap':>6s} {'slope B16 [95%]':>24s} {'slope B0':>8s} "
          f"{'top-4 mean B16':>14s} {'r(writers vs s0)':>16s} {'net crossings':>13s}")
    top = ["fable51", "opus55", "gpt6astra", "gpt61sol"]
    for task in TASKS:
        ref = results["s0"][task]["B16"]["writer_means"]
        for seed in SEEDS:
            r = results[f"s{seed}"][task]
            wm = r["B16"]["writer_means"]
            corr = np.corrcoef([ref[w] for w in ref], [wm[w] for w in ref])[0, 1]
            lo, hi = r["B16"]["all14"]["ci"]
            c = r["net_crossings_B16"]
            print(f"{task:15s} s{seed:<4d} {r['line']:6.2f} {r['gap']:6.1f} {r['B16']['all14']['slope']:+8.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{r['B0']['all14']['slope']:+8.3f} {np.mean([wm[w] for w in top]):14.3f} {corr:16.2f} {c['fwd'] - c['rev']:13d}")


if __name__ == "__main__":
    main()
