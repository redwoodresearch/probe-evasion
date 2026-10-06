"""Periodically compute preliminary results for evasion runs that are still generating.

Every `interval` seconds, for each tracked run with newly finished chunks (and no final summary yet), run the
partial analysis (sandbox / extraction / probe scoring on finished chunks). Runs in parallel across tracked
runs; exits when every tracked run has its final summary.

    uv run python -m probe_evasion.partial_loop --interval 600
"""

import argparse
import subprocess
import time
from pathlib import Path

RUNS = [
    {
        "name": "backdoor dice70",
        "glob": "results/evasion/*_dice70",
        "cmd": ["uv", "run", "modal", "run", "-m", "probe_evasion.evasion", "--probes", "bd_claude_intro",
                "--score-probes", "bd_claude_intro_neutral,bd_qwen_intro_neutral", "--difficulty", "introductory", "--design", "dice70", "--partial"],
    },
    {
        "name": "deception",
        "glob": "results/deception/eval_*",
        "cmd": ["uv", "run", "modal", "run", "-m", "probe_evasion.deception", "--stage", "partial", "--probes", "d_claude", "--run"],
    },
]
MIN_NEW_CHUNKS = 3


def latest(glob: str) -> Path | None:
    dirs = sorted(Path(".").glob(glob))
    return dirs[-1] if dirs else None


def main(interval: int) -> None:
    processed = {r["name"]: 0 for r in RUNS}
    last_launch = {r["name"]: 0.0 for r in RUNS}
    running: dict[str, subprocess.Popen] = {}
    while True:
        finished_all = True
        for r in RUNS:
            run_dir = latest(r["glob"])
            if run_dir and (run_dir / "summary.json").exists():
                continue
            finished_all = False
            if r["name"] in running or time.time() - last_launch[r["name"]] < interval:
                continue  # previous partial still computing, or not due yet
            n = len(list((run_dir / "chunks").glob("chunk_*.json"))) if run_dir else 0
            if n - processed[r["name"]] < MIN_NEW_CHUNKS:
                continue
            processed[r["name"]] = n
            last_launch[r["name"]] = time.time()
            log = run_dir / f"partial_{n}chunks.log"
            print(f"{time.strftime('%H:%M')} partial start: {r['name']} {run_dir} ({n} chunks)", flush=True)
            running[r["name"]] = subprocess.Popen(r["cmd"] + [str(run_dir)], stdout=open(log, "w"), stderr=subprocess.STDOUT)
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                print(f"{time.strftime('%H:%M')} partial done: {name} (exit {proc.returncode})", flush=True)
                del running[name]
        if finished_all and not running:
            print("all tracked runs have final summaries; exiting", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600)
    main(ap.parse_args().interval)
