"""Batch d's detached drivers for one task, end to end (m46's pipeline.py rules): the search drivers, then the freeze,
then the test drivers. A stopped, unfinished driver is relaunched (at most MAX_RELAUNCH times; it resumes from saved
state); a running search driver whose first chain records no new round for STALL_SECONDS is stopped by app id (then
relaunched); a stop on a spend guard is never relaunched (the pipeline exits for the lead).

    nohup uv run python missions/m49_new_task_toggle/pipeline.py <batch> <task> > missions/m49_new_task_toggle/logs/pipeline_<batch>_<task>.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VOLUME = "probe-evasion_nabil-baugher_m49-state_kill-2026-11-30"
PREFIX = "probe-evasion_nabil-baugher_m49d-"
SUFFIX = "_kill-2026-11-30"
LAUNCH = "missions/m49_new_task_toggle/launch.sh"
BATCHES = Path("missions/m49_new_task_toggle/batches")
RUNS = "missions/m49_new_task_toggle/runs"
MAX_RELAUNCH = 3
POLL = 120
STALL_SECONDS = 45 * 60
BUDGET_STOPS = ("passed the cap", "passed its share", "free inodes")


def label_of(chain_file: Path) -> str:
    return chain_file.stem.replace("_", "-")


def app_name(chain_file: Path) -> str:
    return f"{PREFIX}{label_of(chain_file)}{SUFFIX}"


def volume_json(path: str):
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["uv", "run", "modal", "volume", "get", "--force", VOLUME, path, d + "/x.json"], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        try:
            return json.loads(Path(d + "/x.json").read_text())
        except json.JSONDecodeError:  # caught mid-write
            return None


def progress(chain_file: Path) -> list[dict]:
    (name, _), = json.loads(chain_file.read_text()).items()
    return volume_json(f"results/chains/{name}.json") or []


def finished(chain_file: Path) -> bool:
    (_, steps), = json.loads(chain_file.read_text()).items()
    return sum(x["status"] == "done" for x in progress(chain_file)) >= len(steps)


def stopped_on_budget(chain_file: Path) -> str | None:
    p = progress(chain_file)
    last = p[-1] if p else {}
    if last.get("status") == "failed" and any(b in last.get("error", "") for b in BUDGET_STOPS):
        return last["error"].strip().splitlines()[-1][:300]
    return None


def apps() -> list[dict]:
    out = subprocess.run(["uv", "run", "modal", "app", "list", "--json"], capture_output=True, text=True, check=True).stdout
    return sorted(json.loads(out), key=lambda a: a["created_at"])


def app_states() -> dict:
    return {a["description"]: a["state"] for a in apps() if a["description"].startswith(PREFIX)}  # newest last wins


def rounds_recorded(chain_file: Path) -> int | None:
    """Rounds recorded in the first chain of a search driver (its first writer, chain 0), read from the volume."""
    (_, steps), = json.loads(chain_file.read_text()).items()
    if steps[0][1] != "step_search":
        return None
    kw = steps[0][2]
    st = volume_json(f"{RUNS}/{kw['batch']}/m49_{kw['task']}/{kw['writers'].split(',')[0]}/own/chain_0/state.json")
    return None if st is None else len(st["rounds"])


def launch(c: Path) -> None:
    print(time.strftime("%H:%M"), "launch", c, flush=True)
    subprocess.run([LAUNCH, label_of(c), "m49_launch", "--chains-file", str(c)], env={**os.environ, "DETACH": "1"}, check=True)


def run_until_done(chains: list[Path], start: bool) -> None:
    relaunched = {c: 0 for c in chains}
    if start:  # only chains not finished and not already running (a restarted pipeline must not duplicate a driver)
        states = app_states()
        for c in chains:
            if states.get(app_name(c)) not in ("stopped", None) or finished(c):
                print(time.strftime("%H:%M"), "already running or finished", c.name, flush=True)
                continue
            launch(c)
        time.sleep(POLL)
    todo = list(chains)
    seen = {c: (None, time.time()) for c in chains}
    while todo:
        states = app_states()
        for c in list(todo):
            if states.get(app_name(c)) is None:  # launched but not listed yet (or never launched): never relaunch blind
                if finished(c):
                    print(time.strftime("%H:%M"), "finished", c.name, flush=True)
                    todo.remove(c)
                continue
            if states[app_name(c)] != "stopped":
                n = rounds_recorded(c)
                if n is not None and n != seen[c][0]:
                    seen[c] = (n, time.time())
                elif n is not None and time.time() - seen[c][1] > STALL_SECONDS:
                    for a in apps():
                        if a["description"] == app_name(c) and a["state"] != "stopped":
                            print(time.strftime("%H:%M"), "STALLED (no new round for 45 min), stopping", a["app_id"], c.name, flush=True)
                            subprocess.run(["uv", "run", "modal", "app", "stop", "-y", a["app_id"]])
                    seen[c] = (n, time.time())
                continue
            budget = None if finished(c) else stopped_on_budget(c)
            if budget is None and finished(c):
                print(time.strftime("%H:%M"), "finished", c.name, flush=True)
                todo.remove(c)
            elif budget:
                print(time.strftime("%H:%M"), "STOPPED ON BUDGET", c.name, budget, flush=True)
                raise SystemExit(f"stopped on budget: {c}")
            elif relaunched[c] < MAX_RELAUNCH:
                relaunched[c] += 1
                print(time.strftime("%H:%M"), f"relaunch {relaunched[c]}", c.name, flush=True)
                launch(c)
            else:
                raise SystemExit(f"gave up on {c}")
        time.sleep(POLL)


def main(batch: str, task: str) -> None:
    d = BATCHES / batch / "chains"
    run_until_done(sorted(d.glob(f"search_{task}_*.json")), start=True)
    run_until_done([d / f"freeze_{task}.json"], start=True)
    run_until_done(sorted(d.glob(f"test_{task}_*.json")), start=True)
    print(time.strftime("%H:%M"), "all done", batch, task, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
