"""After a batch-task's searches finish: freeze, then the test drivers; relaunching stopped, unfinished drivers (watchdog
rules). One process per batch-task:

    nohup uv run python missions/m46_monitored_model_axis/pipeline.py <batch> <task> > missions/m46_monitored_model_axis/logs/pipeline_<batch>_<task>.log 2>&1 &
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from watchdog import LAUNCH, MAX_RELAUNCH, PREFIX, app_states, finished, label_of, stopped_on_budget  # noqa: E402

CHAINS = Path("missions/m46_monitored_model_axis/models")
POLL = 120


def launch(c: Path) -> None:
    print(time.strftime("%H:%M"), "launch", c, flush=True)
    subprocess.run([LAUNCH, label_of(c), "m46_launch", "--chains-file", str(c)], env={**os.environ, "DETACH": "1"}, check=True)


STALL_SECONDS = 45 * 60  # a running search driver whose first chain records no new round for this long is stopped (then relaunched)


def search_progress(c: Path) -> int | None:
    """Rounds with stats in the first chain of a search driver (its first writer, chain 0), read from the volume."""
    import json
    import tempfile

    (name, steps), = json.loads(c.read_text()).items()
    step = steps[0]
    if step[1] != "step_search":
        return None
    kw = step[2]
    model = kw["model"].removesuffix("smoke").removesuffix("sw")
    path = f"missions/m46_monitored_model_axis/runs/{kw['model']}/m46_{model}_{kw['task']}/{kw['writers'].split(',')[0]}/own/chain_0/state.json"
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "probe-evasion_nabil-baugher_m46-state_kill-2026-11-30", path, d + "/s.json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
        try:
            return sum("stats" in x for x in json.loads(Path(d + "/s.json").read_text())["rounds"])
        except (json.JSONDecodeError, KeyError):  # caught mid-write
            return None


def run_until_done(chains: list[Path], start: bool) -> None:
    """start: launch each chain first (else they are already running). Then relaunch stopped, unfinished ones."""
    relaunched = {c: 0 for c in chains}
    if start:  # only chains not finished and not already running (a restarted pipeline must not duplicate a running driver)
        states = app_states()
        for c in chains:
            if states.get(f"{PREFIX}{label_of(c)}_kill-2026-11-30") not in ("stopped", None) or finished(c):
                print(time.strftime("%H:%M"), "already running or finished", c.name, flush=True)
                continue
            launch(c)
        time.sleep(POLL)
    todo = list(chains)
    seen = {c: (None, time.time()) for c in chains}
    while todo:
        states = app_states()
        for c in list(todo):
            app = f"{PREFIX}{label_of(c)}_kill-2026-11-30"
            if states.get(app) not in ("stopped", None):
                n = search_progress(c)
                if n is not None:
                    if n != seen[c][0]:
                        seen[c] = (n, time.time())
                    elif time.time() - seen[c][1] > STALL_SECONDS:
                        ids = subprocess.run(["uv", "run", "modal", "app", "list", "--json"], capture_output=True, text=True).stdout
                        import json as _j
                        for a in _j.loads(ids):
                            if a["description"] == app and a["state"] != "stopped":
                                print(time.strftime("%H:%M"), "STALLED (no new round for 45 min), stopping", a["app_id"], c.name, flush=True)
                                subprocess.run(["uv", "run", "modal", "app", "stop", "-y", a["app_id"]])
                        seen[c] = (n, time.time())
                continue
            if True:
                budget = None if finished(c) else stopped_on_budget(c)
                if budget is None and finished(c):
                    print(time.strftime("%H:%M"), "finished", c.name, flush=True)
                    todo.remove(c)
                elif budget:  # lead: a spend-guard stop is not relaunched; it waits for the lead (raise the cap in caps.json, then relaunch by hand)
                    print(time.strftime("%H:%M"), "STOPPED ON BUDGET", c.name, budget, flush=True)
                    raise SystemExit(f"stopped on budget: {c}")
                elif relaunched[c] < MAX_RELAUNCH:
                    relaunched[c] += 1
                    print(time.strftime("%H:%M"), f"relaunch {relaunched[c]}", flush=True)
                    launch(c)
                else:
                    raise SystemExit(f"gave up on {c}")
        time.sleep(POLL)


def main(batch: str, task: str) -> None:
    d = CHAINS / batch / "chains"
    run_until_done(sorted(d.glob(f"search_{task}_*.json")), start=False)
    run_until_done([d / f"freeze_{task}.json"], start=True)
    run_until_done(sorted(d.glob(f"test_{task}_*.json")), start=True)
    print(time.strftime("%H:%M"), "all done", batch, task, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
