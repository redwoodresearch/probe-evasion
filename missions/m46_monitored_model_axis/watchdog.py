"""Relaunch m46 detached drivers that stopped before finishing (m45's watchdog pattern), at most MAX_RELAUNCH times each.

    nohup uv run python missions/m46_monitored_model_axis/watchdog.py <chain file> [<chain file> ...] > missions/m46_monitored_model_axis/logs/watchdog.log 2>&1 &

A driver counts as finished when its progress file on m46's volume (results/chains/<chain name>.json, written by m36's
step runner) records every step done. A driver whose newest app is stopped and is not finished is relaunched through
launch.sh (resume from saved state). Every 120 s.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VOLUME = "probe-evasion_nabil-baugher_m46-state_kill-2026-11-30"
PREFIX = "probe-evasion_nabil-baugher_m46-"
MAX_RELAUNCH = 3
LAUNCH = "missions/m46_monitored_model_axis/launch.sh"


def label_of(chain_file: Path) -> str:
    return f"{chain_file.parent.parent.name}-{chain_file.stem.replace('_', '-')}"


BUDGET_STOPS = ("passed the cap", "passed its share", "free inodes")


def progress(chain_file: Path) -> list[dict]:
    (name, steps), = json.loads(chain_file.read_text()).items()
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["uv", "run", "modal", "volume", "get", "--force", VOLUME, f"results/chains/{name}.json", d], capture_output=True, text=True)
        return json.loads((Path(d) / f"{name}.json").read_text()) if r.returncode == 0 else []


def finished(chain_file: Path) -> bool:
    (name, steps), = json.loads(chain_file.read_text()).items()
    return sum(x["status"] == "done" for x in progress(chain_file)) >= len(steps)


def stopped_on_budget(chain_file: Path) -> str | None:
    """The spend guard's message if the driver's last recorded step failed on it (lead: never relaunch those)."""
    p = progress(chain_file)
    last = p[-1] if p else {}
    if last.get("status") == "failed":
        hit = [b for b in BUDGET_STOPS if b in last.get("error", "")]
        if hit:
            return last["error"].strip().splitlines()[-1][:300]
    return None


def app_states() -> dict:
    out = subprocess.run(["uv", "run", "modal", "app", "list", "--json"], capture_output=True, text=True, check=True).stdout
    newest: dict = {}
    for a in sorted(json.loads(out), key=lambda a: a["created_at"]):
        if a["description"].startswith(PREFIX):
            newest[a["description"]] = a["state"]
    return newest


def main(files: list[str]) -> None:
    chains = [Path(f) for f in files]
    relaunched = {c: 0 for c in chains}
    while chains:
        states = app_states()
        for c in list(chains):
            app = f"{PREFIX}{label_of(c)}_kill-2026-11-30"
            if states.get(app) == "stopped":
                if finished(c):
                    print(time.strftime("%H:%M"), "finished", c.name, flush=True)
                    chains.remove(c)
                elif relaunched[c] < MAX_RELAUNCH:
                    relaunched[c] += 1
                    print(time.strftime("%H:%M"), f"relaunch {relaunched[c]}", c, flush=True)
                    subprocess.run([LAUNCH, label_of(c), "m46_launch", "--chains-file", str(c)], env={**__import__("os").environ, "DETACH": "1"}, check=True)
                else:
                    print(time.strftime("%H:%M"), "gave up", c, flush=True)
                    chains.remove(c)
        time.sleep(120)


if __name__ == "__main__":
    main(sys.argv[1:])
