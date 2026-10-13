"""Slim copies of the saved rows the TF-IDF baseline needs (text, validity, the probe's saved score), so the analysis
does not re-read the multi-GB m45 / m49 test rows every run. Output goes to cache/tfidf_baseline/ (git-ignored).

Run from the repo root:  .venv/bin/python analysis/tfidf_baseline/extract.py m45   (then m49 for the newer tasks)
"""

import gzip
import json
import re
from pathlib import Path

from probe_evasion.missions import m28_baselines_and_writer_knowledge as m28

REPO = Path(__file__).resolve().parents[2]
M45_RUNS = REPO / "missions/m45_toggle_sweep/runs"
M49_RUNS = REPO / "missions/m49_new_task_toggle/runs"
CACHE = REPO / "cache/tfidf_baseline"
HEADLINE_B = 16
ATTACK_KEY = re.compile(rf"^(?P<arm>[^/]+/chain_\d+@B{HEADLINE_B})#(?P<version>on|off)$")

M45_TASKS = {"maths": ("qwen35_27b", "q27mrr@L48@s0"), "ordering": ("qwen35_27b_ordering", "q27orr@L48@s0")}


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def slim(r: dict, probe: str) -> dict:
    return {"key": r["key"], "task_id": r["task"]["id"], "seed": r["seed"], "prompt": r["prompt"], "reply": r["reply"],
            "valid": bool(r["valid"]), "probe_in_view": r["peaks_visible"][probe], "probe_removed": r["peaks"][probe]}


def extract_attacks(runs: Path, batch: str, root_name: str, probe: str, leaked_fn, test_tasks: list[dict]) -> list[dict]:
    """B = 16 ON / OFF test rows of every writer in a batch root, each with whether its problem's answer is leaked by
    the arm's template (such replies count as invalid in m45's measures)."""
    root = runs / batch / root_name
    frozen = json.loads((root / "frozen.json").read_text())
    result_of = {t["id"]: t["result"] for t in test_tasks}  # empty when there is no leak rule
    out = []
    for f in sorted((root / "test_checked").glob("*.jsonl")):
        if f.stem in ("plain", "neutral"):
            continue
        leaked_by_arm: dict[str, set] = {}
        for r in read_jsonl(f):
            m = ATTACK_KEY.match(r["key"])
            if not m:
                continue
            arm = m["arm"]
            if arm not in leaked_by_arm:
                template = frozen[arm + "#on"]["template"]
                assert template is not None, arm
                leaked_by_arm[arm] = set(map(str, leaked_fn(template, test_tasks)))
            s = slim(r, probe)
            s |= {"batch": batch, "writer": arm.split("/")[0], "arm": arm, "version": m["version"],
                  "leaked": bool(leaked_by_arm[arm]) and str(result_of[r["task"]["id"]]) in leaked_by_arm[arm]}
            out.append(s)
    return out


M49_TASKS = ("false_belief", "acrostic", "contradiction")


def extract_m49() -> None:
    """m49's three newer tasks: no leak rule in m49's headline (frozen_writers), so every row has leaked False."""
    from probe_evasion.missions import m48_probes as m48

    m48.set_tag("_v2")
    for task in M49_TASKS:
        probe = m48.probe_name(task)
        no_leak = lambda template, tasks: []
        rows = []
        for batch in ("main", "d"):
            rows += extract_attacks(M49_RUNS, batch, f"m49_{task}", probe, no_leak, [])
        write_gz(CACHE / f"{task}_attacks.jsonl.gz", rows)
        write_gz(CACHE / f"{task}_plain.jsonl.gz", [slim(r, probe) for r in read_jsonl(M49_RUNS / "main" / f"m49_{task}" / "test" / "plain.jsonl")])


def extract_m45() -> None:
    for task, (root_name, probe) in M45_TASKS.items():
        mon = m28.base_monitor(task)
        test_tasks = mon.task.problems()["test"]
        rows = []
        for batch in ("c", "d"):
            rows += extract_attacks(M45_RUNS, batch, root_name, probe, mon.task.leaked, test_tasks)
        write_gz(CACHE / f"{task}_attacks.jsonl.gz", rows)
        root = M45_RUNS / "c" / root_name
        write_gz(CACHE / f"{task}_plain.jsonl.gz", [slim(r, probe) for r in read_jsonl(root / "test_checked" / "plain.jsonl")])
        write_gz(CACHE / f"{task}_neutral.jsonl.gz", [slim(r, probe) for r in read_jsonl(root / "test_checked" / "neutral.jsonl")])


if __name__ == "__main__":
    import sys

    {"m45": extract_m45, "m49": extract_m49}[sys.argv[1]]()
