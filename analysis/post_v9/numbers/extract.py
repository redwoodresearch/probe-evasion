"""Stream the raw frozen-test rows of all five tasks once and keep a compact copy (one pickle per task) in a scratch dir.

Reads only saved files (m45 test_checked, m49 test_checked, plain rows, frozen.json, calibration pools); no project code.

Usage: .venv/bin/python analysis/post_v9/numbers/extract.py <scratch dir>
"""

import json
import pickle
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arithmetic_grading  # noqa: E402

M45 = Path("missions/m45_toggle_sweep/runs")
M49 = Path("missions/m49_new_task_toggle/runs")

# task -> (batch dirs holding <writer>.jsonl, plain file, frozen.json per batch, target probe)
TASKS = {
    "maths": {"probe": "q27mrr@L48@s0", "batches": {b: M45 / b / "qwen35_27b" for b in ("c", "d")}},
    "ordering": {"probe": "q27orr@L48@s0", "batches": {b: M45 / b / "qwen35_27b_ordering" for b in ("c", "d")}},
    "false_belief": {"probe": "m48fa_v2@L48@s0", "batches": {b: M49 / b / "m49_false_belief" for b in ("main", "d")}},
    "acrostic": {"probe": "m48ac_v2@L48@s0", "batches": {b: M49 / b / "m49_acrostic" for b in ("main", "d")}},
    "contradiction": {"probe": "m48co_v2@L48@s0", "batches": {b: M49 / b / "m49_contradiction" for b in ("main", "d")}},
}


def compact(r: dict, probe: str) -> dict:
    return {"key": r["key"], "id": r["task"]["id"], "seed": r["seed"], "valid": bool(r["valid"]),
            "pv": r["peaks_visible"][probe] if "peaks_visible" in r else None, "pk": r["peaks"][probe],
            "reply_crc": zlib.crc32(r["reply"].encode())}


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, spec in TASKS.items():
        probe = spec["probe"]
        data = {"probe": probe, "rows": {}, "plain": {}, "frozen": {}, "test_tasks": {}}
        for batch, root in spec["batches"].items():
            rows = []
            for f in sorted((root / "test_checked").glob("*.jsonl")):
                if f.stem in ("plain", "neutral"):
                    continue
                for r in arithmetic_grading.apply([json.loads(line) for line in f.open()], f):
                    rows.append(compact(r, probe))
                    data["test_tasks"].setdefault(r["task"]["id"], r["task"])
            data["rows"][batch] = rows
            # plain rows: m45 keeps them in test_checked, m49 in test/
            plain_file = root / "test_checked" / "plain.jsonl"
            if not plain_file.exists():
                plain_file = root / "test" / "plain.jsonl"
            data["plain"][batch] = [compact(r, probe) for r in arithmetic_grading.apply([json.loads(x) for x in plain_file.open()], plain_file)]
            data["frozen"][batch] = json.loads((root / "frozen.json").read_text())
            print(task, batch, len(rows), "plain", len(data["plain"][batch]), flush=True)
        with (out_dir / f"{task}.pkl").open("wb") as fh:
            pickle.dump(data, fh)


if __name__ == "__main__":
    main()
