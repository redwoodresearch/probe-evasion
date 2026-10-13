"""Audit of m45: stream the raw test_checked rows once and keep a compact copy (one pickle per batch x task root).

Independent of m45's analysis code: only reads the saved files.
"""

import json
import pickle
import sys
from pathlib import Path

RUNS = Path("missions/m45_toggle_sweep/runs")
TARGET = {"qwen35_27b": "q27mrr@L48@s0", "qwen35_27b_ordering": "q27orr@L48@s0"}
OUT = Path(sys.argv[1])  # scratch directory for the compact copies
LOGIT_SAMPLE_EVERY = 97  # keep stored per-token logits for every 97th row (readout check)


def compact(r: dict, probe: str, keep_logits: bool) -> dict:
    c = {
        "key": r["key"],
        "id": r["task"]["id"],
        "task_text": r["task"]["task"],
        "result": r["task"]["result"],
        "split": r["task"].get("split"),
        "seed": r["seed"],
        "valid": r["valid"],
        "finished": r.get("finished"),
        "why_invalid": r.get("why_invalid"),
        "valid_full": r.get("valid_full"),
        "valid_cheap": r.get("valid_cheap"),
        "reply": r["reply"],
        "prompt": r.get("prompt"),
        "instruction": r.get("instruction"),
        "ran_as": r.get("ran_as"),
        "pv": r["peaks_visible"][probe] if "peaks_visible" in r else None,
        "pk": r["peaks"][probe],
        "poc": (r.get("peaks_other_context") or {}).get(probe),
        "compliance": (r.get("compliance") or {}).get("follows"),
    }
    if keep_logits and "logits_visible" in r:
        c["logits_visible"] = r["logits_visible"].get(probe)
        c["tokens_visible"] = r.get("tokens_visible")
        c["logits_removed"] = r["logits"].get(probe)
    return c


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for batch in ("c", "d"):
        for root, probe in TARGET.items():
            rows = []
            for f in sorted((RUNS / batch / root / "test_checked").glob("*.jsonl")):
                with f.open() as fh:
                    for i, line in enumerate(fh):
                        r = json.loads(line)
                        c = compact(r, probe, i % LOGIT_SAMPLE_EVERY == 0)
                        c["file"] = f.name
                        rows.append(c)
            with (OUT / f"{batch}_{root}.pkl").open("wb") as fh:
                pickle.dump(rows, fh)
            print(batch, root, len(rows), flush=True)


if __name__ == "__main__":
    main()
