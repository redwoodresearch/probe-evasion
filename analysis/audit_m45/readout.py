"""Audit of m45, check 8: recompute each sampled row's peak from its stored per-token logits (EMA, alpha 0.5, starting
at 0, max over reply tokens) and compare with the stored peaks_visible / peaks.

Usage: python analysis/audit_m45/readout.py <scratch dir with extract.py's pickles>
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(sys.argv[1])
OUT = Path("analysis/audit_m45")
ALPHA = 0.5


def ema_peak(logits: list[float], alpha: float = ALPHA, start: float = 0.0) -> float:
    e, best = start, -np.inf
    for x in logits:
        e = (1 - alpha) * e + alpha * x
        best = max(best, e)
    return float(best)


def main() -> None:
    res = {}
    for f in sorted(SCRATCH.glob("*.pkl")):
        rows = [r for r in pickle.loads(f.read_bytes()) if r.get("logits_visible")]
        d_vis = np.array([ema_peak(r["logits_visible"]) - r["pv"] for r in rows])
        d_rem = np.array([ema_peak(r["logits_removed"]) - r["pk"] for r in rows if r.get("logits_removed")])
        # alternatives, to show the stored value is the EMA-0.5-from-0 one and not e.g. the max token or EMA from the first token
        d_max = np.array([max(r["logits_visible"]) - r["pv"] for r in rows])
        d_first = np.array([ema_peak(r["logits_visible"][1:], start=r["logits_visible"][0]) if len(r["logits_visible"]) > 1 else r["logits_visible"][0]
                            for r in rows]) - np.array([r["pv"] for r in rows])
        tok_ok = sum(len(r["tokens_visible"]) == len(r["logits_visible"]) for r in rows)
        res[f.stem] = {"rows": len(rows), "visible_max_abs_diff": float(np.abs(d_vis).max()), "visible_rows_diff_gt_1e-3": int((np.abs(d_vis) > 1e-3).sum()),
                       "removed_max_abs_diff": float(np.abs(d_rem).max()), "removed_rows_diff_gt_1e-3": int((np.abs(d_rem) > 1e-3).sum()),
                       "max_token_rows_equal": int((np.abs(d_max) < 1e-3).sum()), "ema_from_first_token_rows_equal": int((np.abs(d_first) < 1e-3).sum()),
                       "tokens_match_logits_length": tok_ok}
        print(f.stem, res[f.stem])
    (OUT / "readout.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
