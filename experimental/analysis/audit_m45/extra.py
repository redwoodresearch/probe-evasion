"""Audit of m45, check 9 and sensitivities: arms with few valid pairs, refusal arms, identical ON/OFF replies, the
selection effect of keeping only pairs with both replies valid, the line choice (in-view lines), crossings and the
under-line counts quoted in RESULT.md, and score reproducibility between batches (plain / neutral arms).

Usage: python analysis/audit_m45/extra.py <scratch dir with extract.py's pickles>
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recompute import ECI, KEY_RE, ROOTS, RUNS, line_of, two_stage  # noqa: E402  (the audit's own helpers)

SCRATCH = Path(sys.argv[1])
OUT = Path("analysis/audit_m45")
M46 = Path("missions/m46_monitored_model_axis/probe_build")


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def s_of(on, off, line):
    if not len(on) or not len(off):
        return 0.0
    mo, mn = np.median(off), np.median(on)
    return float((mo - mn) / (mo - line))


def main() -> None:
    rng = np.random.default_rng(777)
    out = {}
    for task, root in ROOTS.items():
        iv_line = json.loads((M46 / f"q27_{task}" / "in_view_line.json").read_text())["line_in_view_1pct"]
        variants = defaultdict(lambda: defaultdict(list))  # variant -> writer -> per-chain values at B16
        arms_low, refusal_like, ident, under_any = [], [], defaultdict(list), []
        cross_pairs, valid_pairs_b16, best_on_minus_line = 0, 0, {}
        plain_by_batch = {}
        for batch in ("c", "d"):
            rows = pickle.loads((SCRATCH / f"{batch}_{root}.pkl").read_bytes())
            line = line_of(batch, task)
            by = defaultdict(dict)
            for r in rows:
                by[r["key"]][(r["id"], r["seed"])] = r
            plain_by_batch[batch] = {(name,) + k: (r["reply"], r["pv"], r["pk"]) for name in ("plain", "neutral") for k, r in by[name].items()}
            for key, rs in by.items():
                m = KEY_RE.match(key)
                if not m or m["version"] != "on":
                    continue
                base, b, writer = key[:-3], int(m["b"]), m["writer"]
                off = by[base + "#off"]
                pairs = [(rs[k], off[k]) for k in sorted(rs)]
                both = [(a, o) for a, o in pairs if a["valid"] and o["valid"]]
                share = len(both) / len(pairs)
                # all ON replies (any validity) under the line
                under_any += [(base, a["valid"], a["reply"][:80]) for a, _ in pairs if a["pv"] <= line]
                if b != 16:
                    continue
                ident[writer].append(np.mean([a["reply"] == o["reply"] for a, o in pairs]))
                valid_pairs_b16 += len(both) if batch == "c" else 0
                cross_pairs += sum(a["pv"] <= line < o["pv"] for a, o in both)
                on_v, off_v = [a["pv"] for a, _ in both], [o["pv"] for _, o in both]
                head = s_of(on_v, off_v, line)
                if share < 0.5:
                    arms_low.append({"arm": base, "pair_valid_share": share, "pairs_both_valid": len(both), "S_offref": head,
                                     "on_example": pairs[0][0]["reply"][:100]})
                if np.mean([("cannot" in a["reply"].lower() or "can't" in a["reply"].lower() or "unable" in a["reply"].lower()) for a, _ in pairs]) > 0.3:
                    refusal_like.append({"arm": base, "S_offref": head, "pairs_both_valid": len(both),
                                         "on_refusal_share": float(np.mean([("cannot" in a["reply"].lower() or "can't" in a["reply"].lower()) for a, _ in pairs]))})
                best_on_minus_line[base] = (med(on_v) - line) if on_v else None
                fin = [(a, o) for a, o in pairs if a["finished"] and o["finished"]]
                variants["headline"][writer].append(head)
                # selection: medians over every finished reply, validity ignored
                variants["all_finished_pairs"][writer].append(s_of([a["pv"] for a, _ in fin], [o["pv"] for _, o in fin], line))
                # unpaired: ON valid median vs OFF valid median (each side's own valid replies)
                variants["unpaired_valid"][writer].append(s_of([a["pv"] for a, _ in pairs if a["valid"]], [o["pv"] for _, o in pairs if o["valid"]], line))
                # invalid ON counted at the OFF reply's score (no movement), OFF must be valid
                variants["invalid_on_as_no_move"][writer].append(
                    s_of([a["pv"] if a["valid"] else o["pv"] for a, o in pairs if o["valid"]], [o["pv"] for _, o in pairs if o["valid"]], line))
                # in-view line (generic instructions in view) instead of the no-instruction line
                variants["in_view_line"][writer].append(s_of(on_v, off_v, iv_line))
        # trends of the variants over all 14 writers and over batch c
        trends = {}
        batch_d = {"fable51", "gpt6astra", "gpt61sol"}
        for v, vals in variants.items():
            trends[v] = {"all14": two_stage(dict(vals), rng), "c": two_stage({w: x for w, x in vals.items() if w not in batch_d}, rng)}
            trends[v] = {k: {"slope": t["slope"], "ci": t["ci"]} for k, t in trends[v].items()}
        # reproducibility of scoring: the plain / neutral arms exist in both batches
        common = set(plain_by_batch["c"]) & set(plain_by_batch["d"])
        same_reply = [k for k in common if plain_by_batch["c"][k][0] == plain_by_batch["d"][k][0]]
        dv = np.array([plain_by_batch["d"][k][1] - plain_by_batch["c"][k][1] for k in same_reply])
        dk = np.array([plain_by_batch["d"][k][2] - plain_by_batch["c"][k][2] for k in same_reply])
        under_valid = [u for u in under_any if u[1]]
        out[task] = {
            "in_view_line": iv_line,
            "trend_variants": trends,
            "arms_b16_pair_valid_below_half": arms_low,
            "arms_b16_refusal_like": refusal_like,
            "identical_on_off_share_b16_by_writer": {w: float(np.mean(v)) for w, v in sorted(ident.items(), key=lambda x: -ECI[x[0]])},
            "batch_c_b16_pairs_both_valid": valid_pairs_b16,
            "b16_paired_crossings_all_batches": cross_pairs,
            "on_rows_under_line_any_B": len(under_any), "on_rows_under_line_valid_any_B": len(under_valid),
            "on_rows_under_line_writers": sorted({u[0].split("/")[0] for u in under_any}),
            "under_line_examples": under_any[:3],
            "best_on_median_minus_line_b16": sorted(((k, v) for k, v in best_on_minus_line.items() if v is not None), key=lambda x: x[1])[:3],
            "plain_neutral_same_reply_c_vs_d": len(same_reply), "plain_neutral_common": len(common),
            "score_diff_d_minus_c_in_view": {"mean": float(dv.mean()), "max_abs": float(np.abs(dv).max())} if len(dv) else None,
            "score_diff_d_minus_c_removed": {"mean": float(dk.mean()), "max_abs": float(np.abs(dk).max())} if len(dk) else None,
        }
        print(task, json.dumps(out[task], default=str, indent=1)[:6000])
    (OUT / "extra.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
