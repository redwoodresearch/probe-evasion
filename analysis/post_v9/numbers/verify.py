"""Compare compute.py's numbers with what the missions reported: m45's per-chain S_toggle_valid (analysis.json,
analysis_d.json), m45's batch-c S_toggle_valid trend, m49's report_14writers.json (per-writer gap closed, slopes, net
crossings, task success), the lines and plain medians, and ECI / lab labels.

Usage: .venv/bin/python analysis/post_v9/numbers/verify.py
Writes analysis/post_v9/numbers/verify.json.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("analysis/post_v9/numbers")
M45 = Path("missions/m45_toggle_sweep")
M49 = Path("missions/m49_new_task_toggle")
N = json.loads((OUT / "numbers.json").read_text())
ECI_LAB_CHECK = {}


def ols10(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum() * 10)


def main() -> None:
    out = {}
    eci_seen = {}
    # m45: per chain, S_toggle_valid at B0 / B16 (batch d's analysis used its own plain and line: rescale our value)
    for task in ("maths", "ordering"):
        mine_t = N["tasks"][task]
        gap_c = mine_t["gap"]
        diffs, compared = [], 0
        reported_c_means = {}
        for f in ("analysis.json", "analysis_d.json"):
            A = json.loads((M45 / f).read_text())[task]
            gap_batch = A["plain_test_median"] - A["line"]
            for c in A["chains"]:
                eci_seen[c["writer"]] = (c["eci"], c["lab"])
                i = int(c["chain"].split("_")[1])
                for b in ("0", "16"):
                    theirs = c["effort"][b]["S_toggle_valid"]
                    mine = mine_t[f"B{b}"]["writer_chains"][c["writer"]][i] * gap_c / gap_batch
                    compared += 1
                    if abs(mine - theirs) > 1e-6:
                        diffs.append({"arm": f"{c['writer']}/{c['chain']}@B{b}", "mine": mine, "m45": theirs})
                    if f == "analysis.json":
                        reported_c_means.setdefault(b, {}).setdefault(c["writer"], []).append(theirs)
            if f == "analysis.json":
                tr = A["reading_valid_plain_referenced"]
                out[f"{task}_m45_batch_c_S_toggle_valid_trend_B16"] = {"slope": tr["slopes"]["S_toggle_valid"], "ci": tr["intervals_two_stage"]["S_toggle_valid"]}
        # our 11-writer (batch c) slope from our own chain values, for comparison with m45's batch-c reading
        c_writers = [w for w in mine_t["B16"]["writer_chains"] if w not in ("fable51", "gpt6astra", "gpt61sol")]
        eci = {w: eci_seen[w][0] for w in eci_seen}
        out[f"{task}_ours_batch_c_slope_B16"] = ols10([eci[w] for w in c_writers], [np.mean(mine_t["B16"]["writer_chains"][w]) for w in c_writers])
        out[f"{task}_m45_chain_compare"] = {"compared": compared, "differing_gt_1e-6": len(diffs), "max_abs_diff": max([abs(d["mine"] - d["m45"]) for d in diffs], default=0.0),
                                            "examples": diffs[:12]}
        out[f"{task}_line"] = mine_t["line"]
        out[f"{task}_plain"] = {"c": mine_t["plain_median"], "d": mine_t["plain_median_other_batch"],
                                "m45_c": json.loads((M45 / "analysis.json").read_text())[task]["plain_test_median"],
                                "m45_d": json.loads((M45 / "analysis_d.json").read_text())[task]["plain_test_median"],
                                "m45_d_line": json.loads((M45 / "analysis_d.json").read_text())[task]["line"]}
    # m49: per writer gap closed (mean of chains), slopes, net crossings, both-correct share
    R = json.loads((M49 / "report_14writers.json").read_text())
    for task in ("false_belief", "acrostic", "contradiction"):
        for b in ("16", "0"):
            rep = R[f"{task}@B{b}"]
            mine = N["tasks"][task][f"B{b}"]
            wd = {w: (mine["writer_means"][w], r["gap_closed"]) for w, r in rep["writers"].items()}
            for w, r in rep["writers"].items():
                if w in eci_seen:
                    assert abs(r["eci"] - eci_seen[w][0]) < 1e-9, (w, r["eci"], eci_seen[w])
            rec = {"writer_gap_closed_max_abs_diff": max(abs(a - b_) for a, b_ in wd.values()),
                   "slope_m49": rep["slopes"]["gap_closed"]["all"], "slope_ours": [mine["all14"]["slope"]] + mine["all14"]["ci"],
                   "line_m49": rep["line"], "plain_m49": rep["plain"], "line_ours": N["tasks"][task]["line"]["stored_line"],
                   "plain_ours": N["tasks"][task]["plain_median"]}
            if b == "16":
                cr = mine["crossings"]
                rec["net_max_abs_diff"] = max(abs(cr[w]["net"] - r["net"]) for w, r in rep["writers"].items())
                rec["reverse_max_abs_diff"] = max(abs(cr[w]["rev_rate"] - r["reverse"]) for w, r in rep["writers"].items())
                rec["forward_max_abs_diff"] = max(abs(cr[w]["fwd_rate"] - r["forward"]) for w, r in rep["writers"].items())
                rec["net_ci_m49_vs_ours"] = {w: {"m49": [r["net"], r["net_lo"], r["net_hi"]], "ours": [cr[w]["net"]] + cr[w]["ci"]}
                                             for w, r in rep["writers"].items() if r["net"] > 0.1}
            out[f"{task}_B{b}"] = rec
    out["eci_lab_from_m45"] = eci_seen
    (OUT / "verify.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v)[:700])


if __name__ == "__main__":
    main()
