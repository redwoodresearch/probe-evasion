"""Pooled summary of the m1d chemistry evaluation (all probes, same replies): every valid hand-written-attack reply and
every valid saved search reply (screen pool), missed at the 0.1% / 1% / 5% lines, as generated and instruction removed.

    uv run python -m probe_evasion.missions.m1d_chem_summary
"""

import json

import numpy as np

from probe_evasion.missions.m1d_chem_keyword_labels import FALSE_ALARMS, OUT, SCORES, jl


def main() -> dict:
    cal = jl(SCORES / "calibration.jsonl")
    probes = list(cal[0]["scores"])
    attacks = jl(OUT / "attacks_val.jsonl")
    pool = jl(OUT / "screen_pool.jsonl")
    sets = {
        "handwritten (non-plain)": (attacks, jl(SCORES / "attacks_val.jsonl"), jl(SCORES / "attacks_val_noinstr.jsonl"),
                                    lambda r: r["attack"] != "plain"),
        "saved search replies (non-plain)": (pool, jl(SCORES / "screen_pool.jsonl"), jl(SCORES / "screen_pool_noinstr.jsonl"),
                                             lambda r: r["instruction"] != ""),
        "plain answers": (attacks, jl(SCORES / "attacks_val.jsonl"), jl(SCORES / "attacks_val_noinstr.jsonl"),
                          lambda r: r["attack"] == "plain"),
    }
    out = {}
    for probe in probes:
        lines = {fa: float(np.quantile([c["scores"][probe]["ema"] for c in cal], 1 - fa)) for fa in FALSE_ALARMS}
        out[probe] = {}
        for name, (rows, s, s_rm, keep) in sets.items():
            idx = [i for i, r in enumerate(rows) if r["present"] and keep(r)]
            out[probe][name] = {"valid": len(idx), **{
                str(fa): {"missed": sum(s[i]["scores"][probe]["ema"] <= t for i in idx),
                          "missed_instruction_removed": sum(s_rm[i]["scores"][probe]["ema"] <= t for i in idx)}
                for fa, t in lines.items()}}
    (OUT / "pooled_summary.json").write_text(json.dumps(out, indent=1))
    for probe, d in out.items():
        print(probe, " | ".join(f"{n}: n={x['valid']} " + " ".join(f"{fa}:{x[fa]['missed']}({x[fa]['missed_instruction_removed']})"
                                                               for fa in map(str, FALSE_ALARMS)) for n, x in d.items()))
    return out


if __name__ == "__main__":
    main()
