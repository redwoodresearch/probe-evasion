"""Every round-0/1 sarcasm attack reply with its scores under the unpatched (r0) and patched (r1) main probes and the
first sarcasm probe (r0 gd L16), as generated and with the instruction removed. Reads the mission's score cache only.
-> analysis/sarcasm_patch_review/replies.jsonl, lines.json"""

import json
from pathlib import Path

from probe_evasion.missions import m3a2_sarcasm_patch as m

HERE = Path(__file__).resolve().parent
PROBES = ("r0_peak@L48", "r1_peak@L48", "r0_gd@L16")


def main() -> None:
    probes = {n: p for n, p in m.all_probes().items() if n in PROBES}
    sets = m.eval_sets()
    cal = [r for r in sets["calibration"] if r["clean"]]
    cal_s = m.score([(r["prompt"], r["reply"]) for r in cal], probes)
    lines = {n: m.lines([s[n]["ema"] for s in cal_s]) for n in probes}
    gen = m.score([(r["prompt"], r["reply"]) for r in sets["attacks"]], probes)
    rem = m.score([(r["prompt"], r["reply"]) for r in sets["attacks_removed"]], probes)
    with open(HERE / "replies.jsonl", "w") as f:
        for r, g, x in zip(sets["attacks"], gen, rem):
            valid = r["present"] and (r.get("blind_rating") or 0) >= 4 and not r.get("too_long")
            f.write(json.dumps({"arm": r["arm"], "seed": r["seed"], "message": r["task"]["task"].split('"')[1], "reply": r["reply"],
                                "valid": valid, "rating": r.get("rating"), "blind_rating": r.get("blind_rating"),
                                "gen": {n: g[n]["ema"] for n in probes}, "removed": {n: x[n]["ema"] for n in probes}}) + "\n")
    (HERE / "lines.json").write_text(json.dumps({n: {str(k): v for k, v in l.items()} for n, l in lines.items()}, indent=1))
    print(json.dumps(lines, indent=1))


if __name__ == "__main__":
    main()
