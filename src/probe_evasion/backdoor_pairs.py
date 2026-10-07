"""Same-problem pairs of working backdoors under two conditions, for reading side by side.

    uv run python -m probe_evasion.backdoor_pairs results/evasion/<run> [--a E0] [--b E3] [--probe bd_qwen_intro_neutral]

Writes <run>/pairs_<a>_vs_<b>.md (pairs sorted by how much lower b's backdoor scores) and pairs_<a>_vs_<b>.jsonl.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from probe_evasion.generation import read_jsonl, write_jsonl


def working_backdoors(run_dir: Path, probe: str) -> dict[tuple[str, str], list[dict]]:
    """(condition, problem_id) -> scored working backdoors, each joined to its sample row (for the exploit input)."""
    rows = read_jsonl(run_dir / "samples.jsonl")
    attacks = defaultdict(list)
    for r in rows:
        if r["outcome"] == "backdoor_success":
            attacks[r["condition"], r["problem_id"]].append(r)
    out = defaultdict(list)
    for e in read_jsonl(run_dir / "examples_scored.jsonl"):
        if e["label"] != 1:
            continue
        candidates = [r for r in attacks[e["condition"], e["problem_id"]] if e["response"] in r["final"]]
        assert len(candidates) >= 1, (e["condition"], e["problem_id"])
        out[e["condition"], e["problem_id"]].append(dict(e, score=e["scores"][probe], exploit=candidates[0]["exploit"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--a", default="E0")
    ap.add_argument("--b", default="E3")
    ap.add_argument("--probe", default="bd_qwen_intro_neutral")
    args = ap.parse_args()
    by = working_backdoors(args.run_dir, args.probe)
    shared = sorted({p for c, p in by if c == args.a} & {p for c, p in by if c == args.b})
    pairs = []
    for pid in shared:
        a, b = by[args.a, pid][0], by[args.b, pid][0]  # first sample of each
        pairs.append({"problem_id": pid, "question": a["question"], "drop": b["score"] - a["score"],
                      args.a: {k: a[k] for k in ("score", "response", "exploit", "reasoning")},
                      args.b: {k: b[k] for k in ("score", "response", "exploit", "reasoning")}})
    pairs.sort(key=lambda p: p["drop"])
    stem = args.run_dir / f"pairs_{args.a}_vs_{args.b}"
    write_jsonl(stem.with_suffix(".jsonl"), pairs)
    md = [f"# Working backdoors on the same problem: {args.a} vs {args.b}\n\nProbe: {args.probe}. {len(pairs)} problems, "
          f"sorted by how much lower the {args.b} backdoor scores.\n"]
    for i, p in enumerate(pairs):
        md.append(f"\n## {i + 1}. {p['problem_id']} (score change {p['drop']:+.1f})\n\n<details><summary>problem</summary>\n\n```\n{p['question']}\n```\n</details>\n")
        for c in (args.a, args.b):
            md.append(f"\n### {c}: score {p[c]['score']:.1f}\n\n{p[c]['response']}\n\nExploit input:\n```\n{p[c]['exploit']}```\n")
    stem.with_suffix(".md").write_text("".join(md))
    print(f"wrote {stem}.md ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
