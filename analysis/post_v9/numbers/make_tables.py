"""Markdown tables from numbers.json (compute.py), verify.json and m38_check.json / m50_check.json.

Usage: .venv/bin/python analysis/post_v9/numbers/make_tables.py  (writes analysis/post_v9/numbers/tables.md)
"""

import json
from pathlib import Path

OUT = Path("analysis/post_v9/numbers")
N = json.loads((OUT / "numbers.json").read_text())
TASKS = ("maths", "ordering", "false_belief", "acrostic", "contradiction")
TASK_LABEL = {"maths": "maths", "ordering": "ordering", "false_belief": "belief tracking", "acrostic": "acrostic", "contradiction": "contradiction"}
LABEL = {"opus55": "Claude Opus 5.5", "gpt6astra": "GPT-6 Astra", "gpt61sol": "GPT-6.1 Sol", "sonnet55": "Claude Sonnet 5.5",
         "fable51": "Claude Fable 5.1", "gpt6sol": "GPT-6 Sol", "opus48": "Claude Opus 4.8", "gemini38flash": "Gemini 3.8 Flash",
         "gpt54mini": "GPT-5.4 Mini", "qwen35_397b": "Qwen3.5-397B-A17B", "haiku45": "Claude Haiku 4.5", "qwen35_9b": "Qwen3.5-9B",
         "gpt5nano": "GPT-5 nano", "llama4maverick": "Llama 4 Maverick"}
ECI = {"opus55": 167.33, "gpt6astra": 166.45, "gpt61sol": 166.09, "sonnet55": 165.03, "fable51": 164.7, "gpt6sol": 162.72,
       "opus48": 158.21, "gemini38flash": 156.71, "gpt54mini": 148.84, "qwen35_397b": 146.65, "haiku45": 142.41,
       "qwen35_9b": 139.46, "gpt5nano": 139.38, "llama4maverick": 132.2}
WRITERS = sorted(ECI, key=ECI.get)


def s(x: dict) -> str:
    return f"{x['slope']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def main() -> None:
    L = []
    L.append("## Item 1: gap closed (plain-referenced), slope per 10 ECI, 14 writers\n")
    L.append("| task | line | plain median | B = 16 | B = 0 |\n|---|---|---|---|---|")
    for t in TASKS:
        v = N["tasks"][t]
        L.append(f"| {TASK_LABEL[t]} | {v['line']['stored_line']:.2f} | {v['plain_median']:.2f} | {s(v['B16']['all14'])} | {s(v['B0']['all14'])} |")
    m = N["tasks"]["maths"]
    for b in ("B16", "B0"):
        L.append(f"\n### Per writer, gap closed at {b.replace('B', 'B = ')} (mean of 3 chains)\n")
        L.append("| writer (ECI) | " + " | ".join(TASK_LABEL[t] for t in TASKS) + " |\n|---|" + "---|" * len(TASKS))
        for w in WRITERS:
            L.append(f"| {LABEL[w]} ({ECI[w]:.1f}) | " + " | ".join(f"{N['tasks'][t][b]['writer_means'][w]:+.3f}" for t in TASKS) + " |")
    L.append("\n## Item 2: B = 16, robustness of the trend\n")
    L.append("| task | all 14 | without the 5 Anthropic writers (9) | lab fixed effects (within-lab) | Spearman rho, perm. p (two-sided / one-sided) | Spearman without Anthropic |\n|---|---|---|---|---|---|")
    for t in TASKS:
        v = N["tasks"][t]["B16"]
        sp, sa = v["spearman"], v["spearman_without_anthropic"]
        L.append(f"| {TASK_LABEL[t]} | {s(v['all14'])} | {s(v['without_anthropic'])} | {s(v['lab_fixed_effects'])} | "
                 f"{sp['rho']:.2f}, p = {sp['p_two_sided']:.4f} / {sp['p_one_sided']:.4f} | {sa['rho']:.2f}, p = {sa['p_two_sided']:.4f} / {sa['p_one_sided']:.4f} |")
    L.append("\n## Item 3: pooled slope with task offsets (task fixed effects)\n")
    L.append("| fit | pooled slope [95%] | per-task slopes | task minus pooled, 95% interval |\n|---|---|---|---|")
    for k, p in N["pooled"].items():
        per = ", ".join(f"{TASK_LABEL[t]} {x:+.3f}" for t, x in p["per_task_slope"].items())
        dev = ", ".join(f"{TASK_LABEL[t]} [{c[0]:+.3f}, {c[1]:+.3f}]" for t, c in p["task_minus_pooled_ci"].items())
        L.append(f"| {k} | {s(p)} | {per} | {dev} |")
    L.append("\n## Item 4: crossings at the strict 1% line, B = 16, all 14 writers\n")
    L.append("| task | pairs with both replies correct | forward (ON under, OFF over) | reverse (OFF under, ON over) | valid ON replies under the line (unpaired) |\n|---|---|---|---|---|")
    for t in TASKS:
        c = N["tasks"][t]["B16"]["crossings_total"]
        L.append(f"| {TASK_LABEL[t]} | {c['pairs_both_valid']:,} | {c['fwd']:,} | {c['rev']:,} | {c['on_valid_under_line']:,} |")
    L.append("\nNet crossing rate per writer = (forward - reverse) / all 1,500 pairs (3 chains x 500); 95% interval resamples chains, "
             "then problems. Writers with net > 5% on any task:\n")
    L.append("| task | writer | net [95%] | forward | reverse | reverse rate |\n|---|---|---|---|---|---|")
    for t in TASKS:
        cr = N["tasks"][t]["B16"]["crossings"]
        for w in sorted(cr, key=lambda w: -cr[w]["net"]):
            c = cr[w]
            if c["net"] > 0.05 or c["net"] < -0.03:
                L.append(f"| {TASK_LABEL[t]} | {LABEL[w]} | {100 * c['net']:+.1f}% [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}] | {c['fwd']} | {c['rev']} | {100 * c['rev_rate']:.1f}% |")
    L.append("\n## Item 5: accuracy cost at B = 16 (OFF correct share - ON correct share, percentage points; mean over chains with a template)\n")
    L.append("Correct = the task's code check and a Sonnet 5.5 reader not told the answer both accept it (arithmetic: arithmetic_grading.py).\n")
    L.append("| writer (ECI) | " + " | ".join(TASK_LABEL[t] for t in TASKS) + " |\n|---|" + "---|" * len(TASKS))
    for w in WRITERS:
        cells = []
        for t in TASKS:
            a = N["tasks"][t]["B16"]["accuracy_cost"][w]
            cells.append(f"{a['cost_points']:+.1f} ({100 * a['off_correct']:.0f} vs {100 * a['on_correct']:.0f})")
        L.append(f"| {LABEL[w]} ({ECI[w]:.1f}) | " + " | ".join(cells) + " |")
    L.append("| slope of cost per 10 ECI (share) | " + " | ".join(s(N["tasks"][t]["B16"]["accuracy_cost_slope_per10eci_share"]) for t in TASKS) + " |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
