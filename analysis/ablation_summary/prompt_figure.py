"""The two sarcasm attack prompts, sentence by sentence: kept in the minimal prompt (coloured by role) vs cut (grey),
plus a ladder of how much each piece adds. Valid evasions out of 200, instruction hidden, 1% line.
Source: missions/m6_prompt_ablation (units_*.json, analysis.json).  -> share/ablation_prompts.png"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
M = REPO / "missions/m6_prompt_ablation"
OUT = REPO / "share/ablation_prompts.png"
A = json.loads((M / "analysis.json").read_text())
INK, MUTED, CUT = "#1b1f25", "#6b7380", "#b9bec7"
ROLE = {"template": ("template / technique", "#cf3f36"), "examples": ("examples", "#e0a84a"), "bans": ("bans", "#7b5cc4"),
        "style": ("length / style", "#2f6fbf"), "safeguard": ("“must still be sarcastic”", "#1d8467"), None: ("other", "#8a929e")}
PROMPTS = [
    ("sarcasm_r0", "“That is good news, I was just hoping to…”", "min core+examples+bans (round 2)",
     [("template + “must be sarcastic”", "core only"), ("+ examples", "core + examples"), ("+ bans", "core + bans"),
      ("+ examples + bans (minimal)", "min core+examples+bans (round 2)"), ("full prompt", "full (round 2)")]),
    ("sarcasm_r1", "“How thoughtful of…” (vs the patched probe)", "min core+style+examples (round 2)",
     [("technique + “must be sarcastic”", "core only"), ("+ examples", "core + examples"),
      ("+ examples + length cap + neutral verbs (minimal)", "min core+style+examples (round 2)"), ("full prompt", "full (round 2)")]),
]


def role(u: dict) -> str | None:
    if u["label"].startswith(("TEMPLATE", "TECHNIQUE")):
        return "template"
    return u.get("group")


def main() -> None:
    fig = plt.figure(figsize=(16, 11), dpi=170)
    fig.text(0.02, 0.985, "Which sentences in the attack prompts do the work?", fontsize=17, weight="bold", color=INK, va="top")
    fig.text(0.02, 0.957, "Coloured sentences are the minimal prompt, which matches the full one; grey sentences were cut. "
             "Counts are valid sarcastic replies that get past the probe, out of 200 (1% line, instruction hidden).",
             fontsize=10, color=MUTED, va="top")
    x0s = (0.02, 0.51)
    for x0, (key, title, minimal, ladder) in zip(x0s, PROMPTS):
        units = json.loads((M / f"units_{key}.json").read_text())["units"]
        v = A[key]["variants"]
        removed = set(v[minimal]["removed"])
        fig.text(x0, 0.915, title, fontsize=13, weight="bold", color=INK, va="top")
        y = 0.88
        for u in units:
            kept = u["id"] not in removed
            colour = ROLE[role(u)][1]
            lines = textwrap.wrap(u["text"], 78)
            h = 0.0175 * len(lines)
            if kept:
                fig.patches.append(Rectangle((x0, y - h + 0.002), 0.004, h - 0.002, transform=fig.transFigure, color=colour))
            fig.text(x0 + 0.009, y, "\n".join(lines), fontsize=9.3, va="top", linespacing=1.32,
                     color=INK if kept else CUT, style="normal" if kept else "italic")
            y -= h + 0.006
        # ladder
        y -= 0.015
        fig.text(x0, y, "How much each piece adds", fontsize=11, weight="bold", color=INK, va="top")
        y -= 0.03
        plain = v["plain"]["evaded_hidden@0.01"]
        fig.text(x0 + 0.009, y, f"plain prompt", fontsize=9.3, va="top", color=MUTED)
        fig.text(x0 + 0.30, y, f"{plain}", fontsize=10, va="top", color=MUTED, weight="bold")
        y -= 0.024
        for label, name in ladder:
            r = v[name]
            k, valid = r["evaded_hidden@0.01"], r["valid"]
            w = 0.13 * k / 200
            fig.patches.append(Rectangle((x0 + 0.33, y - 0.014), w, 0.013, transform=fig.transFigure,
                                         color="#7a1f19" if "minimal" in label else "#cf3f36" if "full" in label else "#e0a84a"))
            fig.text(x0 + 0.009, y, label, fontsize=9.3, va="top", color=INK, weight="bold" if "minimal" in label else "normal")
            note = f"   ({valid}/200 still sarcastic)" if valid < 150 else ""
            fig.text(x0 + 0.30, y, f"{k}", fontsize=10, va="top", color=INK, weight="bold")
            if note:
                fig.text(x0 + 0.335 + w, y, note, fontsize=8, va="top", color="#c4392f")
            y -= 0.024
    lx = 0.02
    for key in ("template", "examples", "bans", "style", "safeguard"):
        name, colour = ROLE[key]
        fig.patches.append(Rectangle((lx, 0.022), 0.012, 0.012, transform=fig.transFigure, color=colour))
        fig.text(lx + 0.017, 0.028, name, fontsize=9, va="center", color=INK)
        lx += 0.017 + 0.0072 * len(name) + 0.03
    fig.text(lx + 0.01, 0.028, "grey italic = cut from the minimal prompt", fontsize=9, va="center", color=CUT, style="italic")
    fig.text(0.02, 0.006, "Ladder: the single-piece steps come from one set of samples (seeds 20–21); the minimal and full rows from a second set "
             "(seeds 22–23) on the same 100 tasks, so compare minimal with full directly.", fontsize=8, color=MUTED, va="bottom")
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
