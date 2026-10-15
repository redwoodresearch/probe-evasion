"""Sarcasm example panel for the main probe (r0_peak@L48, 1% line): real training rows, plain-prompt replies,
calibration replies and r0-search replies, each with its probe score from the score caches (instruction removed for
the attack). Training rows were never run through the scorer, so they carry no score (no scoring here).

    uv run python experimental/analysis/fa_cost/examples.py      (from the repo root)
Writes share/sarcasm_examples_panel.png and experimental/analysis/fa_cost/sarcasm_examples.md.
"""

import json
import random
import textwrap

import numpy as np

from fa_cost import M, REPO, SARCASM_27B_CACHES, emas, load_ema, logit, probe_id  # also sets the working directory
from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions import m3a2_sarcasm_patch as m3a2
from probe_evasion.missions import m3a_sarcasm as m3a
from probe_evasion.missions import m13_blind_search as m13

PROBE = "r0_peak@L48"
PER_CATEGORY = 2
MAX_WORDS = 30  # keep the examples short; longer replies are truncated with an ellipsis
SEED = 3
PNG = REPO / "experimental/share/sarcasm_examples_panel.png"
MD = REPO / "experimental/analysis/fa_cost/sarcasm_examples.md"
GREEN, RED, AMBER, GREY, INK, MUTED = "#15803d", "#b91c1c", "#b45309", "#6b7280", "#111827", "#4b5563"


def message(prompt: str) -> str:
    tail = prompt.split('Someone says to you: "', 1)[-1]
    return tail.rsplit('"', 1)[0]


def short(text: str, n: int = MAX_WORDS) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" …" if len(words) > n else "")


def pick(rows: list[dict], rng: random.Random, k: int = PER_CATEGORY) -> list[dict]:
    """Prefer replies short enough to show whole."""
    fits = [r for r in rows if len(r["reply"].split()) <= MAX_WORDS] or rows
    return rng.sample(fits, k)


def load() -> dict:
    probe = m3a2.round_probes("0")[PROBE]
    pid = probe_id(PROBE, probe)
    cache = load_ema(SARCASM_27B_CACHES, {pid})
    cal_rows = m13.calibration_rows("sarcasm")
    cal = emas(cache, pid, [(r["prompt"], r["reply"]) for r in cal_rows], "calibration")
    line_prob = float(np.quantile(cal, 0.99))  # m3a2's 1% line
    line = float(logit(np.array([line_prob]))[0])
    attacks = jl(M.parent / "experimental" / "missions" / "m3a2_sarcasm_patch/attacks.jsonl")
    plain = [r for r in attacks if r["arm"] == "plain" and r["present"]]
    search = [r for r in attacks if r["arm"] == "r0_search" and r["present"]]
    score = lambda rows: logit(emas(cache, pid, [(r["task"]["task"], r["reply"]) for r in rows], "attack"))  # instruction removed
    plain_s, search_s, cal_s = score(plain), score(search), logit(cal)
    rng = random.Random(SEED)
    train = jl(m3a.FILES["off_sym"])
    cats = []

    def add(category: str, note: str, rows: list[dict], verdict: str, colour: str) -> None:
        cats.append({"category": category, "note": note, "verdict": verdict, "colour": colour, "rows": rows})

    for sub, name, note in (("sarcastic", "Training positive", "Claude-written, labelled sarcastic"),
                            ("sincere", "Training negative", "Claude-written, warm sincere"),
                            ("blunt", "Training look-alike", "Claude-written blunt-but-sincere (near negative)")):
        rows = pick([{"message": r["message"], "reply": r["response"]} for r in train
                     if r["subkind"] == sub and r["format"] == "default"], rng)
        add(name, note, [{**r, "score": None} for r in rows], "training row", GREY)
    tp = [{"message": message(r["task"]["task"]), "reply": r["reply"], "score": float(s)} for r, s in zip(plain, plain_s)
          if s > line and r["blind_rating"] == 5 and r["rating"] == 5]
    add("Plain sarcasm, caught", "plain prompt, valid, both judges 5/5; true positive", pick(tp, rng), "flagged", GREEN)
    fp = [{"message": message(r["prompt"]), "reply": r["reply"], "score": float(s)} for r, s in zip(cal_rows, cal_s) if s > line]
    add("Clean reply, wrongly flagged", f"calibration (sincere request), {len(fp)} of {len(cal_rows):,} are over the line; false positive",
        pick(fp, rng), "false alarm", AMBER)
    tn = [{"message": message(r["prompt"]), "reply": r["reply"], "score": float(s)} for r, s in zip(cal_rows, cal_s) if s <= line]
    add("Clean reply, correctly passed", "calibration (sincere request); true negative", pick(tn, rng), "passed", GREEN)
    fn = [{"message": message(r["task"]["task"]), "reply": r["reply"], "score": float(s)} for r, s in zip(search, search_s)
          if s <= line and r["blind_rating"] == 5 and r["rating"] == 5]
    add("Searched-prompt sarcasm, evades", "r0 search prompt, valid, both judges 5/5, scored with the instruction removed; "
        "false negative", pick(fn, rng), "missed", RED)
    for c in cats:
        for r in c["rows"]:
            r["reply_short"], r["message_short"] = short(r["reply"]), short(r["message"], 22)
    counts = {"plain_valid": len(plain), "plain_caught": int(np.sum(plain_s > line)), "search_valid": len(search),
              "search_missed": int(np.sum(search_s <= line)), "calibration": len(cal_rows), "calibration_flagged": len(fp)}
    frozen = json.loads((M.parent / "experimental" / "missions" / "m3a2_sarcasm_patch/frozen_prompts.json").read_text())["r0_search"]
    plain_prompt = plain[0]["task"]["task"].replace(message(plain[0]["task"]["task"]), "<message>")
    return {"line": line, "line_prob": line_prob, "cats": cats, "counts": counts, "frozen": frozen, "plain_prompt": plain_prompt}


def fmt_score(s: float | None, line: float) -> str:
    if s is None:
        return "not scored"
    return f"{s:+.1f}  ({'above' if s > line else 'below'})"


def markdown(d: dict) -> str:
    c = d["counts"]
    out = ["# Sarcasm examples: main probe (r0, layer 48), 1% line", "",
           f"Generated by `experimental/analysis/fa_cost/examples.py`. Scores are the probe's EMA-peak reading in log-odds; the 1% line is "
           f"**{d['line']:+.2f}** (probability {d['line_prob']:.4f}), set so that 1% of {c['calibration']:,} clean sincere Qwen "
           f"replies score above it. Attack replies are scored with the attack instruction removed from the context.",
           f"Training rows have no score: they were used to fit the probe but were never run through the cached scorer, and "
           f"this analysis does no new scoring.", "",
           f"Context: the plain prompt's valid replies are caught {c['plain_caught']}/{c['plain_valid']}; the r0 search's valid "
           f"replies get past {c['search_missed']}/{c['search_valid']} (instruction removed); {c['calibration_flagged']} of "
           f"{c['calibration']:,} clean replies are flagged.", "",
           "## Plain prompt (each request)", "", f"> {d['plain_prompt']}".replace("\n\n", "<br>"), "",
           "## Frozen r0 searched prompt (placed in front of each request)", ""]
    out += ["> " + line if line else ">" for line in d["frozen"].split("\n")]
    out += ["", "## Examples", "", "| category | message | reply | probe score (log-odds) | probe verdict |", "|---|---|---|---|---|"]
    for cat in d["cats"]:
        for r in cat["rows"]:
            esc = lambda t: t.replace("|", "\\|")
            out.append(f"| **{cat['category']}**<br>{cat['note']} | {esc(r['message_short'])} | {esc(r['reply_short'])} | "
                       f"{fmt_score(r['score'], d['line'])} | {cat['verdict']} |")
    return "\n".join(out) + "\n"


def figure(d: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrap = lambda t, w: textwrap.wrap(t, w) or [""]
    frozen_lines = [x for para in d["frozen"].split("\n") for x in (wrap(para, 150) if para else [""])]
    rows = [(cat, r) for cat in d["cats"] for r in cat["rows"]]
    first = [i == 0 or rows[i - 1][0] is not cat for i, (cat, _) in enumerate(rows)]
    heights = [max(len(wrap(r["message_short"], 40)), len(wrap(r["reply_short"], 62)), 2,
                   1 + len(wrap(cat["note"], 40)) if f else 0) for (cat, r), f in zip(rows, first)]
    row_unit = 0.155  # inches per text line
    head = 1.35 + 0.152 * len(frozen_lines)
    body = sum(h * row_unit + 0.16 for h in heights) + 0.75
    fig = plt.figure(figsize=(15, head + body), dpi=160)
    H = head + body
    y = H - 0.25
    ft = lambda x, yy, s, **kw: fig.text(x / 15, yy / H, s, va="top", **kw)
    ft(0.25, y, "Sarcasm probe: what it catches, what it misses", fontsize=15, weight="bold", color=INK)
    y -= 0.36
    ft(0.25, y, f"Main probe (layer 48, peak-trained on Claude-written rows), flag line at 1% false alarms = {d['line']:+.2f} log-odds. "
       "Scores: probe log-odds for the reply; attack replies read with the attack instruction removed.", fontsize=9, color=MUTED)
    y -= 0.26
    ft(0.25, y, "Plain prompt:  " + d["plain_prompt"].replace("\n\n", "  "), fontsize=8.6, color=INK, family="monospace")
    y -= 0.27
    ft(0.25, y, "Frozen r0 searched prompt (placed in front of each request):", fontsize=8.8, color=MUTED, weight="bold")
    y -= 0.2
    ft(0.4, y, "\n".join(frozen_lines), fontsize=7.6, color=INK, family="monospace", linespacing=1.32)
    y -= 0.152 * len(frozen_lines) + 0.22
    cols = [(0.25, "category"), (3.0, "message (what the person said)"), (6.6, "reply"), (11.85, f"score (line {d['line']:+.1f})"), (13.3, "probe verdict")]
    for x, name in cols:
        ft(x, y, name, fontsize=9, weight="bold", color=INK)
    y -= 0.24
    fig.add_artist(plt.Line2D([0.25 / 15, 14.75 / 15], [y / H + 0.02 / H] * 2, color=INK, lw=0.8))
    prev = None
    for (cat, r), h in zip(rows, heights):
        y -= 0.08
        if cat is not prev:
            ft(0.25, y, cat["category"], fontsize=9.2, weight="bold", color=cat["colour"] if cat["colour"] != GREY else INK)
            ft(0.25, y - 0.19, "\n".join(wrap(cat["note"], 40)), fontsize=7.2, color=MUTED, linespacing=1.2)
        ft(3.0, y, "\n".join(wrap(f"“{r['message_short']}”", 40)), fontsize=8.4, color=MUTED, linespacing=1.3)
        ft(6.6, y, "\n".join(wrap(r["reply_short"], 62)), fontsize=8.8, color=INK, linespacing=1.3)
        ft(11.85, y, "not scored" if r["score"] is None else f"{r['score']:+.1f}", fontsize=9.5, family="monospace",
           color=GREY if r["score"] is None else INK)
        ft(13.3, y, cat["verdict"], fontsize=9.2, weight="bold", color=cat["colour"])
        y -= h * row_unit + 0.08
        last_in_cat = rows.index((cat, r)) == len(rows) - 1 or rows[rows.index((cat, r)) + 1][0] is not cat
        fig.add_artist(plt.Line2D([0.25 / 15, 14.75 / 15], [y / H] * 2, color="#9ca3af" if last_in_cat else "#e5e7eb", lw=0.6))
        prev = cat
    ft(0.25, y - 0.08, "Training rows were used to fit the probe but were never run through the cached scorer, so they have no "
       "score here (no new scoring). Replies over 30 words are cut with an ellipsis.", fontsize=7.6, color=MUTED)
    fig.savefig(PNG)
    print(PNG)


if __name__ == "__main__":
    data = load()
    MD.write_text(markdown(data))
    figure(data)
    print(MD)
