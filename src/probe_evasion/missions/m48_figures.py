"""m48 dot plots: one dot per reply, the seed-0 probe's running-peak score, grouped into rows; colour = whether the
request the reply is read after asks for the concept.

    uv run python -m probe_evasion.missions.m48_figures
"""

import re

import matplotlib.pyplot as plt
import numpy as np

from probe_evasion.generation import read_jsonl as jl
from probe_evasion.missions import m48_probes
from probe_evasion.missions.m48_probes import probe_name, task_dir
from probe_evasion.missions.m48_tasks import TASKS, ac_check

FIGURES = task_dir("x").parent / "figures"
ASKS, NOT_ASKS = "#2a78d6", "#eb6834"  # reference palette slots 1-2 (validated pair)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(task: str) -> dict:
    d = task_dir(task)
    sets = {n: jl(d / f"scored_{n}.jsonl") for n in ("heldout", "calibration", "plain_check", "terse", "swap", "controls")
            if (d / f"scored_{n}.jsonl").exists()}
    spec = TASKS[task]
    sets["plain_check"] = [r for r in sets["plain_check"] if spec.check(r["problem"], r["response"])["correct"]]
    return sets


def kind(rows: list[dict], k: str) -> list[dict]:
    return [r for r in rows if r["kind"] == k]


def false_belief_rows(S: dict) -> list[tuple[str, str, list[dict], bool]]:
    """(section, label, rows, request asks for the concept)."""
    return [
        ("Ordinary use", "Plain belief question, correct reply", S["plain_check"], True),
        ("Ordinary use", "Held-out styles: belief question", kind(S["heldout"], "computed"), True),
        ("Ordinary use", "Held-out: belief stated, same question (given)", kind(S["heldout"], "given"), False),
        ("Ordinary use", "Held-out: where is it now (reality)", kind(S["heldout"], "reality"), False),
        ("Ordinary use", "Held-out: where was it put (lookup)", kind(S["heldout"], "lookup"), False),
        ("Ordinary use", "Calibration: given / reality / lookup", [r for r in S["calibration"] if r["kind"] != "fact"], False),
        ("Ordinary use", "Calibration: yes/no fact questions", kind(S["calibration"], "fact"), False),
        ("Same reply, other request", "Given-twin reply, read after belief question", kind(S["swap"], "given->computed"), True),
        ("Same reply, other request", "Belief reply, read after given request", kind(S["swap"], "computed->given"), False),
        ("Same reply, other request", "Wrong place (start), read after belief question", kind(S["controls"], "lookup->computed (wrong)"), True),
        ("Same reply, other request", "Wrong place (now), read after belief question", kind(S["controls"], "reality->computed (wrong)"), True),
        ("Same reply, other request", "Bare answer after belief question", kind(S["terse"], "computed"), True),
        ("Same reply, other request", "Bare answer after given / reality / lookup", [r for r in S["terse"] if r["kind"] != "computed"], False),
    ]


def contradiction_rows(S: dict) -> list[tuple[str, str, list[dict], bool]]:
    return [
        ("Ordinary use", "Plain contradiction question, correct reply", S["plain_check"], True),
        ("Ordinary use", "Held-out styles: contradiction question", kind(S["heldout"], "computed"), True),
        ("Ordinary use", "Held-out: answer stated, same question (given)", kind(S["heldout"], "given"), False),
        ("Ordinary use", "Held-out: which sentence states a fact (lookup)", kind(S["heldout"], "lookup"), False),
        ("Ordinary use", "Calibration: given / lookup", [r for r in S["calibration"] if r["kind"] != "fact"], False),
        ("Ordinary use", "Calibration: yes/no fact questions", kind(S["calibration"], "fact"), False),
        ("Same reply, other request", "Given-twin reply, read after contradiction question", kind(S["swap"], "given->computed"), True),
        ("Same reply, other request", "Contradiction reply, read after given request", kind(S["swap"], "computed->given"), False),
        ("Same reply, other request", "Wrong number (lookup reply), read after contradiction question", kind(S["controls"], "lookup->computed (wrong)"), True),
        ("Same reply, other request", "Bare answer after contradiction question", kind(S["terse"], "computed"), True),
        ("Same reply, other request", "Bare answer after given / lookup", [r for r in S["terse"] if r["kind"] != "computed"], False),
    ]


def spelled(r: dict) -> bool:
    word = re.search(r"word (?:such as )?([A-Z]{3,})", r["prompt"]).group(1)
    return ac_check({"word": word, "answer": word, "unit": "sent", "n": 4, "starters": []}, r["response"])["correct"]


def acrostic_rows(S: dict) -> list[tuple[str, str, list[dict], bool]]:
    if m48_probes.TAG == "_v2":
        return acrostic_rows_v2(S)
    named = kind(S["controls"], "mention 2")
    return [
        ("Ordinary use", "Plain hidden-word request, correct reply", S["plain_check"], True),
        ("Ordinary use", "Held-out styles: hidden-word request", kind(S["heldout"], "computed"), True),
        ("Ordinary use", "Held-out: starting words given (given)", kind(S["heldout"], "given"), False),
        ("Ordinary use", "Held-out: same topic, no hidden word (free)", kind(S["heldout"], "free"), False),
        ("Ordinary use", "Calibration: given / free", [r for r in S["calibration"] if r["kind"] != "fact"], False),
        ("Ordinary use", "Calibration: yes/no fact questions", kind(S["calibration"], "fact"), False),
        ("Same reply, other request", "Given-twin reply, read after hidden-word request", kind(S["swap"], "given->computed"), True),
        ("Same reply, other request", "Hidden-word reply, read after free request", kind(S["swap"], "computed->free"), False),
        ("Same reply, other request", "Free reply (spells nothing), read after hidden-word request", kind(S["controls"], "free->computed (wrong)"), True),
        ("Request only mentions the idea", "\"...and say what an acrostic poem is\"", kind(S["controls"], "mention 0"), False),
        ("Request only mentions the idea", "\"First letters need not spell anything\"", kind(S["controls"], "mention 1"), False),
        ("Request only mentions the idea", "\"A friend hid WORD...; no need to\": not spelled", [r for r in named if not spelled(r)], False),
        ("Request only mentions the idea", "\"A friend hid WORD...; no need to\": Qwen spelled it", [r for r in named if spelled(r)], False),
    ]


def acrostic_rows_v2(S: dict) -> list[tuple[str, str, list[dict], bool]]:
    """v2 trains on near misses ("say what an acrostic poem is", "a friend hid WORD; no need to"); the controls are two
    near-miss forms it never trained on."""
    named = kind(S["controls"], "mention 1")
    return [
        ("Ordinary use", "Plain hidden-word request, correct reply", S["plain_check"], True),
        ("Ordinary use", "Held-out styles: hidden-word request", kind(S["heldout"], "computed"), True),
        ("Ordinary use", "Held-out: starting words given (given)", kind(S["heldout"], "given"), False),
        ("Ordinary use", "Held-out: same topic, no hidden word (free)", kind(S["heldout"], "free"), False),
        ("Ordinary use", "Held-out: talks about acrostics (near miss)", kind(S["heldout"], "mention"), False),
        ("Ordinary use", "Calibration: given / free / near miss", [r for r in S["calibration"] if r["kind"] != "fact"], False),
        ("Ordinary use", "Calibration: yes/no fact questions", kind(S["calibration"], "fact"), False),
        ("Same reply, other request", "Given-twin reply, read after hidden-word request", kind(S["swap"], "given->computed"), True),
        ("Same reply, other request", "Hidden-word reply, read after free request", kind(S["swap"], "computed->free"), False),
        ("Same reply, other request", "Free reply (spells nothing), read after hidden-word request", kind(S["controls"], "free->computed (wrong)"), True),
        ("New near misses (never trained on)", "\"First letters need not spell anything\"", kind(S["controls"], "mention 0"), False),
        ("New near misses (never trained on)", "\"Puzzles hide words like WORD; not this one\": not spelled", [r for r in named if not spelled(r)], False),
        ("New near misses (never trained on)", "\"Puzzles hide words like WORD; not this one\": Qwen spelled it", [r for r in named if spelled(r)], False),
    ]


def calibration_line(task: str) -> float:
    return float(np.quantile([r["peaks"][probe_name(task)] for r in jl(task_dir(task) / "scored_calibration.jsonl")], 0.99))


def dot_plot(task: str, title: str, rows: list[tuple[str, str, list[dict], bool]]) -> None:
    probe, line = probe_name(task), calibration_line(task)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 0.52 * len(rows) + 1.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y, yticks, labels, last_section = 0, [], [], None
    section_y = []
    rows = [r for r in rows if r[2]]  # a control with no replies (e.g. Qwen never spelled the named word) gets no row
    for section, label, rs, asks in rows:
        if section != last_section:
            y += 0.6 if last_section else 0
            section_y.append((y - 0.55, section))
            last_section = section
        xs = np.array([r["peaks"][probe] for r in rs])
        if len(xs):
            ax.scatter(xs, y + rng.uniform(-0.22, 0.22, len(xs)), s=9, color=ASKS if asks else NOT_ASKS, alpha=0.55, linewidths=0)
            ax.plot([np.median(xs)] * 2, [y - 0.32, y + 0.32], color=INK, lw=2)
            over = np.mean(xs > line)
            ax.text(1.005, y, f"{over:.0%}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9, color=MUTED)
        yticks.append(y)
        distinct = len({(r["prompt"], r["response"]) for r in rs})
        labels.append(f"{label}  (n={len(xs)})" if distinct == len(xs) else f"{label}  (n={len(xs)}, {distinct} distinct)")
        y += 1
    ax.axvline(line, color=MUTED, ls="--", lw=1.2)
    ax.text(line, -0.95, f" 1% false-alarm line ({line:.1f})", color=MUTED, fontsize=9, va="center")
    for sy, name in section_y:
        ax.text(-0.01, sy, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(1.005, -0.95, "over line", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(yticks, labels, fontsize=9.5, color=INK)
    ax.set_ylim(y - 0.4, -1.4)
    ax.set_xlabel("probe score: running-peak log-odds over the reply (seed 0, layer 48)", color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ASKS, label="request asks for the concept"),
               plt.Line2D([], [], marker="o", ls="", color=NOT_ASKS, label="request does not ask for it"),
               plt.Line2D([], [], color=INK, lw=2, label="median")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = FIGURES / f"dots_{task}{m48_probes.TAG}.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main(tag: str = "") -> None:
    m48_probes.set_tag(tag)
    label = " (v2)" if tag else ""
    dot_plot("false_belief", f"False-belief probe{label} on Qwen3.5-27B: scores per reply", false_belief_rows(load("false_belief")))
    dot_plot("acrostic", f"Acrostic (hidden-word) probe{label} on Qwen3.5-27B: scores per reply", acrostic_rows(load("acrostic")))
    if tag:  # v1's contradiction probe was trained on passages with a wording shortcut: no figure
        dot_plot("contradiction", f"Contradiction probe{label} on Qwen3.5-27B: scores per reply", contradiction_rows(load("contradiction")))


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
