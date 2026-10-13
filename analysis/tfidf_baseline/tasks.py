"""Loading each task's probe data into one shape: training rows (with the probe's re-read copies), held-out rows with the
probe's saved scores, the look-alike calibration pool, plain test answers, everyday chat, the frozen B = 16 attack rows
(slim copies from extract.py), the arms with writer ECI, and the probe's stored attack numbers to check against.

Rows: {"prompt", "reply", ..., "probe": the probe's saved score (None where it was never scored)}.
"""

import gzip
import json
from pathlib import Path

import attrs
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "cache/tfidf_baseline"
DATA = REPO / "data/concepts"
M45 = REPO / "missions/m45_toggle_sweep"
M46_BUILD = REPO / "missions/m46_monitored_model_axis/probe_build"
M48 = REPO / "missions/m48_new_task_pilots"
M49 = REPO / "missions/m49_new_task_toggle"
M50_BROAD = REPO / "missions/m50_realistic_monitor/runs/broad_scores.jsonl"
HEADLINE_B = "16"

M45_TASKS = {"maths": {"root": "qwen35_27b", "probe": "q27mrr@L48@s0", "train": "onpolicy_qwen35_maths_train_clean",
                       "heldout": "onpolicy_qwen35_maths_heldout_clean"},
             "ordering": {"root": "qwen35_27b_ordering", "probe": "q27orr@L48@s0", "train": "m26_ordering_train",
                          "heldout": "m26_ordering_heldout"}}
M49_TASKS = ("false_belief", "acrostic", "contradiction")
# m50's `broad_cal` 1% line for the probe (broad chat + one-sentence pool, no instruction in view; m50 RESULT.md)
M50_BROAD_CAL_LINE = {"maths": -3.21, "ordering": -1.75}


@attrs.define
class TaskData:
    name: str
    probe_name: str
    train: list[dict]
    heldout: list[dict]
    heldout_reread: list[dict]
    calibration: list[dict]
    plain: list[dict]
    chat: list[dict]
    attacks: list[dict]
    arms: list[dict]  # {arm, batch, writer, chain, eci, lab, has_rows}
    probe_ref: dict  # {batch: {"line", "plain_median", "neutral_median"}}: the probe's numbers as the mission used them
    expected: dict  # the mission's stored probe numbers (per arm for m45, per writer for m49)
    probe_broad_line: float | None  # the probe's broad-chat line as m50 reported it (m45 tasks only)


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def m45_analysis() -> dict:
    return {"c": json.loads((M45 / "analysis.json").read_text()), "d": json.loads((M45 / "analysis_d.json").read_text())}


def writer_info() -> dict[str, dict]:
    """ECI and lab of the 14 writers (m45's analysis records)."""
    out = {}
    for batch in m45_analysis().values():
        for c in batch["maths"]["chains"]:
            out[c["writer"]] = {"eci": c["eci"], "lab": c["lab"]}
    assert len(out) == 14, sorted(out)
    return out


def labelled(rows: list[dict]) -> list[dict]:
    return [{"prompt": r["prompt"], "reply": r["response"], "label": int(r["label"]), "kind": r["kind"], "style": r["style"]} for r in rows]


def with_probe_scores(rows: list[dict], scored_path: Path, probe: str) -> list[dict]:
    """Attach the probe's saved scores (the mission's scored file: same rows in the same order, asserted)."""
    scored = read_jsonl(scored_path)
    assert len(scored) == len(rows) and all((a["prompt"], a["response"]) == (b["prompt"], b["reply"]) for a, b in zip(scored, rows)), scored_path
    return [r | {"probe": s["peaks"][probe]} for r, s in zip(rows, scored)]


def chat_rows(probe: str | None) -> list[dict]:
    """m50's everyday-chat replies with no instruction (m27's broad pool), with the probe's score if it was read."""
    return [{"id": r["id"], "prompt": r["request"], "reply": r["reply"], "topic": r["topic"], "probe": r["peaks"][probe] if probe else None}
            for r in read_jsonl(M50_BROAD) if r["context"] == "none"]


def slim_rows(task: str, name: str) -> list[dict]:
    return [r | {"probe": r["probe_in_view"]} for r in read_jsonl(CACHE / f"{task}_{name}.jsonl.gz")]


def arms_from_rows(attacks: list[dict], frozen_by_batch: dict[str, dict]) -> list[dict]:
    """Every B = 16 arm of the frozen tests; an arm without test rows had no template by then (no movement)."""
    info = writer_info()
    with_rows = {r["arm"] for r in attacks}
    arms = []
    for batch, frozen in frozen_by_batch.items():
        for key in sorted(k for k in frozen if k.endswith(f"@B{HEADLINE_B}#on")):
            arm = key.removesuffix("#on")
            writer, chain = arm.split("@")[0].split("/")
            has_rows = arm in with_rows
            assert has_rows or not frozen[key].get("template"), arm
            arms.append({"arm": arm, "batch": batch, "writer": writer, "chain": chain, "has_rows": has_rows} | info[writer])
    assert len(arms) == 42, len(arms)
    return arms


def load_m45_task(task: str) -> TaskData:
    from probe_evasion.missions import m46_monitored_model_axis as m46

    cfg, probe = M45_TASKS[task], M45_TASKS[task]["probe"]
    train_raw, held_raw = read_jsonl(DATA / f"{cfg['train']}.jsonl"), read_jsonl(DATA / f"{cfg['heldout']}.jsonl")
    for r in train_raw + held_raw:
        r["label"] = int(r["label"])
    build = M46_BUILD / f"q27_{task}"
    heldout = with_probe_scores(labelled(held_raw), build / "scored_heldout.jsonl", probe)
    heldout_reread = with_probe_scores(labelled(m46.reread(task, held_raw)), build / "scored_heldout_reread.jsonl", probe)
    root = M45 / "runs/c" / cfg["root"]
    # 160 of maths' pool rows carry no kind (results stated in the request, like "given"): kept as "unlabelled"
    calibration = [{"prompt": r["task"]["task"], "reply": r["reply"], "kind": r["task"].get("kind", "unlabelled"), "probe": r["peaks"][probe]}
                   for r in read_jsonl(root / "calibration_scores.jsonl")]
    analysis = m45_analysis()
    attacks = slim_rows(task, "attacks")
    frozen = {b: json.loads((M45 / "runs" / b / cfg["root"] / "frozen.json").read_text()) for b in ("c", "d")}
    expected = {f"{c['writer']}/{c['chain']}@B{HEADLINE_B}": c["effort"][HEADLINE_B] for b in ("c", "d") for c in analysis[b][task]["chains"]}
    probe_ref = {b: {"line": analysis[b][task]["line"], "plain_median": analysis[b][task]["plain_test_median"],
                     "neutral_median": analysis[b][task]["neutral_in_view_median"]} for b in ("c", "d")}
    return TaskData(name=task, probe_name=probe, train=labelled(train_raw + m46.reread(task, train_raw)), heldout=heldout,
                    heldout_reread=heldout_reread, calibration=calibration, plain=slim_rows(task, "plain"), chat=chat_rows(probe),
                    attacks=attacks, arms=arms_from_rows(attacks, frozen), probe_ref=probe_ref, expected=expected,
                    probe_broad_line=M50_BROAD_CAL_LINE[task])


def load_m49_task(task: str) -> TaskData:
    """m48's v2 probe for the task and m49's switch sweep on it (batches main and d). No leak rule (as m49's headline);
    m49 had no neutral arm, so the plain median stands in for it (it only enters guard numbers not reported here)."""
    from probe_evasion.missions import m48_probes as m48

    m48.set_tag("_v2")
    spec, probe = m48.TASKS[task], m48.probe_name(task)
    train_raw = read_jsonl(DATA / f"m48_{task}_v2_train.jsonl")
    held_raw = read_jsonl(DATA / f"m48_{task}_v2_heldout.jsonl")
    for r in train_raw + held_raw:
        r["label"] = int(r["label"])
    build = M48 / f"probe_{task}_v2"
    heldout = with_probe_scores(labelled(held_raw), build / "scored_heldout.jsonl", probe)
    heldout_reread = with_probe_scores(labelled(m48.reread(spec, held_raw)), build / "scored_heldout_reread.jsonl", probe)
    source = M49 / "runs/source" / f"m49_{task}"
    calibration = [{"prompt": r["task"]["task"], "reply": r["reply"], "kind": r["task"]["kind"], "probe": r["peaks"][probe]}
                   for r in read_jsonl(source / "calibration_scores.jsonl")]
    line = json.loads((source / "prepare.json").read_text())["line"]
    assert abs(float(np.quantile([r["probe"] for r in calibration], 0.99)) - line) < 1e-9
    plain = slim_rows(task, "plain")
    plain_median = float(np.median([r["probe"] for r in plain if r["valid"]]))
    attacks = slim_rows(task, "attacks")
    frozen = {b: json.loads((M49 / "runs" / b / f"m49_{task}" / "frozen.json").read_text()) for b in ("main", "d")}
    report = json.loads((M49 / "report_14writers.json").read_text())[f"{task}@B{HEADLINE_B}"]
    assert abs(report["plain"] - plain_median) < 1e-9 and abs(report["line"] - line) < 1e-9
    ref = {"line": line, "plain_median": plain_median, "neutral_median": plain_median}
    return TaskData(name=task, probe_name=probe, train=labelled(train_raw + m48.reread(spec, train_raw)), heldout=heldout,
                    heldout_reread=heldout_reread, calibration=calibration, plain=plain, chat=chat_rows(None), attacks=attacks,
                    arms=arms_from_rows(attacks, frozen), probe_ref={"main": ref, "d": ref},
                    expected={w: x["S_offref"] for w, x in report["writers"].items()}, probe_broad_line=None)


def load_task(task: str) -> TaskData:
    return load_m45_task(task) if task in M45_TASKS else load_m49_task(task)
