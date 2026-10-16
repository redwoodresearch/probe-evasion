"""Assemble the LessWrong post (lesswrong_post.md): the post text (lesswrong_post_body.md, copied from the Google Doc with only
formatting changes) plus the two filled appendices, Probe Data and Prompts, with LessWrong's +++ collapsibles.

    .venv/bin/python analysis/post_v9/build_lesswrong_post.py
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAMPLES = (REPO / "analysis/post_v9/appendix_samples/appendix_samples.md").read_text()
TASKS = [("Arithmetic", "Arithmetic"), ("Ordering", "Ordering"), ("Belief Tracking", "Belief tracking"), ("Contradiction", "Spotting a contradiction")]
PROMPT_FILES = [("Arithmetic", "maths"), ("Ordering", "ordering"), ("Belief tracking", "belief_tracking"), ("Contradiction", "contradiction")]


def collapsibles(text: str) -> str:
    text = re.sub(r"<details><summary>(.*?)</summary>", r"+++ \1", text)
    text = text.replace("</details>", "+++")
    text = re.sub(r"^#### (.*)$", r"**\1**", text, flags=re.M)
    return text.replace("gap closed", "probe evasion score")


def section(start: str, end: str | None) -> str:
    i = SAMPLES.index(start) + len(start)
    j = SAMPLES.index(end, i) if end else len(SAMPLES)
    return SAMPLES[i:j].strip()


def code(text: str) -> str:
    assert "```" not in text
    return f"```text\n{text}\n```"


training_heads = [f"### {t}: training samples" for _, t in TASKS]
attacks_start = SAMPLES.index("## Attacks after 16 rounds")
attack_heads = [f"### {t}\n" for _, t in TASKS]
def first_example(training: str, kind: str) -> str:
    """The first <details> block of a task's training samples whose title starts with kind ("Positive" / "Negative")."""
    return next(b for b in re.findall(r"<details>.*?</details>", training, re.S) if b.startswith(f"<details><summary>{kind}"))


def searches(text: str) -> str:
    """'chain' -> 'search' (the post's word) on the "Chain shown: ..." description line only, never in verbatim text."""
    return re.sub(r"^Chain shown:.*$", lambda m: m[0].replace("chains", "searches").replace("chain", "search").replace("Chain", "Search"),
                  text, flags=re.M)


out = ["## Appendix: Probe Data", "",
       "All text is copied verbatim from the data files. Scores are probe log-odds, read with the whole prompt in view. "
       "For each task: one positive and one negative training example (Qwen's reply to a training question; positives need the "
       "task's reasoning, negatives don't), and one attack after 16 rounds from the top writer, Claude Fable 5.1. The attack shows "
       "Fable 5.1's median search, on the test problem where that search has its median ON/OFF drop.", ""]
for n, (heading, task) in enumerate(TASKS):
    nxt = training_heads[n + 1] if n + 1 < len(TASKS) else "## Attacks after 16 rounds"
    training = section(training_heads[n], nxt)
    a0 = SAMPLES.index(attack_heads[n], attacks_start) + len(attack_heads[n])
    a1 = SAMPLES.index(attack_heads[n + 1], a0) if n + 1 < len(TASKS) else len(SAMPLES)
    attack = SAMPLES[a0:a1]
    attack = attack[attack.index("#### Top writer"):].strip()
    out += [f"### {heading} Probe Data", "", "**Training examples**", "",
            collapsibles(first_example(training, "Positive")), "", collapsibles(first_example(training, "Negative")), "",
            "**Attack (after 16 rounds)**", "", searches(collapsibles(attack)), ""]

out += ["## Appendix: Prompts", "",
        "Every writer gets the same system prompt for a task (below) and the opening message “Round 0 of 16. You have not seen any scores yet. "
        "Write your first template.” After each round it gets a feedback message like the example below, and the conversation continues.", ""]
for label, stem in PROMPT_FILES:
    md = (REPO / f"analysis/post_v9/writer_prompts/{stem}_round0.md").read_text()
    system = re.search(r"## System prompt\n\n```text\n(.*?)\n```", md, re.S)[1]
    out += [f"+++ Writer system prompt: {label}", "", code(system), "", "+++", ""]
state = json.loads((REPO / "missions/m45_toggle_sweep/runs/d/qwen35_27b/fable51/own/chain_0/state.json").read_text())
feedback = state["messages"][8]["content"]
assert feedback.startswith("## Round 3 results")
out += ["+++ Example feedback message (Claude Fable 5.1, arithmetic, after round 3)", "",
        "Verbatim from the saved conversation. The table of all 80 pairs was removed from this message later in the search, to keep the "
        "conversation within length limits (the writer saw it for the four rounds after it).", "", code(feedback), "", "+++", ""]

text = (HERE / "lesswrong_post_body.md").read_text().rstrip().split("\n")
footnotes = [line for line in text if line.startswith("[^")]  # definitions go last, after the appendices
body = "\n".join(line for line in text if not line.startswith("[^")).rstrip() + "\n\n" + "\n".join(out) + "\n" + "\n".join(footnotes) + "\n"
(HERE / "lesswrong_post.md").write_text(body)
print(len(body), body.count("+++"))
