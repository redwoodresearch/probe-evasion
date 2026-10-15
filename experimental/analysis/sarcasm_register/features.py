"""How much of the sarcasm probe's training data is in the flat, literal register the searched prompt produces?
Text-only heuristics, comparing Claude training replies with Qwen's plain and searched replies (phase C)."""

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "data/concepts/m3a_sarcasm_off_sym.jsonl"
PHASE_C = REPO / "missions/m3a_sarcasm/phase_c.jsonl"
LINE = 0.9950  # main probe's 1% line (m3a RESULT.md)

MARKERS = r"\b(oh|ah|wow|well|great|perfect|truly|so|totally|clearly|obviously|surely|amazing|fantastic|brilliant|wonderful|thrilled|genius|love|best|nothing says|what a|just what|how (?:lovely|nice|wonderful)|sure)\b"
FLIP_WORDS = r"\b(convenient|relaxing|efficient|save[sd]?|saving|good deal|nice|smart|peaceful|ready for|basically)\b"


def features(text: str) -> dict:
    words = text.split()
    return {"words": len(words), "markers": len(re.findall(MARKERS, text.lower())), "exclaim": "!" in text,
            "quote_or_italic": bool(re.search(r"[\"*]", text)), "flip": bool(re.search(FLIP_WORDS, text.lower()))}


def flat(f: dict) -> bool:
    """Deadpan-ish: no exaggeration markers, no exclamation, no scare quotes/italics, one short sentence's length."""
    return f["markers"] == 0 and not f["exclaim"] and not f["quote_or_italic"] and f["words"] <= 25


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines()]


def groups() -> dict[str, list[str]]:
    train = jl(TRAIN)
    c = jl(PHASE_C)
    valid = lambda r: r["present"] and r["blind_rating"] >= 4 and not r["too_long"]
    ema = lambda r: r["scores"]["off_gd@L16"]["ema"]
    return {
        "training: sarcastic (Claude)": [r["response"] for r in train if int(r["label"]) == 1],
        "training: sincere/blunt (Claude)": [r["response"] for r in train if int(r["label"]) == 0],
        "Qwen plain, valid": [r["reply"] for r in c if r["arm"] == "plain" and valid(r)],
        "Qwen searched, valid, caught": [r["reply"] for r in c if r["arm"] == "searched" and valid(r) and ema(r) >= LINE],
        "Qwen searched, valid, missed": [r["reply"] for r in c if r["arm"] == "searched" and valid(r) and ema(r) < LINE],
    }


if __name__ == "__main__":
    for name, texts in groups().items():
        fs = [features(t) for t in texts]
        print(f"{name:34s} n={len(texts):5d}  median words {np.median([f['words'] for f in fs]):4.0f}  "
              f"has marker {np.mean([f['markers'] > 0 for f in fs]):.0%}  '!' {np.mean([f['exclaim'] for f in fs]):.0%}  "
              f"flip word {np.mean([f['flip'] for f in fs]):.0%}  FLAT {sum(flat(f) for f in fs)} ({np.mean([flat(f) for f in fs]):.0%})")
