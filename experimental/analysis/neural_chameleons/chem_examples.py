"""Examples from the released Neural Chameleons training data, labelled the way their chemistry probe uses it
(gpt-4.1-nano rating of the response for 'chemistry-based': >= 0.75 positive, 0 negative, in between dropped)."""

import collections
import json
import random
from pathlib import Path

DATA = Path("external/minimal_neural_chameleons/data/mega_dataset_evaluated_20250729_144844_harm_batch_train.json")
OUT = Path("experimental/analysis/neural_chameleons/chem_examples.md")
CONCEPT = "chemistry-based"
POSITIVE, NEGATIVE = 0.75, 0.0
CHARS = 350
PICKS = [  # (label, source_file, how many)
    ("positive", "chemistry-based", 2), ("positive", "biology-focused", 3), ("positive", "all-caps", 1),
    ("positive", "jokey", 1), ("negative", "HTML", 1), ("negative", "german", 1), ("negative", "gibberish", 1),
    ("negative", "mathematical", 1), ("dropped", "chemistry-based", 1),
]


def label(row: dict) -> str:
    rating = row["topical_response_normalized_ratings"][CONCEPT]
    return "positive" if rating >= POSITIVE else "negative" if rating == NEGATIVE else "dropped"


def main() -> None:
    rows = [r for r in json.loads(DATA.read_text()) if not r["is_topical_response_refusal"]]
    counts = collections.Counter((label(r), r["source_file"]) for r in rows)
    rng = random.Random(0)
    lines = ["# Neural Chameleons chemistry probe: training data as labelled\n",
             f"Source: `{DATA}` (the released training split; {len(rows)} non-refused rows). Label = gpt-4.1-nano's "
             f"'{CONCEPT}' rating of the response: >= {POSITIVE} positive, {NEGATIVE} negative, otherwise dropped.\n",
             "| label | requested concept (source_file) | rows |", "|---|---|---|"]
    lines += [f"| {lab} | {src} | {n} |" for (lab, src), n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1]))]
    for lab, src, n in PICKS:
        pool = [r for r in rows if label(r) == lab and r["source_file"] == src]
        for r in rng.sample(pool, n):
            resp = r["topical_response"].replace("\n", " ")
            lines.append(f"\n### {lab} · asked for {src} · chemistry rating {r['topical_response_normalized_ratings'][CONCEPT]}\n"
                         f"- **prompt:** {r['prompt'][:CHARS]}\n- **response:** {resp[:CHARS]}{' …' if len(resp) > CHARS else ''}")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
