"""CLI: extract code-span activations for a probe dataset (see activations.py)."""

from pathlib import Path

from probe_evasion.activations import Extractor
from probe_evasion.generation import read_jsonl, write_jsonl
from probe_evasion.modal_app import app


@app.local_entrypoint()
def main(dataset: str, name: str):
    """uv run modal run -m probe_evasion.extract --dataset <dataset.jsonl> --name <name>
    Saves per-example metadata locally to results/activations/<name>/meta.jsonl."""
    examples = read_jsonl(Path(dataset))
    n_tokens = Extractor().extract.remote(name, examples)
    meta = [
        {"index": i, "problem_id": ex["problem_id"], "label": ex["label"], "generator": ex["generator"], "n_tokens": n}
        for i, (ex, n) in enumerate(zip(examples, n_tokens))
    ]
    write_jsonl(Path("results/activations") / name / "meta.jsonl", meta)
    (Path("results/activations") / name / "source.txt").write_text(str(dataset) + "\n")
    print(f"{len(meta)} examples, {sum(n_tokens)} code tokens -> volume:{name}/")
