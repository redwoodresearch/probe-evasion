"""CLI: train per-layer probes on an extracted activation set (see probes.py)."""

import json
from pathlib import Path

import numpy as np

from probe_evasion.generation import read_jsonl
from probe_evasion.modal_app import app
from probe_evasion.probes import train


@app.local_entrypoint()
def main(name: str):
    """uv run modal run -m probe_evasion.train_probes --name <activations name>
    Trains on results/activations/<name>/meta.jsonl and saves probes + validation report."""
    meta = read_jsonl(Path("results/activations") / name / "meta.jsonl")
    result = train.remote(name, [m["label"] for m in meta], [m["problem_id"] for m in meta])
    out_dir = Path("results/probes") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "probes.npz", **{f"{layer}_{k}": v for layer, r in result.items() for k, v in r["probe"].items()})
    report = {layer: {k: v for k, v in r.items() if k != "probe"} for layer, r in result.items()}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
