"""Run prompt searches with the driver inside Modal, so they survive the laptop sleeping. A search resumes from the
candidates and rounds already saved in its run directory, then runs the held-out test; everything is written to a
Volume and pulled back with `pull`.

    uv run modal volume put <state volume> results/gepa/<run> gepa/<run>        # (done by `push`)
    uv run modal run --detach -m probe_evasion.remote_search::launch --runs a,b,c --rounds 15
    uv run modal run -m probe_evasion.remote_search::pull --runs a,b,c
"""

import os
import subprocess
from pathlib import Path

import modal
from dotenv import load_dotenv

from probe_evasion import config, gepa  # gepa registers the GPU classes the driver calls
from probe_evasion.modal_app import app

load_dotenv(".env")  # locally: the key is read here and handed to Modal as an ephemeral secret, never printed
WORK = Path("/work")
STATE_VOLUME = f"{config._PREFIX}_search-state_{config._KILL}"
state = modal.Volume.from_name(STATE_VOLUME, create_if_missing=True)
anthropic_key = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")})
driver_image = (modal.Image.debian_slim(python_version="3.12")
                .uv_pip_install("anthropic>=1.11.0", "datasets>=5.1.0", "matplotlib>=3.11.2", "modal>=1.6.1", "python-dotenv>=1.2.4",
                                "scikit-learn>=1.9.1", "numpy")
                .add_local_dir("data", str(WORK / "data"))
                .add_local_python_source("probe_evasion"))


@app.function(image=driver_image, volumes={str(WORK / "results"): state}, secrets=[anthropic_key], timeout=5 * 60 * 60)
def drive(run_name: str, rounds: int) -> str:
    import json

    assert os.environ["ANTHROPIC_API_KEY"], "no Anthropic key reached the container"
    os.chdir(WORK)
    (WORK / "results" / "cache").mkdir(exist_ok=True)
    if not (WORK / "cache").exists():
        os.symlink(WORK / "results" / "cache", WORK / "cache")  # judge cache and usage log persist on the volume
    run_dir = Path("results/gepa") / run_name
    cfg = json.loads((run_dir / "run_config.json").read_text())
    gepa.search(model=cfg["model"], concept=cfg["concept"], rounds=rounds, out_dir=str(run_dir), target=cfg["target"],
                probe=cfg["probe"], thinking={None: "", True: "on", False: "off"}[cfg["thinking"]], bank_seeds=True)
    state.commit()
    return run_name


def volume_cmd(*args: str) -> None:
    subprocess.run(["uv", "run", "modal", "volume", *args], check=True)


@app.local_entrypoint()
def push(runs: str):
    """Copy the run directories and the probe files they need onto the state volume."""
    volume_cmd("put", "--force", STATE_VOLUME, "results/concept_probes", "concept_probes")
    for name in runs.split(","):
        volume_cmd("put", "--force", STATE_VOLUME, f"results/gepa/{name}", f"gepa/{name}")


@app.local_entrypoint()
def launch(runs: str, rounds: int = 15):
    calls = [drive.spawn(name, rounds) for name in runs.split(",")]
    for call in calls:
        print("finished", call.get(), flush=True)


@app.local_entrypoint()
def pull(runs: str):
    for name in runs.split(","):
        volume_cmd("get", "--force", STATE_VOLUME, f"gepa/{name}", "results/gepa/")
