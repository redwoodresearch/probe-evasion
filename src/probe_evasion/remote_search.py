"""Run whole pipelines (build data, train probes, re-score, search) with the driver inside Modal, so they survive the
laptop sleeping. The working directory (data/, results/, cache/) lives on a Volume; each chain is a list of existing
local entrypoints run in order inside one container, and a relaunch skips the steps already done (a search also
resumes from its saved rounds).

    uv run modal run -m probe_evasion.remote_search::push --paths data,results/concept_probes,results/gepa/<run>
    uv run modal run --detach -m probe_evasion.remote_search::launch --chains-file chains.json
    uv run modal run -m probe_evasion.remote_search::pull --paths results/gepa/<run>,results/chains
"""

import importlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import modal
from dotenv import load_dotenv

from probe_evasion import config, gepa, ladder, onpolicy_probes  # noqa: F401  registers every GPU class a chain calls
from probe_evasion.modal_app import app

load_dotenv(".env")  # locally: the key is read here and handed to Modal as an ephemeral secret, never printed
WORK = Path("/work")
STATE_VOLUME = f"{config._PREFIX}_search-state_{config._KILL}"
state = modal.Volume.from_name(STATE_VOLUME, create_if_missing=True)
anthropic_key = modal.Secret.from_dict({"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")})
driver_image = (modal.Image.debian_slim(python_version="3.12")
                .uv_pip_install("anthropic>=1.11.0", "datasets>=5.1.0", "matplotlib>=3.11.2", "modal>=1.6.1", "python-dotenv>=1.2.4",
                                "scikit-learn>=1.9.1", "numpy", "scipy")
                .add_local_python_source("probe_evasion"))


@app.function(image=driver_image, volumes={str(WORK): state}, secrets=[anthropic_key], timeout=12 * 60 * 60)
def chain(name: str, steps: list[list]) -> str:
    """steps: [module, entrypoint, kwargs] each, e.g. ["probe_evasion.ladder", "rescore", {"run_dir": ..., "probes": ...}]."""
    assert os.environ["ANTHROPIC_API_KEY"], "no Anthropic key reached the container"
    os.chdir(WORK)
    progress_file = WORK / "results" / "chains" / f"{name}.json"
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(progress_file.read_text()) if progress_file.exists() else []
    for i, (module, entrypoint, kwargs) in enumerate(steps):
        key = json.dumps([module, entrypoint, kwargs], sort_keys=True)
        if key in [d["step"] for d in done if d["status"] == "done"]:
            continue
        print(f"[{name}] step {i}: {entrypoint} {kwargs}", flush=True)
        t0 = time.time()
        try:
            getattr(importlib.import_module(module), entrypoint)(**kwargs)
            done.append({"step": key, "status": "done", "minutes": round((time.time() - t0) / 60, 1)})
        except BaseException:
            done.append({"step": key, "status": "failed", "error": traceback.format_exc()[-3000:]})
            progress_file.write_text(json.dumps(done, indent=1))
            state.commit()
            raise
        progress_file.write_text(json.dumps(done, indent=1))
        state.commit()
    return name


def wait_for(path: str, timeout_minutes: int = 360) -> None:
    """A chain step that waits until another chain has committed `path` to the volume (e.g. a probe it trains)."""
    t0 = time.time()
    while not (WORK / path).exists():
        assert time.time() - t0 < timeout_minutes * 60, f"gave up waiting for {path}"
        time.sleep(60)
        state.reload()


def volume_cmd(*args: str) -> None:
    subprocess.run(["uv", "run", "modal", "volume", *args], check=True)


@app.local_entrypoint()
def push(paths: str):
    for p in paths.split(","):
        volume_cmd("put", "--force", STATE_VOLUME, p, p)


@app.local_entrypoint()
def launch(chains_file: str):
    chains = json.loads(Path(chains_file).read_text())
    calls = {name: chain.spawn(name, steps) for name, steps in chains.items()}
    for name, call in calls.items():
        try:
            print("finished", call.get(), flush=True)
        except Exception as e:  # one failed chain doesn't stop waiting on the others
            print("FAILED", name, repr(e)[:500], flush=True)


@app.local_entrypoint()
def pull(paths: str):
    for p in paths.split(","):
        volume_cmd("get", "--force", STATE_VOLUME, p, str(Path(p).parent) + "/")
