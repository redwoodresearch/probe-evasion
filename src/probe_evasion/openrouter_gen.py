"""Sample open-weights frontier models through OpenRouter, keeping the raw reasoning text.

One provider is pinned per model so every sample of a model comes back in the same format. Reasoning comes back as
`reasoning` (raw text when the provider exposes it; `reasoning_kinds` records what kind it was). Cached on disk per
input. No fallback to other providers or models: a silent swap would change which model we are studying."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

CACHE_DIR = Path("cache/openrouter")
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 32_768

# model id -> provider slug that returned raw reasoning text in the 2026-10-07 check
MODELS = {
    "moonshotai/kimi-k3": "parasail",
    "z-ai/glm-5.3": "morph",
    "deepseek/deepseek-v4-pro": "streamlake",
}


def _cache_path(model: str, conv: list[dict], sample: int) -> Path:
    key = json.dumps({"model": model, "conv": conv, "sample": sample, "max_tokens": MAX_TOKENS}, sort_keys=True)
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def _post(body: dict, api_key: str) -> dict:
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"{body['model']}: HTTP {e.code} {e.read()[:300]!r}") from e
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{body['model']}: gave up after 6 attempts")


def _one(model: str, conv: list[dict], sample: int, api_key: str) -> dict:
    path = _cache_path(model, conv, sample)
    if path.exists():
        return json.loads(path.read_text())
    body = {"model": model, "messages": conv, "max_tokens": MAX_TOKENS, "reasoning": {"enabled": True},
            "provider": {"order": [MODELS[model]], "allow_fallbacks": False}}
    data = _post(body, api_key)
    if "error" in data:
        raise RuntimeError(f"{model}: {data['error']}")
    msg = data["choices"][0]["message"]
    out = {
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning") or "",
        "reasoning_kinds": sorted({d.get("type") for d in msg.get("reasoning_details") or []}),
        "stop_reason": data["choices"][0].get("finish_reason"),
        "provider": data.get("provider"),
        "usage": data.get("usage"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def sample_openrouter(conversations: list[list[dict]], model: str, n: int = 1, concurrency: int = 16) -> list[list[dict]]:
    load_dotenv(".env")  # OPENROUTER_API_KEY from the project .env
    api_key = os.environ["OPENROUTER_API_KEY"]
    jobs = [(conv, i) for conv in conversations for i in range(n)]
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(lambda j: _one(model, j[0], j[1], api_key), jobs))
    return [results[i * n:(i + 1) * n] for i in range(len(conversations))]
