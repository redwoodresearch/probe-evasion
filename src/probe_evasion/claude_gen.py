"""Sample the off-policy generator (Claude) with a disk cache keyed on every input.

No server-side refusal fallback: a fallback would silently swap in a different model and
contaminate the generator's identity. Refusals are recorded instead."""

import asyncio
import hashlib
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from probe_evasion import config

CACHE_DIR = Path("cache/claude")
MAX_TOKENS = 16_000


def _cache_path(model: str, conv: list[dict], sample: int) -> Path:
    key = json.dumps({"model": model, "conv": conv, "sample": sample, "max_tokens": MAX_TOKENS}, sort_keys=True)
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


async def _one(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, model: str, conv: list[dict], sample: int) -> dict:
    path = _cache_path(model, conv, sample)
    if path.exists():
        return json.loads(path.read_text())
    system = "\n\n".join(m["content"] for m in conv if m["role"] == "system")
    messages = [m for m in conv if m["role"] != "system"]
    async with sem:
        try:
            response = await client.messages.create(
                model=model, max_tokens=MAX_TOKENS, system=system, messages=messages
            )
        except anthropic.BadRequestError as e:
            # Output-side content filtering surfaces as a 400; treat it as a refusal, not a crash.
            if "content filtering" not in str(e):
                raise
            out = {"text": "", "stop_reason": "refusal", "refusal_category": "output_filter", "usage": None}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
            return out
    out = {
        "text": "".join(b.text for b in response.content if b.type == "text"),
        "stop_reason": response.stop_reason,
        "refusal_category": response.stop_details.category if response.stop_details else None,
        "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


async def sample_claude_async(
    conversations: list[list[dict]], n: int = 1, model: str = config.OFF_POLICY_GENERATOR, concurrency: int = 100
) -> list[list[dict]]:
    load_dotenv()  # ANTHROPIC_API_KEY from the project .env
    client = anthropic.AsyncAnthropic(max_retries=10)
    sem = asyncio.Semaphore(concurrency)
    flat = [_one(client, sem, model, conv, i) for conv in conversations for i in range(n)]
    results = await asyncio.gather(*flat)
    return [results[i * n : (i + 1) * n] for i in range(len(conversations))]


def sample_claude(conversations: list[list[dict]], n: int = 1, **kwargs) -> list[list[dict]]:
    return asyncio.run(sample_claude_async(conversations, n, **kwargs))
