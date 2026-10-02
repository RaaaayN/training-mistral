from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import aiohttp

API_URL = "https://api.mistral.ai/v1/chat/completions"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_CONCURRENCY = 8
MAX_RETRIES = 4

log = logging.getLogger(__name__)


class RetryableError(Exception):
    def __init__(self, status: int, retry_after: float) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _retry_after(resp: aiohttp.ClientResponse) -> float:
    try:
        return float(resp.headers.get("Retry-After", 0))
    except ValueError:  # header may be an HTTP date
        return 0.0


async def call_model(session: aiohttp.ClientSession, prompt: str) -> str:
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
    }
    async with session.post(API_URL, json=payload) as resp:
        if resp.status in RETRYABLE_STATUS:
            raise RetryableError(resp.status, _retry_after(resp))
        resp.raise_for_status()  # 400/401/404...: permanent, fail fast
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def call_with_retry(
    session: aiohttp.ClientSession, sem: asyncio.Semaphore, prompt: str
) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:  # held during the request only, not while backing off
                return await call_model(session, prompt)
        except (RetryableError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = max(getattr(e, "retry_after", 0.0), 2**attempt) + random.random()
            log.warning("retry %d after %.1fs: %s", attempt + 1, delay, e)
            await asyncio.sleep(delay)  # never time.sleep in a coroutine
    raise AssertionError("unreachable")


async def run_all(prompts: list[str]) -> list[str | BaseException]:
    """Results are aligned with prompts; failures are returned, not swallowed."""
    headers = {"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"}
    timeout = aiohttp.ClientTimeout(total=60)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        return await asyncio.gather(
            *(call_with_retry(session, sem, p) for p in prompts),
            return_exceptions=True,
        )


def save(prompts: list[str], results: list[str | BaseException], path: Path) -> None:
    rows = [
        {"prompt": p, "error": repr(r)} if isinstance(r, BaseException)
        else {"prompt": p, "result": r}
        for p, r in zip(prompts, results)
    ]
    with path.open("w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    failed = sum("error" in r for r in rows)
    log.info("saved %d rows, %d failed", len(rows), failed)


async def main() -> None:
    prompts = [f"Résume l'article {i}" for i in range(5000)]
    save(prompts, await run_all(prompts), Path("out.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
