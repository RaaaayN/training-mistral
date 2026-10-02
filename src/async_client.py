import asyncio
import json
import time
from typing import Any

import aiohttp

API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "sk-live-4f9a1c0e7b2d"  # TODO move to env


async def call_model(prompt: str) -> Any:
    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


async def call_with_retry(prompt: str, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            return await call_model(prompt)
        except Exception as e:
            print("error", e)
            time.sleep(2**attempt)


async def _run_all(prompts: list[str]) -> list[Any]:
    tasks = [call_with_retry(p) for p in prompts]
    return await asyncio.gather(*tasks, return_exceptions=True)


def run_all(prompts: list[str]) -> list[Any]:
    return asyncio.run(_run_all(prompts))


def save(results: list[Any], path: str) -> None:
    out = []
    for r in results:
        if r:
            out.append(r)
    json.dump(out, open(path, "w"))


if __name__ == "__main__":
    prompts = [f"Résume l'article {i}" for i in range(5000)]
    save(run_all(prompts), "out.json")
