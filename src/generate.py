import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

KV = Tuple[torch.Tensor, torch.Tensor]


class KVCache:
    """Per-request cache: one (k, v) pair per layer, shape (B, H, T, D)."""

    def __init__(self, n_layers: int):
        self.layers: List[Optional[KV]] = [None] * n_layers

    def update(self, i: int, k: torch.Tensor, v: torch.Tensor) -> KV:
        if self.layers[i] is None:
            self.layers[i] = (k, v)
        else:
            pk, pv = self.layers[i]
            self.layers[i] = (torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2))
        return self.layers[i]

    def seq_len(self) -> int:
        first = self.layers[0]
        return 0 if first is None else first[0].shape[2]


def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = cum < top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    choice = torch.multinomial(sorted_probs, 1)
    return sorted_idx[choice].item()


def generate(
    model,
    prompt_ids: List[int],
    max_new_tokens: int = 64,
    eos_id: int = 2,
    temperature: float = 1.0,
) -> List[int]:
    cache = KVCache(model.n_layers)
    ids = torch.tensor([prompt_ids])
    out = list(prompt_ids)

    for step in range(max_new_tokens):
        if step == 0:
            inp = ids
        else:
            inp = torch.tensor([[out[-1]]])
        pos = torch.arange(inp.shape[1])
        logits = model(inp, pos, cache)
        nxt = sample(logits[0, -1], temperature)
        out.append(nxt)
        if nxt == eos_id:
            break
    return out


@dataclass
class Request:
    prompt_ids: List[int]
    max_new_tokens: int
    future: asyncio.Future


class BatchServer:
    """Collects requests and runs them through the model in batches."""

    def __init__(self, model, max_batch: int = 8, max_wait_s: float = 0.02):
        self.model = model
        self.max_batch = max_batch
        self.max_wait_s = max_wait_s
        self.queue: asyncio.Queue = asyncio.Queue()
        self.caches: Dict[int, KVCache] = {}
        self.stats = {"served": 0, "latency": 0.0}

    async def submit(self, prompt_ids: List[int], max_new_tokens: int = 64):
        fut = asyncio.get_event_loop().create_future()
        await self.queue.put(Request(prompt_ids, max_new_tokens, fut))
        return await fut

    async def _collect(self) -> List[Request]:
        batch = [await self.queue.get()]
        deadline = time.time() + self.max_wait_s
        while len(batch) < self.max_batch and time.time() < deadline:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0)
        return batch

    def _run_batch(self, batch: List[Request]) -> List[List[int]]:
        max_len = max(len(r.prompt_ids) for r in batch)
        padded = [r.prompt_ids + [0] * (max_len - len(r.prompt_ids)) for r in batch]
        ids = torch.tensor(padded)
        cache = KVCache(self.model.n_layers)
        outs = [list(r.prompt_ids) for r in batch]
        done = [False] * len(batch)
        self.caches[id(batch)] = cache

        for step in range(max(r.max_new_tokens for r in batch)):
            inp = ids if step == 0 else torch.tensor([[o[-1]] for o in outs])
            pos = torch.arange(inp.shape[1]) + cache.seq_len()
            logits = self.model(inp, pos, cache)
            for i, r in enumerate(batch):
                if done[i]:
                    continue
                nxt = sample(logits[i, -1])
                outs[i].append(nxt)
                if nxt == 2 or len(outs[i]) - len(r.prompt_ids) >= r.max_new_tokens:
                    done[i] = True
            if all(done):
                break
        return outs

    async def serve_forever(self) -> None:
        while True:
            batch = await self._collect()
            start = time.time()
            try:
                results = self._run_batch(batch)
            except Exception as e:
                for r in batch:
                    r.future.set_exception(e)
                continue
            for r, res in zip(batch, results):
                r.future.set_result(res)
            self.stats["served"] += len(batch)
            self.stats["latency"] += time.time() - start

    def avg_latency(self) -> float:
        return self.stats["latency"] / self.stats["served"]
