from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.nn.functional as F

EOS_ID = 2
PAD_ID = 0


class LM(Protocol):
    n_layers: int

    def __call__(
        self,
        ids: torch.Tensor,  # (B, T)
        pos: torch.Tensor,  # (B, T) absolute positions
        cache: "StaticKVCache",
        attn_mask: Optional[torch.Tensor] = None,  # (B, S) 1 = real token, 0 = pad
    ) -> torch.Tensor: ...  # logits (B, T, V)


class StaticKVCache:
    """Pre-allocated (B, H, max_len, D) buffers: no torch.cat per token, no fragmentation."""

    def __init__(self, n_layers: int, max_len: int) -> None:
        self.max_len = max_len
        self.k: list[Optional[torch.Tensor]] = [None] * n_layers
        self.v: list[Optional[torch.Tensor]] = [None] * n_layers
        self.lengths = [0] * n_layers

    def update(
        self, i: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k[i] is None:
            B, H, _, D = k.shape
            self.k[i] = k.new_empty(B, H, self.max_len, D)
            self.v[i] = v.new_empty(B, H, self.max_len, D)
        n, t = self.lengths[i], k.shape[2]
        if n + t > self.max_len:
            raise ValueError("KV cache overflow")
        self.k[i][:, :, n : n + t] = k
        self.v[i][:, :, n : n + t] = v
        self.lengths[i] = n + t
        return self.k[i][:, :, : n + t], self.v[i][:, :, : n + t]

    def seq_len(self) -> int:
        return self.lengths[0]


def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    if temperature <= 0:  # greedy, avoids division by zero
        return int(logits.argmax())
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = (cum - sorted_probs) < top_p  # always keeps the top token
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    return int(sorted_idx[torch.multinomial(sorted_probs, 1)])


@torch.inference_mode()  # no autograd graph: nothing pins activations or the cache
def generate(
    model: LM,
    prompt_ids: list[int],
    max_new_tokens: int = 64,
    eos_id: int = EOS_ID,
    temperature: float = 1.0,
) -> list[int]:
    cache = StaticKVCache(model.n_layers, len(prompt_ids) + max_new_tokens)
    out = list(prompt_ids)
    inp = torch.tensor([prompt_ids])
    for _ in range(max_new_tokens):
        pos = (torch.arange(inp.shape[1]) + cache.seq_len()).unsqueeze(0)  # cache offset
        logits = model(inp, pos, cache)
        nxt = sample(logits[0, -1], temperature)
        out.append(nxt)
        if nxt == eos_id:
            break
        inp = torch.tensor([[nxt]])
    return out


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int
    future: asyncio.Future
    t0: float  # submit time: latency includes queue wait


class BatchServer:
    """Groups requests into batches. One batch = one thread call; the loop stays free."""

    def __init__(
        self,
        model: LM,
        max_batch: int = 8,
        max_wait_s: float = 0.02,
        max_len: int = 256,
        max_queue: int = 64,
    ) -> None:
        self.model = model
        self.max_batch = max_batch
        self.max_wait_s = max_wait_s
        self.max_len = max_len
        self.queue: asyncio.Queue[Request] = asyncio.Queue(maxsize=max_queue)
        self.served = 0
        self.total_latency = 0.0

    async def submit(self, prompt_ids: list[int], max_new_tokens: int = 64) -> list[int]:
        if not prompt_ids or len(prompt_ids) + max_new_tokens > self.max_len:
            raise ValueError(f"prompt + new tokens must fit in {self.max_len}")
        loop = asyncio.get_running_loop()
        req = Request(prompt_ids, max_new_tokens, loop.create_future(), loop.time())
        try:
            self.queue.put_nowait(req)  # backpressure: reject instead of growing forever
        except asyncio.QueueFull:
            raise RuntimeError("server overloaded, retry later") from None
        return await req.future

    async def _collect(self) -> list[Request]:
        loop = asyncio.get_running_loop()
        batch = [await self.queue.get()]
        deadline = loop.time() + self.max_wait_s
        while len(batch) < self.max_batch:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:  # real wait, no busy loop
                batch.append(await asyncio.wait_for(self.queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return batch

    @torch.inference_mode()
    def _run_batch(self, batch: list[Request]) -> list[list[int]]:
        B = len(batch)
        S = max(len(r.prompt_ids) for r in batch)
        steps = max(r.max_new_tokens for r in batch)
        # LEFT padding + attention mask: last column is a real token for every row
        ids = torch.tensor([[PAD_ID] * (S - len(r.prompt_ids)) + r.prompt_ids for r in batch])
        mask = torch.tensor([[0] * (S - len(r.prompt_ids)) + [1] * len(r.prompt_ids) for r in batch])
        pos = (mask.cumsum(1) - 1).clamp(min=0)  # per-row positions ignore the pads
        cache = StaticKVCache(self.model.n_layers, S + steps)  # local: freed on return
        outs = [list(r.prompt_ids) for r in batch]
        done = [False] * B

        for _ in range(steps):
            logits = self.model(ids, pos, cache, mask)
            nxt_ids = []
            for i, r in enumerate(batch):
                if done[i]:
                    nxt_ids.append(PAD_ID)
                    continue
                nxt = sample(logits[i, -1])
                outs[i].append(nxt)
                nxt_ids.append(nxt)
                if nxt == EOS_ID or len(outs[i]) - len(r.prompt_ids) >= r.max_new_tokens:
                    done[i] = True
            if all(done):
                break
            ids = torch.tensor([[t] for t in nxt_ids])
            mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype)], dim=1)
            pos = mask.sum(1, keepdim=True) - 1
        # ponytail: finished rows still ride along; continuous batching (vLLM) removes that waste
        return outs

    async def serve_forever(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [r for r in await self._collect() if not r.future.done()]  # skip cancelled
            if not batch:
                continue
            try:
                results = await asyncio.to_thread(self._run_batch, batch)
            except Exception as e:
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(e)
                continue
            now = loop.time()
            for r, res in zip(batch, results):
                if not r.future.done():  # client may have timed out meanwhile
                    r.future.set_result(res)
                self.served += 1
                self.total_latency += now - r.t0

    def avg_latency(self) -> float:
        return self.total_latency / self.served if self.served else 0.0
