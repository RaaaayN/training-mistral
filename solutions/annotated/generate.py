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
            # REVIEW [MAJEUR] `torch.cat` à chaque token : copie de tout le cache -> O(T²) et fragmentation mémoire. Pré-allouer (static cache de taille max_len).
            self.layers[i] = (torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2))
        return self.layers[i]

    def seq_len(self) -> int:
        first = self.layers[0]
        return 0 if first is None else first[0].shape[2]


def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    # REVIEW [MAJEUR] `temperature=0` -> division par zéro. Gérer le cas greedy (`argmax`).
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # REVIEW [BLOQUANT] Si le 1er token a une proba >= top_p, tout est masqué -> `sum() == 0` -> NaN/crash dans `multinomial`.
    #        Fix: `keep = (cum - sorted_probs) < top_p` (le 1er token est toujours gardé).
    keep = cum < top_p
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    choice = torch.multinomial(sorted_probs, 1)
    # REVIEW [NIT] `.item()` force une synchro GPU à chaque token.
    return sorted_idx[choice].item()


# REVIEW [BLOQUANT] Pas de `torch.no_grad()` / `inference_mode()` : le graphe autograd s'accumule, les activations et le cache restent épinglés -> OOM.
#        Fix: `@torch.inference_mode()`.
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
        # REVIEW [BLOQUANT] Positions fausses dès le step 1 : il manque l'offset du cache (`+ cache.seq_len()`), le token généré est toujours en position 0.
        #        Incohérent avec `_run_batch`, qui l'ajoute.
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
        # REVIEW [MAJEUR] Queue non bornée : pas de backpressure, la mémoire grossit sous charge. `Queue(maxsize=...)` et rejeter quand pleine.
        #        Aucune limite non plus sur la longueur des prompts.
        self.queue: asyncio.Queue = asyncio.Queue()
        # REVIEW [BLOQUANT] Ce dict stocke un KV-cache par batch et n'est JAMAIS vidé -> fuite mémoire GPU garantie en prod. Le cache doit être une variable locale de `_run_batch`.
        self.caches: Dict[int, KVCache] = {}
        self.stats = {"served": 0, "latency": 0.0}

    async def submit(self, prompt_ids: List[int], max_new_tokens: int = 64):
        # REVIEW [MINEUR] `get_event_loop()` déprécié -> `get_running_loop()`. Mémoriser l'heure de soumission pour mesurer la vraie latence.
        fut = asyncio.get_event_loop().create_future()
        # REVIEW [MAJEUR] Valider avant de mettre en file (prompt vide, prompt + max_new_tokens > max_len) pour échouer vite avec un message clair.
        await self.queue.put(Request(prompt_ids, max_new_tokens, fut))
        return await fut

    async def _collect(self) -> List[Request]:
        batch = [await self.queue.get()]
        # REVIEW [MINEUR] `time.time()` n'est pas monotone -> `loop.time()` / `time.monotonic()`.
        deadline = time.time() + self.max_wait_s
        while len(batch) < self.max_batch and time.time() < deadline:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                # REVIEW [MAJEUR] Attente active : la boucle tourne à 100 % CPU pendant `max_wait_s`.
                #        Fix: `await asyncio.wait_for(self.queue.get(), timeout)`.
                await asyncio.sleep(0)
        return batch

    def _run_batch(self, batch: List[Request]) -> List[List[int]]:
        max_len = max(len(r.prompt_ids) for r in batch)
        # REVIEW [BLOQUANT] Padding à DROITE et aucun attention mask : les pads sont attendus, et la génération démarre APRÈS les pads pour les prompts courts.
        #        Fix: left-padding + `attn_mask` + positions calculées par ligne (`mask.cumsum(1) - 1`).
        padded = [r.prompt_ids + [0] * (max_len - len(r.prompt_ids)) for r in batch]
        ids = torch.tensor(padded)
        cache = KVCache(self.model.n_layers)
        outs = [list(r.prompt_ids) for r in batch]
        done = [False] * len(batch)
        # REVIEW [BLOQUANT] Fuite : voir `self.caches`. En plus `id()` peut être réutilisé après garbage collection.
        self.caches[id(batch)] = cache

        for step in range(max(r.max_new_tokens for r in batch)):
            inp = ids if step == 0 else torch.tensor([[o[-1]] for o in outs])
            pos = torch.arange(inp.shape[1]) + cache.seq_len()
            # REVIEW [MAJEUR] Sans `attn_mask`, le modèle ne peut pas ignorer les pads.
            logits = self.model(inp, pos, cache)
            for i, r in enumerate(batch):
                if done[i]:
                    continue
                # REVIEW [MINEUR] Température/top_p non configurables par requête. Séquence terminée (`done[i]`) : elle reste dans le batch et consomme du calcul.
                #        Piste (à citer, pas à exiger) : continuous batching (vLLM).
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
                # REVIEW [BLOQUANT] `_run_batch` est synchrone et lourd (GPU/CPU) : il GÈLE l'event loop, plus aucun `submit` n'avance pendant la génération.
                #        Fix: `await asyncio.to_thread(self._run_batch, batch)` (ou `run_in_executor`).
                results = self._run_batch(batch)
            except Exception as e:
                for r in batch:
                    # REVIEW [MAJEUR] `InvalidStateError` si la future est déjà annulée (client en timeout) -> tester `fut.done()`.
                    #        Une exception fait échouer TOUT le batch ; filtrer les futures annulées avant de lancer le batch.
                    r.future.set_exception(e)
                continue
            for r, res in zip(batch, results):
                # REVIEW [MAJEUR] Même risque `InvalidStateError` : tester `if not r.future.done()`.
                r.future.set_result(res)
            self.stats["served"] += len(batch)
            # REVIEW [MINEUR] Mesure le temps du batch, pas la latence vue par le client (temps d'attente en queue omis). Mesurer depuis `submit`.
            self.stats["latency"] += time.time() - start

    def avg_latency(self) -> float:
        # REVIEW [MINEUR] Division par zéro tant que rien n'a été servi.
        return self.stats["latency"] / self.stats["served"]
