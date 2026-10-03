import asyncio
import json
import time
from typing import Any

import aiohttp

API_URL = "https://api.mistral.ai/v1/chat/completions"
# REVIEW [BLOQUANT][SÉCU] Secret commité en clair. Le révoquer + `os.environ["MISTRAL_API_KEY"]`.
API_KEY = "sk-live-4f9a1c0e7b2d"  # TODO move to env


# REVIEW [MAJEUR] `Any` partout : le retour est un `str`. Typer précisément.
#        [MAJEUR] Une `ClientSession` par appel -> pas de pool de connexions ni keep-alive. Passer une session partagée en paramètre.
async def call_model(prompt: str) -> Any:
    async with aiohttp.ClientSession() as session:
        # REVIEW [MAJEUR] Aucun timeout : une requête bloquée bloque la tâche indéfiniment. `aiohttp.ClientTimeout(total=60)` sur la session.
        resp = await session.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        # REVIEW [BLOQUANT] Pas de `raise_for_status()` : sur 429/500 le JSON d'erreur n'a pas de `choices` -> `KeyError` qui masque la vraie cause.
        #        Fix: lever `RetryableError` pour 429/5xx (avec `Retry-After`), `raise_for_status()` pour le reste (4xx permanentes, pas de retry).
        data = await resp.json()
        return data["choices"][0]["message"]["content"]


# REVIEW [MAJEUR] Aucune borne de concurrence : voir `_run_all`. Passer un `asyncio.Semaphore` ici, tenu pendant la requête seulement.
async def call_with_retry(prompt: str, retries: int = 3) -> Any:
    for attempt in range(retries):
        try:
            return await call_model(prompt)
        # REVIEW [BLOQUANT] `except Exception` retente TOUT (401, 400, bug de parsing...). Ne retenter que les erreurs transitoires (429, 5xx, erreurs réseau, timeout).
        except Exception as e:
            # REVIEW [MINEUR] `print` -> `logging` (niveau, contexte, pas de perte en prod).
            print("error", e)
            # REVIEW [BLOQUANT] `time.sleep` dans une coroutine GÈLE toute l'event loop : le retry annule le bénéfice de l'async.
            #        Fix: `await asyncio.sleep(...)`, avec jitter et en respectant `Retry-After`.
            time.sleep(2**attempt)


# REVIEW [BLOQUANT] Après 3 échecs la fonction renvoie `None` implicitement : échec SILENCIEUX. Re-raise au dernier essai.
async def _run_all(prompts: list[str]) -> list[Any]:
    # REVIEW [BLOQUANT] 5000 coroutines lancées d'un coup : rafale de 429, sockets saturés. `Semaphore(8)` ou un rate limiter.
    tasks = [call_with_retry(p) for p in prompts]
    # REVIEW [MAJEUR] `return_exceptions=True` met les exceptions DANS la liste : il faut les traiter côté appelant (voir `save`).
    return await asyncio.gather(*tasks, return_exceptions=True)


# REVIEW [MINEUR] `asyncio.run` interdit l'appel depuis une loop déjà lancée (notebook, FastAPI). Exposer la coroutine, laisser l'appelant décider.
def run_all(prompts: list[str]) -> list[Any]:
    return asyncio.run(_run_all(prompts))


# REVIEW [MAJEUR] `save` ne reçoit pas les prompts : impossible de relier une réponse à son prompt. Passer `prompts` et garder une ligne par prompt.
def save(results: list[Any], path: str) -> None:
    out = []
    for r in results:
        # REVIEW [BLOQUANT] Ce filtre supprime les `None` -> CASSE l'alignement prompts <-> résultats (réponse C affichée en face de B).
        #        Et il GARDE les exceptions (truthy) -> `json.dump` plante (TypeError). Ne jamais filtrer : `{prompt, result}` ou `{prompt, error: repr(r)}`.
        if r:
            out.append(r)
    # REVIEW [MINEUR] `open()` jamais fermé -> `with path.open("w") as f`. `ensure_ascii=False` pour le français.
    json.dump(out, open(path, "w"))


if __name__ == "__main__":
    # REVIEW [MINEUR] Journaliser le nombre d'échecs à la fin ; sinon un run à 30 % d'erreurs a l'air de réussir.
    prompts = [f"Résume l'article {i}" for i in range(5000)]
    save(run_all(prompts), "out.json")
