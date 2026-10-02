# Corrigés (ne pas lire avant d'avoir fait ta review)

## PR 1 — chunking.py
- `split(" ")` : double espaces/retours ligne → mots vides. Utiliser `split()`.
- `overlap >= size` → boucle infinie (start n'avance pas). Valider `0 <= overlap < size`.
- Chunks de queue redondants : après le dernier chunk complet, `start` reste < len → mini-chunks déjà inclus. Sortir quand `end >= len(words)`.
- Texte vide → `[""]`. Retourner `[]`.
- `batch` : `range(0, len - n, n)` perd le dernier batch (off-by-one). `range(0, len(items), n)`.
- `top_words(seen=[])` : mutable default arg → état partagé entre appels. Utiliser un Counter local.
- Aucun type hint, aucun test.

## PR 2 — async_client.py
- Clé API en dur dans le code (secret leak) → `os.environ`.
- Une `ClientSession` par requête : pas de pool de connexions. Une session partagée.
- 5000 tâches en `gather` sans limite → `asyncio.Semaphore` + rate limiting (429).
- Pas de `raise_for_status()` : une 429/500 renvoie du JSON d'erreur → `KeyError` sur `choices`.
- `time.sleep` dans une coroutine bloque toute l'event loop → `await asyncio.sleep` (+ jitter, respecter `Retry-After`).
- `except Exception` + print, et retourne `None` après l'échec des retries → échec silencieux. Re-raise au dernier essai.
- Pas de timeout (`ClientTimeout`).
- `return_exceptions=True` puis `if r:` : les exceptions (truthy) sont écrites dans le JSON ; en plus `save` casse l'alignement prompt↔résultat (filtre sans index). Et `json.dump` d'exceptions plante.
- `open()` jamais fermé → `with`. `Any` partout → `str`/`str | None`.
- `asyncio.run` dans `run_all` : inutilisable depuis une loop déjà lancée (notebook/FastAPI).

## PR 3 — rag.py
- `search` trie en ordre **croissant** → renvoie les moins similaires. `reverse=True`.
- `np.dot` ≠ cosine : non normalisé, favorise les gros vecteurs.
- `base_meta={}` mutable partagé ET `meta = base_meta` sans copie → tous les chunks partagent le même dict (tous ont le dernier offset/source). `dict(base_meta)`.
- `c.metadata["source"]` → KeyError si absent ; `source: str = None` mal typé (`Optional[str]`).
- Chunking par caractères : coupe les mots/phrases, pas d'overlap.
- Boucle Python au lieu d'une matrice numpy (O(n) python) ; embeddings appelés par doc, pas batchés.
- `embedding: Any`. `np.ndarray` précis.
- Retriever vide → prompt avec contexte vide, le LLM hallucine. Gérer le cas (réponse "je ne sais pas" / seuil de score).
- Prompt : contexte non délimité/numéroté, pas de citation de source, risque de prompt injection via documents ; pas de limite de tokens du contexte.
- `ingest` : ré-ingérer le même doc duplique (pas d'id idempotent).

## PR 4 — transformer_block.py
- `view` sans `transpose(1, 2)` : on mélange T et H. Il faut (B, H, T, D) puis `.transpose(1,2).contiguous().view`.
- Scaling `sqrt(C)` au lieu de `sqrt(d_head)`.
- Masque : `masked_fill(mask == 0, 0.0)` → doit être `-inf`. Et aucun masque causal n'est passé dans `TinyLM.forward` → fuite du futur, loss triviale.
- `softmax(dim=1)` → doit être `dim=-1`.
- Post-norm (`norm(x + sub(x))`) : instable en profondeur ; Mistral/LLaMA-like = pre-norm : `x + attn(norm(x))`, + norme finale avant la head.
- `torch.arange(T)` sans device → crash sur GPU.
- Loss : pas de décalage next-token (`logits[:, :-1]` vs `ids[:, 1:]`) → le modèle apprend à copier.
- Dropout de l'attention à l'intérieur de `forward` OK car `nn.Dropout`, mais on pourrait utiliser `F.scaled_dot_product_attention`.
- ReLU vs SwiGLU/GELU, LayerNorm vs RMSNorm (remarque archi, pas bug). Pas de `max_len` check.

## PR 5 — generate.py
- `KVCache` n'est jamais utilisé par le modèle ici mais : `update` concatène en `cat` à chaque token → O(T²) copies et fragmentation ; pré-allouer (static cache).
- Pas de `torch.no_grad()` / `inference_mode()` : le graphe autograd s'accumule, fuite mémoire énorme, cache attaché au graphe.
- `sample` : `cum < top_p` exclut le premier token si sa proba ≥ top_p → tout à 0 → division par 0 (NaN). Utiliser `cum - sorted_probs < top_p`. `temperature=0` → division par zéro (greedy). `.item()` sync GPU.
- `generate` : `pos = arange(inp.shape[1])` ignore l'offset du cache → positions fausses dès le step 1 (la version batch l'a, pas celle-ci : incohérence).
- Batching : padding à droite avec 0 sans attention mask → les pads sont attendus ; pour la génération il faut du left-padding. Les prompts courts génèrent à partir d'un pad, et `outs[i][-1]` est lu au step 1 sans cohérence avec le cache.
- Les séquences déjà `done` continuent d'être calculées dans le batch (gaspillage) ; pas de continuous batching.
- `self.caches[id(batch)] = cache` jamais supprimé → fuite mémoire du KV-cache (c'est LE bug de prod). `id()` réutilisable.
- `_run_batch` est synchrone et CPU/GPU-bound dans la loop → bloque tous les `submit` pendant la génération ; `run_in_executor`/thread.
- `_collect` : busy loop `sleep(0)` à 100% CPU ; utiliser `asyncio.wait_for(queue.get(), timeout)`. `time.time()` → `time.monotonic()`.
- `set_exception` sur un future déjà annulé → `InvalidStateError` (client timeout) ; vérifier `fut.done()`. `get_event_loop()` déprécié → `get_running_loop()`.
- Une exception dans un batch fait échouer toutes les requêtes ; la boucle `continue` sans libérer le cache.
- `avg_latency` division par zéro ; la latence mesurée est celle du batch, pas par requête (queue wait omis).
- Pas de borne sur la taille de la queue (backpressure) ni sur la longueur des prompts.
