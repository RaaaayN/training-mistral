# Corrigés : ce qu'il faut dire en review

> Ne lis ce fichier qu'après avoir soumis ta review.
> Schéma de chaque point : **constat → impact → correctif**.
> Clôture à dire à chaque fois : « Verdict : *Request changes*. Les bloquants sont X et Y, le reste est mineur. Je veux des tests sur les cas limites avant de merger. »

---

## PR 1 : chunking (`src/chunking.py`)
Ouverture : « Quatre problèmes de correction, un état partagé, et aucun test. »

1. **`chunk_text` boucle à l'infini si `overlap >= size`** : `start = end - overlap` n'avance plus. Valider `0 <= overlap < size` et lever `ValueError`.
2. **Chunks de queue redondants** : après le dernier mot, la boucle continue et crée des mini-chunks déjà inclus. Sortir dès que `end >= len(words)`.
3. **`split(" ")` génère des mots vides** (doubles espaces, retours ligne). Utiliser `split()`. Texte vide → `[""]` au lieu de `[]`.
4. **`batch` perd le dernier batch** : `range(0, len - n, n)` est un off-by-one. Corriger en `range(0, len(items), n)`.
5. **`seen=[]` : argument par défaut mutable.** L'état persiste entre appels, les comptes se cumulent. `Counter` local.
6. **Pas de type hints ni de test.** Ajouter les types et un test paramétré : vide, `overlap == size`, texte plus court qu'un chunk.

---

## PR 2 : client async (`src/async_client.py`)
Ouverture : « Le gain de débit est réel, mais pas sûr en prod. Sécurité, puis correction, puis charge. »

1. **Clé API en dur** : secret commité, à révoquer. Lire `os.environ`.
2. **`time.sleep` dans une coroutine** bloque l'event loop, le retry annule le bénéfice de l'async. `await asyncio.sleep` + jitter, respecter `Retry-After`.
3. **5000 requêtes dans un `gather` sans limite** : cascade de 429, sockets saturés. `asyncio.Semaphore` ou rate limiter.
4. **Pas de `raise_for_status()`** : sur 429/500 le JSON d'erreur n'a pas de `choices` → `KeyError` qui masque la cause. Vérifier le statut, ne retry que 429 et 5xx.
5. **Échec silencieux** : `call_with_retry` avale l'exception, `print`, puis renvoie `None`. Re-raise au dernier essai.
6. **Une `ClientSession` par requête** : pas de pool ni de keep-alive. Session partagée + `ClientTimeout`.
7. **`save` incorrect** : `return_exceptions=True` renvoie des exceptions (truthy) que `save` garde ; le filtre `if r` casse l'alignement prompts↔résultats. Renvoyer `str | Exception`, garder l'index, tracer les échecs.
8. **Détails** : `open()` jamais fermé (`with`) ; `Any` → `str` ; `asyncio.run` interdit l'appel depuis une loop déjà lancée (notebook, FastAPI).

---

## PR 3 : RAG (`src/rag.py`)
Ouverture : « Un bug rend le retrieval faux, un bug de métadonnées est silencieux, et la robustesse manque. »

1. **`search` trie en ordre croissant** : on renvoie les chunks les *moins* similaires. `reverse=True`. Bug le plus grave, invisible sur la démo.
2. **`np.dot` n'est pas un cosinus** : les grosses normes sont favorisées. Normaliser, produit matriciel numpy plutôt qu'une boucle Python.
3. **`meta = base_meta` partage le même dict** entre tous les chunks (tous ont le dernier `offset`), et `base_meta={}` est un défaut mutable. `dict(base_meta)`, défaut `None`.
4. **`c.metadata["source"]` → `KeyError`** si absent ; `source: str = None` mal typé → `Optional[str]`.
5. **Retriever vide** : prompt avec contexte vide, le LLM hallucine. Seuil de score minimal, sinon « je ne sais pas ».
6. **Chunking par caractères** : coupe en plein mot, pas d'overlap. Par phrases ou tokens, avec chevauchement.
7. **Prompt** : contexte non délimité ni numéroté (pas de citation possible), injection de prompt via les documents, pas de limite en tokens. Délimiter, numéroter, traiter comme données, borner.
8. **Ingestion non idempotente** : réingérer duplique. Id stable (hash contenu + source).
9. **Typage et batching** : `embedding: Any` → `Optional[np.ndarray]` ; embeddings calculés doc par doc, à batcher.

---

## PR 4 : Transformer (`src/transformer_block.py`)
Ouverture : « Une loss à 0.02 après 200 steps est un symptôme : le modèle triche (fuite du futur + bugs de tenseurs). »

1. **Aucun masque causal** : `TinyLM.forward` n'en passe pas, chaque position voit le futur, loss triviale. Masque triangulaire inférieur.
2. **Pas de décalage next-token dans la loss** : `logits[t]` vs `ids[t]`, le modèle apprend à copier. `logits[:, :-1]` contre `ids[:, 1:]`.
3. **`masked_fill(mask == 0, 0.0)`** : doit être `-inf`, sinon les positions masquées gardent du poids après softmax.
4. **`softmax(dim=1)`** : mauvais axe, `dim=-1` (sur les clés).
5. **Têtes mal formées** : `view(B, T, H, D)` sans `transpose(1, 2)` mélange têtes et temps. Passer en `(B, H, T, D)`, puis `transpose(1, 2).contiguous().view(B, T, C)`.
6. **Scaling `sqrt(C)`** au lieu de `sqrt(d_head)` : logits trop écrasés.
7. **Post-norm** `norm(x + sub(x))` instable en profondeur. Pre-norm : `x = x + attn(norm1(x))`, norme finale avant la tête. (RMSNorm/SwiGLU : choix d'archi, pas bugs.)
8. **`torch.arange(T)` sans `device`** : plante sur GPU. Pas de vérification `T <= max_len`.
9. **Perf** : `F.scaled_dot_product_attention(..., is_causal=True)` à la place du calcul manuel.

---

## PR 5 : KV-cache et batching (`src/generate.py`)
Ouverture : « Fuite mémoire certaine, bug numérique dans le sampling, et la concurrence bloque la loop. »

**Bloquant**
1. **`self.caches[id(batch)]` jamais supprimé** : chaque batch laisse son KV-cache, la mémoire GPU sature. Supprimer dans un `finally` (ou ne pas stocker). `id()` peut être réutilisé après GC.
2. **Pas de `torch.inference_mode()`** : le graphe autograd s'accumule, le cache y est rattaché. Décorer `generate` et `_run_batch`.
3. **`_run_batch` synchrone dans une coroutine** : la loop est gelée, plus aucun `submit` n'avance. `run_in_executor` ou thread dédié.
4. **`sample` produit des NaN** : si le 1er token a une proba `>= top_p`, `cum < top_p` masque tout → division par zéro. Utiliser `(cum - sorted_probs) < top_p`. `temperature=0` divise aussi par zéro → `argmax`.

**Correction**
5. **Positions incohérentes** : `generate` utilise `arange(inp.shape[1])` sans l'offset du cache, `_run_batch` ajoute `cache.seq_len()`. Position fausse dès le step 1.
6. **Padding à droite sans attention mask** : les pads sont attendus, les prompts courts génèrent après des pads. Left-padding + attention mask.
7. **Futures déjà annulées** : après un timeout client, `set_result` lève `InvalidStateError`. Tester `fut.done()`. `get_event_loop()` → `get_running_loop()`.
8. **Isolation des erreurs** : une exception dans un batch fait échouer toutes les requêtes et le `continue` ne libère rien. Libérer dans le `finally`.

**Perf et scalabilité**
9. **`KVCache.update` fait un `torch.cat` par token** : O(T²) copies, fragmentation. Static cache pré-alloué de taille `max_len`.
10. **`_collect` en attente active** : `sleep(0)` tourne à 100 % CPU. `asyncio.wait_for(queue.get(), timeout)` et `time.monotonic()`.
11. **Séquences terminées restent dans le batch** : calcul gaspillé. Piste : continuous batching (vLLM), à citer, pas à exiger dans cette PR.

**Observabilité et limites**
12. **`avg_latency` divise par zéro** au démarrage, et mesure le batch sans le temps en queue. Mesurer de bout en bout depuis `submit`.
13. **Queue non bornée** : ni backpressure ni limite de longueur de prompt. `Queue(maxsize=...)`, rejeter les prompts trop longs avec une erreur claire.
