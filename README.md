# training-mistral

Entraînement au round **Python Code Review** (Mistral AI).

Chaque PR contient un script imparfait. Revois-la **directement dans l'interface GitHub** :
commentaires inline sur les lignes, puis une "Review" globale (Request changes).
Parle à voix haute comme en entretien : bug, impact, correctif, trade-off.

| PR | Thème | Difficulté | Taille |
|----|-------|-----------|--------|
| 1 | Chunking de texte | ★☆☆☆☆ | ~30 lignes |
| 2 | Client API async (rate limit, erreurs) | ★★☆☆☆ | ~55 lignes |
| 3 | Pipeline RAG (ingestion + retrieval + prompt) | ★★★☆☆ | ~80 lignes |
| 4 | Bloc Transformer PyTorch | ★★★★☆ | ~85 lignes |
| 5 | Génération avec KV-cache + batching serveur | ★★★★★ | ~140 lignes |

Les corrigés commentés sont sur la branche `solutions` — ne les ouvre qu'après ta review.

Grille : bugs/edge cases · typage · perf/scalabilité · robustesse prod · architecture.
