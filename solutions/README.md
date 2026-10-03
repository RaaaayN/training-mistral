# Scripts de correction attendus

| PR | Fichier original | Correction |
|----|------------------|-----------|
| 1 | `src/chunking.py` | `chunking.py` (a un self-check : `python chunking.py`) |
| 2 | `src/async_client.py` | `async_client.py` |
| 3 | `src/rag.py` | `rag.py` (self-check, nécessite numpy) |
| 4 | `src/transformer_block.py` | `transformer_block.py` (self-check de causalité, nécessite torch) |
| 5 | `src/generate.py` | `generate.py` (compile, non exécuté : le modèle n'est pas fourni) |

Le détail de ce qu'il faut dire en review est dans `../SOLUTIONS.md`.

## `annotated/` : la review, directement sur le code
Le code **d'origine** de chaque PR avec les commentaires `# REVIEW [GRAVITÉ]` placés sur les lignes fautives
(BLOQUANT > MAJEUR > MINEUR > NIT) et le correctif attendu. C'est ce que tu dois poster en commentaires inline sur GitHub.
