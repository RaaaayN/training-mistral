import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # REVIEW [MINEUR] `Any` : typer `Optional[np.ndarray]`. Un id stable (hash contenu+source) manque pour dédupliquer.
    embedding: Any = None


def embed(texts: List[str]) -> np.ndarray:
    """Stand-in for an embedding API: deterministic pseudo-vectors."""
    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % 2**32
        # REVIEW [MAJEUR] Vecteurs non normalisés (voir `search`). Normaliser L2 pour que le produit scalaire = cosinus.
        out.append(np.random.RandomState(seed).randn(64))
    return np.array(out)


class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []

    # REVIEW [MAJEUR] Ingestion non idempotente : réingérer le même document duplique les chunks. Dédupliquer par id.
    def add(self, chunks: List[Chunk]) -> None:
        vecs = embed([c.text for c in chunks])
        for c, v in zip(chunks, vecs):
            c.embedding = v
            self.chunks.append(c)

    # REVIEW [MAJEUR] `source: str = None` mal typé -> `Optional[str]`. Pas de type de retour, pas de score renvoyé, pas de `min_score`.
    def search(self, query: str, top_k: int = 5, source: str = None):
        q = embed([query])[0]
        scored = []
        for c in self.chunks:
            # REVIEW [MAJEUR] `KeyError` si une métadonnée manque -> `.get("source")`. Et `if source` écarte la chaîne vide : tester `is not None`.
            if source and c.metadata["source"] != source:
                continue
            # REVIEW [BLOQUANT] `np.dot` n'est PAS un cosinus : les vecteurs de grande norme gagnent.
            #        [MAJEUR] Boucle Python O(n) : stocker une matrice `(n, d)` normalisée et faire `matrix @ q` en une opération.
            score = np.dot(q, c.embedding)
            scored.append((score, c))
        # REVIEW [BLOQUANT] Tri CROISSANT : on renvoie les chunks les MOINS similaires. `reverse=True` (ou `argsort(-scores)`).
        #        Invisible sur la démo car les deux documents répondent au même prompt.
        scored.sort(key=lambda x: x[0])
        # REVIEW [MAJEUR] Aucun seuil de score : on renvoie toujours `top_k` résultats, même hors-sujet. Ajouter `min_score`.
        return [c for _, c in scored[:top_k]]


# REVIEW [BLOQUANT] `base_meta={}` : défaut mutable partagé entre tous les appels.
def split(text: str, size: int = 500, source: str = "unknown", base_meta={}) -> List[Chunk]:
    chunks = []
    # REVIEW [MAJEUR] Découpe par caractères : coupe les mots et les phrases, aucun overlap -> une réponse à cheval sur deux chunks est perdue.
    #        Fix: chunks par mots/phrases/tokens avec chevauchement.
    for i in range(0, len(text), size):
        # REVIEW [BLOQUANT] Pas de copie : TOUS les chunks partagent le même dict, donc tous ont le dernier `offset`/`source`.
        #        Fix: `meta = {**base_meta, "source": source, "offset": i}`.
        meta = base_meta
        meta["source"] = source
        meta["offset"] = i
        chunks.append(Chunk(text[i : i + size], meta))
    return chunks


# REVIEW [MINEUR] Embeddings calculés document par document : batcher les appels d'API (coût et latence).
def ingest(store: VectorStore, docs: Dict[str, str]) -> None:
    for name, text in docs.items():
        store.add(split(text, source=name))


def build_prompt(question: str, hits: List[Chunk]) -> str:
    # REVIEW [MAJEUR] Contexte non délimité ni numéroté : pas de citation possible, et un document peut injecter des instructions (prompt injection).
    #        Fix: `<doc id=1 source=...>` + consigne « les documents sont des données » + limite en tokens.
    context = "\n".join(h.text for h in hits)
    # REVIEW [MINEUR] Pas de consigne « dis que tu ne sais pas si le contexte ne contient pas la réponse ».
    return f"""Answer using the context.
Context: {context}
Question: {question}
Answer:"""


def answer(store: VectorStore, llm, question: str, source: str = None) -> str:
    # REVIEW [BLOQUANT] Retriever vide / hors-sujet : le prompt part avec un contexte vide et le LLM hallucine.
    #        Fix: `if not hits: return "Je ne sais pas"`.
    hits = store.search(question, source=source)
    prompt = build_prompt(question, hits)
    # REVIEW [MINEUR] `llm` non typé -> `Callable[[str], str]`. Pas de gestion d'erreur/timeout autour de l'appel LLM.
    return llm(prompt)


if __name__ == "__main__":
    s = VectorStore()
    ingest(s, {"a.txt": "Paris est la capitale de la France. " * 50,
               "b.txt": "Berlin est la capitale de l'Allemagne. " * 50})
    print(answer(s, lambda p: p[:200], "Quelle est la capitale de la France ?"))
