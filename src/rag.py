import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Any = None


def embed(texts: List[str]) -> np.ndarray:
    """Stand-in for an embedding API: deterministic pseudo-vectors."""
    out = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % 2**32
        out.append(np.random.RandomState(seed).randn(64))
    return np.array(out)


class VectorStore:
    def __init__(self):
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        vecs = embed([c.text for c in chunks])
        for c, v in zip(chunks, vecs):
            c.embedding = v
            self.chunks.append(c)

    def search(self, query: str, top_k: int = 5, source: str = None):
        q = embed([query])[0]
        scored = []
        for c in self.chunks:
            if source and c.metadata["source"] != source:
                continue
            score = np.dot(q, c.embedding)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored[:top_k]]


def split(text: str, size: int = 500, source: str = "unknown", base_meta={}) -> List[Chunk]:
    chunks = []
    for i in range(0, len(text), size):
        meta = base_meta
        meta["source"] = source
        meta["offset"] = i
        chunks.append(Chunk(text[i : i + size], meta))
    return chunks


def ingest(store: VectorStore, docs: Dict[str, str]) -> None:
    for name, text in docs.items():
        store.add(split(text, source=name))


def build_prompt(question: str, hits: List[Chunk]) -> str:
    context = "\n".join(h.text for h in hits)
    return f"""Answer using the context.
Context: {context}
Question: {question}
Answer:"""


def answer(store: VectorStore, llm, question: str, source: str = None) -> str:
    hits = store.search(question, source=source)
    prompt = build_prompt(question, hits)
    return llm(prompt)


if __name__ == "__main__":
    s = VectorStore()
    ingest(s, {"a.txt": "Paris est la capitale de la France. " * 50,
               "b.txt": "Berlin est la capitale de l'Allemagne. " * 50})
    print(answer(s, lambda p: p[:200], "Quelle est la capitale de la France ?"))
