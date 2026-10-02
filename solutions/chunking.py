from __future__ import annotations

from collections import Counter
from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def chunk_text(text: str, size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into chunks of `size` words, sharing `overlap` words."""
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got {overlap=} {size=}")
    words = text.split()  # collapses any whitespace, "" -> []
    chunks: list[str] = []
    for start in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):  # tail already covered: no redundant chunk
            break
    return chunks


def batch(items: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """Yield consecutive batches of n items (the last one may be shorter)."""
    if n <= 0:
        raise ValueError("n must be > 0")
    for i in range(0, len(items), n):
        yield items[i : i + n]


def top_words(chunks: Sequence[str], k: int = 5) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()  # local state: no leak between calls
    for c in chunks:
        counts.update(c.lower().split())
    return counts.most_common(k)


if __name__ == "__main__":
    assert chunk_text("") == []
    assert chunk_text("a  b\nc", size=10, overlap=2) == ["a b c"]
    assert chunk_text("a b c d e", size=3, overlap=1) == ["a b c", "c d e"]
    assert [list(b) for b in batch([1, 2, 3, 4, 5], 2)] == [[1, 2], [3, 4], [5]]
    assert top_words(["le chat"]) == top_words(["le chat"])  # no state shared
    for bad in (3, 4):
        try:
            chunk_text("a b c", size=3, overlap=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("overlap >= size must raise")
    print("ok")
