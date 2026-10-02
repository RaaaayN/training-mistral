from collections import Counter


def chunk_text(text, size=200, overlap=50):
    """Split text into chunks of `size` words, sharing `overlap` words."""
    words = text.split(" ")
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def batch(items, n):
    """Yield consecutive batches of n items."""
    for i in range(0, len(items) - n, n):
        yield items[i : i + n]


def top_words(chunks, k=5, seen=[]):
    for c in chunks:
        seen.extend(c.lower().split())
    return Counter(seen).most_common(k)


if __name__ == "__main__":
    doc = "le chat dort sur le canapé " * 100
    cs = chunk_text(doc, size=20, overlap=5)
    print(len(cs), top_words(cs))
