from collections import Counter


# REVIEW [MAJEUR] Aucun type hint ni validation des paramètres. Ajouter `-> list[str]` et refuser `overlap >= size` (ValueError).
def chunk_text(text, size=200, overlap=50):
    """Split text into chunks of `size` words, sharing `overlap` words."""
    # REVIEW [MAJEUR] `split(" ")` crée des mots vides sur les doubles espaces et les \n ; un texte vide donne `[""]` au lieu de `[]`.
    #        Fix: `text.split()`.
    words = text.split(" ")
    chunks = []
    start = 0
    # REVIEW [BLOQUANT] Boucle infinie si `overlap >= size` (start n'avance plus).
    #        [BLOQUANT] Et même sinon : après le dernier chunk complet, `start < len(words)` reste vrai -> chunks de queue redondants.
    #        Fix: `for start in range(0, len(words), size - overlap)` + `break` dès que `start + size >= len(words)`.
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        # REVIEW [BLOQUANT] Cause de la boucle infinie (voir ci-dessus).
        start = end - overlap
    return chunks


def batch(items, n):
    """Yield consecutive batches of n items."""
    # REVIEW [BLOQUANT] Off-by-one : le dernier batch (partiel ou complet) est perdu. 5 items, n=2 -> [1,2],[3,4] ; le 5 disparaît.
    #        Fix: `range(0, len(items), n)` et valider `n > 0`.
    for i in range(0, len(items) - n, n):
        yield items[i : i + n]


# REVIEW [BLOQUANT] Argument par défaut mutable : la liste est créée UNE fois et partagée entre tous les appels.
#        Fix: `Counter` local, plus besoin du paramètre.
def top_words(chunks, k=5, seen=[]):
    for c in chunks:
        seen.extend(c.lower().split())
    # REVIEW [MAJEUR] Conséquence : les comptes se cumulent d'un appel à l'autre (non déterministe, fuite mémoire lente).
    return Counter(seen).most_common(k)


# REVIEW [MINEUR] Pas de test. Demander un test des cas limites : texte vide, overlap == size, texte plus court qu'un chunk, batch avec reste.
if __name__ == "__main__":
    doc = "le chat dort sur le canapé " * 100
    cs = chunk_text(doc, size=20, overlap=5)
    print(len(cs), top_words(cs))
