import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # REVIEW [NIT] 3 projections séparées : une seule `nn.Linear(d_model, 3*d_model)` + `chunk` est plus rapide.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        # REVIEW [BLOQUANT] Il manque `.transpose(1, 2)` : on reste en (B, T, H, D). Le `@` agit alors sur (H, D) et `scores` vaut (B, T, H, H) :
        #        chaque token compare ses têtes entre elles et ne voit JAMAIS les autres tokens. Aucune erreur car les formes restent valides.
        #        Fix: `(B, T, H, D) -> transpose(1, 2) -> (B, H, T, D)` pour q, k et v.
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head)

        # REVIEW [MAJEUR] Scaling par `sqrt(C)` au lieu de `sqrt(d_head)` : le produit scalaire porte sur d_head éléments.
        scores = q @ k.transpose(-2, -1) / math.sqrt(C)
        if mask is not None:
            # REVIEW [BLOQUANT] Il faut `-inf`, pas `0.0` : avec 0, les positions masquées gardent un poids après le softmax.
            #        Et `mask` est (T, T) : il ne se broadcast pas sur (B, T, H, H). Préférer `F.scaled_dot_product_attention(..., is_causal=True)`.
            scores = scores.masked_fill(mask == 0, 0.0)
        # REVIEW [BLOQUANT] Mauvais axe : on doit normaliser sur les CLÉS (`dim=-1`) pour que chaque ligne (requête) somme à 1.
        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)

        # REVIEW [MAJEUR] Après correction des têtes il faudra `.transpose(1, 2).contiguous().view(B, T, C)` (view exige une mémoire contiguë).
        out = (attn @ v).view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            # REVIEW [NIT] GELU/SwiGLU sont le standard actuel (choix d'architecture, pas un bug).
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # REVIEW [MAJEUR] Post-norm : `norm(x + sub(x))` est instable en profondeur. Pre-norm : `x = x + attn(norm1(x))`.
        #        [MINEUR] Le `mask` est une option : rien ne force le causal. Le rendre non optionnel pour un LM.
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d_model=128, n_heads=4, d_ff=512, n_layers=4, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor):
        B, T = ids.shape
        # REVIEW [MAJEUR] `torch.arange(T)` sans `device=ids.device` -> crash sur GPU. Pas de vérification `T <= max_len` (IndexError obscure).
        x = self.tok(ids) + self.pos(torch.arange(T))
        for blk in self.blocks:
            # REVIEW [BLOQUANT] Aucun masque causal n'est passé aux blocs. Latent aujourd'hui (l'attention ne mélange rien), mais dès que les têtes sont corrigées, chaque position VOIT le futur.
            x = blk(x)
        # REVIEW [MAJEUR] Pre-norm => il faut une norme finale (`self.norm(x)`) avant la tête.
        return self.head(x)

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self(ids)
        # REVIEW [BLOQUANT] C'EST ICI QUE LE MODÈLE TRICHE. Les logits de la position t sont comparés à ids[t], l'entrée elle-même : le modèle apprend à RECOPIER (loss 0.02 en 200 steps).
        #        Fix: décalage next-token : `logits[:, :-1]` contre `ids[:, 1:]`.
        return F.cross_entropy(logits.view(-1, logits.size(-1)), ids.view(-1))
