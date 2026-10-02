from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, C) -> (B, H, T, D): heads must come before time
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))
        # fused kernel: scales by sqrt(d_head), causal -inf mask, softmax over keys
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm: the residual stream stays un-normalized, gradients flow cleanly."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # final norm, required with pre-norm
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.max_len}")
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self(ids)
        # next-token prediction: position t predicts token t+1
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TinyLM(vocab=50).eval()
    a = torch.randint(0, 50, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 50
    # causality: changing the LAST token must not change earlier logits
    assert torch.allclose(m(a)[:, :-1], m(b)[:, :-1], atol=1e-5)
    print("ok")
