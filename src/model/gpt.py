"""GPT causal mínimo estilo GPT-2 (pre-norm, sin flash-attn para CPU)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 256, n_layer: int = 6,
                 n_head: int = 6, n_embd: int = 384, dropout: float = 0.0,
                 bias: bool = False, tie_weights: bool = True):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size, f"T={T} > block={self.block_size}"
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 120, temperature: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.0) -> torch.Tensor:
        self.eval()
        greedy = temperature <= 0
        for _ in range(max_new_tokens):
            cond = idx[:, -self.block_size:]
            logits, _ = self(cond)
            if not greedy:
                logits = logits[:, -1, :] / max(temperature, 1e-5)
            else:
                logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    for t in set(idx[b].tolist()):
                        if logits[b, t] > 0:
                            logits[b, t] /= repetition_penalty
                        else:
                            logits[b, t] *= repetition_penalty
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt], dim=1)
                continue
            if top_k and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                cdf = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
                cut = cdf > top_p
                cut[..., 0] = False
                s[cut] = float("-inf")
                logits = torch.zeros_like(logits).scatter(1, si, s)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
