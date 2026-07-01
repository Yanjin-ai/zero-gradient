"""Phase H / v3.0 base -- a STANDARD multi-layer trainable-attention Transformer (RESEARCH-ONLY).

Deliberately a clean break from the v1.0/v2.0 ZeroBP backbone (last-position collapse + single frozen
reservoir attention): this is a normal pre-LN Transformer with multi-head SELF-attention trained by full
backprop, and NON-collapsing readout (mean-pool over tokens). Pure torch, ZERO dependency on
`kaggle_zerograd_moe` -- this file (and the rest of `phase_h/`) can move to a separate repo unchanged.

Isolation (ADR-005): never imported by the submission path; never imports the submission module.
"""
import math
import torch
import torch.nn as nn


class PhConfig:
    def __init__(self, vocab, seq_len, d_model=128, n_layers=4, n_heads=4, d_ff=None, n_cls=3, dropout=0.1):
        self.vocab = vocab; self.seq_len = seq_len; self.d_model = d_model
        self.n_layers = n_layers; self.n_heads = n_heads; self.d_ff = d_ff or 4 * d_model
        self.n_cls = n_cls; self.dropout = dropout


class Block(nn.Module):                                            # pre-LN Transformer block (bidirectional self-attn + FFN)
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model); self.ln2 = nn.LayerNorm(c.d_model)
        self.attn = nn.MultiheadAttention(c.d_model, c.n_heads, dropout=c.dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(c.d_model, c.d_ff), nn.GELU(),
                                nn.Dropout(c.dropout), nn.Linear(c.d_ff, c.d_model))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        a = self.ln1(x)
        a, _ = self.attn(a, a, a, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class PhTransformer(nn.Module):                                   # token+pos embedding -> N blocks -> mean-pool -> classifier
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model, padding_idx=0)
        self.pos = nn.Embedding(c.seq_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.n_cls)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):                                          # x [B,T] long, pad token = 0
        B, T = x.shape
        pad = (x == 0)                                            # [B,T] True = pad (ignored by attn + pool)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok(x) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)
        h = self.ln(h)
        m = (~pad).float().unsqueeze(-1)                          # non-collapsing readout: mean over non-pad tokens
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class PhCausalLM(nn.Module):                                      # GPT-style decoder (causal) for generative tasks (G2b)
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model, padding_idx=0)
        self.pos = nn.Embedding(c.seq_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, idx, targets=None):                        # idx [B,T]; returns logits [B,T,V] (+loss if targets)
        B, T = idx.shape
        cm = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)   # causal: mask future keys
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        h = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h, attn_mask=cm)                              # no key_padding_mask (causal + right-pad; loss ignores pad)
        logits = self.lm_head(self.ln(h))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, eos=None):                  # greedy decode
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.c.seq_len:])
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos is not None and (nxt == eos).all(): break
        return idx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
