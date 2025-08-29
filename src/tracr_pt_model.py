# src/tracr_pt_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.causal = causal
        proj = n_heads * head_dim
        self.W_q = nn.Linear(d_model, proj, bias=True)
        self.W_k = nn.Linear(d_model, proj, bias=True)
        self.W_v = nn.Linear(d_model, proj, bias=True)
        self.W_o = nn.Linear(proj, d_model, bias=True)

    def forward(self, x):
        B, T, _ = x.shape
        def split(L):
            y = L(x).view(B, T, self.n_heads, self.head_dim)
            return y.permute(0, 2, 1, 3)  # (B, H, T, D)
        q, k, v = split(self.W_q), split(self.W_k), split(self.W_v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.W_o(ctx)

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp, bias=True)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=True)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_mlp):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, n_heads, head_dim, causal=True)
        self.mlp  = MLP(d_model, d_mlp)
    def forward(self, x):
        x = x + self.attn(x)   # Attn → MLP (sequential residuals)
        x = x + self.mlp(x)
        return x

class TracrTransformerPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, d_mlp, n_heads=2, head_dim=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, head_dim, d_mlp) for _ in range(n_layers)
        ])

    def forward(self, token_ids):
        B, T = token_ids.shape
        pos = torch.arange(T, device=token_ids.device)  # positions start at 0
        x = self.token_emb(token_ids) + self.pos_emb(pos)[None, :, :]
        for blk in self.layers:
            x = blk(x)
        return x
