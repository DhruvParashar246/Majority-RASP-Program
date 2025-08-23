# tracr_transformer_pt.py (replace your attention + block signatures with this)

import math
import torch
import torch.nn as nn

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model=24, n_heads=3, head_dim=4, bias=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.proj_dim = n_heads * head_dim  # = 12 (matches your JAX dump)

        self.W_q = nn.Linear(d_model, n_heads * head_dim, bias=bias)  # 24 -> 12
        self.W_k = nn.Linear(d_model, n_heads * head_dim, bias=bias)  # 24 -> 12
        self.W_v = nn.Linear(d_model, n_heads * head_dim, bias=bias)  # 24 -> 12
        self.W_o = nn.Linear(n_heads * head_dim, d_model, bias=bias)  # 12 -> 24


    def forward(self, x):
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,4)
        k = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v)  # (B,H,T,4)
        y = y.transpose(1, 2).contiguous().view(B, T, self.proj_dim)  # (B,T,12)
        return self.W_o(y)  # (B,T,24)

class EncoderBlock(nn.Module):
    def __init__(self, d_model=24, n_heads=3, head_dim=4, d_mlp=4):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model=d_model, n_heads=n_heads, head_dim=head_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=True),  # 24 -> 4
            nn.GELU(),
            nn.Linear(d_mlp, d_model, bias=True),  # 4 -> 24
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class TracrTransformerPT(nn.Module):
    def __init__(self, vocab_size=4, max_seq_len=11, d_model=24, n_heads=3, head_dim=4, n_layers=3, d_mlp=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, head_dim, d_mlp) for _ in range(n_layers)])

    def forward(self, token_ids):
        B, T = token_ids.shape
        x = self.token_emb(token_ids) + self.pos_emb(torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T))
        for blk in self.layers:
            x = blk(x)
        return x  # (B, T, 24)
