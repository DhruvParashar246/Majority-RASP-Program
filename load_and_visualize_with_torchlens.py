import os, sys
sys.path.append(os.path.dirname(__file__))  # add current folder to module search path
from tracr_transformer_pt import TracrTransformerPT
# load_and_visualize_with_torchlens.py
import numpy as np
import torch
import torchlens as tl
from tracr_transformer_pt import TracrTransformerPT

# Instantiate the mirror model with your inferred hyperparams
model = TracrTransformerPT(
    vocab_size=4,
    max_seq_len=11,
    d_model=24,
    n_heads=3,
    head_dim=4,   # ★ critical — this makes 3*4 = 12 projection dim
    n_layers=3,
    d_mlp=4
)


# Load NPZ exported from your JAX/Haiku model
npz = np.load("tracr_majority_params.npz")

# Helper: copy (with optional transpose) into a torch parameter tensor
def copy_(pt_tensor, arr, transpose=False):
    t = torch.tensor(arr)
    if transpose:
        t = t.T
    assert tuple(t.shape) == tuple(pt_tensor.shape), f"Shape mismatch: src {t.shape} != dst {pt_tensor.shape}"
    with torch.no_grad():
        pt_tensor.copy_(t)

sd = model.state_dict()

# ---- Embeddings (same layout as PyTorch, no transpose) ----
copy_(sd["token_emb.weight"], npz["token_embed__embeddings"])
copy_(sd["pos_emb.weight"],   npz["pos_embed__embeddings"])

# ---- Per-layer mappings (notice: JAX Linear 'w' is (in, out); PyTorch is (out, in) -> transpose=True) ----
for i in range(3):  # layers 0..2
    # Attention projections
    copy_(sd[f"layers.{i}.attn.W_q.weight"], npz[f"transformer__layer_{i}__attn__query__w"], transpose=True)  # (24,12)->(12,24)
    copy_(sd[f"layers.{i}.attn.W_q.bias"],   npz[f"transformer__layer_{i}__attn__query__b"])
    copy_(sd[f"layers.{i}.attn.W_k.weight"], npz[f"transformer__layer_{i}__attn__key__w"],   transpose=True)
    copy_(sd[f"layers.{i}.attn.W_k.bias"],   npz[f"transformer__layer_{i}__attn__key__b"])
    copy_(sd[f"layers.{i}.attn.W_v.weight"], npz[f"transformer__layer_{i}__attn__value__w"], transpose=True)
    copy_(sd[f"layers.{i}.attn.W_v.bias"],   npz[f"transformer__layer_{i}__attn__value__b"])

    # Attention output projection ("linear")
    copy_(sd[f"layers.{i}.attn.W_o.weight"], npz[f"transformer__layer_{i}__attn__linear__w"], transpose=True)  # (12,24)->(24,12)
    copy_(sd[f"layers.{i}.attn.W_o.bias"],   npz[f"transformer__layer_{i}__attn__linear__b"])

    # MLP 24->4->24
    copy_(sd[f"layers.{i}.mlp.0.weight"], npz[f"transformer__layer_{i}__mlp__linear_1__w"], transpose=True)  # (24,4)->(4,24)
    copy_(sd[f"layers.{i}.mlp.0.bias"],   npz[f"transformer__layer_{i}__mlp__linear_1__b"])
    copy_(sd[f"layers.{i}.mlp.2.weight"], npz[f"transformer__layer_{i}__mlp__linear_2__w"], transpose=True)  # (4,24)->(24,4)
    copy_(sd[f"layers.{i}.mlp.2.bias"],   npz[f"transformer__layer_{i}__mlp__linear_2__b"])

# Commit weights to the module
model.load_state_dict(sd)

# ---- TorchLens over the SAME model ----
# If you have your exact token-indexing, encode it here. For a quick diagram, dummy IDs in [0..3] work.
x = torch.randint(low=0, high=4, size=(1, 6))  # (B=1, T=6) e.g., [BOS, 1,0,1,1,0] with the right indices if you prefer
model.eval()

# 1) Save full forward history AND 2) render the layered graph (unrolled by layer):
log = tl.log_forward_pass(model, x, vis_opt="unrolled")
tl.show_model_graph(model, (x,), vis_opt="unrolled", file_name="torchlens_majority_graph")
