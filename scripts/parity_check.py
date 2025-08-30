#!/usr/bin/env python3
# scripts/parity_check.py
import sys, json, itertools
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART  = ROOT / "artifacts"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ensure required artifacts exist
assert (ART / "tracr_majority_params.npz").exists(), "Missing NPZ. Run scripts/compile_export.py first."
assert (ART / "input_tokens.json").exists(),          "Missing input tokens. Run scripts/compile_export.py."
assert (ART / "tracr_output.npy").exists(),           "Missing Tracr output. Run scripts/compile_export.py."

# import model
sys.path.append(str(ROOT / "src"))
from tracr_pt_model import TracrTransformerPT

# ---- Load NPZ & infer dims ----
npz = np.load(ART / "tracr_majority_params.npz")
def get(k): return npz.get(k, npz[k.replace("/", "__")])

vocab_size, d_model = get("token_embed/embeddings").shape
max_len   = get("pos_embed/embeddings").shape[0]
proj_dim  = int(get("transformer/layer_0/attn/query/w").shape[1])  # JAX (in,out)
d_mlp     = int(get("transformer/layer_0/mlp/linear_1/w").shape[1])
n_layers  = sum((f"transformer/layer_{i}/attn/query/w".replace("/", "__") in npz) for i in range(64))

# guard: we expect exactly 4 tokens (BOS, 0, 1, PAD)
if vocab_size != 4:
    raise ValueError(f"Expected vocab_size=4, got {vocab_size}. Token mapping search assumes BOS/0/1/PAD.")

# matched config found earlier
n_heads, head_dim = 2, proj_dim // 2

print(f"Inferred -> d_model={d_model}, vocab={vocab_size}, max_seq_len={max_len}, "
      f"layers={n_layers}, proj_dim={proj_dim}, d_mlp={d_mlp}, heads={n_heads}, head_dim={head_dim}")

# ---- Build PT model & load weights ----
model = TracrTransformerPT(vocab_size, max_len, int(d_model), int(n_layers), int(d_mlp),
                           n_heads=int(n_heads), head_dim=int(head_dim)).eval()

def load_linear(L, w_key, b_key):
    w = torch.from_numpy(get(w_key)).float().t().contiguous()  # (in,out) -> (out,in)
    b = torch.from_numpy(get(b_key)).float()
    with torch.no_grad(): L.weight.copy_(w); L.bias.copy_(b)

with torch.no_grad():
    model.token_emb.weight.copy_(torch.from_numpy(get("token_embed/embeddings")).float())
    model.pos_emb.weight.copy_(torch.from_numpy(get("pos_embed/embeddings")).float())

for i in range(n_layers):
    P = f"transformer/layer_{i}"
    blk = model.layers[i]
    load_linear(blk.attn.W_q, f"{P}/attn/query/w",  f"{P}/attn/query/b")
    load_linear(blk.attn.W_k, f"{P}/attn/key/w",    f"{P}/attn/key/b")
    load_linear(blk.attn.W_v, f"{P}/attn/value/w",  f"{P}/attn/value/b")
    load_linear(blk.attn.W_o, f"{P}/attn/linear/w", f"{P}/attn/linear/b")
    load_linear(blk.mlp.fc1,  f"{P}/mlp/linear_1/w", f"{P}/mlp/linear_1/b")
    load_linear(blk.mlp.fc2,  f"{P}/mlp/linear_2/w", f"{P}/mlp/linear_2/b")

# ---- Read tokens & Tracr reference ----
tokens = json.loads((ART / "input_tokens.json").read_text())    # ["BOS", 1, 0, 1, 1, 0]
ref    = torch.from_numpy(np.load(ART / "tracr_output.npy")).float()

# ---- Discover BOS/0/1/PAD mapping once ----
TOKS = ["BOS", "0", "1", "PAD"]
best = None
for perm in itertools.permutations(range(vocab_size), vocab_size):
    tok2id = {TOKS[i]: perm[i] for i in range(vocab_size)}
    ids = [tok2id["BOS"]] + [tok2id[str(t)] for t in tokens[1:]]
    ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad(): out = model(ids)
    md = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, atol=1e-5)
    cand = (tok2id, md, ok)
    if best is None or md < best[1]: best = cand
    if ok: break

tok2id, md, ok = best
(ART / "token_to_id.json").write_text(json.dumps(tok2id, indent=2))

print("\n--- Mapping search ---")
print(f"tok2id={tok2id}, max_diff={md:.6g}, match={ok}")
print("\n--- Sanity Check ---")
print(f"Outputs match: {ok}")
print(f"Max abs diff: {md:.6g}")

# --- Write tracked summary ---
summary = {
    "match": bool(ok),
    "max_abs_diff": float(md),
    "config": {
        "d_model": int(d_model),
        "layers": int(n_layers),
        "proj_dim": int(proj_dim),
        "n_heads": int(n_heads),
        "head_dim": int(head_dim),
        "causal": True,
        "residual_order": "Attn->MLP",
        "scale": "sqrt(head_dim)",
        "pos_shift": 0
    },
    "token_to_id": tok2id,
}
(RESULTS / "parity_summary.json").write_text(json.dumps(summary, indent=2))
print("\nWrote results/parity_summary.json")

sys.exit(0 if ok else 1)
