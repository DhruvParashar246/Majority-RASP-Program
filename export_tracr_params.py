# export_tracr_params.py
import numpy as np
import jax
from tracr.compiler import compiling
from tracr.rasp import rasp

VOCAB = {0, 1}
MAX_SEQ_LEN = 10
COMPILER_BOS = "BOS"

def majority_score_program():
    all_pos = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)
    select_ones = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ)
    num_ones = rasp.Aggregate(select_ones, rasp.tokens)
    seq_len = rasp.Aggregate(all_pos, rasp.tokens * 0 + 1)
    majority_score = num_ones - (seq_len - num_ones)
    return majority_score

print("Compiling RASP → JAX/Haiku transformer…")
compiled = compiling.compile_rasp_to_model(
    majority_score_program(),
    vocab=VOCAB,
    max_seq_len=MAX_SEQ_LEN,
    compiler_bos=COMPILER_BOS,
)
print("Done.")

# --- Inspect param tree to learn names you must map ---
# Depending on Tracr version, compiled.params or compiled.weights exists.
# Print keys so we can map them into PyTorch:
try:
    params = compiled.params
except AttributeError:
    params = compiled.weights  # fallback if older API

flat, treedef = jax.tree_util.tree_flatten(params)
leaves_with_paths = []

def track_paths(path, node):
    if isinstance(node, (dict,)):
        for k,v in node.items():
            track_paths(path + (k,), v)
    else:
        leaves_with_paths.append(("/".join(path), node))

track_paths((), params)

print("\n=== JAX PARAM KEYS (preview) ===")
for k, v in leaves_with_paths:
    print(f"{k}: shape={np.array(v).shape}")
print("=== end ===\n")

# --- Save to NPZ with slash->double_underscore to be filesystem-friendly ---
npz_dict = {}
for k, v in leaves_with_paths:
    safe = k.replace("/", "__")
    npz_dict[safe] = np.array(v)

np.savez("tracr_majority_params.npz", **npz_dict)
print("Exported => tracr_majority_params.npz")
