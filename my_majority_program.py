import os, sys
tracr_path = os.path.join(os.path.dirname(__file__), "Tracr", "tracr")
sys.path.insert(0, tracr_path)

import numpy as np
import jax
import jax.random as random

from tracr.compiler import compiling
from tracr.rasp import rasp

# --- The robust function to import show_model ---
# --- The robust function to import show_model (returns None if not found) ---
def _get_show_model():
    import importlib, pkgutil, tracr
    for modpath in ("tracr.compiler.visualization", "tracr.visualization"):
        try:
            mod = importlib.import_module(modpath)
            fn = getattr(mod, "show_model", None)
            if callable(fn):
                return fn
        except Exception:
            pass
    for _, name, _ in pkgutil.walk_packages(tracr.__path__, tracr.__name__ + "."):
        try:
            mod = importlib.import_module(name)
            fn = getattr(mod, "show_model", None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None  # <- do NOT raise here
# --- Fallback: render a clean Tracr-style diagram from compiled params ---
def render_block_diagram_from_compiled(compiled_model, out_basename="tracr_majority_graph"):
    import re
    from graphviz import Digraph
    # get params from the compiled model
    try:
        params = compiled_model.params
    except AttributeError:
        params = compiled_model.weights

    # flatten nested dict into "path" -> ndarray
    flat = {}
    def walk(path, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(path + (k,), v)
        else:
            flat["/".join(path)] = np.array(node)
    walk((), params)

    # read shapes
    tok_key = next(k for k in flat if k.endswith("token_embed/embeddings"))
    pos_key = next(k for k in flat if k.endswith("pos_embed/embeddings"))
    vocab_size, d_model = flat[tok_key].shape
    max_seq_len = flat[pos_key].shape[0]

    layer_nums = sorted({int(m.group(1))
                         for k in flat
                         for m in [re.search(r"transformer/layer_(\d+)/", k)]
                         if m})
    # attn proj and mlp hidden from layer 0
    proj_dim = flat[f"transformer/layer_{layer_nums[0]}/attn/query/w"].shape[1]
    mlp_hidden = flat[f"transformer/layer_{layer_nums[0]}/mlp/linear_1/w"].shape[1]

    dot = Digraph("tracr_majority_transformer", format="pdf")
    dot.attr(rankdir="LR", fontsize="12", labelloc="t",
             label=f"Tracr-compiled Majority Transformer\n"
                   f"vocab={vocab_size}, d_model={d_model}, layers={len(layer_nums)}, "
                   f"proj_dim={proj_dim}, mlp_hidden={mlp_hidden}")

    with dot.subgraph(name="cluster_embed") as c:
        c.attr(label="Embeddings")
        c.node("tok_emb", f"Token Embedding\n({vocab_size}×{d_model})", shape="box")
        c.node("pos_emb", f"Positional Embedding\n({max_seq_len}×{d_model})", shape="box")
        c.node("sum", "Add", shape="circle")
        c.edges([("tok_emb", "sum"), ("pos_emb", "sum")])

    prev = "sum"
    for i in layer_nums:
        with dot.subgraph(name=f"cluster_layer_{i}") as c:
            c.attr(label=f"Encoder Block {i}")
            c.node(f"attn_{i}", f"MHA proj {proj_dim}", shape="box")
            c.node(f"add_attn_{i}", "Add", shape="circle")
            c.node(f"mlp_{i}", f"MLP {d_model}→{mlp_hidden}→{d_model}", shape="box")
            c.node(f"add_mlp_{i}", "Add", shape="circle")
        dot.edge(prev, f"attn_{i}")
        dot.edge(f"attn_{i}", f"add_attn_{i}")
        dot.edge(f"add_attn_{i}", f"mlp_{i}")
        dot.edge(f"mlp_{i}", f"add_mlp_{i}")
        prev = f"add_mlp_{i}"

    dot.node("out", f"Output\n(seq_len×{d_model})", shape="box")
    dot.edge(prev, "out")
    out_path = dot.render(out_basename, cleanup=True)
    print(f"Saved {out_path}")


show_model = _get_show_model()



VOCAB = {0, 1}
MAX_SEQ_LEN = 10
COMPILER_BOS = "BOS"

# --- Majority Score Program ---
def majority_score_program():
    """
    RASP program that outputs whether 1s or 0s are the majority.
    Positive = majority of 1s, Negative = majority of 0s, 0 = tie.
    """
    # Select all positions
    all_positions = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)

    # Count number of 1s
    select_ones = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ)
    num_ones = rasp.Aggregate(select_ones, rasp.tokens)

    # Count number of 0s (total length - num_ones)
    seq_length = rasp.Aggregate(all_positions, rasp.tokens * 0 + 1)  # sum of 1s over all positions
    num_zeros = seq_length - num_ones

    # Majority score = (#1s - #0s)
    majority_score = num_ones - num_zeros

    return majority_score

# --- Compile ---
print("Compiling majority RASP program to transformer model...")
compiled_model = compiling.compile_rasp_to_model(
    majority_score_program(),
    vocab=VOCAB,
    max_seq_len=MAX_SEQ_LEN,
    compiler_bos=COMPILER_BOS,
)
print("Compilation complete!\n")

# --- Save transformer diagram ---
print("Generating transformer visualization...")
show_model = _get_show_model()
if show_model is not None:
    graph = show_model(compiled_model, max_seq_len=MAX_SEQ_LEN, return_graph=True)
    graph.render("tracr_majority_graph", format="pdf", cleanup=True)
else:
    print("Tracr show_model not found — using fallback renderer.")
    render_block_diagram_from_compiled(compiled_model, out_basename="tracr_majority_graph")
print("Diagram saved as tracr_majority_graph.pdf ✅\n")


# --- Example ---
example_input_sequence = [1, 0, 1, 1, 0]

print(f"Raw input sequence (no BOS): {example_input_sequence}")

# Prepend BOS manually, since Tracr expects it
input_with_bos = [COMPILER_BOS] + example_input_sequence
print(f"Input sequence with BOS: {input_with_bos}")

# Run model
output_logits = compiled_model.apply(input_with_bos)

# Interpret output
vocab_list = sorted(list(VOCAB)) + [COMPILER_BOS]
predicted_tokens = [vocab_list[np.argmax(logits)] for logits in output_logits]

print("\n--- Model Output ---")
print("Raw logits:\n", output_logits)
print("Predicted tokens:", predicted_tokens)

# --- Run RASP directly ---
rasp_output = majority_score_program()(example_input_sequence)
print("Raw RASP output:", rasp_output)
print("\nRASP execution output:", rasp_output)

majority_val = rasp_output[0]
if majority_val > 0:
    print("Majority element: 1 ✅")
elif majority_val < 0:
    print("Majority element: 0 ✅")
else:
    print("Tie between 0 and 1 🤝")
