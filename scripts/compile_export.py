#!/usr/bin/env python3
# scripts/compile_export.py
import os, sys, json
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

# Add local tracr paths if they exist; otherwise rely on pip-installed package
for p in [
    REPO_ROOT / "external" / "Tracr" / "tracr",  # optional vendored location
    REPO_ROOT / "Tracr" / "tracr",               # older local layout
    REPO_ROOT / "tracr",                         # just in case
]:
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

# Now import (works with either local path OR pip-installed package)
from tracr.compiler import compiling
from tracr.rasp import rasp


# -------- Config --------
VOCAB = {0, 1}
MAX_SEQ_LEN = 10
COMPILER_BOS = "BOS"
COMPILER_PAD = "PAD"
CAUSAL = True
EXAMPLE = [1, 0, 1, 1, 0]

ART = REPO_ROOT / "artifacts"
ART.mkdir(exist_ok=True)

def majority_program():
    # majority = 2 * (#ones) - seq_len
    all_pos = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ)
    ones_vec = rasp.Map(lambda t: t, rasp.tokens)           # 0/1 numeric
    num_ones = rasp.Aggregate(all_pos, ones_vec)
    ones_const = rasp.Map(lambda _: 1, rasp.tokens)
    seq_len = rasp.Aggregate(all_pos, ones_const)
    return (2 * num_ones) - seq_len

def compile_tracr(prog):
    kw = dict(vocab=VOCAB, max_seq_len=MAX_SEQ_LEN, compiler_bos=COMPILER_BOS)
    try:
        return compiling.compile_rasp_to_model(prog, compiler_pad=COMPILER_PAD, causal=CAUSAL, **kw)
    except TypeError:
        pass
    try:
        return compiling.compile_rasp_to_model(prog, causal=CAUSAL, **kw)
    except TypeError:
        return compiling.compile_rasp_to_model(prog, **kw)

def get_tok2id_or_fallback(model):
    cands = [
        getattr(model, "tokenizer", None),
        getattr(model, "vocab", None),
        getattr(model, "vocabulary", None),
        getattr(getattr(model, "transformer", None), "tokenizer", None),
        getattr(getattr(model, "transformer", None), "vocab", None),
        getattr(getattr(model, "transformer", None), "vocabulary", None),
    ]
    for obj in cands:
        if obj is None: continue
        if isinstance(obj, dict) and obj: return obj
        if hasattr(obj, "token_to_id") and isinstance(obj.token_to_id, dict):
            return obj.token_to_id
        if hasattr(obj, "id_to_token") and isinstance(obj.id_to_token, dict):
            return {tok: int(i) for i, tok in obj.id_to_token.items()}
    # fallback mapping (deterministic)
    ordered = [COMPILER_BOS] + sorted(list(VOCAB), key=lambda x: repr(x)) + [COMPILER_PAD]
    mapping = {tok: i for i, tok in enumerate(ordered)}
    print("[WARN] Using fallback token mapping:", mapping)
    return mapping

def export_params_npz(compiled, out_path: Path, keys_path: Path):
    try:
        params = compiled.params
    except AttributeError:
        params = compiled.weights

    leaves = []
    def walk(path, node):
        if isinstance(node, dict):
            for k, v in node.items(): walk(path + (k,), v)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node): walk(path + (str(i),), v)
        else:
            leaves.append(("/".join(path), np.array(node)))

    walk((), params)
    npz_dict = {k.replace("/", "__"): v for k, v in leaves}
    np.savez(out_path, **npz_dict)
    keys_path.write_text(json.dumps(sorted(npz_dict.keys()), indent=2))
    print(f"Exported params -> {out_path}  (keys -> {keys_path})")

def main():
    print("Compiling RASP → Tracr transformer…")
    compiled = compile_tracr(majority_program())
    print("Done.\n")

    # tokenizer mapping: only write fallback if no existing discovered mapping
    tok_map_path = ART / "token_to_id.json"
    tok2id = get_tok2id_or_fallback(compiled)
    if not tok_map_path.exists():
        tok_map_path.write_text(json.dumps(tok2id))
        print("Saved token_to_id.json")
    else:
        print("token_to_id.json exists; keeping discovered mapping.")

    # exact tokens used for the reference pass
    tokens = [COMPILER_BOS] + EXAMPLE
    (ART / "input_tokens.json").write_text(json.dumps(tokens))
    print("Saved input_tokens.json")

    # forward pass on assembled model
    out = compiled.apply(tokens)
    arr = np.array(getattr(out, "transformer_output", out), dtype=np.float32)
    if arr.ndim == 2: arr = arr[None, ...]
    np.save(ART / "tracr_output.npy", arr)
    print(f"Saved tracr_output.npy with shape {arr.shape} (dtype={arr.dtype})")

    # export params from THIS compiled model
    export_params_npz(compiled, ART / "tracr_majority_params.npz", ART / "tracr_param_keys.json")

    print("\nNow run:")
    print("  python scripts/parity_check.py")

if __name__ == "__main__":
    main()
