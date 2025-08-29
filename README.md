# Majority RASP → Tracr → PyTorch (Parity-Checked)

![Parity check](https://github.com/DhruvParashar246/Majority-RASP-Program/actions/workflows/parity.yml/badge.svg)

This repo compiles a simple **RASP** program (computes majority of a binary sequence) to a **Tracr** transformer, exports the Tracr model’s parameters, and loads them into a small **PyTorch** mirror. A parity check verifies that the PyTorch forward pass matches Tracr’s outputs exactly (up to numerical tolerance).

- **One-pass compile & export**: compile RASP → Tracr; save reference activations and weights from the **same** compiled model.
- **Deterministic parity**: the checker auto-discovers BOS/0/1/PAD embedding row mapping and writes it to `artifacts/token_to_id.json`.
- **CI**: GitHub Actions runs the parity check on every push/PR.

---

## Table of Contents
- [Project Layout](#project-layout)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Development Tips](#development-tips)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Layout

├─ .github/workflows/parity.yml # CI: fails if parity breaks
├─ scripts/
│ ├─ compile_export.py # compile RASP → Tracr, save outputs, export params
│ └─ parity_check.py # build PT mirror, load weights, verify parity
├─ src/
│ └─ tracr_pt_model.py # minimal PyTorch transformer (Attn→MLP, causal, √d_head)
├─ artifacts/ # build outputs (gitignored; .gitkeep tracked)
│ ├─ input_tokens.json
│ ├─ token_to_id.json
│ ├─ tracr_output.npy
│ ├─ tracr_majority_params.npz
│ └─ tracr_param_keys.json
└─ README.md


> **Note:** `artifacts/` is ignored by git; scripts populate it on each run.

---

## Quickstart

### (recommended) Python 3.11 in a virtualenv
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

### minimal deps
pip install numpy "jax[cpu]" dm-haiku
pip install --index-url https://download.pytorch.org/whl/cpu torch

### install Tracr (via pip, or vendor it under external/Tracr)
pip install git+https://github.com/google-deepmind/tracr.git


Run the pipeline:

### 1) Compile RASP → Tracr, save activations, export params (to artifacts/)
python scripts/compile_export.py

### 2) Verify PyTorch parity (discovers BOS/0/1/PAD mapping & saves it)
python scripts/parity_check.py

Expected:

Outputs match: True
Max abs diff: ~1e-6 (or 0)

## Installation

You can use either approach:

A) Pip-install Tracr (simple; used in CI)
pip install git+https://github.com/google-deepmind/tracr.git

B) Vendor locally
external/Tracr/         # clone of https://github.com/google-deepmind/tracr

compile_export.py auto-detects Tracr in external/Tracr/tracr or Tracr/tracr if present; otherwise it imports the pip-installed package.

## How It Works

### The RASP program

* The RASP program computes a majority score over a binary token sequence:

* Let #1 be the number of ones in the input.

* Let L be the sequence length (excluding BOS).

* The program outputs 2 * #1 - L (positive ⇒ majority ones; negative ⇒ majority zeros; zero ⇒ tie).

### Compile once, compare everywhere

1. Compile the RASP program to a Tracr transformer (Haiku/JAX backend).

2. Save:

  * tracr_output.npy — the Tracr model’s hidden states (reference).

  * tracr_majority_params.npz — all Tracr weights (NPZ).

  * input_tokens.json — the exact token sequence used for the reference pass.

3. Load the NPZ into a tiny PyTorch mirror with the same math:

  * Attn → MLP (sequential residuals)

  * causal attention mask

  * softmax scaling 1 / sqrt(head_dim)

  * head split inferred from NPZ; here it’s n_heads = 2, head_dim = 12 because proj_dim = 24.

4. Discover the true BOS/0/1/PAD row order by trying 24 permutations once; save the mapping to artifacts/token_to_id.json.

5. Assert parity: compare PT vs Tracr tensors element-wise.

This approach avoids:

* Basis drift between separate compiles (we export from the same compiled model we reference).

* Tokenizer ambiguity (we learn and persist the mapping once).

## Troubleshooting

### ModuleNotFoundError: No module named 'tracr'

* Ensure you pip-installed Tracr (see Installation), or vendor it under external/Tracr/.

* compile_export.py already searches common local paths and otherwise imports the pip package.

### Parity mismatch on CI

* The workflow installs Tracr via pip; local vendored code isn’t required.

* Ensure scripts/parity_check.py ends with: 
  * import sys
  * sys.exit(0 if ok else 1)

## Development Tips

* Re-run compile/export whenever you change the RASP program or compiler options (MAX_SEQ_LEN, BOS/PAD, causal).

* Add a Makefile for convenience:

  * build:  ; python scripts/compile_export.py
  * check:  ; python scripts/parity_check.py
  * clean:  ; rm -rf artifacts/*

* Protect main with the Parity check workflow required before merges.

## Citing Tracr

@article{lindner2023tracr,
  title = {Tracr: Compiled Transformers as a Laboratory for Interpretability},
  author = {Lindner, David and Kramár, János and Rahtz, Matthew and McGrath, Thomas and Mikulik, Vladimir},
  journal={arXiv preprint arXiv:2301.05062},
  year={2023}
}
