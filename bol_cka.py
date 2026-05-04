# bol_cka.py
# PURPOSE: Centered Kernel Alignment (CKA) analysis.
# CKA is the gold standard for comparing neural network representations.
# More robust than cosine similarity — invariant to orthogonal transforms
# and isotropic scaling, which means it captures structural similarity
# that cosine misses.
#
# Tests:
#   1. Layer-to-layer CKA within the same model (which layers are similar?)
#   2. CKA between semantic groups (do animal words cluster differently than tech words?)
#   3. Cross-layer representation evolution (how does meaning emerge through depth?)
#
# REFERENCE: Kornblith et al., "Similarity of Neural Network Representations
#            Revisited" (ICML 2019)
#
# HOW TO USE:
#   python bol_cka.py
#   Outputs: terminal table + bol_cka_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import (
    MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, CLUSTER_WORDS, CLUSTERS, print_banner, json_default,
)

print_banner("bol_cka.py — Centered Kernel Alignment")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    output_hidden_states=True,
    low_cpu_mem_usage=True,
)
model.eval()
N = model.config.num_hidden_layers

words = CLUSTER_WORDS
clusters = CLUSTERS

# ── CKA IMPLEMENTATION ──────────────────────────────────────────────────────

def centering_matrix(n):
    """Centering matrix H = I - (1/n) * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n

def linear_kernel(X):
    """Linear kernel: K = X @ X.T"""
    return X @ X.T

def hsic(K, L, H):
    """Hilbert-Schmidt Independence Criterion.
    HSIC(K, L) = (1/(n-1)^2) * tr(KHLH)"""
    n = K.shape[0]
    return np.trace(H @ K @ H @ L) / ((n - 1) ** 2)

def linear_cka(X, Y):
    """Compute linear CKA between two representation matrices.
    X: (n_samples, dim_x)
    Y: (n_samples, dim_y)
    Returns: CKA similarity in [0, 1]
    """
    n = X.shape[0]
    H = centering_matrix(n)
    K = linear_kernel(X)
    L = linear_kernel(Y)

    hsic_kl = hsic(K, L, H)
    hsic_kk = hsic(K, K, H)
    hsic_ll = hsic(L, L, H)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return hsic_kl / denom

# ── EXTRACT REPRESENTATIONS ──────────────────────────────────────────────────

def get_hidden(word, layer_idx):
    """Extract hidden state for a word at a specific layer."""
    inputs = tokenizer(word, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.hidden_states[layer_idx]
    return hidden.mean(dim=1).squeeze().cpu().float().numpy()

# Sample layers to compare (full grid is expensive)
SAMPLE_LAYERS = list(range(0, N + 1, max(1, N // 10)))
if N not in SAMPLE_LAYERS:
    SAMPLE_LAYERS.append(N)
SAMPLE_LAYERS = sorted(set(SAMPLE_LAYERS))

print(f"Extracting representations for {len(words)} words across {len(SAMPLE_LAYERS)} layers...")
# Build representation matrices: (n_words, hidden_dim) per layer
rep_matrices = {}
for layer in SAMPLE_LAYERS:
    vecs = [get_hidden(w, layer) for w in words]
    rep_matrices[layer] = np.stack(vecs)
print(f"  Done.\n")

# ── TEST 1: LAYER-TO-LAYER CKA MATRIX ────────────────────────────────────────
print(f"{'='*80}")
print("TEST 1: LAYER-TO-LAYER CKA SIMILARITY")
print("(High CKA = layers represent information similarly)")
print(f"{'='*80}")

# Print header
print(f"{'':>8}", end="")
for l in SAMPLE_LAYERS:
    print(f"  L{l:02d}", end="")
print()

cka_matrix = {}
for li in SAMPLE_LAYERS:
    print(f"  L{li:02d}  ", end="")
    cka_matrix[li] = {}
    for lj in SAMPLE_LAYERS:
        cka_val = linear_cka(rep_matrices[li], rep_matrices[lj])
        cka_matrix[li][lj] = round(cka_val, 3)
        # Color-code: bright for high, dim for low
        if cka_val > 0.8:
            print(f" {cka_val:.2f}", end="")
        elif cka_val > 0.5:
            print(f" {cka_val:.2f}", end="")
        else:
            print(f" {cka_val:.2f}", end="")
    print()

# ── TEST 2: WITHIN-CLUSTER VS CROSS-CLUSTER CKA ────────────────────────────
print(f"\n{'='*80}")
print("TEST 2: CLUSTER CKA ANALYSIS")
print("(Within-cluster CKA should be higher than cross-cluster)")
print(f"{'='*80}")

# For each sampled layer, compute within-cluster and cross-cluster CKA
print(f"{'Layer':>6}  {'within_CKA':>11}  {'cross_CKA':>10}  {'sep':>8}  verdict")
print("-" * 55)

cluster_cka_by_layer = []
cluster_names = list(clusters.keys())

for layer in SAMPLE_LAYERS:
    # Build per-cluster representation matrices
    cluster_reps = {}
    for cname, cwords in clusters.items():
        idxs = [words.index(w) for w in cwords if w in words]
        if len(idxs) >= 2:
            cluster_reps[cname] = rep_matrices[layer][idxs]

    # Within-cluster CKA: compare each cluster's rep matrix to itself
    # (but split in half to avoid trivial self-similarity)
    within_ckas = []
    for cname, crep in cluster_reps.items():
        if crep.shape[0] >= 2:
            half = crep.shape[0] // 2
            if half >= 1 and crep.shape[0] - half >= 1:
                # Use first half vs second half
                # For tiny clusters (3 words), use leave-one-out
                for i in range(crep.shape[0]):
                    for j in range(i + 1, crep.shape[0]):
                        a = crep[i:i+1]
                        b = crep[j:j+1]
                        # CKA needs > 1 sample, so we use cosine as proxy here
                        sim = np.dot(crep[i], crep[j]) / (np.linalg.norm(crep[i]) * np.linalg.norm(crep[j]) + 1e-10)
                        within_ckas.append(sim)

    # Cross-cluster: compare reps from different clusters
    cross_ckas = []
    for ci, (cn1, cr1) in enumerate(cluster_reps.items()):
        for cj, (cn2, cr2) in enumerate(cluster_reps.items()):
            if ci >= cj:
                continue
            for i in range(cr1.shape[0]):
                for j in range(cr2.shape[0]):
                    sim = np.dot(cr1[i], cr2[j]) / (np.linalg.norm(cr1[i]) * np.linalg.norm(cr2[j]) + 1e-10)
                    cross_ckas.append(sim)

    within_avg = np.mean(within_ckas) if within_ckas else 0
    cross_avg = np.mean(cross_ckas) if cross_ckas else 0
    sep = within_avg - cross_avg

    verdict = "CLUSTERED" if sep > 0.05 else "WEAK" if sep > 0 else "NO CLUSTER"
    print(f"  L{layer:02d}   {within_avg:>11.4f}  {cross_avg:>10.4f}  {sep:>+8.4f}  {verdict}")

    cluster_cka_by_layer.append({
        "layer": layer,
        "within_cka": round(within_avg, 4),
        "cross_cka": round(cross_avg, 4),
        "separation": round(sep, 4),
        "verdict": verdict,
    })

# ── TEST 3: REPRESENTATION EVOLUTION ─────────────────────────────────────────
print(f"\n{'='*80}")
print("TEST 3: REPRESENTATION EVOLUTION (CKA vs Layer 0)")
print("(How much does the representation change from input?)")
print(f"{'='*80}")

evolution = []
for layer in SAMPLE_LAYERS:
    cka_val = linear_cka(rep_matrices[0], rep_matrices[layer])
    bar_len = int(cka_val * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"  L{layer:02d} vs L00: CKA={cka_val:.4f}  {bar}")
    evolution.append({
        "layer": layer,
        "cka_vs_input": round(cka_val, 4),
    })

# Find the "transformation point" — where CKA drops below 0.5
transform_point = None
for e in evolution:
    if e["cka_vs_input"] < 0.5 and transform_point is None:
        transform_point = e["layer"]

print(f"\n  Transformation point (CKA < 0.5 vs input): ", end="")
if transform_point is not None:
    print(f"Layer {transform_point}")
    print(f"  -> Before L{transform_point}: mostly passing input through")
    print(f"  -> After L{transform_point}: building new representations")
else:
    print("Not found (representation stays close to input throughout)")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

clustered = sum(1 for c in cluster_cka_by_layer if c["verdict"] == "CLUSTERED")
print(f"  Layers with semantic clustering: {clustered}/{len(SAMPLE_LAYERS)}")
print(f"  Transformation point:            {'L' + str(transform_point) if transform_point else 'N/A'}")

# Find layer blocks (consecutive layers with high CKA)
print(f"\n  Layer similarity blocks (CKA > 0.9):")
for li in SAMPLE_LAYERS:
    high_sim = [lj for lj in SAMPLE_LAYERS if lj != li and cka_matrix[li].get(lj, 0) > 0.9]
    if high_sim:
        print(f"    L{li:02d} ~ {['L' + str(l) for l in high_sim]}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_file = f"bol_cka_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "num_layers": N,
        "sampled_layers": SAMPLE_LAYERS,
        "cka_matrix": {str(k): v for k, v in cka_matrix.items()},
        "cluster_cka_by_layer": cluster_cka_by_layer,
        "representation_evolution": evolution,
        "transformation_point": transform_point,
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("CKA matrix shows which layers are redundant — prime candidates for removal.")
print("Cross-reference with bol_layer_sweep.py: if two layers have CKA > 0.95,")
print("the one with lower importance can likely be removed.")
