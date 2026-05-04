# bol_semantic_probe.py
# PURPOSE: The octopus test.
# Feed word pairs through the model, extract hidden states,
# measure cosine similarity. If related words cluster better than
# unrelated ones → the model understands without speaking.
# Run on each subject to compare pre vs post Muon geometry.
# SCIENTIFICALLY LOCKED: FP32, multi-layer scan, JSON output.
#
# HOW TO USE:
#   python bol_semantic_probe.py
#   Outputs: terminal table + bol_semantic_probe_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import torch.nn.functional as F
import json
import numpy as np
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import (
    MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, CLUSTER_WORDS, CLUSTERS, print_banner, json_default,
)

print_banner("bol_semantic_probe.py — Cluster Validation")

# Scan multiple layers at once (or set to [-1] for last layer only)
LAYERS = [0, 5, 10, 15, 20, 25, 29, 30]  # Layer indices to probe

words = CLUSTER_WORDS
clusters = CLUSTERS

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

# Clamp LAYERS to valid range
LAYERS = [l for l in LAYERS if l <= N]

def get_hidden(word, layer_idx):
    """Extract hidden state for a word at a specific layer."""
    inputs = tokenizer(word, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.hidden_states[layer_idx]
    return hidden.mean(dim=1).squeeze().cpu().float()

def cos(a, b):
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ── EXTRACT HIDDEN STATES ─────────────────────────────────────────────────────
print(f"Extracting hidden states for {len(words)} words across {len(LAYERS)} layers...")
vectors = {layer: {w: get_hidden(w, layer) for w in words} for layer in LAYERS}
print(f"  Done.\n")

# ── SIMILARITY MATRIX (Last Layer Only for Display) ──────────────────────────
display_layer = LAYERS[-1]
print(f"=== SIMILARITY MATRIX (Layer {display_layer}) ===")
print(f"{'':15}", end="")
for w in words:
    print(f"{w:>10}", end="")
print()

for w1 in words:
    print(f"{w1:15}", end="")
    for w2 in words:
        sim = cos(vectors[display_layer][w1], vectors[display_layer][w2])
        print(f"{sim:10.3f}", end="")
    print()

# ── CLUSTER VALIDATION ACROSS ALL LAYERS ──────────────────────────────────────
# FIX: Compare within-cluster similarity against CROSS-cluster similarity
# (not against "other_words" which was always empty)
print(f"\n{'='*85}")
print("CLUSTER VALIDATION BY LAYER")
print(f"{'='*85}")
print(f"{'Layer':>6}  {'Cluster':<12}  {'within':>8}  {'cross':>9}  {'sep':>8}  {'p-val':>8}  {'verdict':<15}")
print("-" * 85)

# Build a flat list of all cluster words for cross-cluster comparison
all_cluster_words = sum(clusters.values(), [])

layer_results = []

for layer in LAYERS:
    layer_data = {"layer": layer, "clusters": {}}
    for cname, cwords in clusters.items():
        # Within-cluster: pairwise similarity of words in this cluster
        within = [cos(vectors[layer][w1], vectors[layer][w2])
                  for i, w1 in enumerate(cwords) for w2 in cwords[i+1:]]

        # Cross-cluster: similarity of this cluster's words to words in OTHER clusters
        cross_words = [w for cn, cw in clusters.items() if cn != cname for w in cw]
        cross = [cos(vectors[layer][w1], vectors[layer][w2])
                 for w1 in cwords for w2 in cross_words]

        wa = np.mean(within) if within else 0
        ca = np.mean(cross) if cross else 0
        sep = wa - ca

        # Statistical test: is within-cluster similarity significantly higher?
        if len(within) >= 2 and len(cross) >= 2:
            try:
                stat, pval = scipy_stats.mannwhitneyu(within, cross, alternative='greater')
            except ValueError:
                pval = 1.0
        else:
            pval = 1.0

        verdict = ("CLUSTERS" if sep > 0.05 and pval < 0.05
                   else "WEAK"     if sep > 0
                   else "NO CLUSTER")

        sig = "*" if pval < 0.05 else ""
        print(f"  L{layer:02d}   {cname:<12}  {wa:>8.3f}  {ca:>9.3f}  {sep:>+8.3f}  {pval:>7.4f}{sig} {verdict:<15}")
        layer_data["clusters"][cname] = {
            "within_avg": round(wa, 4),
            "cross_avg": round(ca, 4),
            "separation": round(sep, 4),
            "p_value": round(pval, 6),
            "significant": pval < 0.05,
            "verdict": verdict,
        }
    layer_results.append(layer_data)

# ── FIND BEST LAYER FOR CLUSTERING ────────────────────────────────────────────
print(f"\n{'='*85}")
print("BEST LAYER FOR SEMANTIC CLUSTERING")
print(f"{'='*85}")

layer_avg_sep = []
for lr in layer_results:
    avg_sep = np.mean([c["separation"] for c in lr["clusters"].values()])
    layer_avg_sep.append((lr["layer"], avg_sep))

best_layer, best_sep = max(layer_avg_sep, key=lambda x: x[1])
print(f"  Best layer: L{best_layer} (avg separation: {best_sep:+.4f})")

best_data = next(lr for lr in layer_results if lr["layer"] == best_layer)
sorted_clusters = sorted(best_data["clusters"].items(), key=lambda x: x[1]["separation"], reverse=True)
print(f"\n  Cluster rankings at L{best_layer}:")
for rank, (cname, cdata) in enumerate(sorted_clusters, 1):
    sig = " (p<0.05)" if cdata["significant"] else ""
    print(f"    #{rank}  {cname:<12}  sep={cdata['separation']:+.4f}  {cdata['verdict']}{sig}")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
out_file = f"bol_semantic_probe_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "layers_probed": LAYERS,
        "num_words": len(words),
        "clusters": list(clusters.keys()),
        "best_layer": int(best_layer),
        "best_avg_separation": round(float(best_sep), 4),
        "layer_results": layer_results,
        "similarity_matrix_layer": display_layer,
        "similarity_matrix": {
            w1: {w2: round(cos(vectors[display_layer][w1], vectors[display_layer][w2]), 4) for w2 in words}
            for w1 in words
        },
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Cross-reference with bol_silhouette_<LABEL>.json to compare separation metrics.")
