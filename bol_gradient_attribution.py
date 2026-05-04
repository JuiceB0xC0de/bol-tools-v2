# bol_gradient_attribution.py
# PURPOSE: Gradient-based layer importance measurement.
# Instead of zeroing layers (binary on/off, very coarse), this computes
# how much each layer's parameters contribute to the model's ability to
# distinguish related from unrelated word pairs.
#
# Methods:
#   1. Gradient magnitude: ||dL/dW|| per layer — how sensitive is loss to this layer?
#   2. Fisher information: E[||dL/dW||^2] — expected importance under the data distribution
#   3. Gradient flow: track gradient norms through the network to find bottlenecks
#
# This gives a CONTINUOUS importance score instead of the binary "zero it and see what breaks"
# approach. More nuanced and less destructive than ablation.
#
# HOW TO USE:
#   python bol_gradient_attribution.py
#   Outputs: terminal table + bol_gradient_attribution_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import numpy as np
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, EVAL_TEXTS, print_banner, json_default

print_banner("bol_gradient_attribution.py — Gradient Layer Importance")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# NOTE: We need gradients, so we DON'T use torch.no_grad() here
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()  # Still eval mode (no dropout), but we'll compute gradients
N = model.config.num_hidden_layers

eval_texts = EVAL_TEXTS

# ── TEST 1: GRADIENT MAGNITUDE PER LAYER ─────────────────────────────────────
print(f"{'='*75}")
print("TEST 1: GRADIENT MAGNITUDE PER LAYER")
print("(Higher = more important for loss — the model relies on this layer)")
print(f"{'='*75}")

# Tokenize all eval texts
tokenized = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True)
labels = tokenized["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

if DEVICE == "cuda":
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    labels = labels.to(DEVICE)

# Forward + backward
model.zero_grad()
outputs = model(**tokenized, labels=labels)
loss = outputs.loss
loss.backward()

# Collect gradient stats per layer
print(f"\nLoss: {loss.item():.4f}\n")
print(f"{'Layer':>6}  {'grad_norm':>11}  {'grad_mean':>11}  {'param_norm':>11}  {'ratio':>8}  bar")
print("-" * 75)

layer_grad_stats = []
for i in range(N):
    grad_norms = []
    param_norms = []
    grad_means = []

    for name, param in model.named_parameters():
        if f".layers.{i}." not in name:
            continue
        if param.grad is None:
            continue
        if param.ndim < 2:
            continue

        g = param.grad.float()
        grad_norms.append(g.norm().item())
        grad_means.append(g.abs().mean().item())
        param_norms.append(param.data.float().norm().item())

    if not grad_norms:
        continue

    total_grad_norm = np.sqrt(sum(g ** 2 for g in grad_norms))
    total_param_norm = np.sqrt(sum(p ** 2 for p in param_norms))
    ratio = total_grad_norm / (total_param_norm + 1e-10)
    avg_grad_mean = np.mean(grad_means)

    bar = "█" * min(int(ratio * 500), 40)
    print(f"  L{i:02d}   {total_grad_norm:>11.6f}  {avg_grad_mean:>11.8f}  {total_param_norm:>11.2f}  {ratio:>8.6f}  {bar}")

    layer_grad_stats.append({
        "layer": i,
        "gradient_norm": round(total_grad_norm, 6),
        "gradient_mean": round(avg_grad_mean, 8),
        "parameter_norm": round(total_param_norm, 4),
        "gradient_to_param_ratio": round(ratio, 8),
    })

# ── TEST 2: COMPONENT-LEVEL GRADIENT ATTRIBUTION ────────────────────────────
print(f"\n{'='*75}")
print("TEST 2: COMPONENT-LEVEL GRADIENT ATTRIBUTION")
print("(Which component types carry the most gradient signal?)")
print(f"{'='*75}")

component_types = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]

print(f"\n{'Component':<25}  {'total_grad':>12}  {'avg_per_layer':>14}  {'% of total':>11}")
print("-" * 70)

component_grads = {}
total_grad = 0

for comp in component_types:
    grad_sum = 0
    count = 0
    for name, param in model.named_parameters():
        if comp not in name or param.grad is None or param.ndim < 2:
            continue
        grad_sum += param.grad.float().norm().item() ** 2
        count += 1
    grad_sum = np.sqrt(grad_sum)
    total_grad += grad_sum
    component_grads[comp] = {"total": grad_sum, "count": count}

for comp in component_types:
    cg = component_grads[comp]
    pct = (cg["total"] / total_grad * 100) if total_grad > 0 else 0
    avg = cg["total"] / cg["count"] if cg["count"] > 0 else 0
    print(f"  {comp:<23}  {cg['total']:>12.6f}  {avg:>14.6f}  {pct:>10.1f}%")

# ── TEST 3: GRADIENT FLOW (vanishing/exploding gradient detection) ───────────
print(f"\n{'='*75}")
print("TEST 3: GRADIENT FLOW THROUGH LAYERS")
print("(Flat = healthy flow, Decay = vanishing gradients, Spike = instability)")
print(f"{'='*75}")

# Already have layer_grad_stats, compute flow pattern
if len(layer_grad_stats) >= 2:
    norms = [s["gradient_norm"] for s in layer_grad_stats]
    ratios = [s["gradient_to_param_ratio"] for s in layer_grad_stats]

    # Compute flow metrics
    max_norm = max(norms)
    min_norm = min(n for n in norms if n > 0) if any(n > 0 for n in norms) else 1e-10
    flow_ratio = max_norm / min_norm if min_norm > 0 else float('inf')

    # Detect pattern
    first_half_avg = np.mean(norms[:len(norms)//2])
    second_half_avg = np.mean(norms[len(norms)//2:])

    if flow_ratio > 100:
        pattern = "SEVERE GRADIENT IMBALANCE"
    elif flow_ratio > 10:
        pattern = "MODERATE GRADIENT IMBALANCE"
    else:
        pattern = "HEALTHY GRADIENT FLOW"

    direction = "VANISHING (early layers starved)" if first_half_avg < second_half_avg * 0.5 else \
                "EXPLODING (early layers oversaturated)" if first_half_avg > second_half_avg * 2 else \
                "BALANCED"

    print(f"\n  Max/min gradient ratio: {flow_ratio:.2f}x")
    print(f"  Early vs late layer avg: {first_half_avg:.6f} vs {second_half_avg:.6f}")
    print(f"  Pattern: {pattern}")
    print(f"  Direction: {direction}")

    gradient_flow = {
        "max_min_ratio": round(flow_ratio, 4),
        "early_avg": round(first_half_avg, 6),
        "late_avg": round(second_half_avg, 6),
        "pattern": pattern,
        "direction": direction,
    }
else:
    gradient_flow = {"pattern": "INSUFFICIENT_DATA"}

# ── RANK LAYERS BY GRADIENT IMPORTANCE ────────────────────────────────────────
ranked = sorted(layer_grad_stats, key=lambda x: x["gradient_to_param_ratio"], reverse=True)
print(f"\n{'='*75}")
print("LAYER IMPORTANCE RANKING (by gradient-to-parameter ratio)")
print(f"{'='*75}")
for rank, r in enumerate(ranked, 1):
    bar = "█" * min(int(r["gradient_to_param_ratio"] * 500), 30)
    print(f"  #{rank:02d}  Layer {r['layer']:02d}  ratio: {r['gradient_to_param_ratio']:.6f}  {bar}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_file = f"bol_gradient_attribution_{LABEL}.json"

# Convert component_grads for JSON
component_grads_json = {
    k: {"total_gradient_norm": round(v["total"], 6), "layer_count": v["count"]}
    for k, v in component_grads.items()
}

with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "num_layers": N,
        "loss": round(loss.item(), 4),
        "layer_gradient_stats": layer_grad_stats,
        "layer_ranking": [{"rank": i+1, **r} for i, r in enumerate(ranked)],
        "component_attribution": component_grads_json,
        "gradient_flow": gradient_flow,
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Cross-reference with bol_layer_sweep.py: gradient importance should")
print("correlate with ablation damage. If they disagree, investigate why.")
