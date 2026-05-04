# bol_compression_map.py
# PURPOSE: Simulate quantization noise per layer to find compression tolerance.
# Some layers survive int4, some need fp16. Maps the blueprint for 16MB packing.
# Does NOT actually quantize — adds calibrated noise to simulate the effect.
# SCIENTIFICALLY LOCKED: FP32, in-place restore, integrity hash verified.
#
# HOW TO USE:
#   python bol_compression_map.py
#   Outputs terminal table + bol_compression_map_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import hashlib
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import (
    MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, NOISE_LEVELS, COMPRESSION_THRESHOLD, print_banner, json_default,
)

print_banner("bol_compression_map.py — Quantization Tolerance")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)
base.eval()
N = base.config.num_hidden_layers

eval_texts = [
    "The quick brown fox jumps over the lazy dog",
    "Water boils at 100 degrees Celsius",
    "def hello(): return True",
]

# ── BATCH TOKENIZE WITH PROPER PADDING MASK ─��─────────────────────────────��───
tokenized = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True)
labels = tokenized["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

if DEVICE == "cuda":
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    labels = labels.to(DEVICE)

noise_levels = NOISE_LEVELS
THRESHOLD = COMPRESSION_THRESHOLD

def get_batched_loss(model):
    with torch.no_grad():
        outputs = model(**tokenized, labels=labels)
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        return float('nan')
    return loss.item()

# ── INTEGRITY LOCK: Save Original Weights ��───────────────────��─────────────────
print("Saving original weight snapshot (FP32)...")
original_weights = {}
for name, param in base.named_parameters():
    original_weights[name] = param.data.clone()

def verify_integrity(model, original_weights):
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        if name in original_weights:
            hasher.update(param.data.cpu().numpy().tobytes())
    current_hash = hasher.hexdigest()

    hasher_orig = hashlib.sha256()
    for name, tensor in original_weights.items():
        hasher_orig.update(tensor.cpu().numpy().tobytes())
    original_hash = hasher_orig.hexdigest()

    return current_hash == original_hash, original_hash[:16], current_hash[:16]

# ── BASELINE ──────────────────────────────────────────────────────────────────
baseline_loss = get_batched_loss(base)

if math.isnan(baseline_loss) or math.isinf(baseline_loss):
    print("ERROR: Baseline loss is NaN/Inf. Check model loading.")
    exit(1)

print(f"Baseline loss: {baseline_loss:.4f}\n")
print(f"{'Layer':>6}", end="")
for ln in noise_levels:
    print(f"  {ln:>8}", end="")
print(f"  {'tolerance':>10}")
print("-" * 65)

layer_results = []
for i in range(N):
    print(f"  L{i:02d}  ", end="", flush=True)
    row = {"layer": i, "noise_deltas": {}, "tolerance": "fp16"}

    best_tolerance = "fp16"

    layer_param_names = [
        name for name, param in base.named_parameters()
        if f".layers.{i}." in name and param.ndim >= 2
    ]

    for level_name, noise_scale in noise_levels.items():
        try:
            for name in layer_param_names:
                param = base.get_parameter(name)
                orig_tensor = original_weights[name]
                scale = orig_tensor.abs().mean()
                noise = torch.randn_like(orig_tensor) * scale * noise_scale
                if DEVICE == "cuda":
                    noise = noise.to(DEVICE)
                param.data.add_(noise)

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            delta = get_batched_loss(base) - baseline_loss

            if math.isnan(delta) or math.isinf(delta):
                delta = 999.9999

            row["noise_deltas"][level_name] = round(delta, 4)
            print(f"  {delta:>+8.4f}", end="", flush=True)

            if delta < THRESHOLD:
                best_tolerance = level_name

        finally:
            for name in layer_param_names:
                param = base.get_parameter(name)
                param.data.copy_(original_weights[name])

            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    row["tolerance"] = best_tolerance
    tag = ("[compressible]" if "int4" in best_tolerance or "int2" in best_tolerance
           else "[moderate]" if "int6" in best_tolerance
           else "[keep fp16]")
    print(f"  {tag} {best_tolerance:>8}")
    layer_results.append(row)

# ── FINAL INTEGRITY CHECK ────────────────��───────────────────────────────���────
is_intact, orig_hash, final_hash = verify_integrity(base, original_weights)
print(f"\n{'='*65}")
if is_intact:
    print(f"INTEGRITY VERIFIED: Model weights unchanged (Hash: {final_hash})")
else:
    print(f"INTEGRITY FAIL: Model weights drifted!")
    print(f"   Original: {orig_hash}...")
    print(f"   Final:    {final_hash}...")
print(f"{'='*65}")

print(f"\n{'='*65}")
print("COMPRESSION MAP  (compressible -> keep at fp16)")
print(f"{'='*65}")
for r in layer_results:
    tag = ("compressible" if "int4" in r["tolerance"] or "int2" in r["tolerance"]
           else "moderate" if "int6" in r["tolerance"]
           else "keep fp16")
    print(f"  Layer {r['layer']:02d}: [{tag}] tolerates {r['tolerance']}")

out_file = f"bol_compression_map_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({"label": LABEL, "model": MODEL, "baseline_loss": baseline_loss, "layer_results": layer_results}, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Cross-reference with layer_importance_<LABEL>.json to find: removable AND compressible layers.")
