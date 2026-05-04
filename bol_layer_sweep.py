# bol_layer_sweep.py
# PURPOSE: Remove ONE layer at a time, measure loss damage.
# Produces a ranked importance table: most critical → safest to remove.
# SCIENTIFICALLY LOCKED: FP32, LayerNorm protected, integrity hash verified.
#
# HOW TO USE:
#   python bol_layer_sweep.py
#   Output: terminal table + layer_importance_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import hashlib
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, EVAL_TEXTS, print_banner, json_default

print_banner("bol_layer_sweep.py — Layer Importance Ranking")

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

eval_texts = EVAL_TEXTS

# ── BATCH TOKENIZE WITH PROPER PADDING MASK ───────────────────────────────────
tokenized = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True)
labels = tokenized["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

if DEVICE == "cuda":
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    labels = labels.to(DEVICE)

probe_prompt = "The capital of France is"

def get_batched_loss(model):
    """Evaluate loss on all texts in one forward pass."""
    with torch.no_grad():
        outputs = model(**tokenized, labels=labels)
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        return float('nan')
    return loss.item()

def talk(model, prompt, max_new=40):
    """Generate text from a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ── INTEGRITY LOCK: Save Original Weights ──────────────────────────────────────
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

baseline_out = talk(base, probe_prompt)
print(f"BASELINE loss: {baseline_loss:.4f}")
print(f"BASELINE out:  {baseline_out[:80]}\n")
print(f"{'Layer':>7}  {'Loss':>8}  {'Damage':>8}  {'Tag':<12}  Sample output")
print("-" * 90)

results = []
for i in range(N):
    try:
        zeroed_params = []
        for name, param in base.named_parameters():
            if f".layers.{i}." in name:
                if param.ndim >= 2:
                    param.data.zero_()
                    zeroed_params.append(name)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        loss = get_batched_loss(base)

        if math.isnan(loss) or math.isinf(loss):
            loss = 999.9999
            damage = 999.9999
            sample = "NaN ERROR"
            changed = True
            tag = "CRITICAL"
            chg = "NaN"
        else:
            damage = loss - baseline_loss
            sample = talk(base, probe_prompt)
            changed = sample != baseline_out

            tag = ("CRITICAL" if damage > 1.0
                   else "MODERATE" if damage > 0.3
                   else "REMOVABLE")
            chg = "CHANGED" if changed else "stable"

        print(f"  L{i:02d}    {loss:>8.4f}  {damage:>+8.4f}  {tag:<12}  [{chg}] {sample[:50]}")
        results.append({
            "layer": i,
            "loss": round(loss, 4),
            "damage": round(damage, 4),
            "changed": changed,
            "sample": sample[:120],
        })

    finally:
        for name in zeroed_params:
            param = base.get_parameter(name)
            param.data.copy_(original_weights[name])

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# ── FINAL INTEGRITY CHECK ─────────────────────────────────────────────────────
is_intact, orig_hash, final_hash = verify_integrity(base, original_weights)
print(f"\n{'='*70}")
if is_intact:
    print(f"INTEGRITY VERIFIED: Model weights unchanged (Hash: {final_hash})")
else:
    print(f"INTEGRITY FAIL: Model weights drifted!")
    print(f"   Original: {orig_hash}...")
    print(f"   Final:    {final_hash}...")
print(f"{'='*70}")

ranked = sorted(results, key=lambda x: x["damage"], reverse=True)
print(f"\n{'='*70}")
print("LAYER IMPORTANCE RANKING  (most critical -> safest to remove)")
print(f"{'='*70}")
for rank, r in enumerate(ranked, 1):
    tag = ("CRITICAL" if r["damage"] > 1.0
           else "MODERATE" if r["damage"] > 0.3
           else "REMOVABLE")
    print(f"  #{rank:02d}  Layer {r['layer']:02d}  damage: {r['damage']:+.4f}  {tag}")

out_file = f"layer_importance_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({"baseline_loss": baseline_loss, "ranked": ranked, "by_layer": results}, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
