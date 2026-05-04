# bol_component_ablation.py
# PURPOSE: Zero ONE component type across ALL layers simultaneously.
# Answers: if I remove every q_proj in the model, what dies?
# Separates attention surgery from MLP surgery.
# SCIENTIFICALLY LOCKED: FP32, LayerNorm protected, integrity hash verified.
#
# HOW TO USE:
#   python bol_component_ablation.py
#   Outputs: terminal table + bol_component_report_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import hashlib
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, EVAL_TEXTS, PROBES, print_banner, json_default

print_banner("bol_component_ablation.py — Component Importance")

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

COMPONENTS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Use multiple eval texts for more robust loss measurement
eval_texts = EVAL_TEXTS

# ── BATCH TOKENIZE WITH PROPER PADDING MASK ───────────────────────────────────
tokenized = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True)
labels = tokenized["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

if DEVICE == "cuda":
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    labels = labels.to(DEVICE)

def get_loss(model):
    """Evaluate loss with proper padding mask."""
    with torch.no_grad():
        outputs = model(**tokenized, labels=labels)
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        return float('nan')
    return loss.item()

def talk(model, p, max_new=40):
    """Generate text from a single prompt."""
    inp = tokenizer(p, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
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

def zero_component(model, component_name, original_weights):
    zeroed = []
    for name, param in model.named_parameters():
        if component_name in name:
            if param.ndim >= 2:
                param.data.zero_()
                zeroed.append(name)
    return zeroed

def restore_weights(model, zeroed_params, original_weights):
    for name in zeroed_params:
        param = model.get_parameter(name)
        param.data.copy_(original_weights[name])
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ── BASELINE ──────────────────────────────────────────────────────────────────
baseline_loss = get_loss(base)

if math.isnan(baseline_loss) or math.isinf(baseline_loss):
    print("ERROR: Baseline loss is NaN/Inf. Check model loading.")
    exit(1)

baseline_outputs = {p: talk(base, p) for p in PROBES}
print(f"BASELINE loss: {baseline_loss:.4f}\n")

results = []
for comp in COMPONENTS:
    zeroed_params = []
    try:
        zeroed_params = zero_component(base, comp, original_weights)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        loss = get_loss(base)

        if math.isnan(loss) or math.isinf(loss):
            loss_val = 999.9999
            delta = 999.9999
            loss_display = "NaN"
        else:
            loss_val = loss
            delta = loss - baseline_loss
            loss_display = f"{loss:.4f}"

        zeroed_count = len(zeroed_params)
        print(f"\n{'='*60}")
        print(f"ZEROED: {comp}  ({zeroed_count} tensors)  loss: {loss_display}  (delta {delta:+.4f})")
        print(f"{'='*60}")

        row = {"component": comp, "zeroed_tensors": zeroed_count, "loss": round(loss_val, 4), "damage": round(delta, 4), "outputs": {}}
        for p in PROBES:
            out = talk(base, p)
            changed = out != baseline_outputs[p]
            tag = "CHANGED" if changed else "stable"
            print(f"  [{tag:7s}] {p[:35]:<35} -> {out[:55]}")
            row["outputs"][p] = {"text": out, "changed": changed}
        results.append(row)

    finally:
        restore_weights(base, zeroed_params, original_weights)

# ── FINAL INTEGRITY CHECK ─────────────────────────────────────────────────────
is_intact, orig_hash, final_hash = verify_integrity(base, original_weights)
print(f"\n{'='*60}")
if is_intact:
    print(f"INTEGRITY VERIFIED: Model weights unchanged (Hash: {final_hash})")
else:
    print(f"INTEGRITY FAIL: Model weights drifted!")
    print(f"   Original: {orig_hash}...")
    print(f"   Final:    {final_hash}...")
print(f"{'='*60}")

results.sort(key=lambda x: x["damage"], reverse=True)
print(f"\n{'='*60}")
print("COMPONENT IMPORTANCE RANKING")
print(f"{'='*60}")
for r in results:
    tag = ("CRITICAL" if r["damage"] > 1.0
           else "MODERATE" if r["damage"] > 0.3
           else "REMOVABLE")
    print(f"  {r['component']:<30}  delta_loss: {r['damage']:+.4f}  {tag}")

out_file = f"bol_component_report_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({"label": LABEL, "model": MODEL, "baseline_loss": baseline_loss, "results": results}, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
