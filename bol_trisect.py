# bol_trisect.py
# PURPOSE: Quarter-cut and zone experiments.
# Inner core vs outer shell vs top/bottom quarters.
# SCIENTIFICALLY LOCKED: FP32, LayerNorm protected, integrity hash verified.
#
# HOW TO USE:
#   python bol_trisect.py
#   Outputs terminal table + bol_trisect_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import hashlib
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, PROBES, print_banner, json_default

print_banner("bol_trisect.py — Quarter-Cut Experiments")

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

Q1 = N // 4
Q2 = N // 2
Q3 = N - N // 4

print(f"Total layers : {N}")
print(f"Top crust    : 0  - {Q1-1}")
print(f"Top inner    : {Q1} - {Q2-1}")
print(f"Bottom inner : {Q2} - {Q3-1}")
print(f"Bottom crust : {Q3} - {N-1}\n")

prompts = PROBES

eval_texts = [
    "The quick brown fox jumps over the lazy dog",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure",
    "def hello_world(): print('hello world')",
]

# ── BATCH TOKENIZE WITH PROPER PADDING MASK ───────────────────────────────────
tokenized = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True)
labels = tokenized["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100

if DEVICE == "cuda":
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    labels = labels.to(DEVICE)

def get_loss(model):
    with torch.no_grad():
        outputs = model(**tokenized, labels=labels)
    loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        return float('nan')
    return loss.item()

def talk(model, p, max_new=60):
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

def zero_range(model, start, end, original_weights):
    zeroed = []
    for i in range(start, end):
        for name, param in model.named_parameters():
            if f".layers.{i}." in name:
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

def run(model, label, baseline_loss):
    loss = get_loss(model)

    if math.isnan(loss) or math.isinf(loss):
        loss_display = "NaN"
        loss_val = 999.9999
    else:
        loss_display = f"{loss:.4f}"
        loss_val = loss

    damage = loss_val - baseline_loss if not math.isnan(loss_val) else 999.9999

    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"Loss: {loss_display}  (delta: {damage:+.4f})")
    print(f"{'='*65}")
    outputs = []
    for p in prompts:
        out = talk(model, p)
        print(f"  IN:  {p}")
        print(f"  OUT: {out[:80]}...\n")
        outputs.append({"prompt": p, "output": out[:120]})

    return {"label": label, "loss": round(loss_val, 4), "damage": round(damage, 4), "outputs": outputs}

# ── BASELINE ──────────────────────────────────────────────────────────────────
baseline_loss = get_loss(base)

if math.isnan(baseline_loss) or math.isinf(baseline_loss):
    print("ERROR: Baseline loss is NaN/Inf. Check model loading.")
    exit(1)

run(base, f"BASELINE", baseline_loss)
print(f"Baseline Loss: {baseline_loss:.4f}\n")

# FIX: Removed duplicate WORMHOLE config (was identical to OUTER SHELL)
configs = [
    (f"INNER CORE (layers {Q1}-{Q3-1})", [(0, Q1), (Q3, N)]),
    (f"OUTER SHELL (0-{Q1-1} + {Q3}-{N-1})", [(Q1, Q3)]),
    (f"TOP HALF (0-{Q2-1})", [(Q2, N)]),
    (f"BOTTOM HALF ({Q2}-{N-1})", [(0, Q2)]),
    (f"TOP INNER QUARTER ({Q1}-{Q2-1})", [(0, Q1), (Q2, N)]),
    (f"BOTTOM INNER QUARTER ({Q2}-{Q3-1})", [(0, Q2), (Q3, N)]),
    (f"TOP OUTER + BOTTOM INNER (0-{Q1-1} + {Q2}-{Q3-1})", [(Q1, Q2), (Q3, N)]),
    (f"TOP INNER + BOTTOM OUTER ({Q1}-{Q2-1} + {Q3}-{N-1})", [(0, Q1), (Q2, Q3)]),
]

results = []
for label, zero_ranges in configs:
    all_zeroed = []
    try:
        for start, end in zero_ranges:
            zeroed = zero_range(base, start, end, original_weights)
            all_zeroed.extend(zeroed)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        result = run(base, label, baseline_loss)
        results.append(result)

    finally:
        restore_weights(base, all_zeroed, original_weights)

# ── FINAL INTEGRITY CHECK ─────────────────────────────────────────────────────
is_intact, orig_hash, final_hash = verify_integrity(base, original_weights)
print(f"\n{'='*65}")
if is_intact:
    print(f"INTEGRITY VERIFIED: Model weights unchanged (Hash: {final_hash})")
else:
    print(f"INTEGRITY FAIL: Model weights drifted!")
    print(f"   Original: {orig_hash}...")
    print(f"   Final:    {final_hash}...")
print(f"{'='*65}")

out_file = f"bol_trisect_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({"label": LABEL, "model": MODEL, "baseline_loss": baseline_loss, "configs": results}, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("\n=== TRISECT COMPLETE ===")
