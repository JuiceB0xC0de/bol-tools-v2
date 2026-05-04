# bol_wormhole.py
# PURPOSE: Extended wormhole experiments with loss scoring.
# Connects layer zones that were never designed to talk to each other.
# Tests alternating layers, skip patterns, first+last only, etc.
# SCIENTIFICALLY LOCKED: FP32, LayerNorm protected, integrity hash verified.
#
# HOW TO USE:
#   python bol_wormhole.py
#   Outputs terminal table + bol_wormhole_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import json
import hashlib
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, PROBES, print_banner, json_default

print_banner("bol_wormhole.py — Layer Bypass Experiments")

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

prompts = PROBES

eval_text = "The quick brown fox jumps over the lazy dog"

tokenized = tokenizer(eval_text, return_tensors="pt", padding=True, truncation=True)
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

def talk(model, p, max_new=50):
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

def keep_only(model, ranges, original_weights):
    keep = set()
    for s, e in ranges:
        keep.update(range(s, e))

    zeroed = []
    for i in range(N):
        if i not in keep:
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

# ── BASELINE ──────────────────────────────────────────────────────────────────
baseline_loss = get_loss(base)

if math.isnan(baseline_loss) or math.isinf(baseline_loss):
    print("ERROR: Baseline loss is NaN/Inf. Check model loading.")
    exit(1)

print(f"BASELINE loss: {baseline_loss:.4f}\n")

combos = {
    f"Top 25% + Bottom 25% (0-{Q1-1} + {Q3}-{N-1})":               [(0,Q1),(Q3,N)],
    f"Top 25% + Inner-bottom 25% (0-{Q1-1} + {Q2}-{Q3-1})":        [(0,Q1),(Q2,Q3)],
    f"Inner-top 25% + Bottom 25% ({Q1}-{Q2-1} + {Q3}-{N-1})":      [(Q1,Q2),(Q3,N)],
    f"Inner 50% only ({Q1}-{Q3-1})":                                 [(Q1,Q3)],
    f"Every even layer":                                              [(i,i+1) for i in range(0,N,2)],
    f"Every odd layer":                                               [(i,i+1) for i in range(1,N,2)],
    f"First + Last layer only":                                       [(0,1),(N-1,N)],
    f"First 3 + Last 3":                                             [(0,3),(N-3,N)],
    f"First 5 + Last 5":                                             [(0,5),(N-5,N)],
    f"Every 3rd layer":                                               [(i,i+1) for i in range(0,N,3)],
}

results = []
for name, ranges in combos.items():
    zeroed_params = []
    try:
        zeroed_params = keep_only(base, ranges, original_weights)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        loss = get_loss(base)
        active = sum(e-s for s,e in ranges)

        if math.isnan(loss) or math.isinf(loss):
            loss_val = 999.9999
            delta = 999.9999
            loss_display = "NaN"
        else:
            loss_val = loss
            delta = loss - baseline_loss
            loss_display = f"{loss:.4f}"

        print(f"\n{'='*65}")
        print(f"{name}")
        print(f"Active layers: {active}/{N}  Loss: {loss_display}  (delta {delta:+.4f})")
        print(f"{'='*65}")

        outputs = []
        for p in prompts:
            out = talk(base, p)
            print(f"  IN:  {p}")
            print(f"  OUT: {out[:80]}...\n")
            outputs.append({"prompt": p, "output": out[:120]})

        results.append({
            "name": name,
            "active_layers": active,
            "total_layers": N,
            "loss": round(loss_val, 4),
            "delta": round(delta, 4),
            "ranges": ranges,
            "outputs": outputs,
        })

    finally:
        restore_weights(base, zeroed_params, original_weights)

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

out_file = f"bol_wormhole_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({"label": LABEL, "model": MODEL, "baseline_loss": baseline_loss, "combos": results}, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("\n=== WORMHOLE COMPLETE ===")
