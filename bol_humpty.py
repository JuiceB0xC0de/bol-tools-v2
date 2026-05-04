# bol_humpty.py
# PURPOSE: Prove surgery is reversible. Smash everything, restore everything,
# verify exact match with greedy decoding + weight hash.
# Your legal document. Run this once on each subject to confirm restore pipeline.
#
# HOW TO USE:
#   python bol_humpty.py
#   Expected output: "Exact match: True" + "INTEGRITY VERIFIED"
#
# CONFIGURABLE: see bol_config.py

import torch
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, print_banner

print_banner("bol_humpty.py — Reversibility Verification")

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

prompts = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time there was",
]

def talk(m, p, max_new=50):
    inp = tokenizer(p, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def get_weight_hash(model):
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]

# ── BASELINE ──────────────────────────────────────────────────────────────────
print("\n=== BASELINE (greedy) ===")
baseline_outputs = []
for p in prompts:
    out = talk(base, p)
    baseline_outputs.append(out)
    print(f"  {out}\n")

baseline_hash = get_weight_hash(base)
print(f"Baseline weight hash: {baseline_hash}\n")

# ── FREEZE DRY entire model ───────────────────────────────────────────────────
print("=== FREEZE DRY entire model ===")
stash = {name: param.data.clone() for name, param in base.named_parameters()}
print(f"  Saved {len(stash)} tensors\n")

# ── SMASHING all layers ────────────────────────────────────────��──────────────
print("=== SMASHING all layers ===")
smashed_params = []
for name, param in base.named_parameters():
    if ".layers." in name:
        if param.ndim >= 2:
            param.data.zero_()
            smashed_params.append(name)
print(f"  Zeroed {len(smashed_params)} weight matrices\n")

# ���─ SMASHED OUTPUT ────────────────────────────────────────────────────────────
print("=== SMASHED OUTPUT ===")
for p in prompts:
    print(f"  {talk(base, p)}\n")

# ── TRANSPLANT - restoring all layers ──────────────────────────────────────���──
print("=== TRANSPLANT - restoring all layers ===")
for name in smashed_params:
    param = base.get_parameter(name)
    param.data.copy_(stash[name])
print(f"  Restored {len(smashed_params)} tensors\n")

if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ── RESTORED OUTPUT ───────────────────────────────────��───────────────────────
print("=== RESTORED OUTPUT (should match baseline exactly) ===")
restored_outputs = []
for p in prompts:
    out = talk(base, p)
    restored_outputs.append(out)
    print(f"  {out}\n")

restored_hash = get_weight_hash(base)
print(f"Restored weight hash: {restored_hash}\n")

# ── VERIFICATION ──────────────────────────────────────────────────────────────
print("=== VERIFICATION ===")
text_match = all(a == b for a, b in zip(baseline_outputs, restored_outputs))
weight_match = baseline_hash == restored_hash

print(f"  Output match: {text_match}")
print(f"  Weight hash match: {weight_match}")

if text_match and weight_match:
    print(f"\nINTEGRITY VERIFIED: Surgery is reversible (Hash: {restored_hash})")
else:
    print(f"\nINTEGRITY FAIL:")
    if not text_match:
        for i, (a, b) in enumerate(zip(baseline_outputs, restored_outputs)):
            if a != b:
                print(f"  MISMATCH on prompt {i}:")
                print(f"    before: {a}")
                print(f"    after:  {b}")
    if not weight_match:
        print(f"  Baseline hash: {baseline_hash}")
        print(f"  Restored hash: {restored_hash}")
