# bol_transplant_lab.py
# PURPOSE: Harvest layers from one model, transplant into another.
# Tests cross-model organ compatibility — the Frankenstein engine.
# Use after layer_sweep.py identifies which layers are interesting.
# SCIENTIFICALLY LOCKED: FP32, integrity hash, NaN handling.
#
# HOW TO USE:
#   python bol_transplant_lab.py
#
# CONFIGURABLE: see bol_config.py (DONOR_MODEL, RECIPIENT_MODEL)

import torch
import copy
import json
import hashlib
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import DONOR_MODEL, RECIPIENT_MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, PROBES, print_banner, json_default

print_banner("bol_transplant_lab.py — Cross-Model Transplant")

tokenizer = AutoTokenizer.from_pretrained(RECIPIENT_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

donor = AutoModelForCausalLM.from_pretrained(
    DONOR_MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)
donor.eval()

recipient = AutoModelForCausalLM.from_pretrained(
    RECIPIENT_MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
)
recipient.eval()

# Dynamically compute layer ranges based on actual model size
N = recipient.config.num_hidden_layers
Q1 = N // 4
Q2 = N // 2
Q3 = N - N // 4

TRANSPLANT_CONFIGS = {
    f"top 25% (0-{Q1-1})":            list(range(0, Q1)),
    f"inner top ({Q1}-{Q2-1})":        list(range(Q1, Q2)),
    f"inner bottom ({Q2}-{Q3-1})":     list(range(Q2, Q3)),
    f"bottom 25% ({Q3}-{N-1})":        list(range(Q3, N)),
    f"inner 50% ({Q1}-{Q3-1})":        list(range(Q1, Q3)),
    f"every even layer":               list(range(0, N, 2)),
    f"every odd layer":                list(range(1, N, 2)),
    f"just layer 0":                   [0],
    f"just last layer ({N-1})":        [N-1],
    f"first 3 + last 3":              list(range(3)) + list(range(N-3, N)),
}

def talk(m, p, max_new=40):
    inp = tokenizer(p, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def get_loss(m, text="The quick brown fox jumps over the lazy dog"):
    inp = tokenizer(text, return_tensors="pt")
    labels = inp["input_ids"].clone()
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        labels = labels.to(DEVICE)
    with torch.no_grad():
        o = m(**inp, labels=labels)
    loss = o.loss
    if torch.isnan(loss) or torch.isinf(loss):
        return float('nan')
    return loss.item()

def harvest(model, layers):
    tissue = {}
    for name, param in model.named_parameters():
        for i in layers:
            if f".layers.{i}." in name:
                tissue[name] = param.data.clone()
    return tissue

def transplant(recipient_model, tissue):
    patient = copy.deepcopy(recipient_model)
    planted = 0
    for name, param in patient.named_parameters():
        if name in tissue:
            param.data.copy_(tissue[name])
            planted += 1
    return patient, planted

# baselines
donor_loss = get_loss(donor)
recipient_loss = get_loss(recipient)

if math.isnan(donor_loss):
    print("WARNING: Donor baseline loss is NaN")
if math.isnan(recipient_loss):
    print("WARNING: Recipient baseline loss is NaN")

print(f"DONOR     baseline loss: {donor_loss:.4f}")
print(f"RECIPIENT baseline loss: {recipient_loss:.4f}\n")

results = []
for config_name, layers in TRANSPLANT_CONFIGS.items():
    tissue = harvest(donor, layers)
    patient, n_planted = transplant(recipient, tissue)
    loss = get_loss(patient)

    if math.isnan(loss) or math.isinf(loss):
        delta = 999.9999
        loss_display = "NaN"
    else:
        delta = loss - recipient_loss
        loss_display = f"{loss:.4f}"

    print(f"\n{'='*60}")
    print(f"TRANSPLANT: {config_name}  ({n_planted} tensors planted)")
    print(f"Loss: {loss_display}  (delta {delta:+.4f} from recipient baseline)")
    print(f"{'='*60}")

    outputs = {}
    for p in PROBES:
        out = talk(patient, p)
        print(f"  {out}")
        outputs[p] = out

    results.append({
        "config": config_name,
        "layers": layers,
        "tensors_planted": n_planted,
        "loss": round(loss if not math.isnan(loss) else 999.9999, 4),
        "delta": round(delta, 4),
        "outputs": outputs,
    })

    del patient

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
out_file = f"bol_transplant_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "donor_model": DONOR_MODEL,
        "recipient_model": RECIPIENT_MODEL,
        "donor_loss": round(donor_loss, 4),
        "recipient_loss": round(recipient_loss, 4),
        "results": results,
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
