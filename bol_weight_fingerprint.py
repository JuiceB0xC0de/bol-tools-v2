# bol_weight_fingerprint.py
# PURPOSE: Compare weight statistics across Subject A (base), B (Muon/Taoist), C (instruct).
# Measures: mean, std, sparsity, frobenius norm, spectral norm, condition number per layer.
# This is your pre-surgery inspection report — run before ANY cutting.
#
# HOW TO USE:
#   python bol_weight_fingerprint.py
#   Outputs comparison table to terminal + saves weight_fingerprints.json
#
# CONFIGURABLE: see bol_config.py for MODEL/LABEL, edit models dict below for multi-model comparison

import torch
import json
from transformers import AutoModelForCausalLM

from bol_config import MODEL, LABEL, print_banner, json_default

print_banner("bol_weight_fingerprint.py — Pre-Surgery Inspection")

# ── CONFIGURE MODELS TO FINGERPRINT ──────────────────────────────────────────
# Uses the active model from bol_config by default.
# Uncomment additional models for cross-subject comparison.
models = {
    LABEL: MODEL,
    # "Subject_A_base": "HuggingFaceTB/SmolLM2-135M",
    # "Subject_B_muon": "juiceb0xc0de/bol-subject-b-smollm2-muon",
}
# ───────────────────────────────────────────────────────────────────────────────

all_fingerprints = {}

for label, path in models.items():
    print(f"\n{'='*65}")
    print(f"FINGERPRINTING: {label}")
    print(f"{'='*65}")
    print(f"{'Tensor':<55} {'mean':>10} {'std':>10} {'sparse':>8} {'spec_norm':>10}")
    print("-" * 95)

    model = AutoModelForCausalLM.from_pretrained(path)
    fingerprint = []

    for name, param in model.named_parameters():
        if ".layers." not in name:
            continue
        if param.ndim < 2:
            continue

        data = param.data.float()
        stats = {
            "name":        name,
            "shape":       list(param.shape),
            "mean":        round(data.mean().item(), 6),
            "std":         round(data.std().item(), 6),
            "abs_mean":    round(data.abs().mean().item(), 6),
            "max":         round(data.max().item(), 6),
            "min":         round(data.min().item(), 6),
            "sparsity":    round((data.abs() < 1e-6).float().mean().item(), 4),
            "frobenius":   round(data.norm().item(), 4),
        }

        if param.ndim == 2 and max(param.shape) <= 2048:
            try:
                s = torch.linalg.svdvals(data)
                stats["spectral_norm"]    = round(s[0].item(), 4)
                stats["condition_number"] = round((s[0] / (s[-1] + 1e-9)).item(), 4)
            except Exception:
                stats["spectral_norm"]    = None
                stats["condition_number"] = None
        else:
            stats["spectral_norm"]    = None
            stats["condition_number"] = None

        sn = f"{stats['spectral_norm']:.4f}" if stats["spectral_norm"] else "  n/a  "
        print(f"  {name:<53} {stats['mean']:>10.6f} {stats['std']:>10.6f} "
              f"{stats['sparsity']:>8.4f} {sn:>10}")

        fingerprint.append(stats)

    all_fingerprints[label] = fingerprint
    del model
    print(f"\n  -> {len(fingerprint)} tensors fingerprinted for {label}")

with open("weight_fingerprints.json", "w") as f:
    json.dump(all_fingerprints, f, indent=2, default=json_default)
print("\nSaved: weight_fingerprints.json")
