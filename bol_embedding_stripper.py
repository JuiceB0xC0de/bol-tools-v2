# bol_embedding_stripper.py
# PURPOSE: The core scientific experiment of BoL.
# Tests whether semantic separation survives when the learned embedding
# table is destroyed. Three strip modes:
#   1. SHUFFLE — permute embedding rows (destroys semantic mapping, keeps statistics)
#   2. RANDOM  — replace with random vectors (destroys everything, keeps dimensionality)
#   3. ZERO    — zero the table entirely (nuclear option — control condition)
#
# H1 — separation survives → comprehension lives in the transformer weights
# H2 — separation collapses → the embedding WAS the understanding
#
# SCIENTIFICALLY LOCKED: FP32, re-extracts after strip, integrity hash, JSON output.
#
# HOW TO USE:
#   python bol_embedding_stripper.py
#   Outputs: terminal table + bol_embedding_stripper_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import gc
import torch
import torch.nn.functional as F
import json
import hashlib
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats as scipy_stats

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, RELATED_PAIRS, UNRELATED_PAIRS, print_banner, json_default

print_banner("bol_embedding_stripper.py — Octopus Phase Test")

# Use the first 6 pairs (matching original scope)
related = RELATED_PAIRS[:6]
unrelated = UNRELATED_PAIRS[:6]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    output_hidden_states=True,
    low_cpu_mem_usage=True,
)
base.eval()
N = base.config.num_hidden_layers

def get_all_layers(model, word):
    """Extract hidden states for a word at all layers."""
    inp = tokenizer(word, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    return [h[0, -1, :].cpu().float() for h in out.hidden_states]

def cos(a, b):
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def measure_separation(model, label):
    """Measure semantic separation across all layers. Extracts fresh each time."""
    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"{'='*65}")
    print(f"{'Layer':>6}  {'rel avg':>9}  {'unrel avg':>10}  {'sep':>8}  {'p-val':>8}  bar")
    print("-" * 75)

    all_words = list({w for pair in related + unrelated for w in pair})
    word_cache = {w: get_all_layers(model, w) for w in all_words}

    layer_data = []
    for layer in range(N + 1):
        rel_sims = [cos(word_cache[a][layer], word_cache[b][layer]) for a, b in related]
        unrel_sims = [cos(word_cache[a][layer], word_cache[b][layer]) for a, b in unrelated]
        ra = np.mean(rel_sims)
        ua = np.mean(unrel_sims)
        sep = ra - ua

        # Statistical significance: Mann-Whitney U (non-parametric, no normality assumption)
        if len(rel_sims) >= 3 and len(unrel_sims) >= 3:
            try:
                stat, pval = scipy_stats.mannwhitneyu(rel_sims, unrel_sims, alternative='greater')
            except ValueError:
                pval = 1.0
        else:
            pval = 1.0

        bar = "█" * int(max(sep, 0) * 40)
        sig = "*" if pval < 0.05 else ""
        print(f"  L{layer:02d}   {ra:>9.4f}  {ua:>10.4f}  {sep:>+8.4f}  {pval:>7.4f}{sig} {bar}")
        layer_data.append({
            "layer": layer,
            "rel_avg": round(ra, 4),
            "unrel_avg": round(ua, 4),
            "separation": round(sep, 4),
            "p_value": round(pval, 6),
            "significant": pval < 0.05,
            "rel_sims": [round(s, 4) for s in rel_sims],
            "unrel_sims": [round(s, 4) for s in unrel_sims],
        })

    return layer_data

def verify_integrity(model, original_weights):
    """Hashes current weights vs original to prove no drift occurred."""
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]

# ── INTEGRITY LOCK: Save Original Weights ──────────────────────────────────────
print("Saving original weight snapshot (FP32)...")
original_weights = {}
for name, param in base.named_parameters():
    original_weights[name] = param.data.clone()

baseline_hash = verify_integrity(base, original_weights)
print(f"Baseline weight hash: {baseline_hash}\n")

# ── BASELINE MEASUREMENT (FRESH EXTRACTION) ───────────────────────────────────
print("Extracting BASELINE hidden states...")
baseline_data = measure_separation(base, "BASELINE — full model with embeddings")

# ── STRIP MODES ──────────────────────────────────────────────────────────────
STRIP_MODES = {
    "SHUFFLE": "Randomly permute embedding rows (destroy semantic map, keep statistics)",
    "RANDOM":  "Replace with random vectors (destroy everything, keep dimensionality)",
    "ZERO":    "Zero the table entirely (nuclear option — control condition)",
}

all_strip_results = {}

for mode_name, mode_desc in STRIP_MODES.items():
    print(f"\n{'='*65}")
    print(f"STRIP MODE: {mode_name}")
    print(f"  {mode_desc}")
    print(f"{'='*65}")

    # Apply strip
    stripped_params = []
    freed = 0
    for name, param in base.named_parameters():
        if "embed_tokens" in name:
            freed += param.nelement() * param.element_size()
            if mode_name == "ZERO":
                param.data.zero_()
            elif mode_name == "SHUFFLE":
                # Permute rows randomly (each token gets a random OTHER token's embedding)
                perm = torch.randperm(param.data.shape[0])
                param.data.copy_(original_weights[name][perm])
            elif mode_name == "RANDOM":
                # Replace with random vectors matching original statistics
                orig = original_weights[name]
                param.data.copy_(torch.randn_like(orig) * orig.std() + orig.mean())
            stripped_params.append(name)
            print(f"  MODIFIED: {name}  ({param.nelement() * param.element_size() / 1e6:.1f} MB)")

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Measure
    print(f"\nExtracting POST-{mode_name} hidden states...")
    post_data = measure_separation(base, f"POST-{mode_name} — embed_tokens modified")

    # Compare
    print(f"\n{'Layer':>6}  {'Baseline':>10}  {mode_name:>10}  {'Delta':>8}  {'Status':<12}")
    print("-" * 65)

    comparison_data = []
    for b, p in zip(baseline_data, post_data):
        delta = p["separation"] - b["separation"]
        status = ("SURVIVES" if p["separation"] > 0.05 and p["significant"]
                  else "WEAK"     if p["separation"] > 0.02
                  else "COLLAPSED")
        print(f"  L{b['layer']:02d}   {b['separation']:>+10.4f}  {p['separation']:>+10.4f}  {delta:>+8.4f}  {status:<12}")
        comparison_data.append({
            "layer": b["layer"],
            "baseline_sep": b["separation"],
            "stripped_sep": p["separation"],
            "delta": round(delta, 4),
            "status": status,
            "p_value": p["p_value"],
            "significant": p["significant"],
        })

    all_strip_results[mode_name] = {
        "description": mode_desc,
        "freed_mb": round(freed / 1e6, 2),
        "post_data": post_data,
        "comparison": comparison_data,
    }

    # Restore for next mode
    for name in stripped_params:
        param = base.get_parameter(name)
        param.data.copy_(original_weights[name])

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ── FINAL INTEGRITY CHECK ────────────────────────────────────────────────────
final_hash = verify_integrity(base, original_weights)
print(f"\n{'='*65}")
if final_hash == baseline_hash:
    print(f"INTEGRITY VERIFIED: Weights fully restored (Hash: {final_hash})")
else:
    print(f"INTEGRITY FAIL: Weights not restored!")
print(f"{'='*65}")

# ── CROSS-MODE INTERPRETATION ────────────────────────────────────────────────
print(f"\n{'='*65}")
print("CROSS-MODE INTERPRETATION")
print(f"{'='*65}")

for mode_name, result in all_strip_results.items():
    comp = result["comparison"]
    surviving = sum(1 for c in comp if c["status"] == "SURVIVES")
    collapsed = sum(1 for c in comp if c["status"] == "COLLAPSED")
    middle_survive = any(
        c["status"] == "SURVIVES"
        for c in comp if 5 <= c["layer"] <= N - 5
    )
    print(f"\n  {mode_name}:")
    print(f"    Surviving layers (sig sep > 0.05): {surviving}/{N + 1}")
    print(f"    Collapsed layers:                  {collapsed}/{N + 1}")
    print(f"    Middle layers survive:             {'YES' if middle_survive else 'NO'}")

    result["surviving_layers"] = surviving
    result["collapsed_layers"] = collapsed
    result["middle_survive"] = middle_survive

# Key interpretation: SHUFFLE is the real test, ZERO is the control
shuffle = all_strip_results.get("SHUFFLE", {})
zero = all_strip_results.get("ZERO", {})

print(f"\n  {'='*55}")
print("  VERDICT:")
if shuffle.get("middle_survive"):
    print("  H1 SUPPORTED: Separation survives SHUFFLE in transformer weights.")
    print("    -> Comprehension exists independent of learned semantic mapping.")
    print("    -> The 'Octopus Phase' is empirically observable.")
else:
    print("  H2 SUPPORTED: Separation collapses without learned embeddings.")
    print("    -> The embedding layer carries most semantic weight.")
    print("    -> Transformer layers depend on vocabulary input for meaning.")

if zero.get("middle_survive"):
    print("\n  WARNING: ZERO mode also shows survival. This is unexpected")
    print("  (zero inputs should propagate zeros in bias-free architectures).")
    print("  Investigate: possible bias terms, non-zero LayerNorm, or numerical artifacts.")
else:
    print("\n  ZERO mode collapsed as expected (control condition passed).")

print("\n  Both outcomes are valid findings for the BoL submission.")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
out_file = f"bol_embedding_stripper_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "num_layers": N,
        "baseline_hash": baseline_hash,
        "final_hash": final_hash,
        "h1_supported": shuffle.get("middle_survive", False),
        "baseline_data": baseline_data,
        "strip_results": all_strip_results,
        "related_pairs": related,
        "unrelated_pairs": unrelated,
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("This is your core evidence file. Include the hash in your paper.")
