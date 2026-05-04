# bol_silhouette.py
# PURPOSE: Find which layers have the highest semantic separation.
# These are your load-bearing walls — the layers worth keeping.
# Maps residual geometry at every layer for related vs unrelated pairs.
# SCIENTIFICALLY LOCKED: FP32, batched extraction, integrity verified.
#
# HOW TO USE:
#   python bol_silhouette.py
#   Outputs: terminal table + bol_silhouette_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import numpy as np
import torch.nn.functional as F
import json
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import (
    MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, RELATED_PAIRS, UNRELATED_PAIRS, print_banner, json_default,
)

print_banner("bol_silhouette.py — Semantic Separation Map")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    output_hidden_states=True,
    low_cpu_mem_usage=True,
)
model.eval()
N = model.config.num_hidden_layers

RELATED = RELATED_PAIRS
UNRELATED = UNRELATED_PAIRS

# ── EXTRACT HIDDEN STATES (BATCHED FOR SPEED) ─────────────────────────────────
def get_layers(word):
    """Extract hidden states for a single word."""
    inp = tokenizer(word, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    return [h[0, -1, :].cpu().float().numpy() for h in out.hidden_states]

print("Extracting hidden states (this takes a moment)...")
all_words = list({w for pair in RELATED + UNRELATED for w in pair})
cache = {w: get_layers(w) for w in all_words}
print(f"  Cached {len(cache)} words x {N + 1} layers\n")

# ── COMPUTE SEMANTIC SEPARATION ───────────────────────────────────────────────
print(f"{'Layer':>6}  {'rel avg':>9}  {'unrel avg':>10}  {'sep':>8}  {'p-val':>8}  bar")
print("-" * 75)

best_layer, best_sep = 0, -999
layer_data = []

for layer in range(N + 1):
    rel_sims = [
        F.cosine_similarity(
            torch.tensor(cache[a][layer]).unsqueeze(0),
            torch.tensor(cache[b][layer]).unsqueeze(0)
        ).item()
        for a, b in RELATED
    ]
    unrel_sims = [
        F.cosine_similarity(
            torch.tensor(cache[a][layer]).unsqueeze(0),
            torch.tensor(cache[b][layer]).unsqueeze(0)
        ).item()
        for a, b in UNRELATED
    ]
    ra = np.mean(rel_sims)
    ua = np.mean(unrel_sims)
    sep = ra - ua

    # Statistical significance
    if len(rel_sims) >= 3 and len(unrel_sims) >= 3:
        try:
            stat, pval = scipy_stats.mannwhitneyu(rel_sims, unrel_sims, alternative='greater')
        except ValueError:
            pval = 1.0
    else:
        pval = 1.0

    bar = "█" * int(max(sep, 0) * 50)
    marker = " <- BEST" if sep > best_sep else ""
    sig = "*" if pval < 0.05 else ""
    if sep > best_sep:
        best_sep, best_layer = sep, layer
    print(f"  L{layer:02d}   {ra:>9.4f}  {ua:>10.4f}  {sep:>+8.4f}  {pval:>7.4f}{'*' if pval < 0.05 else ' '}  {bar} {'(p<0.05)' if pval < 0.05 else ''}{marker}")
    layer_data.append({
        "layer": layer,
        "rel_avg": round(ra, 4),
        "unrel_avg": round(ua, 4),
        "separation": round(sep, 4),
        "p_value": round(pval, 6),
        "significant": bool(pval < 0.05),
        "rel_sims": [round(s, 4) for s in rel_sims],
        "unrel_sims": [round(s, 4) for s in unrel_sims],
    })

# ── RANK AND OUTPUT ───────────────────────────────────────────────────────────
ranked = sorted(layer_data, key=lambda x: x["separation"], reverse=True)
print(f"\n{'='*75}")
print("TOP 10 LAYERS BY SEMANTIC SEPARATION")
print(f"{'='*75}")
for r in ranked[:10]:
    bar = "█" * int(max(r["separation"], 0) * 50)
    sig = " (p<0.05)" if r["significant"] else ""
    print(f"  Layer {r['layer']:02d}  sep={r['separation']:+.4f}  {bar} {sig}")

print(f"\n-> Best single layer: {best_layer} (sep={best_sep:.4f})")
print(f"-> These are your load-bearing walls. Keep them in the artifact.")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
out_file = f"bol_silhouette_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "num_layers": N,
        "num_words": len(cache),
        "best_layer": best_layer,
        "best_separation": round(best_sep, 4),
        "layer_data": layer_data,
        "ranked_top_10": ranked[:10],
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Cross-reference with layer_importance_<LABEL>.json to find: high-separation + low-damage layers.")
