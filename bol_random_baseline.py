# bol_random_baseline.py
# PURPOSE: Sanity check / control condition.
# Feeds RANDOM token IDs through the model and measures "separation"
# using the same methodology as bol_silhouette.py.
# If random tokens show separation → your methodology is flawed.
# If random tokens show NO separation → your real results are trustworthy.
#
# This is your null hypothesis test. A reviewer will ask for this.
#
# HOW TO USE:
#   python bol_random_baseline.py
#   Outputs: terminal table + bol_random_baseline_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import torch.nn.functional as F
import json
import numpy as np
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import (
    MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, RELATED_PAIRS, UNRELATED_PAIRS, print_banner, json_default,
)

print_banner("bol_random_baseline.py — Null Hypothesis Control")

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
# Replace this:
N = model.config.num_hidden_layers

# With this:
N = getattr(model.config, "num_hidden_layers", None) or \
    getattr(model.config, "num_layers", None) or \
    len(model.model.language_model.layers)
VOCAB_SIZE = model.config.vocab_size

RELATED = RELATED_PAIRS
UNRELATED = UNRELATED_PAIRS

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def get_layers_from_word(word):
    """Extract hidden states for a real word at all layers."""
    inp = tokenizer(word, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    return [h[0, -1, :].cpu().float().numpy() for h in out.hidden_states]

def get_layers_from_random_id():
    """Extract hidden states for a RANDOM token ID at all layers."""
    random_id = torch.randint(0, VOCAB_SIZE, (1, 1))
    attention_mask = torch.ones_like(random_id)
    if DEVICE == "cuda":
        random_id = random_id.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=random_id, attention_mask=attention_mask)
    return [h[0, -1, :].cpu().float().numpy() for h in out.hidden_states]

def measure_separation(cache_a, cache_b, pairs, label):
    """Compute cosine similarity for pairs using pre-cached hidden states."""
    results = []
    for layer in range(N + 1):
        sims = []
        for key_a, key_b in pairs:
            a = torch.tensor(cache_a[key_a][layer]).unsqueeze(0)
            b = torch.tensor(cache_b[key_b][layer]).unsqueeze(0)
            sims.append(F.cosine_similarity(a, b).item())
        results.append({"layer": layer, "sims": sims, "mean": np.mean(sims)})
    return results

# ── CONDITION 1: REAL WORDS (same as silhouette) ─────────────────────────────
print("=== CONDITION 1: Real words (ground truth) ===\n")
all_words = list({w for pair in RELATED + UNRELATED for w in pair})
real_cache = {w: get_layers_from_word(w) for w in all_words}

real_rel = measure_separation(real_cache, real_cache, RELATED, "real_related")
real_unrel = measure_separation(real_cache, real_cache, UNRELATED, "real_unrelated")

# ── CONDITION 2: RANDOM TOKEN IDS ────────────────────────────────────────────
# Generate random "words" — same number of pairs, but meaningless tokens
print("=== CONDITION 2: Random token IDs (null hypothesis) ===\n")

NUM_RANDOM = len(all_words) + 10  # extra to ensure enough unique entries
random_cache = {f"rand_{i}": get_layers_from_random_id() for i in range(NUM_RANDOM)}
random_keys = list(random_cache.keys())

# Create "related" and "unrelated" pairs from random tokens
# (there should be NO difference since they're all random)
n_pairs = len(RELATED)
random_pairs_a = [(random_keys[i], random_keys[i + n_pairs]) for i in range(n_pairs)]
random_pairs_b = [(random_keys[i], random_keys[i + 2 * n_pairs]) for i in range(n_pairs)]

rand_group_a = measure_separation(random_cache, random_cache, random_pairs_a, "random_a")
rand_group_b = measure_separation(random_cache, random_cache, random_pairs_b, "random_b")

# ── CONDITION 3: REAL WORDS WITH SHUFFLED PAIR ASSIGNMENTS ───────────────────
# Same real words, but we randomly reassign which are "related" vs "unrelated"
# If separation persists → it's a property of the words, not the pairing
print("=== CONDITION 3: Real words, shuffled pairings ===\n")

np.random.seed(42)
flat_words = list(all_words)
np.random.shuffle(flat_words)
n = len(RELATED)
shuffled_rel = [(flat_words[i], flat_words[i + n]) for i in range(n)]
shuffled_unrel = [(flat_words[i], flat_words[i + 2 * n]) for i in range(min(n, len(flat_words) - 2 * n))]

shuf_rel = measure_separation(real_cache, real_cache, shuffled_rel, "shuffled_related")
shuf_unrel = measure_separation(real_cache, real_cache, shuffled_unrel, "shuffled_unrelated")

# ── COMPARISON TABLE ──────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("SEPARATION COMPARISON: Real vs Random vs Shuffled")
print(f"{'='*90}")
print(f"{'Layer':>6}  {'Real sep':>10}  {'p-val':>8}  {'Rand sep':>10}  {'p-val':>8}  {'Shuf sep':>10}  {'p-val':>8}")
print("-" * 90)

layer_comparison = []
for layer in range(N + 1):
    # Real: related vs unrelated
    real_r = real_rel[layer]["sims"]
    real_u = real_unrel[layer]["sims"]
    real_sep = np.mean(real_r) - np.mean(real_u)
    try:
        _, real_p = scipy_stats.mannwhitneyu(real_r, real_u, alternative='greater')
    except ValueError:
        real_p = 1.0

    # Random: group A vs group B (should be ~0)
    rand_a = rand_group_a[layer]["sims"]
    rand_b = rand_group_b[layer]["sims"]
    rand_sep = np.mean(rand_a) - np.mean(rand_b)
    try:
        _, rand_p = scipy_stats.mannwhitneyu(rand_a, rand_b, alternative='greater')
    except ValueError:
        rand_p = 1.0

    # Shuffled: shuffled-"related" vs shuffled-"unrelated" (should be ~0)
    shuf_r = shuf_rel[layer]["sims"]
    shuf_u = shuf_unrel[layer]["sims"] if layer < len(shuf_unrel) else shuf_r
    shuf_sep = np.mean(shuf_r) - np.mean(shuf_u)
    try:
        _, shuf_p = scipy_stats.mannwhitneyu(shuf_r, shuf_u, alternative='greater')
    except ValueError:
        shuf_p = 1.0

    real_sig = "*" if real_p < 0.05 else " "
    rand_sig = "*" if rand_p < 0.05 else " "
    shuf_sig = "*" if shuf_p < 0.05 else " "

    print(f"  L{layer:02d}   {real_sep:>+10.4f}  {real_p:>7.4f}{real_sig} {rand_sep:>+10.4f}  {rand_p:>7.4f}{rand_sig} {shuf_sep:>+10.4f}  {shuf_p:>7.4f}{shuf_sig}")

    layer_comparison.append({
        "layer": layer,
        "real_separation": round(real_sep, 4),
        "real_p_value": round(real_p, 6),
        "real_significant": real_p < 0.05,
        "random_separation": round(rand_sep, 4),
        "random_p_value": round(rand_p, 6),
        "random_significant": rand_p < 0.05,
        "shuffled_separation": round(shuf_sep, 4),
        "shuffled_p_value": round(shuf_p, 6),
        "shuffled_significant": shuf_p < 0.05,
    })

# ── INTERPRETATION ────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("INTERPRETATION")
print(f"{'='*90}")

real_sig_layers = sum(1 for c in layer_comparison if c["real_significant"])
rand_sig_layers = sum(1 for c in layer_comparison if c["random_significant"])
shuf_sig_layers = sum(1 for c in layer_comparison if c["shuffled_significant"])

print(f"  Significant layers (p<0.05):")
print(f"    Real word pairs:     {real_sig_layers}/{N+1}")
print(f"    Random token IDs:    {rand_sig_layers}/{N+1}")
print(f"    Shuffled pairings:   {shuf_sig_layers}/{N+1}")

if real_sig_layers > 0 and rand_sig_layers == 0 and shuf_sig_layers <= 2:
    print(f"\n  PASS: Real pairs show significant separation,")
    print(f"        random/shuffled controls do not.")
    print(f"        Your methodology is sound.")
    verdict = "PASS"
elif rand_sig_layers > 2:
    print(f"\n  FAIL: Random tokens show significant separation.")
    print(f"        Your methodology may have a systematic bias.")
    print(f"        Investigate: are all cosine similarities converging to similar values?")
    verdict = "FAIL"
else:
    print(f"\n  INCONCLUSIVE: Results don't clearly distinguish real from random.")
    print(f"        Consider: more word pairs, different similarity metric, or different layers.")
    verdict = "INCONCLUSIVE"

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_file = f"bol_random_baseline_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "num_layers": N,
        "verdict": verdict,
        "sig_layers_real": real_sig_layers,
        "sig_layers_random": rand_sig_layers,
        "sig_layers_shuffled": shuf_sig_layers,
        "layer_comparison": layer_comparison,
        "related_pairs": RELATED,
        "unrelated_pairs": UNRELATED,
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Include this in your paper as Table X: Null Hypothesis Control.")
