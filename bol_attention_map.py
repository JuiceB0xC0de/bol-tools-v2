# bol_attention_map.py
# PURPOSE: Extract and analyze attention patterns before/after surgery.
# If a model "understands," its attention should show structure:
#   - Non-uniform attention distributions (not just attending to everything equally)
#   - Consistent patterns across semantically similar inputs
#   - Head specialization (different heads attend to different things)
#
# Measures:
#   1. Attention entropy per head per layer (low = focused, high = uniform/lost)
#   2. Attention pattern similarity for related vs unrelated word pairs
#   3. Head specialization score (how different are heads from each other?)
#
# HOW TO USE:
#   python bol_attention_map.py
#   Outputs: terminal table + bol_attention_map_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py

import torch
import torch.nn.functional as F
import json
import numpy as np
import math
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, RELATED_PAIRS, UNRELATED_PAIRS, print_banner, json_default

print_banner("bol_attention_map.py — Attention Pattern Analysis")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=DTYPE,
    device_map="auto",
    output_hidden_states=True,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.eval()
N = model.config.num_hidden_layers
NUM_HEADS = model.config.num_attention_heads

RELATED = RELATED_PAIRS
UNRELATED = UNRELATED_PAIRS

# ── EXTRACTION ────────────────────────────────────────────────────────────────

def get_attention(word):
    """Extract attention weights for a word at all layers and heads."""
    inp = tokenizer(word, return_tensors="pt")
    if DEVICE == "cuda":
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp, output_attentions=True)
    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    # Take last token's attention pattern for each head at each layer
    attn_per_layer = []
    for layer_attn in out.attentions:
        # shape: (1, num_heads, seq_len, seq_len)
        # Get last token's attention over all positions, per head
        last_tok_attn = layer_attn[0, :, -1, :].cpu().float()  # (num_heads, seq_len)
        attn_per_layer.append(last_tok_attn)
    return attn_per_layer

def attention_entropy(attn_dist):
    """Compute entropy of an attention distribution. Higher = more uniform."""
    # attn_dist: (seq_len,) — already softmaxed
    # Clamp to avoid log(0)
    p = attn_dist.clamp(min=1e-10)
    return -(p * p.log()).sum().item()

def max_entropy(seq_len):
    """Maximum possible entropy for a distribution of given length."""
    return math.log(seq_len) if seq_len > 0 else 0

def attention_pattern_similarity(attn_a, attn_b):
    """Compare attention patterns between two words using Jensen-Shannon divergence.
    Lower JSD = more similar patterns. Returns 1 - JSD for similarity."""
    sims = []
    for layer_a, layer_b in zip(attn_a, attn_b):
        # Pad to same length (different words may tokenize to different lengths)
        min_heads = min(layer_a.shape[0], layer_b.shape[0])
        min_len = min(layer_a.shape[1], layer_b.shape[1])
        a = layer_a[:min_heads, :min_len]
        b = layer_b[:min_heads, :min_len]
        # Re-normalize after truncation
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        b = b / b.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        # Average JSD across heads
        m = 0.5 * (a + b)
        kl_am = (a * (a / m.clamp(min=1e-10)).log()).sum(dim=-1)
        kl_bm = (b * (b / m.clamp(min=1e-10)).log()).sum(dim=-1)
        jsd = 0.5 * (kl_am + kl_bm)
        sims.append(1.0 - jsd.mean().item())  # 1 - JSD as similarity
    return sims

# ── EXTRACT ATTENTION FOR ALL WORDS ───────────────────────────────────────────
print(f"Extracting attention patterns for all words...")
all_words = list({w for pair in RELATED + UNRELATED for w in pair})
attn_cache = {w: get_attention(w) for w in all_words}
print(f"  Cached {len(attn_cache)} words x {N} layers x {NUM_HEADS} heads\n")

# ── TEST 1: ATTENTION ENTROPY PER LAYER ──────────────────────────────────────
print(f"{'='*75}")
print("TEST 1: ATTENTION ENTROPY BY LAYER")
print("(Low entropy = focused attention, High = uniform/lost)")
print(f"{'='*75}")
print(f"{'Layer':>6}  {'mean_H':>8}  {'norm_H':>8}  {'min_H':>8}  {'max_H':>8}  bar")
print("-" * 75)

entropy_by_layer = []
for layer in range(N):
    layer_entropies = []
    for word, attn_layers in attn_cache.items():
        attn = attn_layers[layer]  # (num_heads, seq_len)
        seq_len = attn.shape[1]
        for head in range(attn.shape[0]):
            h = attention_entropy(attn[head])
            layer_entropies.append(h)

    mean_h = np.mean(layer_entropies)
    max_possible = max_entropy(seq_len)  # using last word's seq_len as representative
    norm_h = mean_h / max_possible if max_possible > 0 else 0

    bar_len = int(norm_h * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"  L{layer:02d}   {mean_h:>8.4f}  {norm_h:>8.4f}  {min(layer_entropies):>8.4f}  {max(layer_entropies):>8.4f}  {bar}")

    entropy_by_layer.append({
        "layer": layer,
        "mean_entropy": round(mean_h, 4),
        "normalized_entropy": round(norm_h, 4),
        "min_entropy": round(min(layer_entropies), 4),
        "max_entropy": round(max(layer_entropies), 4),
        "interpretation": "focused" if norm_h < 0.5 else "moderate" if norm_h < 0.8 else "diffuse",
    })

# ── TEST 2: ATTENTION SIMILARITY FOR RELATED VS UNRELATED PAIRS ─────────────
print(f"\n{'='*75}")
print("TEST 2: ATTENTION PATTERN SIMILARITY (Related vs Unrelated)")
print("(Related pairs should have more similar attention than unrelated)")
print(f"{'='*75}")
print(f"{'Layer':>6}  {'rel_sim':>9}  {'unrel_sim':>10}  {'sep':>8}  {'p-val':>8}  verdict")
print("-" * 75)

attn_separation = []
for layer in range(N):
    rel_sims = []
    for a, b in RELATED:
        sim_per_layer = attention_pattern_similarity(attn_cache[a], attn_cache[b])
        rel_sims.append(sim_per_layer[layer])

    unrel_sims = []
    for a, b in UNRELATED:
        sim_per_layer = attention_pattern_similarity(attn_cache[a], attn_cache[b])
        unrel_sims.append(sim_per_layer[layer])

    rel_avg = np.mean(rel_sims)
    unrel_avg = np.mean(unrel_sims)
    sep = rel_avg - unrel_avg

    if len(rel_sims) >= 3 and len(unrel_sims) >= 3:
        try:
            _, pval = scipy_stats.mannwhitneyu(rel_sims, unrel_sims, alternative='greater')
        except ValueError:
            pval = 1.0
    else:
        pval = 1.0

    verdict = ("STRUCTURED" if sep > 0.02 and pval < 0.05
               else "WEAK" if sep > 0
               else "NO STRUCTURE")
    sig = "*" if pval < 0.05 else " "

    print(f"  L{layer:02d}   {rel_avg:>9.4f}  {unrel_avg:>10.4f}  {sep:>+8.4f}  {pval:>7.4f}{sig} {verdict}")

    attn_separation.append({
        "layer": layer,
        "related_sim": round(rel_avg, 4),
        "unrelated_sim": round(unrel_avg, 4),
        "separation": round(sep, 4),
        "p_value": round(pval, 6),
        "significant": pval < 0.05,
        "verdict": verdict,
    })

# ── TEST 3: HEAD SPECIALIZATION ──────────────────────────────────────────────
print(f"\n{'='*75}")
print("TEST 3: HEAD SPECIALIZATION BY LAYER")
print("(High = heads do different things, Low = redundant heads)")
print(f"{'='*75}")
print(f"{'Layer':>6}  {'head_div':>10}  {'interpret':<15}")
print("-" * 45)

head_specialization = []
for layer in range(N):
    # Collect average attention patterns per head across all words
    head_patterns = [[] for _ in range(NUM_HEADS)]
    for word, attn_layers in attn_cache.items():
        attn = attn_layers[layer]
        for h in range(min(attn.shape[0], NUM_HEADS)):
            head_patterns[h].append(attn[h].numpy())

    # Average pattern per head
    avg_patterns = []
    for h in range(NUM_HEADS):
        if head_patterns[h]:
            # Pad to same length and average
            max_len = max(p.shape[0] for p in head_patterns[h])
            padded = [np.pad(p, (0, max_len - p.shape[0])) for p in head_patterns[h]]
            avg_patterns.append(np.mean(padded, axis=0))

    # Measure diversity: average pairwise cosine distance between heads
    if len(avg_patterns) >= 2:
        dists = []
        for i in range(len(avg_patterns)):
            for j in range(i + 1, len(avg_patterns)):
                a = torch.tensor(avg_patterns[i]).unsqueeze(0)
                b = torch.tensor(avg_patterns[j]).unsqueeze(0)
                sim = F.cosine_similarity(a, b).item()
                dists.append(1.0 - sim)  # distance = 1 - similarity
        diversity = np.mean(dists)
    else:
        diversity = 0.0

    interp = "specialized" if diversity > 0.3 else "moderate" if diversity > 0.1 else "redundant"
    print(f"  L{layer:02d}   {diversity:>10.4f}  {interp:<15}")

    head_specialization.append({
        "layer": layer,
        "head_diversity": round(diversity, 4),
        "interpretation": interp,
    })

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")

structured_layers = sum(1 for a in attn_separation if a["verdict"] == "STRUCTURED")
specialized_layers = sum(1 for h in head_specialization if h["interpretation"] == "specialized")
focused_layers = sum(1 for e in entropy_by_layer if e["interpretation"] == "focused")

print(f"  Structured attention layers (rel > unrel, p<0.05): {structured_layers}/{N}")
print(f"  Specialized head layers (diversity > 0.3):         {specialized_layers}/{N}")
print(f"  Focused attention layers (norm entropy < 0.5):     {focused_layers}/{N}")

if structured_layers > N // 3:
    print(f"\n  The model shows meaningful attention structure.")
    print(f"  Attention patterns differentiate related from unrelated concepts.")
else:
    print(f"\n  Attention patterns don't strongly differentiate semantic relationships.")
    print(f"  This doesn't mean no understanding — it may encode semantics elsewhere.")

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_file = f"bol_attention_map_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "num_layers": N,
        "num_heads": NUM_HEADS,
        "entropy_by_layer": entropy_by_layer,
        "attention_separation": attn_separation,
        "head_specialization": head_specialization,
        "structured_layers": structured_layers,
        "specialized_layers": specialized_layers,
        "focused_layers": focused_layers,
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Cross-reference with bol_silhouette to see if high-separation layers also have structured attention.")
