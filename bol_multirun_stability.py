# bol_multirun_stability.py
# PURPOSE: Run the embedding stripper SHUFFLE test multiple times with
# different random seeds. Reports mean + std of separation scores.
# Proves your results aren't a lucky permutation.
#
# This is your reproducibility evidence. A reviewer will want to see
# that the octopus phase is stable across random seeds, not a fluke.
#
# HOW TO USE:
#   python bol_multirun_stability.py
#   Outputs: terminal table + bol_multirun_stability_<LABEL>.json
#
# CONFIGURABLE: see bol_config.py, N_RUNS below

import torch
import torch.nn.functional as F
import json
import numpy as np
import hashlib
from scipy import stats as scipy_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from bol_config import MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, RELATED_PAIRS, UNRELATED_PAIRS, print_banner, json_default

print_banner("bol_multirun_stability.py — Reproducibility Test")

N_RUNS = 10  # Number of random seeds to test

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
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def measure_separation(model):
    """Measure separation at all layers. Returns list of per-layer separation values."""
    all_words = list({w for pair in related + unrelated for w in pair})
    cache = {w: get_all_layers(model, w) for w in all_words}

    seps = []
    for layer in range(N + 1):
        rel_sims = [cos(cache[a][layer], cache[b][layer]) for a, b in related]
        unrel_sims = [cos(cache[a][layer], cache[b][layer]) for a, b in unrelated]
        seps.append(np.mean(rel_sims) - np.mean(unrel_sims))
    return seps

def verify_hash(model):
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]

# ── SAVE ORIGINAL WEIGHTS ────────────────────────────────────────────────────
print("Saving original weight snapshot...")
original_weights = {}
for name, param in base.named_parameters():
    original_weights[name] = param.data.clone()

baseline_hash = verify_hash(base)

# ── BASELINE (no shuffling) ──────────────────────────────────────────────────
print("Computing baseline separation...")
baseline_seps = measure_separation(base)
print(f"  Baseline best separation: {max(baseline_seps):+.4f} at layer {np.argmax(baseline_seps)}\n")

# ── MULTI-RUN SHUFFLE TEST ───────────────────────────────────────────────────
print(f"Running SHUFFLE test {N_RUNS} times with different random seeds...\n")

all_run_seps = []  # (N_RUNS, N+1) — separation per layer per run

for run in range(N_RUNS):
    seed = run * 42 + 7
    torch.manual_seed(seed)

    # Shuffle embeddings
    embed_names = []
    for name, param in base.named_parameters():
        if "embed_tokens" in name:
            perm = torch.randperm(param.data.shape[0])
            param.data.copy_(original_weights[name][perm])
            embed_names.append(name)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Measure
    seps = measure_separation(base)
    all_run_seps.append(seps)

    best_sep = max(seps)
    best_layer = np.argmax(seps)
    middle_survive = any(s > 0.05 for s, l in zip(seps, range(N + 1)) if 5 <= l <= N - 5)
    status = "H1" if middle_survive else "H2"

    print(f"  Run {run+1:02d} (seed={seed:4d}): best_sep={best_sep:+.4f} at L{best_layer:02d}  [{status}]")

    # Restore
    for name in embed_names:
        param = base.get_parameter(name)
        param.data.copy_(original_weights[name])

# ── VERIFY INTEGRITY ──────────────────────────────────────────────────────────
final_hash = verify_hash(base)
if final_hash == baseline_hash:
    print(f"\nINTEGRITY VERIFIED (Hash: {final_hash})")
else:
    print(f"\nINTEGRITY FAIL!")

# ── STATISTICAL ANALYSIS ─────────────────────────────────────────────────────
all_run_seps = np.array(all_run_seps)  # (N_RUNS, N+1)

print(f"\n{'='*85}")
print(f"STABILITY ANALYSIS ACROSS {N_RUNS} RUNS")
print(f"{'='*85}")
print(f"{'Layer':>6}  {'baseline':>10}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'CV':>8}  stable?")
print("-" * 85)

stability_data = []
for layer in range(N + 1):
    bl = baseline_seps[layer]
    vals = all_run_seps[:, layer]
    mean = np.mean(vals)
    std = np.std(vals)
    cv = abs(std / mean) if abs(mean) > 1e-6 else float('inf')
    stable = "YES" if cv < 0.5 else "MARGINAL" if cv < 1.0 else "NO"

    print(f"  L{layer:02d}   {bl:>+10.4f}  {mean:>+8.4f}  {std:>8.4f}  {min(vals):>+8.4f}  {max(vals):>+8.4f}  {cv:>8.2f}  {stable}")

    stability_data.append({
        "layer": layer,
        "baseline_sep": round(bl, 4),
        "mean_sep": round(mean, 4),
        "std_sep": round(std, 4),
        "min_sep": round(min(vals), 4),
        "max_sep": round(max(vals), 4),
        "cv": round(cv, 4) if cv != float('inf') else None,
        "stable": stable,
    })

# ── OVERALL VERDICT ──────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("OVERALL VERDICT")
print(f"{'='*85}")

# How many runs support H1?
h1_runs = 0
for run_seps in all_run_seps:
    middle_survive = any(s > 0.05 for s, l in zip(run_seps, range(N + 1)) if 5 <= l <= N - 5)
    if middle_survive:
        h1_runs += 1

print(f"  Runs supporting H1 (octopus phase): {h1_runs}/{N_RUNS}")
print(f"  Runs supporting H2 (embedding-dependent): {N_RUNS - h1_runs}/{N_RUNS}")

# Stable layers
stable_layers = sum(1 for s in stability_data if s["stable"] == "YES")
print(f"  Stable layers (CV < 0.5):           {stable_layers}/{N+1}")

# Consistent best layer?
best_layers = [np.argmax(run) for run in all_run_seps]
mode_best = max(set(best_layers), key=best_layers.count)
mode_count = best_layers.count(mode_best)
print(f"  Most common best layer:             L{mode_best:02d} ({mode_count}/{N_RUNS} runs)")

if h1_runs == N_RUNS:
    print(f"\n  REPRODUCIBLE: H1 holds across all {N_RUNS} seeds.")
    print(f"  The octopus phase is not a lucky permutation.")
    reproducibility = "FULLY_REPRODUCIBLE"
elif h1_runs > N_RUNS * 0.8:
    print(f"\n  MOSTLY REPRODUCIBLE: H1 holds in {h1_runs}/{N_RUNS} runs.")
    print(f"  Strong evidence, minor seed sensitivity.")
    reproducibility = "MOSTLY_REPRODUCIBLE"
elif h1_runs > N_RUNS * 0.5:
    print(f"\n  MIXED: H1 holds in {h1_runs}/{N_RUNS} runs.")
    print(f"  The effect exists but is seed-sensitive. Report with caveats.")
    reproducibility = "MIXED"
else:
    print(f"\n  NOT REPRODUCIBLE: H1 only holds in {h1_runs}/{N_RUNS} runs.")
    print(f"  The effect may be a permutation artifact.")
    reproducibility = "NOT_REPRODUCIBLE"

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_file = f"bol_multirun_stability_{LABEL}.json"
with open(out_file, "w") as f:
    json.dump({
        "label": LABEL,
        "model": MODEL,
        "num_layers": N,
        "n_runs": N_RUNS,
        "reproducibility": reproducibility,
        "h1_runs": h1_runs,
        "h2_runs": N_RUNS - h1_runs,
        "most_common_best_layer": int(mode_best),
        "best_layer_consistency": f"{mode_count}/{N_RUNS}",
        "baseline_separations": [round(s, 4) for s in baseline_seps],
        "stability_by_layer": stability_data,
        "all_run_best_layers": [int(l) for l in best_layers],
    }, f, indent=2, default=json_default)
print(f"\nSaved: {out_file}")
print("Include this as your reproducibility evidence.")
print("Report: mean +/- std separation, and the H1/H2 consistency ratio.")
