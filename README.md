# Blocks of Life (BoL) Test Suite

**Research Question:** Can a neural network understand language without being able to speak it?

The "Octopus Theory" — named for the octopus that can open a jar through comprehension without language — proposes that there was a moment in transformer development where the model understood but couldn't communicate. This suite reverse-engineers transformer internals to find empirical evidence of non-communicative comprehension.

All tests run in FP32 Scientific Integrity Mode with SHA256 weight verification.

---

## Order of Use

Run these in sequence. Each phase builds on the previous one's output.

### Phase 1: Setup & Baseline
1. `bol_config.py` — Set your model, tokenizer, and label
2. `bol_switch.py` — (Optional) CLI shortcut to swap models
3. `bol_weight_fingerprint.py` — Record pre-surgery weight statistics

### Phase 2: Core Measurements
4. `bol_silhouette.py` — Map semantic separation at every layer
5. `bol_semantic_probe.py` — Validate cluster geometry
6. `bol_random_baseline.py` — Establish null hypothesis baseline

### Phase 3: Surgical Analysis
7. `bol_layer_sweep.py` — Ablate layers one at a time
8. `bol_component_ablation.py` — Ablate component types across all layers
9. `bol_embedding_stripper.py` — The core octopus test (SHUFFLE/RANDOM/ZERO)
10. `bol_trisect.py` — Quarter-cut zone bypass experiments
11. `bol_wormhole.py` — Extended layer bypass experiments
12. `bol_transplant_lab.py` — Cross-model layer transplant

### Phase 4: Deep Analysis
13. `bol_attention_map.py` — Attention entropy and head specialization
14. `bol_cka.py` — Representation similarity across layers
15. `bol_gradient_attribution.py` — Gradient-based layer importance

### Phase 5: Validation & Reproducibility
16. `bol_multirun_stability.py` — Reproducibility across 10 random seeds
17. `bol_humpty.py` — Reversibility verification (smash and restore)
18. `bol_compression_map.py` — Quantization noise tolerance per layer

### Phase 6: Build
19. `bol_artifact_builder.py` — Assemble final 16MB artifact for Parameter Golf

---

## File Reference

### bol_config.py
**Single source of truth.** Every script imports MODEL, TOKENIZER, LABEL, DEVICE, DTYPE, test data, and cluster definitions from here. Change models once, all scripts pick it up. TOKENIZER can be set separately from MODEL for surgically pruned models whose tokenizer lives in a different repo.

**Importance:** Without centralized config, switching subjects required editing 18 files. This eliminated hours of debugging from mismatched settings.

### bol_switch.py
**CLI model switcher.** `python bol_switch.py "repo/model" "label" [--tokenizer "repo/tok"]` rewrites bol_config.py in place.

**Importance:** One-command model switching for rapid cross-subject comparison.

### bol_weight_fingerprint.py
**Pre-surgery weight statistics.** Records per-layer weight norms, means, stds, and SHA256 hashes before any experiments touch the model.

**Importance:** Provides the "before" snapshot. If post-experiment weights don't match, you know something went wrong. Essential for reproducibility claims in a paper.

### bol_silhouette.py
**Semantic separation map.** For every layer, measures cosine similarity between related word pairs (dog/wolf) vs unrelated pairs (dog/democracy). Computes separation score with Mann-Whitney U significance testing.

**Importance:** This is how you find load-bearing layers. High separation = the layer is doing semantic work. Cross-subject comparison revealed that Muon training eliminated the L11 spike seen in base models and distributed comprehension smoothly — a key finding.

### bol_semantic_probe.py
**Cluster validation.** Groups words into semantic clusters (animals, vehicles, emotions, etc.) and measures within-cluster vs cross-cluster cosine similarity at each layer.

**Importance:** Validates silhouette findings with a different methodology. If both agree on which layers matter, the evidence is stronger. Also reveals whether the model organizes concepts into human-recognizable categories.

### bol_random_baseline.py
**Null hypothesis control.** Compares three conditions: real word pairs, random token IDs, and shuffled pairings. If random tokens show the same separation as real words, your other results are meaningless.

**Importance:** Every scientific claim needs a null hypothesis. This is the control that makes all other tests credible. Without it, a reviewer could argue the separation patterns are artifacts of the architecture, not learned comprehension.

### bol_layer_sweep.py
**Single-layer ablation.** Zeros out one layer at a time and measures perplexity damage on evaluation texts.

**Importance:** Complementary to silhouette. Silhouette tells you which layers carry meaning; layer sweep tells you which layers the model actually relies on for generation. A layer can be semantically rich but redundant if zeroing it causes no damage.

### bol_component_ablation.py
**Component-type ablation.** Zeros one component type (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) across ALL layers simultaneously.

**Importance:** Reveals which architectural components carry the most weight. If zeroing all k_proj barely hurts but zeroing all v_proj destroys output, that tells you where comprehension lives at the component level.

### bol_embedding_stripper.py
**The core octopus experiment.** Three modes:
- **SHUFFLE**: Randomly permutes embedding weights (primary test — destroys learned mappings but preserves weight distribution)
- **RANDOM**: Replaces embeddings with random values (secondary control)
- **ZERO**: Zeros all embeddings (baseline control — produces all-zero hidden states in bias-free architectures)

Measures whether post-strip transformer layers still show semantic separation.

**Importance:** This is THE experiment. If shuffled embeddings still produce semantic separation in the transformer layers, it means comprehension lives in the weights, not the input encoding. This is the strongest direct test of the octopus hypothesis.

### bol_trisect.py
**Quarter-cut zone bypass.** Divides the model into zones and bypasses specific quarters to test which regions are essential.

**Importance:** Tests whether comprehension is localized to a region or distributed. Combined with layer sweep and silhouette data, builds a complete map of where understanding lives.

### bol_wormhole.py
**Extended layer bypass.** Similar to trisect but with more flexible skip patterns, testing larger contiguous bypasses.

**Importance:** Extends trisect findings. If you can skip 50% of layers and still get semantic output, that's strong evidence for distributed (not localized) comprehension.

### bol_transplant_lab.py
**Cross-model layer transplant.** Takes layers from a donor model and inserts them into a recipient model. Measures whether transplanted layers maintain function.

**Importance:** Tests whether semantic knowledge is layer-portable across models. If a layer trained with Muon works when transplanted into an AdamW model, it suggests the representation format is universal, not optimizer-specific.

### bol_attention_map.py
**Attention pattern analysis.** Measures attention entropy (how focused vs diffuse each head is), pattern similarity across heads, and head specialization.

**Importance:** Reveals HOW the model processes meaning. Low entropy = focused attention (syntactic?). High entropy = distributed attention (semantic?). Head specialization shows whether different heads learned different roles.

### bol_cka.py
**Centered Kernel Alignment.** Computes representation similarity between every pair of layers. Produces a layer-to-layer similarity matrix and tracks how representations evolve through the network.

**Importance:** Shows where the model's internal representation changes most. Sharp CKA drops between adjacent layers = major transformation happening there. Smooth CKA = redundant layers (compression candidates).

### bol_gradient_attribution.py
**Gradient-based importance.** Measures gradient magnitude per layer and per component during forward/backward pass.

**Importance:** Provides a continuous (not binary) importance score, unlike ablation which is all-or-nothing. Catches layers that are important but whose ablation is compensated by neighboring layers.

### bol_multirun_stability.py
**Reproducibility test.** Runs the SHUFFLE embedding experiment 10 times with different random seeds and measures variance.

**Importance:** If results change wildly across seeds, they're noise. If they're stable, they're signal. Reviewers will ask about reproducibility — this answers them preemptively.

### bol_humpty.py
**Reversibility verification.** Smashes model weights, then restores from the fingerprint. Verifies SHA256 hashes match.

**Importance:** Proves your experimental pipeline doesn't permanently corrupt models. If Humpty can put it back together, your surgery was clean.

### bol_compression_map.py
**Quantization noise simulation.** Adds calibrated noise (int8/int6/int4/int2 levels) to each layer and measures damage.

**Importance:** Maps which layers tolerate compression and which don't. Directly informs the artifact builder about which layers can be quantized to fit the 16MB Parameter Golf limit without destroying comprehension.

### bol_artifact_builder.py
**Final assembly.** Builds the 16MB artifact for Parameter Golf submission using keep/compress decisions from all previous tests.

**Importance:** This is the deliverable. Everything else feeds into this. The artifact must demonstrate the research findings within the competition constraints.

---

## Output Files

Every script saves results as `bol_<test>_<LABEL>.json` in the working directory. These JSON files contain all raw data, computed metrics, and statistical tests for cross-referencing between experiments.

## Switching Models

```bash
# Option 1: Edit bol_config.py directly
# Option 2: Use the CLI tool
python bol_switch.py "org/model-name" "my-label" --tokenizer "org/tokenizer"
```

For pruned models where the tokenizer lives in a different repo, always set TOKENIZER explicitly.
