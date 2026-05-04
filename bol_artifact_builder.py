# bol_artifact_builder.py
# PURPOSE: Assemble the final 16MB submission artifact.
# Takes your chosen layers (from silhouette + sweep analysis),
# strips embeddings, and packs everything into a safetensors file.
# Run LAST after all analysis is complete.
#
# HOW TO USE:
#   1. Run bol_silhouette.py + bol_layer_sweep.py first
#   2. Adjust KEEP_LAYERS in bol_config.py with your chosen layer indices
#   3. python bol_artifact_builder.py
#   4. Check size output — iterate until it fits under 16MB
#
# CONFIGURABLE: see bol_config.py

import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

from bol_config import MODEL, TOKENIZER, LABEL, TARGET_BYTES, KEEP_LAYERS, KEEP_EMBEDDINGS, print_banner

print_banner("bol_artifact_builder.py — Final Assembly")

OUTPUT_DIR = "./bol_artifact"

model     = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

artifact    = {}
total_bytes = 0
skipped     = []

for name, param in model.named_parameters():
    # always keep final layer norm
    if name == "model.norm.weight":
        artifact[name] = param.data.clone().to(torch.float16)
        total_bytes += artifact[name].nelement() * 2
        continue

    # embeddings + lm_head
    if "embed_tokens" in name or "lm_head" in name:
        if KEEP_EMBEDDINGS:
            artifact[name] = param.data.clone().to(torch.float16)
            total_bytes += artifact[name].nelement() * 2
        else:
            skipped.append(name)
        continue

    # transformer layers
    keep = any(f".layers.{i}." in name for i in KEEP_LAYERS)
    if keep:
        artifact[name] = param.data.clone().to(torch.float16)
        total_bytes += artifact[name].nelement() * 2
    else:
        skipped.append(name)

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_file(artifact, f"{OUTPUT_DIR}/model.safetensors")
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n{'='*60}")
print(f"ARTIFACT REPORT")
print(f"{'='*60}")
print(f"Output dir:       {OUTPUT_DIR}")
print(f"Layers kept:      {KEEP_LAYERS}")
print(f"Embeddings:       {'yes' if KEEP_EMBEDDINGS else 'STRIPPED (octopus mode)'}")
print(f"Tensors included: {len(artifact)}")
print(f"Tensors skipped:  {len(skipped)}")
print(f"Raw size (fp16):  {total_bytes / 1e6:.2f} MB")
print(f"Target:           {TARGET_BYTES / 1e6:.2f} MB")

if total_bytes <= TARGET_BYTES:
    headroom = TARGET_BYTES - total_bytes
    print(f"Status:           FITS  ({headroom/1e6:.2f} MB headroom for probe head)")
else:
    over = total_bytes - TARGET_BYTES
    print(f"Status:           OVER by {over/1e6:.2f} MB")
    mb_per_layer = total_bytes / len(KEEP_LAYERS) / 1e6 if KEEP_LAYERS else 0
    layers_to_drop = int(over / (mb_per_layer * 1e6)) + 1
    print(f"Suggestion:       Remove ~{layers_to_drop} more layer(s) from KEEP_LAYERS")
    print(f"Per-layer cost:   ~{mb_per_layer:.2f} MB each")
