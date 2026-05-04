# bol_config.py
# SINGLE SOURCE OF TRUTH for model, label, and shared settings.
# Change these ONCE here, then every script picks them up.
#
# USAGE: from bol_config import *

import numpy as np
import torch
from transformers import AutoTokenizer

# ── MODEL CONFIG (CHANGE THESE WHEN SWITCHING SUBJECTS) ──────────────────────
MODEL = "google/gemma-4-E2B-it"
TOKENIZER = MODEL
LABEL = "gpt-merged"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── HARDWARE ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # FP32 for scientific accuracy — do not change

# ── SHARED TEST DATA ─────────────────────────────────────────────────────────
EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "In the beginning was the word and the word was with God",
    "def hello_world(): print('hello world')",
    "The population of Tokyo is approximately 14 million people",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure",
]

PROBES = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time there was",
    "The meaning of life is",
    "def fibonacci(n):",
]

RELATED_PAIRS = [
    ("dog", "wolf"), ("cat", "lion"), ("river", "ocean"),
    ("happy", "joyful"), ("cold", "frozen"), ("king", "queen"),
    ("python", "code"), ("car", "vehicle"), ("run", "sprint"),
]

UNRELATED_PAIRS = [
    ("dog", "democracy"), ("cat", "algebra"), ("river", "justice"),
    ("happy", "concrete"), ("cold", "philosophy"), ("king", "equation"),
    ("python", "banana"), ("car", "silence"), ("run", "orange"),
]

# ── SEMANTIC PROBE CLUSTERS ──────────────────────────────────────────────────
CLUSTER_WORDS = [
    "dog", "wolf", "puppy",
    "cat", "kitten", "tiger",
    "car", "truck", "vehicle",
    "banana", "apple", "fruit",
    "happy", "joy", "sadness",
    "code", "python", "programming",
    "silence", "peace", "war",
]

CLUSTERS = {
    "animals": ["dog", "wolf", "puppy"],
    "felines": ["cat", "kitten", "tiger"],
    "vehicles": ["car", "truck", "vehicle"],
    "food": ["banana", "apple", "fruit"],
    "emotions": ["happy", "joy", "sadness"],
    "tech": ["code", "python", "programming"],
    "abstract": ["silence", "peace", "war"],
}

# ── TRANSPLANT CONFIG ────────────────────────────────────────────────────────
DONOR_MODEL = "juiceb0xc0de/bol-llama-3b-muon"
RECIPIENT_MODEL = "juiceb0xc0de/bol-llama-3b-muon"

# ── COMPRESSION MAP ──────────────────────────────────────────────────────────
NOISE_LEVELS = {
    "int8": 0.004,
    "int6": 0.016,
    "int4": 0.063,
    "int2": 0.250,
}
COMPRESSION_THRESHOLD = 0.5

# ── ARTIFACT BUILDER ─────────────────────────────────────────────────────────
TARGET_BYTES = 16_000_000
KEEP_LAYERS = list(range(7, 23))
KEEP_EMBEDDINGS = False

# ── BANNER ───────────────────────────────────────────────────────────────────
def print_banner(script_name):
    """Print the standard BoL header."""
    print(f"{'='*65}")
    print(f"  {script_name}")
    print(f"{'='*65}")
    print(f"  Scientific Integrity Mode: ON (FP32)")
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {MODEL}")
    print(f"  Label:  {LABEL}")
    print(f"{'='*65}\n")


def json_default(o):
    """Serialize NumPy scalars and arrays in BoL result dumps."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
