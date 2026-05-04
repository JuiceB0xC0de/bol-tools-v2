#!/usr/bin/env python3
# bol_switch.py — Switch model/tokenizer/label in one command.
#
# USAGE:
#   python bol_switch.py "HuggingFaceTB/SmolLM2-1.7B" "smollm2-1.7b"
#   python bol_switch.py "meta-llama/Llama-3.2-1B" "llama-3.2-1b"
#   python bol_switch.py "juiceb0xc0de/temp-drop-llama3.2-3b" "temp-drop" --tokenizer "meta-llama/Llama-3.2-3B"
#
# That's it. One command, everything updates.

import sys
import re
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python bol_switch.py MODEL_REPO LABEL [--tokenizer TOKENIZER_REPO]")
        print()
        print("Examples:")
        print('  python bol_switch.py "HuggingFaceTB/SmolLM2-1.7B" "smollm2-1.7b"')
        print('  python bol_switch.py "meta-llama/Llama-3.2-1B" "llama-1b"')
        print('  python bol_switch.py "juiceb0xc0de/pruned-model" "pruned" --tokenizer "meta-llama/Llama-3.2-3B"')
        sys.exit(1)

    model = sys.argv[1]
    label = sys.argv[2]
    tokenizer = None

    if "--tokenizer" in sys.argv:
        idx = sys.argv.index("--tokenizer")
        if idx + 1 < len(sys.argv):
            tokenizer = sys.argv[idx + 1]

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bol_config.py")

    with open(config_path, "r") as f:
        content = f.read()

    # Replace MODEL line
    content = re.sub(
        r'^MODEL\s*=\s*"[^"]*".*$',
        f'MODEL     = "{model}"',
        content, count=1, flags=re.MULTILINE
    )

    # Replace TOKENIZER line
    if tokenizer:
        content = re.sub(
            r'^TOKENIZER\s*=\s*.*$',
            f'TOKENIZER = "{tokenizer}"',
            content, count=1, flags=re.MULTILINE
        )
    else:
        content = re.sub(
            r'^TOKENIZER\s*=\s*.*$',
            f'TOKENIZER = None',
            content, count=1, flags=re.MULTILINE
        )

    # Replace LABEL line
    content = re.sub(
        r'^LABEL\s*=\s*"[^"]*".*$',
        f'LABEL     = "{label}"',
        content, count=1, flags=re.MULTILINE
    )

    with open(config_path, "w") as f:
        f.write(content)

    # Show what was set
    tok_display = tokenizer if tokenizer else model
    print(f"  Model:     {model}")
    print(f"  Tokenizer: {tok_display}")
    print(f"  Label:     {label}")
    print(f"\nUpdated: {config_path}")

if __name__ == "__main__":
    main()
