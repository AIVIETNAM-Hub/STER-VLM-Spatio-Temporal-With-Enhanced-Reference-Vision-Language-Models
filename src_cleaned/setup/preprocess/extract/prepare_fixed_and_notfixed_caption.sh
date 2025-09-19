#!/usr/bin/env bash
set -euo pipefail

cd "/home/s48gb/Desktop/GenAI4E/aicity_02"
source .venv/bin/activate

# Require HF token for Qwen
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  echo "Please export HUGGINGFACEHUB_API_TOKEN (or place API_KEY in .env)."
  echo "Example: export HUGGINGFACEHUB_API_TOKEN='hf_xxx...'"
  exit 1
fi

cd src/preprocess/extract_info

# Run decomposition for WTS and BDD splits
python extract_fixed_and_not_fixed_caption.py --dataset wts --split train
python extract_fixed_and_not_fixed_caption.py --dataset wts --split val
python extract_fixed_and_not_fixed_caption.py --dataset bdd --split train
python extract_fixed_and_not_fixed_caption.py --dataset bdd --split val

echo "Done."
