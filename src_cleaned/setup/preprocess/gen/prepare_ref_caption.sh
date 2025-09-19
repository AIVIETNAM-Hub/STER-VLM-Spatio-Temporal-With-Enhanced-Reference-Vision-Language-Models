#!/bin/bash

# Prepare data
echo "Transforming dataset!"
cd "/home/s48gb/Desktop/GenAI4E/aicity_02/"
source .venv/bin/activate

# Require HF token
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  echo "Please export HUGGINGFACEHUB_API_TOKEN (or place API_KEY in .env)."
  echo "Example: export HUGGINGFACEHUB_API_TOKEN='hf_xxx...'"
  exit 1
fi

cd "selectFrames_Triet/pipeline/temp"

# Parse argument
MODE=$1

if [ "$MODE" = "train_val" ]; then
  echo "Running for train + val splits"

  python transform.py --split train
  python transform.py --split val

  python gen_ref.py --split train --max-images-per-request 16
  python gen_ref.py --split val   --max-images-per-request 16

elif [ "$MODE" = "test" ]; then
  echo "Running for test split"

  python transform.py --split test
  python gen_ref.py --split test --max-images-per-request 16

else
  echo "Usage: $0 {train_val|test}"
  exit 1
fi

echo "Done."
