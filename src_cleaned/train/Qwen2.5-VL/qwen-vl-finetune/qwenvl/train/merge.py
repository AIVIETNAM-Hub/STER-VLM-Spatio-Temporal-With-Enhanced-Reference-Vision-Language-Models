import os
import sys
import json
import torch
import argparse
import transformers
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from transformers import set_seed
set_seed(4242)
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Qwen2VLImageProcessor,
    AutoTokenizer
)

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False






lora_checkpoint_path = Path("/home/s48gb/Desktop/GenAI4E/aicity_02/experiments/qwen_vl_total_caption_enhanced/checkpoints/checkpoint-1600")

main_training_output_dir = lora_checkpoint_path.parent.parent

print(f"DEBUG: LoRA Checkpoint Path: '{lora_checkpoint_path}'")
print(f"DEBUG: Main Training Output Dir (for configs): '{main_training_output_dir}'")

adapter_config_path = lora_checkpoint_path / "adapter_config.json"
with open(adapter_config_path, 'r') as f:
    adapter_config = json.load(f)
base_model_name = adapter_config.get("base_model_name_or_path")

print(f"INFO: Using base model name from LoRA adapter config: {base_model_name}")


base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map='cuda'
)
print(f"INFO: load base model")

model = PeftModel.from_pretrained(base_model, str(lora_checkpoint_path))
print(f"INFO: peft model")

model.merge_and_unload()
print(f"INFO: load merge and unload")

model = model.to(torch.bfloat16)
print(f"INFO: turn into bfloat 16")


print("oke")

# Set the directory where you want to save the merged model
output_dir = Path("/home/s48gb/Desktop/GenAI4E/aicity_02/experiments/qwen_vl_total_caption_enhanced/merge_ckpt")

# Create the directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Save the model and tokenizer
model.save_pretrained(output_dir)
print(f"INFO: Merged model saved to {output_dir}")

# Optionally save the tokenizer too
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)
print("INFO: Tokenizer saved")




