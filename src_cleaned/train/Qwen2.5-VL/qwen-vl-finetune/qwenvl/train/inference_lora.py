import os
import sys
import json
import torch
import argparse
import transformers
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
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

# Import your training components
from qwenvl.data import data_list
from qwenvl.data.data_qwen import LazySupervisedDataset, DataCollatorForInference
from qwenvl.train.argument import DataArguments, InferenceArguments
from PIL import Image



def contains_ordered_substrings(gt: str) -> bool:
    pos_a = gt.find("a.")
    if pos_a == -1:
        return False
    pos_b = gt.find("b.", pos_a + 1)
    if pos_b == -1:
        return False
    pos_c = gt.find("c.", pos_b + 1)
    if pos_c == -1:
        return False
    pos_d = gt.find("d.", pos_c + 1)
    return pos_d != -1



def inference():
    parser = transformers.HfArgumentParser((DataArguments, InferenceArguments))
    data_args, inference_args = parser.parse_args_into_dataclasses()

    lora_checkpoint_path = Path(inference_args.checkpoint_folder)

    main_training_output_dir = lora_checkpoint_path.parent.parent

    print(f"DEBUG: LoRA Checkpoint Path: '{lora_checkpoint_path}'")
    print(f"DEBUG: Main Training Output Dir (for configs): '{main_training_output_dir}'")
    print(f"DEBUG: Dataset for inference: '{data_args.dataset_use}'")
    print(f"DEBUG: Inference output JSON path: '{inference_args.output_path}'")

    adapter_config_path = lora_checkpoint_path / "adapter_config.json"
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path")

    print(f"INFO: Using base model name from LoRA adapter config: {base_model_name}")

    data_args.model_type = "qwen2.5vl"

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, str(lora_checkpoint_path), ephemeral_gpu_offload=True)
    model = model.merge_and_unload()
    model.to(torch.bfloat16)

    model = model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(lora_checkpoint_path),
        model_max_length=2500,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True
    )
    print(f"INFO: Loaded tokenizer from specific LoRA checkpoint: {lora_checkpoint_path}")


    image_processor = Qwen2VLImageProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
    

    data_args.image_processor = image_processor


    if tokenizer.pad_token_id is None:
        print("INFO: tokenizer.pad_token_id is None, setting to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id


    training_config_path = main_training_output_dir / "configs" / "training_config.json"
    if training_config_path.exists():
        with open(training_config_path, 'r') as f:
            full_training_config = json.load(f)
        saved_data_args = full_training_config.get("data_args", {})


    inference_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, inference_mode=True
    )
    data_collator = DataCollatorForInference(tokenizer=tokenizer)
    batch_size = getattr(inference_args, "batch_size", 32)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Total samples: {len(inference_dataset)}")



    results = []
    device = model.device


    ######DEBUG
    # data_dict = inference_dataset[0]
    # processed_data_dict = {}
    # for key, value in data_dict.items():
    #     if isinstance(value, torch.Tensor):
    #         processed_data_dict[key] = value.to(device)
    #     else:
    #         processed_data_dict[key] = value
    # input_ids = processed_data_dict["input_ids"]
    # pixel_values = processed_data_dict.get("pixel_values")
    # image_grid_thw = processed_data_dict.get("image_grid_thw")
    # position_ids = processed_data_dict["position_ids"]

    # inputs = {
    #     "input_ids": input_ids,
    #     "position_ids": position_ids,
    # }
    # inputs["attention_mask"] = input_ids.ne(tokenizer.pad_token_id)
    # if pixel_values is not None:
    #     inputs["pixel_values"] = pixel_values
    # if image_grid_thw is not None:
    #     inputs["image_grid_thw"] = image_grid_thw

    # temp_decoded_prompt = tokenizer.decode(input_ids[0])
    # print(f"DEBUG (Sample {0}): Decoded prompt for generation: {temp_decoded_prompt}")



    for batch in tqdm(inference_loader, desc="Inferencing batches"):
        # Move inputs to GPU
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Prepare generation inputs
        gen_inputs = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if batch.get("pixel_values") is not None:
            gen_inputs["pixel_values"]    = batch["pixel_values"]
            gen_inputs["image_grid_thw"]  = batch["image_grid_thw"]

        with torch.no_grad():
            generated = model.generate(
                **gen_inputs,
                max_new_tokens=1024,
                min_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True,
                tokenizer=tokenizer,
                stop_strings=["<|im_start|>", "\n\n", "<|im_end|>"],
            )

        prompt_lens = (batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1)

        all_texts = []
        for gen_ids, plen in zip(generated, prompt_lens):
            new_tokens = gen_ids[plen:].cpu().tolist()
            text = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            all_texts.append(text)

        start_idx = len(results)
        for idx, text in enumerate(all_texts):
            sample_idx = start_idx + idx
            orig = inference_dataset.list_data_dict[sample_idx]
            gt = ""
            for conv in orig.get("conversations", []):
                if conv.get("from") in ["gpt", "assistant"]:
                    gt = conv.get("value", "").strip()
                    break

            result = {
                "prediction": text,
                "ground_truth": gt,
                "image_paths": orig.get("images", []),
                "sample": orig
            }
            print(result)
            results.append(result)

    output_data = {
        "metadata": {
            "lora_checkpoint_path": str(lora_checkpoint_path),
            "main_training_output_dir": str(main_training_output_dir),
            "base_model_name": base_model_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(inference_dataset),
            "used_lazy_dataset": True
        },
        "results": results
    }
    output_path = Path(inference_args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n✅ Inference completed!")
    print(f"💾 Results saved to: {output_path}")


if __name__ == "__main__":
    inference()
