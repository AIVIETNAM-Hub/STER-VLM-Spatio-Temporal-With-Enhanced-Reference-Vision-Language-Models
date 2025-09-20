# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import uuid
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pprint import pprint
import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    EarlyStoppingCallback
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import Qwen2VLImageProcessor, Trainer, TrainerCallback, AutoProcessor

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
        AdaLoraConfig
    )

    from transformers import BitsAndBytesConfig
    PEFT_AVAILABLE = True
except Exception as e:
    PEFT_AVAILABLE = False
    print("PEFT not available. LoRA/QLoRA/AdaLoRA features will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available. W&B logging will be disabled.")




local_rank = None





        




class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")


def rank0_print(*args):
    try:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(*args)
        else:
            print(*args)
    except Exception:
        print(*args)


def setup_project_structure(training_args: TrainingArguments):

    if training_args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = str(uuid.uuid4())[:8]
        training_args.run_name = f"run_{timestamp}_{run_id}"
    
    project_dir = Path(training_args.experiment_dir) / training_args.project_name
    run_dir = project_dir / training_args.run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "configs").mkdir(exist_ok=True)

    training_args.output_dir = str(run_dir / "checkpoints")
    training_args.logging_dir = str(run_dir / "logs")

    return run_dir

def save_training_config(run_dir: Path, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    config_file = run_dir / 'configs' / "training_config.json"

    configs = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": training_args.to_dict(),
        "timestamp": datetime.now().isoformat()
    }
    with open(config_file, 'w') as f:
        json.dump(
            configs, f, indent=2, default=str
        )
    
    rank0_print(f"Training configuration saved to: {config_file}")


# def setup_quantization_config(
#     model_args: ModelArguments
# ):
#     if not model_args.use_qlora:
#         return None 
    
#     if not PEFT_AVAILABLE:
#         raise ImportError(f"PEFT is required for Qlora")

#     return BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type=model_args.qlora_quant_type,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=model_args.qlora_double_quant,
#     )



def setup_adapters(model, model_args: ModelArguments):
    target_modules = model_args.lora_target_modules.split(',') if model_args.lora_target_modules else None

    efficient_config = None
    message = "No efficient fine-tuning applied!"

    if model_args.use_dora and PEFT_AVAILABLE:
        efficient_config = {
            'r': model_args.lora_r,
            'lora_alpha': model_args.lora_alpha,
            'target_modules': target_modules,
            'lora_dropout': model_args.lora_dropout,
            'bias': 'none',
            'task_type': TaskType.CAUSAL_LM,
            'use_dora': True  
        }
        message = f"DoRA enabled with rank {model_args.lora_r} on modules {model_args.lora_target_modules}"

    elif model_args.use_lora and PEFT_AVAILABLE:
        efficient_config = {
            'r': model_args.lora_r,
            'lora_alpha': model_args.lora_alpha,
            'target_modules': target_modules,
            'lora_dropout': model_args.lora_dropout,
            'bias': 'none',
            'task_type': TaskType.CAUSAL_LM
        }
        message = f"LoRA enabled with rank {model_args.lora_r} on modules {model_args.lora_target_modules}"

    if efficient_config:
        config = LoraConfig(**efficient_config)
        model = get_peft_model(model, config)
        print(message)  

    return model 


def setup_monitoring(training_args: TrainingArguments, run_dir: Path):
    callbacks = []

    if training_args.early_stopping_patience > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=0.0
        )
        callbacks.append(early_stopping)
    
    if training_args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project = training_args.project_name,
            name = training_args.run_name,
            config = {
                "model_name": training_args.run_name,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
            }
        )
        
        rank0_print("WnB logging enabled")
    
    return callbacks



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  


def set_model(model_args, model):

    if hasattr(model, 'peft_config') or 'PeftModel' in str(type(model)):
        print(f"Detected LORA model - LoRA parameters are automatically trainable")


        if model_args.tune_mm_mlp:
            for n, p in model.named_parameters():
                if 'merger' in n or 'mm_projector' in n:
                    p.requires_grad = True
                    print(f"Enabled gradient for: {n}")

        return 

    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        print(f"")
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False




def debug_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found!")
    


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )



    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    local_rank = training_args.local_rank
    run_dir = setup_project_structure(training_args)
    save_training_config(run_dir=run_dir, model_args=model_args, data_args=data_args, training_args=training_args)

    os.makedirs(training_args.output_dir, exist_ok=True)


    adapter_path = model_args.model_name_or_path.strip()
    base_path = getattr(model_args, "base_model_or_path", None) or "Qwen/Qwen2.5-VL-7B-Instruct"

    if adapter_path == "":
        print("Model_name_or_path is empty. Using base model instead.")
        adapter_path = base_path
        is_adapter_checkpoint = False
    else:
        is_adapter_checkpoint = (
            os.path.isdir(adapter_path) and
            "adapter_model.safetensors" in os.listdir(adapter_path)
        )

    print(f"{adapter_path=}")
    print(f"{base_path=}")


    if is_adapter_checkpoint:
        print(f"🔗 Loading base model from {base_path} and applying adapter from {adapter_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            device_map=None if training_args.deepspeed else "auto",
        )


        data_args.image_processor = AutoProcessor.from_pretrained(
            base_path,  
            use_fast=True
        ).image_processor

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        data_args.model_type = "qwen2.5vl"

    else:
        print(f"📦 Loading full model from {adapter_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            adapter_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            device_map=None if training_args.deepspeed else 'auto'

        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            adapter_path,
            use_fast=True
        ).image_processor

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            adapter_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        data_args.model_type = "qwen2.5vl"


    model = setup_adapters(model, model_args)

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    
    set_model(model_args, model)

    debug_trainable_parameters(model)

    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    

  
        

    if data_module["eval_dataset"] is not None:
        if data_args.validation_batch_size is not None:
            training_args.per_device_eval_batch_size = data_args.validation_batch_size
        else:
            training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
        
        if training_args.eval_strategy == "no":
            training_args.eval_strategy = data_args.eval_strategy
        if training_args.eval_steps is None or training_args.eval_steps <= 0:
            training_args.eval_steps = data_args.eval_steps
        



    callbacks = setup_monitoring(training_args, run_dir)
    callbacks.append(LossLoggingCallback()) 

 
    

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer, 
        callbacks=callbacks,
        **data_module 
    )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


    summary_file = run_dir/  "training_summary.json"
    summary = {
        'completed_at': datetime.now().isoformat(),
        'final_loss': trainer.state.log_history[-1].get('train_loss', None) if trainer.state.log_history else None  ,
        'total_steps': trainer.state.global_step,
        'run_dir': str(run_dir),
        'model_path': training_args.output_dir,
    }

    with open(summary_file, 'w') as f:
        json.dump(
            summary, f, indent=2, default=str
        )

    rank0_print(f"Training completed. Summary saved to: {summary_file}")

    if training_args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
