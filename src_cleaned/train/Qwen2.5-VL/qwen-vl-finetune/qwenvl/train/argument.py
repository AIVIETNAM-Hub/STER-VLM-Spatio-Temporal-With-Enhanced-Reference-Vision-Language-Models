import transformers
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Either a full-model checkpoint or an adapter folder"}
    )
    base_model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Base model to load config and core weights from"}
    )
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning"})
    use_dora: bool = field(default=False, metadata={"help": "Enable DoRA fine-tuning"})

    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules (comma-separated)"}
    )




@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 350)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_images_per_sample: int = field(default=6, metadata={"help": "Maximum number of images per sample"})
    
    dataset_percentage: float = field(default=1.0, metadata={"help": "Percentage of total dataset to use (0.0-1.0, for debugging)"})
    train_split_ratio: float = field(default=0.9, metadata={"help": "Ratio of selected data to use for training (0.0-1.0)"})
    val_split_ratio: float = field(default=0.1, metadata={"help": "Ratio of selected data to use for validation (0.0-1.0)"})
    validation_split_seed: int = field(default=42, metadata={"help": "Random seed for train/val split"})
    shuffle_before_split: bool = field(default=True, metadata={"help": "Shuffle data before splitting"})

    validation_mode: str = field(default="split", metadata={"help": "Validation mode: 'split' (auto split), 'separate' (provide separate files), 'none' (no validation)"})
    val_dataset_use: Optional[str] = field(default="aicitydataset_fixed_val", metadata={"help": "Separate validation dataset name (when validation_mode='separate')"})

    skip_validation: bool = field(default=False, metadata={"help": "Skip validation entirely"})
    validation_subset_size: Optional[int] = field(default=None, metadata={"help": "Limit validation to N samples (None for all)"})
    validation_batch_size: Optional[int] = field(default=None, metadata={"help": "Validation batch size (None to use train batch size)"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    project_name: str = field(default="qwen_vl_finetune", metadata={"help": "Project name"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Run name (auto-generated if None)"})
    experiment_dir: str = field(default="./experiments", metadata={"help": "Base directory for experiments"})

    optimizer_type: str = field(
        default="adamw", 
        metadata={"help": "Optimizer type: adamw, adam, sgd, adafactor, lion"}
    )
    scheduler_type: str = field(
        default="cosine", 
        metadata={"help": "LR scheduler: linear, cosine, polynomial, constant, constant_with_warmup"}
    )
    cosine_restarts: int = field(default=0, metadata={"help": "Number of cosine restarts"})
    poly_power: float = field(default=1.0, metadata={"help": "Polynomial decay power"})
    
    max_steps: int = field(default=-1, metadata={"help": "Maximum number of training steps (-1 for epoch-based)"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Ratio of warmup steps to total steps"})
    warmup_steps: int = field(default=0, metadata={"help": "Number of warmup steps (overrides warmup_ratio if > 0)"})
    
    base_learning_rate: Optional[float] = field(default=None, metadata={"help": "Base learning rate (alias for learning_rate)"})
    adapter_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for adapter parameters"})
    
    optimizer_eps: float = field(default=1e-8, metadata={"help": "Optimizer epsilon"})
    optimizer_betas: str = field(default="0.9,0.999", metadata={"help": "Optimizer betas (comma-separated)"})

    use_deepspeed: bool = field(default=False, metadata={"help": "Enable DeepSpeed"})
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "DeepSpeed config file"})
    use_flash_attention: bool = field(default=True, metadata={"help": "Enable Flash Attention 2"})
    
    
    use_wandb: bool = field(default=False, metadata={"help": "Enable Weights & Biases logging"})
    use_tensorboard: bool = field(default=True, metadata={"help": "Enable TensorBoard logging"})
    log_model_checkpoints: bool = field(default=False, metadata={"help": "Log model checkpoints to wandb"})
    
    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy: no, steps, epoch"})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy: no, steps, epoch"})
    eval_steps: int = field(default=500, metadata={"help": "Number of update steps between evaluations"})
    save_steps: int = field(default=500, metadata={"help": "Number of update steps between saves"})
    eval_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "Number of predictions steps to accumulate before moving to CPU"})
    compute_metrics: bool = field(default=True, metadata={"help": "Whether to compute custom metrics during evaluation"})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "Metric for best model selection"})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at the end of training"})
    greater_is_better: bool = field(default=False, metadata={"help": "Whether higher metric is better"})

    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})
    early_stopping_patience: int = field(default=0, metadata={"help": "Early stopping patience (0 to disable)"})
    
    gradient_accumulation_steps: int = field(default=1)
    max_grad_norm: float = field(default=1.0)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    
    fp16_full_eval: bool = field(default=False, metadata={"help": "Use fp16 for full evaluation"})
    bf16_full_eval: bool = field(default=True, metadata={"help": "Use bf16 for full evaluation"})

    save_validation_predictions: bool = field(
        default=True,
        metadata={"help": "Whether to save validation predictions and ground truth"}
    )
    
    save_predictions_every_n_evals: int = field(
        default=2,
        metadata={"help": "Save predictions every N evaluation steps"}
    )
    
    max_validation_samples_to_save: int = field(
        default=100,
        metadata={"help": "Maximum number of validation samples to save per evaluation"}
    )


@dataclass
class InferenceArguments:
    checkpoint_folder : str = field(default='')
    output_path: str = field(default='')

