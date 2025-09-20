# Training Guide

This repo will contain all the instruction to fine-tune the model.

## Setup

### Installation

1. Download uv:
   ```bash
   pip install uv
   ```

2. At the root repo `AICITY`, install using uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Setup the Dataset

1. Head to the `__init__` file in the `aicity_02/src_cleaned/train/Qwen2.5-VL/qwen-vl-finetune/qwenvl/data`

2. In here you will see some examples of the way to setup the dataset. In the previous section, you have the json path of training and validation set. Create a constant of training and validation variables, and then register in the `data_dict` variables.

   Example:
   ```python
   TRAIN_SET = {
       'annotation_path': "/path to the train set"
   }
   VAL_SET = {
       'annotation_path': "/path to the val set"
   }

   data_dict = {
       ...,
       'train_set': TRAIN_SET,
       'val_set': VAL_SET
   }
   ```

### Setup the Training Script

Head to the training script: `aicity_02/src/train/Qwen2.5-VL/qwen-vl-finetune/scripts`

You will see multiple training scripts here, corresponding to the model we used in the paper. Run these scripts to obtain the checkpoints:

- **train_caption_fixed.sh**: This is the configuration for the fixed captioning model. This is the model that only learns the characteristics that do not change overtime (environment, clothes and characteristics of pedestrian). In this config, you need to fill in the train and val dataset that you have set in the `data_dict`.

- **train_caption_not_fixed.sh**: The same with the previous script. But this time, you plug in the dataset for the not-fixed training.

- **train_vqa_fixed.sh**: This is the training script for the QA checkpoint. In the `MODEL_NAME`, fill in the checkpoint path of model trained by `train_caption_fixed.sh`, and plug in the right QA dataset. All the configs have been set.

- **train_vqa_not_fixed.sh**: This is the training script for the not fixed QA checkpoint. In the `MODEL_NAME`, fill in the checkpoint path of model trained by `train_caption_not_fixed.sh`, and plug in the right QA dataset. All the configs have been set.

- **train_caption_total.sh**: This is the training script for the final model. Enter the appropriate dataset.

## Training

Run the above files. For the VQA, you need to have the best checkpoint from the corresponding model mentioned.

## Inference

Head to the `aicity_02/src/train/Qwen2.5-VL/qwen-vl-finetune/scripts/inference.sh`. Fill in the variable and run inference.

Also for the final total caption model, remember to preprocess the first 2 fixed and not fixed caption model's results based on the preprocess guide, and remember to register them.