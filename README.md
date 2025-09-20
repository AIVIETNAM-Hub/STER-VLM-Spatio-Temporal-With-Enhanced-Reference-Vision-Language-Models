# STER-VLM: Spatio-Temporal With Enhanced Reference Vision Language Models



# Prerequisite
This repo use uv for environment configuration. First install uv
```bash
pip install uv
```

Next, run the following command to sync and activate the virtual environment
```bash
uv sync
source .venv/bin/activate
# .venv/Scripts/activate.bat (Windows)
```

# Data Preparation
## Download the dataset, prepare the workspace
- Download the [WTS](https://github.com/woven-visionai/wts-dataset?tab=readme-ov-file) dataset and unzip the folder. Also, prepare the preprocess folder(<preprocessed_folder_path>)

## Prepare the training/validation code
- <dataset_path>: the directory to dataset e.g /home/.../wts_dataset_zip
- <preprocessed_path>: the directory to store preprocessed data

1. Extract information from the dataset
```bash
./src_cleaned/setup/extract_draw_trainval.sh <dataset_path> <preprocessed_path>
```
This script will prepare the images for the training
2. Caption Decomposition
Run the following file to preprocess the caption. Make sure to align the arguments described in the script. More information in [setup README](./src_cleaned/setup/README.md)
```bash
./src_cleaned/setup/preprocess/extract/prepare_fixed_and_notfixed_caption.sh
```

3. Generate the reference caption (More instruction on [Ref README](./src_cleaned/setup/preprocess/gen/README.md))
```bash
./src_cleaned/setup/preprocess/gen/prepare_ref_caption.sh train_val
```

4. Transform into Qwen-2.5-VL format
```bash
./src_cleaned/setup/transform_trainval.sh <preprocessed_path>
```
# Preparing the TestSet
The preparation for the testset is instructed in [setup README](./src_cleaned/setup/README.md)

# Training and Inference
Up until now, you have already have the training dataset(train/val) for fixed, not fixed and total caption. Head to the [Tuning Guide](./src_cleaned/train/README.md)

# Evaluation

After that, from the prediction of the model (which already have the ground truth attach), run the file with the correct arguments described in the script. [Evaluation Script](./src_cleaned/train/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/evaluation.py)

# PostProcessing for Scoring

After obtaining the result from each model. Using the postprocessing script from [Postprocessing caption](./src_cleaned/postprocess/postprocess_caption.py) and [PostProcessing vqa](./src_cleaned/postprocess/postprocess_vqa.py).




