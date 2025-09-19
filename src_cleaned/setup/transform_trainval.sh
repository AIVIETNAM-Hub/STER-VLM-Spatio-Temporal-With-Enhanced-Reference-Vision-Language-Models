#!/bin/bash

splits=("train" "val")
preprocessed_data_path="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned"
if [ -z "$1" ]; then
    echo "No preprocessed data path provided. Using default specified in file."
else
    preprocessed_data_path="$1"
    echo "Using provided preprocessed data path: $preprocessed_data_path"
fi

for split in "${splits[@]}"; do
    echo "Matching images with captions for split: $split"
    python src_cleaned/setup/preprocess/transform/match_caption.py --split "$split" --preprocessed_data_path "$preprocessed_data_path"
done

for split in "${splits[@]}"; do
    echo "Transforming component data for Qwen-2.5-VL for split: $split"
    python src_cleaned/setup/preprocess/transform/transform_component.py --split "$split" --preprocessed_data_path "$preprocessed_data_path"
done

for split in "${splits[@]}"; do
    echo "Transforming merge data for Qwen-2.5-VL for split: $split"
    python src_cleaned/setup/preprocess/transform/transform_merge_enhanced.py --split "$split" --preprocessed_data_path "$preprocessed_data_path"
done
