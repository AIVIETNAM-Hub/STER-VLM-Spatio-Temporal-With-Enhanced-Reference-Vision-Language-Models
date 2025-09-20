#!/bin/bash

splits=("train" "val")

data_path="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip"
if [ -z "$1" ]; then
    echo "No data path provided. Using default specified in file."
else
    data_path="$1"
    echo "Using provided data path: $data_path"
fi

preprocessed_data_path="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned"
if [ -z "$2" ]; then
    echo "No preprocessed data path provided. Using default specified in file."
else
    preprocessed_data_path="$2"
    echo "Using provided preprocessed data path: $preprocessed_data_path"
fi

echo "Start extractinng caption, VQA, and bounding box annotations for train and validation splits."
for split in "${splits[@]}"; do
    echo "Processing split: $split"
    python src_cleaned/setup/preprocess/extract/extract_caption.py --split "$split" --data_path "$data_path" --output_path "$preprocessed_data_path"
    python src_cleaned/setup/preprocess/extract/extract_vqa.py --split "$split" --data_path "$data_path" --output_path "$preprocessed_data_path"
    python src_cleaned/setup/preprocess/extract/extract_bbox_annotated.py --split "$split" --data_path "$data_path" --output_path "$preprocessed_data_path"
    python src_cleaned/setup/preprocess/extract/extract_bbox_generated.py --split "$split" --data_path "$data_path" --output_path "$preprocessed_data_path"
done

echo "Start drawing bounding boxes for train and validation splits."
for split in "${splits[@]}"; do
    echo "Drawing bounding boxes for split: $split"
    python src_cleaned/setup/preprocess/draw/wts_draw.py --split "$split" --data_path "$data_path" --bbox_path "$preprocessed_data_path" --output_path "$preprocessed_data_path"
    python src_cleaned/setup/preprocess/draw/bdd_draw.py --split "$split" --data_path "$data_path" --bbox_path "$preprocessed_data_path" --output_path "$preprocessed_data_path"
done
