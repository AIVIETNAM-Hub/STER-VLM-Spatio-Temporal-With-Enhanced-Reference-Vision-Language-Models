#!/bin/bash

data_path="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test"
if [ -z "$1" ]; then
    echo "No data path provided. Using default specified in file."
else
    data_path="$1"
    echo "Using provided data path: $data_path"
fi

preprocessed_data_path="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test"
if [ -z "$2" ]; then
    echo "No preprocessed data path provided. Using default specified in file."
else
    preprocessed_data_path="$2"
    echo "Using provided preprocessed data path: $preprocessed_data_path"
fi

echo "Processing test data"

echo "Extracting VQA data for test data"
python src_cleaned/setup/preprocess_test/extract_vqa.py --data_path "$data_path" --output_path "$preprocessed_data_path"

echo "Extracting annotated bounding boxes for test data"
python src_cleaned/setup/preprocess_test/extract_bbox_annotated.py --data_path "$data_path" --output_path "$preprocessed_data_path"

echo "Extracting generated bounding boxes for test data"
python src_cleaned/setup/preprocess_test/extract_bbox_generated.py --data_path "$data_path" --output_path "$preprocessed_data_path"

echo "Drawing bounding boxes for test data"
python src_cleaned/setup/preprocess_test/draw_bbox.py --data_path "$data_path" --bbox_path "$preprocessed_data_path" --output_path "$preprocessed_data_path"