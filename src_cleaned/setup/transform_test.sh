#!/bin/bash

data_path="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test"
if [ -z "$1" ]; then
    echo "No prediction path provided. Using default specified in file."
else
    data_path="$1"
    echo "Using provided prediction path: $data_path"
fi

preprocessed_data_path="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test"
if [ -z "$2" ]; then
    echo "No prediction path provided. Using default specified in file."
else
    preprocessed_data_path="$2"
    echo "Using provided prediction path: $preprocessed_data_path"
fi

prediction_path="/home/s48gb/Desktop/GenAI4E/aicity_02/test_res"
if [ -z "$3" ]; then
    echo "No prediction path provided. Using default specified in file."
else
    prediction_path="$3"
    echo "Using provided prediction path: $prediction_path"
fi

echo "Transforming component data for Qwen-2.5-VL for test"
python src_cleaned/setup/preprocess_test/transform_component.py --data_path "$data_path" --preprocessed_data_path "$preprocessed_data_path"
