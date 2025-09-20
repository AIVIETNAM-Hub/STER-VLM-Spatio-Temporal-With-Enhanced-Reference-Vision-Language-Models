import os
import json
import argparse

def extract_scene_name(image_path):
    return os.path.basename(os.path.dirname(image_path))

def main():
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)

    
    # All file paths
    base_input = "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset"
    base_output = "/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/data"
    args = parser.parse_args()
    
    INPUT_FILENAMES = {
        'train': "train_data/total_train_caption.json",
        'val': "train_data/total_val_caption.json",
        'test': "test/test_data/test_merge_caption.json"
    }
    
    
    OUTPUT_FILENAMES = {
        'train': "total_train_caption_transformed.json",
        'val': "total_val_caption_transformed.json",
        'test': "total_test_caption_transformed.json"
    }
    
    key = args.split
    input_file = os.path.join(base_input, INPUT_FILENAMES[key])
    output_file = os.path.join(base_output, OUTPUT_FILENAMES[key])
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    scenes_dict = {}
    
    for entry in data:
        images = entry.get("images", [])
        for image_path in images:
            scene_name = extract_scene_name(image_path)
            scenes_dict.setdefault(scene_name, [])
            if image_path not in scenes_dict[scene_name]:
                scenes_dict[scene_name].append(image_path)
    
    with open(output_file, "w") as f:
        json.dump(scenes_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Transformed data saved to {output_file}")

if __name__ == "__main__":
    main()
