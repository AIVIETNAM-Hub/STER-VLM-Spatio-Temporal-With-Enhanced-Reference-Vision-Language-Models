import os
import json
from tqdm import tqdm
import argparse

import sys
sys.path.insert(1, 'src_cleaned/utils')
from choose_image import choose_best_image, find_best_image_for_environment, find_best_image_for_appearance

def match_wts_spatial_captions_with_images(split, preprocessed_data_path):
    """
    Match fixed captions with images for appearance and environment.

    Args:
        args: Command line arguments containing paths and split information.

    Returns:
        res_dict: List of dictionaries containing video names, image paths, and captions.
    """
    res_dict = []

    # Get extracted captions
    with open(os.path.join(preprocessed_data_path, f'wts_{split}_fnf_processed.json'), 'r') as f:
        spatial_captions = json.load(f)
    
    """
    Format
    {
        [
            "20230707_8_SN46_T1": {
                'image': [2 cái image path]
                'caption_veh': "caption về vehicle",
                'caption_ped': "caption về pedestrian"
            },
        ]
    }
    """
    for video_name in tqdm(spatial_captions.keys()):
        if 'normal' in video_name:
            appearance_image_path = find_best_image_for_appearance(f"{preprocessed_data_path}/drawn_bbox_annotated/{split}/normal_trimmed/{video_name}", bbox_json_path=f"{preprocessed_data_path}/wts_{split}_annotated_bbox.json")
            environment_image_path = find_best_image_for_environment(f"{preprocessed_data_path}/drawn_bbox_annotated/{split}/normal_trimmed/{video_name}")
        else:
            appearance_image_path = find_best_image_for_appearance(f"{preprocessed_data_path}/drawn_bbox_annotated/{split}/{video_name}", bbox_json_path=f"{preprocessed_data_path}/wts_{split}_annotated_bbox.json")
            environment_image_path = find_best_image_for_environment(f"{preprocessed_data_path}/drawn_bbox_annotated/{split}/{video_name}")
        
        image_list = appearance_image_path + environment_image_path
        image_list = sorted(image_list, key=lambda x: int(x.split('_')[-2][-1]))  # Sắp xếp theo phase number
        image_list = [img for img in image_list if 'vehicle' in img] + [img for img in image_list if 'vehicle' not in img]  # Đưa vehicle view lên trước
        res_dict.append({
            video_name: { 
                'image': image_list,
                'caption_veh': spatial_captions[video_name][0]['vehicle_fixed'],
                'caption_ped': spatial_captions[video_name][0]['pedestrian_fixed']
            }
        })

    return res_dict

def match_bdd_spatial_captions_with_images(split, preprocessed_data_path):
    """
    Input:
        split: train, val
        preprocessed_data_path: path to the preprocessed data directory
    Output:
        res_dict: List of dictionaries, each dictionary contains video name, image paths, and captions
   
    """
    res_dict = []
    
    # Get extracted captions
    with open(os.path.join(preprocessed_data_path, f'bdd_{split}_fnf_processed.json'), 'r') as f:
        fixed_captions = json.load(f)
    
    """
    Format
    {
        [
            "20230707_8_SN46_T1": {
                'image': [2 cái image path]
                'caption_veh': "caption về vehicle",
                'caption_ped': "caption về pedestrian"
            },
        ]
    }
    """
    for video_name in tqdm(fixed_captions.keys()):
        video_name = video_name.split('.mp4')[0]
        appearance_image_path = find_best_image_for_appearance(f"{preprocessed_data_path}/drawn_bbox_annotated/external/{split}/{video_name}", bbox_json_path=f"{preprocessed_data_path}/bdd_{split}_annotated_bbox.json")
        environment_image_path = find_best_image_for_environment(f"{preprocessed_data_path}/drawn_bbox_annotated/external/{split}/{video_name}")
        
        image_list = appearance_image_path + environment_image_path
        image_list = sorted(image_list, key=lambda x: int(x.split('_')[-2][-1]))  # Sắp xếp theo phase number
        res_dict.append({
            video_name: { 
                'image': image_list,
                'caption_veh': fixed_captions[f"{video_name}.mp4"][0]['vehicle_fixed'],
                'caption_ped': fixed_captions[f"{video_name}.mp4"][0]['pedestrian_fixed']
            }
        })

    return res_dict

# Temporal captions

def match_wts_one_scenario_caption(scenario_folder, caption_arr, bbox_json_path=None):
    """
    Match WTS one scenario caption with images
    
    Args:
        scenario_folder (str): Path to the folder containing the scenario
        caption_arr (list): List of caption dictionaries for different phases
        bbox_json_path (str, optional): Path to the bounding box file. Defaults to None
    Returns:
        scenario_caption_phase_arr (list): List of dictionaries containing matched captions and images for each phase
    """

    # Check if the scenario exists in the caption dictionary
    scenario_caption_phase_arr = []
    for phase in range(5):
        phase_obj = {}
        phase_obj['labels'] = str(phase)
        img_list = [f for f in os.listdir(scenario_folder) if (f.endswith('.jpg')) and (f"phase{phase}" in os.path.basename(f))]
        img_list = sorted(img_list, key=lambda x: str(x.split('_')[:-3]))  # Sort by phase number
        scenario = scenario_folder.split('/')[-1]
        best_images = choose_best_image(scenario, img_list, phase, bbox_json_path)
        best_images = sorted(best_images, key=lambda x: int(x.split('_')[-1].split('.jpg')[0])) # sort by frame number
        phase_obj['image'] = best_images
        for caption_obj in caption_arr:
            if caption_obj['labels'] == str(phase) and caption_obj['view'] == 'overhead_view':
                phase_obj['caption_ped'] = caption_obj['pedestrian_not_fixed']
                phase_obj['caption_veh'] = caption_obj['vehicle_not_fixed']
                break
        scenario_caption_phase_arr.append(phase_obj)
    return scenario_caption_phase_arr

def match_bdd_one_scenario_caption(scenario_folder, caption_arr, bbox_json_path=None):
    """
    Match BDD one scenario caption with images.

    Args:
        scenario_folder (str): Path to the folder containing the scenario
        caption_arr (list): List of caption dictionaries for different phases
        bbox_json_path (str, optional): Path to the bounding box file. Defaults to None
    Returns:
        scenario_caption_phase_arr (list): List of dictionaries containing matched captions and images for each phase
    """

    # Check if the scenario exists in the caption dictionary
    scenario_caption_phase_arr = []
    for phase in range(5):
        phase_obj = {}
        phase_obj['labels'] = str(phase)
        img_list = [f for f in os.listdir(scenario_folder) if (f.endswith('.jpg')) and (f"phase{phase}" in os.path.basename(f))]
        img_list = sorted(img_list, key=lambda x: str(x.split('_')[:-3]))  # Sort by phase number
        scenario = scenario_folder.split('/')[-1] + ".mp4"
        best_images = choose_best_image(scenario, img_list, phase, bbox_json_path)
        best_images = sorted(best_images, key=lambda x: int(x.split('_')[-1].split('.')[0])) # sort by frame number
        phase_obj['image'] = best_images
        for caption_obj in caption_arr:
            if caption_obj['labels'] == str(phase):
                phase_obj['caption_ped'] = caption_obj['pedestrian_not_fixed']
                phase_obj['caption_veh'] = caption_obj['vehicle_not_fixed']
                break
        scenario_caption_phase_arr.append(phase_obj)
    return scenario_caption_phase_arr

def match_wts_temporal_captions(split, preprocessed_data_path):
    """
    Match temporal captions with images for WTS dataset

    Args:
        split (str): The dataset split (train, val)
        preprocessed_data_path (str): Path to the preprocessed data directory
    Returns:
        scenario_caption_arr (list): List of dictionaries containing scenario names and matched captions
    """

    scenario_caption_arr = []
    drawn_bbox_path = f"{preprocessed_data_path}/drawn_bbox_generated"
    bbox_file = f"{preprocessed_data_path}/wts_{split}_generated_bbox.json"
    caption_file = f"{preprocessed_data_path}/wts_{split}_fnf_processed.json"
    scenario_caption_arr = []

    with open(caption_file, 'r') as f:
        caption_data = json.load(f)

    # Get all subfolders in the scenario folder outside NORMAL_TRIMMED for the current split 
    scenario_names = [f.path for f in os.scandir(os.path.join(drawn_bbox_path, split)) if f.is_dir()]
    for scenario_name in tqdm(scenario_names):
        scenario_name = os.path.basename(scenario_name)
        if scenario_name == "normal_trimmed":
            continue
        if scenario_name in caption_data:
            caption_arr = caption_data[scenario_name]
            matched_captions = match_wts_one_scenario_caption(os.path.join(drawn_bbox_path, split, scenario_name), caption_arr, bbox_file)
            scenario_caption_arr.append({scenario_name: matched_captions})

    # Get all subfolders in the scenario folder inside NORMAL_TRIMMED for the current split  
    scenario_names = [f.path for f in os.scandir(os.path.join(drawn_bbox_path, split, "normal_trimmed")) if f.is_dir()]
    for scenario_name in tqdm(scenario_names):
        scenario_name = os.path.basename(scenario_name)
        if scenario_name in caption_data:
            caption_arr = caption_data[scenario_name]
            matched_captions = match_wts_one_scenario_caption(os.path.join(drawn_bbox_path, split, "normal_trimmed", scenario_name), caption_arr, bbox_file)
            scenario_caption_arr.append({scenario_name: matched_captions})
    return scenario_caption_arr

def match_bdd_temporal_captions(split, preprocessed_data_path):
    """
    Match temporal captions with images for BDD dataset

    Args:
        split (str): The dataset split (train, val)
        preprocessed_data_path (str): Path to the preprocessed data directory
    Returns:
        scenario_caption_arr (list): List of dictionaries containing scenario names and matched captions
    """

    drawn_bbox_path = f"{preprocessed_data_path}/drawn_bbox_generated/external"
    bbox_file = f"{preprocessed_data_path}/bdd_{split}_generated_bbox.json"
    caption_file = f"{preprocessed_data_path}/bdd_{split}_fnf_processed.json"
    scenario_caption_arr = []

    with open(caption_file, 'r') as f:
        caption_data = json.load(f)

    scenario_names = [f.path for f in os.scandir(os.path.join(drawn_bbox_path, split)) if f.is_dir()]
    for scenario_name in tqdm(scenario_names):
        scenario_name = os.path.basename(scenario_name)
        scenario_name_ext = os.path.basename(scenario_name + '.mp4')
        if scenario_name_ext in caption_data:
            caption_arr = caption_data[scenario_name_ext]
            matched_captions = match_bdd_one_scenario_caption(os.path.join(drawn_bbox_path, split, scenario_name), caption_arr, bbox_file)
            scenario_caption_arr.append({scenario_name_ext: matched_captions})
    return scenario_caption_arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--preprocessed_data_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned', help='Path to preprocessed data')

    args = parser.parse_args()
    
    wts_matched_spatial_caption_dict = match_wts_spatial_captions_with_images(args.split, args.preprocessed_data_path)
    wts_matched_spatial_caption_json_path = os.path.join(args.preprocessed_data_path, f"wts_{args.split}_spatial_pair.json")
    with open(wts_matched_spatial_caption_json_path, 'w') as f:
        json.dump(wts_matched_spatial_caption_dict, f, indent=4)
    print(f"Matched spatial captions with images saved to {wts_matched_spatial_caption_json_path}")

    wts_matched_temporal_caption_dict = match_wts_temporal_captions(args.split, args.preprocessed_data_path)
    wts_matched_temporal_caption_json_path = os.path.join(args.preprocessed_data_path, f"wts_{args.split}_temporal_pair.json")
    with open(wts_matched_temporal_caption_json_path, 'w') as f:
        json.dump(wts_matched_temporal_caption_dict, f, indent=4)
    print(f"Matched temporal captions with images saved to {wts_matched_temporal_caption_json_path}")


    bdd_matched_spatial_caption_dict = match_bdd_spatial_captions_with_images(args.split, args.preprocessed_data_path)
    bdd_json_path = os.path.join(args.preprocessed_data_path, f"bdd_{args.split}_spatial_pair.json")
    with open(bdd_json_path, 'w') as f:
        json.dump(bdd_matched_spatial_caption_dict, f, indent=4)
    print(f"Matched spatial captions with images saved to {bdd_json_path}")

    bdd_matched_temporal_caption_dict = match_bdd_temporal_captions(args.split, args.preprocessed_data_path)
    bdd_matched_temporal_caption_json_path = os.path.join(args.preprocessed_data_path, f"bdd_{args.split}_temporal_pair.json")
    with open(bdd_matched_temporal_caption_json_path, 'w') as f:
        json.dump(bdd_matched_temporal_caption_dict, f, indent=4)
    print(f"Matched temporal captions with images saved to {bdd_matched_temporal_caption_json_path}")
