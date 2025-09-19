import os
import json
from tqdm import tqdm
import argparse

import sys
sys.path.insert(1, 'src_cleaned/utils')
from choose_image import choose_best_image
from choose_prompt import get_prompt_merge, get_prompt_merge_with_reference
from choose_prompt import determine_reference_for_caption

def transform_wts_caption(args):
    """
    [
        {
            "image": ["path/to/image", "path/to/image2",...],
            "conversations": [
                {
                    "from": "user",
                    "value": "Picture 1: <image>\n<image>\nYou are a expert in analyzing fine-grained video understanding in traffic scenarios.\nCONTEXT:..."
                },
                {
                    "from": "gpt",
                    "value": "{rewrite prompt}"
                }
            ]
        },
        ...
    ]
    """

    res_caption_dict = []
    caption_spatial_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_spatial_pair.json')
    caption_temporal_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_temporal_pair.json')
    reference_for_caption_path = os.path.join(args.preprocessed_data_path, f'total_{args.split}_caption_transformed.json')
    with open(caption_spatial_path, 'r') as f:
        caption_spatial_json = json.load(f)
    with open(caption_temporal_path, 'r') as f:
        caption_temporal_json = json.load(f)
    with open(reference_for_caption_path, 'r') as f:
        reference_for_caption_json = json.load(f)

    # sort caption by scenario name
    caption_spatial_json = sorted(caption_spatial_json, key=lambda x: list(x.keys())[0])
    caption_temporal_json = sorted(caption_temporal_json, key=lambda x: list(x.keys())[0])

    bbox_folder = os.path.join(args.preprocessed_data_path, "drawn_bbox_generated", args.split)
    bbox_path = os.path.join(args.preprocessed_data_path, f"wts_{args.split}_generated_bbox.json")
    caption_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_caption.json')
    with open(caption_path, 'r') as f:
        caption_gt_json = json.load(f)

    for i in tqdm(range(len(caption_spatial_json)), desc="Processing WTS scenarios"):
        scenario = list(caption_spatial_json[i].keys())[0]  # Get the scenario name
        spatial_caption_ped = caption_spatial_json[i][scenario]['caption_ped']
        spatial_caption_veh = caption_spatial_json[i][scenario]['caption_veh']
        try:
            temporal_captions = caption_temporal_json[i][scenario]
        except IndexError:
            print(f"Warning: No temporal captions found for scenario {scenario}. Skipping.")
            continue
        
        # Get the best images for each phase
        for type in ['pedestrian', 'vehicle']:
            for phase_info in temporal_captions:
                phase_number = phase_info['labels']
                if type == 'pedestrian':
                    phase_caption = phase_info.get(f'caption_ped', '')
                else:
                    phase_caption = phase_info.get(f'caption_veh', '')

                if 'normal' in scenario:
                    scene_name = os.path.join(bbox_folder, 'normal_trimmed', scenario)
                else:
                    scene_name = os.path.join(bbox_folder, scenario)

                image_list = [os.path.join(scene_name, f) for f in os.listdir(scene_name) if (f.endswith('.jpg')) and (f"phase{phase_number}" in os.path.basename(f))]
                best_images = choose_best_image(scenario, image_list, int(phase_number), bbox_path)
                best_images = sorted(best_images, key=lambda x: int(x.split('_')[-1].split('.')[0])) # sort by frame number

                reference_for_caption = determine_reference_for_caption(reference_for_caption_json, scenario, best_images, 'person' if type == 'pedestrian' else 'vehicle')
                conversations = []
                if reference_for_caption:
                    conversations.append({
                        "from": "user",
                        "value": get_prompt_merge_with_reference(
                            len(best_images),
                            spatial_caption_ped if type == 'pedestrian' else spatial_caption_veh,
                            phase_caption,
                            phase_number,
                            type,
                            reference_for_caption
                        )
                    })
                else:
                    conversations.append({
                        "from": "user",
                        "value": get_prompt_merge(
                            len(best_images),
                            spatial_caption_ped if type == 'pedestrian' else spatial_caption_veh,
                            phase_caption,
                            phase_number,
                            type,
                        )
                    })
                conversations.append({
                    "from": "gpt",
                    "value": caption_gt_json[scenario][str(phase_number)][('caption_pedestrian' if type == 'pedestrian' else 'caption_vehicle')]
                })
                res_caption_dict.append({
                    "ped_or_veh": "ped" if type == 'pedestrian' else "veh",
                    "images": best_images,
                    "conversations": conversations  
                })
    return res_caption_dict


def transform_bdd_caption(args):
    """
    [
        {
            "images": ["path/to/image", "path/to/image2",...],
            "conversations": [
                {
                    "from": "user",
                    "value": "Picture 1: <image>\n<image>\nYou are a expert in analyzing fine-grained video understanding in traffic scenarios.\nCONTEXT:..."
                },
                {
                    "from": "gpt",
                    "value": "{rewrite prompt}"
                }
            ]
        },
        ...
    ]
    """
    res_caption_dict = []
    caption_spatial_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_spatial_pair.json')
    caption_temporal_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_temporal_pair.json')
    reference_for_caption_path = os.path.join(args.preprocessed_data_path, f'total_{args.split}_caption_transformed.json')
    with open(caption_spatial_path, 'r') as f:
        caption_spatial_json = json.load(f)
    with open(caption_temporal_path, 'r') as f:
        caption_temporal_json = json.load(f)
    with open(reference_for_caption_path, 'r') as f:
        reference_for_caption_json = json.load(f)

    # sort caption by scenario name
    caption_spatial_json = sorted(caption_spatial_json, key=lambda x: list(x.keys())[0])
    caption_temporal_json = sorted(caption_temporal_json, key=lambda x: list(x.keys())[0])

    bbox_folder = os.path.join(args.preprocessed_data_path, "drawn_bbox_generated/external", args.split)
    bbox_path = os.path.join(args.preprocessed_data_path, f"bdd_{args.split}_generated_bbox.json")
    caption_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_caption.json')
    with open(caption_path, 'r') as f:
        caption_gt_json = json.load(f)

    for i in tqdm(range(len(caption_spatial_json)), desc="Processing BDD scenarios"):
        scenario_without_ext = list(caption_spatial_json[i].keys())[0]  # Get the scenario name
        scenario = scenario_without_ext + '.mp4'  
        spatial_caption_ped = caption_spatial_json[i][scenario_without_ext]['caption_ped']
        spatial_caption_veh = caption_spatial_json[i][scenario_without_ext]['caption_veh']
        try:
            temporal_captions = caption_temporal_json[i][scenario]
        except IndexError:
            print(f"Warning: No temporal captions found for scenario {scenario}. Skipping.")
            continue
        
        # Get the best images for each phase
        for type in ['pedestrian', 'vehicle']:
            for phase_info in temporal_captions:
                phase_number = phase_info['labels']
                if type == 'pedestrian':
                    phase_caption = phase_info.get(f'caption_ped', '')
                else:
                    phase_caption = phase_info.get(f'caption_veh', '')

                scene_name = os.path.join(bbox_folder, scenario_without_ext)

                image_list = [os.path.join(scene_name, f) for f in os.listdir(scene_name) if (f.endswith('.jpg')) and (f"phase{phase_number}" in os.path.basename(f))]
                best_images = choose_best_image(scenario, image_list, int(phase_number), bbox_path)
                best_images = sorted(best_images, key=lambda x: int(x.split('_')[-1].split('.')[0])) # sort by frame number

                reference_for_caption = determine_reference_for_caption(reference_for_caption_json, scenario_without_ext, best_images, 'person' if type == 'pedestrian' else 'vehicle')
                conversations = []
                if reference_for_caption:
                    conversations.append({
                        "from": "user",
                        "value": get_prompt_merge_with_reference(
                            len(best_images),
                            spatial_caption_ped if type == 'pedestrian' else spatial_caption_veh,
                            phase_caption,
                            phase_number,
                            type,
                            reference_for_caption
                        )
                    })
                else:
                    conversations.append({
                        "from": "user",
                        "value": get_prompt_merge(
                            len(best_images),
                            spatial_caption_ped if type == 'pedestrian' else spatial_caption_veh,
                            phase_caption,
                            phase_number,
                            type,
                        )
                    })
                conversations.append({
                    "from": "gpt",
                    "value": caption_gt_json[scenario][str(phase_number)][('caption_pedestrian' if type == 'pedestrian' else 'caption_vehicle')]
                })
                res_caption_dict.append({
                    "ped_or_veh": "ped" if type == 'pedestrian' else "veh",
                    "images": best_images,
                    "conversations": conversations  
                })
    return res_caption_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train, val')
    parser.add_argument('--preprocessed_data_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned', help='Path to preprocessed data')

    args = parser.parse_args()

    wts_merge_caption = transform_wts_caption(args)
    bdd_merge_caption = transform_bdd_caption(args)
    merged_caption = wts_merge_caption + bdd_merge_caption
    
    for sample in merged_caption:
        #delete scenario, type, and phase
        if 'scenario' in sample:
            del sample['scenario']
        if 'type' in sample:
            del sample['type']
        if 'phase' in sample:
            del sample['phase']

    os.makedirs(os.path.join(args.preprocessed_data_path, 'train_data'), exist_ok=True)
    output_path = os.path.join(args.preprocessed_data_path, f'train_data/total_{args.split}_caption_enhanced.json')
    with open(output_path, 'w') as f:
        json.dump(merged_caption, f, indent=4)

    print(f'Merged captions saved to {output_path}')

    # Optionally, you can also print the number of entries in the merged caption
    print(f'Total entries in merged caption: {len(merged_caption)}')
    print(f'WTS entries: {len(wts_merge_caption)}, BDD entries: {len(bdd_merge_caption)}')

