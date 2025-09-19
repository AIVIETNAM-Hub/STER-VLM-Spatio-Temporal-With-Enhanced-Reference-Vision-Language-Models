import os
import json
import argparse
from tqdm import tqdm

import sys
sys.path.insert(1, 'src_cleaned/utils')
from choose_prompt import get_prompt_merge, get_prompt_merge_with_reference
from choose_prompt import determine_reference_for_caption


def match_spatial_result(prediction_path):
    infer_res = {}
    inference_spatial_path = os.path.join(prediction_path, 'caption_spatial.json')
    with open(inference_spatial_path, 'r') as f:
        inference_spatial_json = json.load(f)['results']

    for sample in inference_spatial_json:
        prediction = sample['prediction']
        image_paths = sample['image_paths']
        scenario_name = image_paths[0].split('/')[-2] 

        if scenario_name not in infer_res:
            infer_res[scenario_name] = {
                'caption_1': '',
                'caption_2': '',
            }
        
        if infer_res[scenario_name]['caption_1'] == '':
            infer_res[scenario_name]['caption_1'] = prediction
        else:
            infer_res[scenario_name]['caption_2'] = prediction
    
    return infer_res

def match_temporal_result(prediction_path):
    infer_res = {}
    inference_temporal_path = os.path.join(prediction_path, 'caption_temporal.json')
    with open(inference_temporal_path, 'r') as f:
        inference_temporal_json = json.load(f)['results']

    for sample in inference_temporal_json:
        prediction = sample['prediction']
        image_paths = sample['image_paths']
        scenario_name = image_paths[0].split('/')[-2] 
        phase_number = image_paths[0].split('/')[-1].split('_')[-3][-1]
        if '0' > phase_number or phase_number > '4':
            phase_number = image_paths[0].split('/')[-1].split('_')[-4][-1]

        if scenario_name not in infer_res:
            infer_res[scenario_name] = {}
        if str(phase_number) not in infer_res[scenario_name]:
            infer_res[scenario_name][str(phase_number)] = {
                'caption_ped': {},
                'caption_veh': {},
                'image_paths': {}
            }

        infer_res[scenario_name][str(phase_number)]['image_paths'] = image_paths
        if prediction.startswith("The pedestrian"):
            infer_res[scenario_name][str(phase_number)]['caption_ped'] = prediction
        elif prediction.startswith("The vehicle"):
            infer_res[scenario_name][str(phase_number)]['caption_veh'] = prediction
        else:
            if prediction.startswith("He") or prediction.startswith("She") or prediction.startswith("A pedestrian") or prediction.startswith("A person") or prediction.startswith("A male") or prediction.startswith("A female") or prediction.startswith("A young") or prediction.startswith("A middle-aged"):
                infer_res[scenario_name][str(phase_number)]["caption_pedestrian"] = prediction
            else:
                infer_res[scenario_name][str(phase_number)]["caption_vehicle"] = prediction
    return infer_res

def transform_merge_format(transformed_inference_spatial_caption, transformed_inference_temporal_caption):
    """
    res_caption:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\nYou are ..."
                },
            }
        }
    ]
    """

    """
    transformed_inference_spatial_caption:
    {
        "20230707_12_SN17_T1": {
            "caption_1": "..."
            "caption_2": "...",
        },
        ...
    }

    transformed_inference_temporal_caption:
    {
        "20230707_12_SN17_T1": {
            "0": {
                "caption_ped": "...",
                "caption_veh": "...",
                "image_paths": ["20230707_12_SN17_T1/000000.jpg", "..."]
            },
            "1": {
                ...
            },
        },
        ...
    }
    """
    reference_for_caption_path = os.path.join(args.preprocessed_data_path, f'total_{args.split}_caption_transformed.json')
    with open(reference_for_caption_path, 'r') as f:
        reference_for_caption_json = json.load(f)

    res_caption = []
    for scenario in tqdm(transformed_inference_temporal_caption.keys(), desc="Processing scenarios"):
        if scenario == 'normal_trimmed':
            continue
        if scenario not in transformed_inference_spatial_caption:
            print(f"Scenario {scenario} not found in transformed_inference_temporal_caption, need to recheck")
            continue

        spatial_caption = transformed_inference_spatial_caption[scenario]
        for phase in range(5):
            if str(phase) not in transformed_inference_temporal_caption[scenario]:
                print(f"Phase {phase} not found in transformed_inference_temporal_caption for scenario {scenario}, need to recheck")
                continue

            temporal_caption = transformed_inference_temporal_caption[scenario][str(phase)]
            image_paths = temporal_caption['image_paths']
            
            scenario_without_ext = scenario.split('.mp4')[0]
            reference_for_caption = determine_reference_for_caption(reference_for_caption_json, scenario_without_ext, image_paths, 'person')
            conversation = []
            if 'caption_ped' in temporal_caption and temporal_caption['caption_ped']:
                if reference_for_caption:
                    prompt = get_prompt_merge_with_reference(len(image_paths), spatial_caption['caption_1'], temporal_caption['caption_ped'], phase, 'pedestrian', reference_for_caption)
                else:
                    prompt = get_prompt_merge(len(image_paths), spatial_caption['caption_1'], temporal_caption['caption_ped'], phase, 'pedestrian')
                conversation.append({
                    "from": "user",
                    "value": prompt
                })
                res_caption.append({
                    "scenario": scenario,
                    "phase": phase,
                    "images": image_paths,
                    "conversations": conversation
                })

            reference_for_caption = determine_reference_for_caption(reference_for_caption_json, scenario_without_ext, image_paths, 'vehicle')
            conversation = []
            if 'caption_veh' in temporal_caption and temporal_caption['caption_veh']:
                if reference_for_caption:
                    prompt = get_prompt_merge_with_reference(len(image_paths), spatial_caption['caption_1'], temporal_caption['caption_veh'], phase, 'vehicle', reference_for_caption)
                else:
                    prompt = get_prompt_merge(len(image_paths), spatial_caption['caption_1'], temporal_caption['caption_veh'], phase, 'vehicle')
                conversation.append({
                    "from": "user",
                    "value": prompt
                })
                res_caption.append({
                    "scenario": scenario,
                    "phase": phase,
                    "images": image_paths,
                    "conversations": conversation
                })
    
    return res_caption

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_data_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test', help='Path to preprocessed data')
    parser.add_argument('--prediction_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/prediction_path', help='Path to the folder containing inference results')

    args = parser.parse_args()

    transformed_inference_spatial_caption = match_spatial_result(args.prediction_path)
    transformed_inference_temporal_caption = match_temporal_result(args.prediction_path)

    transformed_merge_caption = transform_merge_format(transformed_inference_spatial_caption, transformed_inference_temporal_caption)

    os.makedirs(os.path.join(args.preprocessed_data_path, 'test_data'), exist_ok=True)
    output_path = os.path.join(args.preprocessed_data_path, 'test_data', 'test_merge_caption_enhanced.json')
    with open(output_path, 'w') as f:
        json.dump(transformed_merge_caption, f, indent=4)
    print(f"Transformed merge caption saved to {output_path}")
