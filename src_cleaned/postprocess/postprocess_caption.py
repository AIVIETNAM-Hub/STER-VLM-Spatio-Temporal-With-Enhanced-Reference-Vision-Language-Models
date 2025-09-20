import os
import json
from tqdm import tqdm
import argparse

def transform_submission_format(result_caption_path):
    '''
    Args:
        result_caption_path: Path to the JSON file containing the inference results for captions.

    Returns:
        res_caption: A dictionary containing the transformed caption data in the required submission format.

    Example of res_caption format:
    {
        "20230707_12_SN17_T1": [
            {
                "labels": [
                    "4"
                ],
                "caption_pedestrian": "...",
                "caption_vehicle": "...",
            },
            {
                "labels": [
                    "3"
                ],
                "caption_pedestrian": "...",
                "caption_vehicle": "...",
            },
            ...
        ],
        "20230707_12_SN17_T2": [
            ...
        ],
    }
    '''

    scene_cap = {}
    with open(result_caption_path, 'r') as f:
        inference_caption_json = json.load(f)['results']

    for sample in tqdm(inference_caption_json):
        caption = sample['prediction']

        scenario = sample['sample']['scenario']
        labels = str(sample['sample']['phase'])

        if '\nassistant\n' in caption:
            caption = caption.split('\nassistant\n')[1].strip()
        elif 'assistant\n' in caption:
            caption = caption.split('assistant\n')[1].strip()
        else:
            caption = caption.strip()

        if scenario not in scene_cap:
            scene_cap[scenario] = {}
        if labels not in scene_cap[scenario]:
            scene_cap[scenario][labels] = {}
        if 'caption_pedestrian' not in scene_cap[scenario][labels]:
            scene_cap[scenario][labels]['caption_pedestrian'] = ""
        if 'caption_vehicle' not in scene_cap[scenario][labels]:
            scene_cap[scenario][labels]['caption_vehicle'] = ""

        if caption.startswith("The pedestrian"):
            scene_cap[scenario][labels]["caption_pedestrian"] = caption
        elif caption.startswith("The vehicle"):
            scene_cap[scenario][labels]["caption_vehicle"] = caption
        else:
            if caption.startswith("He") or caption.startswith("She") or caption.startswith("A pedestrian") or caption.startswith("A person") or caption.startswith("A male") or caption.startswith("A female") or caption.startswith("A young") or caption.startswith("A middle-aged"):
                scene_cap[scenario][labels]["caption_pedestrian"] = caption
            else:
                scene_cap[scenario][labels]["caption_vehicle"] = caption

    res_caption = {}
    for scenario in scene_cap:
        for label in scene_cap[scenario]:
            if scenario not in res_caption:
                res_caption[scenario] = []
            res_caption[scenario].append({
                "labels": [label],
                "caption_pedestrian": scene_cap[scenario][label].get("caption_pedestrian", ""),
                "caption_vehicle": scene_cap[scenario][label].get("caption_vehicle", "")
            })
    
    # sort the captions by labels
    for scenario in res_caption:
        res_caption[scenario].sort(key=lambda x: x['labels'][0], reverse=True)

    return res_caption

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_caption_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/experiments/Trainvalonepunchman/caption_total_test.json', help='Path to inferred caption')
    parser.add_argument('--submission_folder_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/submission/vLastHope', help='Folder path to save Caption submission')

    args = parser.parse_args()

    res_caption = transform_submission_format(args.result_caption_path)

    os.makedirs(args.submission_folder_path, exist_ok=True)
    res_caption_json_path = os.path.join(args.submission_folder_path, f'caption.json')
    with open(res_caption_json_path, 'w+') as f:
        json.dump(res_caption, f, indent=4)
        print(f"Transformed caption data saved to {res_caption_json_path}")