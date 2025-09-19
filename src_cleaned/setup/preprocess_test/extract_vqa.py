import os
import json
from tqdm import tqdm
import argparse

phase_to_number_dict = {
    'avoidance': '4',
    'action': '3',
    'judgement': '2',
    'recognition': '1',
    'prerecognition': '0',
}

def extract_vqa(vqa_path):
    '''
    Extracts VQA from the test directory

    Args:
        data_path (str): Path to the dataset directory containing VQA data.
    Returns:
        dict: A dictionary containing the extracted VQA data structured by video names and views.
    
    Example structure:
    {
        "20230707_17_CY23_T1": {
            "environment": [
                {
                    "question": "What is the weather condition?",
                    "a": "Sunny",
                    "b": "Rainy",
                    "c": "Cloudy",
                    "d": "Snowy",
                    "correct": "a"
                },
                ...
            ],
            "overhead_view": {
                "0": {
                    "start_time": xxx,
                    "end_time": xxx,
                    "conversations": [
                    {
                        "question": "What is the orientation of the pedestrian's body?",
                        "a": "Same direction as the vehicle",
                        "b": "Diagonally to the left, in the opposite direction to the vehicle",
                        "c": "Perpendicular to the vehicle and to the right",
                        "d": "Diagonally to the right, in the same direction as the vehicle",
                        "correct": "c"
                    },
                ],
                },
                ...
            }
        },
        "20230728_31_CN37_T1": {
            ...
        },
        ...
    }
    '''
    res_vqa_dict = {}
    with open(os.path.join(vqa_path, 'WTS_VQA_PUBLIC_TEST.json'), 'r') as f:
        video_vqa_dict = json.load(f)
    
    for scene_info in tqdm(video_vqa_dict):
        video_name = scene_info['videos'][0]
        video = os.path.splitext(video_name)[0]
        if "event_phase" not in scene_info:
            # VQA env
            if video not in res_vqa_dict:
                res_vqa_dict[video] = {}
            if 'environment' not in res_vqa_dict[video]:
                res_vqa_dict[video]['environment'] = {}
            res_vqa_dict[video]['environment'] = scene_info['conversations']
        else:
            # VQA 
            view = 'vehicle_view' if 'vehicle_view' in video else 'overhead_view'
            for phase_info in scene_info['event_phase']:
                phase_number = phase_to_number_dict[phase_info["labels"][0]]
                st_time = float(phase_info['start_time'])
                ed_time = float(phase_info['end_time'])
                conversations = phase_info['conversations']
                
                if video not in res_vqa_dict:
                    res_vqa_dict[video] = {}
                if view not in res_vqa_dict[video]:
                    res_vqa_dict[video][view] = {}
                if phase_number not in res_vqa_dict[video][view]:
                    res_vqa_dict[video][view][phase_number] = {}
                    res_vqa_dict[video][view][phase_number].update({
                        'start_time': float(st_time),
                        'end_time': float(ed_time),
                        'conversations': conversations
                    })
                else:
                    res_vqa_dict[video][view][phase_number]['conversations'].extend(conversations)

    return res_vqa_dict
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test", help='Data directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test", help='Directory for saving json file')

    args = parser.parse_args()
    if not os.path.exists(args.data_path + '/SubTask2-VQA'):
        print(f"Data path {args.data_path} does not exist. Please check the path.")
        exit(1)
    os.makedirs(args.output_path, exist_ok=True)

    args.data_path = os.path.join(args.data_path, 'SubTask2-VQA')
    res_vqa_dict = extract_vqa(args.data_path)

    with open(os.path.join(args.output_path, f'test_vqa.json'), 'w') as f:
        f.write(json.dumps(res_vqa_dict, indent=4))
    
    print(f"Extracted test VQA saved to {os.path.join(args.output_path, 'test_vqa.json')}")