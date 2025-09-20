import os
import json
from tqdm import tqdm
import argparse

def extract_wts_caption(scenario_path, caption_path, normal_trimmed=False):
    """
    Extracts captions from the given scenario and caption paths for the WTS dataset

    Args:
        scenario_path (str): Path to the directory containing scenario videos
        caption_path (str): Path to the directory containing caption annotations
        normal_trimmed (bool): Boolean indicating whether to use the normal trimmed version of the data
    Returns:
        res_caption_dict (dict): A dictionary containing the extracted captions in a structured format

    Example of the output format for res_caption_dict:
    {
        "20230707_17_CY23_T1": {
            "4": {
                "start_time": 0.00,
                "end_time": 1.00,
                "caption_pedestrian": "A",
                "caption_vehicle": "B",
            },
            "3": {
                ...
            }
            ...
        },
        "20230728_31_CN37_T1": {
            ...
        },
        ...
    }
    """
    res_caption_dict = {}
    if normal_trimmed:
        scenario_path = scenario_path + '/normal_trimmed'
        caption_path = caption_path + '/normal_trimmed'
    
    for scenario in tqdm(os.listdir(scenario_path)):
        if scenario == 'normal_trimmed': # Skip folder normal_trimmed (when os.listdir) 
            continue

        view = 'overhead_view'
        if os.path.exists(f'{caption_path}/{scenario}/overhead_view/{scenario}_caption.json'):
            view = 'overhead_view'
        else:
            view = 'vehicle_view'
        scenario_caption_path = f'{caption_path}/{scenario}/{view}/{scenario}_caption.json'

        try:
            scenario_dict = json.load(open(scenario_caption_path))['event_phase'] 
        except:
            # print(f"Error loading {scenario_caption_path}")
            continue
    
        for phase in scenario_dict:
            st_time = float(phase['start_time'])
            ed_time = float(phase['end_time'])
            
            if scenario not in res_caption_dict:
                res_caption_dict[scenario] = {}
            if phase['labels'][0] not in res_caption_dict[scenario]:
                res_caption_dict[scenario][phase['labels'][0]] = {}
            
            res_caption_dict[scenario][phase['labels'][0]].update({
                'start_time': st_time,
                'end_time': ed_time,
                'caption_pedestrian': phase['caption_pedestrian'],
                'caption_vehicle': phase['caption_vehicle'],
            })
    return res_caption_dict

def extract_bdd_caption(scenario_path, caption_path):
    """
    Extracts captions from the given scenario and caption paths for the BDD dataset

    Args:
        scenario_path (str): Path to the directory containing scenario videos
        caption_path (str): Path to the directory containing caption annotations
    Returns:
        res_caption_dict (dict): A dictionary containing the extracted captions in a structured format

    Example of the output format for res_caption_dict:
    {
        "video1004.mp4": {
            "0": {
                "start_time": 0.00,
                "end_time": 1.00,
                "caption_pedestrian": "A",
                "caption_vehicle": "B",
            },
            "1": {
                ...
            },
            ...
        },
        "video100.mp4": {
            ...
        },
        ...
    }
    """

    res_caption_dict = {}
    for scenario in tqdm(os.listdir(scenario_path)): 
        scenario_caption_path = caption_path + f"/{scenario.replace('.mp4', '')}_caption.json"

        try:
            scenario_caption_dict = json.load(open(scenario_caption_path))['event_phase'] 
        except:
            # print(f"Error loading {scenario_caption_path}")
            continue
        
        scenario_caption_dict.sort(key=lambda x: float(x['start_time']))
        for i, phase in enumerate(scenario_caption_dict):
            st_time = float(phase['start_time'])
            ed_time = float(phase['end_time'])
           
            if scenario not in res_caption_dict:
                res_caption_dict[scenario] = {}

            if str(i) not in res_caption_dict[scenario]:
                res_caption_dict[scenario][str(i)] = {}
            
            res_caption_dict[scenario][str(i)].update({
                'start_time': st_time,
                'end_time': ed_time,
                'caption_pedestrian': phase['caption_pedestrian'],
                'caption_vehicle': phase['caption_vehicle'],
            })
            
    return res_caption_dict
                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip", help='Data directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Directory for saving json file')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # WTS
    scenario_path = args.data_path + f'/videos/{args.split}'
    caption_path = args.data_path + f'/annotations/caption/{args.split}'
    wts_caption_dict = extract_wts_caption(scenario_path, caption_path, normal_trimmed=False)
    wts_caption_dict.update(extract_wts_caption(scenario_path, caption_path, normal_trimmed=True))
    with open(os.path.join(args.output_path, f'wts_{args.split}_caption.json'), 'w') as f:
        f.write(json.dumps(wts_caption_dict, indent=4))
    print(f"WTS {args.split} captions extracted and saved to {args.output_path}")

    # BDD
    args.data_path = args.data_path + "/external/BDD_PC_5K"
    scenario_path = args.data_path + f'/videos/{args.split}'
    caption_path = args.data_path + f'/annotations/caption/{args.split}'
    bdd_caption_dict = extract_bdd_caption(scenario_path, caption_path)
    with open(os.path.join(args.output_path, f'bdd_{args.split}_caption.json'), 'w') as f:
        f.write(json.dumps(bdd_caption_dict, indent=4))
    print(f"BDD {args.split} captions extracted and saved to {args.output_path}")