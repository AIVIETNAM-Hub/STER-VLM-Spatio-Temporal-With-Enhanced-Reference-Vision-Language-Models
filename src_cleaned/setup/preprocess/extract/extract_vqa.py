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

def extract_wts_phase_vqa(scenario_path, vqa_path, normal_trimmed=False):
    """
    Extract VQA of each phases from vqa files for WTS dataset
    
    Args:
        scenario_path (str): Path to the directory containing scenario files
        vqa_path (str): Path to the directory containing VQA annotations
        normal_trimmed (bool): Boolean indicating whether to use the normal trimmed version of the data
    Returns:
        res_vqa_dict (dict): A dictionary containing the extracted VQA of each phases in a structured format
    
    Example of the output format for res_vqa_dict:
    {
        "20230707_17_CY23_T1": {
            "20230707_17_CY23_T1_Camera2_2": {
                "4": {
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
                },
                "3": {
                    ...
                }
                ...
            ...
            "20230707_17_CY23_T1_Camera2_3": {
                ...
            },
        },
        "20230728_31_CN37_T1": {
            ...
        },
        ...
    }
    """
    res_vqa_dict = {}
    if normal_trimmed: 
        scenario_path = scenario_path + '/normal_trimmed'
        vqa_path = vqa_path + '/normal_trimmed'
    
    for scenario in tqdm(os.listdir(scenario_path)):
        if scenario == 'normal_trimmed': # Folder normal_trimmed 
            continue

        for view in ['overhead_view', 'vehicle_view']:
            scenario_vqa_path = f'{vqa_path}/{scenario}/{view}/{scenario}.json'

            # vehicle_view may not exist
            try:
                scenario_vqa_dict = json.load(open(scenario_vqa_path))[0]['event_phase']
            except:
                # print(f"Error loading {scenario_vqa_path}")
                continue
            
            for i, phase in enumerate(scenario_vqa_dict):
                phase_number = phase_to_number_dict[phase["labels"][0]]
                st_time = float(phase['start_time'])
                ed_time = float(phase['end_time'])
                conversations = scenario_vqa_dict[i]['conversations']
                
                if scenario not in res_vqa_dict:
                    res_vqa_dict[scenario] = {}
                if view not in res_vqa_dict[scenario]:
                    res_vqa_dict[scenario][view] = {}
                if phase_number not in res_vqa_dict[scenario][view]:
                    res_vqa_dict[scenario][view][phase_number] = {}
                
                res_vqa_dict[scenario][view][phase_number].update({
                    'start_time': st_time,
                    'end_time': ed_time,
                    'conversations': conversations
                })
    return res_vqa_dict

def extract_wts_environment_vqa(scenario_path, vqa_path, normal_trimmed=False):
    """
    Extract VQA environment from vqa files for WTS dataset

    Args:
        scenario_path (str): Path to the directory containing scenario files
        vqa_path (str): Path to the directory containing VQA annotations
        normal_trimmed (bool): Boolean indicating whether to use the normal trimmed version of the data
    Returns:
        res_env_vqa_dict (dict): A dictionary containing the extracted VQA environment in a structured format
    """
    res_env_vqa_dict = {}
    if normal_trimmed:
        scenario_path = scenario_path + '/normal_trimmed'
        vqa_path = vqa_path + '/normal_trimmed'

    for scenario in tqdm(os.listdir(scenario_path)):
        if scenario == 'normal_trimmed': # Folder normal_trimmed 
            continue
        
        scenario_vqa_path = f'{vqa_path}/{scenario}/environment/{scenario}.json'
        try:
            scenario_vqa_dict = json.load(open(scenario_vqa_path))[0]
        except:
            # print(f"Error loading {scenario_vqa_path}")
            continue

        if scenario not in res_env_vqa_dict:
            res_env_vqa_dict[scenario] = {}

        res_env_vqa_dict[scenario].update({
            'environment': scenario_vqa_dict['environment']
        })
    return res_env_vqa_dict

def extract_bdd_phase_vqa(scenario_path, vqa_path):
    """
    Extract VQA of each phases from vqa files for BDD dataset

    Args:
        scenario_path (str): Path to the directory containing scenario files
        vqa_path (str): Path to the directory containing VQA annotations
    Returns:
        res_vqa_dict (dict): A dictionary containing the extracted VQA of each phases in a structured format

    Example of the output format for res_vqa_dict:
    {
        "20230707_17_CY23_T1": {
            "20230707_17_CY23_T1_Camera2_2": {
                "4": {
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
                        ...
                    ]
                },
                "3": {
                    ...
                }
                ...
            },
            "20230707_17_CY23_T1_Camera2_3": {
                ...
            },
        },
        "20230728_31_CN37_T1": {
            ...
        },
        ...
    }
    """
    res_vqa_dict = {}
    if not os.path.exists(scenario_path):
        # print(f"scenario path {scenario_path} does not exist.")
        return
    for scenario in tqdm(os.listdir(scenario_path)):
        scenario_name = os.path.splitext(scenario)[0]
        view = 'overhead_view'
        scenario_vqa_path = f'{vqa_path}/{scenario_name}/{view}/{scenario_name}.json'
        try:
            scenario_vqa_dict = json.load(open(scenario_vqa_path))[0]['event_phase']
        except:
            print(scenario_name, "   ", view)
        
        for i, phase in enumerate(scenario_vqa_dict):
            phase_number = phase_to_number_dict[phase["labels"][0]]
            st_time = float(phase['start_time'])
            ed_time = float(phase['end_time'])
            conversations = scenario_vqa_dict[i]['conversations']
            
            if scenario_name not in res_vqa_dict:
                res_vqa_dict[scenario_name] = {}
            if view not in res_vqa_dict[scenario_name]:
                res_vqa_dict[scenario_name][view] = {}
            if phase_number not in res_vqa_dict[scenario_name][view]:
                res_vqa_dict[scenario_name][view][phase_number] = {}
            
            res_vqa_dict[scenario_name][view][phase_number].update({
                'start_time': st_time,
                'end_time': ed_time,
                'conversations': conversations
            })
    return res_vqa_dict

def extract_bdd_environment_vqa(scenario_path, vqa_path):
    """
    Extract VQA environment from vqa files for BDD dataset

    Args:
        scenario_path (str): Path to the directory containing scenario files
        vqa_path (str): Path to the directory containing VQA annotations
    Returns:
        res_env_vqa_dict (dict): A dictionary containing the extracted VQA environment in a structured format
    """
    res_env_vqa_dict = {}
    if not os.path.exists(scenario_path):
        # print(f"scenario path {scenario_path} does not exist.")
        return
    
    for scenario in tqdm(os.listdir(scenario_path)):
        scenario_name = os.path.splitext(scenario)[0]
        view = 'environment'
        scenario_vqa_path = f'{vqa_path}/{scenario_name}/{view}/{scenario_name}.json'
        scenario_vqa_dict = json.load(open(scenario_vqa_path))[0]

        if scenario_name not in res_env_vqa_dict:
            res_env_vqa_dict[scenario_name] = {}

        res_env_vqa_dict[scenario_name].update({
            'environment': scenario_vqa_dict['environment']
        })
    return res_env_vqa_dict
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip", help='Data directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Directory for saving json file')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # WTS
    scenario_path = args.data_path + f'/videos/{args.split}'
    vqa_path = args.data_path + f'/annotations/vqa/{args.split}'
    wts_phase_vqa_dict = extract_wts_phase_vqa(scenario_path, vqa_path)
    wts_phase_vqa_dict.update(extract_wts_phase_vqa(scenario_path, vqa_path, normal_trimmed=True))
    wts_env_vqa_dict = extract_wts_environment_vqa(scenario_path, vqa_path)
    wts_env_vqa_dict.update(extract_wts_environment_vqa(scenario_path, vqa_path, normal_trimmed=True))
    wts_vqa_json_path = os.path.join(args.output_path, f'wts_{args.split}_phase_vqa.json')
    wts_env_vqa_json_path = os.path.join(args.output_path, f'wts_{args.split}_env_vqa.json')

    with open(wts_vqa_json_path, 'w') as f:
        f.write(json.dumps(wts_phase_vqa_dict, indent=4))
    with open(wts_env_vqa_json_path, 'w') as f:
        f.write(json.dumps(wts_env_vqa_dict, indent=4))
    print(f"Extracted VQA data saved to {wts_vqa_json_path} and {wts_env_vqa_json_path}")

    # BDD
    args.data_path += "/external/BDD_PC_5K"
    scenario_path = args.data_path + f'/videos/{args.split}'
    vqa_path = args.data_path + f'/annotations/vqa/{args.split}'
    bdd_phase_vqa_dict = extract_bdd_phase_vqa(scenario_path, vqa_path)
    bdd_env_vqa_dict = extract_bdd_environment_vqa(scenario_path, vqa_path)

    bdd_vqa_json_path = os.path.join(args.output_path, f'bdd_{args.split}_phase_vqa.json')
    bdd_env_vqa_json_path = os.path.join(args.output_path, f'bdd_{args.split}_env_vqa.json')
    with open(bdd_vqa_json_path, 'w') as f:
        f.write(json.dumps(bdd_phase_vqa_dict, indent=4))
    with open(bdd_env_vqa_json_path, 'w') as f:
        f.write(json.dumps(bdd_env_vqa_dict, indent=4))
    print(f"Extracted VQA data saved to {bdd_vqa_json_path} and {bdd_env_vqa_json_path}")
    