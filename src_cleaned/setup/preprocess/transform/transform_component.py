import os
import json
from tqdm import tqdm
import argparse

import sys
sys.path.insert(1, 'src_cleaned/utils')
from choose_prompt import get_prompt_spatial_pedestrian, get_prompt_spatial_vehicle, get_prompt_env_vqa
from choose_prompt import get_prompt_temporal_pedestrian, get_prompt_temporal_vehicle, get_prompt_phase_vqa

def format_vqa_pair(pair):
    '''
    {
        "question": "What is the age group of the pedestrian?",
        "a": "60s",
        "b": "80s",
        "c": "30s",
        "d": "50s",
        "correct": "c"
    }
    '''
    question = pair["question"]
    for key in ['a', 'b', 'c', 'd']:
        if key in pair:
            question += f"\n{key}. {pair[key]}"
    return {
        "question": question,
        "answer": pair["correct"] + f". {pair[pair['correct']]}",
    }

def get_image_for_wts_ped_vqa(image_path, bbox_data, scenario):
    max_bbox_area, second_max_bbox_area = 0.0, 0.0
    candidate_image_1, candidate_image_2 = None, None
    for img in image_path:
        img_name = img.split('/')[-1].split('.jpg')[0]
        idx = -1
        for idx in range(-1, -5, -1):
            if 'phase' in img_name.split('_')[idx]:
                break
        phase_number = img_name.split('_')[idx][-1] if 'phase' != img_name.split('_')[idx] else img_name.split('_')[idx+1]
        camera_name = '_'.join(img_name.split('_')[:idx])

        # Get and compare bbox area
        current_bbox = bbox_data[scenario][camera_name][phase_number]['ped_bbox']
        if not current_bbox:
            continue
        area_bbox = current_bbox[-2] * current_bbox[-1]
        if area_bbox > max_bbox_area:
            max_bbox_area = area_bbox
            candidate_image_1 = img
        elif area_bbox > second_max_bbox_area:
            second_max_bbox_area = area_bbox
            candidate_image_2 = img

    if candidate_image_1 is None or candidate_image_2 is None:
        image_path = [image_path[0], image_path[-1]] # first overhead + last vehicle view
        # print(f"Warning: Not enough pedestrian bounding box found for scenario {scenario}. Using first and last images instead.")
    else:
        image_path = [candidate_image_1, candidate_image_2] # overhead + vehicle view
    return image_path

def transform_wts_spatial_format(args):
    '''
    [
        {
            "image": ["path/to/image", "path/to/image2",...],
            "conversations": [
                {
                    "from": "user",
                    "value": "Picture 1: <image>\n<image>\n...\nDescribe the appearance of the pedestrian in the image. Include details such as color, make, model, and any visible features."
                },
                {
                    "from": "gpt",
                    "value": "{rewrite prompt}"
                }
            ]
        },
        ...
    ]
    '''
    res_caption_dict = []
    res_vqa_dict = []
    preprocessed_data_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_spatial_pair.json')
    vqa_env_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_env_vqa.json')
    bbox_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_annotated_bbox.json')

    try:
        with open(os.path.join(preprocessed_data_path), 'r') as f:
            spatial_captions = json.load(f)
    except FileNotFoundError:
        print(f"File wts_{args.split}_spatial_pair not found in {preprocessed_data_path}. Please check the path.")
        return

    try:
        with open(os.path.join(vqa_env_path), 'r') as f:
            vqa_env = json.load(f)
    except FileNotFoundError:
        print(f"File wts_{args.split}_env_vqa.json not found in {preprocessed_data_path}. Please check the path.")
        return
    
    try:
        with open(os.path.join(bbox_path), 'r') as f:
            bbox_data = json.load(f)
    except FileNotFoundError:
        print(f"File wts_{args.split}_bbox.json not found in {preprocessed_data_path}. Please check the path.")
        return

    for item in tqdm(spatial_captions):
        item_keys = list(item.keys())
        scenario = item_keys[0] 
        image_path = item[scenario]['image']
        image_path = list(dict.fromkeys(image_path)) 
        conversations = []
        
        if not image_path:
            print(f"Image path is empty for item: {item}")
            continue
        
        ped_caption = item[scenario]['caption_ped']
        if ped_caption:
            conversations.append({
                "from": "user",
                "value": f"{get_prompt_spatial_pedestrian(len(image_path))}"
            })
            conversations.append({
                "from": "gpt",
                "value": ped_caption
            })
            res_caption_dict.append({
                "ped_or_veh": "ped",
                "images": image_path,
                "conversations": conversations
            })
        conversations = []
        
        # Xử lý caption cho vehicle
        veh_caption = item[scenario]['caption_veh']
        if veh_caption:
            conversations.append({
                "from": "user",
                "value": f"{get_prompt_spatial_vehicle(len(image_path))}"  
            })
            conversations.append({
                "from": "gpt",
                "value": veh_caption
            })
            res_caption_dict.append({
                "ped_or_veh": "veh",
                "images": image_path,
                "conversations": conversations
            })

        conversations = []
        # VQA environment
        if scenario not in vqa_env:
            # print(f"{scenario} not found in VQA environment data. Some videos do not have VQA for them, please ignore this message")
            continue
        item_env_vqa = vqa_env.get(scenario)['environment']
        if item_env_vqa:
            ped_image_path = get_image_for_wts_ped_vqa(image_path, bbox_data, scenario)
            env_image_path = image_path
            for pair in item_env_vqa:
                conversations = []
                formatted_pair = format_vqa_pair(pair)

                # Only Image token for first question in group 
                if 'pedestrian' in formatted_pair['question'].lower():
                    image_token = "<image>\n" * len(ped_image_path)
                else:
                    image_token = "<image>\n" * len(env_image_path)

                # Conversation formatting
                vqa_prompt = get_prompt_env_vqa()
                conversations.append({
                    "from": "user",
                    "value": image_token + vqa_prompt + "\nQ: " + formatted_pair['question']
                })
                conversations.append({
                    "from": "gpt",
                    "value": formatted_pair['answer']
                })
                
                if 'pedestrian' in formatted_pair['question'].lower():
                    res_vqa_dict.append({
                        "images": ped_image_path,
                        "conversations": conversations
                    })
                else:
                    res_vqa_dict.append({
                        "images": env_image_path,
                        "conversations": conversations
                    })

    return res_caption_dict, res_vqa_dict

def transform_wts_temporal_format(args):
    '''
    [
        {
            "image": ["path/to/image", "path/to/image2",...],
            "conversations": [
                {
                    "from": "user",
                    "value": "Picture 1: <image>\n<image>\n...\nDescribe the appearance of the pedestrian in the image. Include details such as color, make, model, and any visible features."
                },
                {
                    "from": "gpt",
                    "value": "{rewrite prompt}"
                }
            ]
        },
        ...
    ]
    '''
    res_caption_dict = []
    res_vqa_dict = []
    preprocessed_data_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_temporal_pair.json')
    vqa_path = os.path.join(args.preprocessed_data_path, f'wts_{args.split}_phase_vqa.json')
    
    try:
        with open(os.path.join(preprocessed_data_path), 'r') as f:
            temporal_captions = json.load(f)
    except FileNotFoundError:
        print(f"File wts_{args.split}_temporal_pair.json not found in {preprocessed_data_path}. Please check the path.")
        return

    try:
        with open(os.path.join(vqa_path), 'r') as f:
            vqa = json.load(f)
    except FileNotFoundError:
        print(f"File wts_{args.split}_vqa.json not found in {preprocessed_data_path}. Please check the path.")
        return

    for item in tqdm(temporal_captions):
        scenario = list(item.keys())[0]
        item_vqa = vqa[scenario] if scenario in vqa else {}
        overhead_vqa = item_vqa['overhead_view'] if item_vqa.get('overhead_view') else None
        vehicle_vqa = item_vqa['vehicle_view'] if item_vqa.get('vehicle_view') else None
        for phase_info in item[scenario]:
            phase_number = phase_info['labels']
            image_path = phase_info['image']
            image_path = list(dict.fromkeys(image_path)) 
            for i in range(len(image_path)):
                if 'normal' in scenario:
                    image_path[i] = os.path.join(args.preprocessed_data_path, f'drawn_bbox_generated/{args.split}/normal_trimmed/{scenario}', image_path[i])
                else:
                    image_path[i] = os.path.join(args.preprocessed_data_path, f'drawn_bbox_generated/{args.split}/{scenario}', image_path[i])

            conversations = []
            
            if not image_path:
                print(f"Image path is empty for item: {item}")
                continue

            ped_caption = phase_info['caption_ped']
            if ped_caption:
                conversations.append({
                    "from": "user",
                    "value": f"{get_prompt_temporal_pedestrian(len(image_path), phase_number)}"
                })
                conversations.append({
                    "from": "gpt",
                    "value": ped_caption
                })
                res_caption_dict.append({
                    "ped_or_veh": "ped",
                    "images": image_path,
                    "conversations": conversations
                })
            conversations = []
            
            # Xử lý caption cho vehicle
            veh_caption = phase_info['caption_veh']
            if veh_caption:
                conversations.append({
                    "from": "user",
                    "value": f"{get_prompt_temporal_vehicle(len(image_path), phase_number)}"
                })
                conversations.append({
                    "from": "gpt",
                    "value": veh_caption
                })
                res_caption_dict.append({
                    "ped_or_veh": "veh",
                    "images": image_path,
                    "conversations": conversations
                })

            conversations = []
        
            if overhead_vqa and phase_number in overhead_vqa:
                overhead_phase_vqa = overhead_vqa[phase_number]['conversations']
                for pair in overhead_phase_vqa:
                    conversations = []
                    formatted_pair = format_vqa_pair(pair)
                    if 'pedestrian' in formatted_pair['question'].lower():
                        vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number, pedestrian_pov=True)
                    else:
                        vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number, pedestrian_pov=False)
                    conversations.append({
                        "from": "user",
                        "value": vqa_prompt + "\nQ: " + formatted_pair['question']
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": formatted_pair['answer']
                    })
                    res_vqa_dict.append({
                        "images": image_path,
                        "conversations": conversations
                    })
            if vehicle_vqa and phase_number in vehicle_vqa:
                vehicle_phase_vqa = vehicle_vqa[phase_number]['conversations']
                for pair in vehicle_phase_vqa:
                    conversations = []
                    formatted_pair = format_vqa_pair(pair)
                    if 'pedestrian' in formatted_pair['question'].lower():
                        vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number, pedestrian_pov=True)
                    else:
                        vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number, pedestrian_pov=False)
                    conversations.append({
                        "from": "user",
                        "value": vqa_prompt + "\nQ: " + formatted_pair['question']
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": formatted_pair['answer']
                    })
                    res_vqa_dict.append({
                        "images": image_path,
                        "conversations": conversations
                    })

    return res_caption_dict, res_vqa_dict

def get_image_for_bdd_ped_vqa(image_path, bbox_data, scenario):
    max_bbox_area, second_max_bbox_area = 0.0, 0.0
    candidate_image_1, candidate_image_2 = None, None
    for img in image_path:
        img_name = img.split('/')[-1].split('.jpg')[0]
        idx = -1
        for idx in range(-1, -5, -1):
            if 'phase' in img_name.split('_')[idx]: 
                break
        phase_number = img_name.split('_')[idx][-1] if 'phase' != img_name.split('_')[idx] else img_name.split('_')[idx+1] # Sẽ sửa lại chung với phần vẽ not fix WTS

        # Get and compare bbox area
        current_bbox = bbox_data[scenario][phase_number]['ped_bbox']
        if not current_bbox:
            continue
        area_bbox = current_bbox[-2] * current_bbox[-1]
        if area_bbox > max_bbox_area:
            max_bbox_area = area_bbox
            candidate_image_1 = img
        elif area_bbox > second_max_bbox_area:
            second_max_bbox_area = area_bbox
            candidate_image_2 = img

    if candidate_image_1 is None or candidate_image_2 is None:
        image_path = [image_path[0], image_path[-1]] # first overhead + last vehicle view
        # print(f"Warning: Not enough pedestrian bounding box found for scenario {scenario}. Using first and last images instead.")
    else:
        image_path = [candidate_image_1, candidate_image_2] # overhead + vehicle view
    return image_path

def transform_bdd_spatial_format(args):
    '''
    [
        {
            "image": ["path/to/image", "path/to/image2",...],
            "conversations": [
                {
                    "from": "user",
                    "value": "Picture 1: <image>\n<image>\n...\nDescribe the appearance of the pedestrian in the image. Include details such as color, make, model, and any visible features."
                },
                {
                    "from": "gpt",
                    "value": "{rewrite prompt}"
                }
            ]
        },
        ...
    ]
    '''
    res_caption_dict = []
    res_vqa_dict = []
    preprocessed_data_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_spatial_pair.json')
    vqa_env_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_env_vqa.json')
    bbox_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_annotated_bbox.json')

    try:
        with open(os.path.join(preprocessed_data_path), 'r') as f:
            fixed_captions = json.load(f)
    except FileNotFoundError:
        print(f"File bdd_{args.split}_spatial_pair not found in {preprocessed_data_path}. Please check the path.")
        return

    try:
        with open(os.path.join(vqa_env_path), 'r') as f:
            vqa_env = json.load(f)
    except FileNotFoundError:
        print(f"File bdd_{args.split}_env_vqa.json not found in {preprocessed_data_path}. Please check the path.")
        return
    
    try:
        with open(os.path.join(bbox_path), 'r') as f:
            bbox_data = json.load(f)
    except FileNotFoundError:
        print(f"File bdd_{args.split}_drawn_bbox_generated.json not found in {preprocessed_data_path}. Please check the path.")
        return

    for item in tqdm(fixed_captions):
        item_keys = list(item.keys())
        scenario = item_keys[0] 
        image_path = item[scenario]['image']
        image_path = list(dict.fromkeys(image_path)) 
        conversations = []
        
        if not image_path:
            print(f"Image path is empty for item: {item}")
            continue
        
        ped_caption = item[scenario]['caption_ped']
        if ped_caption:
            conversations.append({
                "from": "user",
                "value": f"{get_prompt_spatial_pedestrian(len(image_path))}"
                # "value": f"{get_prompt_temporal_pedestrian(len(image_path), phase_number)}"
            })
            conversations.append({
                "from": "gpt",
                "value": ped_caption
            })
            res_caption_dict.append({
                "ped_or_veh": "ped",
                "images": image_path,
                "conversations": conversations
            })
        conversations = []
        
        # Xử lý caption cho vehicle
        veh_caption = item[scenario]['caption_veh']
        if veh_caption:
            conversations.append({
                "from": "user",
                "value": f"{get_prompt_spatial_vehicle(len(image_path))}"  
                # "value": f"{get_prompt_temporal_vehicle(len(image_path), phase_number)}"
            })
            conversations.append({
                "from": "gpt",
                "value": veh_caption
            })
            res_caption_dict.append({
                "ped_or_veh": "veh",
                "images": image_path,
                "conversations": conversations
            })

        conversations = []
        # VQA environment
        if scenario not in vqa_env:
            # print(f"{scenario} not found in VQA environment data. Some videos do not have VQA for them, please ignore this message")
            continue

        item_env_vqa = vqa_env.get(scenario)['environment']
        if item_env_vqa:
            ped_image_path = get_image_for_bdd_ped_vqa(image_path, bbox_data, scenario+'.mp4')
            env_image_path = image_path
            for pair in item_env_vqa:
                conversations = []
                formatted_pair = format_vqa_pair(pair)

                # Only Image token for first question in group 
                if 'pedestrian' in formatted_pair['question'].lower():
                    image_token = "<image>\n" * len(ped_image_path)
                else:
                    image_token = "<image>\n" * len(env_image_path)

                # Conversation formatting
                vqa_prompt = get_prompt_env_vqa()
                conversations.append({
                    "from": "user",
                    "value": image_token + vqa_prompt + "\nQ: " + formatted_pair['question']
                })
                conversations.append({
                    "from": "gpt",
                    "value": formatted_pair['answer']
                })
                
                if 'pedestrian' in formatted_pair['question'].lower():
                    res_vqa_dict.append({
                        "images": ped_image_path,
                        "conversations": conversations
                    })
                else:
                    res_vqa_dict.append({
                        "images": env_image_path,
                        "conversations": conversations
                    })

    return res_caption_dict, res_vqa_dict

def transform_bdd_temporal_format(args):
    '''
    [
        {
            "image": ["path/to/image", "path/to/image2",...],
            "conversations": [
                {
                    "from": "user",
                    "value": "Picture 1: <image>\n<image>\n...\nDescribe the appearance of the pedestrian in the image. Include details such as color, make, model, and any visible features."
                },
                {
                    "from": "gpt",
                    "value": "{rewrite prompt}"
                }
            ]
        },
        ...
    ]
    '''
    res_caption_dict = []
    res_vqa_dict = []
    preprocessed_data_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_temporal_pair.json')
    vqa_path = os.path.join(args.preprocessed_data_path, f'bdd_{args.split}_phase_vqa.json')

    try:
        with open(os.path.join(preprocessed_data_path), 'r') as f:
            fixed_captions = json.load(f)
    except FileNotFoundError:
        print(f"File bdd_{args.split}_temporal_pair.json not found in {preprocessed_data_path}. Please check the path.")
        return

    try:
        with open(os.path.join(vqa_path), 'r') as f:
            vqa = json.load(f)
    except FileNotFoundError:
        print(f"File bdd_{args.split}_phase_vqa.json not found in {preprocessed_data_path}. Please check the path.")
        return


    for item in tqdm(fixed_captions):
        scenario = list(item.keys())[0]  
        scenario_name = scenario.split('.mp4')[0]  # Remove the .mp4 extension if it exists
        item_vqa = vqa[scenario_name] if scenario_name in vqa else {}
        overhead_vqa = item_vqa['overhead_view'] if item_vqa.get('overhead_view') else None
        for phase_info in item[scenario]:
            phase_number = phase_info['labels']
            image_path = phase_info['image']
            image_path = list(dict.fromkeys(image_path)) 
            for i in range(len(image_path)):
                image_path[i] = os.path.join(args.preprocessed_data_path, f'drawn_bbox_generated/external/{args.split}/{scenario_name}', image_path[i])

            conversations = []
            
            if not image_path:
                print(f"Image path is empty for item: {item}")
                continue
            
            ped_caption = phase_info['caption_ped']
            if ped_caption:
                conversations.append({
                    "from": "user",
                    "value": f"{get_prompt_temporal_pedestrian(len(image_path), phase_number)}"
                })
                conversations.append({
                    "from": "gpt",
                    "value": ped_caption
                })
                res_caption_dict.append({
                    "ped_or_veh": "ped",
                    "images": image_path,
                    "conversations": conversations
                })
            conversations = []
            
            # Xử lý caption cho vehicle
            veh_caption = phase_info['caption_veh']
            if veh_caption:
                conversations.append({
                    "from": "user",
                    "value": f"{get_prompt_temporal_vehicle(len(image_path), phase_number)}"
                })
                conversations.append({
                    "from": "gpt",
                    "value": veh_caption
                })
                res_caption_dict.append({
                    "ped_or_veh": "veh",
                    "images": image_path,
                    "conversations": conversations
                })
                
            vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number)
            if overhead_vqa and phase_number in overhead_vqa:
                overhead_phase_vqa = overhead_vqa[phase_number]['conversations']
                for pair in overhead_phase_vqa:
                    conversations = []
                    formatted_pair = format_vqa_pair(pair)
                    if 'pedestrian' in formatted_pair['question'].lower():
                        vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number, pedestrian_pov=True)
                    else:
                        vqa_prompt = get_prompt_phase_vqa(len(image_path), phase_number, pedestrian_pov=False)
                    conversations.append({
                        "from": "user",
                        "value": vqa_prompt + "\nQ: " + formatted_pair['question']
                    })
                    conversations.append({
                        "from": "gpt",
                        "value": formatted_pair['answer']
                    })
                    res_vqa_dict.append({
                        "images": image_path,
                        "conversations": conversations
                    })
                    

    return res_caption_dict, res_vqa_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--preprocessed_data_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned', help='Path to preprocessed data')

    args = parser.parse_args()

    # WTS
    spatial_caption, env_vqa = transform_wts_spatial_format(args)
    temporal_caption, phase_vqa = transform_wts_temporal_format(args)
    # BDD 
    bdd_spatial_caption, bdd_env_vqa = transform_bdd_spatial_format(args)
    bdd_temporal_caption, bdd_phase_vqa = transform_bdd_temporal_format(args)

    spatial_caption.extend(bdd_spatial_caption)
    temporal_caption.extend(bdd_temporal_caption)
    env_vqa.extend(bdd_env_vqa)
    phase_vqa.extend(bdd_phase_vqa)

    # Save
    os.makedirs(os.path.join(args.preprocessed_data_path, 'train_data'), exist_ok=True)
    spatial_caption_output_path = os.path.join(args.preprocessed_data_path, 'train_data', f'transformed_{args.split}_spatial_caption.json')
    with open(spatial_caption_output_path, 'w') as f:
        json.dump(spatial_caption, f, indent=4)
        print(f"Transformed spatial caption data saved to {spatial_caption_output_path}")
    temporal_caption_output_path = os.path.join(args.preprocessed_data_path, 'train_data', f'transformed_{args.split}_temporal_caption.json')
    with open(temporal_caption_output_path, 'w') as f:
        json.dump(temporal_caption, f, indent=4)
        print(f"Transformed temporal caption data saved to {temporal_caption_output_path}")

    env_vqa_output_path = os.path.join(args.preprocessed_data_path, 'train_data', f'transformed_{args.split}_env_vqa.json')
    with open(env_vqa_output_path, 'w') as f:
        json.dump(env_vqa, f, indent=4)
        print(f"Transformed env VQA data saved to {env_vqa_output_path}")
    phase_vqa_output_path = os.path.join(args.preprocessed_data_path, 'train_data', f'transformed_{args.split}_phase_vqa.json')
    with open(phase_vqa_output_path, 'w') as f:
        json.dump(phase_vqa, f, indent=4)
        print(f"Transformed phase VQA data saved to {phase_vqa_output_path}")