import os
import json
from tqdm import tqdm
import argparse

import sys
sys.path.insert(1, 'src_cleaned/utils')
from choose_image import find_best_image_for_appearance, find_best_image_for_environment
from choose_image import choose_best_image
from choose_prompt import get_prompt_spatial_pedestrian, get_prompt_spatial_vehicle, get_prompt_env_vqa
from choose_prompt import get_prompt_temporal_pedestrian, get_prompt_temporal_vehicle, get_prompt_phase_vqa

def format_vqa_pair(pair):
    """
    {
        "question": "What is the age group of the pedestrian?",
        "a": "60s",
        "b": "80s",
        "c": "30s",
        "d": "50s",
        "correct": "c"
    }
    """
    question = pair["question"]
    for key in ['a', 'b', 'c', 'd']:
        if key in pair:
            question += f"\n{key}. {pair[key]}"
    return {
        "id": pair["id"],
        "question": question,
        # "answer": ""
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
    """
    res_caption_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:caption>You are ..."
                },
            }
        },
        ...
    ]

    res_vqa_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:multiple-choice qa>You are ...\nQ: What is the age group of the pedestrian?\na. 60s\nb. 80s\nc. 30s\nd. 50s"
                },
            ]
        },
        ...
    ]
    """
    res_caption_dict = []
    res_vqa_dict = []

    vqa_path = os.path.join(args.preprocessed_data_path, 'test_vqa.json')
    bbox_data_path = os.path.join(args.preprocessed_data_path, 'wts_test_annotated_bbox.json')
    with open(vqa_path, 'r') as f:
        vqa_data_json = json.load(f)
    with open(bbox_data_path, 'r') as f:
        bbox_data = json.load(f)

    all_scenarios = os.listdir(args.video_path) + os.listdir(args.video_path + '/normal_trimmed')
    for scenario in tqdm(all_scenarios):
        if scenario == 'normal_trimmed':
            continue
        if 'normal' in scenario:
            image_path = os.path.join(args.preprocessed_data_path, 'bbox_annotated', 'normal_trimmed', scenario)
        else:
            image_path = os.path.join(args.preprocessed_data_path, 'bbox_annotated', scenario)

        appearance_image_path = find_best_image_for_appearance(image_path, bbox_data_path)
        environment_image_path = find_best_image_for_environment(image_path)
        chosen_images = appearance_image_path + environment_image_path
        chosen_images = sorted(chosen_images, key=lambda x: int(x.split('_')[-2][-1]))  # Sắp xếp theo phase number
        chosen_images = [img for img in chosen_images if 'vehicle' in img] + [img for img in chosen_images if 'vehicle' not in img]  # Đưa vehicle view lên trước
        chosen_images = list(set(chosen_images))  # Loại bỏ trùng lặp
        image_path = [os.path.join(scenario, img) for img in chosen_images]
        image_path_for_caption = [img for img in image_path if 'phase2' not in img]  # Loại bỏ phase2 images
        
        conversations = []
        conversations.append({
            "from": "user",
            "value": get_prompt_spatial_pedestrian(len(image_path_for_caption))
        })
        res_caption_dict.append({
            "scenario": scenario,
            "images": image_path_for_caption,
            "conversations": conversations
        })

        conversations = []
        conversations.append({
            "from": "user",
            "value": get_prompt_spatial_vehicle(len(image_path_for_caption))
        })
        res_caption_dict.append({
            "scenario": scenario,
            "images": image_path_for_caption,
            "conversations": conversations
        })

        # VQA
        all_views = os.listdir(f"{args.video_path}/{scenario}") if 'normal' not in scenario else os.listdir(f"{args.video_path}/normal_trimmed/{scenario}")
        for view in all_views:
            all_cameras = os.listdir(f"{args.video_path}/{scenario}/{view}") if 'normal' not in scenario else os.listdir(f"{args.video_path}/normal_trimmed/{scenario}/{view}")
            for camera_name in all_cameras:
                camera_name = camera_name.replace('.mp4', '')
                if camera_name not in vqa_data_json:
                    # print(f"Warning: Camera {camera_name} not found in VQA data for scenario {scenario}. Skipping.")
                    continue
                if 'environment' not in vqa_data_json[camera_name]:
                    # print(f"Warning: No environment data for camera {camera_name} in scenario {scenario}. Skipping.")
                    continue
                vqa_env_scenario = vqa_data_json[camera_name]['environment']
                ped_image_path = get_image_for_wts_ped_vqa(image_path, bbox_data, scenario)
                env_image_path = image_path
                for pair in vqa_env_scenario:
                    formatted_pair = format_vqa_pair(pair)
                    if 'pedestrian' in formatted_pair['question'].lower():
                        image_token = "<image>\n" * len(ped_image_path)
                    else:
                        image_token = "<image>\n" * len(env_image_path)

                    vqa_prompt = get_prompt_env_vqa()
                    conversations = []
                    conversations.append({
                        "from": "user",
                        "value": image_token + vqa_prompt + "\nQ: " + formatted_pair['question']
                    })
                    
                    if 'pedestrian' in formatted_pair['question'].lower():
                        res_vqa_dict.append({
                            "id_vqa": formatted_pair['id'],
                            "images": ped_image_path,
                            "conversations": conversations
                        })
                    else:
                        res_vqa_dict.append({
                            "id_vqa": formatted_pair['id'],
                            "images": env_image_path,
                            "conversations": conversations
                        })
                
    return res_caption_dict, res_vqa_dict

def transform_wts_temporal_format(args):
    """
    res_caption_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:caption-JUDGEMENT>You are ..."
                },
            }
        },
        ...
    ]

    res_vqa_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:multiple-choice-qa-JUDGEMENT>You are ...\nQ: What is the age group of the pedestrian?\na. 60s\nb. 80s\nc. 30s\nd. 50s"
                },
            ]
        },
        ...
    ]
    """
    res_caption_dict = []
    res_vqa_dict = []

    vqa_path = os.path.join(args.preprocessed_data_path, 'test_vqa.json')
    bbox_data_path = os.path.join(args.preprocessed_data_path, 'wts_test_generated_bbox.json')
    bbox_img_path = os.path.join(args.preprocessed_data_path, 'bbox_generated')
    with open(vqa_path, 'r') as f:
        vqa_data_json = json.load(f)

    all_scenarios = os.listdir(args.video_path) + os.listdir(args.video_path + '/normal_trimmed')
    for scenario in tqdm(all_scenarios):
        if scenario == 'normal_trimmed':
            continue
        if 'normal' in scenario:
            image_path = os.listdir(os.path.join(bbox_img_path, 'normal_trimmed', scenario))
        else:
            image_path = os.listdir(os.path.join(bbox_img_path, scenario))

        for phase_number in range(5):
            # Check phase exist ?
            all_views = os.listdir(f"{args.video_path}/{scenario}") if 'normal' not in scenario else os.listdir(f"{args.video_path}/normal_trimmed/{scenario}")
            if 'normal' in scenario:
                caption_file_path = os.path.join(args.caption_folder_path, 'normal_trimmed', scenario, f'{all_views[0]}/{scenario}_caption.json')
            else:
                caption_file_path = os.path.join(args.caption_folder_path, scenario, f'{all_views[0]}/{scenario}_caption.json')

            with open(caption_file_path, 'r') as f:
                caption_data = json.load(f)["event_phase"]
            
            exist_phase = False
            for idx in range(len(caption_data)):
                if caption_data[idx]['labels'][0] == str(phase_number):
                    exist_phase = True
                    break
            if not exist_phase:
                print(f"Warning: Phase {phase_number} not found in scenario {scenario}. Skipping.")
                continue

            # Choose best images for phase
            phase_image_path = [img for img in image_path if f'phase_{phase_number}' in img]
            best_images = choose_best_image(scenario, phase_image_path, phase_number, bbox_data_path)
            best_images = sorted(best_images, key=lambda x: int(x.split('_')[-2]))
            if 'normal' in scenario:
                best_images = [os.path.join(bbox_img_path, 'normal_trimmed', scenario, img) for img in best_images] # aldready has {scenario} in img
            else:
                best_images = [os.path.join(bbox_img_path, scenario, img) for img in best_images]

            conversations = []
            conversations.append({
                "from": "user",
                "value": get_prompt_temporal_pedestrian(len(best_images), str(phase_number))
            })
            res_caption_dict.append({
                "scenario": scenario,
                "phase_number": str(phase_number),
                "images": best_images,
                "conversations": conversations
            })

            conversations = []
            conversations.append({
                "from": "user",
                "value": get_prompt_temporal_vehicle(len(best_images), str(phase_number))
            })
            res_caption_dict.append({
                "scenario": scenario,
                "phase_number": str(phase_number),
                "images": best_images,
                "conversations": conversations
            })

            # VQA
            
            all_views = os.listdir(f"{args.video_path}/{scenario}") if 'normal' not in scenario else os.listdir(f"{args.video_path}/normal_trimmed/{scenario}")
            for view in all_views:
                all_cameras = os.listdir(f"{args.video_path}/{scenario}/{view}") if 'normal' not in scenario else os.listdir(f"{args.video_path}/normal_trimmed/{scenario}/{view}")
                for camera_name in all_cameras:
                    camera_name = camera_name.replace('.mp4', '')
                    if camera_name not in vqa_data_json:
                        # print(f"Warning: Camera {camera_name} not found in VQA data for scenario {scenario}. Skipping.")
                        continue
                    vqa_scenario = vqa_data_json[camera_name][view]
                    if str(phase_number) not in vqa_scenario:
                        print(f"Warning: Phase {phase_number} not found in VQA data for camera {camera_name} in scenario {scenario}. Skipping.")
                        continue
                    vqa_prompt = get_prompt_phase_vqa(len(best_images), str(phase_number))
                    if 'conversations' not in vqa_scenario[str(phase_number)]:
                        print(f"Warning: No conversations found for phase {phase_number} in camera {camera_name} of scenario {scenario}. Skipping.")
                        continue
                    for pair in vqa_scenario[str(phase_number)]['conversations']:
                        formatted_pair = format_vqa_pair(pair)
                        conversations = []
                        conversations.append({
                            "from": "user",
                            "value": vqa_prompt + "\nQ: " + formatted_pair['question']
                        })
                        res_vqa_dict.append({
                            "id_vqa": formatted_pair['id'],
                            "images": best_images,
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
        phase_number = img_name.split('_')[idx][-1] if 'phase' != img_name.split('_')[idx] else img_name.split('_')[idx+1]
        camera_name = '_'.join(img_name.split('_')[:idx])

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
    """
    res_caption_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:caption>You are ..."
                },
            }
        },
        ...
    ]

    res_vqa_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:multiple-choice qa>You are ...\nQ: What is the age group of the pedestrian?\na. 60s\nb. 80s\nc. 30s\nd. 50s"
                },
            ]
        },
        ...
    ]
    """
    res_caption_dict = []
    res_vqa_dict = []

    vqa_path = os.path.join(args.preprocessed_data_path, 'test_vqa.json')
    bbox_data_path = os.path.join(args.preprocessed_data_path, 'bdd_test_annotated_bbox.json')
    with open(vqa_path, 'r') as f:
        vqa_data_json = json.load(f)
    with open(bbox_data_path, 'r') as f:
        bbox_data = json.load(f)

    all_scenarios = os.listdir(args.video_path)
    for scenario in tqdm(all_scenarios):
        if 'video' not in scenario:
            continue
        scenario_name = scenario.split('.mp4')[0]
        image_path = os.path.join(args.preprocessed_data_path, 'bbox_annotated/external', scenario_name)
        appearance_image_path = find_best_image_for_appearance(image_path, bbox_data_path, external=True)
        environment_image_path = find_best_image_for_environment(image_path)
        chosen_images = appearance_image_path + environment_image_path
        chosen_images = sorted(chosen_images, key=lambda x: int(x.split('_')[-2][-1]))  # Sắp xếp theo phase number
        chosen_images = [img for img in chosen_images if 'vehicle' in img] + [img for img in chosen_images if 'vehicle' not in img]  # Đưa vehicle view lên trước
        chosen_images = list(set(chosen_images))  # Loại bỏ trùng lặp
        image_path = [os.path.join(scenario_name, img) for img in chosen_images]
        image_path_for_caption = [image for image in image_path if 'phase2' not in image]
        
        conversations = []
        conversations.append({
            "from": "user",
            "value": get_prompt_spatial_pedestrian(len(image_path_for_caption))
        })
        res_caption_dict.append({
            "scenario": scenario_name,
            "images": image_path_for_caption,
            "conversations": conversations
        })

        conversations = []
        conversations.append({
            "from": "user",
            "value": get_prompt_spatial_vehicle(len(image_path_for_caption))
        })
        res_caption_dict.append({
            "scenario": scenario_name,
            "images": image_path_for_caption,
            "conversations": conversations
        })

        # VQA
        if scenario_name not in vqa_data_json:
            continue
        if 'environment' not in vqa_data_json[scenario_name]:
            continue
        vqa_env_scenario = vqa_data_json[scenario_name]['environment']
        ped_image_path = get_image_for_bdd_ped_vqa(image_path, bbox_data, scenario)
        env_image_path = image_path
        for pair in vqa_env_scenario:
            formatted_pair = format_vqa_pair(pair)
            if 'pedestrian' in formatted_pair['question'].lower():
                image_token = "<image>\n" * len(ped_image_path)
            else:
                image_token = "<image>\n" * len(env_image_path)
            vqa_prompt = get_prompt_env_vqa()
            conversations = []
            conversations.append({
                "from": "user",
                "value": image_token + vqa_prompt + "\nQ: " + formatted_pair['question']
            })
            if 'pedestrian' in formatted_pair['question'].lower():
                res_vqa_dict.append({
                    "id_vqa": formatted_pair['id'],
                    "images": ped_image_path,
                    "conversations": conversations
                })
            else:
                res_vqa_dict.append({
                    "id_vqa": formatted_pair['id'],
                    "images": env_image_path,
                    "conversations": conversations
                })
                
    return res_caption_dict, res_vqa_dict

def transform_bdd_temporal_format(args):
    """
    res_caption_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:caption-JUDGEMENT>You are ..."
                },
            }
        },
        ...
    ]

    res_vqa_dict:
    [
        {
            "images": ["20230707_12_SN17_T1/000000.jpg", "..."],
            "conversations": [
                {
                    "from": "user",
                    "value": "<image>\n<task:multiple-choice-qa-JUDGEMENT>You are ...\nQ: What is the age group of the pedestrian?\na. 60s\nb. 80s\nc. 30s\nd. 50s"
                },
            ]
        },
        ...
    ]
    """
    res_caption_dict = []
    res_vqa_dict = []

    vqa_path = os.path.join(args.preprocessed_data_path, 'test_vqa.json')
    bbox_data_path = os.path.join(args.preprocessed_data_path, 'bdd_test_generated_bbox.json')
    bbox_img_path = os.path.join(args.preprocessed_data_path, 'bbox_generated/external')
    with open(vqa_path, 'r') as f:
        vqa_data_json = json.load(f)

    all_scenarios = os.listdir(args.video_path)
    for scenario in tqdm(all_scenarios):
        if 'video' not in scenario:
            continue
        scenario_name = scenario.split('.mp4')[0]
        image_path = os.listdir(os.path.join(bbox_img_path, scenario_name))

        for phase_number in range(5):
            # Check phase exist ?
            caption_file_path = os.path.join(args.caption_folder_path, f'{scenario_name}_caption.json')

            with open(caption_file_path, 'r') as f:
                caption_data = json.load(f)["event_phase"]
            
            exist_phase = False
            for idx in range(len(caption_data)):
                if caption_data[idx]['labels'][0] == str(phase_number):
                    exist_phase = True
                    break
            if not exist_phase:
                print(f"Warning: Phase {phase_number} not found in scenario {scenario_name}. Skipping.")
                continue

            # Choose best images for phase
            phase_image_path = [img for img in image_path if f'phase{phase_number}' in img]
            best_images = choose_best_image(scenario, phase_image_path, phase_number, bbox_data_path)
            best_images = sorted(best_images, key=lambda x: int(x.split('_')[-1].split('.')[0])) # video8250_phase0_bbox_367.jpg
            best_images = [os.path.join(bbox_img_path, scenario_name, img) for img in best_images]

            conversations = []
            conversations.append({
                "from": "user",
                "value": get_prompt_temporal_pedestrian(len(best_images), str(phase_number))
            })
            res_caption_dict.append({
                "scenario": scenario_name,
                "phase_number": str(phase_number),
                "images": best_images,
                "conversations": conversations
            })

            conversations = []
            conversations.append({
                "from": "user",
                "value": get_prompt_temporal_vehicle(len(best_images), str(phase_number))
            })
            res_caption_dict.append({
                "scenario": scenario_name,
                "phase_number": str(phase_number),
                "images": best_images,
                "conversations": conversations
            })

            # VQA

            if scenario_name not in vqa_data_json:
                continue
            vqa_scenario = vqa_data_json[scenario_name]['overhead_view']
            if str(phase_number) not in vqa_scenario:
                continue
            vqa_prompt = get_prompt_phase_vqa(len(best_images), str(phase_number))
            if 'conversations' not in vqa_scenario[str(phase_number)]:
                print(f"Warning: No conversations found for phase {phase_number} of scenario {scenario}. Skipping.")
                continue
            for pair in vqa_scenario[str(phase_number)]['conversations']:
                formatted_pair = format_vqa_pair(pair)
                conversations = []
                conversations.append({
                    "from": "user",
                    "value": vqa_prompt + "\nQ: " + formatted_pair['question']
                })
                res_vqa_dict.append({
                    "id_vqa": formatted_pair['id'],
                    "images": best_images,
                    "conversations": conversations
                })

    return res_caption_dict, res_vqa_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test', help='Path to the test dataset')
    parser.add_argument('--preprocessed_data_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test', help='Path to preprocessed data')

    args = parser.parse_args()
    if not os.path.exists(args.data_path + '/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/'):
        print(f"Data path {args.data_path} does not exist. Please check the path.")
        exit(1)
    os.makedirs(args.preprocessed_data_path, exist_ok=True)

    args.video_path = os.path.join(args.data_path, 'SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/videos/test/public')
    args.caption_folder_path = os.path.join(args.data_path, 'SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/annotations/caption/test/public_challenge')

    wts_spatial_caption, wts_env_vqa = transform_wts_spatial_format(args)
    wts_temporal_caption, wts_phase_vqa = transform_wts_temporal_format(args)
    bdd_spatial_caption, bdd_env_vqa = transform_bdd_spatial_format(args)
    bdd_temporal_caption, bdd_phase_vqa = transform_bdd_temporal_format(args)

    spatial_caption = wts_spatial_caption + bdd_spatial_caption
    env_vqa = wts_env_vqa + bdd_env_vqa
    temporal_caption = wts_temporal_caption + bdd_temporal_caption
    phase_vqa = wts_phase_vqa + bdd_phase_vqa

    os.makedirs(os.path.join(args.preprocessed_data_path, 'test_data'), exist_ok=True)
    output_spatial_path = os.path.join(args.preprocessed_data_path, 'test_data', 'test_spatial_caption.json')
    output_env_vqa_path = os.path.join(args.preprocessed_data_path, 'test_data', 'test_env_vqa.json')
    output_temporal_path = os.path.join(args.preprocessed_data_path, 'test_data', 'test_temporal_caption.json')
    output_phase_vqa_path = os.path.join(args.preprocessed_data_path, 'test_data', 'test_phase_vqa.json')

    with open(output_spatial_path, 'w') as f:
        json.dump(spatial_caption, f, indent=4)
    with open(output_env_vqa_path, 'w') as f:
        json.dump(env_vqa, f, indent=4)
    with open(output_temporal_path, 'w') as f:
        json.dump(temporal_caption, f, indent=4)
    with open(output_phase_vqa_path, 'w') as f:
        json.dump(phase_vqa, f, indent=4)
    
    print(f"Transformed WTS spatial caption saved to {output_spatial_path}")
    print(f"Transformed WTS environment VQA saved to {output_env_vqa_path}")
    print(f"Transformed WTS temporal caption saved to {output_temporal_path}")
    print(f"Transformed WTS phase VQA saved to {output_phase_vqa_path}")
