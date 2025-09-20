import os
import json
from tqdm import tqdm
import argparse

def extract_wts_bbox_annotated(video_path, bbox_path, caption_path, normal_trimmed=False):
    """
    Extracts bounding box annotations from the given video and bbox paths in WTS dataset

    Args:
        video_path (str): Path to the video directory
        bbox_path (str): Path to the bbox annotations directory
        caption_path (str): Path to the caption annotations directory
        normal_trimmed (bool): If True, processes the 'normal_trimmed' subdirectory
    Returns:
        dict: A dictionary containing bounding box annotations

    Example res_bbox_dict format:
    {
        "20230707_17_CY23_T1": {
            "20230707_17_CY23_T1_Camera2_2": {
                "4": {
                    "frame_id": 1118,
                    "start_time": xxx,
                    "end_time": xxx,
                    "ped_bbox": null,
                    "veh_bbox": [
                        1280.3,
                        814.4,
                        639.7,
                        265.6
                    ]
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
    res_bbox_dict = {}
    if normal_trimmed:
        video_path = video_path + '/normal_trimmed'
        caption_path = caption_path + '/normal_trimmed'

    for scenario in tqdm(os.listdir(video_path)):
        if scenario == 'normal_trimmed': # Skip folder normal_trimmed (when os.listdir)
            continue

        for view in ['overhead_view', 'vehicle_view']:
            if view not in os.listdir(f'{video_path}/{scenario}/'):
                continue
            scenario_caption_path = f'{caption_path}/{scenario}/{view}/{scenario}_caption.json'

            try:
                scenario_caption_dict = json.load(open(scenario_caption_path))['event_phase'] 
            except:
                # print(f"Error loading {scenario_caption_path}")
                continue

            fps = 30.0  
            for camera_name in os.listdir(f'{video_path}/{scenario}/{view}/'):
                camera_name = os.path.splitext(camera_name)[0]
                if normal_trimmed:
                    scenario_ped_bbox_path = f'{bbox_path}/pedestrian/test/public/normal_trimmed/{scenario}/{view}/{camera_name}_bbox.json'
                    # print(f"Processing pedestrian bbox file: {scenario_ped_bbox_path}")
                    scenario_veh_bbox_path = f'{bbox_path}/vehicle/test/public/normal_trimmed/{scenario}/{view}/{camera_name}_bbox.json'
                else:
                    scenario_ped_bbox_path = f'{bbox_path}/pedestrian/test/public/{scenario}/{view}/{camera_name}_bbox.json'
                    scenario_veh_bbox_path = f'{bbox_path}/vehicle/test/public/{scenario}/{view}/{camera_name}_bbox.json'

                ped_res_bbox_dict, veh_res_bbox_dict = None, None
                if os.path.exists(scenario_ped_bbox_path):
                    ped_res_bbox_dict = json.load(open(scenario_ped_bbox_path))['annotations']
    
                if os.path.exists(scenario_veh_bbox_path):
                    veh_res_bbox_dict = json.load(open(scenario_veh_bbox_path))['annotations']

                for phase in scenario_caption_dict:
                    st_time = float(phase['start_time'])
                    ed_time = float(phase['end_time'])
                    st_frame = int(st_time * fps)

                    # phase may not have bbox (due to occlusion, out of view,...)
                    ped_bbox, veh_bbox = None, None
                    if ped_res_bbox_dict is not None:
                        for i in range(len(ped_res_bbox_dict)):
                            if int(ped_res_bbox_dict[i]['phase_number']) == int(phase['labels'][0]):
                                ped_bbox = ped_res_bbox_dict[i]['bbox']
                                break
                    if veh_res_bbox_dict is not None:
                        for i in range(len(veh_res_bbox_dict)):    
                            if int(veh_res_bbox_dict[i]['phase_number']) == int(phase['labels'][0]):
                                veh_bbox = veh_res_bbox_dict[i]['bbox']
                                break
                    
                    if scenario not in res_bbox_dict:
                        res_bbox_dict[scenario] = {}
                    if camera_name not in res_bbox_dict[scenario]:
                        res_bbox_dict[scenario][camera_name] = {}
                    if phase['labels'][0] not in res_bbox_dict[scenario][camera_name]:
                        res_bbox_dict[scenario][camera_name][phase['labels'][0]] = {}
                    res_bbox_dict[scenario][camera_name][phase['labels'][0]].update({
                        'frame_id': st_frame,
                        'start_time': st_time,
                        'end_time': ed_time,
                        'ped_bbox': ped_bbox if ped_bbox else None,
                        'veh_bbox': veh_bbox if veh_bbox else None,
                    })
    return res_bbox_dict

def extract_bdd_bbox_annotated(video_path, bbox_path, caption_path):
    """
    Extracts bounding box annotations from the given video and bbox paths in BDD dataset

    Args:
        video_path (str): Path to the video directory
        anno_path (str): Path to the bbox annotations directory
        caption_path (str): Path to the caption annotations directory
    Returns:
        dict: A dictionary containing bounding box annotations

    res_bbox_dict:
    {
        "video1004.mp4": {
            "0": {
                "frame_id": 189,
                "start_time": 6.326,
                "end_time": 7.757,
                "ped_bbox": [
                    ...
                ],
                "veh_bbox": null
            },
            "1": {
                "frame_id": 233,
                "start_time": 7.791,
                "end_time": 7.857,
                "ped_bbox": [
                    787,
                    315,
                    27,
                    79
                ],
                "veh_bbox": null
            },
            "2": {
                "frame_id": 236,
                "start_time": 7.891,
                "end_time": 8.723,
                "ped_bbox": [
                    787,
                    315,
                    27,
                    79
                ],
                "veh_bbox": null
            },
            ...
        },
        ...
    }
    """
    
    res_bbox_dict = {}
    for scenario in tqdm(os.listdir(video_path)):
        scenario_caption_path = caption_path + f"/{scenario.replace('.mp4', '')}_caption.json"

        try:
            scenario_caption_dict = json.load(open(scenario_caption_path))
        except:
            # print(f"Error loading {scenario_caption_path}")
            continue

        fps = float(scenario_caption_dict['fps'])
        scenario_caption_dict = scenario_caption_dict['event_phase'] 
        
        scenario_ped_bbox_path = bbox_path + f"/test/public/{scenario.replace('.mp4','_bbox.json')}"
        ped_bbox_dict = None
        if os.path.exists(scenario_ped_bbox_path):
            ped_bbox_dict = json.load(open(scenario_ped_bbox_path))['annotations']

        scenario_caption_dict.sort(key=lambda x: float(x['start_time']))
        # sort by start time
        for i, phase in enumerate(scenario_caption_dict):
            st_time = float(phase['start_time'])
            ed_time = float(phase['end_time'])
            st_frame = int(st_time * fps)
           
            # phase may not have bbox (due to occlusion, out of view,...)
            ped_bbox, veh_bbox = None, None
            if ped_bbox_dict is not None:
                for j in range(len(ped_bbox_dict)):
                    if int(ped_bbox_dict[j]['phase_number']) == int(i):
                        ped_bbox = ped_bbox_dict[j]['bbox']
                        break
            
            if scenario not in res_bbox_dict:
                res_bbox_dict[scenario] = {}

            if str(i) not in res_bbox_dict[scenario]:
                res_bbox_dict[scenario][str(i)] = {}
            
            res_bbox_dict[scenario][str(i)].update({
                'frame_id': st_frame,
                'start_time': st_time,
                'end_time': ed_time,
                'ped_bbox': ped_bbox if ped_bbox else None,
                'veh_bbox': veh_bbox if veh_bbox else None,
            })
    return res_bbox_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test", help='Data directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test/", help='Directory for saving json file')

    args = parser.parse_args()
    if not os.path.exists(args.data_path + '/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/'):
        print(f"Data path {args.data_path} does not exist. Please check the path.")
        exit(1)
    os.makedirs(args.output_path, exist_ok=True)
    args.data_path = os.path.join(args.data_path, 'SubTask1-Caption/')

    video_path = args.data_path + f'WTS_DATASET_PUBLIC_TEST/videos/test/public'
    bbox_path = args.data_path + f'WTS_DATASET_PUBLIC_TEST_BBOX/annotations/bbox_annotated'
    caption_path = args.data_path + f'WTS_DATASET_PUBLIC_TEST/annotations/caption/test/public_challenge'

    wts_bbox_dict = extract_wts_bbox_annotated(video_path, bbox_path, caption_path)
    wts_bbox_dict.update(extract_wts_bbox_annotated(video_path, bbox_path, caption_path, normal_trimmed=True))
    with open(os.path.join(args.output_path, f'test_wts_bbox_annotated.json'), 'w') as f:
        f.write(json.dumps(wts_bbox_dict, indent=4))
    print(f"Extracted bbox data saved to {os.path.join(args.output_path, 'test_wts_bbox_annotated.json')}")

    
    video_path = args.data_path + f'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public'
    bbox_path = args.data_path + f'WTS_DATASET_PUBLIC_TEST_BBOX/external/BDD_PC_5K/annotations/bbox_annotated'
    caption_path = args.data_path + f'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/annotations/caption/test/public_challenge'

    bdd_bbox_dict = extract_bdd_bbox_annotated(video_path, bbox_path, caption_path)
    with open(os.path.join(args.output_path, f'test_bdd_bbox_annotated.json'), 'w') as f:
        f.write(json.dumps(bdd_bbox_dict, indent=4))
    print(f"Extracted bbox data saved to {os.path.join(args.output_path, 'test_bdd_bbox_annotated.json')}")
    