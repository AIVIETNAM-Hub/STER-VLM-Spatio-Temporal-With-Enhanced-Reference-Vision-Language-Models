import os
import json
from tqdm import tqdm
import argparse
import pandas as pd

def extract_wts_bbox_annotated(split, data_path, video_path, bbox_path, caption_path, normal_trimmed=False):
    '''
    Extract annotated bounding boxes from WTS dataset.

    Args:
        split (str): Dataset split to process (train, val)
        data_path (str): Path to the WTS dataset
        video_path (str): Path to the video files
        bbox_path (str): Path to the bounding box annotations
        caption_path (str): Path to the caption annotations
        normal_trimmed (bool): Whether to use normal trimmed videos
    Returns:
        dict: A dictionary containing bounding box annotations for each video and phase

    Example of res_bbox_dict format:
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
            },
            "20230707_17_CY23_T1_Camera2_3": {
                ...
            },
            ...
        },
        "20230728_31_CN37_T1": {
            ...
        },
        ...
    }
    '''
    res_bbox_dict = {}
    if normal_trimmed:
        # Pseudo df of main view
        video_path = video_path + '/normal_trimmed'
        caption_path = caption_path + '/normal_trimmed'
        main_view_df = pd.DataFrame(columns=['Scene', 'viewpoint1'])
        main_view_df['Scene'] = os.listdir(video_path)
        main_view_df['viewpoint1'] = main_view_df['Scene'].apply(lambda x: os.listdir(f'{video_path}/{x}/overhead_view/')[0]) 
    else:
        main_view_df = pd.read_csv(f"{data_path}/view_used_as_main_reference_for_multiview_scenario.csv")
    
    for scenario in tqdm(os.listdir(video_path)):
        if scenario == 'normal_trimmed': # Skip folder normal_trimmed (when os.listdir)
            continue

        for view in ['overhead_view', 'vehicle_view']:
            if view == 'overhead_view':
                scenario_main_view = main_view_df[main_view_df['Scene'] == scenario].drop(columns=['Scene'])
                all_main_viewpoints = scenario_main_view.values[0]
            else:
                try:
                    all_main_viewpoints = os.listdir(f'{video_path}/{scenario}/vehicle_view/')
                except:
                    continue

            scenario_caption_path = f'{caption_path}/{scenario}/{view}/{scenario}_caption.json'

            try:
                scenario_caption_dict = json.load(open(scenario_caption_path))['event_phase'] 
            except:
                # print(f"Error loading {scenario_caption_path}")
                continue

            fps = 30.0 # Approximate FPS of all scenarios in WTS dataset
            # take first frame of each phase
            for camera_name in all_main_viewpoints:
                if camera_name is None or type(camera_name) == float:
                    continue

                camera_name = os.path.splitext(camera_name)[0]
                if normal_trimmed:
                    scenario_ped_bbox_path = f'{bbox_path}/bbox_annotated/pedestrian/{split}/normal_trimmed/{scenario}/{view}/{camera_name}_bbox.json'
                    scenario_veh_bbox_path = f'{bbox_path}/bbox_annotated/vehicle/{split}/normal_trimmed/{scenario}/{view}/{camera_name}_bbox.json'
                else:
                    scenario_ped_bbox_path = f'{bbox_path}/bbox_annotated/pedestrian/{split}/{scenario}/{view}/{camera_name}_bbox.json'
                    scenario_veh_bbox_path = f'{bbox_path}/bbox_annotated/vehicle/{split}/{scenario}/{view}/{camera_name}_bbox.json'

                # bbox file may not exist
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

def extract_bdd_bbox_annotated(split, video_path, anno_path, caption_path):
    """
    Extract annotated bounding boxes from BDD dataset.

    Args:
        split (str): Dataset split to process (train, val)
        video_path (str): Path to the video files
        anno_path (str): Path to the bounding box annotations
        caption_path (str): Path to the caption annotations
    Returns:
        dict: A dictionary containing bounding box annotations for each video and phase

    Example of res_bbox_dict format:
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

        # vehicle_view may not exist
        try:
            scenario_caption_dict = json.load(open(scenario_caption_path))
        except:
            # print(f"Error loading {scenario_caption_path}")
            continue

        fps = float(scenario_caption_dict['fps'])
        scenario_caption_dict = scenario_caption_dict['event_phase'] 
        
        scenario_ped_bbox_path = anno_path + f"/bbox_annotated/{split}/{scenario.replace('.mp4','_bbox.json')}"

        # bbox file may not exist
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
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip", help='Data directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Directory for saving json file')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # WTS
    video_path = args.data_path + f'/videos/{args.split}'
    bbox_path = args.data_path + f'/annotations'
    caption_path = args.data_path + f'/annotations/caption/{args.split}'
    wts_bbox_dict = extract_wts_bbox_annotated(args.split, args.data_path, video_path, bbox_path, caption_path, normal_trimmed=False)
    wts_bbox_dict.update(extract_wts_bbox_annotated(args.split, args.data_path, video_path, bbox_path, caption_path, normal_trimmed=True))

    wts_output_path = os.path.join(args.output_path, f'wts_{args.split}_annotated_bbox.json')
    with open(wts_output_path, 'w') as f:
        f.write(json.dumps(wts_bbox_dict, indent=4))
    print(f"WTS {args.split} annotated bounding boxes extracted and saved to {wts_output_path}")

    # BDD
    args.data_path += '/external/BDD_PC_5K'
    video_path = args.data_path + f'/videos/{args.split}'
    bbox_path = args.data_path + f'/annotations'
    caption_path = args.data_path + f'/annotations/caption/{args.split}'
    bdd_bbox_dict = extract_bdd_bbox_annotated(args.split, video_path, bbox_path, caption_path)

    bdd_output_path = os.path.join(args.output_path, f'bdd_{args.split}_annotated_bbox.json')
    with open(bdd_output_path, 'w') as f:
        f.write(json.dumps(bdd_bbox_dict, indent=4))
    print(f"BDD {args.split} annotated bounding boxes extracted and saved to {bdd_output_path}")
