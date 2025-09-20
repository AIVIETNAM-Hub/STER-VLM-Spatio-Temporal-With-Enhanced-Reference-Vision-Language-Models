import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import ffmpeg

def get_bbox(ped_bbox_file, veh_bbox_file, frame_number):
    """
    Get pedestrian and vehicle bounding boxes for a specific frame number

    Args:
        ped_bbox_file (str): Path to the pedestrian bounding box file
        veh_bbox_file (str): Path to the vehicle bounding box file
        frame_number (int): The frame number to extract bounding boxes for
    Returns:
        ped_bbox (list or None): Bounding box for pedestrians in the format [x, y, width, height] or None if not found
        veh_bbox (list or None): Bounding box for vehicles in the format [x, y, width, height] or None if not found
    """
    # read bbox data from files
    # bbox format: [x, y, width, height]
    ped_bbox = None
    veh_bbox = None
    if os.path.exists(ped_bbox_file): 
        with open(ped_bbox_file, 'r') as f:
            bbox_data = json.load(f)
            bbox_data = bbox_data['annotations'] 

        for frame in bbox_data:
            if frame['image_id'] == frame_number:
                ped_bbox = frame['bbox']
                break

    if os.path.exists(veh_bbox_file):
        with open(veh_bbox_file, 'r') as f:
            bbox_data = json.load(f)
            bbox_data = bbox_data['annotations']

        for frame in bbox_data:
            if frame['image_id'] == frame_number:
                veh_bbox = frame['bbox']
                break
    
    return ped_bbox, veh_bbox

def extract_info_frame(ped_bbox_file, veh_bbox_file, video_view, caption_obj):
    """
    Extract bounding box information for a specific video view and caption object.
    Args:
        ped_bbox_file (str): Path to the pedestrian bounding box file
        veh_bbox_file (str): Path to the vehicle bounding box file
        video_view (str): The view of the video (e.g., "overhead_view", "vehicle_view")
        caption_obj (list): List of caption objects containing phase information
    Returns:
        bbox (dict): A dictionary containing bounding box information for each phase
    """
    #print(f"Extracting frames from {video_file} for view {video_view}...")
    bbox = {}

    # read caption object phase by phase
    for phase in caption_obj:
        if phase['view'] != video_view:
            continue
        phase_id = phase['labels']
        start_time = float(phase['start_time'])
        end_time = float(phase['end_time'])
        # extract 3 frames for each phase
        for i in range(3):
            frame_number = int((start_time + (end_time - start_time) * (i / 3.0))*30) # assuming 30 FPS

            ped_bbox, veh_bbox = get_bbox(ped_bbox_file, veh_bbox_file, frame_number)
            if bbox.get(phase_id) is None:
                bbox[phase_id] = []
            bbox[phase_id].append({
                "frame_id": frame_number,
                "ped_bbox": ped_bbox,
                "veh_bbox": veh_bbox,
            })

    return bbox

def extract_wts_bbox_generated(split, data_path, preprocessed_data_path, video_folder, normal_trimmed):
    """
    Extract generated bounding boxes for WTS dataset. 
    Args:
        split (str): The dataset split to process (e.g., "train", "val")
        data_path (str): Path to the WTS dataset root directory
        preprocessed_data_path (str): Path to the preprocessed data directory
        video_folder (str): Path to the folder containing video files
        normal_trimmed (bool): Whether to process the normal trimmed videos
    Returns:
        res_bbox_dict (dict): A dictionary containing bounding box information for each video and phase
    """
    res_bbox_dict = {}
    ped_bbox_folder = f"{data_path}/annotations/bbox_generated/pedestrian"
    veh_bbox_folder = f"{data_path}/annotations/bbox_generated/vehicle"

    # load main_viewpoint_csv
    df = pd.read_csv(f"{data_path}/view_used_as_main_reference_for_multiview_scenario.csv")
    caption_file = f"{preprocessed_data_path}/wts_{split}_fnf_processed.json"
    # load caption file
    with open(caption_file, 'r') as f:
        caption_data = json.load(f)

        split_name = split
        if normal_trimmed:
            split_name = f"{split}/normal_trimmed"

        # get all subfolders in the video folder
        scene_names = [os.path.basename(f.path) for f in os.scandir(f"{video_folder}/{split_name}") if f.is_dir()]
        for scene_name in tqdm(scene_names, desc=f"Processing split: {split_name}"):
            # Skip folder normal_trimmed (when os.listdir)
            if scene_name == "normal_trimmed":
                continue

            # overhead view
            views = ["overhead_view", "vehicle_view"]
            for view in views:
                view_folder = f"{video_folder}/{split_name}/{scene_name}/{view}"
                if os.path.exists(view_folder):
                    video_names = [os.path.basename(f).replace(".mp4","") for f in os.listdir(view_folder) if f.endswith('.mp4')]
                    
                    for video_name in video_names:
                        # check if video file is in the main viewpoint if it is in overhead_view (not apply to normal_trimmed)
                        if view == "overhead_view" and not normal_trimmed:
                            # check if video file is main viewpoint
                            row = df[df["Scene"] == scene_name]
                            viewpoint_cols = ["Viewpoint1", "Viewpoint2", "Viewpoint3", "Viewpoint4"]
                            found = any(f"{video_name}.mp4" == row[col].values[0] for col in viewpoint_cols if pd.notna(row[col].values[0]))

                            if not found:
                                continue
                        
                        # print(f"Extracting frames for: \n Video: {video_name} \n Scene: {scene_name} \n View: {view} \n Split: {split_name}")
                        # prepare paths
                        video_file_path = f"{video_folder}/{split_name}/{scene_name}/{view}/{video_name}.mp4"
                        ped_bbox_file = f"{ped_bbox_folder}/{split_name}/{scene_name}/{view}/{video_name}_bbox.json"
                        veh_bbox_file = f"{veh_bbox_folder}/{split_name}/{scene_name}/{view}/{video_name}_bbox.json"

                        # extract frame info from video
                        bbox = extract_info_frame(ped_bbox_file, veh_bbox_file, view, caption_data[scene_name])
                        video_name = os.path.basename(video_file_path)
                        video_name = video_name.replace('.mp4', '') # remove .mp4
                        if res_bbox_dict.get(scene_name) is None:
                            res_bbox_dict[scene_name] = {}
                        res_bbox_dict[scene_name][video_name] = bbox
    
    return res_bbox_dict
    
def extract_bdd_bbox_generated(split, video_path, anno_path, caption_path):
    """
    Extract generated bounding boxes for BDD dataset.
    Args:
        split (str): The dataset split to process (e.g., "train", "val")
        video_path (str): Path to the folder containing video files
        anno_path (str): Path to the folder containing annotation files
        caption_path (str): Path to the folder containing caption files
    Returns:
        res_bbox_dict (dict): A dictionary containing bounding box information for each video and phase
    
    res_bbox_dict:
    {
        "video1004.mp4": {
            "0": [
                    {
                        "fps": 30.0,
                        "frame_id": 189,
                        "time": 6.326,
                        "ped_bbox": [x, y, w, h],
                        "veh_bbox": None
                    },
                    {
                        "fps": 30.0,
                        "frame_id": 223,
                        "time": 7.757,
                        "ped_bbox": [x, y, w, h],
                        "veh_bbox": None
                    },
                    {
                        ...
                    },
                ]
            "1": [
                ...
            ]
        
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

        # Some scenarios have wrong fps, so we use the fps from the scenario itself
        scenario_file_path = os.path.join(video_path, scenario)
        try:
            probe = ffmpeg.probe(scenario_file_path)
            fps = float(probe['streams'][0]['avg_frame_rate'].split('/')[0]) / float(probe['streams'][0]['avg_frame_rate'].split('/')[1])
        except ffmpeg.Error as e:
            print(f"Error probing scenario {scenario_file_path}: {e}")
            continue
        
        # delete later
        fake_fps = float(scenario_caption_dict['fps'])  

        scenario_caption_dict = scenario_caption_dict['event_phase'] 
        
        scenario_ped_bbox_path = anno_path + f"/bbox_generated/{split}/{scenario.replace('.mp4','_bbox.json')}"

        # bbox file may not exist
        ped_bbox_dict = None
        if os.path.exists(scenario_ped_bbox_path):
            ped_bbox_dict = json.load(open(scenario_ped_bbox_path))['annotations']

        scenario_caption_dict.sort(key=lambda x: float(x['start_time']))
        # sort by start time
        ped_bbox_dict = pd.DataFrame(ped_bbox_dict) if ped_bbox_dict is not None else None
        lists = []
        skip_scenario = False
        for i, phase in enumerate(scenario_caption_dict):
            st_time = float(phase['start_time'])
            ed_time = float(phase['end_time'])
            st_frame = max(int(round(st_time * fps)), 1) # rounding error may occur
            ed_frame = int(ed_time * fps)

            # 3 frames for each phase
            times = [st_time, st_time + (ed_time - st_time) / 3, st_time + 2 * (ed_time - st_time) / 3]
            frames = [st_frame, st_frame + (ed_frame - st_frame) // 3, st_frame + 2 * (ed_frame - st_frame) // 3]
            cur_list = []
            # enough frames for the phase
            if ed_frame - st_frame + 1 >= 3: 
                for frame, time in zip(frames, times):
                    try:
                        ped_bbox = ped_bbox_dict.query("image_id == @frame") if len(ped_bbox_dict) > 0 else None
                    except KeyError:
                        print(f"scenario {scenario} does not have bbox for the whole scenario. Please ignore this error because its original dataset fault")
                        ped_bbox = None
                        
                    if abs(fake_fps - fps) >= 2:
                        try:
                            # real_frame_in_bbox_284 / real_frame_in_scenario_142 ~ fake_caption_fps_59.04 / fps_30.09
                            halfFrame = int(frame * (fake_fps / fps))
                            ped_bbox = ped_bbox_dict.query("@halfFrame - 1 <= image_id and image_id <= @halfFrame + 1") if len(ped_bbox_dict) > 0 else None
                        except KeyError:
                            print(f"scenario {scenario} does not have bbox for the whole scenario. Please ignore this error because its original dataset fault")
                            ped_bbox = None

                    ped_bbox = list(ped_bbox['bbox']) if ped_bbox is not None else None
                    ped_bbox = ped_bbox[len(ped_bbox)//2] if ped_bbox is not None and len(ped_bbox) > 0 else None # take middle bbox if there are multiple bboxes
                    cur_list.append({
                        'fps': fps,
                        'time': time,
                        'frame_id': frame,
                        'ped_bbox': ped_bbox if ped_bbox else None,
                        'veh_bbox': None
                    })
            else: # not enough frames, only 2 or 1 frames
                ed_frame_prev = int(float(scenario_caption_dict[i-1]['end_time']) * fps) if i-1 >= 0 else st_frame-2
                st_frame_nxt = int(float(scenario_caption_dict[i+1]['start_time']) * fps) if i+1 < len(scenario_caption_dict) else st_frame+2
                frames = [st_frame, st_frame+1, st_frame+2]
                if st_frame_nxt < st_frame+2:
                    frames = [st_frame-1, st_frame, st_frame+1]
                    if ed_frame_prev > st_frame-1:
                        frames = [st_frame, st_frame+1]
                        print(f"scenario {scenario} has only 2 frames for phase {i}")
                        # skip_scenario = True
                        # break

                for frame, time in zip(frames, times):
                    try:
                        ped_bbox = ped_bbox_dict.query("image_id == @frame") if len(ped_bbox_dict) > 0 else None
                    except KeyError:
                        print(f"scenario {scenario} does not have bbox for the whole scenario. Please ignore this error because its original dataset fault")
                        ped_bbox = None
                    ped_bbox = list(ped_bbox['bbox']) if ped_bbox is not None else None
                    ped_bbox = ped_bbox[0] if ped_bbox is not None and len(ped_bbox) > 0 else None # take first bbox if there are multiple bboxes (due to annotated error)
                    cur_list.append({
                        'fps': fps,
                        'time': time,
                        'frame_id': frame,
                        'ped_bbox': ped_bbox if ped_bbox else None,
                        'veh_bbox': None
                    })
            lists.append(cur_list)
                
        if not skip_scenario:
            for i in range(len(scenario_caption_dict)):
                if scenario not in res_bbox_dict:
                    res_bbox_dict[scenario] = {}

                if str(i) not in res_bbox_dict[scenario]:
                    res_bbox_dict[scenario][str(i)] = {}
                    
                res_bbox_dict[scenario][str(i)] = lists[i]
                
    return res_bbox_dict
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames with bounding boxes from videos")
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip/", help='Path to the dataset root directory')
    parser.add_argument('--preprocessed_data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset", help='Path to the preprocessed data directory')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Path to the output folder')
    
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # WTS
    video_path = args.data_path + f"/videos"
    wts_bbox_dict = extract_wts_bbox_generated(args.split, args.data_path, args.preprocessed_data_path, video_path, normal_trimmed=False)
    wts_bbox_dict.update(extract_wts_bbox_generated(args.split, args.data_path, args.preprocessed_data_path, video_path, normal_trimmed=True))

    wts_output_path = os.path.join(args.output_path, f'wts_{args.split}_generated_bbox.json')
    with open(wts_output_path, 'w') as f:
        json.dump(wts_bbox_dict, f, indent=4)
    print(f"Generated bounding boxes extracted and saved to {wts_output_path}")

    # BDD
    args.data_path += "/external/BDD_PC_5K"
    video_path = args.data_path + f'/videos/{args.split}'
    anno_path = args.data_path + f'/annotations'
    caption_path = args.data_path + f'/annotations/caption/{args.split}'
    bdd_bbox_dict = extract_bdd_bbox_generated(args.split, video_path, anno_path, caption_path)

    bdd_output_path = os.path.join(args.output_path, f'bdd_{args.split}_generated_bbox.json')
    with open(bdd_output_path, 'w') as f:
        json.dump(bdd_bbox_dict, f, indent=4)
    print(f"Generated bounding boxes extracted and saved to {bdd_output_path}")