import os
import json
from tqdm import tqdm
import argparse
import ffmpeg
import pandas as pd

def get_bbox(ped_bbox_file, veh_bbox_file, frame_number):
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


def extract_info_frame(ped_bbox_file, veh_bbox_file, caption_folder, scene_name, video_view):
    bbox = {}

    # read caption object phase by phase
    caption_file = f"{caption_folder}/{scene_name}/{video_view}/{scene_name}_caption.json"
    if not os.path.exists(caption_file):
        print(f"Caption file not found: {caption_file}")
        return bbox
    with open(caption_file, 'r') as f:
        caption_obj = json.load(f)['event_phase']

    for phase in caption_obj:
        phase_id = phase['labels'][0]
        start_time = float(phase['start_time'])
        end_time = float(phase['end_time'])
        # extract 3 frames for each phase
        for i in range(3):
            frame_number = int((start_time + (end_time - start_time) * (i / 3.0))*30.0) # assuming 30 FPS

            ped_bbox, veh_bbox = get_bbox(ped_bbox_file, veh_bbox_file, frame_number)
            if bbox.get(phase_id) is None:
                bbox[phase_id] = []
            bbox[phase_id].append({
                "frame_id": frame_number,
                "ped_bbox": ped_bbox,
                "veh_bbox": veh_bbox,
            })
    return bbox

def extract_wts_bbox_generated(data_path, normal_trimmed):
    video_folder = os.path.join(data_path, "WTS_DATASET_PUBLIC_TEST/videos/test/public")
    ped_bbox_folder = os.path.join(data_path, "WTS_DATASET_PUBLIC_TEST_BBOX/annotations/bbox_generated/pedestrian/test/public")
    veh_bbox_folder = os.path.join(data_path, "WTS_DATASET_PUBLIC_TEST_BBOX/annotations/bbox_generated/vehicle/test/public")
    caption_folder = os.path.join(data_path, "WTS_DATASET_PUBLIC_TEST/annotations/caption/test/public_challenge")

    # load ref_main_viewpoint_file
    res_bbox_dict = {}
    if normal_trimmed:
        video_folder = f"{video_folder}/normal_trimmed"
        ped_bbox_folder = f"{ped_bbox_folder}/normal_trimmed"
        veh_bbox_folder = f"{veh_bbox_folder}/normal_trimmed"
        caption_folder = f"{caption_folder}/normal_trimmed"

    # get all subfolders in the video folder
    scene_names = [os.path.basename(f.path) for f in os.scandir(f"{video_folder}") if f.is_dir()]
    for scene_name in tqdm(scene_names, desc=f"Processing:"):
        # normal_trimmed skipped
        if scene_name == "normal_trimmed":
            continue

        # overhead view
        views = ["overhead_view", "vehicle_view"]
        for view in views:
            view_folder = f"{video_folder}/{scene_name}/{view}"
            if os.path.exists(view_folder):
                video_names = [os.path.basename(f).replace(".mp4","") for f in os.listdir(view_folder) if f.endswith('.mp4')]
                for video_name in video_names:
                    # print(f"Extracting frames for: \n Video: {video_name} \n Scene: {scene_name} \n View: {view}")
                    # prepare paths
                    video_file_path = f"{video_folder}/{scene_name}/{view}/{video_name}.mp4"
                    ped_bbox_file = f"{ped_bbox_folder}/{scene_name}/{view}/{video_name}_bbox.json"
                    veh_bbox_file = f"{veh_bbox_folder}/{scene_name}/{view}/{video_name}_bbox.json"

                    # extract frame info from video
                    bbox = extract_info_frame(ped_bbox_file, veh_bbox_file, caption_folder, scene_name, view)
                    video_name = os.path.basename(video_file_path)
                    video_name = video_name.replace('.mp4', '')
                    if res_bbox_dict.get(scene_name) is None:
                        res_bbox_dict[scene_name] = {}
                    res_bbox_dict[scene_name][video_name] = bbox
    return res_bbox_dict

def extract_bdd_bbox_generated(data_path):
    '''
    res_bbox_dict:

    {
        "video1004.mp4": {
            "0": [
                    {
                        "fps": 30.0,
                        "frame_id": 189,
                        "time": 6.326,
                        "bbox": [x, y, w, h],
                    },
                    {
                        "fps": 30.0,
                        "frame_id": 223,
                        "time": 7.757,
                        "bbox": [x, y, w, h],
                    },
                    {
                        "fps": 30.0,
                        "frame_id": 236,
                        "time": 7.891,
                        "bbox": [x, y, w, h],
                    }
                ]
            "1": [
                ...
            ]
        
            ...
        },
        ...
    }
    '''

    video_path = data_path + f'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public'
    caption_path = data_path + f'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/annotations/caption/test/public_challenge'
    anno_path = data_path + f'WTS_DATASET_PUBLIC_TEST_BBOX/external/BDD_PC_5K/annotations'
    
    res_bbox_dict = {}
    for video in tqdm(os.listdir(video_path)):
        video_caption_path = caption_path + f"/{video.replace('.mp4', '')}_caption.json"

        # vehicle_view may not exist
        try:
            video_caption_dict = json.load(open(video_caption_path))
        except:
            # print(f"Error loading {video_caption_path}")
            continue

        # Some videos have wrong fps, so we use the fps from the video file
        video_file_path = os.path.join(video_path, video)
        try:
            probe = ffmpeg.probe(video_file_path)
            fps = float(probe['streams'][0]['avg_frame_rate'].split('/')[0]) / float(probe['streams'][0]['avg_frame_rate'].split('/')[1])
        except ffmpeg.Error as e:
            print(f"Error probing video {video_file_path}: {e}")
            continue

        fake_fps = float(video_caption_dict['fps'])  

        video_caption_dict = video_caption_dict['event_phase'] 
        
        video_ped_bbox_path = anno_path + f"/bbox_generated/test/public/{video.replace('.mp4','_bbox.json')}"
        # bbox file may not exist
        ped_bbox_dict = None
        if os.path.exists(video_ped_bbox_path):
            ped_bbox_dict = json.load(open(video_ped_bbox_path))['annotations']

        video_caption_dict.sort(key=lambda x: float(x['start_time']))
        # sort by start time
        ped_bbox_dict = pd.DataFrame(ped_bbox_dict) if ped_bbox_dict is not None else None
        lists = []
        skip_video = False
        for i, phase in enumerate(video_caption_dict):
            st_time = float(phase['start_time'])
            ed_time = float(phase['end_time'])
            st_frame = max(int(round(st_time * fps)), 1)
            ed_frame = int(ed_time * fps)
            times = [st_time, st_time + (ed_time - st_time) / 3, st_time + 2 * (ed_time - st_time) / 3]
            
            frames = [st_frame, st_frame + (ed_frame - st_frame) // 3, st_frame + 2 * (ed_frame - st_frame) // 3]
            # phase may not have bbox (due to occlusion, out of view,...)
            cur_list = []
            if ed_frame - st_frame + 1 >= 3:
                for frame, time in zip(frames, times):
                    try:
                        ped_bbox = ped_bbox_dict.query("image_id == @frame") if len(ped_bbox_dict) > 0 else None
                    except KeyError:
                        print(f"Video {video} does not have bbox for the whole video. Please ignore this error because its original dataset fault")
                        ped_bbox = None
                        
                    if abs(fake_fps - fps) >= 2:
                        try:
                            # real_frame_in_bbox_284 / real_frame_in_video_142 ~ fake_caption_fps_59.04 / fps_30.09
                            halfFrame = int(frame * (fake_fps / fps))
                            ped_bbox = ped_bbox_dict.query("@halfFrame - 1 <= image_id and image_id <= @halfFrame + 1") if len(ped_bbox_dict) > 0 else None
                        except KeyError:
                            print(f"Video {video} does not have bbox for the whole video. Please ignore this error because its original dataset fault")
                            ped_bbox = None

                    ped_bbox = list(ped_bbox['bbox']) if ped_bbox is not None else None
                    ped_bbox = ped_bbox[len(ped_bbox)//2] if ped_bbox is not None and len(ped_bbox) > 0 else None
                    cur_list.append({
                        'fps': fps,
                        'time': time,
                        'frame_id': frame,
                        'ped_bbox': ped_bbox if ped_bbox else None,
                        'veh_bbox': None
                    })
            else:
                ed_frame_prev = int(float(video_caption_dict[i-1]['end_time']) * fps) if i-1 >= 0 else st_frame-2
                st_frame_nxt = int(float(video_caption_dict[i+1]['start_time']) * fps) if i+1 < len(video_caption_dict) else st_frame+2
                frames = [st_frame, st_frame+1, st_frame+2]
                if st_frame_nxt < st_frame+2:
                    frames = [st_frame-1, st_frame, st_frame+1]
                    if ed_frame_prev > st_frame-1:
                        frames = [st_frame, st_frame+1]
                        print(f"Video {video} has only 2 frames for phase {i}")
                        # skip_video = True
                        # break

                for frame, time in zip(frames, times):
                    try:
                        ped_bbox = ped_bbox_dict.query("image_id == @frame") if len(ped_bbox_dict) > 0 else None
                    except KeyError:
                        print(f"Video {video} does not have bbox for the whole video. Please ignore this error because its original dataset fault")
                        ped_bbox = None
                    ped_bbox = list(ped_bbox['bbox']) if ped_bbox is not None else None
                    ped_bbox = ped_bbox[0] if ped_bbox is not None and len(ped_bbox) > 0 else None
                    cur_list.append({
                        'fps': fps,
                        'time': time,
                        'frame_id': frame,
                        'ped_bbox': ped_bbox if ped_bbox else None,
                        'veh_bbox': None
                    })
            lists.append(cur_list)
                
        if not skip_video:
            for i in range(len(video_caption_dict)):
                if video not in res_bbox_dict:
                    res_bbox_dict[video] = {}

                if str(i) not in res_bbox_dict[video]:
                    res_bbox_dict[video][str(i)] = {}
                    
                res_bbox_dict[video][str(i)] = lists[i]
                
    return res_bbox_dict
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract bbox from video and save to json file")
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test", help='Path to the test dataset')
    parser.add_argument('--output_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test', help='Path to preprocessed data')

    args = parser.parse_args()
    if not os.path.exists(args.data_path + '/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/'):
        print(f"Data path {args.data_path} does not exist. Please check the path.")
        exit(1)
    os.makedirs(args.output_path, exist_ok=True)
    args.data_path = os.path.join(args.data_path, 'SubTask1-Caption/')

    wts_bbox_dict = extract_wts_bbox_generated(args.data_path, normal_trimmed=False)
    wts_bbox_dict.update(extract_wts_bbox_generated(args.data_path, normal_trimmed=True))

    wts_bbox_json_path = os.path.join(args.output_path, 'test_wts_bbox_generated.json')
    with open(wts_bbox_json_path, 'w') as f:
        json.dump(wts_bbox_dict, f, indent=4)
    print(f"Saved WTS bbox to {wts_bbox_json_path}")

    bdd_bbox_dict = extract_bdd_bbox_generated(args.data_path)

    bdd_bbox_json_path = os.path.join(args.output_path, 'test_bdd_bbox_generated.json')
    with open(bdd_bbox_json_path, 'w') as f:
        json.dump(bdd_bbox_dict, f, indent=4)
    print(f"Saved BDD bbox to {bdd_bbox_json_path}")
