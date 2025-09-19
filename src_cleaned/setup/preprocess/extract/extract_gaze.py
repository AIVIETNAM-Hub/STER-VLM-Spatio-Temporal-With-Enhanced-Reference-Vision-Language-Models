import os
import json
import cv2
from tqdm import tqdm
import argparse

def extract_wts_gaze(data_path, split):
    """
    Extracts 3d gaze from the given scenario and gaze paths for the WTS dataset

    Args:
        gaze_path (str): Path to the directory containing gaze annotations
        normal_trimmed (bool): Boolean indicating whether to use the normal trimmed version of the data
    Returns:
        res_gaze_dict (dict): A dictionary containing the extracted gaze data in a structured format

    Example of the output format for res_gaze_dict:
    {
        "20230707_17_CY23_T1": {
            "20230707_17_CY23_T1_4_camera": [
                {
                    "head" : [120, 120],
                    "gaze" : [0.5, 0.5],
                    "frame_id": 120
                }
                ...
            ],
            ...
        }
    }
    """
    res_gaze_dict = {}
    video_path = data_path + f'/videos/{split}'
    gaze_path = data_path + f'/annotations/3D_gaze/{split}'
    caption_path = data_path + f'/annotations/caption/{split}'
    head_path = data_path + f'/annotations/head/{split}'

    for scenario in tqdm(os.listdir(gaze_path)):
        if scenario == 'normal_trimmed' or scenario == '.DS_Store': # Skip folder normal_trimmed (when os.listdir)
            continue
        res_gaze_dict[scenario] = {}
        for item in os.listdir(f'{gaze_path}/{scenario}'):
            if item == 'normal_trimmed' or item == '.DS_Store':  # Skip folder normal_trimmed
                continue

            camera_name = item.replace('_gaze.json', '')
            try:
                
                camera_gaze_path = f'{gaze_path}/{scenario}/{camera_name}_gaze.json'
                gaze_data = json.load(open(camera_gaze_path))['annotations']
            except:
                print(f"Error loading {camera_gaze_path}")
                continue

            try:
                camera_head_path = f'{head_path}/{scenario}/{camera_name.replace("-", "_")}_head.json'
                head_data = json.load(open(camera_head_path))['annotations']
            except:
                print(f"Error loading {camera_head_path, camera_gaze_path}") 
                continue
            try: 
                camera_caption_path = f'{caption_path}/{scenario}/overhead_view/{scenario}_caption.json'
                caption_data = json.load(open(camera_caption_path))['event_phase']
            except:
                print(f"Error loading {caption_path}")
                continue

            res_gaze_dict[scenario][camera_name.replace("-", "_")] = []
            caption_data.sort(key=lambda x: float(x['start_time']))
            phase0 = caption_data[0]['start_time'] if caption_data else None
            cap = cv2.VideoCapture(f'{video_path}/{scenario}/overhead_view/{camera_name.replace("-", "_")}.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            for gaze in gaze_data:
                
                frame_id = int(gaze['image_id'])
                head_coord =  None
                for head in head_data:
                    
                    if head['image_id'] <= int(frame_id + float(phase0) * fps) + 1:
                        head_coord = head['head']
                        break
                
                res_gaze_dict[scenario][camera_name.replace("-", "_")].append({
                    'head': head_coord,
                    'gaze': gaze['gaze'],
                    'frame_id': int(frame_id + float(phase0) * fps)
                })
    return res_gaze_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Split to process: train or val')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip", help='Data directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Directory for saving json file')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # WTS
    gaze_data_dict = extract_wts_gaze(args.data_path, args.split)

    with open(f'{args.output_path}/wts_{args.split}_gaze.json', 'w') as f:
        json.dump(gaze_data_dict, f, indent=4)
    print(f'WTS {args.split} gaze data extracted and saved to {args.output_path}/wts_{args.split}_gaze.json')