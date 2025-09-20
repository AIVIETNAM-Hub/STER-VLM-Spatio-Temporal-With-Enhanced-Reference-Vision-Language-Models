import os
import json
import cv2
from tqdm import tqdm
import argparse
import math
import re

bbox_anno_dict = {}
gaze_anno_dict = {}

# --------- Utility functions ---------
def get_frame(video_path, frame_number):
    """
    Extract a specific frame from a video file at a given frame number

    Args:
        video_path (str): Path to the video file
        frame_number (int): The frame number to extract
    Returns:
        frame (ndarray or None): The extracted frame at the specified frame number, or None if the frame could not be read
    """
  
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    _, frame = cap.read()

    if frame is None:
        cap2 = cv2.VideoCapture(video_path)
        print(f"Frame {frame_number} not found in video {video_path}, trying to read from the beginning")
        cnt = 0

        while cnt < frame_number - 1:
            cnt += 1
            _, frame = cap2.read()

            if frame is None:
                print(f"Frame {frame_number} not found in video {video_path} at count {cnt}")
                cap2.release()
                cap.release()
                return None

        cap2.release()
        cap.release()
        if frame is None:
            print(f"Frame {frame_number} and cnt {cnt} not found in video {video_path}")
            cap2.set(cv2.CAP_PROP_POS_FRAMES, cnt-2)
            _, frame = cap2.read()  
            return frame

    cap.release()
    return frame

def get_phase_frame_numbers(filename):
    """ 
    Extracts the scenario name, phase, and frame number from the filename

    Args:
        filename (str): The filename to extract information from
    Returns:
        tuple: A tuple containing the scenario name, phase, and frame number
    """

    match = re.search(r'^(.*)_phase_(\d+)_frame_(\d+)\.jpg$', filename)
    if match:
        scenario_name = match.group(1)
        phase = int(match.group(2))
        frame = int(match.group(3))

    return scenario_name, phase, frame

def draw_line(img, x_head, y_head, x, y, z, scale=100):
    """
    Draw a line in the image based on the 3D coordinates

    Args:
        img (numpy.ndarray): The image on which to draw
        x_head (int): X coordinate of the head position
        y_head (int): Y coordinate of the head position
        x (float): X coordinate of the gaze point in 3D space
        y (float): Y coordinate of the gaze point in 3D space
        z (float): Z coordinate of the gaze point in 3D space
        scale (int): Scale factor for the gaze line length
    Returns:
        numpy.ndarray: The image with the gaze line drawn
    """
    if z == 0:
        return img
    dx = x * scale / abs(z)
    dy = y * scale / abs(z)

    # If the gaze line is too long, scale it down
    if abs(dx) * 4 >= img.shape[0] or abs(dy) * 4 >= img.shape[1]:
        scale = math.min(img.shape[0] / abs(dx) / 4, img.shape[1] / abs(dy) / 4)

        dx *= scale
        dy *= scale

    end_x = int(round(x_head + dx))
    end_y = int(round(y_head - dy))

    start = (int(round(x_head)), int(round(y_head)))
    end = (end_x, end_y)

    cv2.arrowedLine(img, start, end, (255, 0, 255), 3, tipLength=0.1)
    return img

# --------- Drawing bounding boxes for SPATIAL CAPTIONS ---------
def draw_bbox_annotated(split, video_path, output_path, normal_trimmed=False):
    """
    Draw bounding boxes on frames for spatial caption and save to output path

    Args:
        split (str): The dataset split (train, val, test)
        video_path (str): Path to the video directory
        output_path (str): Path to save the output images with bounding boxes
        normal_trimmed (bool): If True, process only normal trimmed videos
    Returns:
        None
    """
    if normal_trimmed:
        video_path = video_path + '/normal_trimmed/'
        scenario_list = list(filter(lambda x: 'normal' in x, list(bbox_anno_dict.keys())))
    else:
        scenario_list = list(filter(lambda x: 'normal' not in x, list(bbox_anno_dict.keys())))

    for scenario in tqdm(scenario_list, total=len(scenario_list)):
        for camera_name in bbox_anno_dict[scenario]:
            view = 'overhead_view' if 'vehicle_view' not in camera_name else 'vehicle_view'
            view_video_path = video_path + f'{scenario}/{view}'
            for phase in bbox_anno_dict[scenario][camera_name]:
                start_frame = bbox_anno_dict[scenario][camera_name][phase]['frame_id']
                img = get_frame(os.path.join(view_video_path, camera_name + '.mp4'), start_frame)
                # print(os.path.join(view_video_path, camera_name + '.mp4'))
                # print(f'Processing {scenario}/{view}/{camera_name}.mp4 | phase {phase} | frame {start_frame}')
                
                ped_bbox = bbox_anno_dict[scenario][camera_name][phase]['ped_bbox']
                if ped_bbox:
                    x = int(ped_bbox[0])
                    y = int(ped_bbox[1])
                    w = int(ped_bbox[2])
                    h = int(ped_bbox[3])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


                veh_bbox = bbox_anno_dict[scenario][camera_name][phase]['veh_bbox']
                if veh_bbox:
                    x = int(veh_bbox[0])
                    y = int(veh_bbox[1])
                    w = int(veh_bbox[2])
                    h = int(veh_bbox[3])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if normal_trimmed:
                    if not os.path.exists(output_path + f'/{split}/normal_trimmed/{scenario}'):
                        os.makedirs(output_path + f'/{split}/normal_trimmed/{scenario}')
                    try:
                        cv2.imwrite(output_path + f'/{split}/normal_trimmed/{scenario}/{camera_name}_phase{phase}_bbox.jpg', img)
                    except Exception as e:
                        print(f'Error writing image: {output_path}/{split}/normal_trimmed/{scenario}/{camera_name}_phase{phase}_bbox.jpg')
                        print(e)
                else:
                    if not os.path.exists(output_path + f'/{split}/{scenario}'):
                        os.makedirs(output_path + f'/{split}/{scenario}')
                    try:
                        cv2.imwrite(output_path + f'/{split}/{scenario}/{camera_name}_phase{phase}_bbox.jpg', img)
                    except Exception as e:
                        print(f'Error writing image: {output_path}/{split}/{scenario}/{camera_name}_phase{phase}_bbox.jpg')
                        print(e)

# --------- Drawing bounding boxes for not TEMPORAL CAPTIONS ---------
def draw_bbox_generated(split, video_path, output_path, normal_trimmed=False):
    """
    Draw bounding boxes on frames for temporal caption and save to output path

    Args:
        split (str): The dataset split (train, val, test)
        video_path (str): Path to the video directory
        output_path (str): Path to save the output images with bounding boxes
        normal_trimmed (bool): If True, process only normal trimmed videos
    Returns:
        Folders with images containing bounding boxes for each video and camera view
    """

    if normal_trimmed:
        video_path = video_path + '/normal_trimmed/'
        scenario_list = list(filter(lambda x: 'normal' in x, list(bbox_anno_dict_generated.keys())))
    else:
        scenario_list = list(filter(lambda x: 'normal' not in x, list(bbox_anno_dict_generated.keys())))

    for scenario in tqdm(scenario_list, total=len(scenario_list)):
        for camera_name in bbox_anno_dict_generated[scenario]:
            view = 'overhead_view' if 'vehicle_view' not in camera_name else 'vehicle_view'
            view_video_path = video_path + f'{scenario}/{view}'
            phase0 = 1e15
            for phase in bbox_anno_dict_generated[scenario][camera_name]:
                for item in bbox_anno_dict_generated[scenario][camera_name][phase]:
                    if 'vehicle_view' not in camera_name:
                        phase0 = min(phase0, item['frame_id'])

            for phase in bbox_anno_dict_generated[scenario][camera_name]:
                for item in bbox_anno_dict_generated[scenario][camera_name][phase]:
                    start_frame = item['frame_id']
                    img = get_frame(os.path.join(view_video_path, camera_name + '.mp4'), start_frame)

                    # load gaze and head data
                    
                    
                    # draw bbox
                    ped_bbox = item['ped_bbox']
                    if ped_bbox:
                        x = int(ped_bbox[0])
                        y = int(ped_bbox[1])
                        w = int(ped_bbox[2])
                        h = int(ped_bbox[3])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    veh_bbox = item['veh_bbox']
                    if veh_bbox:
                        x = int(veh_bbox[0])
                        y = int(veh_bbox[1])
                        w = int(veh_bbox[2])
                        h = int(veh_bbox[3])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Find the gaze and head data for the specific frame
                    frame_gaze, frame_head = None, None
                    try:
                        for item in gaze_anno_dict[scenario][camera_name]:
                            if item['frame_id'] == start_frame:
                                frame_gaze = item['gaze']
                                frame_head = item['head']
                                break
                    except Exception as e:
                  #      print(f'Error: {e}')
                        pass
                            
                    # draw head
                    try:
                        if frame_head is not None:
                            curr_bbox_head = frame_head['head']
                            cur_x = int(curr_bbox_head[0]) 
                            cur_y = int(curr_bbox_head[1]) 
                        elif frame_gaze is not None:
                            cur_x = int(x + w / 2)
                            cur_y = int(y + h / 2)
                    except Exception as e:
                        #print(f"This scenario {head_file} does not contains head")
                        pass

                    # draw gaze
                    contain_gaze = False
                    try: 
                        if frame_gaze is not None:
                            x = float(frame_gaze['gaze'][0])
                            y = float(frame_gaze['gaze'][1])
                            z = float(frame_gaze['gaze'][2])
                            img = draw_line(img, cur_x, cur_y, x, y, z)
                            contain_gaze = True
                            #cv2.putText(img, f'Gaze: ({x}, {y}, {z})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        #print(f"This {e} scenario {gaze_file_path} does not contains gaze information")
                        pass

                    if contain_gaze:
                        print(f"Contain gaze for {scenario}/{camera_name} phase {phase} | frame {start_frame})")
                        cv2.circle(img, (cur_x, cur_y), 5, (0, 255, 255), -1)

                    if normal_trimmed:
                        if not os.path.exists(output_path + f'/{split}/normal_trimmed/{scenario}'):
                            os.makedirs(output_path + f'/{split}/normal_trimmed/{scenario}')
                        try:
                            cv2.imwrite(output_path + f'/{split}/normal_trimmed/{scenario}/{camera_name}_phase{phase}_bbox_{start_frame}.jpg', img)
                        except Exception as e:
                            print(f'Error writing image: {output_path}/{split}/normal_trimmed/{scenario}/{camera_name}_phase{phase}_bbox_{start_frame}.jpg')
                            print(e)
                    else:
                        if not os.path.exists(output_path + f'/{split}/{scenario}'):
                            os.makedirs(output_path + f'/{split}/{scenario}')
                        try:
                            cv2.imwrite(output_path + f'/{split}/{scenario}/{camera_name}_phase{phase}_bbox_{start_frame}.jpg', img)
                        except Exception as e:
                            print(f'Error writing image: {output_path}/{split}/{scenario}/{camera_name}_phase{phase}_bbox_{start_frame}.jpg')
                            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split to process (train, val)')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip", help='Data directory path')
    parser.add_argument('--bbox_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Extracted bbox json directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Output path for two folders: drawn_bbox_annotated and drawn_bbox_generated')

    args = parser.parse_args()

    # ------ draw bbox for SPATIAL CAPTIONS ------

    output_path_annotated = os.path.join(args.output_path, 'drawn_bbox_annotated')
    if not os.path.exists(output_path_annotated):
        os.makedirs(output_path_annotated)
    try:
        bbox_anno_dict = json.load(open(args.bbox_path + f'/wts_{args.split}_annotated_bbox.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)

    video_path = args.data_path + f'/videos/{args.split}/'
    draw_bbox_annotated(args.split, video_path, output_path_annotated)
    draw_bbox_annotated(args.split, video_path, output_path_annotated, normal_trimmed=True)


    # ------ draw bbox and 3d gaze for TEMPORAL CAPTIONS ------

    output_path_generated = os.path.join(args.output_path, 'drawn_bbox_generated')
    video_path = args.data_path + f'/videos/{args.split}/'

    if not os.path.exists(output_path_generated):
        os.makedirs(output_path_generated)
    try:
        bbox_anno_dict_generated = json.load(open(args.bbox_path + f'/wts_{args.split}_generated_bbox.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)


    try: 
        gaze_anno_dict = json.load(open(args.bbox_path + f'/wts_{args.split}_gaze.json'))
    except Exception as e:
        print(f'Error: {e}')

    # draw_bbox_generated(args.split, video_path, output_path_generated)
    # draw_bbox_generated(args.split, video_path, output_path_generated, normal_trimmed=True)

