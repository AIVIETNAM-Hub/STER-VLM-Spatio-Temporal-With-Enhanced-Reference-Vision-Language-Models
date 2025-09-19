import os
import json
import cv2
import ffmpeg
from tqdm import tqdm
import argparse

bbox_anno_dict = {}

def check_rotation(path_video_file):
    """
    Check the rotation of a video file using ffmpeg metadata

    Args:
        path_video_file (str): Path to the video file.
    Returns:
        rotateCode (int or None): The rotation code if the video is rotated, otherwise None.
    """
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the value
    # we are looking for
    rotateCode = None
    if 'rotate' not in meta_dict['streams'][0]['tags']:
        return rotateCode
    
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode

def correct_rotation(frame, rotateCode):
    """
    Rotate the image frame based on the provided rotation code

    Args:
        frame (ndarray): The image frame to be rotated
        rotateCode (int or None): The rotation code to apply (e.g., cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, etc.)
    Returns:
        frame (ndarray): The rotated image frame

    If rotateCode is None, apply a 180-degree rotation twice to fix frames with wrong dimensions
    """
    if rotateCode is not None:
        frame = cv2.rotate(frame, rotateCode)
    else:
        frame = cv2.rotate(frame, cv2.ROTATE_180) 
        frame = cv2.rotate(frame, cv2.ROTATE_180) 
    return frame

def get_frame(video_path, start_time):
    """
    Extract a specific frame from a video file at a given time

    Args:
        video_path (str): Path to the video file
        start_time (float): The start time in seconds
    Returns:
        frame (ndarray or None): The extracted frame at the specified time, or None if the frame could not be read
    
    Try to read a specific frame directly from a video file at a given start time
    If the direct read fails, iterates through the video frames to find the frame
    """
    # Try direct frame read
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(start_time*fps)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    rotateCode = check_rotation(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read() 
    
    if not ret:
        cap.release()
        # Loop through video to find the frame
        # print(f'Processing video {video_path} / {frame_number} / {start_time}')
        cap2 = cv2.VideoCapture(video_path)
        total_frame = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_number - 1):
            ret, frame = cap2.read()
            if not ret:
                print(f"Error: Could not read frame {frame_number} from {video_path} {i}/{total_frame} frames read")
                cap2.release()
                return None
        if frame is None:
            print(f"Error: Frame {frame_number} is None in {video_path}")
            cap2.release()
            return None
        frame = correct_rotation(frame, rotateCode)
        cap2.release()
        return frame
    
    frame = correct_rotation(frame, rotateCode)
    cap.release()
    cv2.destroyAllWindows()
    return frame

# --------- Drawing bounding boxes for SPATIAL CAPTIONS ---------
def draw_bbox_annotated(split, video_path, output_path):
    video_list = list(bbox_anno_dict.keys())
    for video in tqdm(video_list):
        video_name = os.path.splitext(video)[0]
        view_video_path = video_path + f'{video_name}'
        # print(view_video_path + '.mp4')
        for phase in bbox_anno_dict[video]:
            start_frame = bbox_anno_dict[video][phase]['frame_id']
            start_time = bbox_anno_dict[video][phase]['start_time']
            
            img = get_frame(view_video_path + '.mp4', start_time)

            if img is None:
                print(f"Error: Could not get frame {start_frame} from {view_video_path}.mp4")
                continue
            # print(f'Processing {view_video_path}.mp4 | phase {phase} | frame {start_frame}')
            
            bbox = bbox_anno_dict[video][phase]['ped_bbox']
            if bbox:
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # print(bbox_anno_dict[video][phase])
            else:
                print(f'No bounding box found for {video_name} phase {phase}')

            if not os.path.exists(output_path + f'/external/{split}/{video_name}'):
                os.makedirs(output_path + f'/external/{split}/{video_name}')

            try:
                cv2.imwrite(output_path + f'/external/{split}/{video_name}/{video_name}_phase{phase}_bbox.jpg', img)
            except Exception as e:
                print(f'Error writing image: {output_path}/external/{split}/{video_name}/{video_name}_phase{phase}_bbox.jpg')
                print(f'Error: {e}')
                return None
        
# --------- Drawing bounding boxes for TEMPORAL CAPTIONS ---------
def draw_bbox_generated(split, video_path, output_path):
    video_list = list(bbox_anno_dict_not_fixed.keys())
    for video in tqdm(video_list):
        video_name = os.path.splitext(video)[0]
        view_video_path = video_path + f'{video_name}'
        # print(view_video_path + '.mp4')
        for phase in bbox_anno_dict_not_fixed[video]:
            lists = bbox_anno_dict_not_fixed[video][phase]
            for data in lists:
                time = data['time']
                # fps = data['fps']
                frame = data['frame_id']
                bbox = data['ped_bbox']
                # print(f'Processing {view_video_path}.mp4 | phase {phase} | frame {frame} | time {time}')
                # print("bbox: ", bbox)
                img = get_frame(view_video_path + '.mp4', time)

                if img is None:
                    print(f"Error: Could not get frame {frame} from {view_video_path}.mp4")
                    continue
                # print(f'Processing {view_video_path}.mp4 | phase {phase} | frame {start_frame}')
                
                if bbox:
                    x = int(bbox[0])
                    y = int(bbox[1])
                    w = int(bbox[2])
                    h = int(bbox[3])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if not os.path.exists(output_path + f'/external/{split}/{video_name}'):
                    os.makedirs(output_path + f'/external/{split}/{video_name}')

                try:
                    cv2.imwrite(output_path + f'/external/{split}/{video_name}/{video_name}_phase{phase}_bbox_{frame}.jpg', img)
                except Exception as e:
                    print(f'Error writing image: {output_path}/external/{split}/{video_name}/{video_name}_phase{phase}_bbox_{frame}.jpg')
                    print(f'Error: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'], help='Dataset split to process (train, val)')
    parser.add_argument('--data_path', type=str, default="/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_zip", help='Data directory path')
    parser.add_argument('--bbox_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Extracted bbox json directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned", help='Output path for two folders: drawn_bbox_annotated and drawn_bbox_generated')

    args = parser.parse_args()
    args.data_path += f'/external/BDD_PC_5K'

    # ------ draw bbox for SPATIAL CAPTIONS ------

    output_path_fixed = os.path.join(args.output_path, 'drawn_bbox_annotated')
    if not os.path.exists(output_path_fixed):
        os.makedirs(output_path_fixed)
    try:
        bbox_anno_dict = json.load(open(args.bbox_path + f'/bdd_{args.split}_annotated_bbox.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
    video_path = args.data_path + f'/videos/{args.split}/'
    draw_bbox_annotated(args.split, video_path, output_path_fixed)

    # ------ draw bbox for not TEMPORAL CAPTIONS ------

    output_path_not_fixed = os.path.join(args.output_path, 'drawn_bbox_generated')
    if not os.path.exists(output_path_not_fixed):
        os.makedirs(output_path_not_fixed)
    try:
        bbox_anno_dict_not_fixed = json.load(open(args.bbox_path + f'/bdd_{args.split}_generated_bbox.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
    video_path = args.data_path + f'/videos/{args.split}/'
    draw_bbox_generated(args.split, video_path, output_path_not_fixed)
    print(f'Finished drawing bbox for {args.split} split.')

