import os
import json
import cv2
from tqdm import tqdm
import argparse
import ffmpeg

bbox_anno_dict = {}

def get_frame(video_path, frame_number):
    # Try direct frame read
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

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

def wts_draw_bbox_spatial(video_path, output_path, normal_trimmed=False):
    if normal_trimmed:
        video_path = video_path + '/normal_trimmed/'
        video_list = list(filter(lambda x: 'normal' in x, list(bbox_anno_dict.keys())))
    else:
        video_list = list(filter(lambda x: 'normal' not in x, list(bbox_anno_dict.keys())))
    
    for video in tqdm(video_list, total=len(video_list)):
        for camera_name in bbox_anno_dict[video]:
            view = 'overhead_view' if 'vehicle_view' not in camera_name else 'vehicle_view'
            view_video_path = video_path + f'{video}/{view}'
            for phase in bbox_anno_dict[video][camera_name]:
                start_frame = bbox_anno_dict[video][camera_name][phase]['frame_id']
                img = get_frame(os.path.join(view_video_path, camera_name + '.mp4'), start_frame)
                
                ped_bbox = bbox_anno_dict[video][camera_name][phase]['ped_bbox']
                if ped_bbox:
                    x = int(ped_bbox[0])
                    y = int(ped_bbox[1])
                    w = int(ped_bbox[2])
                    h = int(ped_bbox[3])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                veh_bbox = bbox_anno_dict[video][camera_name][phase]['veh_bbox']
                if veh_bbox:
                    x = int(veh_bbox[0])
                    y = int(veh_bbox[1])
                    w = int(veh_bbox[2])
                    h = int(veh_bbox[3])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if normal_trimmed:
                    if not os.path.exists(output_path + f'/normal_trimmed/{video}'):
                        os.makedirs(output_path + f'/normal_trimmed/{video}')
                    try:
                        cv2.imwrite(output_path + f'/normal_trimmed/{video}/{camera_name}_phase{phase}_bbox.jpg', img)
                    except Exception as e:
                        print(f'Error writing image: {output_path}/normal_trimmed/{video}/{camera_name}_phase{phase}_bbox.jpg')
                        print(e)
                else:
                    if not os.path.exists(output_path + f'/{video}'):
                        os.makedirs(output_path + f'/{video}')
                    try:
                        cv2.imwrite(output_path + f'/{video}/{camera_name}_phase{phase}_bbox.jpg', img)
                    except Exception as e:
                        print(f'Error writing image: {output_path}/{video}/{camera_name}_phase{phase}_bbox.jpg')
                        print(e)

def wts_draw_bbox_temporal(video_path, output_path, normal_trimmed=False):
    if normal_trimmed:
        video_path = video_path + '/normal_trimmed/' 
        
        video_list = list(filter(lambda x: 'normal' in x, list(bbox_anno_dict.keys())))
    else:
        video_list = list(filter(lambda x: 'normal' not in x, list(bbox_anno_dict.keys())))
    
    for video in tqdm(video_list, total=len(video_list)):
        for camera_name in bbox_anno_dict[video]:
            view = 'overhead_view' if 'vehicle_view' not in camera_name else 'vehicle_view'
            view_video_path = video_path + f'{video}/{view}'

            for phase in bbox_anno_dict[video][camera_name]:
                
                for bbox_info in bbox_anno_dict[video][camera_name][phase]:
                    img = get_frame(os.path.join(view_video_path, camera_name + '.mp4'), bbox_info['frame_id'])

                    if img is None:
                        print(f"Skipping {video}/{view}/{camera_name}.mp4 | phase {phase} | frame {bbox_info['frame_id']} due to read error")
                        continue

                    if bbox_info['ped_bbox']:
                        x  = int(bbox_info['ped_bbox'][0])
                        y  = int(bbox_info['ped_bbox'][1])
                        w  = int(bbox_info['ped_bbox'][2])
                        h  = int(bbox_info['ped_bbox'][3])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if bbox_info['veh_bbox']:
                        x  = int(bbox_info['veh_bbox'][0])
                        y  = int(bbox_info['veh_bbox'][1])
                        w  = int(bbox_info['veh_bbox'][2])
                        h  = int(bbox_info['veh_bbox'][3])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    if normal_trimmed:
                        if not os.path.exists(output_path + f'/normal_trimmed/{video}'):
                            os.makedirs(output_path + f'/normal_trimmed/{video}')
                        cv2.imwrite(os.path.join(output_path, 'normal_trimmed', video, f'{camera_name}_phase{phase}_bbox_{bbox_info["frame_id"]}.jpg'), img)
                    else:
                        if not os.path.exists(output_path + f'/{video}'):
                            os.makedirs(output_path + f'/{video}')
                        cv2.imwrite(os.path.join(output_path, video, f'{camera_name}_phase{phase}_bbox_{bbox_info["frame_id"]}.jpg'), img)

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
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
    if rotateCode is not None:
        frame = cv2.rotate(frame, rotateCode)
    else:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def get_frame_time(video_path, start_time):
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

def bdd_draw_bbox_spatial(video_path, output_path):
    video_list = list(bbox_anno_dict.keys())
    for video in tqdm(video_list):
        video_name = os.path.splitext(video)[0]
        view_video_path = video_path + f'{video_name}'
        # print(view_video_path + '.mp4')
        for phase in bbox_anno_dict[video]:
            start_frame = bbox_anno_dict[video][phase]['frame_id']
            start_time = bbox_anno_dict[video][phase]['start_time']
            
            img = get_frame_time(view_video_path + '.mp4', start_time)

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

            if not os.path.exists(output_path + f'/{video_name}'):
                os.makedirs(output_path + f'/{video_name}')
            
            try:
                cv2.imwrite(output_path + f'/{video_name}/{video_name}_phase{phase}_bbox.jpg', img)
            except Exception as e:
                print(f'Error writing image: {output_path}/{video_name}/{video_name}_phase{phase}_bbox.jpg')
                print(f'Error: {e}')

def bdd_draw_bbox_temporal(video_path, output_path):
    video_list = list(bbox_anno_dict.keys())
    for video in tqdm(video_list):
        video_name = os.path.splitext(video)[0]
        view_video_path = video_path + f'{video_name}'
        # print(view_video_path + '.mp4')
        for phase in bbox_anno_dict[video]:
            lists = bbox_anno_dict[video][phase]
            for data in lists:
                time = data['time']
                frame_number = data['frame_id']
                bbox = data['ped_bbox']
                img = get_frame_time(view_video_path + '.mp4', time)

                if img is None:
                    print(f"Error: Could not get frame {frame_number} from {view_video_path}.mp4")
                    continue
                
                if bbox:
                    x = int(bbox[0])
                    y = int(bbox[1])
                    w = int(bbox[2])
                    h = int(bbox[3])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if not os.path.exists(output_path + f'/{video_name}'):
                    os.makedirs(output_path + f'/{video_name}')
                
                try:
                    cv2.imwrite(output_path + f'/{video_name}/{video_name}_phase{phase}_bbox_{frame_number}.jpg', img)
                except Exception as e:
                    print(f'Error writing image: {output_path}/{video_name}/{video_name}_phase{phase}_bbox_{frame_number}.jpg')
                    print(f'Error: {e}')
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default= "/home/s48gb/Desktop/GenAI4E/Track2_Aicity/wts_dataset_test", help='Data directory path')
    parser.add_argument('--bbox_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test", help='Extracted bbox json directory path')
    parser.add_argument('--output_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/test", help='Output directory path')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    if not os.path.exists(args.data_path + '/SubTask1-Caption/WTS_DATASET_PUBLIC_TEST/'):
        print(f"Data path {args.data_path} does not exist. Please check the path.")
        exit(1)
    args.data_path = os.path.join(args.data_path, 'SubTask1-Caption')

    try:
        bbox_anno_dict = json.load(open(args.bbox_path + f'/test_wts_bbox_annotated.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)

    video_path = args.data_path + f'/WTS_DATASET_PUBLIC_TEST/videos/test/public/'
    output_path = os.path.join(args.output_path, 'bbox_annotated')  
    # wts_draw_bbox_spatial(video_path, output_path)
    # wts_draw_bbox_spatial(video_path, output_path, normal_trimmed=True)

    try:
        bbox_anno_dict = json.load(open(args.bbox_path + f'/test_wts_bbox_generated.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
    video_path = args.data_path + f'/WTS_DATASET_PUBLIC_TEST/videos/test/public/'
    output_path = os.path.join(args.output_path, 'bbox_generated')
    wts_draw_bbox_temporal(video_path, output_path)
    wts_draw_bbox_temporal(video_path, output_path, normal_trimmed=True)

    try:
        bbox_anno_dict = json.load(open(args.bbox_path + f'/test_bdd_bbox_annotated.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
    video_path = args.data_path + f'/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public/'
    output_path = os.path.join(args.output_path, 'bbox_annotated/external')
    # bdd_draw_bbox_spatial(video_path, output_path)

    try:
        bbox_anno_dict = json.load(open(args.bbox_path + f'/test_bdd_bbox_generated.json'))
    except Exception as e:
        print(f'Error: {e}')
        exit(1)
    video_path = args.data_path + f'/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public/'
    output_path = os.path.join(args.output_path, 'bbox_generated/external')
    bdd_draw_bbox_temporal(video_path, output_path)

