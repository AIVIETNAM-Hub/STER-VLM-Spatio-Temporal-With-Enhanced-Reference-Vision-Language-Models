import os
import json

# --------------------------------- Spatial ---------------------------------
def find_best_image_for_appearance(image_path, bbox_json_path):
    """
    Find the best images for the appearance view based on the bounding box information

    Args:
        image_path (str): Path to the directory containing images
        bbox_json_path (str): Path to the bounding box JSON file
    Returns:
        res_image_list (list): List of selected image paths for the appearance view
    """
    res_image_path = ""

    all_images = os.listdir(image_path)
    try:
        json_bbox = json.load(open(bbox_json_path, 'r'))
    except FileNotFoundError:
        print(f"File {bbox_json_path} not found. Please check the path.")

    bbox_data = json_bbox[image_path.split('/')[-1]] if 'external' not in image_path else json_bbox[f"{image_path.split('/')[-1]}.mp4"]
    res_image_list = []
    best_view = None
    best_area = 0
    # Exist vehicle view, usually images in phase 3 or 4 are the clearest
    vehicle_view_images = [img for img in all_images if 'vehicle' in img]
    for img in vehicle_view_images:
        phase_number = img.split('_')[-2][-1]  # Get phase number from image name
        if phase_number < "3":
            continue
        camera_name = "_".join(img.split('_')[:-2])  # Get camera name without phase
        bbox = bbox_data[camera_name][phase_number] if camera_name in bbox_data and phase_number in bbox_data[camera_name] else None
        if bbox is not None:
            if 'ped_bbox' in bbox and bbox['ped_bbox'] != None:
                # Calculate bbox area
                _, _, width, height = bbox['ped_bbox']
                area = width * height
                if area > best_area:
                    best_area = area
                    best_view = img
    if best_view:
        res_image_list.append(f"{image_path}/{best_view}")

    # Select overhead view image with the clearest pedestrian bbox
    overhead_view_images = [img for img in all_images if 'vehicle' not in img]
    overhead_view_images.sort(key=lambda x: int(x.split('_')[-2][-1]))  # Sort by phase number
    res_image_path = ""
    for img in overhead_view_images:
        phase_number = img.split('_')[-2][-1]  # Get phase number from image name
        camera_name = "_".join(img.split('_')[:-2])  # Get camera name without phase
        bbox = bbox_data[camera_name][phase_number] if camera_name in bbox_data and phase_number in bbox_data[camera_name] else None
        if bbox is not None:
            if 'ped_bbox' in bbox and bbox['ped_bbox'] != None:
                # Calculate bbox area
                _, _, width, height = bbox['ped_bbox']
                area = width * height
                if area > best_area:
                    best_area = area
                    best_view = img
    if res_image_path:
        res_image_list.append(f"{image_path}/{best_view}")

    if len(res_image_list) < 2:
        # If not enough images, take additional phase 3 or 4 images from vehicle view or overhead view
        for img in overhead_view_images:
            if 'phase3' in img or 'phase4' in img:
                if f"{image_path}/{img}" not in res_image_list:
                    res_image_list.append(f"{image_path}/{img}")
                    if len(res_image_list) >= 2:
                        break
                    
    return res_image_list

def find_best_image_for_environment(image_path):
    """
    Find the best images for the environment view based on the bounding box information.

    Args:
        image_path (str): Path to the directory containing images.
    Returns:
        res_image_list (list): List of selected image paths for the environment view.
    """
    res_image_list = []
    
    all_images = os.listdir(image_path)
    vehicle_view_images = [img for img in all_images if 'vehicle' in img]
    vehicle_view_images.sort(key=lambda x: int(x.split('_')[-2][-1]))  # Sort by phase number
    if vehicle_view_images:
        # Exist vehicle view, usually images in phase 0 or 1 are the clearest
        for img in vehicle_view_images:
            if 'phase0' in img or 'phase1' in img:
                res_image_list.append(f"{image_path}/{img}")
                break

    overhead_view_images = [img for img in all_images if 'vehicle' not in img]
    overhead_view_images.sort(key=lambda x: int(x.split('_')[-2][-1]))  # Sort by phase number
    while len(res_image_list) < 2: # If not enough images, take phase 0 or 1 images since they are usually clearer
        for img in overhead_view_images:
            if 'phase0' in img or 'phase1' in img:
                if f"{image_path}/{img}" not in res_image_list:
                    res_image_list.append(f"{image_path}/{img}")
                    break 
    
    return res_image_list

# --------------------------------- Temporal ---------------------------------
def get_k_max_ped_bbox(scene_name, image_list, phase, bbox_path, k):
    '''
    Args:
        scene_name (str): Name of the scene, e.g., "230922_0000.."
        image_list (list): List of image file names in the scene folder
        phase (int): Phase number (0-4)
        bbox_path (str): Path to the bounding box JSON file
        k (int): Number of largest bounding boxes to return
    
    Returns:
        list: List of image file names with the largest pedestrian bounding boxes

    If no bounding boxes are found, returns None
    Else return k largest pedestrian bounding boxes
    '''
    if bbox_path is None:
        print(f"BBox not exist {bbox_path} for scene {scene_name}, return None")
        return None
    
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
        try:
            bbox = bbox[scene_name]
        except KeyError:
            print(f"Scene {scene_name} not found in bbox file {bbox_path}, return None")
            return None

    bbox_list = []
    if 'video' in scene_name:
        if str(phase) in bbox:
            phase_info = bbox[str(phase)]
            for bbox_info in phase_info:
                pedestrian_bbox = bbox_info.get('ped_bbox', None)
                frame_id = bbox_info.get('frame_id', None)
                if pedestrian_bbox is not None:
                    _, _, w, h = pedestrian_bbox
                    for img in image_list:
                        if frame_id is not None and f"phase{phase}_bbox_{frame_id}" in img:
                            bbox_list.append((img, w*h))
        else:
            print(f"Phase {phase} not found in bbox file {bbox_path} for scene {scene_name}, return None")
            return None
    else:
        for camera in bbox:
            if str(phase) in bbox[camera]:
                phase_info = bbox[camera][str(phase)]
                for bbox_info in phase_info:
                    pedestrian_bbox = bbox_info.get('ped_bbox', None)
                    frame_id = bbox_info.get('frame_id', None)
                    if pedestrian_bbox is not None:
                        _, _, w, h = pedestrian_bbox
                        for img in image_list:
                            if frame_id is not None and f"{camera}_phase{phase}_bbox_{frame_id}" in img:
                                bbox_list.append((img, w*h))
            else:
                print(f"Phase {phase} not found in bbox file {bbox_path} for scene {scene_name}, return None")
                return None
    # Sort the bounding boxes by area in descending order
    bbox_list.sort(key=lambda x: x[1], reverse=True)
    # Prioritise vehicle view images
    bbox_list = [img for img in bbox_list if 'vehicle' in img[0]] + [img for img in bbox_list if 'vehicle' not in img[0]]
    # Get array of images of k largest bounding boxes
    k_largest_bboxes = [img[0] for img in bbox_list[:k]]
    
    return k_largest_bboxes

def get_k_max_ped_veh_bbox(scene_name, image_list, phase, bbox_path, k):
    '''
    Args:
        scene_name (str): Name of the scene, e.g., "230922_0000.."
        image_list (list): List of image file names in the scene folder
        phase (int): Phase number (0-4)
        bbox_path (str): Path to the bounding box JSON file
        k (int): Number of largest bounding boxes to return
    
    Returns:
        list: List of image file names with the largest pedestrian and vehicle bounding boxes

    If no bounding boxes are found, returns None
    Else return k largest bounding boxes
    '''
    if bbox_path is None:
        print(f"BBox not exist {bbox_path} for scene {scene_name}, return None")
        return None
    
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
        try:
            bbox = bbox[scene_name]
        except KeyError:
            print(f"Scene {scene_name} not found in bbox file {bbox_path}, return None")
            return None

    bbox_list = []
    if 'video' in scene_name:
        if str(phase) in bbox:
            phase_info = bbox[str(phase)]
            for bbox_info in phase_info:
                pedestrian_bbox = bbox_info.get('ped_bbox', None)
                vehicle_bbox = bbox_info.get('veh_bbox', None)
                frame_id = bbox_info.get('frame_id', None)
                cur_area = 0
                if pedestrian_bbox is not None:
                    _, _, w, h = pedestrian_bbox
                    cur_area += w * h
                if vehicle_bbox is not None:
                    _, _, w, h = vehicle_bbox
                    cur_area += w * h
                if cur_area >= 0:
                    for img in image_list:
                        if frame_id is not None and f"phase{phase}_bbox_{frame_id}" in img:
                            bbox_list.append((img, cur_area))
        else:
            print(f"Phase {phase} not found in bbox file {bbox_path} for scene {scene_name}, return None")
            return None
    else:
        for camera in bbox:
            if str(phase) in bbox[camera]:
                phase_info = bbox[camera][str(phase)]
                for bbox_info in phase_info:
                    pedestrian_bbox = bbox_info.get('ped_bbox', None)
                    vehicle_bbox = bbox_info.get('veh_bbox', None)
                    frame_id = bbox_info.get('frame_id', None)
                    cur_area = 0
                    if pedestrian_bbox is not None:
                        _, _, w, h = pedestrian_bbox
                        cur_area += w * h
                    if vehicle_bbox is not None:
                        _, _, w, h = vehicle_bbox
                        cur_area += w * h
                    if cur_area >= 0:
                        for img in image_list:
                            if frame_id is not None and f"{camera}_phase{phase}_bbox_{frame_id}" in img:
                                bbox_list.append((img, cur_area))
            else:
                print(f"Phase {phase} not found in bbox file {bbox_path} for scene {scene_name}, return None")
                return None
                    
    # Sort the bounding boxes by area in descending order
    bbox_list.sort(key=lambda x: x[1], reverse=True)
    # Prioritise vehicle view images
    bbox_list = [img for img in bbox_list if 'vehicle' in img[0]] + [img for img in bbox_list if 'vehicle' not in img[0]]
    # Get array of images of k largest bounding boxes
    k_largest_bboxes = [img[0] for img in bbox_list[:k]]
    
    return k_largest_bboxes  

def get_group_max_ped_veh_bbox(scene_name, image_list, phase, bbox_path):
    '''
    Args:
        scene_name (str): Name of the scene, e.g., "230922_0000.."
        image_list (list): List of image file names in the scene folder
        phase (int): Phase number (0-4)
        bbox_path (str): Path to the bounding box JSON file
    
    Returns:
        list: List of image file names with the largest sum area of pedestrian and vehicle bounding boxes

    If no bounding boxes are found, returns None
    Else return list of images with largest bounding boxes of the same camera group
    '''
    if bbox_path is None:
        print(f"BBox not exist {bbox_path} for scene {scene_name}, return None")
        return None
    
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
        try:
            bbox = bbox[scene_name]
        except KeyError:
            print(f"Scene {scene_name} not found in bbox file {bbox_path}, return None")
            return None

    bbox_list = []
    if 'video' in scene_name:
        if str(phase) in bbox:
            phase_info = bbox[str(phase)]
            for bbox_info in phase_info:
                pedestrian_bbox = bbox_info.get('ped_bbox', None)
                vehicle_bbox = bbox_info.get('veh_bbox', None)
                frame_id = bbox_info.get('frame_id', None)
                cur_area = 0
                if pedestrian_bbox is not None:
                    _, _, w, h = pedestrian_bbox
                    cur_area += w * h
                if vehicle_bbox is not None:
                    _, _, w, h = vehicle_bbox
                    cur_area += w * h
                if cur_area >= 0:
                    for img in image_list:
                        if frame_id is not None and f"phase{phase}_bbox_{frame_id}" in img:
                            bbox_list.append((img, cur_area))
        else:
            print(f"Phase {phase} not found in bbox file {bbox_path} for scene {scene_name}, return None")
            return None
    else:
        for camera in bbox:
            if str(phase) in bbox[camera]:
                phase_info = bbox[camera][str(phase)]
                for bbox_info in phase_info:
                    pedestrian_bbox = bbox_info.get('ped_bbox', None)
                    vehicle_bbox = bbox_info.get('veh_bbox', None)
                    frame_id = bbox_info.get('frame_id', None)
                    cur_area = 0
                    if pedestrian_bbox is not None:
                        _, _, w, h = pedestrian_bbox
                        cur_area += w * h
                    if vehicle_bbox is not None:
                        _, _, w, h = vehicle_bbox
                        cur_area += w * h
                    if cur_area >= 0:
                        for img in image_list:
                            if frame_id is not None and f"{camera}_phase{phase}_bbox_{frame_id}" in img:
                                bbox_list.append((img, cur_area))
            else:
                print(f"Phase {phase} not found in bbox file {bbox_path} for scene {scene_name}, return None")
                return None
                    
    # Sort the bounding boxes by area in descending order
    bbox_list.sort(key=lambda x: x[1], reverse=True)
    # Prioritise vehicle view images
    bbox_list = [img for img in bbox_list if 'vehicle' in img[0]] + [img for img in bbox_list if 'vehicle' not in img[0]]
    # Get array of images of k largest bounding boxes
    camera_groups = []
    camera_images = []
    area_of_groups = []
    for img, area in bbox_list:
        if "_".join(img.split('/')[-1].split('_')[:-3]) not in camera_groups:
            camera_groups.append("_".join(img.split('/')[-1].split('_')[:-3])) # get the camera group name
            camera_images.append([img])
            area_of_groups.append(area)
        else:
            camera_images[camera_groups.index("_".join(img.split('/')[-1].split('_')[:-3]))].append(img)
            area_of_groups[camera_groups.index("_".join(img.split('/')[-1].split('_')[:-3]))] += area

    # Sort the camera groups by area in descending order
    sorted_groups = sorted(zip(camera_groups, camera_images, area_of_groups), key=lambda x: x[2], reverse=True)
    
    if "20230929_39_SY19_T1" in scene_name:
        for i in range(len(sorted_groups)):
            print(f"Group {sorted_groups[i][0]} has {len(sorted_groups[i][1])} images with area {sorted_groups[i][2]}")

    if len(sorted_groups) > 0:
        k_largest_bboxes = []
        for i in range(len(sorted_groups)):
            if len(sorted_groups[i][1]) < 3:
                print("skip", sorted_groups[i][0], "with only", len(sorted_groups[i][1]), "images")
                continue
            else:
                k_largest_bboxes = sorted_groups[i][1]
                break
        return k_largest_bboxes
    else:
        return None

def choose_best_image(scene_name, image_list, phase_number, bbox_path=None):
    best_images = []
    vehicle_view = [img for img in image_list if 'vehicle' in img]
    overview_view = [img for img in image_list if 'vehicle' not in img]
    image_list = overview_view + vehicle_view #rearrange

    # get the best image based on pedestrian bbox (prioritise vehicle view)
    ped_img = get_k_max_ped_bbox(scene_name, image_list, phase_number, bbox_path, 1)
    if ped_img:
        best_images.append(ped_img[0]) # if exist, take
    elif vehicle_view:
        best_images.append(vehicle_view[0] if vehicle_view else image_list[0]) # if no pedestrian bbox, just take the first vehicle view. else overhead view
    else:
        best_images.append(image_list[0]) # if no pedestrian bbox and no vehicle view (bdd), just take the first image
    # now best_images always has at least one image, which is the best pedestrian bbox image

    if phase_number < 3: # phase 0 1 2
        # if no pedestrian bbox, no vehicle view to take
        if 'vehicle' not in best_images[0]:
            # get the best image based on pedestrian and vehicle bbox in all images
            imgs_bbox = get_k_max_ped_veh_bbox(scene_name, image_list, phase_number, bbox_path, 3)
        else:
            # get the best image based on pedestrian and vehicle bbox in overview
            imgs_bbox = get_k_max_ped_veh_bbox(scene_name, overview_view, phase_number, bbox_path, 3)

        if imgs_bbox:
            for i in range(len(imgs_bbox)):
                if imgs_bbox[i] not in best_images:
                    best_images.append(imgs_bbox[i])
                    break
            if len(best_images) < 2:
                best_images.append(image_list[1]) # if no veh+ped max bbox, just take the second image
        else:
            best_images.append(image_list[len(image_list)//2]) # if no veh+ped max bbox, just take the middle image

        if best_images[0] == best_images[1]:
            # if the first two images are the same, take the next one
            for img in image_list:
                if img != best_images[0]:
                    best_images[1] = img
                    break
    else: 
        if len(image_list) < 3:
            return image_list # if not enough images, just return all images
        imgs_bbox = get_group_max_ped_veh_bbox(scene_name, image_list, phase_number, bbox_path)
        if imgs_bbox:
            best_images = imgs_bbox
        else:
            best_images = image_list[:3] # if no veh+ped max bbox, just take the first 3 images

    return best_images
