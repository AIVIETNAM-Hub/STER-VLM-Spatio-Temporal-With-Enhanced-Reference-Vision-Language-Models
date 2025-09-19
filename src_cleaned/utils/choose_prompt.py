import os
import json

# =================================================== Spatial caption + vqa ===================================================

def get_prompt_spatial_pedestrian(image_len):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    prompt += """<task:caption>
You are analyzing a traffic or pedestrian surveillance image.
Your task is to write a concise yet detailed description of the central pedestrian (the one in the green bounding box, or if no box exists, the person closest to interacting with a vehicle).Use the following hints to guide your observation, but do not feel compelled to mention every single one - only include details that genuinely add useful context.
Hints (use as needed):\n- Gender and estimated age group (e.g., 30, 60, 40)\n- Height in centimeters (e.g., 170 cm, 180 cm)\n- Clothing details:\n+ Headwear (hat, helmet): type and color if you can see it\n+ Upper body (e.g., T-shirt, jacket, coat; color)\n+ Lower body (e.g., slacks, short pants, skirt; color)\n- If an obstacle is near the pedestrian (to the left, right, in front, behind), estimate its size (e.g., 1 m tall, 0.5 m wide)\n- Environmental context that matters:\n+ Weather (light rain, hail, snow, cloudy, etc.)\n+ Lighting (dark, bright, dim)\n+ Road surface (asphalt, dirt, gravel) and condition (dry, wet, frozen)\n+ Type of road (main road, residential, parking lot, etc.)\n+ Sidewalk presence (left, right, both, or none) and condition\n+ Roadside stripes or markings (left, right, both, or none)\n+ Traffic volume (heavy, light, normal)\n+ Road inclination (level, uphill, downhill)\n+ Number of lanes and traffic direction (e.g., \u201ctwo-way, two lanes\u201d)\n+ Street lights (present or not)\nNow, write a cohesive paragraph that weaves in the most relevant details. Keep your description succinct\u2014no more than 7-8 sentences."""
    return prompt

def get_prompt_spatial_vehicle(image_len):
    prompt = ""
    if image_len > 0:
        for i in range(1, image_len+1):
            prompt += "<image>\n"

    prompt += """<task:caption>
You are analyzing a traffic or pedestrian surveillance image.
Your task is to write a concise yet detailed description of the central pedestrian (the one in the green bounding box, or if no box exists, the person closest to interacting with a vehicle).Use the following hints to guide your observation, but do not feel compelled to mention every single one - only include details that genuinely add useful context.
Hints (use as needed):\n- Gender and estimated age group (e.g., 30, 60, 40)\n- Height in centimeters (e.g., 170 cm, 180 cm)\n- Clothing details:\n+ Headwear (hat, helmet): type and color if you can see it\n+ Upper body (e.g., T-shirt, jacket, coat; color)\n+ Lower body (e.g., slacks, short pants, skirt; color)\n- If an obstacle is near the pedestrian (to the left, right, in front, behind), estimate its size (e.g., 1 m tall, 0.5 m wide)\n- Environmental context that matters:\n+ Weather (light rain, hail, snow, cloudy, etc.)\n+ Lighting (dark, bright, dim)\n+ Road surface (asphalt, dirt, gravel) and condition (dry, wet, frozen)\n+ Type of road (main road, residential, parking lot, etc.)\n+ Sidewalk presence (left, right, both, or none) and condition\n+ Roadside stripes or markings (left, right, both, or none)\n+ Traffic volume (heavy, light, normal)\n+ Road inclination (level, uphill, downhill)\n+ Number of lanes and traffic direction (e.g., \u201ctwo-way, two lanes\u201d)\n+ Street lights (present or not)\nNow, write a cohesive paragraph that weaves in the most relevant details. Keep your description succinct\u2014no more than 7-8 sentences."""
    return prompt

def get_prompt_env_vqa():
    return """<task:multiple-choice qa>
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). From the following choices, select the most accurate answer based on the image. Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
# =================================================== Temporal caption + vqa ===================================================

def get_prompt_temporal_pedestrian(image_len, phase):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    if phase == "0": # ------------------------------------- phase 0 ------------------------------------
        prompt += """[task:caption-PRERECOGNITION]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness(crosswalks, traffic signals, vehicles, etc). 
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Your output caption should be around 2-4 concises sentences."""
    elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
        prompt += """[task:caption-RECOGNITION]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Your output caption should be around 2-4 concises sentences."""
    elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
        prompt += """[task:caption-JUDGEMENT]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Your output caption should be around 2-4 concises sentences."""
    elif phase == "3":  # ------------------------------------- phase 3 ------------------------------------
        prompt += """[task:caption-ACTION]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Your output caption should be around 2-4 concises sentences."""
    else:  # ------------------------------------- phase 4 ------------------------------------
        prompt += """[task:caption-AVOIDANCE]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Your output caption should be around 2-4 concises sentences. """
    return prompt

def get_prompt_temporal_vehicle(image_len, phase):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    if phase == "0": # ------------------------------------- phase 0 ------------------------------------
        prompt += """[task:caption-PRERECOGNITION]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness of the pedestrian(crosswalks, traffic signals, vehicles, etc). 

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
        prompt += """[task:caption-RECOGNITION]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian in front of the point of the vehicle's view.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
        prompt += """[task:caption-JUDGEMENT]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    elif phase == "3":   # ------------------------------------- phase 3 ------------------------------------
        prompt += """[task:caption-ACTION]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize in front of the vehicle view.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    else:   # ------------------------------------- phase 4 ------------------------------------
        prompt += """[task:caption-AVOIDANCE]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid of the pedestrian.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    return prompt

def get_prompt_phase_vqa(image_len, phase, pedestrian_pov=False):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"
    
    if not pedestrian_pov:
        if phase == "0": # ------------------------------------- phase 0 ------------------------------------
            prompt += """[task:multiple-choice-qa-PRERECOGNITION]
This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness(crosswalks, traffic signals, vehicles, etc). 
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
            prompt += """[task:multiple-choice-qa-RECOGNITION]
This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
            prompt += """[task:multiple-choice-qa-JUDGEMENT]
This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "3":  # ------------------------------------- phase 3 ------------------------------------
            prompt += """[task:multiple-choice-qa-ACTION]
This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        else:
            prompt += """[task:multiple-choice-qa-AVOIDANCE]
This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
    else:
        if phase == "0": # ------------------------------------- phase 0 ------------------------------------
            prompt += """[task:multiple-choice-qa-PRERECOGNITION]
This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness(crosswalks, traffic signals, vehicles, etc). 
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
            prompt += """[task:multiple-choice-qa-RECOGNITION]
This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
            prompt += """[task:multiple-choice-qa-JUDGEMENT]
This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "3":  # ------------------------------------- phase 3 ------------------------------------
            prompt += """[task:multiple-choice-qa-ACTION]
This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        else:
            prompt += """[task:multiple-choice-qa-AVOIDANCE]
This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
    return prompt

def get_prompt_temporal_pedestrian_with_reference(image_len, phase, reference):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    if phase == "0": # ------------------------------------- phase 0 ------------------------------------
        prompt += f"""[task:caption-PRERECOGNITION]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness(crosswalks, traffic signals, vehicles, etc). 
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Your output caption should be around 2-4 concises sentences."""
    elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
        prompt += f"""[task:caption-RECOGNITION]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Your output caption should be around 2-4 concises sentences."""
    elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
        prompt += f"""[task:caption-JUDGEMENT]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Your output caption should be around 2-4 concises sentences."""
    elif phase == "3":  # ------------------------------------- phase 3 ------------------------------------
        prompt += f"""[task:caption-ACTION]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Your output caption should be around 2-4 concises sentences."""
    else:  # ------------------------------------- phase 4 ------------------------------------
        prompt += f"""[task:caption-AVOIDANCE]
You are a traffic understanding model analyzing a traffic-related image, enriched with 3D scene reconstructions and 3D gaze information. Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). **Alert**: If the bounding boxes are not clear, focus on the pedestrain and the vehicle that nearest together.
Tags:[gaze3d] this tag will tell you that you can use the magenta-color arrow indicating pedestrian's gaze (if there is one) as additional visual information to give out decision.

This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid.
Use the following detailed aspects to guide your analysis:
1. Body orientation[gaze3d]: diagonal, straight, or perpendicular to the vehicle; moving in the same or opposite direction; to the left, right, or center. 
2. Position[gaze3d]: in front of, behind, or beside the vehicle; directly or diagonally; relative or adjacent.
3. Line of sight[gaze3d]: where the pedestrian is looking—direction of travel, crossing destination, nearby vehicles, objects like phones or the road.
4. Visual state[gaze3d]: e.g., scanning the scene, eyes closed, attentive.
5. Action: e.g., walking, standing, crossing, squatting, falling, rushing, Collision, thrown back
6. Direction and speed[gaze3d]: where and how fast the pedestrian is moving.
7. Fine-grained action[gaze3d]: Focus on the action type(cross, fall, rush, travel), any Modifier / Location Detail(prohibited place, diagonally, near moving/parked vehicle, near but not on crosswalk, ignoring traffic signal, backward, forward, in car lane,...)
8. Pedestrian awareness[gaze3d]: does the pedestrian notice the vehicle? 

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Your output caption should be around 2-4 concises sentences. """
    return prompt

def get_prompt_temporal_vehicle_with_reference(image_len, phase, reference):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    if phase == "0": # ------------------------------------- phase 0 ------------------------------------
        prompt += f"""[task:caption-PRERECOGNITION]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness of the pedestrian(crosswalks, traffic signals, vehicles, etc). 

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
        prompt += f"""[task:caption-RECOGNITION]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian in front of the point of the vehicle's view.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
        prompt += f"""[task:caption-JUDGEMENT]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    elif phase == "3":   # ------------------------------------- phase 3 ------------------------------------
        prompt += f"""[task:caption-ACTION]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize in front of the vehicle view.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    else:   # ------------------------------------- phase 4 ------------------------------------
        prompt += f"""[task:caption-AVOIDANCE]
You are a traffic understanding model analyzing a image related to traffic awareness . 
Focus on interactions between the pedestrian (in the green box) and vechile (in the red box). Your goal is to generate a short, clear caption describing their spatial relationship, intent, and motion, under the point of view of the vehicle

This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid of the pedestrian.

Include the following aspects:
- Action and state: is the vehicle starting, stopping, swerving, turning, or parked? Include phases like "about to", "just started", or "just finished".
- Pedestrian visibility: is the pedestrian visible from the vehicle's view?
- Position of the vehicle relative to the pedestrian
- The speed of the vehicle: (5km/h, 10km/h, etc.)

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  
Output:
Write a concise caption combining all these observations. Focus on intent, spatial layout, and motion—avoid redundancy.

Example: "The vehicle is traveling at a constant speed of 5km/h on a main road in an urban area. It is positioned behind and to the left of a pedestrian who is wearing a white T-shirt and gray slacks. The pedestrian is a male in his 20s with a height of 170 cm. The vehicle can see the pedestrian as they have a clear field of view. The vehicle is far away from the pedestrian and there is usual traffic volume on the road, which consists of two-way traffic." """
    return prompt

def get_prompt_phase_vqa_with_reference(image_len, phase, reference, pedestrian_pov=False):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"
    
    if not pedestrian_pov:
        if phase == "0": # ------------------------------------- phase 0 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-PRERECOGNITION]
This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness(crosswalks, traffic signals, vehicles, etc). 
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-RECOGNITION]
This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-JUDGEMENT]
This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "3":  # ------------------------------------- phase 3 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-ACTION]
This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        else:
            prompt += f"""[task:multiple-choice-qa-AVOIDANCE]
This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
    else:
        if phase == "0": # ------------------------------------- phase 0 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-PRERECOGNITION]
This is the **PRERECOGNITION** phase, where the timing is before the start of an environment awareness(crosswalks, traffic signals, vehicles, etc). 
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "1": # ------------------------------------- phase 1 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-RECOGNITION]
This is the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "2":  # ------------------------------------- phase 2 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-JUDGEMENT]
This is the **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        elif phase == "3":  # ------------------------------------- phase 3 ------------------------------------
            prompt += f"""[task:multiple-choice-qa-ACTION]
This is the **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
        else:
            prompt += f"""[task:multiple-choice-qa-AVOIDANCE]
This is the **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid.
You are analyzing a traffic camera or pedestrian surveillance camera or vehicle's dashcam image(s). Focus on the behaviour and spatial context around the pedestrian (in the green box and/or a magenta-color arrow indicating pedestrian's gaze if there is one) and the vechicle (in the red box). From the following choices, select the most accurate answer (1 choice from multiple choice) based on the pedestrian's Point of View in the given image(s). Only respond with the letter and the answer (e.g., "a. 60s"). Do not explain or rephrase.

Here are some references for the given images (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

Examples:
Q: [question]?\na. OptionA\nb. OptionB\nc. OptionC\nd. OptionD  
A: c. OptionC\n"""
    return prompt

# =================================================== Merge caption ===================================================

def get_prompt_merge(image_len, fixed_caption, phase_caption, phase_number, type):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    if phase_number == "0":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images and initial captions. The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian in front of the point of the vehicle's view.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
PRERECOGNITION caption: {phase_caption}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images and the initial captions. You caption should be around 4-6 sentences."""
    elif phase_number == "1":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images and initial captions. The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian in front of the point of the vehicle's view.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
RECOGNITION caption: {phase_caption}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images and the initial captions. You caption should be around 4-6 sentences."""
        pass
    elif phase_number == "2":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images and initial captions. The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the  **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
JUDGEMENT caption: {phase_caption}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images and the initial captions. You caption should be around 4-6 sentences."""
    elif phase_number == "3":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images and initial captions. The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the  **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize in front of the vehicle view.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
ACTION caption: {phase_caption}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images and the initial captions. You caption should be around 4-6 sentences."""
    else:
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images and initial captions. The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the  **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid of the pedestrian.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
AVOIDANCE caption: {phase_caption}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images and the initial captions. You caption should be around 4-6 sentences."""
    return prompt

def get_prompt_merge_with_reference(image_len, fixed_caption, phase_caption, phase_number, type, reference):
    prompt = ""
    if image_len > 0:
        for _ in range(image_len):
            prompt += "<image>\n"

    if phase_number == "0":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images, initial captions and some references (only use the given references as clues, do not copy-paste or depend on them). The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian in front of the point of the vehicle's view.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
PRERECOGNITION caption: {phase_caption}
Here are the references (only use the given references as clues, do not copy-paste or depend on them) 
{reference}  

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images, the initial captions and the additional references. You caption should be around 4-6 sentences."""
    elif phase_number == "1":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images, initial captions and some references (only use the given references as clues, do not copy-paste or depend on them). The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the **RECOGNITION** phase, where the timing is from the start of the environment awareness(crosswalks, traffic signals, vehicles, etc), until a judgement is made. There should be some awareness clues of the pedestrian in front of the point of the vehicle's view.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
RECOGNITION caption: {phase_caption}
Here are the references (carefully use the references as clues, not copy-paste or depend on them): 
{reference}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images, the initial captions and the additional references. You caption should be around 4-6 sentences."""
        pass
    elif phase_number == "2":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images, initial captions and some references (only use the given references as clues, do not copy-paste or depend on them). The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the  **JUDGEMENT** phase. In principle, it is the moment from which environmental awareness is completed until the start of an action. The pedestrian should be fully awareness of the surrounding situation.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
JUDGEMENT caption: {phase_caption}
Here are the references (carefully use the references as clues, not copy-paste or depend on them): 
{reference}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images, the initial captions and the additional references. You caption should be around 4-6 sentences."""
    elif phase_number == "3":
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images, initial captions and some references (only use the given references as clues, do not copy-paste or depend on them). The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the  **ACTION** phase. In principle, this is the start moment of any part of the body of the pedestrian(eye and ears) up to the time a result(e.g, collision) occurs. The action should be clear to recognize in front of the vehicle view.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
ACTION caption: {phase_caption}
Here are the references (carefully use the references as clues, not copy-paste or depend on them): 
{reference}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images, the initial captions and the additional references. You caption should be around 4-6 sentences."""
    else:
        prompt += f"""You are a expert in analyzing fine-grained video understanding in traffic scenarios. 

CONTEXT: You are given some visual related images, initial captions and some references (only use the given references as clues, do not copy-paste or depend on them). The images will related to spatial characteristics , like the characteristic of the pedestrian(gender, age, height, clothes, his surrounding, etc.), the condition of the environment(Weather, lighting condition, road surface, type of road, sidewalk, traffic volume, road inclination, etc), and the phase of the  **AVOIDANCE** phase. In principle, this is the time after avoidability is clear until the time of avoidance happened or failure to avoid of the pedestrian.

Here are the initial captions:
SPATIAL caption: {fixed_caption}
AVOIDANCE caption: {phase_caption}
Here are the references (carefully use the references as clues, not copy-paste or depend on them): 
{reference}

GOAL: Your objective is to generate a response for the [{type.upper()} CAPTION], using the clues from the images, the initial captions and the additional references. You caption should be around 4-6 sentences."""
    return prompt

# =================================================== Determine ===================================================

def determine_reference_for_caption(reference_for_captions, scenario, best_images, ped_or_veh):
    '''
    Args:
        reference_for_captions (dict): Dictionary of reference captions for each scenario.
        scenario (str): Scenario name, e.g., "230922_12_SN17..."
        best_images (list): List of best images for the scenario.
        ped_or_veh (str): Type of caption, either 'person' or 'vehicle'.
    
    Returns:
        str: The reference caption for the given scenario and phase.

    Format:
    Reference for person in image 1 (from vehicle dashcam):
    - position: ...
    - action: ... 
    - speed: ...
    Reference for vehicle in image 2 (from traffic camera):
    - position: ...
    - action: ...
    - speed: ...
    ....
    
    '''
    if scenario not in reference_for_captions:
        # print(f"Warning: No reference caption found for scenario {scenario}.")
        return ""

    all_references = []    
    for ref in reference_for_captions[scenario]:
        if ref["agent"] == ped_or_veh:
            for i, image in enumerate(best_images):
                if image == ref["image_path"] and ref["caption"]:
                    all_references.append((i, ref))

    res_reference = ""
    if len(all_references) > 0:
        all_references.sort(key=lambda x: x[0])  # Sort by index of the image
        for i, ref in all_references:
            cur_ref_caption = f"Reference for {ref['agent']} in image {i+1} (from {'vehicle dashcam' if ('vehicle' in ref['image_path'] or 'video' in ref['image_path']) else 'traffic camera'}):\n"
            for key, value in ref["caption"].items():
                cur_ref_caption += f"{key}: {value}\n"
            res_reference += cur_ref_caption
        return res_reference.strip()
    else: 
        return None
