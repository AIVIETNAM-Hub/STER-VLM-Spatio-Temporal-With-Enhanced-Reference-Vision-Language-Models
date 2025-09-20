#!/usr/bin/env python3
import os
import json
import time
import base64
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import requests
import re
import io
from PIL import Image

PROMPT = """ 
You are analyzing a sequence of images (frames 1, 2, 3, ..., N) from a traffic awareness video. These images show real-world, complex intersections where the vehicle (in the colored bounding box, you can reference bounding boxes of neighbor frames) and the person (in the colored bounding box, sometimes with an attention arrow, you can reference bounding boxes of neighbor frames) may interact in ways that highlight potential risks, near-misses, or accidents such as collisions between cars and pedestrians.
Your primary task first is to provide detailed traffic safety context for each frame, focusing on both routine and dangerous events, and clearly describing potential or actual incidents.

IMPORTANT VIEWPOINT NOTE (take time to consider carefully because it's not easy to infer correctly):
For each frame, first determine the camera viewpoint:

If you see a black border at the bottom or the perspective appears to be from inside or just above a vehicle, assume the view is from the vehicle itself (such as dashcam or rearview). In these vehicle-perspective frames, the vehicle may not have a bounding box, and all relative positions, orientations, and movements should be judged as if you are seated in the vehicle looking forward—use "driver's view" as your frame of reference.
If there is an overhead or street camera view (where both vehicle and pedestrian usually have bounding boxes, and the view is from high above), base your spatial reasoning on the visible positions of the agents in the scene.
Always clarify the camera viewpoint before making spatial, directional, or relational judgments for agents in the image.
By first determining the camera viewpoint for each frame, you can accurately identify which vehicle and person to describe:
- If the frame is from a vehicle’s viewpoint (such as a dashcam or rear-view camera), assume the vehicle in question is the one recording the footage. The person to focus on will usually be the individual appearing within a bounding box in the frame; generate captions describing the interaction between this vehicle (the one with the camera view) and the pedestrian. If in some frames a bounding box is missing (for either vehicle or person), use information from adjacent frames where the bounding box is present to identify and maintain focus on the correct agents throughout the sequence.
- If the frame is from an overhead or surveillance viewpoint (where both vehicles and pedestrians are typically within bounding boxes), track the vehicle and person by their bounding boxes. If in some frames a bounding box is missing (for either vehicle or person), use information from adjacent frames where the bounding box is present to identify and maintain focus on the correct agents throughout the sequence.

**For each agent in the bounding boxes in each image, include the following:**
- Position (describe in no more than ten words): Always describe the position of the agent relative to the specific other agent of interest (i.e., for the "Vehicle," state its position relative to the "Person"; for the "Person," state their position relative to the "Vehicle") from the agent’s own forward-facing perspective. Do not describe the relative position to other vehicles, the camera, or the image frame itself. To infer this position, first determine the agent’s true direction of movement: for a person, use face and body orientation; for a vehicle, use the front or, if from a dashcam/rearview, use that as forward. Example: “to the left behind the pedestrian” (means to the pedestrian’s left, not the image’s left), “directly in front of the vehicle” (means in front of that specific vehicle).
- Proximity (describe in no more than ten words): Always describe how close the agent is to the other agent of interest only (not the camera or other objects). For the person, indicate their distance from the specific vehicle; for the vehicle, indicate its distance from the specific person. Example: “close to the pedestrian,” “very near the vehicle,” “far from the person.” Never describe proximity to the camera, background, or unrelated vehicles/persons.
Do not mention position or proximity to any other vehicle, the camera, or the image borders—only describe the spatial relation between the agent of focus and the single other agent (vehicle <-> person) that is being tracked in this scene.

- Action/movement (describe in at most ten words): Action belongs to one of the following categories:
    - Moving: walking, cycling, driving, etc.
    - Stopped: waiting, yielding, stopped at a red light, etc.
    - Collision: hitting, crashing, person falling down, etc.
    - Dangerous proximity: dangerously close to, almost hit, etc.
- (If moving only) **Direction of movement by the agent's own forward-facing perspective (describe in at most ten words):**
    - Very important: Never judge direction from a single frame. Always analyze the change in the agent's position and orientation using at least two neighboring frames (previous and next) for reference. 
    - When stating direction, always include "driver's view" or "person's view".
    - If the agent’s path is curving (evidenced by position change, body orientation, wheels, or path arc), use "turning right" or "turning left" appropriately. For this, activate your thinking mode, chain of thought, reasoning mode, analyze deeply step by step based on frames.
    - Only use "going straight" if the path over several frames (BASED ON THE IMAGE CONTEXT, not based on the positions on the frames, I mean) is completely straight with no indication of a curve.
    - Base this from the agent's perspective.
    - If the direction is ambiguous, state "direction undiscernible."
- (For person) Attention and focus (describe in at most five words): Direction of gaze or attention as indicated by red arrow, if no attetion arrow is detected, infer the attention of the person based on the information about the context.

**Instructions for output:**
- For each image, output the image order with the tag <image #num> (for example "<image 1>" and clearly separate each agent’s description with the agent label (“Vehicle:” or “Person:”).
- If multiple agents are present, only describe the one who appears inside a bounding box!
- Do NOT provide any overall summary or information not directly visible or justifiable by tracking the sequence.
- Do NOT assume or imagine events outside what can be seen and tracked in the sequence.
- If the direction is still unclear after comparison and inference, state "direction undiscernible".
- If there is no description for the image, just output "No description available for this image."
- There is at most one vehicle and one person need to focus in each image.


**Example for vehicle:**
Vehicle:
- Position: to the left and in front of the pedestrian
- Proximity: near the pedestrian
- Action: turning right

**Example for person:**
Person:
- Position: to the right and behind the vehicle
- Proximity: very near the vehicle
- Action: walking
- Attention: looking slightly forward/right

"""

SPLIT_INPUTS = {
    'train': '/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/data/total_train_caption_transformed.json',
    'val': '/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/data/total_val_caption_transformed.json',
    'test': '/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/data/total_test_caption_transformed.json',
}
SPLIT_OUTPUTS = {
    'train': '/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/train_pipeline/final_generated_total_train_caption_real.json',
    'val':   '/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/val_pipeline/final_generated_total_val_caption_real.json',
    'test':  '/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/val_pipeline/final_generated_total_test_caption_real.json',
}

def parse_agent_caption_lines(lines):
    attr_map = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            attr_map[key.strip().lower()] = val.strip()
    return attr_map

def parse_multiimage_gemini_caption(response_text, image_count):
    img_caption_dict = {}
    curr_imgnum = None
    curr_caption_lines = []
    agent_captions = []
    agent = None

    lines = response_text.splitlines()
    for line in lines:
        line = line.strip()
        m = re.match(r"^\*?\s*<image\s*#?(\d+)>", line, re.IGNORECASE) or re.match(r"^\*?\s*Image\s*#?(\d+):?", line, re.IGNORECASE)
        if m:
            if curr_imgnum is not None:
                if agent and curr_caption_lines:
                    agent_captions.append({
                        "agent": agent,
                        "caption_dict": parse_agent_caption_lines(curr_caption_lines)
                    })
                img_caption_dict[curr_imgnum] = agent_captions
            curr_imgnum = int(m.group(1)) - 1
            curr_caption_lines = []
            agent_captions = []
            agent = None
            continue

        m_agent = re.match(r"^\**\s*(Vehicle|Person):\s*\**", line, re.IGNORECASE)
        if m_agent:
            if agent and curr_caption_lines:
                agent_captions.append({
                    "agent": agent,
                    "caption_dict": parse_agent_caption_lines(curr_caption_lines)
                })
            agent = m_agent.group(1).lower()
            curr_caption_lines = []
            continue

        if agent is not None:
            curr_caption_lines.append(line)

    if agent and curr_caption_lines:
        agent_captions.append({
            "agent": agent,
            "caption_dict": parse_agent_caption_lines(curr_caption_lines)
        })
    if curr_imgnum is not None and agent_captions:
        img_caption_dict[curr_imgnum] = agent_captions

    for idx in range(image_count):
        if idx not in img_caption_dict:
            img_caption_dict[idx] = []

    return img_caption_dict

# helper: read token (prefers HUGGINGFACEHUB_API_TOKEN; falls back to .env API_KEY)
def read_api_key():
    # try environment first
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token
    # else try .env fallback
    load_dotenv(dotenv_path='/home/s48gb/Desktop/GenAI4E/aicity_02/selectFrames_Triet/pipeline/temp/.env')
    api_key = os.getenv("API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        raise RuntimeError("Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN or API_KEY in .env")
    return api_key

# convert image file -> data URI
def image_path_to_datauri(p):
    with open(p, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

# send a chunk (list of datauris) to HF model
def call_hf_multi_image(api_token, prompt_text, datauris, model_id="Qwen/Qwen2.5-VL-72B-Instruct", timeout=180):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
    payload = {
        "inputs": {
            "text": prompt_text,
            "images": datauris
        },
        "options": {"wait_for_model": True}
    }
    # retries
    for attempt in range(1, 5):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                try:
                    resp_json = r.json()
                except Exception:
                    return r.text
                # typical: dict with 'generated_text' or list; try to extract readable string
                if isinstance(resp_json, dict):
                    for k in ("generated_text", "text", "output", "data"):
                        if k in resp_json and isinstance(resp_json[k], str):
                            return resp_json[k]
                    return json.dumps(resp_json, ensure_ascii=False)
                elif isinstance(resp_json, list) and len(resp_json) > 0:
                    first = resp_json[0]
                    if isinstance(first, dict):
                        for k in ("generated_text", "text", "output"):
                            if k in first and isinstance(first[k], str):
                                return first[k]
                        return json.dumps(first, ensure_ascii=False)
                    elif isinstance(first, str):
                        return first
                    else:
                        return json.dumps(resp_json, ensure_ascii=False)
                else:
                    return str(resp_json)
            else:
                if r.status_code in (429, 503, 502):
                    wait = min(60, attempt * 5)
                    print(f"HF returned {r.status_code}. Backing off {wait}s (attempt {attempt})")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
        except requests.exceptions.RequestException as e:
            wait = min(60, attempt * 5)
            print(f"Request error: {e}. Retrying {wait}s (attempt {attempt})")
            time.sleep(wait)
            continue
    raise RuntimeError("HF inference failed after retries")

def build_chunk_prompt(base_prompt, absolute_start_idx, total_images):
    # Ask model to use absolute numbering inside the scene (so parser can work)
    instr = f"\n\nIMPORTANT: Number images absolutely from 1..{total_images} and use tags like <image 1>, <image 2>, ... . The first image in this request is absolute image {absolute_start_idx}.\n\n"
    return instr + base_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train','val','test'], required=True)
    parser.add_argument('--model', default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument('--max-images-per-request', type=int, default=16,
                        help="Chunk size for how many images to send in one HF request")
    args = parser.parse_args()

    input_path = SPLIT_INPUTS[args.split]
    output_path = SPLIT_OUTPUTS[args.split]
    api_token = read_api_key()
    model_id = args.model
    chunk_size = max(1, int(args.max_images_per_request))

    with open(input_path, "r") as f:
        data = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, "r") as g:
            output_dict = json.load(g)
    else:
        output_dict = {}

    for scene in tqdm(list(data.keys()), desc=f"Scenes ({args.split})"):
        image_paths = data[scene]
        # skip already-processed scenes (non-empty)
        if scene in output_dict and output_dict[scene]:
            continue

        # load image paths that exist
        valid_images = []
        for p in image_paths:
            if os.path.exists(p):
                valid_images.append(p)
            else:
                print(f"Warning: image not found {p}")

        if not valid_images:
            output_dict[scene] = []
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(output_dict, out_f, indent=2, ensure_ascii=False)
            continue

        # prepare datauris in chunks, call HF
        total_images = len(valid_images)
        chunk_texts = []
        try:
            for start in range(0, total_images, chunk_size):
                end = min(total_images, start + chunk_size)
                chunk_paths = valid_images[start:end]
                datauris = [image_path_to_datauri(p) for p in chunk_paths]
                prompt = build_chunk_prompt(PROMPT, absolute_start_idx=start+1, total_images=total_images)
                prompt += f"\n\nThese images correspond to absolute indices {start+1}..{end} in the scene."
                resp_text = call_hf_multi_image(api_token, prompt, datauris, model_id=model_id)
                chunk_texts.append(resp_text)
                time.sleep(0.8)
        except Exception as e:
            print(f"HF failed for scene {scene}: {e}")
            # skip scene if HF fails
            continue

        combined_text = "\n\n".join(chunk_texts)
        # parse using the original parser (expecting <image N> tags)
        img_caption_dict = parse_multiimage_gemini_caption(combined_text, total_images)

        # write output: for each image index, append vehicle & person dicts (same format you requested)
        output_dict[scene] = []
        for idx, img_path in enumerate(valid_images):
            agent_caption_lookup = {item["agent"]: item["caption_dict"] for item in img_caption_dict.get(idx, [])}
            for agent in ["vehicle", "person"]:
                output_dict[scene].append({
                    "image_path": img_path,
                    "agent": agent,
                    "caption": agent_caption_lookup.get(agent, {})
                })
        # save after every scene
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(output_dict, out_f, indent=2, ensure_ascii=False)

    print("All done. Saved to:", output_path)


if __name__ == "__main__":
    main()
