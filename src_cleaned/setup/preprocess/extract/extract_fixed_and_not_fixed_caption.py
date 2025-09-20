#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import re
import requests

from collections import defaultdict
from dotenv import load_dotenv

# ---------- Helper I/O ----------
def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def read_json_file_safe(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    else:
        return {}

def write_json_file(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def split_sentences(text):
    if not text:
        return []
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences

# ---------- Prompt construction (unchanged) ----------
def prepare_llm_prompt(ped_caption, veh_caption, phase_label):  
    critical_rule = (  
        "!!! CRITICAL RULE—NO EXCEPTIONS:\n"  
        "- DO NOT put speed, position, movement, action, or awareness/attention facts (e.g., stationary, comes to a halt, is moving, is standing, positioned in front/behind, line of sight, looking, noticing, etc) in ANY 'fixed' field.\n"  
        "- 'fixed' fields ONLY contain [appearance] (e.g., color, gender, age, clothing, height, accessories, etc.) "  
        "or static [environment] facts (e.g., weather, brightness, time of day, road type, number of lanes, "  
        "presence of sidewalks, permanent objects—NOT position, movement, or awareness).\n"  
        "- If you are unsure, ASSIGN TO 'not_fixed'.\n"  
        "- EVERY sentence in every output field MUST start with the proper subject as in the input (e.g., 'The pedestrian ...', 'He ...', 'The vehicle ...'). DO NOT output fragments or omit subjects. NEVER begin a sentence with a verb, adjective, or preposition.\n"  
        "- NEVER leave a field blank if there is any relevant information from the input; if not, use \"N/A\".\n"  
        "\nBAD EXAMPLES (NEVER allowed in 'fixed' or in any field):\n"  
        '  "The vehicle is stationary."\n'  
        '  "The vehicle is maintaining a constant speed of 0 km/h."\n'  
        '  "The vehicle is positioned diagonally to the left in front of the pedestrian."\n'  
        '  "The pedestrian is standing in front of the vehicle."\n'  
        '  "She is closely watching the vehicle."\n'  
        '  "Diagonally to the left in front of the vehicle on a bright and level asphalt road."\n'  
        '  "Standing still in the street."\n'  
        '  "Is positioned in front of the vehicle."\n'  
        '  "Wearing a blue jacket."\n'  
        '  "":\n'  
        "--- End BAD EXAMPLES ---\n"  
        "\nSUBJECT-RETENTION EXAMPLES:\n"  
        "INCORRECT OUTPUT (missing subject):\n"  
        '  \"pedestrian_not_fixed\": \"Standing still on the road. Diagonally to the right of the car.\"\n'  
        "CORRECTED OUTPUT:\n"  
        '  \"pedestrian_not_fixed\": \"The pedestrian was standing still on the road. He was positioned diagonally to the right of the car.\"\n'  
        "\nPLACEMENT OF ENVIRONMENT FACTS:\n"  
        "GOOD EXAMPLE ('fixed'):\n"  
        '  \"She was in an urban environment on a clear, dimly lit weekday.\"\n'  
        "BAD EXAMPLE ('not_fixed'):\n"  
        '  \"She finds herself in an urban environment on a clear, dimly lit weekday.\"\n'  
        "  (This sentence should be in 'fixed' with subject and in a stative form, e.g. \"She was in ...\")\n"  
        "\nSPECIAL NOTE:\n"  
        "If a sentence describes only static appearance or environment but is phrased with verbs like 'finds herself' or 'is present in', it BELONGS in 'fixed'. "  
        "Rephrase to a stative form if needed (e.g., 'She was in an urban environment on a clear, dimly lit weekday.'). "  
        "Only assign to 'not_fixed' if the sentence contains or implies change, awareness, movement, position, or action."  
        "\n"  
    )  
    fixed_summary = (  
        "'fixed' = ONLY physical description or environment that NEVER changes (gender, clothes, body, weather, time, road type, number of lanes, presence of sidewalks, permanent landmarks). "  
        "ANY sentence with position, movement, awareness, speed, action, orientation, or location goes to 'not_fixed'. "  
        "If uncertain, always put it in 'not_fixed'.\n"  
    )  

    tag_definitions = (  
        "Tag definitions:\n"  
        "- [appearance]: Physical look, identity, or attributes (clothing, age, gender, height, color, accessories, etc). No actions, intent, nor position.\n"  
        "- [location]: Spatial position/relative orientation (e.g., in front/left/behind; standing in street). No movement or static environment.\n"  
        "- [environment]: Permanent scene/background facts (weather, lighting, road material, number of lanes, day, time, etc.) No actions or positions.\n"  
        "- [attention]: Gaze, agent awareness/focus. (e.g., looking at/aware/unaware of another; NOT fixed)\n"  
        "- [action]: Movement or behavioral state (walking, driving, stopping, standing, speeding, reacting, seeing). Verbs of doing/change. NOT fixed.\n"  
    )  

    instructions = (  
        "INSTRUCTIONS:\n"  
        "1. Split BOTH the pedestrian and vehicle captions into their original, full sentences. Do not paraphrase, merge, split, or drop sentences.\n"  
        "2. Tag EACH sentence with ONE tag: [appearance], [location], [environment], [attention], or [action].\n"  
        "3. Prepare output fields:\n"  
        "   - 'pedestrian_fixed': All [appearance] and [environment] sentences for the pedestrian caption, ONLY if truly invariant, combined as a string.\n"  
        "   - 'vehicle_fixed': All [appearance] and [environment] sentences for the vehicle caption, ONLY if truly invariant, combined as a string.\n"  
        "   - 'pedestrian_not_fixed': Any [location], [attention], [action], PLUS any [appearance]/[environment] sentences that may change, from the pedestrian.\n"  
        "   - 'vehicle_not_fixed': Any [location], [attention], [action], PLUS any [appearance]/[environment] sentences that may change, from the vehicle.\n"  
        "4. Never place the same sentence in both 'fixed' and 'not_fixed'.\n"  
        "5. If ANY doubt, put in 'not_fixed'.\n"  
        "6. OUTPUT ONLY a valid JSON object, with these fields:\n"  
        '{\n'  
        '  "pedestrian_fixed": "...",\n'  
        '  "vehicle_fixed": "...",\n'  
        '  "pedestrian_not_fixed": "...",\n'  
        '  "vehicle_not_fixed": "..."\n'  
        '}\n'  
        "No extra notes, preamble, or output. Review final JSON BEFORE output: double-check that NO field called 'fixed' contains any description of action, position, movement, speed, orientation, awareness, attention, or location. If even a single phrase relates to these, it must go in 'not_fixed'.\n"  
        "FAILURE to follow these rules will make the result unusable.\n"  
    )  
    few_shot_example = (  
        "\nEXAMPLE:\n"  
        "Caption pedestrian:\n"  
        "[appearance] The pedestrian, a male in his 30s,\n"  
        "[action][location] The pedestrian was standing still on a main road in an urban area.\n"  
        "[appearance] He was wearing a gray T-shirt and black short pants.\n"  
        "[environment] The weather was cloudy with bright visibility, and the road surface was dry and level.\n"  
        "[location] The pedestrian's body was oriented in the same direction as the vehicle that approached him.\n"  
        "[location] He was positioned diagonally to the right, in front of the vehicle.\n"  
        "[attention] Despite being unaware of the vehicle's presence, he closely watched the parked vehicle in his line of sight.\n"  
        "[location] The relative distance between the pedestrian and the vehicle was far.\n"  
        "[environment] The road had a one-way traffic flow with three lanes and sidewalks on both sides.\n"  
        "[environment] It was a weekday, and the pedestrian seemed to have no awareness of the usual traffic volume on this road.\n"  
        "\n"  
        "Caption vehicle:\n"  
        "[action] The vehicle was moving at a constant speed of 10 km/h.\n"
        "[location] It was positioned behind a pedestrian and was quite far away from them.\n"
        "[location] The vehicle had a clear view of the pedestrian.\n"
        "[action] It was going straight ahead without any change in direction.\n"
        "[appearance] The environment conditions indicated that the pedestrian was a male in his 30s with a height of 160 cm.\n"
        "[appearance] He was wearing a gray T-shirt and black short pants.\n"
        "[environment] The event took place in an urban area on a weekday.\n"
        "[environment] The weather was cloudy but the brightness was bright.\n"
        "[environment] The road surface was dry and level, made of asphalt.\n"
        "[environment] The traffic volume was usual on the main road that had one-way traffic with three lanes.\n"
        "[environment] Sidewalks were present on both sides of the road.\n"
        "\n"
        "Correct Output:\n"
        "{\n"
        "  \"pedestrian_fixed\": \"The pedestrian, a male in his 30s. He was wearing a gray T-shirt and black short pants. "
        "The weather was cloudy with bright visibility, and the road surface was dry and level. "
        "The road had a one-way traffic flow with three lanes and sidewalks on both sides. It was a weekday.\",\n"
        "  \"vehicle_fixed\": \"The environment conditions indicated that the pedestrian was a male in his 30s with a height of 160 cm. "
        "He was wearing a gray T-shirt and black short pants. The event took place in an urban area on a weekday. "
        "The weather was cloudy but the brightness was bright. The road surface was dry and level, made of asphalt. "
        "The traffic volume was usual on the main road that had one-way traffic with three lanes. "
        "Sidewalks were present on both sides of the road.\",\n"
        "  \"pedestrian_not_fixed\": \"The pedestrian was standing still on a main road in an urban area. "
        "The pedestrian's body was oriented in the same direction as the vehicle that approached him. "
        "He was positioned diagonally to the right, in front of the vehicle. "
        "Despite being unaware of the vehicle's presence, he closely watched the parked vehicle in his line of sight. "
        "The relative distance between the pedestrian and the vehicle was far. "
        "The pedestrian seemed to have no awareness of the usual traffic volume on this road.\",\n"
        "  \"vehicle_not_fixed\": \"The vehicle was moving at a constant speed of 10 km/h. "
        "It was positioned behind a pedestrian and was quite far away from them. "
        "The vehicle had a clear view of the pedestrian. "
        "It was going straight ahead without any change in direction.\"\n"
        "}\n"
        "----\n"
    )

    output_warning = (
        "FINAL CHECK: Before outputting, review your answer and ensure that NO phrase in any 'fixed' field describes action, position, movement, speed, direction, awareness, attention, orientation, or location as these MUST be in 'not_fixed'. "
        "Output only a valid JSON object in the structure above. DO NOT provide any commentary or explanation. "
        "Never output any blank field; if there is no information for a field, output \"N/A\" as its value."
    )
    
    clarification = (
        "\nCLARIFICATION ON AMBIGUOUS STATIC DESCRIPTIONS:\n"
        "If a sentence describes only appearance or permanent environment (weather, time, location type, etc) but uses verbs such as 'is located', 'was found', 'was spotted', 'is present', or similar, TREAT IT AS [fixed]. \n"
        "EXAMPLES – these MUST go in 'fixed' (after rephrasing if needed):\n"
        '  "She is located in an urban area on a weekday." → [fixed]\n'
        '  "He was spotted on a clear and bright weekday morning." → [fixed].\n'
        '  "The scene is located on a dry, level road." → [fixed]\n'
        "Unless a sentence describes changing position, movement, orientation, awareness, speed, or agent action/intent, assign to 'fixed'.\n"
        "You may reword such sentences for clarity, e.g., 'She was in an urban area on a weekday.'\n"
        "Do NOT place factually static descriptions in 'not_fixed' only because of verbs such as 'is located' or 'was spotted'.\n"
    )

    prompt = (
        "You are an expert at analyzing structured information from traffic scene captions. "
        "You will be given a pedestrian and a vehicle caption for the SAME scene and a single time phase.\n"
        + critical_rule
        + fixed_summary
        + tag_definitions
        + instructions
        + clarification
        + few_shot_example
        + output_warning
        + f"\n\nINPUT CAPTIONS FOR THIS PHASE ONLY:\nPhase '{phase_label}' Pedestrian: {ped_caption if ped_caption else 'N/A'}\nPhase '{phase_label}' Vehicle: {veh_caption if veh_caption else 'N/A'}\n"
    )
    return prompt

# ---------- Hugging Face API integration ----------
def read_api_key(hint_env_var="HUGGINGFACEHUB_API_TOKEN", dotenv_path=None):
    # prefer environment variable
    token = os.environ.get(hint_env_var)
    if token:
        return token
    # else try load .env (optional path)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()  # load from current working dir or parent
    token = os.environ.get(hint_env_var) or os.environ.get("API_KEY")
    if not token:
        raise RuntimeError("Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN or API_KEY in a .env file.")
    return token

def llm_call(prompt, api_key=None, max_retries=5, model="Qwen/Qwen2.5-72B-Instruct", max_tokens=1200, temperature=0.0, timeout=120):
    """
    Calls Hugging Face inference API for text generation.
    Returns parsed JSON object extracted from model text (the same way previous llm_call did).
    """
    if api_key is None:
        api_key = read_api_key()

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            # you can add other generation parameters here if desired
        },
        "options": {"wait_for_model": True}
    }

    json_pattern = re.compile(r'\{[\s\S]*\}')

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            # 429/503 backoff
            if resp.status_code in (429, 503, 502):
                wait = min(60, attempt * 5)
                print(f"[llm_call] HF returned {resp.status_code}. Backing off {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()

            # Try to extract text
            try:
                resp_json = resp.json()
            except ValueError:
                # Not JSON response — use text
                raw_text = resp.text
            else:
                # The HF inference response may be:
                # - {"generated_text": "..."}
                # - {"error": "..."}
                # - a list of outputs
                raw_text = ""
                if isinstance(resp_json, dict):
                    # Check common fields
                    for k in ("generated_text", "text", "response", "output", "data"):
                        if k in resp_json and isinstance(resp_json[k], str):
                            raw_text = resp_json[k]
                            break
                    if not raw_text:
                        # sometimes model returns {"outputs":[{"generated_text": "..."}]}
                        if "outputs" in resp_json and isinstance(resp_json["outputs"], list) and resp_json["outputs"]:
                            first = resp_json["outputs"][0]
                            if isinstance(first, dict) and "generated_text" in first:
                                raw_text = first["generated_text"]
                elif isinstance(resp_json, list) and resp_json:
                    first = resp_json[0]
                    if isinstance(first, dict):
                        for k in ("generated_text", "text", "response", "output"):
                            if k in first and isinstance(first[k], str):
                                raw_text = first[k]
                                break
                    elif isinstance(first, str):
                        raw_text = first
                if not raw_text and isinstance(resp_json, (dict, list)):
                    # fallback to converting whole JSON to text
                    raw_text = json.dumps(resp_json, ensure_ascii=False)

            content = (raw_text or "").strip()

            # Try to find JSON object inside the model output
            match = json_pattern.search(content)
            if match:
                content = match.group()
            else:
                # No JSON discovered — log and retry up to max_retries
                print(f"[llm_call][No JSON found][Attempt {attempt}/{max_retries}]")
                print(f"---- HF output start ----\n{content}\n---- HF output end ----")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise RuntimeError(f"Could not find JSON in HF response:\n{content}")

            # Remove trailing commas before a closing bracket
            content = re.sub(r',(\s*[\]}])', r'\1', content)

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[llm_call][JSON ERROR][Attempt {attempt}/{max_retries}] {e}")
                print(f"---- Extracted content start ----\n{content}\n---- Extracted content end ----")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise RuntimeError(f"Could not parse HF response as JSON:\n{content}") from e

        except requests.exceptions.RequestException as e:
            print(f"[llm_call][HTTP ERROR][Attempt {attempt}/{max_retries}] {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Failed after {max_retries} attempts.")
                raise

# ---------- Processing logic (unchanged) ----------
def process_wts(data):
    output = {}

    for video_id, views in data.items():
        video_results = []

        for view in ['overhead_view', 'vehicle_view']:
            if view not in views:
                continue

            phases = views[view]
            phase_labels = sorted(phases.keys(), key=int)

            for phase in phase_labels:
                phase_data = phases[phase]
                ped_caption = phase_data.get('caption_pedestrian', "")
                veh_caption = phase_data.get('caption_vehicle', "")
                prompt = prepare_llm_prompt(ped_caption, veh_caption, phase)

                llm_response = llm_call(prompt)

                pedestrian_fixed = llm_response.get("pedestrian_fixed", "")
                pedestrian_not_fixed = llm_response.get("pedestrian_not_fixed", "")
                vehicle_fixed = llm_response.get("vehicle_fixed", "")
                vehicle_not_fixed = llm_response.get("vehicle_not_fixed", "")

                res = {}
                res["labels"] = phase
                res["view"] = view
                if phase == "0":
                    res["pedestrian_fixed"] = pedestrian_fixed
                    res["vehicle_fixed"] = vehicle_fixed
                res["pedestrian_not_fixed"] = pedestrian_not_fixed
                res["vehicle_not_fixed"] = vehicle_not_fixed
                res["start_time"] = str(phase_data.get("start_time", ""))
                res["end_time"] = str(phase_data.get("end_time", ""))

                video_results.append(res)
        output[video_id] = video_results

    return output


def process_bdd(data, output_path, prior_results=None):
    if prior_results is None:
        prior_results = {}
    counter = 0
    
    output = dict(prior_results)
    
    for video_id, phases in data.items():
        if video_id in output and isinstance(output[video_id], list) and len(output[video_id]) == len(phases):
            counter += 1
            continue
        
        counter += 1
        print(counter, video_id)
        video_results = output.get(video_id, [])
        processed_phases = {res["labels"] for res in video_results}
        
        for phase, phase_data in phases.items():
            if phase in processed_phases:
                continue
            ped_caption = phase_data.get('caption_pedestrian', "")
            veh_caption = phase_data.get('caption_vehicle', "")
            prompt = prepare_llm_prompt(ped_caption, veh_caption, phase)
            llm_response = llm_call(prompt)

            pedestrian_fixed = llm_response.get("pedestrian_fixed", "")
            pedestrian_not_fixed = llm_response.get("pedestrian_not_fixed", "")
            vehicle_fixed = llm_response.get("vehicle_fixed", "")
            vehicle_not_fixed = llm_response.get("vehicle_not_fixed", "")
            
            res = {}
            res["labels"] = phase
            if phase == "0":
                res["pedestrian_fixed"] = pedestrian_fixed
                res["vehicle_fixed"] = vehicle_fixed
            res["pedestrian_not_fixed"] = pedestrian_not_fixed
            res["vehicle_not_fixed"] = vehicle_not_fixed
            res["start_time"] = str(phase_data.get("start_time", ""))
            res["end_time"] = str(phase_data.get("end_time", ""))
            video_results.append(res)

            # save as the process goes
            output[video_id] = video_results
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(output, f, indent=2)
    
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['wts', 'bdd'], required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--base_path', type=str, default="/home/s48gb/Desktop/GenAI4E/aicity_02/all_infer")
    args = parser.parse_args()
    INPUT_FILENAMES = {
        ("wts", "train"): "wts_train_caption.json",
        ("wts", "val"): "wts_val_caption.json",
        ("bdd", "train"): "bdd_train_caption.json",
        ("bdd", "val"): "bdd_val_caption.json",
    }
    OUTPUT_FILENAMES = {
        ("wts", "train"): "wts_train_fnf_processed.json",
        ("wts", "val"): "wts_val_fnf_processed.json",
        ("bdd", "train"): "bdd_train_fnf_processed.json",
        ("bdd", "val"): "bdd_val_fnf_processed.json",
    }
    
    key = (args.dataset, args.split)
    input_file = os.path.join(args.base_path, INPUT_FILENAMES[key])
    output_file = os.path.join(args.base_path, OUTPUT_FILENAMES[key])

    print(f"Processing {input_file} into {output_file}")
    data = read_json_file(input_file)
    if args.dataset == "wts":
        result = process_wts(data)
    elif args.dataset == "bdd":
        partial_output = read_json_file_safe(output_file)
        result = process_bdd(data, output_file, prior_results=partial_output)
    else:
        raise ValueError("Unknown dataset!")
    write_json_file(output_file, result)
    print("Done.")

if __name__ == "__main__":
    main()
