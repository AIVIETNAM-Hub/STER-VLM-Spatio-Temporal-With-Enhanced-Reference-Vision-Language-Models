# Caption Decomposition

This folder contains code to decompose *raw captions* for traffic-scene clips into two categories per agent:

- **`fixed`** — static facts (appearance, environment). **Must not** contain movement, position, or attention.  
- **`not_fixed`** — dynamic facts (action, movement, position, attention) and any ambiguous facts.

Decomposition is performed by a Qwen2.5-72B-Instruct using a strict instruction prompt.  
The main script is:

- `extract_fixed_and_not_fixed_caption.py` — runs decomposition for two datasets (`wts` and `bdd`) and saves JSON outputs.

A convenience shell script is included (example below) to run decomposition for train/val splits.

---

## What does the code do?
1. Read an input JSON that contains per-video (or per-view) caption text split into phases.
2. For every phase, send the pedestrian + vehicle captions + a long, prescriptive prompt to a LLM (Qwen2.5-72B-Instruct) and ask it to output a JSON object with four fields:
   - `pedestrian_fixed`, `vehicle_fixed` — static appearance/environment facts,
   - `pedestrian_not_fixed`, `vehicle_not_fixed` — dynamic facts and ambiguous facts.
3. Save results incrementally to an output JSON. For BDD processing the script supports resuming based on partial outputs.

---

## Project layout

```bash
src/preprocess/extract_info/
├─ extract_fixed_and_not_fixed_caption.py # main script
├─ prepare_fixed_and_notfixed_caption.sh # example .sh to start ollama and run script (user-provided)
```


Content of Bash Script:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "/home/s48gb/Desktop/GenAI4E/aicity_02"
source .venv/bin/activate

# Require HF token for Qwen
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  echo "Please export HUGGINGFACEHUB_API_TOKEN (or place API_KEY in .env)."
  echo "Example: export HUGGINGFACEHUB_API_TOKEN='hf_xxx...'"
  exit 1
fi

cd src/preprocess/extract_info

# Run decomposition for WTS and BDD splits
python extract_fixed_and_not_fixed_caption.py --dataset wts --split train
python extract_fixed_and_not_fixed_caption.py --dataset wts --split val
python extract_fixed_and_not_fixed_caption.py --dataset bdd --split train
python extract_fixed_and_not_fixed_caption.py --dataset bdd --split val

echo "Done."

```



## Inputs & outputs

The Python script defaults to --base_path
``` /home/s48gb/Desktop/GenAI4E/aicity_02/all_infer.```

**Input filenames (inside `base_path`):**

- `wts_train_caption.json`

- `wts_val_caption.json`

- `bdd_train_caption.json`

- `bdd_val_caption.json`

**Output filenames (inside `base_path`):**

- `wts_train_fnf_processed.json`

- `wts_val_fnf_processed.json`

- `bdd_train_fnf_processed.json`

- `bdd_val_fnf_processed.json`


## Output JSON structure

For WTS the script produces a mapping video_id -> list_of_phase_results where each element (per phase) is:

```json
{
  "labels": "0",                 // phase label (string)
  "view": "overhead_view",       // in WTS many entries include 'view'
  "pedestrian_fixed": "...",     // present only for phase == "0"
  "vehicle_fixed": "...",        // present only for phase == "0"
  "pedestrian_not_fixed": "...",
  "vehicle_not_fixed": "...",
  "start_time": "123.4",
  "end_time": "124.0"
}
```

For BDD the script produces similar lists per video_id.


## Decomposition rules (summary of prompt)

The prompt enforces strict rules. Key rules (short):

  - `fixed` **MUST contain only static appearance/environment facts**, such as:
  
    - color, clothing, gender, age, height, weather, road type, number of lanes, presence of sidewalks, permanent objects, time-of-day.
    - **No** position, movement, awareness, speed, orientation, or action in fixed.

  - `not_fixed` **MUST contain all position/movement/attention/action facts**, and any ambiguous facts or anything that could change.

- Every sentence in the output must start with an explicit subject as in the input (e.g., "The pedestrian ...", "He ...", "The vehicle ..."). Do not output fragments without subjects.

- If in doubt, place the sentence in not_fixed.

- Output must be a single valid JSON object with exactly these four fields:
  - `pedestrian_fixed`, `vehicle_fixed`, `pedestrian_not_fixed`, `vehicle_not_fixed`

- If nothing belongs to a field, use "N/A" (the prompt explicitly enforces no blank fields).

- A long few-shot prompt with many examples and strict rules is embedded in prepare_llm_prompt() to maximize consistency.

## How the LLM is called (implementation details)

- The script calls the Hugging Face inference API endpoint for model Qwen/Qwen2.5-72B-Instruct (default). Example request target:

```
POST https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct
Authorization: Bearer <HUGGINGFACEHUB_API_TOKEN>
```

- The script:

  - Reads HUGGINGFACEHUB_API_TOKEN from the environment (or API_KEY from .env),

  - Posts the prompt to HF with generation parameters,

  - Extracts the JSON object from the model text using the same regex+cleanup logic you had previously,


# How to run?

- `cd` to the correct path which has `prepare_fixed_and_notfixed_caption.sh` and run the Bash Script file.

Or the terminal and run each line in bash script file manually.