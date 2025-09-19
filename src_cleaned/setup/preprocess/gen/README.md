# Caption Generation & Preprocessing Pipeline

This folder is used for:
- group frame image paths into scenes,
- call a multimodal LLM (Qwen2.5-VL-72B-Instruct on Hugging Face) to generate caption for each keyframe,
- parse model output into structured caption JSON usable by downstream code.

---

## What this repo does
1. **`transform.py`** — transforms dataset entries into a scene-to-image-paths JSON (`total_<split>_caption_transformed.json`).

E.g:
```json
"video6086": [
    "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/bbox_extracted_not_fixed_external/video6086/video6086_phase0_bbox_519.jpg",
    "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/bbox_extracted_not_fixed_external/video6086/video6086_phase0_bbox_541.jpg",
    ...
]
```
2. **`gen_ref.py`** — sends scene images and a long task prompt to the HF model `Qwen/Qwen2.5-VL-72B-Instruct`, parses the response and writes structured captions (vehicle/person) per image.
3. **`prepare_ref_caption.sh`** — convenience runner that executes `transform.py` and `gen_ref.py` for `train`, `val`, and `test`.

---

## Project structure (important files)
```
selectFrames_Triet/pipeline/
├─ temp/
│ ├─ transform.py
│ ├─ gen_ref.py
│ ├─ prepare_ref_caption.sh
│ └─ .env # optional: API_KEY=hf_...
├─ data/
│ ├─ total_train_caption_transformed.json
│ ├─ total_val_caption_transformed.json
│ └─ total_test_caption_transformed.json
├─ train_pipeline/
│ └─ final_generated_total_train_caption_real.json
├─ val_pipeline/
│ └─ final_generated_total_val_caption_real.json
└─ ...
```



> Note: `transform.py` creates `total_<split>_caption_transformed.json` in `data/`. `gen_ref.py` reads those files and writes final JSON to the `train_pipeline/` / `val_pipeline/` paths defined in the script.

---

## Input & output formats

### Input: scene JSON (output of `transform.py`)
`total_<split>_caption_transformed.json` is a mapping `scene_name -> [image_path1, image_path2, ...]`:

```json
{
  "20230707_12_SN17_T1": [
    "/abs/path/sceneA/frame0001.jpg",
    "/abs/path/sceneA/frame0002.jpg"
  ],
  "20230707_13_SN42_T4": [
    "/abs/path/sceneB/frame0001.jpg"
  ]
}
```




### Output: structured captions (from gen_ref.py)

For each image the script emits two objects (vehicle & person). caption is a dict of parsed fields or {}:

```json
{
  "20230707_12_SN17_T1": [
    {
      "image_path": "/abs/path/sceneA/frame0001.jpg",
      "agent": "vehicle",
      "caption": {"position":"to the left and in front of the pedestrian", "proximity":"near the pedestrian", "action":"turning right"}
    },
    {
      "image_path": "/abs/path/sceneA/frame0001.jpg",
      "agent": "person",
      "caption": {"position":"to the right and behind the vehicle", "proximity":"very near the vehicle", "action":"walking", "attention":"looking slightly forward/right"}
    },
    ...
  ]
}
```

If the model produced no structured values for an agent, caption is {}.

--- 

## Setup

- Need to specify an environment, install needed libraries.

- Sample:
```bash
# Activate virtualenv
cd /home/s48gb/Desktop/GenAI4E/aicity_02/
source .venv/bin/activate

# Ensure HF token exported
export HUGGINGFACEHUB_API_TOKEN="hf_xxx..."

# Run the convenience script (transform + generate) for train/val/test
cd selectFrames_Triet/pipeline/temp
./prepare_ref_caption.sh
```