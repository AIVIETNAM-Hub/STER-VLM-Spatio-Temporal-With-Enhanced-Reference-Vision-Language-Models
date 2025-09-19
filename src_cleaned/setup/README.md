# Contents
1. What is a 'setup' folder
2. How to run setup
3. Purpose of each component 
4. Structure of each file

# 1. What is a 'setup' folder
Containing code and scripts to extract and transform the WTS and BDD dataset into training, validation, and test format

# 2. How to run setup

To preprocess **training and validation set**, run:
```
source .venv/bin/activate (Linux)
# .venv/Scripts/activate.bat (Windows)

# <dataset_path>: the directory to dataset e.g /home/.../wts_dataset_zip
# <preprocessed_path>: the directory to store preprocessed data

# Extract information from dataset
setup/extract_draw_trainval.sh <dataset_path> <preprocessed_path>

# Run decompose captions and references generation
setup/preprocess/extract/prepare_fixed_and_notfixed_caption.sh
setup/preprocess/gen/prepare_ref_caption.sh train_val

# Transform into Qwen-2.5-VL format
setup/transform_trainval.sh <preprocessed_path>

```

To preprocess **test set**, run:
```
source .venv/bin/activate (Linux)
# .venv/Scripts/activate.bat (Windows)

# <dataset_path>: the directory to dataset e.g /home/.../wts_dataset_zip
# <preprocessed_path>: the directory to store preprocessed data
# <prediction_path>: the directory to inferred caption e.g. /home/.../prediction/

# Extract information from dataset
setup/extract_draw_test.sh <dataset_path> <preprocessed_path>

# Run references generation
setup/preprocess/gen/prepare_ref_caption.sh test

setup/transform_test.sh <dataset_path> <preprocessed_path>
# Then infer using spatial-invariant and temporal-variant model

# After running inference on spatial-invariant and temporal-variant model
python setup_preprocess_test/transform_merge_enhanced.py --preprocessed_data_path <preprocessed_path> --prediction_path <prediction_path>

```

# 3. Purpose of each component 
## Overall folder structure
```txt
src_cleaned/setup
├── preprocess
│   ├── draw
│   │   ├── bdd_draw.py                     #Draws bounding boxes and gaze 3D for images from BDD_PC_5K dataset
│   │   └── wts_draw.py                     #Draws bounding boxes and gaze 3D for images from WTS dataset
│   ├── extract
│   │   ├── extract_bbox_annotated.py       #Extracts human-annotated bounding boxes from both datasets
│   │   ├── extract_bbox_generated.py       #Extracts machine-generated bounding boxes from both datasets
│   │   ├── extract_caption.py              #Extracts caption from caption json file
│   │   ├── extract_gaze.py                 #Extracts gaze 3D from gaze json file
│   │   └── extract_vqa.py                  #Extracts vqa from vqa json file
│   └── transform
│       ├── match_caption.py                #Matchs decomposed captions with appropriate images
│       ├── transform_component.py          #Transforms captions and VQA for spatial-invariant and temporal-variant models into Qwen-2.5-VL format
│       └── transform_merge_enhanced.py     #Transforms caption and VQA for merge model into Qwen-2.5-VL format
├── preprocess_test
│   ├── draw_bbox.py                        #Draws bounding boxes and gaze 3D for images from both datasets
│   ├── extract_bbox_annotated.py           
│   ├── extract_bbox_generated.py           
│   ├── extract_vqa.py                      
│   ├── transform_component.py
│   └── transform_merge_enhanced.py
├── README.md                               #Readme file demonstrates usage of setup folder
├── extract_draw_test.sh                    #Script to run train and val data
├── extract_draw_trainval.sh                #Script to run test data
├── transform_test.sh                       #Script to transform train and val data
└── transform_trainval.sh                   #Script to transform test data
```

## Input and output of each file
### Folder "Preprocess"
**extract_bbox_annotated.py**:
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for bbox json
- Output:
    - Two extracted bbox files **wts\_{split}\_annotated_bbox.json** and **bdd\_{split}\_annotated_bbox.json** for each split (Total 4 files).

**extract_bbox_generated.py**:
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for bbox json
- Output:
    - Two extracted bbox files **wts\_{split}\_generated_bbox.json** and **bdd\_{split}\_generated_bbox.json** for each split (Total 4 files).

**extract_caption.py**:
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for caption json
- Output:
    - Two caption files **wts\_{split}\_caption.json** and **bdd\_{split}\_captio_.json** for each split (Total 4 files).

**extract_gaze.py**:
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for gaze json
- Output:
    - One gaze file each split **wts_{split}_gaze.json**.

**extract_vqa.py**:
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for vqa json
- Output:
    - Four vqa files **wts\_{split}\_phase_vqa.json** + **wts\_{split}\_env_vqa.json** + **bdd\_{split}\_phase_vqa.json** and **bdd\_{split}\_env_vqa.json** for each split (Total 8 files)


**wts_draw.py**: 
- Requirements:
    - Two extracted bbox files each split: **wts\_{split}\_annotated_bbox.json** and **wts\_{split}\_generated_bbox.json** from **extract_bbox_annotated.py** and **extracted_bbox_generated.py**
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - bbox_path: the extracted bbox directory path
    - output_path: the output path for two "drawn image" folders
- Output:
    - Folder **drawn_bbox_annotated** contains images that added human-annotated bounding boxes.
    - Folder **drawn_bbox_generated** contains images that added machine-generated bounding boxes.

**bdd_draw.py**: 
- Requirements:
    - Two extracted bbox files each split: **bdd\_{split}\_annotated_bbox.json** and **bdd\_{split}\_generated_bbox.json** from **extract_bbox_annotated.py** and **extracted_bbox_generated.py**
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - bbox_path: the extracted bbox directory path
    - output_path: the output path for two "drawn image" folders
- Output:
    - Folder **drawn_bbox_annotated/external**.
    - Folder **drawn_bbox_generated/external**.


**match_caption.py**: 
- Requirements:
    - Two drawn bbox folders **drawn_bbox_annotated/external** and **drawn_bbox_generated/external** 
    - Two extracted bbox files for each split **wts\_{split}\_generated_bbox.json** and **bdd\_{split}\_generated_bbox.json**
    - Two decomposed captions files for each split **wts_{split}_fnf_processed.json** and **bdd_{split}_fnf_processed.json**
- Arguments: 
    - split: 'train' or 'val'
    - preprocessed_path: the directory to preprocessed dataset folder e.g. /home/.../preprocessed_dataset/
- Output:
    - **wts\_{split}\_spatial_pair.json**: contains spatial-invariant captions for {split} of wts dataset.
    - **wts\_{split}\_temporal_pair.json**: contains temporal-variant captions for {split} of wts dataset.
    - **bdd\_{split}\_spatial_pair.json**: contains spatial-invariant captions for {split} of bdd dataset.
    - **bdd\_{split}\_temporal_pair.json**: contains temporal-variant captions for {split} of bdd dataset.

**transform_component.py**: 
- Requirements:
    - Two extracted bbox files each split: **bdd\_{split}\_annotated_bbox.json** and **bdd\_{split}\_annotated_bbox.json** from **extract_bbox_annotated.py**
    - Four caption files each split: **wts\_{split}\_spatial_pair.json** + **wts\_{split}\_temporal_pair.json** + **bdd\_{split}\_spatial_pair.json** and **bdd\_{split}\_temporal_pair.json**
    - Four vqa files each split: **wts\_{split}\_phase_vqa.json** + **wts\_{split}\_env_vqa.json** + **bdd\_{split}\_phase_vqa.json** and **bdd\_{split}\_env_vqa.json**
- Arguments: 
    - split: 'train' or 'val'
    - preprocessed_path: the directory to preprocessed dataset folder e.g. /home/.../preprocessed_dataset/
- Output:
    - **transformed\_{split}\_spatial_caption.json**: contains spatial-invariant captions for {split} in Qwen-2.5-VL format.
    - **transformed\_{split}\_temporal_caption.json**: contains temporal-variant captions for {split} in Qwen-2.5-VL format.
    - **transformed\_{split}\_env_vqa.json**: contains environment vqa for {split} in Qwen-2.5-VL format.
    - **transformed\_{split}\_phase_vqa.json**: contains vqa of each phases for {split} in Qwen-2.5-VL format.

**transform_merge_enhanced.py**: 
- Requirements:
    - Two folders **drawn_bbox_annotated/external** and **drawn_bbox_generated/external** 
    - Two files **wts\_{split}\_generated_bbox.json** and **bdd\_{split}\_generated_bbox.json** (Total 4 files)
    - One file **total_{split}_caption_transformed.json** contains hints from LVLM
- Arguments: 
    - split: 'train' or 'val'
    - preprocessed_path: the directory to preprocessed dataset folder e.g. /home/.../preprocessed_dataset/
- Output:
    - **total_{split}_caption_enhanced.json**: contains ground truth and enhanced prompt for merge model.



### Folder "Preprocess_test"
**extract_bbox_annotated.py**:
- Arguments: 
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for bbox json
- Output:
    - Two extracted bbox files **test_wts_bbox_annotated.json** and **test_bdd_bbox_annotated.json**.

**extract_bbox_generated.py**:
- Arguments: 
    - split: 'train' or 'val'
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for bbox json
- Output:
    - Two extracted bbox files **test_wts_bbox_generated.json** and **test_bdd_bbox_generated.json**.

**extract_vqa.py**:
- Arguments: 
    - data_path: the directory to dataset e.g /home/.../wts_dataset_zip
    - output_path: the output path for vqa json
- Output:
    - One vqa file: **test_vqa.json**.


**draw_bbox.py**: 
- Requirements:
    - Two extracted bbox files: **wts\_{split}\_annotated_bbox.json** and **wts\_{split}\_generated_bbox.json** from **extract_bbox_annotated.py** and **extracted_bbox_generated.py**
- Arguments:
    - data_path: the directory to dataset e.g /home/.../wts_dataset_test
    - bbox_path: the extracted bbox directory path
    - output_path: the output path for two "drawn image" folders
- Output:
    - Folder **bbox_annotated** contains images that added human-annotated bounding boxes.
    - Folder **bbox_generated** contains images that added machine-generate bounding boxes.


**transform_component.py**: 
- Requirements:
    - Two extracted bbox files: **bdd\_{split}\_annotated_bbox.json** and **bdd\_{split}\_generated_bbox.json** from **extract_bbox_annotated.py** and **extracted_bbox_generated.py**
    - One vqa file: **test_vqa.json**
- Arguments: 
    - split: 'train' or 'val'
    - preprocessed_path: the directory to preprocessed dataset folder e.g. /home/.../preprocessed_dataset/
- Output:
    - **test_spatial_caption.json**: contains spatial-invariant captions for {split} in Qwen-2.5-VL format.
    - **test_temporal_caption.json**: contains temporal-variant captions for {split} in Qwen-2.5-VL format.
    - **test_env_vqa.json**: contains environment vqa for {split} in Qwen-2.5-VL format.
    - **test_phase_vqa.json**: contains vqa of each phases for {split} in Qwen-2.5-VL format.

**transform_merge_enhanced.py**: 
- Requirements:
    - Two extracted bbox files: **wts\_{split}\_generated_bbox.json** and **bdd\_{split}\_generated_bbox.json**
    - One hint file **total_{split}_caption_transformed.json** contains hints from LVLM
- Arguments: 
    - preprocessed_path: the directory to preprocessed dataset folder e.g. /home/.../preprocessed_dataset/
    - prediction_path: the directory to inferred caption e.g. /home/.../prediction/
- Output:
    - **test_merge_caption_enhanced.json**: contains ground truth and enhanced prompts for merge model.

# 4. Structure of each file
Extracted human-annotated bbox follows the format (**\*_annotated_bbox.json**):
```
{
    "20230707_17_CY23_T1": {
        "20230707_17_CY23_T1_Camera2_2": {
            "4": {
                "frame_id": 1118,
                "start_time": xxx,
                "end_time": xxx,
                "ped_bbox": null,
                "veh_bbox": [
                    1280.3,
                    814.4,
                    639.7,
                    265.6
                ]
            },
            "3": {
                ...
            }
            ...
        },
        "20230707_17_CY23_T1_Camera2_3": {
            ...
        },
        ...
    },
    "20230728_31_CN37_T1": {
        ...
    },
    ...
}
```
Extracted machine-generated bbox follows the format (**\*_generated_bbox.json**):
```
{
    "video1004.mp4": {
        "0": [ # phase
                {
                    "fps": 30.0,
                    "frame_id": 189,
                    "time": 6.326,
                    "ped_bbox": [x, y, w, h],
                    "veh_bbox": None
                },
                {
                    "fps": 30.0,
                    "frame_id": 223,
                    "time": 7.757,
                    "ped_bbox": [x, y, w, h],
                    "veh_bbox": None
                },
                {
                    ...
                },
            ]
        "1": [
            ...
        ]
    
        ...
    },
    ...
}
```
Extracted caption follows the format (**\*_caption.json**):
```
{

    "video1004.mp4": {
        "0": {
            "start_time": 0.00,
            "end_time": 1.00,
            "caption_pedestrian": "A",
            "caption_vehicle": "B",
        },
        "1": {
            ...
        },
        ...
    },
    "video100.mp4": {
        ...
    },
    ...
}
```
Extracted gaze follows the format (**\*_gaze.json**):
```
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
```
Extracted VQA about environment follows the format (**\*_env_vqa.json**):
```
{
    "20230707_17_CY23_T1": {
        "environment": [
            {
                "question": "What is the road inclination in the scene?",
                "a": "Steep downhill",
                "b": "Level",
                "c": "Gentle downhill",
                "d": "Steep uphill",
                "correct": "b"
            },
            ...
        ]
    },
    "20230728_31_CN37_T1": {
        ...
    },
    ...
}
```
Extracted VQA about each phases follows the format (**\*_phase_vqa.json**):
```
{
    "20230707_17_CY23_T1": {
        "20230707_17_CY23_T1_Camera2_2": {
            "4": {
                "start_time": xxx,
                "end_time": xxx,
                "conversations": [
                    {
                        "question": "What is the orientation of the pedestrian's body?",
                        "a": "Same direction as the vehicle",
                        "b": "Diagonally to the left, in the opposite direction to the vehicle",
                        "c": "Perpendicular to the vehicle and to the right",
                        "d": "Diagonally to the right, in the same direction as the vehicle",
                        "correct": "c"
                    },
                    ...
                ]
            },
            "3": {
                ...
            }
            ...
        },
        "20230707_17_CY23_T1_Camera2_3": {
            ...
        },
    },
    "20230728_31_CN37_T1": {
        ...
    },
    ...
}
```
Matched spatial captions and images follows the format (**\*_spatial_pair.json**):
```
{
    [
        "20230707_8_SN46_T1": {
            'image': ["path/to/images",...],
            'caption_veh': "caption",
            'caption_ped': "caption"
        },
    ]
}
```
Matched temporal captions and images follows the format (**\*_temporal_pair.json**):
```
{
    [
        "20230707_8_SN46_T1": [
            'labels: "0"
            'image': ["path/to/images",...],
            'caption_veh': "caption",
            'caption_ped': "caption"
        ],
        ...
    ]
}
```
Transformed spatial captions (into Qwen format) follows the format (**transformed_\*.json** and **total_{split}_caption_enhanced.json**):
```
[
    {
        "image": ["path/to/image", "path/to/image2",...],
        "conversations": [
            {
                "from": "user",
                "value": "Picture 1: <image>\n<image>\n...\nDescribe the appearance of the pedestrian in the image. Include details such as color, make, model, and any visible features."
            },
            {
                "from": "gpt",
                "value": "{rewrite prompt}"
            }
        ]
    },
    ...
]
```