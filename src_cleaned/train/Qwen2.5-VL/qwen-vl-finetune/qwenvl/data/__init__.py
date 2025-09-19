import re
import os
from pathlib import Path


CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

AICITYDATASET_TRAIN_FIXED_CAPTION = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/train_fixed_caption.json"
}

AICITYDATASET_TRAIN_NOT_FIXED_CAPTION = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/train_not_fixed_caption.json"
}

AICITYDATASET_TRAIN_FIXED_VQA = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/train_fixed_vqa.json"
}

AICITYDATASET_TRAIN_NOT_FIXED_VQA = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/train_not_fixed_vqa.json"
}

AICITYDATASET_VALID_FIXED_CAPTION = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_fixed_caption.json"
}

AICITYDATASET_VALID_NOT_FIXED_CAPTION = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_not_fixed_caption.json"
}

AICITYDATASET_VALID_FIXED_VQA = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_fixed_vqa.json"
}

AICITYDATASET_VALID_NOT_FIXED_VQA = {
    "annotation_path": "/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_not_fixed_vqa.json"
}



AICITYDATASET_TRAIN_FIXED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/train_fixed.json'
}


AICITYDATASET_VAL_FIXED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_fixed.json'
}



AICITYDATASET_TRAIN_FIXED_CAPTION_TOTAL = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/total_train_caption.json'
}


AICITYDATASET_VAL_FIXED_CAPTION_TOTAL = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/total_val_caption.json'
}



AICITYDATASET_TEST_FIXED_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/test_data/test_fixed_caption.json'
}


AICITYDATASET_TEST_FIXED_VQA = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/test_data/test_fixed_vqa.json'
}

AICITYDATASET_TEST_NOT_FIXED_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/test_data/test_not_fixed_caption.json'
}


AICITYDATASET_TEST_NOT_FIXED_VQA = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/test_data/test_not_fixed_vqa.json'
}

AICITYDATASET_TEST_TOTAL_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/test/test_data/test_merge_caption.json'
}



AICITYDATASET_TRAIN_NOT_FIXED_VQA_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/train_not_fixed_vqa_enhanced.json'
}
AICITYDATASET_VAL_NOT_FIXED_VQA_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_not_fixed_vqa_enhanced.json'
}

AICITYDATASET_VAL_TOTAL_CAPTION_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/total_val_caption_enhanced.json'
}

AICITYDATASET_VAL_DIRECTION_NOT_FIXED_VQA_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/debug/val_not_fixed_vqa_enhanced.json'
}

AICITYDATASET_TRAINVAL_FIXED_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/trainval_fixed_caption.json'
}

AICITYDATASET_TRAINVAL_NOT_FIXED_CAPTION_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/trainval_not_fixed_caption_enhanced.json'
}

AICITYDATASET_VAL_NOT_FIXED_CAPTION_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_not_fixed_caption_enhanced.json'
}

AICITYDATASET_VAL_MERGE_CAPTION_ENHANCED_FROM_INFER = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_merge_caption_enhanced_from_infer.json'
}
AICITYDATASET_TRAINVAL_MERGE_CAPTION_ENHANCED_FROM_GT = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/total_trainval_caption_enhanced.json'
}

AICITYDATASET_DEBUG_VQA_NOT_FIXED_ENHANCED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/debug/vqa_not_fixed_debug.json'
}

ABLA_TRAIN_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/ablation/train_caption.json'
}
ABLA_VAL_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/ablation/val_caption.json'
}

AICITYDATASET_VAL_TOTAL_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/total_val_caption.json'
}

VAL_MERGE_CAPTION_ENHANCED_FROM_INFER = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/train_data/val_merge_caption_enhanced_from_infer.json'
}

VQA_PART1 = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/ablation/val_vqa_1.json'
}

ABLATION_FRAME_SELECTION_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/frameselection/test.json'
}

ABLATION_TEXTUAL_PROMPT_FIXED_CAPTION_NO_PED = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_fixed_cap_no_ped.json'
}

ABLATION_TEXTUAL_PROMPT_FIXED_CAPTION_NO_ENV = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_fixed_cap_no_env.json'
}

ABLATION_TEXTUAL_PROMPT_NOT_FIXED_CAPTION_NO_ACTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_not_fixed_cap_no_act.json'
}
ABLATION_TEXTUAL_PROMPT_NOT_FIXED_CAPTION_NO_ATTENTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_not_fixed_cap_no_attn.json'
}
ABLATION_TEXTUAL_PROMPT_NOT_FIXED_CAPTION_NO_LOCATION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_not_fixed_cap_no_loc.json'
}

ABLATION_FULL_FIXED_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/train_data/transformed_val_spatial_caption.json'
}

ABLATION_FULL_NOT_FIXED_CAPTION = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/preprocessed_dataset/cleaned/train_data/transformed_val_temporal_caption.json'
}

NOT_FIXED_NO_TEXTUAL = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_no_textual_again.json'
}

NOT_FIXED_NO_VISUAL = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_no_visual_again.json'
}

FIXED_NO_TEXTUAL = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_fixed_no_textual_again.json'
}

FIXED_NO_VISUAL = {
    'annotation_path': '/home/s48gb/Desktop/GenAI4E/aicity_02/local/ablationPrompt/val_fixed_no_visual_again.json'
}


data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "aicitydataset_train_fixed_caption": AICITYDATASET_TRAIN_FIXED_CAPTION,
    "aicitydataset_train_not_fixed_caption": AICITYDATASET_TRAIN_NOT_FIXED_CAPTION,
    "aicitydataset_train_fixed_vqa": AICITYDATASET_TRAIN_FIXED_VQA,
    "aicitydataset_train_not_fixed_vqa": AICITYDATASET_TRAIN_NOT_FIXED_VQA,
    "aicitydataset_val_fixed_caption": AICITYDATASET_VALID_FIXED_CAPTION,
    "aicitydataset_val_not_fixed_caption": AICITYDATASET_VALID_NOT_FIXED_CAPTION,
    "aicitydataset_val_fixed_vqa": AICITYDATASET_VALID_FIXED_VQA,
    "aicitydataset_val_not_fixed_vqa": AICITYDATASET_VALID_NOT_FIXED_VQA,

    "aicitydataset_train_fixed": AICITYDATASET_TRAIN_FIXED,
    "aicitydataset_val_fixed": AICITYDATASET_VAL_FIXED,

    "aicitydataset_train_fixed_caption_total": AICITYDATASET_TRAIN_FIXED_CAPTION_TOTAL,
    "aicitydataset_val_fixed_caption_total": AICITYDATASET_VAL_FIXED_CAPTION_TOTAL,

    "aicitydataset_test_fixed_caption": AICITYDATASET_TEST_FIXED_CAPTION,
    'aicitydataset_test_fixed_vqa': AICITYDATASET_TEST_FIXED_VQA,

    "aicitydataset_test_not_fixed_caption": AICITYDATASET_TEST_NOT_FIXED_CAPTION,
    "aicitydataset_test_not_fixed_vqa": AICITYDATASET_TEST_NOT_FIXED_VQA,

    'aicitydataset_test_total_caption': AICITYDATASET_TEST_TOTAL_CAPTION,

    'aicitydataset_val_total_caption_enhanced': AICITYDATASET_VAL_TOTAL_CAPTION_ENHANCED,

    'aicitydataset_train_not_fixed_vqa_enhanced': AICITYDATASET_TRAIN_NOT_FIXED_VQA_ENHANCED,
    'aicitydataset_val_not_fixed_vqa_enhanced': AICITYDATASET_VAL_NOT_FIXED_VQA_ENHANCED,

    'aicitydataset_val_direction_not_fixed_vqa_enhanced': AICITYDATASET_VAL_DIRECTION_NOT_FIXED_VQA_ENHANCED,

    'aicitydataset_trainval_fixed_caption': AICITYDATASET_TRAINVAL_FIXED_CAPTION,
    'aicitydataset_trainval_not_fixed_caption_enhanced': AICITYDATASET_TRAINVAL_NOT_FIXED_CAPTION_ENHANCED,

    'aicitydataset_val_not_fixed_caption_enhanced': AICITYDATASET_VAL_NOT_FIXED_CAPTION_ENHANCED,
    
    'aicitydataset_val_merge_caption_enhanced_from_infer': AICITYDATASET_VAL_MERGE_CAPTION_ENHANCED_FROM_INFER,
    'aicitydataset_debug_vqa_not_fixed_enhanced': AICITYDATASET_DEBUG_VQA_NOT_FIXED_ENHANCED,

    'aicitydataset_trainval_merge_caption_enhanced_from_gt': AICITYDATASET_TRAINVAL_MERGE_CAPTION_ENHANCED_FROM_GT,

    'ablation_train_caption': ABLA_TRAIN_CAPTION,
    'ablation_val_caption': ABLA_VAL_CAPTION,

    'aicitydataset_val_total_caption': AICITYDATASET_VAL_TOTAL_CAPTION,
    'val_merge_caption_enhanced_from_infer': VAL_MERGE_CAPTION_ENHANCED_FROM_INFER,

    'vqa_part1': VQA_PART1,

    'ablation_frame_selection_caption': ABLATION_FRAME_SELECTION_CAPTION,
    'ablation_textual_prompt_fixed_caption_no_ped': ABLATION_TEXTUAL_PROMPT_FIXED_CAPTION_NO_PED,
    'ablation_textual_prompt_fixed_caption_no_env': ABLATION_TEXTUAL_PROMPT_FIXED_CAPTION_NO_ENV,
    'ablation_full_fixed_caption': ABLATION_FULL_FIXED_CAPTION,

    'ablation_textual_prompt_not_fixed_caption_no_action': ABLATION_TEXTUAL_PROMPT_NOT_FIXED_CAPTION_NO_ACTION,
    'ablation_textual_prompt_not_fixed_caption_no_attention': ABLATION_TEXTUAL_PROMPT_NOT_FIXED_CAPTION_NO_ATTENTION,
    'ablation_textual_prompt_not_fixed_caption_no_location': ABLATION_TEXTUAL_PROMPT_NOT_FIXED_CAPTION_NO_LOCATION,
    'ablation_full_not_fixed_caption': ABLATION_FULL_NOT_FIXED_CAPTION,

    'not_fixed_no_textual': NOT_FIXED_NO_TEXTUAL,
    'not_fixed_no_visual': NOT_FIXED_NO_VISUAL,

    'fixed_no_textual': FIXED_NO_TEXTUAL,
    'fixed_no_visual': FIXED_NO_VISUAL,
    
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def validate_local_dataset(config):
    annotation_path = config["annotation_path"]
    data_path = config["data_path"]

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    print(f"✓ Local dataset validated: {annotation_path}")
    return True





def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            if config.get("dataset_type") == "local_multi_image":
                validate_local_dataset(config)
            config_list.append(config)

            
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["aicitydataset_train_fixed_caption",
                    "aicitydataset_train_not_fixed_caption",
                    "aicitydataset_train_fixed_vqa",
                    "aicitydataset_train_not_fixed_vqa",
                    "aicitydataset_val_fixed_caption",
                    "aicitydataset_val_not_fixed_caption",
                    "aicitydataset_val_fixed_vqa",
                    "aicitydataset_val_not_fixed_vqa",
                    "aicitydataset_test_total_caption",
                    "aicitydataset_val_not_fixed_vqa_enhanced",
                    "aicitydataset_val_total_caption_enhanced",
                    "aicitydataset_train_not_fixed_vqa_enhanced",
                    "aicitydataset_val_direction_not_fixed_vqa_enhanced",
                    "aicitydataset_trainval_fixed_caption",
                    "aicitydataset_trainval_not_fixed_caption_enhanced",
                    "aicitydataset_val_not_fixed_caption_enhanced",
                    "aicitydataset_val_merge_caption_enhanced_from_infer",
                    'aicitydataset_debug_vqa_not_fixed_enhanced',
                    "aicitydataset_trainval_merge_caption_enhanced_from_gt",
                    "ablation_train_caption",
                    "ablation_val_caption",
                    "aicitydataset_val_total_caption",
                    "val_merge_caption_enhanced_from_infer",
                    "vqa_part1",
                    "ablation_frame_selection_caption",
                    "ablation_textual_prompt_fixed_caption_no_ped",
                    "ablation_textual_prompt_fixed_caption_no_env",
                    "ablation_textual_prompt_not_fixed_caption_no_action",
                    "ablation_textual_prompt_not_fixed_caption_no_attention",
                    "ablation_textual_prompt_not_fixed_caption_no_location",
                    'ablation_full_fixed_caption',
                    'ablation_full_not_fixed_caption',
                    "not_fixed_no_textual",
                    "not_fixed_no_visual",
                    "fixed_no_textual",
                    "fixed_no_visual",
                    ]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
