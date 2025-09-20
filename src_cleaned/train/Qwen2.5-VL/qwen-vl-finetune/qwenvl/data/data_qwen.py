import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
import transformers
from typing import Any
from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None




def pad_sequence_left(sequences, batch_first=False, padding_value=0):
    """
    Pad sequences to the same length with left padding.
    Fixed version that properly handles all sequences.
    """
    max_len = max(seq.size(0) for seq in sequences)
    
    if batch_first:
        out_dims = (len(sequences), max_len) + sequences[0].shape[1:]
    else:
        out_dims = (max_len, len(sequences)) + sequences[0].shape[1:]

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, max_len - length:] = tensor
        else:
            out_tensor[max_len - length:, i] = tensor
    
    return out_tensor

def pad_and_cat_left(tensor_list):
    max_length = max(
        tensor.shape[2] for tensor in tensor_list
    )
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (pad_length, 0), "constant", 1)
        padded_tensors.append(padded_tensor)
    
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor



def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    


def validate_local_sample(
    sample, sample_index=0, max_images=5
):
    sample_id = sample.get('id', f"sample_{sample_index}")
    required_fields = ['images', 'conversations']
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing required field '{field}' in sample {sample_id}")

    images = sample['images']
    if not isinstance(images, list):
        raise ValueError(f"'images' must be a list, got {type(images)} in sample {sample_id}")
    
    if len(images) < 1 or len(images) > max_images:
        raise ValueError(f"Number of images must be 1-{max_images}, got {len(images)} in sample {sample_id}")


    conversations = sample["conversations"]
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"'conversations' must be a list with at least 2 entries in sample {sample_id}")
    
    human_message = next((conv for conv in conversations if conv.get("from") == "human"), None)
    if human_message:
        image_token_count = human_message['value'].count('<image>')
        if image_token_count != len(images):
            rank0_print(f"Warning: Image token count ({image_token_count}) doesn't match number of images ({len(images)}) in sample {sample_id}")
    
    return True

def process_local_dataset(annotations: list[dict], max_images=5) -> None:
    for i, sample in enumerate(annotations):
        try:
            validate_local_sample(sample, sample_index=i, max_images=max_images)
        except Exception as e:
            rank0_print(f"Error processing sample {i}: {e}")
            continue
    


    



def preprocess_qwen_2_visual(
    sources,
    tokenizer,
    grid_thw_image: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])

                    
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

               

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

   
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def inference_preprocess_qwen_2_visual(
    sources: list[list[dict]],
    tokenizer,
    grid_thw_image: list[int]
) -> dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids = []

    for i, source in enumerate(sources):
        messages_for_template = [{"role": "system", "content": system_message}]
        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            
            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for k in range(len(parts) - 1):
                        new_parts.append(parts[i])

                    
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            if role == 'assistant':
                continue
            messages_for_template.append({"role": role, "content": content})

        encoded_prompt = tokenizer.apply_chat_template(
            messages_for_template,
            add_generation_prompt=True,
            tokenize=True
        )
        input_ids.append(encoded_prompt)
    
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    return dict(input_ids=input_ids_tensor)







class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args,inference_mode:bool=False):
        super(LazySupervisedDataset, self).__init__()

        self.inference_mode = inference_mode

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))


            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            ####!TODO: TEMPORALY REMOVE MISMATCH BETWEEN List Of Images and image token

            filtered_annotations = []
            for ann in annotations:
                if 'images' in ann:
                    image_count = len(ann['images'])
                    image_token_count = 0
                    for conv in ann["conversations"]:
                        if conv.get("from") == "user":
                            image_token_count += conv.get("value", "").count("<image>")
                    if image_count != image_token_count:
                     
                        continue
                        
                    if image_token_count == 0:
                        continue
                        
                
                filtered_annotations.append(ann)
            
          

            list_data_dict += filtered_annotations

            ###########################!TODO: 

        

        print(f"Total training samples: {len(list_data_dict)}")

        if not self.inference_mode:
            random.shuffle(list_data_dict)  

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

      

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        for attempt_idx in range(num_base_retries):
            
                sample = self._get_item(i)
                return sample
        

        for attempt_idx in range(num_base_retries):
            
            next_index = min(i + 1, len(self.list_data_dict) - 1)
            sample = self._get_item(next_index)
            return sample
       

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME


        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        
     
        image_files = self.list_data_dict[i]["images"]
        
        processed_images = []
        processed_grid_thws = []

        
        for img_file in image_files:
            img_tensor, img_grid_thw = self.process_image_unified(img_file)
            processed_images.append(img_tensor)
            processed_grid_thws.append(img_grid_thw)
        
        image = processed_images
        grid_thw = processed_grid_thws

        grid_thw_merged = copy.deepcopy(grid_thw)
        grid_thw_merged = [
            merged_thw.prod() // self.data_args.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]

        
        if grid_thw_merged is None:
            print(self.list_data_dict[i])
            raise ValueError("dnaosodnad")

        
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        
        preprocess_inputs = {
            'sources': chat_sources,
            'tokenizer': self.tokenizer,
            'grid_thw_image': grid_thw_merged
        }
        data_dict = preprocess_qwen_2_visual(**preprocess_inputs) if not self.inference_mode else inference_preprocess_qwen_2_visual(**preprocess_inputs)


        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
       

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        data_dict["pixel_values"] = torch.cat(image, dim=0)
        data_dict["image_grid_thw"] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thw], dim=0
        )
    
        

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        if getattr(self.tokenizer, 'padding_side', 'right') == 'left':
            input_ids = pad_sequence_left(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = pad_sequence_left(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            position_ids = pad_and_cat_left(position_ids)
        else:

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch

@dataclass
class DataCollatorForInference(object):
    """Collate examples for inference (no labels)."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst['input_ids'].squeeze(0) for inst in instances]
        position_ids = [inst['position_ids'] for inst in instances]

        if getattr(self.tokenizer, 'padding_side', 'right') == 'left':
            input_ids = pad_sequence_left(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            position_ids = pad_and_cat_left(position_ids)
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        batch = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
        }
        images = [inst['pixel_values'] for inst in instances if 'pixel_values' in inst]
        if images:
            batch['pixel_values'] = torch.cat(images, dim=0)
            grid_thw = [inst['image_grid_thw'] for inst in instances if 'image_grid_thw' in inst]
            batch['image_grid_thw'] = torch.cat(grid_thw, dim=0)

       

        return batch



@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch

def split_dataset(data_list, train_ratio=0.9, val_ratio=0.1, seed=42, shuffle=True, dataset_percentage=1.0):
    """
    Split dataset into train and validation sets with flexible ratio control

    Args:
        data_list: List of data samples
        train_ratio: Ratio for training data (0.0-1.0)
        val_ratio: Ratio for validation data (0.0-1.0)  
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        dataset_percentage: Percentage of total dataset to use (0.0-1.0)
    
    Returns:
        tuple(train_data, val_data)
    
    Note: train_ratio and valid_ratio are applied to the selected data percentage -> No sum up to 1. Remaining data is discarded
    """    

    if train_ratio < 0 or train_ratio > 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if val_ratio < 0 or val_ratio > 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if dataset_percentage <= 0 or dataset_percentage > 1:
        raise ValueError(f"dataset_percentage must be between 0 and 1, got {dataset_percentage}")


    total_samples = len(data_list)
    if total_samples == 0:
        return [], []

    if dataset_percentage < 1.0:
        import random
        random.seed(seed)
        selected_size = int(total_samples * dataset_percentage)

        if shuffle:
            data_list = data_list.copy()
            random.shuffle(data_list)
        
        data_list = data_list[:selected_size]
        rank0_print(f"🔍 Using {dataset_percentage*100:.1f}% of dataset: {selected_size}/{total_samples} samples")
    elif shuffle:
        import random
        random.seed(seed)
        data_list = data_list.copy()
        random.shuffle(data_list)
    
    selected_samples = len(data_list)

    train_size = int(selected_samples * train_ratio)
    val_size = int(selected_samples * val_ratio)

    train_data = data_list[:train_size]
    val_data = data_list[:val_size]

    actual_train_ratio = len(train_data) / selected_samples
    actual_val_ratio = len(val_data) / selected_samples
    unused_ratio = 1.0 - actual_train_ratio - actual_val_ratio

    rank0_print(f"   Dataset split summary:")
    rank0_print(f"   Original total: {total_samples} samples")
    rank0_print(f"   Selected: {selected_samples} samples ({dataset_percentage*100:.1f}%)")
    rank0_print(f"   Train: {len(train_data)} samples ({actual_train_ratio*100:.1f}% of selected)")
    rank0_print(f"   Validation: {len(val_data)} samples ({actual_val_ratio*100:.1f}% of selected)")
    if unused_ratio > 0:
        rank0_print(f"   Unused: {selected_samples - len(train_data) - len(val_data)} samples ({unused_ratio*100:.1f}% of selected)")
    rank0_print(f"   Split seed: {seed}")
    
    return train_data, val_data






def load_separate_validation_dataset(data_args):
    if not data_args.val_dataset_use:
        raise ValueError("val_dataset_use must be specified when using separate validation dataset")

    val_dataset_names = data_args.val_dataset_use.split(",")
    val_dataset_list = data_list(val_dataset_names)

    val_data_dict = []

    for data in val_dataset_list:
        annotations = json.load(open(data["annotation_path"], "r"))

        process_local_dataset(annotations=annotations, max_images=getattr(data_args, 'max_images_per_sample', 6))

        sampling_rate = data.get("sampling_rate", 1.0)
        if sampling_rate < 1.0:
            annotations = random.sample(
                annotations, int(len(annotations) * sampling_rate)
            )
            rank0_print(f"Sampled {len(annotations)} validation examples from dataset {data}")
        
        val_data_dict += annotations

    return val_data_dict







def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning with flexible validation options."""
    
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    eval_dataset = None
    
    validation_mode = getattr(data_args, 'validation_mode', 'split')
    skip_validation = getattr(data_args, 'skip_validation', False)
    
    if skip_validation or validation_mode == 'none':
        rank0_print("Validation disabled")
        eval_dataset = None
        
    elif validation_mode == 'separate':
        rank0_print("Using separate validation dataset")
        val_data = load_separate_validation_dataset(data_args)
        
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        eval_dataset.list_data_dict = val_data
        
        if data_args.validation_subset_size is not None and data_args.validation_subset_size > 0:
            subset_size = min(data_args.validation_subset_size, len(val_data))
            eval_dataset.list_data_dict = val_data[:subset_size]
            rank0_print(f"Limited validation to {subset_size} samples")
            
    elif validation_mode == 'split':
        rank0_print("Auto-splitting dataset for validation")
        
        dataset_percentage = getattr(data_args, 'dataset_percentage', 1.0)
        train_ratio = getattr(data_args, 'train_split_ratio', 0.9)
        val_ratio = getattr(data_args, 'val_split_ratio', 0.1)
        
        if val_ratio <= 0:
            rank0_print("Validation ratio is 0, disabling validation")
            eval_dataset = None
        else:
            train_data, val_data = split_dataset(
                train_dataset.list_data_dict,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=data_args.validation_split_seed,
                shuffle=data_args.shuffle_before_split,
                dataset_percentage=dataset_percentage
            )
            
            train_dataset.list_data_dict = train_data
            
            if len(val_data) > 0:
                eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
                eval_dataset.list_data_dict = val_data
                
                if data_args.validation_subset_size is not None and data_args.validation_subset_size > 0:
                    subset_size = min(data_args.validation_subset_size, len(val_data))
                    eval_dataset.list_data_dict = val_data[:subset_size]
                    rank0_print(f"🔍 Limited validation to {subset_size} samples")
            else:
                rank0_print("No validation data after split")
                eval_dataset = None
    else:
        raise ValueError(f"Unknown validation_mode: {validation_mode}. Use 'split', 'separate', or 'none'")
    
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    rank0_print(f"Data collator created with tokenizer padding_side: {tokenizer.padding_side}")

    return dict(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        data_collator=data_collator
    )


if __name__ == "__main__":
    pass
