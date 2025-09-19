import os
import json
import argparse

def transform_submission_format(result_spatial_vqa_json, result_temporal_vqa_json):
    '''
    Args:
        result_spatial_vqa_json: Path to the JSON file containing the spatial VQA inference results.
        result_temporal_vqa_json: Path to the JSON file containing the temporal VQA inference results.
        
    Returns:
        res_vqa: A list of dictionaries containing the transformed VQA data in the required submission format.

    Example of res_vqa format:
    [
        {
            'id': 'a1',
            'correct': 'a'
        },
        {
            'id': 'b2',
            'correct': 'b'
        },
        ...
    ]
    '''

    res_vqa = []
    with open(result_spatial_vqa_json, 'r') as f:
        inference_spatial_vqa_json = json.load(f)['results']
    with open(result_temporal_vqa_json, 'r') as f:
        inference_temporal_vqa_json = json.load(f)['results']

    for sample in inference_spatial_vqa_json:
        id = sample['id_vqa']
        correct = sample['prediction'][0]

        res_vqa.append({
            'id': id,
            'correct': correct
        })

    for sample in inference_temporal_vqa_json:
        id = sample['sample']['id_vqa']
        correct = sample['prediction']
        
        if '\nassistant\n' in correct:
            correct = correct.split('\nassistant\n')[-1].strip()[0]
        else:
            correct = correct.strip()[0]

        res_vqa.append({
            'id': id,
            'correct': correct
        })

    return res_vqa
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_spatial_vqa_json', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/test_res/vqa_fixed_v2.json', help='Path to inferred spatial vqa')
    parser.add_argument('--result_temporal_vqa_json', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/test_res/vqa_not_fixed_withenhanced.json', help='Path to inferred temporal vqa')
    parser.add_argument('--submission_folder_path', type=str, default='/home/s48gb/Desktop/GenAI4E/aicity_02/submission/v4', help='Folder path to save VQA submission')

    args = parser.parse_args()

    res_vqa = transform_submission_format(args.result_spatial_vqa_json, args.result_temporal_vqa_json)

    os.makedirs(args.submission_folder_path, exist_ok=True)
    res_vqa_json_path = os.path.join(args.submission_folder_path, f'vqa.json')
    with open(res_vqa_json_path, 'w+') as f:
        json.dump(res_vqa, f, indent=4)
        print(f"Transformed caption data saved to {res_vqa_json_path}")