import sys
import json
import argparse
from pathlib import Path
from evaluate import load
from tqdm import tqdm

import nltk
from nltk import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import TreebankWordTokenizer


ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT_DIR))

print(ROOT_DIR)
from evaluating.evaluation.cider.cider import Cider



try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpus/wordnet')
except:
    nltk.download('wordnet', quiet=True)






def load_entries(predictions_path: Path) -> list[dict[str, str]]:
    with predictions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data['results']
    filtered = []
    for e in tqdm(results, desc='Reading prediction/gt...'):
        pred = e.get("prediction", "").strip()

        gt = e['sample']['conversations'][1]['value']
        ped_or_veh = e['sample'].get("ped_or_veh", "ped")
        internal_or_external = "external" if e['image_paths'][0].split('/')[-1].startswith("video") else "internal"
        # gt = e.get("ground_truth", "").strip()
        if pred and gt:
            filtered.append({"prediction": pred, "ground_truth": gt, "ped_or_veh": ped_or_veh, "internal_or_external": internal_or_external})

    print(f"{len(filtered)=}")
    return filtered



def acc_score(predictions: list[str], references: list[str]) -> float:
    score = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            score += 1
    
    print("Number of correct predictions:", score)
    return score / len(predictions)


def tokenize_sentence(sentence):
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(sentence)
    if len(words) == 0:
        return ""
    return " ".join(words)


# Compute BLEU-4 score on a single sentence
def compute_bleu_single(tokenized_hypothesis, tokenized_reference):
    # convert tokenized sentence (joined by spaces) into list of words
    tokenized_hypothesis = tokenized_hypothesis.split(" ")
    tokenized_reference = tokenized_reference.split(" ")

    return sentence_bleu([tokenized_reference], tokenized_hypothesis,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=bleu_score.SmoothingFunction().method3)


# Compute METEOR score on a single sentence
def compute_meteor_single(tokenized_hypothesis, tokenized_reference):
    # convert tokenized sentence (joined by spaces) into list of words
    tokenized_hypothesis = tokenized_hypothesis.split(" ")
    tokenized_reference = tokenized_reference.split(" ")

    return meteor_score([tokenized_reference], tokenized_hypothesis)


# Compute ROUGE-L score on a single sentence
def compute_rouge_l_single(sentence_hypothesis, sentence_reference):
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = rouge_l_scorer.score(sentence_hypothesis, sentence_reference)
    rouge_l_score = rouge_score['rougeL']
    return rouge_l_score.fmeasure


# Compute CIDEr score on a single sentence
def compute_cider_single(sentence_hypothesis, sentence_reference):
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score([sentence_reference], [sentence_hypothesis])

    return cider_score


# Compute metrics based for a single caption.
def compute_metrics_single(pred, gt):
    tokenized_pred = tokenize_sentence(pred)
    tokenized_gt = tokenize_sentence(gt)

    bleu_score = compute_bleu_single(tokenized_pred, tokenized_gt)
    meteor_score = compute_meteor_single(tokenized_pred, tokenized_gt)
    rouge_l_score = compute_rouge_l_single(pred, gt)
    cider_score = compute_cider_single([tokenized_pred], [tokenized_gt])

    return {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "rouge-l": rouge_l_score,
        "cider": cider_score,
    }

def compute_metrics_from_predictions(
    predictions_path: Path,
    output_file: Path,  # model_path removed because unused,
    type_: str,
    dataset: str = 'internal'
) -> dict[str, float]:
    """
    Load raw predictions, compute BLEU, METEOR, ROUGE-L, CIDEr, and Accuracy.

    Args:
        predictions_path: JSON file with [{"prediction": str, "ground_truth": str}, ...].
        output_dir: Directory where metric summaries could be saved (if needed).
    Returns:
        A dict mapping metric names to their float scores.
    """


    output_file.parent.mkdir(parents=True, exist_ok=True)

    

    entries = load_entries(predictions_path)

    ped_idx = [1] * len(entries)
    veh_idx = [0] * len(entries)
    
    for i, e in enumerate(entries):
        if e["ped_or_veh"] == "veh":
            ped_idx[i] = 0
            veh_idx[i] = 1

    for i, e in enumerate(entries):
        if e["internal_or_external"] != dataset:
            ped_idx[i] = 0
            veh_idx[i] = 0

    ped_predictions = [e["prediction"] for i, e in enumerate(entries) if ped_idx[i]]
    ped_references = [e["ground_truth"] for i, e in enumerate(entries) if ped_idx[i]]

    ped_metrics = {}
    for i, (pred, gt) in tqdm(enumerate(zip(ped_predictions, ped_references)), total=len(ped_predictions), desc='Computing ped_metrics...'):
        single_metrics = compute_metrics_single(pred, gt)
        for key, value in single_metrics.items():
            if key not in ped_metrics:
                ped_metrics[key] = []
            ped_metrics[key].append(value)

    for key in ped_metrics:
        ped_metrics[key] = sum(ped_metrics[key]) / len(ped_metrics[key])

    ped_metrics["avg_score"] = (ped_metrics["bleu"] + ped_metrics["meteor"] + ped_metrics["rouge-l"] + ped_metrics["cider"] * 0.1) / 4 * 100


    veh_references = [e["ground_truth"] for i, e in enumerate(entries) if veh_idx[i]]
    veh_predictions = [e["prediction"] for i, e in enumerate(entries) if veh_idx[i]]
    veh_metrics = {}
    for i, (pred, gt) in tqdm(enumerate(zip(veh_predictions, veh_references)), total=len(veh_predictions), desc='Computing veh_metrics...'):
        single_metrics = compute_metrics_single(pred, gt)
        for key, value in single_metrics.items():
            if key not in veh_metrics:
                veh_metrics[key] = []
            veh_metrics[key].append(value)

    for key in veh_metrics:
        veh_metrics[key] = sum(veh_metrics[key]) / len(veh_metrics[key])

    if veh_metrics:
        veh_metrics["avg_score"] = (veh_metrics["bleu"] + veh_metrics["meteor"] + veh_metrics["rouge-l"] + veh_metrics["cider"] * 0.1) / 4 * 100

    metrics = {}
    for key in ped_metrics:
        if key != "avg_score":
            if key not in veh_metrics:
                metrics[key] = ped_metrics[key]
            else:
                metrics[key] = (ped_metrics[key] + veh_metrics[key]) / 2.0
    
    metrics["ped_avg_score"] = ped_metrics["avg_score"]
    metrics["veh_avg_score"] = veh_metrics["avg_score"] if veh_metrics else 0.0

    metrics["avg_score"] = (metrics["bleu"]*100 + metrics["meteor"]*100 + metrics["rouge-l"]*100 + metrics["cider"] * 10) / 4


    # predictions = [tokenize_sentence(e["prediction"]) for e in entries]
    # references = [tokenize_sentence(e["ground_truth"]) for e in entries]

    # bleu = load("bleu")
    # meteor = load("meteor")
    # rouge = load("rouge")  
    # cider_scorer = Cider()

    # bleu_res = bleu.compute(predictions=predictions, references=[[r] for r in references])
    # bleu_score = bleu_res.get("bleu", 0.0)

    # print(f"BLEU SCORE: {bleu_score}")

    # meteor_res = meteor.compute(predictions=predictions, references=[[r] for r in references])
    # meteor_score = meteor_res.get("meteor", 0.0)
    # print(f"METEOR SCORE: {meteor_res}")


    acc_res = 0
    if type_ != 'cap':
        predictions = [e["prediction"] for e in entries]
        references = [e["ground_truth"] for e in entries]
        
        print("Computing ACC score...")
        acc_res = acc_score(predictions=predictions, references=references)
    
        print(f"ACC SCORE: {acc_res}")

    # rouge_res = rouge.compute(predictions=predictions, references=references)
    # rouge_l_score = rouge_res.get("rougeL", 0.0)
    # print(f"ROUGE SCORE: {rouge_res}")

   
    # cider_score, _ = cider_scorer.compute_score(
    #     gts=[[r] for r in references],
    #     res=[[p] for p in predictions]
    # )
    # print(f"CIDER SCORE: {cider_score}")


    # metrics =  {
    #     "bleu": float(bleu_score),
    #     "meteor": float(meteor_score),
    #     "rouge-l": float(rouge_l_score),
    #     "cider": float(cider_score),   
    # }

    if type_ != 'cap':
        metrics.update({
            'acc_score': acc_res
        })


    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute text-generation metrics from saved predictions."
    )
    parser.add_argument(
        "--predictions_path",
        type=Path,
        required=True,
        help="Path to JSON file with raw predictions."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Filename where metric summaries will be saved."
    )
    parser.add_argument(
        "--type_",
        default='cap',
        help="Caption (cap) or VQA (acc)."
    )
    args = parser.parse_args()

    internal_metrics = compute_metrics_from_predictions(
        predictions_path=args.predictions_path,
        output_file=args.output_file,
        type_ = args.type_,
        dataset='internal'
    )

    external_metrics = compute_metrics_from_predictions(
        predictions_path=args.predictions_path,
        output_file=args.output_file.with_suffix('.external.json'),
        type_ = args.type_,
        dataset='external'
    )

    metrics = {}

    for key in internal_metrics:
        metrics[key + "_internal"] = internal_metrics[key]

    for key in external_metrics:
        metrics[key + "_external"] = external_metrics[key]

    metrics["mean_score"] = (metrics["avg_score_internal"] + metrics["avg_score_external"]) / 2.0
    print("\n🎯 Evaluation Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name.upper():<10}: {value:.4f}")


    with open(str(args.output_file), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()