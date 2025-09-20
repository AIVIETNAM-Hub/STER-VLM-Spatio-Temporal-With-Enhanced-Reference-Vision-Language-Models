import sys
import json
import argparse
from pathlib import Path
from evaluate import load
import nltk
from tqdm import tqdm



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
        pred = pred.split('\nassistant\n')[1].strip()

        gt = e['sample']['conversations'][1]['value']
        # gt = e.get("ground_truth", "").strip()
        if pred and gt:
            filtered.append({"prediction": pred, "ground_truth": gt})
        
    print(f"{len(filtered)=}")
    return filtered



def acc_score(predictions: list[str], references: list[str]) -> float:
    score = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            score += 1
    
    print("Number of correct predictions:", score)
    return score / len(predictions)




def compute_metrics_from_predictions(
    predictions_path: Path,
    output_file: Path,  # model_path removed because unused,
    type_: str
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

    predictions = [e["prediction"] for e in entries]
    references = [e["ground_truth"] for e in entries]

    bleu = load("bleu")
    meteor = load("meteor")
    rouge = load("rouge")  
    cider_scorer = Cider()

    bleu_res = bleu.compute(predictions=predictions, references=[[r] for r in references])
    bleu_score = bleu_res.get("bleu", 0.0)

    print(f"BLEU SCORE: {bleu_score}")

    meteor_res = meteor.compute(predictions=predictions, references=[[r] for r in references])
    meteor_score = meteor_res.get("meteor", 0.0)
    print(f"METEOR SCORE: {meteor_res}")


    acc_res = 0
    if type_ != 'cap':

        acc_res = acc_score(predictions=predictions, references=references)
    
        print(f"ACC SCORE: {acc_res}")

    rouge_res = rouge.compute(predictions=predictions, references=references)
    rouge_l_score = rouge_res.get("rougeL", 0.0)
    print(f"ROUGE SCORE: {rouge_res}")

   
    cider_score, _ = cider_scorer.compute_score(
        gts=[[r] for r in references],
        res=[[p] for p in predictions]
    )
    print(f"CIDER SCORE: {cider_score}")


    metrics =  {
        "bleu": float(bleu_score),
        "meteor": float(meteor_score),
        "rouge-l": float(rouge_l_score),
        "cider": float(cider_score),   
    }

    if type_ != 'cap':
        metrics.update({
            'acc_score': acc_res
        })

    

    with open(str(output_file), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


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
        help="Filename where metric summaries will be saved."
    )
    args = parser.parse_args()

    metrics = compute_metrics_from_predictions(
        predictions_path=args.predictions_path,
        output_file=args.output_file,
        type_ = args.type_
    )

    print("\n🎯 Evaluation Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name.upper():<10}: {value:.4f}")


if __name__ == "__main__":
    main()