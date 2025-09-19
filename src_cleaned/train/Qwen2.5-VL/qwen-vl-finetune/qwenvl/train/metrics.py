"""
Metrics calculation module for Qwen-VL fine-tuning evaluation.
Location: qwenvl/train/metrics.py

This file should be created at: qwen-vl-finetune/qwenvl/train/metrics.py
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime   
import json
from transformers import EvalPrediction
import nltk
from pathlib import Path    
import torch
from nltk import bleu_score
from transformers import EvalPrediction, PreTrainedTokenizer
from nltk.tokenize import TreebankWordTokenizer
from rouge_score import rouge_scorer
import sys
import os
from multiprocessing import Pool, cpu_count

from evaluate import load



root_dir = os.path.dirname(
    os.path.abspath(
        os.path.join(
        __file__, '../../../../../'
        )
    )
)


def acc_score(predictions: list[str], references: list[str]) -> float:
    score = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            score += 1
    
    return score / len(predictions)

sys.path.insert(0, root_dir)

from evaluating.evaluation.cider.cider import Cider


try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpus/wordnet')
except:
    nltk.download('wordnet', quiet=True)


def tokenize_sentence(sentence):
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(sentence)
    if len(words) == 0:
        return ""
    return " ".join(words)


class ValidationMetricsSaver:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        save_validation_predictions: bool = True,
        save_predictions_every_n_evals: int = 2,
        max_validation_samples_to_save: int = 100,
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.save_validation_predictions = save_validation_predictions
        self.save_every = save_predictions_every_n_evals
        self.max_save = max_validation_samples_to_save
        self.ignore_index = ignore_index

        self.validation_dir = self.output_dir / "validation_results"
        if self.save_validation_predictions:
            self.validation_dir.mkdir(parents=True, exist_ok=True)

        self.eval_count = 0

        self._all_pred_ids: List[np.ndarray] = []
        self._all_label_ids: List[np.ndarray] = []

        
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.cider = Cider()
        self.bleu_eval = load("bleu")
        self.meteor_eval = load("meteor")
        self.acc_eval = load("accuracy")

    def set_eval_dataset(self, eval_dataset: Any):
        self._current_eval_dataset = eval_dataset

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        self.eval_count += 1

        print(f"🔁 [Eval #{self.eval_count}] compute_metrics() called")

        if isinstance(eval_pred.predictions, np.ndarray):
            preds = torch.from_numpy(eval_pred.predictions)
        else:
            preds = eval_pred.predictions
        token_ids = torch.argmax(preds, dim=-1).cpu().numpy()
        labels = eval_pred.label_ids

        print(f"🔎 Decoding predictions and labels...")

        decoded_preds = self._batch_decode(token_ids)
        decoded_labels = self._batch_decode_labels(labels)

        print(f"✅ Done decoding {len(decoded_preds)} predictions")

        bleu_res = self.bleu_eval.compute(
            predictions=decoded_preds,
            references=[[lbl] for lbl in decoded_labels]
        )
        meteor_res = self.meteor_eval.compute(
            predictions=decoded_preds,
            references=[[lbl] for lbl in decoded_labels]
        )
        acc_res = acc_score(
            predictions=decoded_preds,
            references=decoded_labels
        )

        rouge_l_vals = [
            self.rouge.score(p, g)["rougeL"].fmeasure
            for p, g in zip(decoded_preds, decoded_labels)
        ]
        rouge_l_mean = float(np.mean(rouge_l_vals)) if rouge_l_vals else 0.0

        cider_score = None
        if self.eval_count % 5 == 0:
            print("📈 Computing CIDEr (every 5 evals)...")
            cider_score, _ = self.cider.compute_score(decoded_labels, decoded_preds)
        else:
            print("⚠️ Skipping CIDEr computation (not 5th eval)")

        metrics: Dict[str, float] = {
            "bleu": bleu_res.get("bleu", 0.0),
            "meteor": meteor_res.get("meteor", 0.0),
            "rouge-l": rouge_l_mean,
            "accuracy": acc_res.get("accuracy", 0.0),
        }
        if cider_score is not None:
            metrics["cider"] = float(cider_score)

        if self.save_validation_predictions and (self.eval_count % self.save_every == 0):
            print(f"💾 Saving predictions at eval #{self.eval_count}")
            self._save_results(decoded_preds, decoded_labels, metrics)
        else:
            print(f"📉 Skipping prediction save at eval #{self.eval_count}")

        return metrics

    def _batch_decode(self, token_ids: np.ndarray) -> List[str]:
        seqs = []
        for seq in token_ids:
            seq = seq[seq != self.tokenizer.pad_token_id]
            seqs.append(list(seq))
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _batch_decode_labels(self, labels: np.ndarray) -> List[str]:
        seqs = []
        for seq in labels:
            seq = seq[seq != self.ignore_index]
            seq = seq[seq != self.tokenizer.pad_token_id]
            seqs.append(list(seq))
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _save_results(
        self,
        preds: List[str],
        labels: List[str],
        metrics: Dict[str, float]
    ) -> None:
        step_dir = self.validation_dir / f"eval_{self.eval_count}"
        step_dir.mkdir(parents=True, exist_ok=True)

        samples = [
            {"id": i, "prediction": p, "ground_truth": g}
            for i, (p, g) in enumerate(zip(preds, labels))
        ][: self.max_save]

        timestamp = datetime.now().isoformat()
        json_data = {
            "metadata": {
                "eval_count": self.eval_count,
                "timestamp": timestamp,
                "total": len(labels),
                "saved": len(samples)
            },
            "samples": samples,
            "overall": metrics
        }
        with open(step_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        with open(step_dir / "summary.txt", "w") as f:
            f.write(f"Evaluation {self.eval_count} @ {timestamp}\n")
            f.write("=" * 40 + "\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        print(f"✅ Saved validation results to {step_dir}")


def create_validation_metrics_saver(
    tokenizer: PreTrainedTokenizer,
    training_args: Any,
    eval_dataset: Any = None,
) -> ValidationMetricsSaver:
    saver = ValidationMetricsSaver(
        tokenizer=tokenizer,
        output_dir=training_args.output_dir,
        save_validation_predictions=getattr(training_args, 'save_validation_predictions', True),
        save_predictions_every_n_evals=getattr(training_args, 'save_predictions_every_n_evals', 2),
        max_validation_samples_to_save=getattr(training_args, 'max_validation_samples_to_save', 100),
        ignore_index=getattr(training_args, 'ignore_index', -100),
    )
    if eval_dataset is not None:
        saver.set_eval_dataset(eval_dataset)
    return saver
