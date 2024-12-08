from typing import List, Dict
import json
import pandas as pd
import re
import numpy as np
from numpy import ndarray
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util


class TLQAEvaluation:
    """
    TLQAEvaluation evaluates LLM outputs for timeline-based question answering tasks.
    """

    def __init__(self, test_data: str):
        """
        Initialize with test data from JSON file.
        """
        self.test_data = self.load_test_data(test_data)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Possible metrics
        self.metrics = {
            "rouge": self.rouge_scores,
            "bleu": self.bleu_score,
            "f1": self.f1_score,
            "exact_match": self.exact_match,
            "completeness": self.completeness,
            "temporal_alignment": self.temporal_alignment,
            "temporal_ordering": self.temporal_ordering,
        }

    @staticmethod
    def load_test_data(file_path: str) -> pd.DataFrame:
        """
        Load the test data from a JSON file into a DataFrame.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalizes the input text by removing punctuation, special characters, extra whitespace,
        and converting all characters to lowercase.
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def evaluate(self, predictions: List[Dict], metric: str, normalize: bool = False):
        """
        General evaluation method for various metrics.
        Normalize is required for f1, EM, and completeness.
        Args:
            predictions (List[Dict]): List of predicted answers as dictionaries.
            metric (str): The metric to evaluate (e.g., 'cosine_similarity', 'rouge', 'f1').
            normalize (bool): Whether to normalize the text before evaluation.
            output_file (str): File to save the results, ground truth, and predictions.

        Returns:
            float or dict: The computed metric score(s).
        """
        if metric in ["rouge", "bleu"]:
            # Use joined strings for these metrics
            ground_truths = ["\n".join(gt["final_answers"]) for gt in self.test_data.to_dict(orient="records")]
            predictions = ["\n".join(pred["answers"]) for pred in predictions]

        elif metric in ["f1", "exact_match", "temporal_alignment", "temporal_ordering", "completeness"]:
            # Use lists for these metrics
            ground_truths = [gt["final_answers"] for gt in self.test_data.to_dict(orient="records")]
            predictions = [pred["answers"] for pred in predictions]

            # Normalize if required
            if normalize:
                ground_truths = [[self.normalize_text(answer) for answer in gt] for gt in ground_truths]
                predictions = [[self.normalize_text(answer) for answer in pred] for pred in predictions]
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Compute the metric
        results = self.metrics[metric](predictions, ground_truths)

        return results

    @staticmethod
    def bleu_score(predictions: List[str], ground_truths: List[str]) -> ndarray:
        """
        Computes the average BLEU score between predicted and ground truth sentences.
        """
        bleu_scores = [sentence_bleu([gt.split()], pred.split()) for pred, gt in zip(predictions, ground_truths)]
        return np.mean(bleu_scores)

    @staticmethod
    def rouge_scores(predictions: List[str], ground_truths: List[str]) -> dict[str, ndarray]:
        """
        Computes the average ROUGE-1, ROUGE-2, and ROUGE-L scores between predicted and ground truth sentences.
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_1, rouge_2, rouge_l = [], [], []

        for pred, gt in zip(predictions, ground_truths):
            scores = scorer.score(gt, pred)
            rouge_1.append(scores['rouge1'].fmeasure)
            rouge_2.append(scores['rouge2'].fmeasure)
            rouge_l.append(scores['rougeL'].fmeasure)

        return {
            "ROUGE-1": np.mean(rouge_1),
            "ROUGE-2": np.mean(rouge_2),
            "ROUGE-L": np.mean(rouge_l)
        }

    @staticmethod
    def f1_score(predictions: List[List[str]], ground_truths: List[List[str]]) -> dict[str, ndarray]:
        """
        Computes Precision, Recall, and F1 score between predicted and ground truth lists of answers.
        """
        precisions, recalls, f1_scores = [], [], []

        for pred_list, gt_list in zip(predictions, ground_truths):
            pred_set, gt_set = set(pred_list), set(gt_list)
            intersection = len(pred_set & gt_set)

            precision = intersection / len(pred_set) if pred_set else 0
            recall = intersection / len(gt_set) if gt_set else 0

            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        return {
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "F1": np.mean(f1_scores)
        }

    @staticmethod
    def exact_match(predictions: List[List[str]], ground_truths: List[List[str]]) -> ndarray:
        """
        Computes the Exact Match score for lists of predictions and ground truths.
        """
        matches = [
            1 if set(pred_list) == set(gt_list) else 0
            for pred_list, gt_list in zip(predictions, ground_truths)
        ]
        return np.mean(matches)

    def completeness(self, predictions: List[List[str]], ground_truths: List[List[str]],
                     threshold: float = 0.7) -> ndarray:
        """
        Computes Completeness score for lists of predictions and ground truths using cosine similarity.

        Args:
            predictions (List[List[str]]): List of predicted answers.
            ground_truths (List[List[str]]): List of ground truth answers.
            threshold (float): Similarity threshold to consider a match (default=0.7).

        Returns:
            float: Mean completeness score across all examples.
        """
        completeness_scores = []

        for pred_list, gt_list in zip(predictions, ground_truths):
            # Compute embeddings for ground truths and predictions
            gt_embeddings = self.model.encode(gt_list, convert_to_tensor=True)
            pred_embeddings = self.model.encode(pred_list, convert_to_tensor=True)

            # Compute cosine similarity between predictions and ground truths
            similarities = util.cos_sim(pred_embeddings, gt_embeddings).cpu().numpy()

            # For each ground truth, check if any prediction passes the similarity threshold
            matched_count = sum(
                any(sim >= threshold for sim in similarities[:, i])
                for i in range(len(gt_list))
            )

            # Completeness = Fraction of ground truth items covered by predictions
            completeness = matched_count / len(gt_list) if gt_list else 0
            completeness_scores.append(completeness)

        return np.mean(completeness_scores)

    @staticmethod
    def temporal_alignment(predictions: List[List[str]], ground_truths: List[List[str]]) -> ndarray:
        """
        Computes the normalized alignment between predicted years and ground truth years.
        Normalizes by both the size of predicted and ground truth years to balance precision and recall.

        Args:
            predictions (List[List[str]]): List of predicted answers as strings (e.g., ["1990-1995", "2000"]).
            ground_truths (List[List[str]]): List of ground truth answers as strings.

        Returns:
            float: Mean normalized alignment score across all examples.
        """

        def extract_years(text: str) -> set:
            """Extracts individual years and year ranges from text."""
            years = []
            ranges = re.findall(r"(\d{4})-(\d{4})", text)
            for start, end in ranges:
                years.extend(range(int(start), int(end) + 1))
            individual_years = re.findall(r"(?<!-)\b\d{4}\b(?!-)", text)
            years.extend(int(year) for year in individual_years)
            return set(years)

        def compute_alignment(pred_years: set, gt_years: set) -> float:
            if not pred_years and not gt_years:
                return 1.0  # Perfect alignment for empty prediction and ground truth.
            if not pred_years or not gt_years:
                return 0.0  # Mismatch if one is empty.
            overlap = len(pred_years & gt_years)
            return 2 * overlap / (len(pred_years) + len(gt_years))

        alignments = []
        for pred_list, gt_list in zip(predictions, ground_truths):
            pred_years = {year for pred in pred_list for year in extract_years(pred)}
            gt_years = {year for gt in gt_list for year in extract_years(gt)}
            alignment = compute_alignment(pred_years, gt_years)
            alignments.append(alignment)

        return np.mean(alignments)

    @staticmethod
    def temporal_ordering(predictions: List[List[str]], ground_truths: List[List[str]]) -> ndarray:
        """
        Computes the correctness of chronological ordering in predictions.
        """

        def extract_years_ordered(text: str) -> List[int]:
            years = []
            ranges = re.findall(r"(\d{4})-(\d{4})", text)
            for start, end in ranges:
                years.extend(range(int(start), int(end) + 1))
            individual_years = re.findall(r"(?<!-)\b\d{4}\b(?!-)", text)
            years.extend(int(year) for year in individual_years)
            return sorted(years)

        order_scores = []
        for pred_list, gt_list in zip(predictions, ground_truths):
            pred_years = [year for pred in pred_list for year in extract_years_ordered(pred)]
            order_score = int(pred_years == sorted(pred_years))  # 1 if ordered, 0 otherwise
            order_scores.append(order_score)
        return np.mean(order_scores)

