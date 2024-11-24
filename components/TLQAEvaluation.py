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
            "cosine_similarity": self.cosine_similarity,
            "rouge": self.rouge_l_similarity,
            "bleu": self.bleu_score,
            "f1": self.f1_score,
            "exact_match": self.exact_match,
            "temporal_consistency": self.temporal_consistency,
            "completeness": self.completeness,
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
        Normalize is applied for f1, EM, and completeness
        """
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def evaluate(self, predictions: List[Dict], metric: str, normalize: bool = False):
        """
        General evaluation method for various metrics.
        Normalize is required for f1, EM, and completeness
        Args:
            predictions (List[Dict]): List of predicted answers as dictionaries.
            metric (str): The metric to evaluate (e.g., 'cosine_similarity', 'rouge', 'f1').
            normalize (bool): Whether to normalize the text before evaluation.

        Returns:
            float or dict: The computed metric score(s).
        """
        if metric in ["cosine_similarity", "rouge", "bleu"]:
            # Use joined strings for these metrics
            ground_truths = ["\n".join(gt["final_answers"]) for gt in self.test_data.to_dict(orient="records")]
            predictions = ["\n".join(pred["answers"]) for pred in predictions]

        elif metric in ["f1", "exact_match", "temporal_consistency", "completeness"]:
            # Use lists for these metrics
            ground_truths = [gt["final_answers"] for gt in self.test_data.to_dict(orient="records")]
            predictions = [pred["answers"] for pred in predictions]

            # Normalize if required
            if normalize:
                ground_truths = [[self.normalize_text(answer) for answer in gt] for gt in ground_truths]
                predictions = [[self.normalize_text(answer) for answer in pred] for pred in predictions]
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return self.metrics[metric](predictions, ground_truths)

    def cosine_similarity(self, predictions: List[str], ground_truths: List[str]) -> float:
        """
        Computes the average cosine similarity between predicted and ground truth sentences.
        """
        pred_embeddings = self.model.encode(predictions, convert_to_tensor=True)
        gt_embeddings = self.model.encode(ground_truths, convert_to_tensor=True)
        similarities = util.cos_sim(pred_embeddings, gt_embeddings)
        return similarities.diagonal().mean().item()

    @staticmethod
    def rouge_l_similarity(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Computes the average ROUGE-L similarity between predicted and ground truth sentences.
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(gt, pred)['rougeL'].fmeasure for pred, gt in zip(predictions, ground_truths)]
        return np.mean(scores)

    @staticmethod
    def bleu_score(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Computes the average BLEU score between predicted and ground truth sentences.
        """
        bleu_scores = [sentence_bleu([gt.split()], pred.split()) for pred, gt in zip(predictions, ground_truths)]
        return np.mean(bleu_scores)

    @staticmethod
    def f1_score(predictions: List[List[str]], ground_truths: List[List[str]]) -> Dict[str, float]:
        """
        Computes Precision, Recall, and F1 score between predicted and ground truth lists of answers.

        Args:
            predictions (List[List[str]]): List of predicted answers.
            ground_truths (List[List[str]]): List of ground truth answers.

        Returns:
            Dict[str, float]: Average Precision, Recall, and F1 scores.
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
    def exact_match(predictions: List[List[str]], ground_truths: List[List[str]]) -> float:
        """
        Computes the Exact Match score for lists of predictions and ground truths.
        """
        matches = [
            1 if set(pred_list) == set(gt_list) else 0
            for pred_list, gt_list in zip(predictions, ground_truths)
        ]
        return np.mean(matches)

    @staticmethod
    def completeness(predictions: List[List[str]], ground_truths: List[List[str]]) -> float:
        """
        Computes Completeness score for lists of predictions and ground truths.

        Args:
            predictions (List[List[str]]): List of predicted answers.
            ground_truths (List[List[str]]): List of ground truth answers.

        Returns:
            float: Completeness score across all examples.
        """
        completeness_scores = []

        for pred_list, gt_list in zip(predictions, ground_truths):
            gt_set = set(gt_list)
            intersection = len(gt_set & set(pred_list))

            # Completeness = Fraction of ground truth items covered by predictions
            completeness = intersection / len(gt_set) if gt_set else 0
            completeness_scores.append(completeness)

        return np.mean(completeness_scores)

    @staticmethod
    def temporal_consistency(predictions: List[List[str]], ground_truths: List[List[str]]) -> float:
        """
        Computes Temporal Consistency for lists of predictions and ground truths.
        """

        def extract_years(text: str) -> set:
            years = []
            ranges = re.findall(r"(\d{4})-(\d{4})", text)
            for start, end in ranges:
                years.extend(range(int(start), int(end) + 1))
            individual_years = re.findall(r"(?<!-)\b\d{4}\b(?!-)", text)
            years.extend(int(year) for year in individual_years)
            return set(years)

        def overlap_ratio(pred_years: set, gt_years: set) -> float:
            return len(pred_years & gt_years) / len(gt_years) if gt_years else 0

        scores = []
        for pred_list, gt_list in zip(predictions, ground_truths):
            total_overlap = 0
            for gt in gt_list:
                gt_years = extract_years(gt)
                best_overlap = max(
                    (overlap_ratio(extract_years(pred), gt_years) for pred in pred_list), default=0
                )
                total_overlap += best_overlap
            scores.append(total_overlap / len(gt_list) if gt_list else 0)

        return np.mean(scores) if scores else 0.0
