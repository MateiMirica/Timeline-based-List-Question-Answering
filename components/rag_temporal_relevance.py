import json
import random
import re
import numpy as np

from datetime import datetime
from sentence_transformers import SentenceTransformer


def load_mapping(mapping_file: str):
    with open(mapping_file, 'r') as f:
        return json.load(f)


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


class TemporalScore:
    def __init__(self, infobox):
        self.timestamp = self.find_timestamp(infobox)
        self.difference = self.compute_difference(self.timestamp)

    def find_timestamp(self, infobox: str):
        # Regular expression to match full dates in formats like "16 January 2023"
        full_date_matches = re.findall(
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(1[0-9]{3}|2[0-9]{3})\b',
            infobox
        )

        latest_date = None

        # Process full dates
        for day, month, year in full_date_matches:
            try:
                # Convert to datetime object to ensure validity
                date_obj = datetime.strptime(
                    f"{int(day):02d}-{datetime.strptime(month, '%B').month:02d}-{year}",
                    "%d-%m-%Y"
                )
                if latest_date is None or date_obj > latest_date:
                    latest_date = date_obj
            except ValueError:
                # Skip invalid dates
                continue

        # Regular expression to match years only
        year_matches = re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', infobox)

        # Process years if no full dates are found or to compare against full dates
        for year in year_matches:
            try:
                year_date_obj = datetime.strptime(f"01-01-{year}", "%d-%m-%Y")
                if latest_date is None or year_date_obj > latest_date:
                    latest_date = year_date_obj
            except ValueError:
                continue

        return latest_date.strftime("%d-%m-%Y") if latest_date else None

    def compute_difference(self, timestamp):
        if timestamp is None:
            return None
        current_date = datetime.now()
        extracted_date = datetime.strptime(timestamp, "%d-%m-%Y")
        difference_in_days = (current_date - extracted_date).days
        return difference_in_days


def compute_temporal_score(mapping):
    temporal_scores = {}
    for key, value in mapping.items():
        temporal_scores[key] = TemporalScore(value).difference
    return temporal_scores


class WelfordMeanAndStdDev:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, new_value):
        """Update running statistics with a new value using Welford's algorithm."""
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self._m2 += delta * delta2

    def get_std(self):
        """Return the current standard deviation."""
        if self.count < 2:
            return 0.0
        return (self._m2 / (self.count - 1)) ** 0.5


def compute_similarity_stats_with_sampling(mapping, questions_embedding,
                                           compute_embeddings_func, sample_size=0.05,
                                           random_seed=42):
    """
    Compute mean and standard deviation of cosine similarities using sampling.

    Args:
        mapping: Dictionary of document keys to content
        compute_embeddings_func: Function to compute embeddings
        sample_size: Float between 0 and 1 indicating the fraction to sample
        random_seed: Integer for reproducibility

    Returns:
        tuple: (mean_similarity, std_similarity, sample_count)
    """
    random.seed(random_seed)
    stats = WelfordMeanAndStdDev()

    # Convert mapping items to list and sample
    items = list(mapping.items())
    sample_count = int(len(items) * sample_size)
    sampled_items = random.sample(items, sample_count)

    for index_d, (key, value) in enumerate(sampled_items):
        if index_d % 1000 == 0:
            print(f"Computing document {index_d}/{sample_count}")

        doc_content = value.replace("Infobox:", key)
        doc_embedding = compute_embeddings_func([doc_content])[0]

        for question_embedding in questions_embedding:
            similarity = cosine_similarity(doc_embedding, question_embedding)
            stats.update(similarity)

    return stats.mean, stats.get_std()


class WikipediaRAGTemporal:
    def __init__(self, mapping_file, top_k=10):
        self.top_k = top_k
        self.mapping = load_mapping(mapping_file)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()

    def preprocess(self, questions, output_file, alpha_scale=1):
        """
        Precompute top-10 contexts for each question in the test set and save to file.
        """
        preprocessed_contexts = {question: [] for question in questions}
        print("Computing embeddings")
        questions_embedding = self.compute_embeddings(questions)

        print("Computing temporal scores")
        # compute the timestamp for each document and calculate the temporal score
        temporal_scores = compute_temporal_score(self.mapping)

        # compute the mean and standard deviation of the temporal scores
        temporal_scores_values = list(temporal_scores.values())
        # Remove None values
        filtered_scores = [score for score in temporal_scores_values if score is not None]

        # Check if there are valid scores left after filtering
        mean_temporal = np.mean(filtered_scores)
        std_temporal = np.std(filtered_scores)

        print("Mean temporal score:", mean_temporal)
        print("Standard deviation of temporal score:", std_temporal)

        ### For time constraint resons, we have precomputed the mean and standard deviation of the cosine similarity scores
        ### if you want to compute it, you can uncomment the following code. The mean and std was computed on 1% of the data

        # print("Computing mean and standard deviation of cosine similarity scores")
        # mean_similarity, std_similarity = compute_similarity_stats_with_sampling(
        #     self.mapping,
        #     questions_embedding,
        #     self.compute_embeddings
        # )
        # print("Mean similarity:", mean_similarity)
        # print("Standard deviation of similarity:", std_similarity)

        mean_similarity, std_similarity = 0.09534184219638123, 0.09368471193925024

        index_d = 0
        for key, value in self.mapping.items():
            doc_content = value.replace("Infobox:", key)
            doc_embedding = self.compute_embeddings([doc_content])[0]
            if index_d % 1000 == 0:
                print("Computing: " + str(index_d))
            index_d += 1

            for index, question in enumerate(questions):
                question_embedding = questions_embedding[index]
                similarity = cosine_similarity(doc_embedding, question_embedding)

                # Compute the temporal relevance score if the document has a timestamp
                if key in temporal_scores and temporal_scores[key] is not None:
                    scaled_temporal_relevance = ((((alpha_scale / temporal_scores[key]) - mean_temporal)
                                                 / std_temporal)
                                                 * std_similarity + mean_similarity)

                    temporal_rank = similarity + scaled_temporal_relevance
                else:
                    temporal_rank = similarity

                # Add the document to the top-k list for the question if applicable
                if len(preprocessed_contexts[question]) < self.top_k:
                    preprocessed_contexts[question].append((temporal_rank, doc_content))
                    preprocessed_contexts[question].sort(reverse=True, key=lambda x: x[0])
                elif temporal_rank > preprocessed_contexts[question][-1][0]:
                    preprocessed_contexts[question][-1] = (temporal_rank, doc_content)
                    preprocessed_contexts[question].sort(reverse=True, key=lambda x: x[0])

        # Format the output to only include document contents
        formatted_contexts = {
            question: [doc for _, doc in preprocessed_contexts[question]]
            for question in preprocessed_contexts
        }

        # Save precomputed contexts to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_contexts, f, indent=2)
        print(f"Preprocessed contexts saved to {output_file}")

    def retrieve(self, question, k, preprocessed_file):
        """
        Retrieve the top-k most similar contexts for a given question from the preprocessed file.
        """
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            preprocessed_contexts = json.load(f)

        if question not in preprocessed_contexts:
            raise ValueError(f"Question '{question}' not found in preprocessed contexts.")
        return preprocessed_contexts[question][:k]


def extract_questions():
    # Load train and val datasets
    with open('../data/test_TLQA.json', 'r') as f:
        training_data = json.load(f)

    questions = []
    for item in training_data:
        question = item['question']
        questions.append(question)
    return questions


# Main Script
if __name__ == "__main__":
    questions = extract_questions()
    preprocessor = WikipediaRAGTemporal('../data/title_to_infobox.json', top_k=10)

    output_file = "../data/preprocessed_contexts_time_based.json"

    preprocessor.preprocess(questions, output_file)
