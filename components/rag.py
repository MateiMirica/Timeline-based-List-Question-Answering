import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def load_mapping(mapping_file: str):
    with open(mapping_file, 'r') as f:
        return json.load(f)

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

class WikipediaRAG:
    def __init__(self, mapping_file, top_k=10):
        self.top_k = top_k
        self.mapping = load_mapping(mapping_file)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()

    def preprocess(self, questions, output_file):
        """
        Precompute top-10 contexts for each question in the test set and save to file.
        """
        preprocessed_contexts = {question: [] for question in questions}
        print("Computing embeddings")
        questions_embedding = self.compute_embeddings(questions)

        index_d = 0
        for key, value in self.mapping.items():
            print(f"Processing document: {key}")
            doc_content = value.replace("Infobox:", key)
            doc_embedding = self.compute_embeddings([doc_content])[0]
            print("Computing: " + str(index_d))
            index_d += 1

            for index, question in enumerate(questions):
                question_embedding = questions_embedding[index]
                similarity = cosine_similarity(doc_embedding, question_embedding)

                # Add the document to the top-k list for the question if applicable
                if len(preprocessed_contexts[question]) < self.top_k:
                    preprocessed_contexts[question].append((similarity, doc_content))
                    preprocessed_contexts[question].sort(reverse=True, key=lambda x: x[0])
                elif similarity > preprocessed_contexts[question][-1][0]:
                    preprocessed_contexts[question][-1] = (similarity, doc_content)
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
    preprocessor = WikipediaRAG('../data/title_to_infobox.json', top_k=10)

    output_file = "../data/preprocessed_contexts.json"

    # Step 2: Precompute contexts for all questions and save to file
    preprocessor.preprocess(questions, output_file)

    # Step 3: Retrieve top-k contexts for a specific question from the file
    # k = 3
    # question = "What is the population of Tokyo?"
    # top_k_contexts = preprocessor.retrieve(question, k, output_file)
    # print(f"\nTop-{k} contexts for question '{question}':")
    # for i, context in enumerate(top_k_contexts, 1):
    #     print(f"{i}. {context}")
