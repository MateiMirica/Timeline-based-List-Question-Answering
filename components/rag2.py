import os
import json
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_mapping(mapping_file: str):
    with open(mapping_file, 'r') as f:
        return json.load(f)

class WikipediaRAG:
    def __init__(self, mapping_file, top_k=10):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(persist_directory="./chroma_title_and_summary", embedding_function=self.embeddings)
        self.top_k = top_k
        self.mapping = load_mapping(mapping_file)

        docs = []
        for key, value in self.mapping.items():
            doc_content = value.replace("Infobox:", key)
            doc = Document(page_content=doc_content, metadata={"title": key})
            docs.append(doc)

        print("Storing documents...")

        self.vector_store.add_documents(docs)

        # Persist the vector store to save the data
        self.vector_store.persist()

    def preprocess(self, questions, output_file):
        """
        Precompute top-10 contexts for each question in the test set and save to file.
        """
        preprocessed_contexts = {}
        index = 0
        for question in questions:
            print("Processing question: " + str(index))
            docs = self.vector_store.similarity_search(question, self.top_k)
            context = []
            for doc in docs:
                context.append(doc.page_content)

            preprocessed_contexts[question] = context

        # Save precomputed contexts to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_contexts, f, indent=2)
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
    #
    # # Step 3: Retrieve top-k contexts for a specific question from the file
    # k = 3
    # question = "What is the population of Tokyo?"
    # top_k_contexts = preprocessor.retrieve(question, k, output_file)
    # print(f"\nTop-{k} contexts for question '{question}':")
    # for i, context in enumerate(top_k_contexts, 1):
    #     print(f"{i}. {context}")
