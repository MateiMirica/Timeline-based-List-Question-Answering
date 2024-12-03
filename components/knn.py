from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class KNN:
    def __init__(self, k=3, model_name='all-MiniLM-L6-v2'):
        self.k = k
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.outputs = None

    def fit(self, input_prompts, output_prompts):
        self.inputs = input_prompts
        self.outputs = output_prompts
        self.embeddings = self.model.encode(input_prompts, convert_to_tensor=True).cpu().numpy()

        self.knn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.knn.fit(self.embeddings)

    def query(self, new_input):
        if self.embeddings is None:
            raise ValueError("The model has not been fitted yet.")

        query_embedding = self.model.encode([new_input], convert_to_tensor=True).cpu().numpy()

        distances, indices = self.knn.kneighbors(query_embedding)

        closest_prompts = [(self.inputs[idx], self.outputs[idx]) for idx in indices[0]]
        return closest_prompts

def main():
    # Example input-output prompts
    training_inputs = [
        "Translate English to French: How are you?",
        "Translate English to Spanish: Good morning.",
        "What is the capital of France?",
        "Summarize the article in one sentence."
    ]

    training_outputs = [
        "French: Comment ça va?",
        "Spanish: Buenos días.",
        "The capital of France is Paris.",
        "The article discusses recent advancements in AI."
    ]

    knn_classifier = KNN(k=2)

    knn_classifier.fit(training_inputs, training_outputs)

    query_input = "Translate English to French: Good evening."
    closest_prompts = knn_classifier.query(query_input)

    print("Query Input:", query_input)
    print("\nClosest Outputs:")
    for (input, output) in closest_prompts:
        print(f"- Input: {input}, Output: {output}")

if __name__ == "__main__":
    main()