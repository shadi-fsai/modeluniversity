import requests

from typing import List


class OllamaBatchEmbeddingWrapper:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.cache = {}  # To cache already computed embeddings

    def __call__(self, input):
        # ChromaDB calls this for single or batch embeddings
        if isinstance(input, str):  # Single string case
            if input in self.cache:
                return self.cache[input]
            embedding = self.embedding_function(input)  # Call single embedding
            self.cache[input] = embedding
            return embedding
        elif isinstance(input, list):  # Batch case
            embeddings = []
            uncached_inputs = []
            for text in input:
                if text in self.cache:
                    embeddings.append(self.cache[text])
                else:
                    uncached_inputs.append(text)

            # Get embeddings for uncached inputs
            if uncached_inputs:
                new_embeddings = self.embedding_function.batch_embed(uncached_inputs)
                for text, embedding in zip(uncached_inputs, new_embeddings):
                    self.cache[text] = embedding
                    embeddings.append(embedding)
            return embeddings
        else:
            raise ValueError(
                f"Input to embedding function must be a string or list of strings, got {type(input)}"
            )


class OllamaEmbeddingFunction:
    def __init__(
        self,
        endpoint: str = "http://localhost:11434/api/embeddings",
        model: str = "jina/jina-embeddings-v2-small-en",
    ):
        self.endpoint = endpoint
        self.model = model

    def __call__(self, input: str) -> list[float]:
        """Single input embedding"""
        payload = {
            "model": self.model,
            "prompt": input,
        }
        response = requests.post(self.endpoint, json=payload)
        response.raise_for_status()
        return response.json().get("embedding", [])

    def batch_embed(self, inputs: list[str]) -> list[list[float]]:
        """Batch embedding for multiple inputs"""
        return [self.__call__(text) for text in inputs]
