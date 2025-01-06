import json
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import colored
from src.modeluniversity.ollama_embeddings import (
    OllamaBatchEmbeddingWrapper,
    OllamaEmbeddingFunction,
)
import os
from .config import settings


class OpenTextBook:
    _collection = None
    _client = None
    _idCounter = 0

    def __init__(
        self,
        embedding_function=None,
        file_with_questions: Path = Path("training_questions.json"),
        collections_name="textbook",
    ):
        self._client = chromadb.PersistentClient(path="./db")
        self.file_with_questions = file_with_questions
        self.collections_name = collections_name

        try:
            if embedding_function is None:
                self._collection = self._client.get_or_create_collection(
                    self.collections_name
                )
            else:
                self._collection = self._client.get_or_create_collection(
                    self.collections_name,
                    embedding_function=embedding_function,
                )
            print(colored(f"Constructing {self.collections_name} DB", "green"))
        except chromadb.errors.InvalidCollectionException as e:
            print(
                colored(f"Failed to initialize {self.collections_name} DB: {e}", "red")
            )
            raise

        try:
            with open(self.file_with_questions, "r") as file:
                training_questions = json.load(file)

                for entry in training_questions:
                    if isinstance(entry, str):
                        entry = json.loads(entry)
                    content = (
                        f"In the topic of {entry['topic']} and subtopic of {entry['subtopic']}, "
                        f"The answer to the following question '{entry['question']}' is {entry['answer']}."
                        f"{entry['explanation']}"
                    )
                    self.add_content(
                        content,
                        {"topic": entry["topic"], "subtopic": entry["subtopic"]},
                    )
        except Exception as e:
            print(colored(f"Error loading training questions: {e}", "red"))
            raise

        self._idCounter = 0
        print(colored("Textbook initialized", "green"))

    def add_content(self, content, metadatadict):
        textsplitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = textsplitter.create_documents([content])
        documents, metadata, ids = [], [], []

        for chunk in chunks:
            documents.append(chunk.page_content)
            metadata.append(metadatadict)
            ids.append(f"entry_{self._idCounter}")
            self._idCounter += 1

        try:
            self._collection.upsert(documents=documents, metadatas=metadata, ids=ids)
        except Exception as e:
            print(colored(f"Error during upsert: {e}", "red"))
            raise

    def query(self, query_texts, n_results):
        try:
            result = self._collection.query(
                query_texts=query_texts, n_results=n_results
            )
            return result.get("documents", [])
        except Exception as e:
            print(colored(f"Query failed: {e}", "red"))
            return []


def create_textbook_instance():
    use_custom_embeddings = (
        os.getenv("USE_CUSTOM_EMBEDDINGS", "false").lower() == "true"
    )

    if use_custom_embeddings:
        print(colored("Using custom embeddings from Ollama", "green"))
        if not settings.ollama_base_url or not settings.ollama_embedding_model:
            raise ValueError(
                "Ollama base URL or embedding model is not configured properly."
            )
        embedding_function = OllamaEmbeddingFunction(
            endpoint=f"{settings.ollama_base_url}/api/embeddings",
            model=settings.ollama_embedding_model,
        )
        # Wrap the embedding function with the batch-aware wrapper
        embedding_function = OllamaBatchEmbeddingWrapper(embedding_function)

    else:
        print("Using default embeddings instead of Ollama")
        embedding_function = None

    return OpenTextBook(embedding_function=embedding_function)
