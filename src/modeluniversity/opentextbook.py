import json
# import the chromadb module
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import colored

class OpenTextBook:
    _collection = None
    _client = None
    _idCounter = 0

    def __init__(self):
        self._client = chromadb.PersistentClient("./db")
        try:
            self._collection = self._client.get_collection("textbook")
            print (colored("Constructing Textbook DB", "green"))            
        except chromadb.errors.InvalidCollectionException:
            self._collection = self._client.get_or_create_collection("textbook")

            with open("training_questions.json", "r") as file:
                training_questions = json.load(file)

                for entry in training_questions:
                    entry = json.loads(entry)
                    content = "In the topic of " + entry["topic"] + " and subtopic of " + entry["subtopic"] +\
                        ", The answer to the following question \" " + entry["question"] + "\" is " + entry["answer"] + "." + entry["explanation"]
                    self.add_content(content, {"topic" : entry["topic"], "subtopic" : entry["subtopic"]})

        self._idCounter = 0
        print (colored("Textbook initialized", "green"))


    def add_content(self, content, metadatadict):
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200, length_function=len,
            is_separator_regex=False)
        chunks = textsplitter.create_documents([content])
        documents = []
        metadata = []
        ids = []

        for chunk in chunks:
            documents.append(chunk.page_content)
            metadata.append(metadatadict)
            ids.append(f"entry_{self._idCounter}")
            self._idCounter += 1
        
        self._collection.upsert(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

    def query(self, query_texts, n_results):
        return self._collection.query(
            query_texts=query_texts,
            n_results=n_results
        )["documents"]

