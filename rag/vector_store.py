import os
import shutil
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "interview_rag")


class RAGStore:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        self.persist_directory = CHROMA_DIR
        self.collection_name = COLLECTION_NAME

    def reset_store(self) -> None:
        if os.path.exists(self.persist_directory):
            print(f"\n=== DEBUG: REMOVING OLD CHROMA STORE AT {self.persist_directory} ===")
            shutil.rmtree(self.persist_directory)

    def build_from_documents(self, documents: List[Document]) -> Chroma:
        print("\n=== DEBUG: BUILDING CHROMA FROM DOCUMENTS ===")
        print("Total documents:", len(documents))

        for i, doc in enumerate(documents[:5], start=1):
            print(f"\n--- DOCUMENT {i} BEFORE INDEXING ---")
            print("FILE:", doc.metadata.get("file_name"))
            print("TYPE:", doc.metadata.get("source_type"))
            print("WORKSPACE:", doc.metadata.get("workspace_id"))
            print("CHUNK INDEX:", doc.metadata.get("chunk_index"))
            print("CONTENT PREVIEW:", doc.page_content[:400])
            print("HAS original_content:", "original_content" in doc.metadata)

        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

        print("\n=== DEBUG: CHROMA BUILD COMPLETE ===")
        return db

    def load_db(self) -> Chroma:
        print("\n=== DEBUG: LOADING CHROMA DB ===")
        print("persist_directory:", self.persist_directory)
        print("collection_name:", self.collection_name)

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def has_data(self) -> bool:
        """
        Check whether the persisted Chroma collection already has documents.
        """
        try:
            db = self.load_db()
            count = db._collection.count()
            print("\n=== DEBUG: CHROMA DOCUMENT COUNT ===")
            print(count)
            return count > 0
        except Exception as e:
            print("\n=== DEBUG: FAILED TO CHECK CHROMA COUNT ===")
            print(e)
            return False

    def get_retriever(self, workspace_id: str, k: int = 3):
        print("\n=== DEBUG: RETRIEVER CONFIG ===")
        print("workspace_id =", workspace_id)
        print("k =", k)

        db = self.load_db()

        retriever = db.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"workspace_id": workspace_id},
            }
        )

        return retriever

    def debug_retrieve_without_filter(self, query: str, k: int = 3):
        print("\n=== DEBUG: RETRIEVE WITHOUT FILTER ===")
        db = self.load_db()
        retriever = db.as_retriever(search_kwargs={"k": k})
        chunks = retriever.invoke(query)

        print("Retrieved without filter:", len(chunks))
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- NO FILTER CHUNK {i} ---")
            print("FILE:", chunk.metadata.get("file_name"))
            print("TYPE:", chunk.metadata.get("source_type"))
            print("WORKSPACE:", chunk.metadata.get("workspace_id"))
            print("CONTENT PREVIEW:", chunk.page_content[:400])

        return chunks