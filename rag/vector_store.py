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
            shutil.rmtree(self.persist_directory)

    def build_from_documents(self, documents: List[Document]) -> Chroma:
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return db

    def load_db(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def get_retriever(self, workspace_id: str, k: int = 3):
        db = self.load_db()
        retriever = db.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"workspace_id": workspace_id},
            }
        )
        return retriever