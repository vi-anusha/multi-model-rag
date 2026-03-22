import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.answer_generation import answer_query
from rag.vector_store import RAGStore
from rag.index_docs import index_workspace_if_needed

WORKSPACE_ID = "test"
QUESTION = "Have you worked on RAG system explain about it?"
MODE = "answer_as_me"  # "ask", "answer_as_me", "answer_as_expert"


def main():
    print("\n=== DEBUG: START TEST SCRIPT ===")
    print("WORKSPACE_ID =", WORKSPACE_ID)
    print("QUESTION =", QUESTION)
    print("MODE =", MODE)

    # Auto-index if DB is empty
    index_workspace_if_needed(workspace_id=WORKSPACE_ID, reset=False)

    # Optional extra debug: check retrieval without filter
    store = RAGStore()
    store.debug_retrieve_without_filter(query=QUESTION, k=3)

    answer, docs = answer_query(
        workspace_id=WORKSPACE_ID,
        query=QUESTION,
        mode=MODE,
    )

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== SOURCES ===\n")
    for i, doc in enumerate(docs, start=1):
        print(
            f"[{i}] {doc.metadata.get('file_name')} | "
            f"{doc.metadata.get('source_type')} | "
            f"chunk={doc.metadata.get('chunk_index')}"
        )
        print(doc.metadata.get("original_content")[:500])  # Print original content preview
        print("-" * 80)


if __name__ == "__main__":
    main()