import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.answer_generation import HistoryAwareAnswerGenerator
from rag.vector_store import RAGStore
from rag.index_docs import index_workspace_if_needed

WORKSPACE_ID = "test"
MODE = "answer_as_me"  # "ask", "answer_as_me", "answer_as_expert"

def main():
    print("\n=== DEBUG: START TEST SCRIPT ===")
    print("WORKSPACE_ID =", WORKSPACE_ID)
    print("MODE =", MODE)

    # Force rebuild once while debugging
    index_workspace_if_needed(workspace_id=WORKSPACE_ID, reset=True)

    store = RAGStore()
    generator = HistoryAwareAnswerGenerator()

    while True:
        question = input("\nYour question (or 'quit'): ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        # Optional debug: retrieval without filter
        store.debug_retrieve_without_filter(query=question, k=3)

        answer, docs, search_query = generator.ask(
            workspace_id=WORKSPACE_ID,
            user_question=question,
            mode=MODE,
            k=3,
        )

        print("\n=== SEARCH QUERY USED ===\n")
        print(search_query)

        print("\n=== ANSWER ===\n")
        print(answer)

        print("\n=== SOURCES ===\n")
        for i, doc in enumerate(docs, start=1):
            print(
                f"[{i}] {doc.metadata.get('file_name')} | "
                f"{doc.metadata.get('source_type')} | "
                f"chunk={doc.metadata.get('chunk_index')}"
            )
            print(doc.page_content[:300])
            print("-" * 80)


if __name__ == "__main__":
    main()