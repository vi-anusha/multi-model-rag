from pathlib import Path

from rag.ingestion_pipeline import ingest_file
from rag.vector_store import RAGStore

VALID_SOURCE_TYPES = {
    "resume",
    "jd",
    "project_note",
    "star_story",
    "skill_note",
    "company_note",
}

VALID_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx", ".md"}


def index_workspace_if_needed(workspace_id: str, reset: bool = False):
    store = RAGStore()

    if not reset and store.has_data():
        print("\n=== DEBUG: CHROMA ALREADY HAS DATA, SKIPPING RE-INDEX ===")
        return

    data_dir = Path("data") / workspace_id

    if not data_dir.exists():
        raise FileNotFoundError(f"Workspace folder not found: {data_dir}")

    print("\n=== DEBUG: STARTING AUTO-INDEX ===")
    print("Workspace folder:", data_dir.resolve())

    all_docs = []

    for source_dir in data_dir.iterdir():
        if not source_dir.is_dir():
            continue

        source_type = source_dir.name
        print("\nSource dir:", source_type)

        if source_type not in VALID_SOURCE_TYPES:
            print("Skipping unknown source type:", source_type)
            continue

        for file_path in source_dir.iterdir():
            if not file_path.is_file():
                continue

            if file_path.name.startswith("."):
                print("Skipping hidden file:", file_path.name)
                continue

            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                print("Skipping unsupported file type:", file_path.name)
                continue

            print("\nProcessing file:", file_path.name)

            try:
                docs = ingest_file(
                    file_path=str(file_path),
                    workspace_id=workspace_id,
                    source_type=source_type,
                )

                print(f"Docs returned from ingest_file: {len(docs)}")
                all_docs.extend(docs)

            except Exception as e:
                print(f"\n=== DEBUG: FAILED TO INGEST FILE {file_path.name} ===")
                print(e)
                continue

    print("\n=== DEBUG: DOCUMENTS BEFORE INDEXING ===")
    print("Total documents to index:", len(all_docs))

    for i, doc in enumerate(all_docs[:5], start=1):
        print(f"\n--- DOC {i} ---")
        print("FILE:", doc.metadata.get("file_name"))
        print("TYPE:", doc.metadata.get("source_type"))
        print("WORKSPACE:", doc.metadata.get("workspace_id"))
        print("CHUNK INDEX:", doc.metadata.get("chunk_index"))
        print("FILE EXT:", doc.metadata.get("file_ext"))
        print("CONTENT:", doc.page_content[:300])
        print("METADATA:", doc.metadata)

    if not all_docs:
        raise ValueError(f"No supported documents found to index for workspace: {workspace_id}")

    if reset:
        store.reset_store()

    store.build_from_documents(all_docs)
    print(f"\n=== DEBUG: INDEXED {len(all_docs)} CHUNKS SUCCESSFULLY ===")