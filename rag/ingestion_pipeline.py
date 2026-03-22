import json
from pathlib import Path
from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def partition_document(file_path: str):
    """Extract elements from PDF."""
    print(f"Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in the PDF
        extract_image_block_to_payload=True # Store images as base64 data you can actually use
    )

    print(f"Extracted {len(elements)} elements")
    return elements


def create_chunks_by_title(elements):
    """Create intelligent chunks based on document structure, not just arbitrary character counts."""
    print("Creating smart chunks...")

    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    print(f"Created {len(chunks)} chunks")
    return chunks


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk."""
    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(element.metadata, "image_base64"):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:
TEXT CONTENT:
{text}

"""

        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:

1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:
"""

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        print(f"AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks, workspace_id: str, source_type: str, file_path: str):
    """Attaching app metadata."""
    print("Processing chunks with AI summaries...")

    langchain_documents = []
    total_chunks = len(chunks)
    file_name = Path(file_path).name

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"Processing chunk {current_chunk}/{total_chunks}")

        content_data = separate_content_types(chunk)

        print(f"Types found: {content_data['types']}")
        print(f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        if content_data["tables"] or content_data["images"]:
            print("Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"],
                    content_data["tables"],
                    content_data["images"],
                )
                print("AI summary created successfully")
            except Exception as e:
                print(f"AI summary failed: {e}")
                enhanced_content = content_data["text"]
        else:
            enhanced_content = content_data["text"]

        # Creating Langchain document based on the summary and storing original content in metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "workspace_id": workspace_id,
                "source_type": source_type,
                "file_name": file_name,
                "file_path": file_path,
                "chunk_index": i,
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                    }
                ),
            },
        )

        langchain_documents.append(doc)

    print(f"Processed {len(langchain_documents)} chunks")
    return langchain_documents


def ingest_file(file_path: str, workspace_id: str, source_type: str):
    """Single entry point for the rest of the app."""
    elements = partition_document(file_path)
    chunks = create_chunks_by_title(elements)
    docs = summarise_chunks(chunks, workspace_id, source_type, file_path)
    return docs
