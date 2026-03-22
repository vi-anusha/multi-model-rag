import os
import json
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from rag.vector_store import RAGStore

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _parse_original_content(doc: Document) -> dict:
    """
    Read original_content from metadata.
    Falls back safely if missing or malformed.
    """
    raw = doc.metadata.get("original_content")

    if not raw:
        return {
            "raw_text": doc.page_content,
            "tables_html": [],
            "images_base64": [],
        }

    if isinstance(raw, dict):
        return {
            "raw_text": raw.get("raw_text", doc.page_content),
            "tables_html": raw.get("tables_html", []),
            "images_base64": raw.get("images_base64", []),
        }

    try:
        parsed = json.loads(raw)
        return {
            "raw_text": parsed.get("raw_text", doc.page_content),
            "tables_html": parsed.get("tables_html", []),
            "images_base64": parsed.get("images_base64", []),
        }
    except Exception:
        return {
            "raw_text": doc.page_content,
            "tables_html": [],
            "images_base64": [],
        }


def build_multimodal_message(query: str, docs: List[Document], mode: str = "ask") -> HumanMessage:
    mode_instruction = {
        "ask": "Answer using only the retrieved context.",
        "answer_as_me": (
            "Answer as the candidate in first person. "
            "Use only the retrieved context and do not invent experience."
        ),
        "generate_questions": (
            "Generate realistic interview questions using only the retrieved context. "
            "Group them into technical, behavioral, and gap-based questions."
        ),
        "gap_analysis": (
            "Compare the candidate evidence with the job-related context. "
            "List strengths, likely gaps, and preparation areas."
        ),
    }.get(mode, "Answer using only the retrieved context.")

    message_content = [
        {
            "type": "text",
            "text": f"""
You are an interview prep assistant.

Instruction:
{mode_instruction}

Rules:
- Use only the provided context.
- If the context is insufficient, say so clearly.
- Do not fabricate skills, projects, outcomes, or numbers.
- When tables are present, use the table data carefully.
- When images are present, inspect them before answering.

User request:
{query}

Retrieved context is provided below source by source.
""".strip(),
        }
    ]

    for i, doc in enumerate(docs, start=1):
        original = _parse_original_content(doc)

        raw_text = original.get("raw_text", "")
        tables_html = original.get("tables_html", [])
        images_base64 = original.get("images_base64", [])

        source_header = f"""
[Source {i}]
file: {doc.metadata.get("file_name")}
type: {doc.metadata.get("source_type")}
chunk_index: {doc.metadata.get("chunk_index")}
""".strip()

        source_text = f"{source_header}\n\nRAW TEXT:\n{raw_text}\n"

        if tables_html:
            source_text += "\nTABLES:\n"
            for t_idx, table_html in enumerate(tables_html, start=1):
                source_text += f"\nTable {t_idx}:\n{table_html}\n"

        message_content.append(
            {
                "type": "text",
                "text": source_text,
            }
        )

        for image_base64 in images_base64:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                }
            )

    return HumanMessage(content=message_content)


def answer_query(
    workspace_id: str,
    query: str,
    mode: str = "ask",
) -> Tuple[str, List[Document]]:
    store = RAGStore()
    retriever = store.get_retriever(workspace_id=workspace_id, k=3)
    chunks = retriever.invoke(query)

    message = build_multimodal_message(query=query, docs=chunks, mode=mode)

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    answer = llm.invoke([message]).content

    return answer, chunks
