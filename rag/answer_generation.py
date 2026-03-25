import os
import json
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from rag.vector_store import RAGStore

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


class HistoryAwareAnswerGenerator:
    def __init__(self):
        self.model = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
        self.store = RAGStore()
        self.chat_history: List = []

    def _parse_original_content(self, doc: Document) -> dict:
        raw = doc.metadata.get("original_content")

        if not raw:
            return {
                "raw_text": "",
                "tables_html": [],
                "images_base64": [],
            }

        if isinstance(raw, dict):
            return {
                "raw_text": raw.get("raw_text", ""),
                "tables_html": raw.get("tables_html", []),
                "images_base64": raw.get("images_base64", []),
            }

        try:
            parsed = json.loads(raw)
            return {
                "raw_text": parsed.get("raw_text", ""),
                "tables_html": parsed.get("tables_html", []),
                "images_base64": parsed.get("images_base64", []),
            }
        except Exception as e:
            print("\n=== DEBUG: FAILED TO PARSE original_content ===")
            print(e)
            print("RAW original_content value:")
            print(raw)
            return {
                "raw_text": "",
                "tables_html": [],
                "images_base64": [],
            }

    def reformulate_query(self, user_question: str) -> str:
        print("\n=== DEBUG: REFORMULATING QUERY ===")

        if not self.chat_history:
            print("No history found. Using original question.")
            return user_question

        messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new user question into a standalone, "
                    "searchable query for document retrieval. "
                    "Return only the rewritten query and nothing else."
                )
            ),
        ] + self.chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = self.model.invoke(messages)
        search_query = result.content.strip()

        print("Original question:", user_question)
        print("Rewritten query:", search_query)

        return search_query

    def build_multimodal_message(
        self,
        user_question: str,
        search_query: str,
        docs: List[Document],
        mode: str = "ask",
    ) -> List:
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

        content_parts = [
            {
                "type": "text",
                "text": f"""
                    You are an interview prep assistant.

                    Instruction:
                    {mode_instruction}

                    Rules:
                    - Use only the provided context.
                    - Use the raw original chunk content below, not summarized retrieval text.
                    - If the context is insufficient, say so clearly.
                    - Do not fabricate skills, projects, outcomes, or numbers.
                    - Use tables carefully when present.
                    - Inspect images when present.

                    Original user question:
                    {user_question}

                    Search query used for retrieval:
                    {search_query}
                    """.strip(),
            }
        ]

        for i, doc in enumerate(docs, start=1):
            original = self._parse_original_content(doc)

            raw_text = original.get("raw_text", "")
            tables_html = original.get("tables_html", [])
            images_base64 = original.get("images_base64", [])

            print(f"\n=== DEBUG: ORIGINAL CONTENT FOR CHUNK {i} ===")
            print("file_name:", doc.metadata.get("file_name"))
            print("source_type:", doc.metadata.get("source_type"))
            print("chunk_index:", doc.metadata.get("chunk_index"))
            print("raw_text preview:", raw_text[:300])
            print("tables count:", len(tables_html))
            print("images count:", len(images_base64))

            source_text = f"""
            [Source {i}]
            file: {doc.metadata.get("file_name")}
            type: {doc.metadata.get("source_type")}
            chunk_index: {doc.metadata.get("chunk_index")}

            RAW TEXT:
            {raw_text}
            """.strip()

            if tables_html:
                source_text += "\n\nTABLES:\n"
                for t_idx, table_html in enumerate(tables_html, start=1):
                    source_text += f"\nTable {t_idx}:\n{table_html}\n"

            content_parts.append(
                {
                    "type": "text",
                    "text": source_text,
                }
            )

            for image_base64 in images_base64:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    }
                )

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful interview prep assistant that answers only from the "
                    "provided retrieved documents and the ongoing conversation history."
                )
            )
        ] + self.chat_history + [
            HumanMessage(content=content_parts)
        ]

        return messages

    def ask(
        self,
        workspace_id: str,
        user_question: str,
        mode: str = "ask",
        k: int = 3,
    ) -> Tuple[str, List[Document], str]:
        print("\n=== DEBUG: USER QUESTION ===")
        print(user_question)

        print("\n=== DEBUG: WORKSPACE_ID ===")
        print(workspace_id)

        # Step 1: history-aware reformulation
        search_query = self.reformulate_query(user_question)

        # Step 2: retrieval
        retriever = self.store.get_retriever(workspace_id=workspace_id, k=k)
        chunks = retriever.invoke(search_query)

        print("\n=== DEBUG: RETRIEVED CHUNKS COUNT ===")
        print(len(chunks))

        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- RETRIEVED CHUNK {i} ---")
            print("FILE:", chunk.metadata.get("file_name"))
            print("TYPE:", chunk.metadata.get("source_type"))
            print("WORKSPACE:", chunk.metadata.get("workspace_id"))
            print("CHUNK INDEX:", chunk.metadata.get("chunk_index"))
            print("RETRIEVED PAGE_CONTENT PREVIEW:", chunk.page_content[:300])

        if not chunks:
            print("\n=== DEBUG: NO CHUNKS RETURNED FROM RETRIEVER ===")

        # Step 3: answer generation from original metadata content
        messages = self.build_multimodal_message(
            user_question=user_question,
            search_query=search_query,
            docs=chunks,
            mode=mode,
        )

        print("\n=== DEBUG: FINAL MESSAGE COUNT ===")
        print(len(messages))

        answer = self.model.invoke(messages).content

        # Step 4: update history
        self.chat_history.append(HumanMessage(content=user_question))
        self.chat_history.append(AIMessage(content=answer))

        return answer, chunks, search_query