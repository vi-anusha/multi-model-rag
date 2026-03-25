from pathlib import Path
import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

VALID_SOURCE_TYPES = {
    "resume",
    "jd",
    "project_note",
    "star_story",
    "skill_note",
    "company_note",
}


def get_workspace_dir(workspace_id: str) -> Path:
    return Path("data") / workspace_id


def save_uploaded_file(workspace_id: str, source_type: str, uploaded_file):
    workspace_dir = get_workspace_dir(workspace_id)
    target_dir = workspace_dir / source_type
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


# =========================
# CHAT PERSISTENCE
# =========================

def get_chat_history_path(workspace_id: str) -> Path:
    workspace_dir = get_workspace_dir(workspace_id)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir / "chat_history.json"


def load_chat_history(workspace_id: str):
    path = get_chat_history_path(workspace_id)

    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_chat_history(workspace_id: str, chat_messages):
    path = get_chat_history_path(workspace_id)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(chat_messages, f, ensure_ascii=False, indent=2)


def clear_chat_history(workspace_id: str):
    path = get_chat_history_path(workspace_id)

    if path.exists():
        path.unlink()


# =========================
# GENERATOR HISTORY SYNC
# =========================

def rebuild_generator_history(generator, chat_messages):
    generator.chat_history = []

    for msg in chat_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            generator.chat_history.append(HumanMessage(content=content))
        elif role == "assistant":
            generator.chat_history.append(AIMessage(content=content))


def ensure_chat_state(workspace_id: str):
    from rag.answer_generation import HistoryAwareAnswerGenerator

    if "rag_generator" not in st.session_state:
        st.session_state.rag_generator = HistoryAwareAnswerGenerator()

    if "loaded_workspace_id" not in st.session_state:
        st.session_state.loaded_workspace_id = None

    # Load chat only when:
    # 1. first time
    # 2. workspace changed
    if (
        "chat_messages" not in st.session_state
        or st.session_state.loaded_workspace_id != workspace_id
    ):
        st.session_state.chat_messages = load_chat_history(workspace_id)
        st.session_state.loaded_workspace_id = workspace_id

        rebuild_generator_history(
            st.session_state.rag_generator,
            st.session_state.chat_messages,
        )