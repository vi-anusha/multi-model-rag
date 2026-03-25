from pathlib import Path
import streamlit as st


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


def ensure_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "rag_generator" not in st.session_state:
        from rag.answer_generation import HistoryAwareAnswerGenerator
        st.session_state.rag_generator = HistoryAwareAnswerGenerator()
