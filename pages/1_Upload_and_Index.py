import streamlit as st
from pathlib import Path

from ui.helpers import VALID_SOURCE_TYPES, save_uploaded_file
from rag.index_docs import index_workspace_if_needed

st.title("Upload and Index Documents")

workspace_id = st.session_state.get("workspace_id", "test")
st.write(f"Current workspace: **{workspace_id}**")

source_type = st.selectbox("Select source type", sorted(list(VALID_SOURCE_TYPES)))
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "doc", "docx", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Save uploaded files"):
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(workspace_id, source_type, uploaded_file)
            saved_files.append(file_path)

        st.success("Files saved successfully.")
        for f in saved_files:
            st.write(f"- {f}")

st.divider()

st.subheader("Index Documents")

col1, col2 = st.columns(2)

with col1:
    if st.button("Index if needed"):
        try:
            index_workspace_if_needed(workspace_id=workspace_id, reset=False)
            st.success("Indexing completed or existing index reused.")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

with col2:
    if st.button("Force Rebuild Index"):
        try:
            index_workspace_if_needed(workspace_id=workspace_id, reset=True)
            st.success("Index rebuilt successfully.")
        except Exception as e:
            st.error(f"Rebuild failed: {e}")

workspace_path = Path("data") / workspace_id
if workspace_path.exists():
    st.subheader("Current workspace files")
    for source_dir in workspace_path.iterdir():
        if source_dir.is_dir():
            st.write(f"**{source_dir.name}**")
            for file_path in source_dir.iterdir():
                if file_path.is_file():
                    st.write(f"- {file_path.name}")