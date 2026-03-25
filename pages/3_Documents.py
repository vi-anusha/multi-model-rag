import streamlit as st
from pathlib import Path

st.title("Documents")

workspace_id = st.session_state.get("workspace_id", "test")
workspace_path = Path("data") / workspace_id

st.write(f"Current workspace: **{workspace_id}**")

if not workspace_path.exists():
    st.warning("No documents found for this workspace.")
else:
    for source_dir in workspace_path.iterdir():
        if source_dir.is_dir():
            st.subheader(source_dir.name)
            files = [f for f in source_dir.iterdir() if f.is_file()]
            if not files:
                st.write("No files")
            else:
                for file_path in files:
                    st.write(f"- {file_path.name}")