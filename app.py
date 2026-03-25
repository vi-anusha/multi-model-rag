import streamlit as st

st.set_page_config(page_title="Interview RAG Assistant", layout="wide")

st.title("Interview RAG Assistant")
st.write("Upload your resume, JD, and notes. Then chat with your interview preparation assistant.")

if "workspace_id" not in st.session_state:
    st.session_state.workspace_id = "test"

workspace = st.text_input("Workspace ID", value=st.session_state.workspace_id)
st.session_state.workspace_id = workspace

st.info(f"Current workspace: {st.session_state.workspace_id}")

st.markdown("""
### Available pages
- **Upload and Index**: upload resume, JD, project notes, stories
- **Chat**: ask questions about the uploaded documents
- **Documents**: view uploaded files
""")

