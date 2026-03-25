import streamlit as st

from ui.helpers import (
    ensure_chat_state,
    save_chat_history,
    clear_chat_history,
)

st.set_page_config(page_title="Interview RAG Chat", layout="wide")

# =========================
# TERMINAL STYLE UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0b0f10;
    color: #d7ffd9;
    font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
}

.block-container {
    padding-top: 1rem;
    max-width: 1200px;
}

.terminal-shell {
    background: #111315;
    border: 1px solid #2a2f33;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1rem;
}

.terminal-topbar {
    background: #1b1f22;
    padding: 10px;
}

.dot {
    height: 12px;
    width: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
}
.red { background: #ff5f57; }
.yellow { background: #febc2e; }
.green { background: #28c840; }

[data-testid="stChatMessage"] {
    background: #0f1315;
    border: 1px solid #263238;
    border-radius: 10px;
    padding: 10px;
}

[data-testid="stMarkdownContainer"] p {
    color: #d7ffd9 !important;
}

div[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0;
    background: #0b0f10;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("Interview RAG Terminal")

workspace_id = st.session_state.get("workspace_id", "test")
st.caption(f"workspace: {workspace_id}")

# =========================
# LOAD CHAT STATE
# =========================
ensure_chat_state(workspace_id)

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""

# =========================
# MODE SELECTOR
# =========================
mode = st.selectbox(
    "mode",
    ["ask", "answer_as_me", "generate_questions", "gap_analysis"],
)

# =========================
# CLEAR CHAT
# =========================
if st.button("clear chat"):
    st.session_state.chat_messages = []

    from rag.answer_generation import HistoryAwareAnswerGenerator
    st.session_state.rag_generator = HistoryAwareAnswerGenerator()

    st.session_state.last_docs = []
    st.session_state.last_search_query = ""

    clear_chat_history(workspace_id)
    st.rerun()

# =========================
# LAYOUT
# =========================
left, right = st.columns([2, 1])

# =========================
# CHAT HISTORY (TOP)
# =========================
with left:
    st.subheader("conversation")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# =========================
# RETRIEVAL PANEL
# =========================
with right:
    st.subheader("retrieval")

    if st.session_state.last_search_query:
        st.write("search query")
        st.code(st.session_state.last_search_query)

    docs = st.session_state.last_docs

    if docs:
        st.write("sources")

        for i, doc in enumerate(docs, start=1):
            with st.expander(
                f"{i}. {doc.metadata.get('file_name')} | {doc.metadata.get('source_type')}"
            ):
                st.write(f"chunk: {doc.metadata.get('chunk_index')}")
                st.text(doc.page_content[:1000])

# =========================
# CHAT INPUT (BOTTOM)
# =========================
question = st.chat_input("ask your question...")

# =========================
# HANDLE QUESTION
# =========================
if question:
    # Save user message
    st.session_state.chat_messages.append(
        {"role": "user", "content": question}
    )
    save_chat_history(workspace_id, st.session_state.chat_messages)

    with left:
        # Show user message
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                try:
                    generator = st.session_state.rag_generator

                    answer, docs, search_query = generator.ask(
                        workspace_id=workspace_id,
                        user_question=question,
                        mode=mode,
                        k=3,
                    )

                    st.markdown(answer)

                    # Save assistant response
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    save_chat_history(workspace_id, st.session_state.chat_messages)

                    # Save retrieval info
                    st.session_state.last_docs = docs
                    st.session_state.last_search_query = search_query

                except Exception as e:
                    st.error(f"error: {e}")