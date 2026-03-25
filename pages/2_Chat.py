import streamlit as st

from ui.helpers import ensure_chat_state

st.title("Chat with Interview RAG")

workspace_id = st.session_state.get("workspace_id", "test")
st.write(f"Current workspace: **{workspace_id}**")

ensure_chat_state()

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""

mode = st.selectbox(
    "Mode",
    ["ask", "answer_as_me", "generate_questions", "gap_analysis"],
)

if st.button("Clear Chat"):
    st.session_state.chat_messages = []
    from rag.answer_generation import HistoryAwareAnswerGenerator
    st.session_state.rag_generator = HistoryAwareAnswerGenerator()
    st.session_state.last_docs = []
    st.session_state.last_search_query = ""
    st.rerun()

left, right = st.columns([2, 1])

with left:
    st.subheader("Conversation")

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

with right:
    st.subheader("Retrieval Info")

    search_query = st.session_state.get("last_search_query")
    if search_query:
        st.write("**Search query used:**")
        st.code(search_query)

    docs = st.session_state.get("last_docs", [])
    if docs:
        st.write("**Retrieved sources:**")
        for i, doc in enumerate(docs, start=1):
            with st.expander(
                f"{i}. {doc.metadata.get('file_name')} | {doc.metadata.get('source_type')}"
            ):
                st.write(f"Chunk index: {doc.metadata.get('chunk_index')}")
                st.text(doc.page_content[:1000])

question = st.chat_input("Ask a question about your documents")

if question:
    st.session_state.chat_messages.append(
        {"role": "user", "content": question}
    )

    with left:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    generator = st.session_state.rag_generator
                    answer, docs, search_query = generator.ask(
                        workspace_id=workspace_id,
                        user_question=question,
                        mode=mode,
                        k=3,
                    )

                    st.markdown(answer)

                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    st.session_state.last_docs = docs
                    st.session_state.last_search_query = search_query

                except Exception as e:
                    st.error(f"Failed to answer: {e}")