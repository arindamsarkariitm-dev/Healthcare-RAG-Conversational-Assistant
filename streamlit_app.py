import os

import streamlit as st

from chatbot import HealthcareRAGAssistant
from ingest import build_vector_store, VECTOR_STORE_DIR


def ensure_vector_store() -> None:
    if not VECTOR_STORE_DIR.exists() or not any(VECTOR_STORE_DIR.iterdir()):
        build_vector_store()


def format_sources(sources):
    return ", ".join(sources) if sources else "No source citations available."


def format_retrieval_items(items):
    lines = []
    for idx, item in enumerate(items, start=1):
        location = []
        if item.get("page") is not None:
            location.append(f"page {item['page']}")
        if item.get("chunk_index") is not None:
            location.append(f"chunk {item['chunk_index']}")
        location_text = f" ({', '.join(location)})" if location else ""
        lines.append(
            f"### {idx}. Source: {item['source']}{location_text}\n"
            f"- **Score:** {item['score']:.4f}\n"
            f"- **Content:** {item['content']}"
        )
    return "\n\n".join(lines)


def summarize_retrieval(items):
    if not items:
        return None
    scores = [item["score"] for item in items]
    sources = {item["source"] for item in items}
    return {
        "top_score": max(scores),
        "average_score": sum(scores) / len(scores),
        "retrieved_chunks": len(items),
        "distinct_sources": len(sources),
    }


st.set_page_config(page_title="Healthcare RAG Assistant", layout="centered")
st.title("Healthcare RAG Assistant")
st.write(
    "Ask healthcare questions based on patient education guides and hospital FAQs. "
    "Answers are retrieved from source documents and grounded in the retrieved content."
)

api_key = os.environ.get("New_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.warning(
        "OpenAI API key not found. Please set the environment variable `New_OPENAI_API_KEY` or `OPENAI_API_KEY` before running the app."
    )
    st.stop()

with st.sidebar:
    st.header("Instructions")
    st.write(
        "1. Place PDF documents in the `docs/` folder.\n"
        "2. Click **Rebuild vector store** after adding or updating documents.\n"
        "3. Ask a question and review sources and retrieval context."
    )
    if st.button("Rebuild vector store"):
        try:
            build_vector_store()
            st.success("Vector store rebuilt successfully.")
            st.session_state.assistant = HealthcareRAGAssistant()
        except Exception as exc:
            st.error(f"Unable to rebuild vector store: {exc}")

ensure_vector_store()

if "assistant" not in st.session_state:
    st.session_state.assistant = HealthcareRAGAssistant()
    st.session_state.messages = []
    st.session_state.retrievals = []
    st.session_state.question = ""
    st.session_state.reset_message = ""

if "question" not in st.session_state:
    st.session_state.question = ""

if "reset_message" not in st.session_state:
    st.session_state.reset_message = ""


def reset_conversation():
    st.session_state.assistant.reset()
    st.session_state.messages = []
    st.session_state.retrievals = []
    st.session_state.question = ""
    st.session_state.reset_message = "Conversation history cleared."

question = st.text_area("Ask a healthcare question", height=140, key="question")
cols = st.columns([3, 1])
with cols[1]:
    st.button("Reset conversation", on_click=reset_conversation)

if st.session_state.reset_message:
    st.success(st.session_state.reset_message)
    st.session_state.reset_message = ""

if st.button("Get answer"):
    if not question.strip():
        st.warning("Please enter a question before asking.")
    else:
        with st.spinner("Retrieving relevant healthcare content..."):
            response = st.session_state.assistant.chat(question)
            st.session_state.messages.append(
                {
                    "question": question,
                    "answer": response["answer"],
                    "sources": response["sources"],
                }
            )
            st.session_state.retrievals = st.session_state.assistant.retrieve(question)

tab1, tab2 = st.tabs(["Conversation", "Retrieval Details"])

with tab1:
    for idx, message in enumerate(st.session_state.messages, start=1):
        st.markdown(f"**Q{idx}:** {message['question']}")
        st.markdown(f"**A{idx}:** {message['answer']}")
        st.markdown(f"**Sources:** {format_sources(message['sources'])}")
        st.divider()

with tab2:
    if st.session_state.retrievals:
        st.markdown("### Retrieved passages and scores")
        stats = summarize_retrieval(st.session_state.retrievals)
        if stats:
            st.metric("Top retrieval score", f"{stats['top_score']:.4f}")
            st.metric("Average retrieval score", f"{stats['average_score']:.4f}")
            st.metric("Chunks retrieved", stats['retrieved_chunks'])
            st.metric("Distinct sources", stats['distinct_sources'])
        st.markdown(format_retrieval_items(st.session_state.retrievals))
    else:
        st.info("Ask a question to see retrieved source passages and similarity scores.")

st.subheader("Feedback")
feedback_col1, feedback_col2 = st.columns(2)
with feedback_col1:
    rating = st.radio("Was this answer helpful?", ["Yes", "No"], horizontal=True)
with feedback_col2:
    comment = st.text_input("Optional feedback")
if st.button("Submit feedback"):
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        st.session_state.assistant.save_feedback(last["question"], last["answer"], rating, comment)
        st.success("Thanks for your feedback.")
    else:
        st.warning("Ask a question before submitting feedback.")
