import os
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from pdf_loader import load_pdf
from text_splitter import split_documents
from vector_store import create_vector_store, load_vector_store
from retriever import get_retriever
from client import get_llm
from prompts import rag_prompt
from log_config import get_logger

# ---------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------

logger = get_logger(__name__)
load_dotenv()

logger.info("Starting RAG PDF Assistant")

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="centered"
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None
    logger.info("Session retriever initialized")

if "llm" not in st.session_state:
    st.session_state.llm = None
    logger.info("Session LLM initialized")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Chat history initialized")

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ---------------------------------------------------
# UI HEADER
# ---------------------------------------------------

st.title("📚 RAG PDF Assistant")
st.caption("Ask questions directly from your PDF document")

# ---------------------------------------------------
# PDF INGESTION
# ---------------------------------------------------

st.subheader("📄 Upload PDF")

uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)

if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("📥 Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                logger.info(f"PDF upload started: {uploaded_file.name}")

                os.makedirs("data", exist_ok=True)
                pdf_path = f"data/{uploaded_file.name}"

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                logger.info("PDF saved locally")

                documents = load_pdf(pdf_path)
                logger.info(f"Loaded {len(documents)} pages")

                chunks = split_documents(documents)
                logger.info(f"Split into {len(chunks)} chunks")

                vector_db = create_vector_store(chunks)
                logger.info("Vector store created")

                st.session_state.retriever = get_retriever(vector_db)
                st.session_state.llm = get_llm()

                st.session_state.pdf_name = uploaded_file.name

                st.success("✅ PDF processed successfully")
                logger.info("PDF ingestion completed")

            except Exception as e:
                logger.exception("PDF processing failed")
                st.error(str(e))

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------

st.divider()
st.subheader("💬 Chat History")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Assistant:** {chat['answer']}")
        st.caption(chat["time"])
        st.markdown("---")
else:
    st.info("No questions asked yet.")

# ---------------------------------------------------
# QUERY SECTION
# ---------------------------------------------------

st.divider()
st.subheader("❓ Ask Question")

if st.session_state.retriever and st.session_state.llm:

    query = st.text_input(
        "Enter your question",
        placeholder="Ask something from the PDF..."
    )

    if st.button("🚀 Ask") and query:

        logger.info(f"User query: {query}")

        with st.spinner("Thinking..."):
            try:
                # FIX: use invoke instead of similarity_search
                docs = st.session_state.retriever.invoke(query)

                logger.info(f"Retrieved {len(docs)} chunks")

                context = "\n\n".join(doc.page_content for doc in docs)

                messages = rag_prompt.format_messages(
                    context=context,
                    question=query
                )

                response = st.session_state.llm.invoke(messages)

                answer = response.content

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "time": datetime.now().strftime("%I:%M %p")
                })

                logger.info("Answer generated successfully")

                st.rerun()

            except Exception as e:
                logger.exception("Query processing failed")
                st.error(str(e))


else:
    st.warning("⚠️ Please upload and process a PDF first.")
