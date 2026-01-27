import streamlit as st
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from pdf_loader import load_pdf
from text_splitter import split_documents
from vector_store import create_vector_store, load_vector_store
from retriever import get_retriever
from client import get_llm
from prompts import rag_prompt
from log_config import get_logger

from langchain.chains import RetrievalQA

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        min-height: 400px;
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.8rem 0;
        margin-left: 15%;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant message */
    .assistant-message {
        background: #f7f9fc;
        color: #1e1e1e;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.8rem 0;
        margin-right: 15%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2.5rem;
        background: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        color: #666;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Timestamp styling */
    .timestamp {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .assistant-message .timestamp {
        color: #999;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        border-left: 4px solid #17a2b8;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border-left: 4px solid #ffc107;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border-left: 4px solid #dc3545;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader */
    .uploadedFile {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
    }
    
    /* Source document expander */
    .source-doc {
        background: #f9f9f9;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #666;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
    logger.info("Initialized QA chain in session state")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Initialized chat history in session state")

if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None

# Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">📚 RAG PDF Assistant</div>
        <div class="header-subtitle">Upload your PDF and ask intelligent questions - Powered by Google Gemini</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF file to start asking questions"
    )
    
    if uploaded_file is not None:
        st.markdown(f'<div class="status-badge status-success">✅ {uploaded_file.name}</div>', 
                    unsafe_allow_html=True)
        
        if st.button("🔄 Process PDF", use_container_width=True):
            with st.spinner("Processing PDF... This may take a moment"):
                try:
                    logger.info(f"Starting PDF processing: {uploaded_file.name}")
                    
                    # Create data directory
                    os.makedirs("data", exist_ok=True)
                    pdf_path = f"data/{uploaded_file.name}"
                    
                    # Save uploaded file
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    logger.info(f"PDF saved to {pdf_path}")
                    
                    # Load and process PDF
                    docs = load_pdf(pdf_path)
                    logger.info(f"Loaded {len(docs)} pages from PDF")
                    
                    chunks = split_documents(docs)
                    logger.info(f"Split into {len(chunks)} chunks")
                    
                    # Create vector store
                    vector_db = create_vector_store(chunks)
                    logger.info("Vector store created successfully")
                    
                    # Create retriever and QA chain
                    retriever = get_retriever(vector_db)
                    llm = get_llm()
                    
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": rag_prompt},
                        return_source_documents=True
                    )
                    
                    st.session_state.pdf_name = uploaded_file.name
                    logger.info("QA chain initialized successfully")
                    
                    st.success("✅ PDF processed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    st.markdown("---")
    
    # Load existing index if available
    if st.session_state.qa_chain is None:
        if st.button("📂 Load Existing Index", use_container_width=True):
            with st.spinner("Loading existing vector store..."):
                try:
                    vector_db = load_vector_store()
                    if vector_db:
                        retriever = get_retriever(vector_db)
                        llm = get_llm()
                        
                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=retriever,
                            chain_type="stuff",
                            chain_type_kwargs={"prompt": rag_prompt},
                            return_source_documents=True
                        )
                        
                        logger.info("Loaded existing FAISS index")
                        st.success("✅ Loaded existing index!")
                    else:
                        st.warning("⚠️ No existing index found")
                except Exception as e:
                    logger.error(f"Error loading index: {str(e)}", exc_info=True)
                    st.error(f"❌ Error loading index: {str(e)}")
    
    st.markdown("---")
    
    # Chat history management
    st.markdown("### 💬 Chat History")
    
    if st.session_state.chat_history:
        st.markdown(f'<div class="status-badge status-info">📊 {len(st.session_state.chat_history)} messages</div>', 
                    unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            logger.info("Chat history cleared")
            st.rerun()
    else:
        st.info("No messages yet")
    
    st.markdown("---")
    
    # Current document info
    st.markdown("### 📄 Current Document")
    if st.session_state.pdf_name:
        st.success(f"📗 {st.session_state.pdf_name}")
    else:
        st.info("No document loaded")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    num_sources = st.slider("Number of source chunks", 1, 8, 4)
    show_sources = st.checkbox("Show source documents", value=True)

# Main chat interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    for idx, chat in enumerate(st.session_state.chat_history):
        timestamp = chat.get('timestamp', datetime.now().strftime("%I:%M %p"))
        
        # User message
        st.markdown(f"""
            <div class="user-message">
                <strong>You</strong><br>
                {chat['user']}
                <div class="timestamp">{timestamp}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Assistant</strong><br>
                {chat['ai']}
                <div class="timestamp">{timestamp}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show sources if enabled
        if show_sources and 'sources' in chat and chat['sources']:
            with st.expander(f"📄 View {len(chat['sources'])} Source Documents"):
                for i, doc in enumerate(chat['sources'], 1):
                    st.markdown(f"""
                        <div class="source-doc">
                            <strong>Source {i}</strong><br>
                            {doc.page_content[:300]}...
                        </div>
                    """, unsafe_allow_html=True)
                    if i < len(chat['sources']):
                        st.markdown("---")
else:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">👋</div>
            <h3>Welcome to RAG PDF Assistant!</h3>
            <p>Upload a PDF document from the sidebar and start asking questions.</p>
            <p>I'll help you find answers based on the document's content.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Query input
st.markdown("---")

if st.session_state.qa_chain:
    query = st.chat_input("💬 Ask a question about your PDF document...")
    
    if query:
        logger.info(f"User query: {query}")
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from QA chain
                response = st.session_state.qa_chain.invoke({"query": query})
                
                answer = response["result"]
                sources = response.get("source_documents", [])
                
                logger.info(f"Generated response with {len(sources)} sources")
                
                # Add to chat history
                timestamp = datetime.now().strftime("%I:%M %p")
                st.session_state.chat_history.append({
                    "user": query,
                    "ai": answer,
                    "sources": sources,
                    "timestamp": timestamp
                })
                
                logger.info("Added to chat history")
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                st.error(f"❌ Error: {str(e)}")
        
        st.rerun()
else:
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 12px;">
            <h4 style="color: #667eea;">⚠️ Please upload and process a PDF document first</h4>
            <p style="color: #666;">Use the sidebar to upload your document and get started!</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: white; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 12px;">
        <p style="margin: 0; font-size: 1rem;">
            🚀 Powered by <strong>LangChain</strong> + <strong>Streamlit</strong> + <strong>Google Gemini</strong>
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            Built with ❤️ for intelligent document analysis
        </p>
    </div>
""", unsafe_allow_html=True)