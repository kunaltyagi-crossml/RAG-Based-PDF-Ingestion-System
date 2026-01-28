# 📚 RAG PDF Assistant - Intelligent Document Q&A System

This project demonstrates how to build an intelligent document analysis system using LangChain, Google Gemini API, and FAISS vector database, with support for accurate question-answering, source attribution, and strict document-boundary enforcement.

---

## 🚀 Project Overview

**RAG PDF Assistant** is an AI-powered application that enables intelligent question-answering from PDF documents using Retrieval-Augmented Generation (RAG) technology. Upload any PDF document and ask questions to get accurate, source-backed answers extracted directly from the document content.

### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
1. **Information Retrieval** - Finding relevant document sections using vector similarity
2. **Language Generation** - Using LLM to generate answers from retrieved context
3. **Source Grounding** - Ensuring all responses are backed by actual document content

### Key Capabilities

The system is organized around **4 core functionalities**:

#### 1. Part 1 - PDF Processing & Vectorization
* Load PDF documents using PyPDF
* Split text into semantic chunks (1000 chars, 200 overlap)
* Generate embeddings using Google Generative AI
* Store vectors in FAISS index for fast retrieval

#### 2. Part 2 - LLM + Document Integration
* Integrate Google Gemini with document context
* Automatically retrieve relevant sections for queries
* Example queries:
  * `"What is the main topic of this document?"`
  * `"Who are the authors and what are their affiliations?"`
  * `"Summarize the key findings in section 3"`

#### 3. Part 3 - Intelligent Response System
* Answer questions ONLY from document content
* Clear "I don't know" responses for out-of-scope queries
* Example behaviors:
  * Document question: `"What methodology was used?"` → Detailed answer with quotes
  * Off-topic question: `"What is the capital of France?"` → "I don't know based on the provided document"
* Logs reasoning process and source attribution

#### 4. Part 4 - Advanced Features
* **Dual API Mode** - Traditional and message-based LangChain syntax
* **Source Attribution** - View exact document sections used
* **Conversation History** - Contextual follow-up questions
* **Temperature Control** - Adjust response creativity
* Example query: `"Compare the approaches in sections 2 and 4, then extract all dates mentioned"`

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/kunaltyagi-crossml/RAG-Based-PDF-Ingestion-System.git
cd RAG_BASED_CHATBOT
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install langchain langchain-community langchain-google-genai langchain-core langchain-text-splitters faiss-cpu pypdf streamlit python-dotenv tiktoken
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a file named `.env` in the root directory of this project and add your API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 📖 Usage

### 1. Core RAG Application

The Core RAG Application supports:
* PDF document upload and processing
* Semantic question answering
* Source document attribution
* Multi-turn conversations
* Temperature and retrieval control

**Run the application:**
```bash
streamlit run streamlit.py
```

The app will open in your browser at `http://localhost:8501`

#### Example Queries:

**Document-Based Questions (Will Get Answers):**
```
✅ "What is the main topic of this document?"
✅ "Who created Python and when?"
✅ "Summarize the methodology section"
✅ "List all the key findings"
✅ "What does the document say about machine learning?"
```

**Off-Topic Questions (Will Get "I Don't Know"):**
```
❌ "What is the capital of France?"
❌ "Tell me about quantum physics"
❌ "What's the weather today?"
❌ "Recommend similar papers"
```

The examples are invoked automatically and demonstrate:
* Basic question-answering
* Multi-turn conversations
* Document summarization
* Information extraction
* Fact-checking

---

## 🔧 How It Works

### Processing Pipeline

1. **PDF Upload** → User selects PDF document
2. **Text Extraction** → PyPDF extracts text from all pages
3. **Text Chunking** → Split into 1000-char chunks with 200-char overlap
4. **Embedding Generation** → Google Generative AI creates vector embeddings
5. **Vector Storage** → FAISS stores embeddings for fast similarity search
6. **Query Processing** → User asks question
7. **Retrieval** → FAISS finds top K most similar chunks (default: 4)
8. **Answer Generation** → Gemini generates answer using retrieved context
9. **Response** → User receives answer with optional source attribution

### System Behavior

* **User Input** → Received as text query through Streamlit interface
* **System Prompts** → Detailed instructions in `prompts.py` enforce document-only responses
* **Tool Selection** → Automatically chooses between:
  * Traditional RetrievalQA chain
  * Modern message-based API
* **Tool Outputs** → Retrieved document chunks are treated as authoritative
* **Final Responses** → Human-friendly, accurate, and honest (says "I don't know" when appropriate)

### Key Design Principles

✅ **Document Boundaries** - Strict adherence to uploaded content only  
✅ **No Hallucination** - Never fabricates information  
✅ **Transparent Sourcing** - Shows which sections support answers  
✅ **Clear Limitations** - Explicitly states when information is unavailable  
✅ **Evidence-Based** - Quotes relevant passages  

---

## 🎯 Learning Outcomes

By exploring this project, you will:

* ✅ Understand **RAG architecture** and how it prevents hallucination
* ✅ Learn **vector similarity search** using FAISS
* ✅ Master **LangChain integration** with Google Gemini
* ✅ Design **strict system prompts** with DOs and DON'Ts
* ✅ Implement **document-boundary enforcement** in AI systems
* ✅ Handle **Pydantic validation** and message schemas
* ✅ Build **production-ready LLM applications** with proper error handling
* ✅ Create **beautiful UIs** with Streamlit and custom CSS
* ✅ Implement **comprehensive logging** for debugging
* ✅ Use both **traditional and modern** LangChain APIs

---

## 🎨 Key Features

### 1. Intelligent Document Processing
- Upload PDF documents of any size
- Automatic text extraction and semantic chunking
- Fast vector similarity search using FAISS
- Persistent storage for quick reloading

### 2. Accurate Question Answering
- Answers strictly from document content
- Quotes relevant passages for evidence
- Clear "I don't know" for unavailable information
- No fabrication or hallucination

### 3. Beautiful User Interface
- Modern gradient design with smooth animations
- Real-time chat interface
- Interactive conversation history
- Source document viewer
- Mobile-responsive layout

### 4. Advanced Configuration
- **Dual API Support** - Traditional RetrievalQA or message-based
- **Temperature Control** - Adjust response creativity (0.0 - 1.0)
- **Retrieval Settings** - Configure number of source chunks (1-8)
- **Logging System** - Comprehensive debugging logs
- **Session Management** - Clear history and start fresh

### 5. Developer-Friendly
- Well-organized modular code
- Comprehensive docstrings
- Type hints throughout
- Detailed logging
- Example scripts included

---

## 💻 Technical Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Google Gemini 1.5 Flash | Answer generation |
| **Framework** | LangChain | RAG orchestration |
| **Vector DB** | FAISS (IndexFlatL2) | Similarity search |
| **Embeddings** | Google Generative AI | Text vectorization |
| **PDF Parser** | PyPDF | Document processing |
| **Web UI** | Streamlit | User interface |
| **Logging** | Python logging | Debug & monitoring |

### Python Libraries
```python
langchain>=0.1.0                    # LLM framework
langchain-community>=0.0.20         # Community tools
langchain-google-genai>=0.0.6       # Google AI integration
langchain-core>=0.1.0               # Core components
langchain-text-splitters>=0.0.1     # Text processing
faiss-cpu>=1.7.4                    # Vector search
pypdf>=3.17.0                       # PDF parsing
streamlit>=1.29.0                   # Web interface
python-dotenv>=1.0.0                # Environment config
tiktoken>=0.5.2                     # Token management
```

---

## 📊 Performance Metrics

| PDF Size | Pages | Processing Time | Query Response | RAM Usage |
|----------|-------|----------------|----------------|-----------|
| Small | 1-10 | 5-15 seconds | 2-3 seconds | 4GB |
| Medium | 10-50 | 30-90 seconds | 3-5 seconds | 8GB |
| Large | 50-200 | 2-5 minutes | 5-8 seconds | 16GB |
| Very Large | 200+ | 5-10 minutes | 8-12 seconds | 32GB |

*Note: Times vary based on CPU speed, network latency, and document complexity*

---

## 🚀 Future Enhancements

- [ ] **Multiple PDF Support** - Upload and query across multiple documents
- [ ] **Export Functionality** - Download chat history as PDF/JSON/CSV
- [ ] **Custom Prompts UI** - Edit system prompts in the interface
- [ ] **PDF Highlighting** - Visual highlighting of source sections
- [ ] **Conversation Persistence** - Save and load chat sessions
- [ ] **OCR Integration** - Process scanned PDFs with text recognition
- [ ] **Multi-format Support** - DOCX, TXT, HTML, Markdown files
- [ ] **Advanced Search** - Hybrid keyword + semantic search
- [ ] **Batch Processing** - Process multiple files simultaneously
- [ ] **REST API** - Programmatic access for integrations

#
