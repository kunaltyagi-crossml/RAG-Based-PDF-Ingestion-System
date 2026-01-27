import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from cred import GOOGLE_API_KEY
from log_config import get_logger

logger = get_logger(__name__)

FAISS_PATH = "faiss_index"


def get_embeddings():
    """
    Initialize and return Google Generative AI embeddings.
    
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    try:
        logger.info("Initializing Google Generative AI embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}", exc_info=True)
        raise


def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks using latest LangChain syntax.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        FAISS vector store instance
    """
    try:
        logger.info(f"Creating vector store from {len(chunks)} chunks")
        embeddings = get_embeddings()
        
        # Get embedding dimension
        embedding_dim = len(embeddings.embed_query("hello world"))
        logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Initialize FAISS vector store with new syntax
        vector_db = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Add documents to the vector store
        vector_db.add_documents(chunks)
        logger.info(f"Added {len(chunks)} documents to vector store")
        
        # Save the vector store locally
        vector_db.save_local(FAISS_PATH)
        logger.info(f"Vector store created and saved to {FAISS_PATH}")
        
        return vector_db
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise


def load_vector_store():
    """
    Load an existing FAISS vector store from disk.
    
    Returns:
        FAISS vector store instance or None if not found
    """
    try:
        if os.path.exists(FAISS_PATH):
            logger.info(f"Loading existing vector store from {FAISS_PATH}")
            embeddings = get_embeddings()
            
            vector_db = FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Vector store loaded successfully")
            return vector_db
        else:
            logger.warning(f"Vector store path {FAISS_PATH} does not exist")
            return None
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
        return None