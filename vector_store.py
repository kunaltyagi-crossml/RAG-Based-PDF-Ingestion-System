import os
from langchain_community.vectorstores import FAISS
from cred import GOOGLE_API_KEY
from log_config import get_logger
from client import get_embeddings

logger = get_logger(__name__)

FAISS_PATH = "faiss_index"


def create_vector_store(chunks):
    """
    Create a FAISS vector store from document chunks.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        FAISS vector store instance
    """
    try:
        logger.info(f"Creating vector store from {len(chunks)} chunks")
        embeddings = get_embeddings()
        
        vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        
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