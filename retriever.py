from log_config import get_logger

logger = get_logger(__name__)


def get_retriever(vector_db, search_type: str = "similarity", k: int = 4):
    """
    Create a retriever from the vector database.
    
    Args:
        vector_db: FAISS vector store instance
        search_type: Type of search ("similarity" or "mmr")
        k: Number of documents to retrieve
        
    Returns:
        Retriever instance
    """
    try:
        logger.info(f"Creating retriever (search_type={search_type}, k={k})")
        
        retriever = vector_db.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        logger.info("Retriever created successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {str(e)}", exc_info=True)
        raise