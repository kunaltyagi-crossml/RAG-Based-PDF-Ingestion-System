from langchain_text_splitters import RecursiveCharacterTextSplitter
from log_config import get_logger

logger = get_logger(__name__)


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    try:
        logger.info(f"Splitting {len(documents)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Successfully split into {len(chunks)} chunks")
        
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}", exc_info=True)
        raise