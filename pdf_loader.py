from langchain_community.document_loaders import PyPDFLoader
from log_config import get_logger

logger = get_logger(__name__)


def load_pdf(pdf_path: str):
    """
    Load a PDF document and return its pages as documents.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects, one per page
    """
    try:
        logger.info(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages from PDF")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}", exc_info=True)
        raise