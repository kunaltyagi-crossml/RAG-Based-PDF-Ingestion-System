from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from cred import GOOGLE_API_KEY
from log_config import get_logger

logger = get_logger(__name__)


def get_llm(model: str = "gemini-2.5-flash-lite", temperature: float = 0):
    """
    Initialize and return a Google Generative AI LLM.
    
    Args:
        model: Model name to use
        temperature: Temperature for response generation (0-1)
        
    Returns:
        ChatGoogleGenerativeAI instance
    """
    try:
        logger.info(f"Initializing LLM (model={model}, temperature={temperature})")
        
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY
        )
        
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
        raise

def get_embeddings():
    """
    Initialize and return Google Generative AI embeddings.
    
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    try:
        logger.info("Initializing Google Generative AI embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {str(e)}", exc_info=True)
        raise
