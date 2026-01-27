from langchain_google_genai import ChatGoogleGenerativeAI
from cred import GOOGLE_API_KEY
from log_config import get_logger

logger = get_logger(__name__)


def get_llm(model: str = "gemini-1.5-flash", temperature: float = 0):
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