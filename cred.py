from dotenv import load_dotenv
import os
from log_config import get_logger

logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API keys
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API keys
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    raise ValueError(
        "GOOGLE_API_KEY is not set. Please add it to your .env file:\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )

logger.info("API credentials loaded successfully")