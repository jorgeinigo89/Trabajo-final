import os

from dotenv import load_dotenv
from my_package.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class Settings:
    """Class to manage environment variables and project configurations."""

    def __init__(self):
        logger.info("Loading environment variables...")
        load_dotenv()  # This looks for the .env file

        self.database_url = os.getenv("DATABASE_URL")
        self.api_key = os.getenv("FINANCIAL_API_KEY")

        if not self.database_url:
            logger.warning("DATABASE_URL not found. Check your .env file.")
        else:
            logger.info("Settings loaded successfully.")
