"""
Configuration file for AI-powered Test Case Generator PoC
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
EXPORTS_DIR = BASE_DIR / "exports"
EXAMPLES_DIR = BASE_DIR / "examples"

# Ensure export directories exist
EXPORTS_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True)

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-4-turbo"  # Using GPT-4 as specified

# Agentic Configuration
MAX_ITERATIONS = 5
COVERAGE_THRESHOLD = 0.80
MIN_COVERAGE_IMPROVEMENT = 0.05  # Minimum improvement to continue iterating

# Generation Configuration
TEMPERATURE = 0.7
MAX_TOKENS = 2000
TIMEOUT = 60  # API timeout in seconds

# UI Configuration
STREAMLIT_PAGE_TITLE = "AI Test Case Generator"
STREAMLIT_PAGE_ICON = "🧪"
STREAMLIT_LAYOUT = "wide"

# Quality Metrics
BLEU_WEIGHTS = (0.25, 0.25, 0.25, 0.25)  # 4-gram BLEU

# Export Configuration
PDF_OPTIONS = {
    'page-size': 'A4',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': "UTF-8",
    'enable-local-file-access': None
}

# Example user stories
EXAMPLE_STORIES = [
    {
        "title": "User Registration",
        "story": "As a user, I want to register an account with email and password so that I can access the platform.",
        "context": "Email must be valid format. Password minimum 8 characters with numbers and special characters."
    },
    {
        "title": "File Upload",
        "story": "As a user, I want to upload files up to 10MB so that I can share documents with my team.",
        "context": "Supported formats: PDF, DOCX, TXT. File size limit: 10MB."
    },
    {
        "title": "Shopping Cart",
        "story": "As a customer, I want to add items to my shopping cart and see the total price so that I can review before checkout.",
        "context": "Support quantity updates, item removal, and price calculations including tax."
    }
]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
