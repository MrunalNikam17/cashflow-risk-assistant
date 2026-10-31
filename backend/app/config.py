import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if item.strip()
]

MAX_SESSION_COUNT = int(os.getenv("MAX_SESSION_COUNT", "50"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(2 * 60 * 60)))
UPLOAD_MAX_BYTES = int(os.getenv("UPLOAD_MAX_BYTES", str(5 * 1024 * 1024)))
UPLOAD_MAX_FILES = int(os.getenv("UPLOAD_MAX_FILES", "10"))
API_TITLE = "Cashflow & Risk Assistant API"
API_VERSION = "0.1.0"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
