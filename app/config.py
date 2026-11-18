from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
EMPLOYEES_FILE = Path(os.getenv("EMPLOYEES_FILE", ROOT / "employees.json"))
TICKETS_FILE = Path(os.getenv("TICKETS_FILE", ROOT / "tickets.json"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hr_it_policies")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
MAX_TOOL_CALLS = 5
FALLBACK = "I couldn't find this in the provided documents."
