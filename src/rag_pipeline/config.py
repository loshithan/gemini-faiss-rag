from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Chunking (measured in words)
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# Retrieval
TOP_K = 4

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # gemini-embedding-001 supports 768, 1536 or 3072
EMBED_BATCH_SIZE = 100  # max texts per embed_content request
GENERATION_MODEL = "gemini-3.8-flash"

# Environment
API_KEY_ENV_VAR = "GOOGLE_API_KEY"  # name of env var containing API key; the value is read in embedder.get_client()
