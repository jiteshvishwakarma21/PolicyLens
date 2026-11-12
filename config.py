"""Central configuration for the PolicyLens project."""

# Embedding model
EMBEDDING_MODEL = "intfloat/e5-base-v2"
EMBEDDING_BATCH_SIZE = 32

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

# Dataset paths
DEFAULT_INPUT = "data/pdfs"
DEFAULT_MANIFEST = "data/manifest.csv"

# ChromaDB
CHROMA_DB_PATH = "data/chroma_db"
DEFAULT_COLLECTION = "policylens_final_30docs"
FIXED_COLLECTION = "policylens_fixed_30docs"
