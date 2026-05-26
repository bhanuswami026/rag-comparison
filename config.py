"""Shared configuration for the RAG comparison app."""

EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

LLM_PROVIDERS = {
    "Gemini Flash mini": {
        "provider": "gemini",
        "model": "gemini-2.5-flash-lite",
        "api_key_env": "GEMINI_API_KEY",
    },
    "GPT-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_TOP_K = 4
MIN_TOP_K = 1
MAX_TOP_K = 8
