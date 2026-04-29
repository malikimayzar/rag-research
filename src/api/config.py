from enum import Enum
from pathlib import Path
from pydantic_settings import BaseSettings


class RetrievalBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"


class Settings(BaseSettings):
    # ── API ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8003

    # ── Backend ────────────────────────────────────────────
    retrieval_backend: RetrievalBackend = RetrievalBackend.QDRANT
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC

    # ── Paths ──────────────────────────────────────────────
    data_path: Path = Path("data/processed")
    qdrant_path: Path = Path("data/qdrant_storage")
    bm25_chunks_path: str = "data/processed/chunks_semantic.json"

    # ── Retrieval ──────────────────────────────────────────
    default_top_k: int = 10
    candidate_k: int = 50

    hybrid_rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Filtering ──────────────────────────────────────────
    score_gap_threshold: float = 2.0
    min_score_threshold: float = 0.0

    # ── Generation ─────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_model_fast: str = "llama-3.1-8b-instant"
    generation_max_tokens: int = 512
    generation_temperature: float = 0.1
    generation_max_chars: int = 3000

    # ── Multi-query ────────────────────────────────────────
    use_multi_query: bool = False
    use_hyde: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()