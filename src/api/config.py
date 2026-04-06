from enum import Enum
from pydantic_settings import BaseSettings
from pathlib import Path

class RetrievalBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"

class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Backend selection
    retrieval_backend: RetrievalBackend = RetrievalBackend.QDRANT
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    
    # Paths
    data_path: Path = Path("data/processed")
    qdrant_path: Path = Path("data/qdrant_storage")
    
    # Retrieval config
    default_top_k: int = 5
    hybrid_rrf_k: int = 60
    
    # Generation
    groq_api_key: str
    groq_model: str = "mixtral-8x7b-32768"
    
    class Config:
        env_file = ".env"
settings = Settings()