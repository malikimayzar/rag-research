"""
model_registry.py — Global singleton untuk semua model.
Load SEKALI saat startup, reuse selamanya.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class ModelRegistry:
    _instance = None

    def __init__(self):
        print("[ModelRegistry] Loading embedder...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print("[ModelRegistry] Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)
        print("[ModelRegistry] All models loaded. Will reuse for all queries.")

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance
