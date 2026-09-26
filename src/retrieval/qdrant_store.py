from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

CHUNK_ID_NAMESPACE = uuid.UUID('12345678-1234-5678-1234-567812345678')

COLLECTION_NAME = "rag_research"
QDRANT_PATH     = "./data/qdrant_storage"
VECTOR_DIM      = 768
BATCH_SIZE      = 128


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id:   str
    text:     str
    score:    float
    metadata: dict

class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        qdrant_path: str = QDRANT_PATH,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        vector_dim: int = VECTOR_DIM,
    ):
        self.collection_name = os.getenv("QDRANT_COLLECTION", collection_name)
        self.vector_dim = vector_dim

        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        qdrant_path = os.getenv("QDRANT_PATH", qdrant_path or QDRANT_PATH)

        # ── Qdrant client ──────────────────────────────────────────
        if qdrant_url:
            print(f"[QdrantStore] Connecting to Qdrant Cloud: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            print(f"[QdrantStore] Using local Qdrant storage: {qdrant_path}")
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=qdrant_path)

        # ── Embedder dari registry — tidak reload ──────────────────
        from src.retrieval.model_registry import ModelRegistry
        self.embedder = ModelRegistry.get().embedder
        print(f"[QdrantStore] Embedder reused from ModelRegistry")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )
            print(f"[QdrantStore] Collection '{self.collection_name}' created")
        else:
            count = self.client.count(self.collection_name).count
            print(f"[QdrantStore] Collection '{self.collection_name}' exists ({count} vectors)")

    def index_chunks(self, chunks: List[dict], reset: bool = False) -> None:
        if reset:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()

        print(f"[QdrantStore] Indexing {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        all_embeddings = self.embedder.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False
        )

        points = []
        for chunk, emb in zip(chunks, all_embeddings):
            point_id = uuid.uuid5(CHUNK_ID_NAMESPACE, chunk["chunk_id"]).int >> 64
            vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "doc_id":   chunk["doc_id"],
                    "text":     chunk["text"],
                    **chunk.get("metadata", {}),
                }
            ))

        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i+BATCH_SIZE]
            )
        print(f"[QdrantStore] Indexing done.")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        query_emb = self.embedder.encode([f"Represent this sentence: {query}"], show_progress_bar=False)[0]
        query_emb = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
        qdrant_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=filter_doc_id))])
            if filter_doc_id else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=k,
            query_filter=qdrant_filter,
        )

        return [
            RetrievalResult(
                chunk_id=r.payload["chunk_id"],
                doc_id=r.payload["doc_id"],
                text=r.payload["text"],
                score=r.score,
                metadata={
                    key: v for key, v in r.payload.items()
                    if key not in ("chunk_id", "doc_id", "text")
                },
            )
            for r in response.points
        ]
    def close(self):
        if hasattr(self, 'client'):
            try:
                self.client.close()
                print("[QdrantStore] Connection closed gracefully.")
            except Exception:
                pass