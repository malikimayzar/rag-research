from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# --- Config ---
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM       = 384
COLLECTION_NAME  = "rag_research"
QDRANT_PATH      = "./data/qdrant_storage"
BATCH_SIZE       = 128

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
        embed_model: str = EMBED_MODEL,
        vector_dim: int = VECTOR_DIM,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim

        if qdrant_url:
            print(f"[QdrantStore] Connecting to Qdrant Cloud: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            print(f"[QdrantStore] Using local Qdrant storage: {qdrant_path}")
            Path(qdrant_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=qdrant_path)

        print(f"[QdrantStore] Loading embedding model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)
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
            texts, batch_size=BATCH_SIZE, show_progress_bar=True
        )

        points = []
        for chunk, emb in zip(chunks, all_embeddings):
            point_id = abs(hash(chunk["chunk_id"])) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id":   chunk["doc_id"],
                        "text":     chunk["text"],
                        **chunk.get("metadata", {}),
                    }
                )
            )

        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i+BATCH_SIZE]
            )
        print(f"[QdrantStore] [OK] Indexing done.")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_doc_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        query_emb = self.embedder.encode([query])[0].tolist()
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
