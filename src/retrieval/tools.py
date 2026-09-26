from __future__ import annotations

import asyncio
from typing import Any

from src.retrieval.qdrant_store import QdrantVectorStore, RetrievalResult
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.controller.confidence_engine import ConfidenceEngine
from src.generation.generator import GroqGenerator

class RetrievalTools:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        retriever: MasterHybridRetriever,
        confidence_engine: ConfidenceEngine,
        generator: GroqGenerator,
    ):
        self.vector_store = vector_store
        self.retriever = retriever
        self.confidence_engine = confidence_engine
        self.generator = generator

    async def search_dense(self, query: str, k: int = 10) -> dict[str, Any]:
        chunks = self.vector_store.search(query, k=k)
        return {"tool": "search_dense", "query": query, "chunks": chunks}

    async def search_bm25(self, query: str, k: int = 10) -> dict[str, Any]:
        chunks = self.retriever._bm25_search(query, k)
        return {"tool": "search_bm25", "query": query, "chunks": chunks}

    async def search_hybrid(
        self,
        query: str,
        k: int = 5,
        use_multi_query: bool = False,
        use_hyde: bool = False,
    ) -> dict[str, Any]:

        self.retriever.use_multi_query = use_multi_query
        self.retriever.use_hyde = use_hyde

        chunks = await asyncio.to_thread(
            self.retriever.search,
            query,
            top_k=k
        )

        return {
            "tool": "search_hybrid",
            "query": query,
            "chunks": chunks,
            "num_chunks": len(chunks),
        }
    async def rerank_candidates(self, candidates: list, query: str, top_n: int = 5) -> dict[str, Any]:
        if not candidates:
            return {"tool": "rerank_candidates", "query": query, "chunks": []}

        pairs = []
        for hit in candidates:
            text = hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
            pairs.append([query, text])

        try:
            scores = self.retriever.reranker.predict(pairs)
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: item[1],
                reverse=True,
            )
            ordered = [item[0] for item in ranked][:top_n]
            return {"tool": "rerank_candidates", "query": query, "chunks": ordered}
        except Exception as exc:
            return {"tool": "rerank_candidates", "query": query, "chunks": candidates[:top_n], "error": str(exc)}

    async def assess_retrieval_quality(self, chunks: list, query: str) -> dict[str, Any]:
        confidence = self.confidence_engine.calculate_confidence(chunks)
        return {"tool": "assess_retrieval_quality", "query": query, "confidence": confidence}

    async def generate_answer(
        self,
        query: str,
        chunks: list,
        max_tokens: int = 200,
        temperature: float = 0.0,
        source_chunk_id: str | None = None,
        min_top1_score: float = 0.0,
    ) -> dict[str, Any]:
        response = self.generator.generate(
            query=query,
            chunks=chunks,
            max_tokens=max_tokens,
            temperature=temperature,
            source_chunk_id=source_chunk_id,
            min_top1_score=min_top1_score,
        )
        return {
            "tool": "generate_answer",
            "query": query,
            "chunks": chunks,
            "response": response,
            "status": response.status,
            "confidence": response.confidence_score,
        }

    async def call_tool(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool == "search_hybrid":
            return await self.search_hybrid(**params)
        if tool == "search_dense":
            return await self.search_dense(**params)
        if tool == "search_bm25":
            return await self.search_bm25(**params)
        if tool == "rerank_candidates":
            return await self.rerank_candidates(**params)
        if tool == "assess_retrieval_quality":
            return await self.assess_retrieval_quality(**params)
        if tool == "generate_answer":
            return await self.generate_answer(**params)
        return {"tool": tool, "error": "unknown tool"}