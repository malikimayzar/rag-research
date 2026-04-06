from __future__ import annotations

import time
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("rag.api")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
from src.api.config import settings

# ── Lazy globals (loaded once at startup) ──────────────────────
store = None
retriever = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, retriever, generator
    logger.info("[STARTUP] Loading RAG Research Master Engine...")

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.generation.generator import GroqGenerator

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator(model=settings.groq_model)

    count = store.client.count(store.collection_name).count
    logger.info(f"[STARTUP] Ready — {count} vectors loaded")
    yield
    logger.info("[SHUTDOWN] RAG Research Master Engine stopped")

app = FastAPI(
    title="RAG Research Master Engine",
    description="Hybrid retrieval (Rust chunks + Qdrant + BGE Reranker) + Groq generation",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Request / Response Models ───────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    method: str = "hybrid"  

class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float

class RetrieveResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    retrieval_method: str
    latency_ms: float

class GenerateRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True

class GenerateResponse(BaseModel):
    query: str
    answer: str
    contexts: list[str]
    retrieval_method: str
    latency_retrieval_ms: float
    latency_generation_ms: float
    model: str

# ── Endpoints ───────────────────────────────────────────────────
@app.get("/health")
async def health():
    if store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    count = store.client.count(store.collection_name).count
    return {
        "status": "ready",
        "service": "rag-research",
        "version": "2.0.0",
        "vectors": count,
        "retrieval": "hybrid_rrf+bge_reranker",
        "generator": "llama-3.1-8b-instant",
        "chunks": "rust_semantic_982",
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.time()
    results = await run_in_threadpool(
        retriever.search,
        query=req.query,
        top_k=req.top_k,
        use_reranker=req.use_reranker,
    )
    latency_ms = round((time.time() - t0) * 1000, 2)

    return RetrieveResponse(
        query=req.query,
        results=[
            ChunkResult(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                text=r["text"],
                score=round(r["retrieval_score"], 4),
            )
            for r in results
        ],
        retrieval_method="hybrid_multiquery_hyde_rerank" if req.use_reranker else "hybrid_rrf_multiquery",
        latency_ms=latency_ms,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Retrieve
    t0 = time.time()
    chunks = await run_in_threadpool(
        retriever.search,
        query=req.query,
        top_k=req.top_k,
        use_reranker=req.use_reranker,
    )
    latency_retrieval_ms = round((time.time() - t0) * 1000, 2)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # Generate
    t1 = time.time()
    response = await run_in_threadpool(generator.generate, req.query, chunks)
    latency_generation_ms = round((time.time() - t1) * 1000, 2)

    logger.info(json.dumps({
        "event": "generate",
        "query": req.query,
        "top_score": round(chunks[0].get("retrieval_score", chunks[0].get("score", 0)), 4) if chunks else None,
        "num_chunks_retrieved": req.top_k,
        "num_chunks_used": len(response.retrieved_chunks),
        "rejected": response.answer.startswith("The provided context does not"),
        "latency_retrieval_ms": latency_retrieval_ms,
        "latency_generation_ms": latency_generation_ms,
    }))

    return GenerateResponse(
        query=req.query,
        answer=response.answer,
        contexts=[c["text"] for c in response.retrieved_chunks],
        retrieval_method=response.retrieval_method,
        latency_retrieval_ms=latency_retrieval_ms,
        latency_generation_ms=latency_generation_ms,
        model=response.model,
    )

@app.get("/metrics")
async def metrics():
    if store is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    count = store.client.count(store.collection_name).count
    return {
        "vectors_indexed": count,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunking": "rust_semantic",
        "retrieval_methods": ["dense", "bm25", "hybrid_rrf", "hybrid_rrf+bge_reranker"],
        "reranker": "BAAI/bge-reranker-base",
        "llm": "groq/llama-3.3-70b-versatile",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_host, reload=False)