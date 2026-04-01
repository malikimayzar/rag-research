from __future__ import annotations

import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# ── Lazy globals (loaded once at startup) ──────────────────────
store = None
retriever = None
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, retriever, generator
    print("[STARTUP] Loading RAG Research Master Engine...")

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.generation.generator import GroqGenerator

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator(model="llama-3.3-70b-versatile")

    count = store.client.count(store.collection_name).count
    print(f"[STARTUP] Ready — {count} vectors loaded")
    yield
    print("[SHUTDOWN] RAG Research Master Engine stopped")


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
    method: str = "hybrid"  # hybrid, dense, bm25 — for mcp-gateway compat


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
        "generator": "groq/llama-3.3-70b-versatile",
        "chunks": "rust_semantic_982",
    }


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.time()
    results = retriever.search(
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
        retrieval_method="hybrid_rrf+bge_reranker" if req.use_reranker else "hybrid_rrf",
        latency_ms=latency_ms,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Retrieve
    t0 = time.time()
    chunks = retriever.search(
        query=req.query,
        top_k=req.top_k,
        use_reranker=req.use_reranker,
    )
    latency_retrieval_ms = round((time.time() - t0) * 1000, 2)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # Generate
    t1 = time.time()
    response = generator.generate(req.query, chunks)
    latency_generation_ms = round((time.time() - t1) * 1000, 2)

    return GenerateResponse(
        query=req.query,
        answer=response.answer,
        contexts=[c["text"] if isinstance(c, dict) else c.text for c in chunks],
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
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
