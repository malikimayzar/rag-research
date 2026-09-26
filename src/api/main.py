from __future__ import annotations

import time
import logging
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.api.config import settings

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("rag.api")

# Lazy globals 
store = None
retriever = None
generator = None
confidence_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, retriever, generator, confidence_engine
    logger.info("[STARTUP] Initializing RAG Research Master Engine (PHASE 1 - BASELINE)...")

    try:
        from src.controller.confidence_engine import ConfidenceEngine
        from src.retrieval.qdrant_store import QdrantVectorStore
        from src.retrieval.hybrid_retriever import MasterHybridRetriever
        from src.generation.generator import GroqGenerator

        confidence_engine = ConfidenceEngine()
        store = QdrantVectorStore()
        retriever = MasterHybridRetriever(
            vector_store=store,
            bm25_chunks_path=settings.bm25_chunks_path,
            rrf_k=settings.hybrid_rrf_k,
            use_multi_query=settings.use_multi_query,
            use_hyde=settings.use_hyde,
        )

        generator = GroqGenerator(model=settings.groq_model)
        count = store.client.count(store.collection_name).count
        logger.info(f"[STARTUP] Success — {count} vectors loaded into memory")
        logger.info(
            f"[STARTUP] CONFIG — multi_query={settings.use_multi_query} | "
            f"hyde={settings.use_hyde} | model={settings.groq_model}"
        )
    except Exception as e:
        logger.error(f"[STARTUP] Critical Error: {str(e)}")
        raise e
    yield

    if store:
        logger.info("[SHUTDOWN] Closing Vector Store connections...")
        store.close()

app = FastAPI(
    title="RAG Research Master Engine",
    description="Production-ready RAG (PHASE 1 - Simplified Baseline)",
    version="1.0.0-baseline",
    lifespan=lifespan,
)

# Pydantic Models 
class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)

class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = False
class GenerateResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[ChunkResult]
    retrieval_method: str
    latency_retrieval_ms: float
    latency_generation_ms: float
    latency_total_ms: float
    model: str
    confidence: float
    decision: str
    retry_triggered: bool = False
    failure_type: str = "none"

# Helpers 
def _format_chunks(chunks: list, limit: int | None = None) -> list[ChunkResult]:
    target = chunks[:limit] if limit else chunks
    return [
        ChunkResult(
            chunk_id=c.get("chunk_id", "unk"),
            doc_id=c.get("doc_id", "unk"),
            text=c.get("text", ""),
            score=round(float(c.get("retrieval_score", 0)), 4),
            metadata=c.get("metadata", {})
        )
        for c in target
    ]


def _log_query_event(**event) -> None:
    """Emit one machine-readable event for every completed API query."""
    logger.info(json.dumps(event, ensure_ascii=False, default=str))


def _abstain_response(
    *,
    query: str,
    chunks: list,
    retrieval_method: str,
    latency_retrieval_ms: float,
    latency_total_ms: float,
    confidence: float,
    retry_triggered: bool,
    failure_type: str,
) -> GenerateResponse:
    return GenerateResponse(
        query=query,
        answer="The available information is not sufficient to provide a reliable answer.",
        retrieved_chunks=_format_chunks(chunks, limit=2),
        retrieval_method=retrieval_method,
        latency_retrieval_ms=latency_retrieval_ms,
        latency_generation_ms=0.0,
        latency_total_ms=latency_total_ms,
        model="none",
        confidence=confidence,
        decision="REJECT",
        retry_triggered=retry_triggered,
        failure_type=failure_type,
    )

# Endpoints
@app.get("/health")
async def health():
    if not store:
        raise HTTPException(status_code=503, detail="Store not initialized")
    count = store.client.count(store.collection_name).count
    return {
        "status": "ready",
        "vectors": count,
        "mode": "PHASE1_BASELINE",
        "multi_query": settings.use_multi_query,
        "hyde": settings.use_hyde,
        "reranker": True,
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(req: GenerateRequest):
    if not all([retriever, generator, confidence_engine]):
        raise HTTPException(status_code=503, detail="Engine not ready")
    rid = str(uuid.uuid4())[:8]
    t_api_start = time.time()
    retry_triggered = False

    try:
        # Retrieval is deliberately bounded: initial attempt plus one BM25-heavy retry.
        t_retrieval_start = time.time()
        chunks = []
        conf_eval = {"confidence_score": 0.0, "decision": "REJECT", "signals": {}}
        for attempt in range(2):
            search_kwargs = {"query": req.query, "top_k": req.top_k}
            if attempt == 1:
                retry_triggered = True
                search_kwargs.update({"dense_weight": 0.3, "bm25_weight": 0.7})

            chunks = await run_in_threadpool(retriever.search, **search_kwargs)
            conf_eval = confidence_engine.calculate_confidence(chunks)
            if chunks and conf_eval["decision"] != "REJECT":
                break

        latency_retrieval_ms = round((time.time() - t_retrieval_start) * 1000, 2)
        confidence = conf_eval["confidence_score"]
        decision   = conf_eval["decision"]

        logger.info(
            f"[{rid}] RETRIEVAL DONE | "
            f"chunks={len(chunks)} | confidence={confidence} | decision={decision} | "
            f"retrieval_ms={latency_retrieval_ms}"
        )

        # NO RESULTS 
        if not chunks:
            logger.warning(f"[{rid}] NO_RESULTS | query='{req.query[:60]}'")
            result = _abstain_response(
                query=req.query, chunks=[], retrieval_method="none",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                confidence=0.0, retry_triggered=retry_triggered,
                failure_type="retrieval_miss",
            )
            _log_query_event(query_id=rid, query=req.query, answer_status="abstained", failure_type=result.failure_type, retry_triggered=retry_triggered, latency_ms={"retrieval": latency_retrieval_ms, "generation": 0.0, "total": result.latency_total_ms})
            return result

        # REJECT
        if decision == "REJECT":
            logger.warning(
                f"[{rid}] REJECT | query='{req.query[:60]}' | "
                f"confidence={confidence} | signals={conf_eval['signals']}"
            )
            result = _abstain_response(
                query=req.query, chunks=chunks, retrieval_method="rejected_low_confidence",
                latency_retrieval_ms=latency_retrieval_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                confidence=confidence, retry_triggered=retry_triggered,
                failure_type="low_confidence",
            )
            _log_query_event(query_id=rid, query=req.query, answer_status="abstained", failure_type=result.failure_type, retry_triggered=retry_triggered, latency_ms={"retrieval": latency_retrieval_ms, "generation": 0.0, "total": result.latency_total_ms})
            return result
        
        if decision == "PARTIAL_TRUST":
            logger.warning(
                f"[{rid}] PARTIAL_TRUST | query='{req.query[:60]}' | "
                f"confidence={confidence} | generating with top-3 chunks only"
            )
            t_gen_start = time.time()
            response = await run_in_threadpool(
                generator.generate,
                req.query,
                chunks[:3],  
            )
            latency_generation_ms = round((time.time() - t_gen_start) * 1000, 2)

            result = GenerateResponse(
                query=req.query,
                answer=f"[Low confidence — verify independently] {response.answer}",
                retrieved_chunks=_format_chunks(chunks, limit=3),
                retrieval_method=response.retrieval_method,
                latency_retrieval_ms=latency_retrieval_ms,
                latency_generation_ms=latency_generation_ms,
                latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
                model=response.model,
                confidence=confidence,
                decision=decision,
                retry_triggered=retry_triggered,
            )
            _log_query_event(query_id=rid, query=req.query, answer_status="answered", failure_type="none", retry_triggered=retry_triggered, latency_ms={"retrieval": latency_retrieval_ms, "generation": latency_generation_ms, "total": result.latency_total_ms})
            return result

        # GENERATE 
        t_gen_start = time.time()
        response = await run_in_threadpool(generator.generate, req.query, chunks)
        latency_generation_ms = round((time.time() - t_gen_start) * 1000, 2)
        latency_total_ms = round((time.time() - t_api_start) * 1000, 2)
        logger.info(
            f"[{rid} COMPLETE]"
            f"retrieval_ms={latency_retrieval_ms}|"
            f"generation_ms={latency_generation_ms}|"
        )
        result = GenerateResponse(
            query=req.query,
            answer=response.answer,
            retrieved_chunks=_format_chunks(chunks),
            retrieval_method=response.retrieval_method,
            latency_retrieval_ms=latency_retrieval_ms,
            latency_generation_ms=latency_generation_ms,
            latency_total_ms=latency_total_ms,
            model=response.model,
            confidence=confidence,
            decision=decision,
            retry_triggered=retry_triggered,
        )
        _log_query_event(query_id=rid, query=req.query, answer_status="answered", failure_type="none", retry_triggered=retry_triggered, latency_ms={"retrieval": latency_retrieval_ms, "generation": latency_generation_ms, "total": latency_total_ms})
        return result
    
    except Exception as e:
        logger.error(f"[{rid}] CRITICAL ERROR | {str(e)}", exc_info=True)
        result = _abstain_response(
            query=req.query, chunks=[], retrieval_method="failed_safe_abstain",
            latency_retrieval_ms=0.0,
            latency_total_ms=round((time.time() - t_api_start) * 1000, 2),
            confidence=0.0, retry_triggered=retry_triggered,
            failure_type="pipeline_error",
        )
        _log_query_event(query_id=rid, query=req.query, answer_status="abstained", failure_type=result.failure_type, retry_triggered=retry_triggered, latency_ms={"retrieval": 0.0, "generation": 0.0, "total": result.latency_total_ms})
        return result


@app.get("/metrics")
async def metrics():
    if not store:
        raise HTTPException(status_code=503, detail="Service not ready")
    count = store.client.count(store.collection_name).count
    return {
        "vectors_indexed": count,
        "engine_version": "1.0.0-baseline",
        "phase": "PHASE_1",
        "retrieval_stack": ["Qdrant", "BM25"],
        "reranker": settings.reranker_model,
        "multi_query": settings.use_multi_query,
        "hyde": settings.use_hyde,
        "governor": "ConfidenceEngine_v1",
        "llm_provider": "Groq",
        "llm_model": settings.groq_model,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=False)
