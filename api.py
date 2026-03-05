from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from sentence_transformers import SentenceTransformer

from src.retrieval.bm25_retriever import load_bm25, bm25_search
from src.retrieval.embedder import load_index, dense_search
from src.retrieval.hybrid_retriever import hybrid_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Retrieval Service", version="1.0.0")

bm25 = None
bm25_chunks = None
dense_index = None
embed_model = None

@app.on_event("startup")
def startup():
    global bm25, bm25_chunks, dense_index, embed_model
    logger.info("Loading BM25 index...")
    bm25, bm25_chunks = load_bm25("data/processed/index_bm25")
    logger.info("Loading dense index...")
    dense_index = load_index("data/processed/index_minilm")
    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("All indexes loaded")

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    method: str = "bm25"

class RetrieveResponse(BaseModel):
    results: list[dict]
    method: str
    total: int

def normalize(raw: list) -> list:
    out = []
    for r in raw:
        if isinstance(r, dict):
            out.append({
                "text": r.get("text", r.get("content", "")),
                "score": float(r.get("score", r.get("rrf_score", 0.0))),
                "chunk_id": str(r.get("chunk_id", r.get("id", ""))),
            })
        else:
            out.append({"text": str(r), "score": 0.0, "chunk_id": ""})
    return out

@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-research"}

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if req.method not in ["bm25", "dense", "hybrid"]:
        raise HTTPException(status_code=400, detail="method must be bm25, dense, or hybrid")

    try:
        if req.method == "bm25":
            raw = bm25_search(req.query, bm25, bm25_chunks, req.top_k)
        elif req.method == "dense":
            raw = dense_search(req.query, dense_index, embed_model, req.top_k)
        else:
            result = hybrid_search(req.query, dense_index, bm25, bm25_chunks, embed_model, req.top_k)
            raw = result.get("hybrid", [])
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")

    normalized = normalize(raw)
    return RetrieveResponse(results=normalized, method=req.method, total=len(normalized))
