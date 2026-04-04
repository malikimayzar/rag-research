from __future__ import annotations
import argparse
import json
import os
import time
import re

from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever, MULTI_QUERY_PROMPT, HYDE_PROMPT
from src.generation.generator import GroqGenerator, build_context
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# Helpers 
def compute_context_overlap(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    answer_words = set(re.sub(r'[^\w\s]', '', answer.lower()).split())
    context_text = " ".join(contexts).lower()
    context_words = set(re.sub(r'[^\w\s]', '', context_text).split())
    # Hapus stopwords basic
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                 "at", "to", "for", "of", "and", "or", "but", "it", "this",
                 "that", "with", "as", "by", "from", "be", "has", "have"}
    answer_words -= stopwords
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return round(len(overlap) / len(answer_words), 4)

def classify_error(
    retrieved_chunks: list,
    reranked_chunks: list,
    answer: str,
    ground_truth: Optional[str],
    source_chunk_id: Optional[str],
    context_overlap: float,
) -> dict:
    # Cek apakah source chunk ada di retrieved
    retrieved_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
    reranked_ids  = [c.get("chunk_id", "") for c in reranked_chunks]

    def _is_hit(source_id, id_set):
        if not source_id:
            return None
        return any(rid == source_id or rid.startswith(source_id + "_sub") for rid in id_set)

    retrieval_hit = _is_hit(source_chunk_id, retrieved_ids)
    ranking_hit   = _is_hit(source_chunk_id, reranked_ids)

    # Classify
    if source_chunk_id and not retrieval_hit:
        failure_type = "retrieval_miss"
        reason = f"Source chunk '{source_chunk_id}' tidak ditemukan di retrieved candidates"
    elif source_chunk_id and retrieval_hit and not ranking_hit:
        failure_type = "ranking_error"
        reason = f"Source chunk ditemukan tapi tidak masuk top-k setelah reranking"
    elif context_overlap < 0.3:
        failure_type = "generation_fail"
        reason = f"Context overlap rendah ({context_overlap}) — answer tidak grounded di context"
    else:
        failure_type = "ok"
        reason = "Pipeline berjalan normal"

    return {
        "failure_type": failure_type,
        "reason": reason,
        "retrieval_hit": retrieval_hit,
        "ranking_hit": ranking_hit,
    }

# Main Runner 
def run_single_query(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
    ground_truth: Optional[str] = None,
    source_chunk_id: Optional[str] = None,
    save_output: bool = True,
) -> dict:
    timings = {}
    output = {}

    print(f"\n{'='*60}")
    print(f"  QUERY: {query}")
    print(f"  top_k={top_k} | reranker={use_reranker}")
    print(f"{'='*60}\n")

    # Init 
    store     = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator(model="llama-3.3-70b-versatile")
    groq      = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Multi-Query + HyDE parallel 
    print("[Step 1+2] Running Multi-Query expansion + HyDE (parallel)...")
    t0 = time.time()

    expanded_queries = [query]
    hyde_doc = None

    def call_groq(prompt, temperature, max_tokens):
        resp = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_mq   = executor.submit(call_groq, MULTI_QUERY_PROMPT.format(query=query), 0.7, 100)
        f_hyde = executor.submit(call_groq, HYDE_PROMPT.format(query=query), 0.5, 150)

    try:
        raw = f_mq.result(timeout=5)
        alternatives = [q.strip() for q in raw.split("\n") if q.strip()][:2]
        expanded_queries = [query] + alternatives
    except Exception as e:
        print(f"  [WARN] Multi-Query failed: {e}")

    try:
        hyde_doc = f_hyde.result(timeout=5)
    except Exception as e:
        print(f"  [WARN] HyDE failed: {e}")

    timings["query_expansion_ms"]  = round((time.time() - t0) * 1000, 2)
    timings["hyde_ms"]             = timings["query_expansion_ms"]  # parallel, sama

    print(f"  Expanded queries : {expanded_queries}")
    print(f"  HyDE doc         : {hyde_doc[:80] if hyde_doc else 'None'}...")
    print(f"  Latency          : {timings['query_expansion_ms']}ms\n")

    # Dense + BM25 per query 
    print("[Step 3] Running dense + BM25 retrieval...")
    candidate_k = min(top_k * 4, 20)

    t0 = time.time()
    all_dense_hits = []
    for q in expanded_queries:
        all_dense_hits.append(retriever.vector_store.search(q, k=candidate_k))
    if hyde_doc:
        all_dense_hits.append(retriever.vector_store.search(hyde_doc, k=candidate_k))
    timings["embedding_ms"] = round((time.time() - t0) * 1000, 2)

    t0 = time.time()
    all_bm25_hits = []
    import numpy as np
    for q in expanded_queries:
        scores = retriever._bm25.get_scores(q.lower().split())
        top_idx = np.argsort(scores)[::-1][:candidate_k]
        all_bm25_hits.append([
            {"chunk_id": retriever._chunks[i]["chunk_id"],
             "text": retriever._chunks[i]["text"],
             "score": float(scores[i]),
             "doc_id": retriever._chunks[i]["doc_id"]}
            for i in top_idx if scores[i] > 0
        ])
    timings["bm25_ms"] = round((time.time() - t0) * 1000, 2)

    print(f"  Dense sources : {len(all_dense_hits)} | BM25 sources: {len(all_bm25_hits)}")
    print(f"  Embedding     : {timings['embedding_ms']}ms")
    print(f"  BM25          : {timings['bm25_ms']}ms\n")

    # Qdrant search latency 
    t0 = time.time()
    _ = retriever.vector_store.search(query, k=1)  # warmup probe
    timings["qdrant_search_ms"] = round((time.time() - t0) * 1000, 2)

    # RRF Fusion 
    print("[Step 5] RRF fusion...")
    t0 = time.time()
    fused_results = retriever._rrf_fuse(all_dense_hits, all_bm25_hits, candidate_k)
    timings["rrf_ms"] = round((time.time() - t0) * 1000, 2)

    # Format pre-rerank chunks
    pre_rerank_chunks = []
    for i, r in enumerate(fused_results[:top_k * 2]):
        pre_rerank_chunks.append({
            "rank": i + 1,
            "chunk_id": r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id"),
            "doc_id":   r.doc_id   if hasattr(r, "doc_id")   else r.get("doc_id"),
            "text":     (r.text    if hasattr(r, "text")      else r.get("text", ""))[:200] + "...",
            "score":    round(r.score if hasattr(r, "score") else r.get("score", 0), 4),
        })
    print(f"  Fused candidates: {len(fused_results)} | RRF: {timings['rrf_ms']}ms\n")

    # Reranking 
    print("[Step 6] Reranking...")
    t0 = time.time()
    rerank_candidates = fused_results[:10]

    if use_reranker and rerank_candidates:
        def _best_window(query: str, text: str, window: int = 600, step: int = 200) -> str:
            if len(text) <= window:
                return text
            q_words = set(query.lower().split())
            best_score, best_start = -1, 0
            for start in range(0, len(text) - window, step):
                snippet = text[start:start + window]
                score = sum(1 for w in q_words if w in snippet.lower())
                if score > best_score:
                    best_score, best_start = score, start
            return text[best_start:best_start + window]

        pairs = [
            [query, _best_window(query, r.text if hasattr(r, "text") else r.get("text", ""))]
            for r in rerank_candidates
        ]
        rerank_scores = retriever.reranker.predict(pairs)
        for i, hit in enumerate(rerank_candidates):
            score = float(rerank_scores[i])
            if hasattr(hit, "score"):
                hit.score = score
            else:
                hit["score"] = score
        rerank_candidates = sorted(
            rerank_candidates,
            key=lambda x: x.score if hasattr(x, "score") else x.get("score", 0),
            reverse=True,
        )

    timings["rerank_ms"] = round((time.time() - t0) * 1000, 2)

    post_rerank_chunks = []
    for i, r in enumerate(rerank_candidates[:top_k]):
        post_rerank_chunks.append({
            "rank":   i + 1,
            "chunk_id": r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id"),
            "doc_id":   r.doc_id   if hasattr(r, "doc_id")   else r.get("doc_id"),
            "text":     (r.text if hasattr(r, "text") else r.get("text", ""))[:200] + "...",
            "score":    round(r.score if hasattr(r, "score") else r.get("score", 0), 4),
        })

    print(f"  Reranked top-{top_k}: {timings['rerank_ms']}ms\n")

    # Build context 
    final_chunks_full = rerank_candidates[:top_k]
    final_context = build_context(final_chunks_full, max_chars=3000)

    # Generation 
    print("[Step 8] Generating answer...")
    t0 = time.time()
    response = generator.generate(query, final_chunks_full)
    timings["generation_ms"] = round((time.time() - t0) * 1000, 2)
    print(f"  Generation: {timings['generation_ms']}ms\n")

    # Confidence v1 
    context_texts = [
        r.text if hasattr(r, "text") else r.get("text", "")
        for r in final_chunks_full
    ]
    confidence_v1 = compute_context_overlap(response.answer, context_texts)

    # Error decomposition 
    post_rerank_full = [
        {"chunk_id": r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id")}
        for r in rerank_candidates[:top_k]
    ]
    pre_rerank_full = [
        {"chunk_id": r.chunk_id if hasattr(r, "chunk_id") else r.get("chunk_id")}
        for r in fused_results[:candidate_k]
    ]
    error_info = classify_error(
        retrieved_chunks=pre_rerank_full,
        reranked_chunks=post_rerank_full,
        answer=response.answer,
        ground_truth=ground_truth,
        source_chunk_id=source_chunk_id,
        context_overlap=confidence_v1,
    )

    # Total latency 
    timings["total_ms"] = round(
        timings["query_expansion_ms"] +
        timings["embedding_ms"] +
        timings["bm25_ms"] +
        timings["rerank_ms"] +
        timings["generation_ms"],
        2
    )

    # Assemble output 
    output = {
        "query":             query,
        "expanded_queries":  expanded_queries,
        "hyde_doc":          hyde_doc,
        "pre_rerank_chunks": pre_rerank_chunks,
        "reranked_chunks":   post_rerank_chunks,
        "final_context":     final_context[:500] + "..." if len(final_context) > 500 else final_context,
        "answer":            response.answer,
        "ground_truth":      ground_truth,
        "latency_breakdown": timings,
        "confidence_v1":     confidence_v1,
        "error_decomposition": error_info,
        "retrieval_method":  "hybrid_multiquery_hyde_rerank" if use_reranker else "hybrid_multiquery_rrf",
        "config": {
            "top_k":         top_k,
            "use_reranker":  use_reranker,
            "reranker":      "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "generator":     "llama-3.3-70b-versatile",
            "embedding":     "all-MiniLM-L6-v2",
            "chunking":      "rust_semantic_982",
        }
    }

    # Print summary 
    print(f"{'='*60}")
    print(f"  ANSWER:\n  {response.answer[:300]}")
    print(f"\n  CONFIDENCE V1 (context_overlap): {confidence_v1}")
    print(f"  ERROR TYPE: {error_info['failure_type']}")
    print(f"  REASON: {error_info['reason']}")
    print(f"\n  LATENCY BREAKDOWN:")
    for k, v in timings.items():
        bar = "█" * int(v / 100)
        print(f"    {k:<25} {v:>8.1f}ms  {bar}")
    print(f"{'='*60}\n")

    # Save output 
    if save_output:
        out_path = Path("results/logs/single_query_debug.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if out_path.exists():
            with open(out_path) as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        existing.append(output)
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[Saved] → {out_path}")
    return output

# CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Single Query Debugger")
    parser.add_argument("--query",      type=str, default=None)
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--no-reranker",action="store_true")
    parser.add_argument("--gt",         type=str, default=None, help="Ground truth answer")
    parser.add_argument("--source",     type=str, default=None, help="Source chunk ID")
    parser.add_argument("--from-gt",    action="store_true", help="Run dari ground_truth_qa.json (first 3)")
    args = parser.parse_args()

    if args.from_gt:
        with open("data/processed/ground_truth_qa.json") as f:
            gt_data = json.load(f)
        for item in gt_data[:3]:
            run_single_query(
                query=item["question"],
                top_k=args.top_k,
                use_reranker=not args.no_reranker,
                ground_truth=item.get("ground_truth"),
                source_chunk_id=item.get("source_chunk"),
            )
    else:
        query = args.query or "What is attention mechanism in transformer models?"
        run_single_query(
            query=query,
            top_k=args.top_k,
            use_reranker=not args.no_reranker,
            ground_truth=args.gt,
            source_chunk_id=args.source,
        )

# Singleton cache 
_STORE     = None
_RETRIEVER = None
_GENERATOR = None

def get_components():
    global _STORE, _RETRIEVER, _GENERATOR
    if _STORE is None:
        from src.retrieval.qdrant_store import QdrantVectorStore
        from src.retrieval.hybrid_retriever import MasterHybridRetriever
        from src.generation.generator import GroqGenerator
        _STORE     = QdrantVectorStore()
        _RETRIEVER = MasterHybridRetriever(vector_store=_STORE)
        _GENERATOR = GroqGenerator(model="llama-3.3-70b-versatile")
        print("[Cache] Models loaded once — reuse for subsequent queries")
    return _STORE, _RETRIEVER, _GENERATOR