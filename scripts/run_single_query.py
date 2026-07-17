from __future__ import annotations
import argparse
import asyncio
import json
import os
import time
import re
import numpy as np
import uuid

from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever, MULTI_QUERY_PROMPT, HYDE_PROMPT
from src.generation.generator import GroqGenerator, build_context
from src.controller.policy_engine import PolicyEngine
from src.api.config import settings
from src.controller.confidence_engine import ConfidenceEngine
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# Constants
SECTION_SCORE_DELTA: dict[str, float] = {
    "abstract":     +0.1,
    "introduction": +0.1,
    "conclusion":   +0.05,
    "body":         +0.2,
    "references":   -2.0,
    "bibliography": -2.0,
}

BLOCKED_SECTIONS = {"references", "bibliography"} 
MAX_REF_IN_CONTEXT = 1  
SCORE_DISTRIBUTION_PATH = "data/debug/score_distribution.json"
RRF_K = 60
RRF_CIRCUIT_BREAKER_PERCENTILE = 0.50
RERANK_LATENCY_BUDGET_MS = 700.0
RERANK_GUARD_MAX_PAIRS = 6
MIN_CANDIDATES_FOR_RERANK = 3
RERANK_GAP_SKIP_THRESHOLD = 0.005

def load_retrieval_confidence_threshold(path: str = SCORE_DISTRIBUTION_PATH) -> float:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        mean = float(data.get("top1_mean", 0.0))
        std = float(data.get("top1_std", 0.0))
        threshold = max(mean - std, 0.0)
        print(
            f"[CALIBRATION] Retrieval threshold loaded from {path}: "
            f"mean={mean:.4f}, std={std:.4f}, threshold={threshold:.4f}"
        )
        return threshold
    except Exception as exc:
        print(f"[CALIBRATION] Could not load score distribution from {path}: {exc}")
        return 0.0

RETRIEVAL_MIN_TOP1_SCORE = 0.0 
MIN_CHUNK_LENGTH = 80
MAX_RETRIES = 2
RETRIEVAL_CONF_THRESHOLD = 0.3
MAX_RETRY_ATTEMPTS = 2
MAX_GENERATION_MS = 15000

_DOMAIN_KEYWORDS = [
    "attention", "transformer", "model", "neural", "learning",
    "retrieval", "embedding", "language", "training", "inference",
    "classification", "generation", "encoder", "decoder", "vector",
    "dataset", "evaluation", "benchmark", "fine-tuning", "pre-training",
]

def should_skip_reranker(candidate_count: int, top_gap: float, elapsed_ms: float) -> bool:
    if candidate_count < MIN_CANDIDATES_FOR_RERANK:
        return True
    if elapsed_ms > RERANK_LATENCY_BUDGET_MS:
        return True
    if top_gap > RERANK_GAP_SKIP_THRESHOLD:
        return True
    return False

def is_valid_query(q: str) -> bool:
    q_lower = q.lower()
    return any(kw in q_lower for kw in _DOMAIN_KEYWORDS)

def reciprocal_rank_fusion(results_lists: list[list], k: int = RRF_K) -> list:
    rrf_scores: dict[str, float] = {}
    id_to_item:  dict[str, object] = {}

    for results in results_lists:
        for rank, item in enumerate(results):
            cid = item.get("chunk_id") if isinstance(item, dict) else getattr(item, "chunk_id", "")
            if not cid:
                continue
            rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            id_to_item[cid]  = item  

    fused = sorted(
        id_to_item.values(),
        key=lambda x: rrf_scores.get(
            x.get("chunk_id") if isinstance(x, dict) else getattr(x, "chunk_id", ""),
            0.0,
        ),
        reverse=True,
    )

    for item in fused:
        cid   = item.get("chunk_id") if isinstance(item, dict) else getattr(item, "chunk_id", "")
        score = round(rrf_scores.get(cid, 0.0), 6)
        if isinstance(item, dict):
            item["score"] = score
        elif hasattr(item, "score"):
            item.score = score
    return fused

def rrf_circuit_breaker_threshold(num_lists: int, rank_depth: int, k: int = RRF_K) -> tuple[float, float, float]:
    if num_lists <= 0 or rank_depth <= 0:
        return 0.0, 0.0, 0.0
    min_score = 1.0 / (k + rank_depth)
    max_score = num_lists * (1.0 / (k + 1))
    threshold = min_score + RRF_CIRCUIT_BREAKER_PERCENTILE * (max_score - min_score)
    return threshold, min_score, max_score
 
# Helpers
def compute_context_overlap(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    answer_words  = set(re.sub(r'[^\w\s]', '', answer.lower()).split())
    context_words = set(re.sub(r'[^\w\s]', '', " ".join(contexts).lower()).split())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                 "at", "to", "for", "of", "and", "or", "but", "it", "this",
                 "that", "with", "as", "by", "from", "be", "has", "have"}
    answer_words -= stopwords
    if not answer_words:
        return 0.0
    return round(len(answer_words & context_words) / len(answer_words), 4)

def classify_error(
    retrieved_chunks: list,
    reranked_chunks: list,
    source_chunk_id: Optional[str],
    context_overlap: float,
    blocked_chunk_ids: Optional[set] = None,  
) -> dict:
    def _get_cid(c):
        return c.get("chunk_id", "") if isinstance(c, dict) else getattr(c, "chunk_id", "")

    retrieved_ids = [_get_cid(c) for c in retrieved_chunks]
    reranked_ids  = [_get_cid(c) for c in reranked_chunks]
    blocked_ids   = blocked_chunk_ids or set()

    def _is_hit(source_id, id_set):
        if not source_id:
            return None
        return any(rid == source_id or rid.startswith(source_id + "_sub") for rid in id_set)
    
    retrieval_hit = _is_hit(source_chunk_id, retrieved_ids)
    ranking_hit   = _is_hit(source_chunk_id, reranked_ids)
    source_was_blocked = source_chunk_id and source_chunk_id in blocked_ids

    if source_chunk_id and not retrieval_hit:
        failure_type = "retrieval_miss"
        reason = f"Source chunk '{source_chunk_id}' tidak ditemukan di retrieved candidates"
    elif source_chunk_id and retrieval_hit and source_was_blocked:
        failure_type = "filtering_error"
        reason = (
            f"Source chunk ditemukan di rank retrieval, tapi dibuang oleh section filter "
            f"(section=references) ” jawaban ada tapi pipeline sendiri yang block"
        )
    elif source_chunk_id and retrieval_hit and not ranking_hit:
        failure_type = "ranking_error"
        reason = "Source chunk ditemukan tapi tidak masuk top-k setelah reranking"
    elif context_overlap < 0.3:
        failure_type = "generation_fail"
        reason = f"Context overlap rendah ({context_overlap}) ” answer tidak grounded di context"
    else:
        failure_type = "ok"
        reason = "Pipeline berjalan normal"

    return {
        "failure_type":       failure_type,
        "reason":             reason,
        "retrieval_hit":      retrieval_hit,
        "ranking_hit":        ranking_hit,
        "source_was_blocked": source_was_blocked,
    }

# Object-safe accessors
def _get_attr(chunk, key: str, default=None):
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)

def _get_section(chunk) -> str:
    meta = _get_attr(chunk, "metadata", {}) or {}
    return meta.get("section", "body").lower()

def _chunk_to_summary(chunk, rank: int) -> dict:
    text = _get_attr(chunk, "text", "") or ""
    score = _get_attr(chunk, "rerank_score", None)
    if score is None:
        score = _get_attr(chunk, "retrieval_score", None)
    if score is None:
        score = _get_attr(chunk, "score", 0.0)

    return {
        "rank": rank,
        "chunk_id": _get_attr(chunk, "chunk_id", ""),
        "doc_id": _get_attr(chunk, "doc_id", ""),
        "section": _get_section(chunk),
        "score": round(float(score), 4),
        "length": len(text),
        "text": text[:200] + "...",
    }

# Reference filter
def is_valid_chunk(chunk) -> bool:
    return True 

# Context diversity ” soft reference limit
def limit_reference_chunks(chunks: list, max_ref: int = MAX_REF_IN_CONTEXT) -> tuple[list, set]:
    result      = []
    blocked_ids = set()
    ref_count   = 0

    for c in chunks:
        section = _get_section(c)
        if section in BLOCKED_SECTIONS:
            if ref_count >= max_ref:
                cid = _get_attr(c, "chunk_id", "")
                blocked_ids.add(cid)
                continue
            ref_count += 1
        result.append(c)

    if blocked_ids:
        print(
            f"  [REF_LIMIT] Allowed {ref_count} reference chunk(s), "
            f"capped {len(blocked_ids)} (max_ref={max_ref})"
        )
    return result, blocked_ids

def enforce_reference_policy(chunks: list, allow_ref: bool, max_ref_ratio: float, top_k: int) -> list:
    if not allow_ref or max_ref_ratio <= 0.0:
        return [c for c in chunks if _get_section(c) not in BLOCKED_SECTIONS][:top_k]

    non_ref_chunks = [c for c in chunks if _get_section(c) not in BLOCKED_SECTIONS]
    ref_chunks = [c for c in chunks if _get_section(c) in BLOCKED_SECTIONS]
    max_ref_budget = int(top_k * max_ref_ratio)

    selected = non_ref_chunks[:top_k]
    remaining_slots = top_k - len(selected)
    if remaining_slots > 0 and max_ref_budget > 0:
        selected += ref_chunks[:min(remaining_slots, max_ref_budget)]
    return selected

# Chunk quality guard
def is_quality_chunk(chunk) -> bool:
    text = _get_attr(chunk, "text", "") or ""
    return len(text) >= MIN_CHUNK_LENGTH

# Chunk quality logging (called on fused results before reranking)
def log_chunk_quality(chunks: list, label: str = "fused") -> None:
    bad = [c for c in chunks if not is_quality_chunk(c)]
    ref = [c for c in chunks if _get_section(c) in BLOCKED_SECTIONS]
    print(
        f"  [CHUNK_QUALITY:{label}] total={len(chunks)} | "
        f"too_short(<{MIN_CHUNK_LENGTH}chars)={len(bad)} | "
        f"references={len(ref)}"
    )

# Source chunk tracker
def track_source_chunk(source_chunk_id: Optional[str], fused_results: list) -> None:
    if not source_chunk_id:
        return

    for rank, r in enumerate(fused_results):
        cid = _get_attr(r, "chunk_id", "")
        if cid == source_chunk_id or cid.startswith(source_chunk_id + "_sub"):
            score = round(_get_attr(r, "score", 0) or 0, 4)
            print(
                f"  [SOURCE_TRACK] [OK] Found in fused | "
                f"rank={rank + 1}/{len(fused_results)} | score={score} | chunk_id={cid}"
            )
            return

    print(
        f"  [SOURCE_TRACK] [ERROR] NOT FOUND in fused results | "
        f"source_chunk_id='{source_chunk_id}' ” this is a retrieval miss"
    )

# Decision Step
def decide_action(query: str, groq_client: Groq) -> str:
    prompt = f"""
        Decide what to do next for this query:
        Options: "retrieve" or "answer"

        Query: {query}

        Respond with exactly one word: retrieve OR answer
    """
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        decision = resp.choices[0].message.content.strip().lower()
        return "retrieve" if "retrieve" in decision else "answer"
    except Exception:
        return "retrieve"

# Main Runner 
def run_single_query(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
    ground_truth: Optional[str] = None,
    source_chunk_id: Optional[str] = None,
    save_output: bool = True,
    mode: Literal["baseline", "full"] = "full",
) -> dict:
    timings: dict[str, float] = {}

    print(f"\n{'='*60}")
    print(f"  QUERY  : {query}")
    print(f"  MODE   : {mode.upper()}")
    print(f"  top_k  ={top_k} | reranker={use_reranker and mode == 'full'}")
    print(f"{'='*60}\n")

    store      = QdrantVectorStore()
    retriever  = MasterHybridRetriever(vector_store=store)
    generator  = GroqGenerator(model="llama-3.3-70b-versatile")
    groq       = Groq(api_key=os.getenv("GROQ_API_KEY"))
    confidence_engine = ConfidenceEngine()

    retrieval_confidence = {
        "confidence_score": 0.0,
        "decision": "REJECT"
    }
    policy   = PolicyEngine()
    decision = policy.resolve(query)

    q_type          = decision["query_type"]
    top_k           = decision["retrieval"]["top_k"]
    use_hyde        = decision["retrieval"]["use_hyde"]
    use_multi_query = decision["retrieval"]["use_multi_query"]
    allow_ref       = decision["reference"]["allow_references"]
    max_ref_ratio   = decision["reference"]["max_ref_ratio"]
    gen_max_tokens  = decision["generation"]["max_tokens"]
    gen_temperature = decision["generation"]["temperature"]

    print(f"[POLICY] query_type={q_type} | hyde={use_hyde} | multi_query={use_multi_query}")
    print(f"         allow_ref={allow_ref} | max_ref_ratio={max_ref_ratio}")
    print(f"         max_tokens={gen_max_tokens} | temperature={gen_temperature}")
    print(f"  Strategy: multi_query={use_multi_query} | hyde={use_hyde}")
    print(f"  Gen policy: max_tokens={gen_max_tokens} | temperature={gen_temperature}")

    # BASELINE MODE    
    if mode == "baseline":
        print("[Baseline] Dense-only retrieval...")
        expanded_queries = [query]
        hyde_doc         = None
        t0 = time.time()
        fused_results = list(retriever.vector_store.search(query, k=top_k))
        timings["embedding_ms"]        = round((time.time() - t0) * 1000, 2)
        timings["query_expansion_ms"]  = 0.0
        timings["hyde_ms"]             = 0.0
        timings["bm25_ms"]             = 0.0
        timings["rrf_ms"]              = 0.0
        timings["qdrant_search_ms"]    = timings["embedding_ms"]
        timings["rerank_ms"]           = 0.0

        rerank_candidates = fused_results[:top_k]
        retrieval_confidence = confidence_engine.calculate_confidence(rerank_candidates)
        print(f"  Dense hits: {len(fused_results)} | Embedding: {timings['embedding_ms']}ms")
        print(f"  [CONF_ENGINE] baseline confidence={retrieval_confidence['confidence_score']:.4f} decision={retrieval_confidence['decision']}")
        log_chunk_quality(fused_results, label="baseline_dense")

        track_source_chunk(source_chunk_id, fused_results)
        print()

    # Adaptive Proto-Agent    
    else:
        print("[Step 1+2] Adaptive expansion...")
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

        if use_multi_query:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_mq    = executor.submit(call_groq, MULTI_QUERY_PROMPT.format(query=query), 0.7, 100)
                f_hyde = executor.submit(call_groq, HYDE_PROMPT.format(query=query), 0.5, 150) if use_hyde else None

            try:
                raw = f_mq.result(timeout=5)
                alternatives = [q.strip() for q in raw.split("\n") if q.strip()][:2]
                expanded_queries = [query] + alternatives
            except Exception as e:
                print(f"  [WARN] Multi-Query failed: {e}")

            if use_hyde and f_hyde:
                try:
                    hyde_doc = f_hyde.result(timeout=5)
                except Exception as e:
                    print(f"  [WARN] HyDE failed: {e}")

        timings["query_expansion_ms"] = round((time.time() - t0) * 1000, 2)
        timings["hyde_ms"]            = timings["query_expansion_ms"]

        # Domain filter on expanded queries
        n_before = len(expanded_queries)
        expanded_queries = [q for q in expanded_queries if is_valid_query(q)]
        n_dropped = n_before - len(expanded_queries)
        if n_dropped:
            print(f"  [DOMAIN_FILTER] Dropped {n_dropped} off-topic expanded query(ies)")
        if not expanded_queries:
            expanded_queries = [query]
            print("  [DOMAIN_FILTER] All expansions dropped ” fallback to original query")
        print(f"  Expanded : {expanded_queries}")
        print(f"  HyDE     : {hyde_doc[:80] if hyde_doc else 'None'}...")
        print(f"  Latency  : {timings['query_expansion_ms']}ms\n")
        candidate_k = 30

        t0 = time.time()
        all_dense_hits: list[list] = []
        for q in expanded_queries:
            all_dense_hits.append(list(retriever.vector_store.search(q, k=candidate_k)))
        if hyde_doc:
            all_dense_hits.append(list(retriever.vector_store.search(hyde_doc, k=candidate_k)))
        timings["embedding_ms"] = round((time.time() - t0) * 1000, 2)

        # BM25 per expanded query
        print("BM25 retrieval...")
        t0 = time.time()
        all_bm25_hits: list[list] = []
        for q in expanded_queries:
            scores  = retriever._bm25.get_scores(q.lower().split())
            top_idx = np.argsort(scores)[::-1][:candidate_k]
            all_bm25_hits.append([
                {
                    "chunk_id": retriever._chunks[i]["chunk_id"],
                    "text":     retriever._chunks[i]["text"],
                    "score":    float(scores[i]),
                    "doc_id":   retriever._chunks[i]["doc_id"],
                    "metadata": retriever._chunks[i].get("metadata", {}),
                }
                for i in top_idx if scores[i] > 0
            ])
        timings["bm25_ms"] = round((time.time() - t0) * 1000, 2)
        print(f"  Dense lists: {len(all_dense_hits)} | BM25 lists: {len(all_bm25_hits)}")
        print(f"  Embedding: {timings['embedding_ms']}ms | BM25: {timings['bm25_ms']}ms\n")

        # Qdrant warmup probe
        t0 = time.time()
        _ = retriever.vector_store.search(query, k=1)
        timings["qdrant_search_ms"] = round((time.time() - t0) * 1000, 2)

        # Real RRF across all dense + BM25 lists
        print("[Step 5] Real RRF fusion...")
        t0 = time.time()
        fused_results = reciprocal_rank_fusion(all_dense_hits + all_bm25_hits)
        timings["rrf_ms"] = round((time.time() - t0) * 1000, 2)
        print(f"  Fused candidates: {len(fused_results)} | RRF: {timings['rrf_ms']}ms")
        log_chunk_quality(fused_results, label="post_rrf")
 
        valid_chunks = [
            c for c in fused_results
            if _get_section(c) not in BLOCKED_SECTIONS and is_quality_chunk(c)
        ]
        usable_ratio = len(valid_chunks) / max(len(fused_results), 1)
        print(f"  [QUALITY] usable_chunks={len(valid_chunks)}/{len(fused_results)} ({usable_ratio:.0%})")
        if usable_ratio < 0.5:
            print("  [QUALITY] WARNING: <50% usable ” corpus dirty or retrieval broken")

        track_source_chunk(source_chunk_id, fused_results)
        print()

        # CrossEncoder reranking + section penalty
        print("[Step 6] Reranking + section penalty...")
        t0 = time.time()
        # Drop quality-failing chunks before reranker sees them
        candidates_for_rerank = [
            c for c in fused_results[:candidate_k]
            if is_quality_chunk(c)
        ]
        rerank_n = min(candidate_k, top_k * 2)
        rerank_candidates = candidates_for_rerank[:rerank_n]

        if not allow_ref:
            rerank_candidates = [
                c for c in rerank_candidates
                if _get_section(c) not in BLOCKED_SECTIONS
            ]

        rrf_lists_count = len(all_dense_hits) + len(all_bm25_hits)
        rrf_threshold, rrf_min, rrf_max = rrf_circuit_breaker_threshold(
            num_lists=rrf_lists_count,
            rank_depth=candidate_k,
        )
        top_rrf_score = _get_attr(rerank_candidates[0], "score", 0) if rerank_candidates else 0

        if not (use_reranker and mode == "full"):
            skip_reranker = True 
            skip_reason = "use_reranker=False"
        elif len(rerank_candidates) < 2:
            skip_reranker = True
            skip_reason = "too few candidates to rerank"
        elif top_rrf_score >= rrf_threshold:
            skip_reranker = True
            skip_reason = f"top_rrf_score={top_rrf_score:.4f} >= threshold={rrf_threshold:.4f} (confident, skip)"
        else:
            skip_reranker = False
            skip_reason = f"top_rrf_score={top_rrf_score:.4f} < threshold={rrf_threshold:.4f} (ambiguous, rerank)"
        print(f"    [CIRCUIT_BREAKER] {skip_reason} -> {'skip' if skip_reranker else 'run'} reranker")
        
        def _best_window(q: str, text: str, window: int = 600, step: int = 200) -> str:
            if len(text) <= window:
                return text
            q_words = set(q.lower().split())
            best_score, best_start = -1, 0
            for start in range(0, len(text) - window, step):
                snippet = text[start:start + window]
                sc = sum(1 for w in q_words if w in snippet.lower())
                if sc > best_score:
                    best_score, best_start = sc, start
            return text[best_start:best_start + window]

        if not skip_reranker:
            pairs = [
                [query, _best_window(query, _get_attr(r, "text", "") or "")]
                for r in rerank_candidates
            ]
            if len(pairs) > RERANK_GUARD_MAX_PAIRS:
                first_pairs = pairs[:RERANK_GUARD_MAX_PAIRS]
                first_candidates = rerank_candidates[:RERANK_GUARD_MAX_PAIRS]
                first_scores = list(retriever.reranker.predict(first_pairs))
                elapsed_ms = (time.time() - t0) * 1000
                if elapsed_ms > RERANK_LATENCY_BUDGET_MS:
                    print(
                        f"  [RERANK_GUARD] elapsed={elapsed_ms:.1f}ms > "
                        f"{RERANK_LATENCY_BUDGET_MS:.0f}ms; truncating pairs "
                        f"{len(pairs)} -> {RERANK_GUARD_MAX_PAIRS}"
                    )
                    rerank_candidates = first_candidates
                    rerank_scores = first_scores
                else:
                    remaining_scores = list(retriever.reranker.predict(pairs[RERANK_GUARD_MAX_PAIRS:]))
                    rerank_scores = first_scores + remaining_scores
            else:
                rerank_scores = list(retriever.reranker.predict(pairs))

            for i, hit in enumerate(rerank_candidates):
                base_score  = float(rerank_scores[i])
                section     = _get_section(hit)
                delta       = SECTION_SCORE_DELTA.get(section, 0.0)
                final_score = base_score + delta
                if hasattr(hit, "score"):
                    hit.score = final_score
                else:
                    hit["score"] = final_score

            rerank_candidates = sorted(
                rerank_candidates,
                key=lambda x: _get_attr(x, "score", 0) or 0,
                reverse=True,
            )
        before_score_filter = len(rerank_candidates)
        rerank_candidates = [c for c in rerank_candidates if (_get_attr(c, "score", 0) or 0) >= 0.0]
        removed_score_filter = before_score_filter - len(rerank_candidates)
        if removed_score_filter:
            print(
                f"  [RERANK_FILTER] Dropped {removed_score_filter} chunk(s) "
                f"with cross-encoder score < 0.0"
            )

        timings["rerank_ms"] = round((time.time() - t0) * 1000, 2)
        print(f"  Reranked top-{top_k}: {timings['rerank_ms']}ms\n")

    # SELF-REFLECTION RETRY LOOP â†’ generate
    pre_rerank_chunks  = [_chunk_to_summary(c, i + 1) for i, c in enumerate(fused_results[:top_k * 2])]
    post_rerank_chunks = [_chunk_to_summary(c, i + 1) for i, c in enumerate(rerank_candidates[:top_k])]
    if allow_ref:
        print(f"  [REF_POLICY] Query '{query[:50]}... references ALLOWED (citation query)")
        filtered_chunks = [c for c in rerank_candidates if is_quality_chunk(c)]
        blocked_ids = set()
    else:
        print(f"  [REF_POLICY] Query '{query[:50]}... references FILTERED (non-citation query)")
        filtered_chunks = [
            c for c in rerank_candidates
            if is_valid_chunk(c) and is_quality_chunk(c)
        ]
        filtered_chunks, blocked_ids = limit_reference_chunks(
            filtered_chunks,
                max_ref=0,
            )
        
    # Context Budgeting System ” Hard Limit Reference Ratio
    MAX_REF_RATIO  = max_ref_ratio 
    quality_filtered = [c for c in filtered_chunks if is_quality_chunk(c)]

    final_chunks_full = enforce_reference_policy(
        quality_filtered,
        allow_ref=allow_ref,
        max_ref_ratio=MAX_REF_RATIO,
        top_k=top_k,
    )

    final_selected_ids = {_get_attr(c, "chunk_id", "") for c in final_chunks_full}
    additional_blocked = {
        _get_attr(c, "chunk_id", "")
        for c in quality_filtered
        if _get_section(c) in BLOCKED_SECTIONS and _get_attr(c, "chunk_id", "") not in final_selected_ids
    }
    blocked_ids.update(additional_blocked)

    n_allowed_ref = sum(1 for c in final_chunks_full if _get_section(c) in BLOCKED_SECTIONS)
    actual_ref_ratio = round(n_allowed_ref / max(len(final_chunks_full), 1), 3)
    print(
        f"  [CONTEXT] final={len(final_chunks_full)} chunks | "
        f"ref_in_context={n_allowed_ref} | "
        f"ref_ratio={actual_ref_ratio} | "
        f"blocked={len(blocked_ids)}\n"
    )
    max_ref_budget = int(top_k * MAX_REF_RATIO) if allow_ref else 0
    confidence_engine = ConfidenceEngine()
    print("[Generation] Self-reflection retry loop...")
    retry_count = 0
    allow_all_refs_on_retry = False
    t0_gen = time.time()

    while retry_count < MAX_RETRY_ATTEMPTS:
        if timings.get("generation_ms", 0) >= MAX_GENERATION_MS:
            print("  [GEN_GUARD] generation exceeded soft cap; forcing abstain")
            break

        final_context = build_context(final_chunks_full, max_chars=3000)

        if len(final_context) > 2500:
            final_context = final_context[:2500]

        print(f"  [GEN CONFIG] max_tokens={gen_max_tokens} | temperature={gen_temperature} | context_len={len(final_context)}")
        pre_gen_confidence = confidence_engine.calculate_confidence(final_chunks_full)
        retrieval_confidence = pre_gen_confidence
        if pre_gen_confidence["decision"] == "REJECT":
            print(f" [PRE_GEN_GATE] Confidence REJECT ({pre_gen_confidence['confidence_score']:.3f}) skip LLM call")
            from src.generation.generator import _make_abstain_response
            response = _make_abstain_response(
                query=query,
                chunks=final_chunks_full,
                model="llama-3.3-70b-versatile",
                reason="pre_gen confidence REJECT", 
            )
            break 

        try:
            response = generator.generate(
                query,
                final_chunks_full,
                max_tokens=gen_max_tokens,
                temperature=gen_temperature,
                source_chunk_id=source_chunk_id,
                min_top1_score=RETRIEVAL_MIN_TOP1_SCORE,
            )
        except Exception as e:
            print(f" [GENERATION_ERROR] {type(e).__name__}: {e} - aborting to abstain")
            from src.generation.generator import _make_abstain_response
            response = _make_abstain_response(
                query=query,
                chunks=final_chunks_full,
                model="llama-3.3-70b-versatile",
                reason=f"generation_exception: {type(e).__name__}: {e}",
            )
            break
        context_texts = [_get_attr(r, "text", "") or "" for r in final_chunks_full]
        confidence_v1 = compute_context_overlap(response.answer, context_texts)
        retrieval_confidence = confidence_engine.calculate_confidence(final_chunks_full)

        if confidence_v1 >= 0.3:
            print(f"  [RETRY] Success at attempt {retry_count + 1} | overlap={confidence_v1}")
            break

        if retry_count >= 1:
            print("  [RETRY] Max retry attempts reached; stopping retry loop")
            break

        if response.status == "INSUFFICIENT_CONTEXT" and q_type == "factual" and retry_count == 0:
            print("[RETRY] Factual insufficient context immediate reference escalation")
            allow_all_refs_on_retry = True

        print(f"[RETRY] Low context overlap ({confidence_v1}) â†’ retrying retrieval...")

        candidate_k_retry = top_k * (2 ** retry_count)
        _use_reranker_retry = retry_count < 1          
        _bm25_weight_retry  = 0.3 + (0.2 * retry_count) 
        _bm25_weight_retry  = min(_bm25_weight_retry, 0.7)
        _dense_weight_retry = 1.0 - _bm25_weight_retry

        if allow_all_refs_on_retry:
            candidate_k_retry = min(top_k * 4, 30)
            _use_reranker_retry = False
            _bm25_weight_retry = 0.6
            _dense_weight_retry = 0.4
            print("  [RETRY] Factual escalation: allow references, stronger BM25 signal")

        print(
            f"  [RETRY] attempt={retry_count + 1} | candidate_k={candidate_k_retry} | "
            f"reranker={_use_reranker_retry} | bm25_w={_bm25_weight_retry:.1f}"
        )

        try:
            fused_results = list(retriever.search(
                query,
                top_k=candidate_k_retry,
                bm25_weight=_bm25_weight_retry,
                dense_weight=_dense_weight_retry,
            ))
            
        except Exception as e:
            print(f"  [RETRY] Fallback dense-only (retriever.search() error: {type(e).__name__}: {e})")
            try:
                fused_results = list(retriever.vector_store.search(query, k=candidate_k_retry))
            except Exception as e2:
                print(f"  [RETRY] Dense-only fallback ALSO failed ({type(e2).__name__}: {e2}) — aborting to abstain")
                from src.generation.generator import _make_abstain_response
                response = _make_abstain_response(
                    query=query,
                    chunks=final_chunks_full,
                    model="llama-3.3-70b-versatile",
                    reason=f"retrieval_exception: {type(e2).__name__}: {e2}",
                )
                break

        # allow all references (don't limit ” last resort for factual QA)
        if allow_all_refs_on_retry:
            final_chunks_full = [c for c in fused_results[:top_k] if is_quality_chunk(c)]
            retry_count = MAX_RETRIES
        elif retry_count >= 2:
            print("  [RETRY] Escalation: references unrestricted (last resort)")
            final_chunks_full = [c for c in fused_results[:top_k] if is_quality_chunk(c)]
        else:
            retry_filtered, _ = limit_reference_chunks(fused_results[:top_k], max_ref=MAX_REF_IN_CONTEXT)
            final_chunks_full = [c for c in retry_filtered if is_quality_chunk(c)]
        print(f"  [RETRY] usable after filter={len(final_chunks_full)}")
        retry_count += 1

    timings["generation_ms"] = round((time.time() - t0_gen) * 1000, 2)
    print(f"  Final generation: {timings['generation_ms']}ms | retries={retry_count}\n")
    final_status = response.status

    # Enforce retrieval confidence safety
    if retrieval_confidence["decision"] == "REJECT" and final_status == "ANSWERED":
        print("  [CONF_ENGINE] Retrieval confidence REJECT -> abstaining to prevent hallucination")
        final_status = "INSUFFICIENT_CONTEXT"
        response.answer = "INSUFFICIENT_CONTEXT"
        response.supporting_sources = []

    context_texts = [_get_attr(r, "text", "") or "" for r in final_chunks_full]
    confidence_v1 = compute_context_overlap(response.answer, context_texts)

    # FAILURE-AWARE ADAPTATION LOGIC 
    error_info = classify_error(
    retrieved_chunks=final_chunks_full,
    reranked_chunks=rerank_candidates[:top_k],
    source_chunk_id=source_chunk_id,
    context_overlap=confidence_v1,
    blocked_chunk_ids=blocked_ids,
)

    # Adapt based on failure type
    if error_info["failure_type"] == "retrieval_miss":
        print("[ADAPT] Retrieval miss â†’ future runs shift to BM25-heavy")
    elif error_info["failure_type"] == "filtering_error":
        print("[ADAPT] Filtering error detected â†’ consider increasing MAX_REF_IN_CONTEXT")
    elif error_info["failure_type"] == "ranking_error":
        print("[ADAPT] Ranking error â†’ reranker mis-ordered, consider section boost tuning")

    # Confidence calibration ” penalize when pipeline has known failures
    raw_confidence = confidence_v1
    failure_type   = error_info["failure_type"]
    if failure_type == "retrieval_miss":
        confidence_v1 = round(confidence_v1 * 0.4, 4)   
    elif failure_type == "filtering_error":
        confidence_v1 = round(confidence_v1 * 0.6, 4)  
    elif failure_type == "ranking_error":
        confidence_v1 = round(confidence_v1 * 0.7, 4)   
    elif failure_type == "generation_fail":
        confidence_v1 = round(confidence_v1 * 0.5, 4)

    if raw_confidence != confidence_v1:
        print(
            f"  [CONFIDENCE] Calibrated: {raw_confidence} â†’ {confidence_v1} "
            f"(penalty for {failure_type})"
        )

    timings["total_ms"] = round(
        timings.get("query_expansion_ms", 0) +
        timings.get("embedding_ms", 0) +
        timings.get("bm25_ms", 0) +
        timings.get("rerank_ms", 0) +
        timings.get("generation_ms", 0),
        2,
    )
    # debug_full_chunks ” full score transparency
    debug_source      = final_chunks_full
    debug_full_chunks = [
        {
            "chunk_id": _get_attr(r, "chunk_id", ""),
            "doc_id":   _get_attr(r, "doc_id", ""),
            "section":  _get_section(r),
            "score":    round(
                        _get_attr(r, "rerank_score", None) or
                        _get_attr(r, "retrieval_score", None) or
                        _get_attr(r, "score", 0) or 0,
                        4
                        ),
            "length":   len(_get_attr(r, "text", "") or ""),
            "text":     (_get_attr(r, "text", "") or "")[:300],
        }
        for r in debug_source
    ]

    # Log Ref Ratio ” metric utama untuk monitor reference leakage
    ref_count   = sum(1 for d in debug_full_chunks if d["section"] in BLOCKED_SECTIONS)
    total_count = len(debug_full_chunks)
    ref_ratio   = round(ref_count / total_count, 3) if total_count > 0 else 0.0
    print(f"\n  REF RATIO        : {ref_ratio} ({ref_count}/{total_count})")
    if ref_ratio >= 0.2 and not allow_ref:
        print(f"  [REF_WARN] ref_ratio={ref_ratio} melebihi target 0.2 ” pertimbangkan penalty -3.0")

    # Exact Match Metric
    exact_match = None
    if ground_truth:
        exact_match = response.answer.strip().lower() == ground_truth.strip().lower()

    # Assemble output
    output = {
        "query":                query,
        "query_id":             str(uuid.uuid4()),
        "mode":                 mode,
        "query_type":           q_type,
        "expanded_queries":     expanded_queries if mode == "full" else [query],
        "hyde_doc":             hyde_doc,
        "pre_rerank_chunks":    pre_rerank_chunks,
        "reranked_chunks":      post_rerank_chunks,
        "retrieval_scores":     [c.get("score") if isinstance(c, dict) else getattr(c, "score", None) for c in (post_rerank_chunks or [])],
        "final_context":        final_context[:500] + "..." if len(final_context) > 500 else final_context,
        "answer":               response.answer,
        "status":               final_status,
        "confidence_score":     response.confidence_score,
        "retrieval_confidence": retrieval_confidence,
        "supporting_sources":   response.supporting_sources,
        "ground_truth":         ground_truth,
        "latency_breakdown":    timings,
        "confidence_v1":        confidence_v1,
        "error_decomposition":  error_info,
        "failure_type":         error_info.get("failure_type", "none") if isinstance(error_info, dict) else "none",
        "retry_triggered":      retry_count > 0,
        "exact_match":          exact_match,
        "ref_ratio":            ref_ratio,
        "retrieval_method":     (
            "dense_only" if mode == "baseline"
            else f"adaptive_agent_{q_type}_multiquery_hyde_rerank_rrf"
        ),
        "debug_full_chunks":    debug_full_chunks,
        "config": {
            "mode":             mode,
            "top_k":            top_k,
            "use_reranker":     use_reranker and mode == "full",
            "query_type":       q_type,
            "use_multi_query":  use_multi_query,
            "use_hyde":         use_hyde,
            "min_chunk_length": MIN_CHUNK_LENGTH,
            "reranker":         "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "generator":        "llama-3.3-70b-versatile",
            "embedding":        "all-MiniLM-L6-v2",
            "chunking":         "semantic",
            "rrf_k":            60,
            "retries_used":     retry_count,
            "candidate_k":      settings.candidate_k,
            "max_tokens":       gen_max_tokens,
            "temperature":      gen_temperature,
            "domain_filter":    True,
            "ref_penalty":      SECTION_SCORE_DELTA.get("references", -2.0),
            "ref_filter_mode":  "citation_aware" if allow_ref else "strict",
            "max_ref_ratio":    MAX_REF_RATIO,
            "max_ref_budget":   max_ref_budget,
        },
    }

    # Print summary
    print(f"{'='*60}")
    print(f"  MODE          : {mode.upper()}")
    print(f"  QUERY_TYPE    : {q_type}")
    print(f"  ANSWER        : {response.answer[:300]}")
    print(f"\n  STATUS (final): {final_status}")
    print(f"  CONFIDENCE (LLM) : {response.confidence_score:.2f}")
    print(f"  CONFIDENCE OVERLAP    : {confidence_v1}")
    print(f"  ERROR TYPE       : {error_info['failure_type']}")
    print(f"  REASON           : {error_info['reason']}")
    print(f"  RETRIES          : {retry_count}")

    # Print exact match real metric
    if exact_match is not None:
        print(f"  EXACT MATCH      : {exact_match}")
    print(f"\n  DEBUG CHUNKS (section | score | length):")
    for d in debug_full_chunks:
        flag = " BLOCKED" if d["section"] in BLOCKED_SECTIONS else ""
        short_flag = " SHORT "  if d["length"] < MIN_CHUNK_LENGTH else ""
        print(
            f"    [{d['chunk_id']}] "
            f"section={d['section']:<15} "
            f"score={d['score']:.4f} "
            f"len={d['length']}"
            f"{flag}{short_flag}"
        )

    print(f"\n  LATENCY BREAKDOWN:")
    for k, v in timings.items():
        bar = "|" * int(v / 100)
        print(f"    {k:<25} {v:>8.1f}ms  {bar}")
        
    def to_serializable(obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    if save_output:
        out_path = Path(f"results/logs/single_query_{mode}.json")
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
            json.dump(existing, f, indent=2, ensure_ascii=False, default=to_serializable)
        print(f"[Saved]  {out_path}")

    # dump trace for query-level debugging and audit
    trace_path = Path("results/debug/trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_entry = {
        "query":           query,
        "query_type":      q_type,
        "top_k":           top_k,
        "allow_ref":       allow_ref,
        "max_ref_ratio":   max_ref_ratio,
        "final_ref_ratio": ref_ratio,
        "retries":         retry_count,
        "error":           error_info,
        "status":          final_status,
        "exact_match":     exact_match,
        "confidence_v1":   confidence_v1,
        "final_chunks":   debug_full_chunks,
        "timings":         timings,
    }
    existing_trace = []
    if trace_path.exists():
        with open(trace_path, "r") as f:
            try:
                existing_trace = json.load(f)
            except Exception:
                existing_trace = []
    existing_trace.append(trace_entry)
    with open(trace_path, "w") as f:
        json.dump(existing_trace, f, indent=2, ensure_ascii=False)
    print(f"[Trace] {trace_path}")
    return output

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RAG Proto-Agent Debugger")
    parser.add_argument("--query",       type=str,  default=None)
    parser.add_argument("--top_k",       type=int,  default=5)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--gt",          type=str,  default=None)
    parser.add_argument("--source",      type=str,  default=None)
    parser.add_argument("--from-gt",     action="store_true")
    parser.add_argument("--agent",       action="store_true", help="Use the new agentic loop instead of the legacy pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "full"],
        default="full",
        help="baseline=dense-only | full=adaptive proto-agent",
    )
    args = parser.parse_args()

    if args.agent:
        from src.controller.agent import Agent

        query = args.query or "What is attention mechanism in transformer models?"
        agent = Agent()
        response = asyncio.run(agent.run(query, source_chunk_id=args.source))
        print(f"\n[AGENT] status={response.status.value} | steps={response.state.step_count}")
        print(f"[AGENT] confidence={response.state.confidence_score:.4f}")
        print(f"[AGENT] answer:\n{response.answer}\n")
    elif args.from_gt:
        with open("data/processed/ground_truth_qa.json") as f:
            gt_data = json.load(f)
        for item in gt_data[:3]:
            run_single_query(
                query=item["question"],
                top_k=args.top_k,
                use_reranker=not args.no_reranker,
                ground_truth=item.get("ground_truth"),
                source_chunk_id=item.get("source_chunk"),
                mode=args.mode,
            )
    else:
        query = args.query or "What is attention mechanism in transformer models?"
        run_single_query(
            query=query,
            top_k=args.top_k,
            use_reranker=not args.no_reranker,
            ground_truth=args.gt,
            source_chunk_id=args.source,
            mode=args.mode,
        )

# Singleton cache
_STORE:      QdrantVectorStore | None      = None
_RETRIEVER:  MasterHybridRetriever | None = None
_GENERATOR:  GroqGenerator | None          = None

def get_components():
    global _STORE, _RETRIEVER, _GENERATOR
    if _STORE is None:
        _STORE      = QdrantVectorStore()
        _RETRIEVER  = MasterHybridRetriever(vector_store=_STORE)
        _GENERATOR  = GroqGenerator(model="llama-3.3-70b-versatile")
        print("[Cache] Components loaded reuse for subsequent queries")
    return _STORE, _RETRIEVER, _GENERATOR