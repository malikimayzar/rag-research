from __future__ import annotations
from src.api.config import settings

import logging
import json
import os
import numpy as np
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.retrieval.qdrant_store import QdrantVectorStore, RetrievalResult
from src.controller.policy_engine import PolicyEngine
from rank_bm25 import BM25Okapi
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from dotenv import load_dotenv

logger = logging.getLogger("rag.retriever")
load_dotenv()

def rrf_fuse(
    all_hits: List[List],
    k: int = 60,
    weights: Optional[List[float]] = None,
    section_boost_fn=None,
) -> List:
    if weights is not None and len(weights) != len(all_hits):
        raise ValueError(f"weights length ({len(weights)}) != all_hits length ({len(all_hits)})")

    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, Any] = {}

    for list_idx, hits in enumerate(all_hits):
        w = weights[list_idx] if weights is not None else 1.0
        for rank, hit in enumerate(hits):
            cid = hit.get("chunk_id") if isinstance(hit, dict) else getattr(hit, "chunk_id", None)
            if not cid:
                continue
            boost = 1.0
            if section_boost_fn is not None:
                meta = (hit.get("metadata", {}) if isinstance(hit, dict)
                        else getattr(hit, "metadata", {})) or {}
                boost = section_boost_fn(meta)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + w * boost / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = hit

    result = []
    for cid, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        item = chunk_map[cid]
        rrf_score = round(score, 6)
        if isinstance(item, dict):
            item["retrieval_score"] = rrf_score
        elif hasattr(item, "score"):
            item.score = rrf_score
        result.append(item)

    return result


MULTI_QUERY_PROMPT = """Generate 2 alternative search queries for the following question.

STRICT RULES:
- Stay in the SAME technical domain as the original query
- Do NOT switch fields (e.g., from machine learning to neuroscience)
- Preserve key technical terms (e.g., "transformer", "attention")
- Rephrase, don't reinterpret

Original query: {query}

Alternative queries:"""

HYDE_PROMPT = """Write a short hypothetical passage (2-3 sentences) that would directly answer this question.
Be specific and technical. Write as if you are an expert answering from a research paper.

Question: {query}

Hypothetical passage:"""

class MasterHybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_chunks_path: str = "data/processed/chunks_semantic.json",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
        use_multi_query: bool = False,
        use_hyde: bool = False,
    ):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.use_multi_query = use_multi_query
        self.use_hyde = use_hyde
        self._policy = PolicyEngine()
        self._chunks = []
        self._bm25 = None

        if Path(bm25_chunks_path).exists():
            with open(bm25_chunks_path, "r") as f:
                self._chunks = json.load(f)
            tokenized_corpus = [c["text"].lower().split() for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"[MasterRetriever] BM25 loaded: {len(self._chunks)} chunks")

        from src.retrieval.model_registry import ModelRegistry
        self.reranker = ModelRegistry.get().reranker
        logger.info(f"[MasterRetriever] Reranker loaded (disabled in baseline)")

        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq = Groq(api_key=api_key)
            logger.info(f"[MasterRetriever] Groq available (expansion disabled)")
        else:
            self.groq = None
            logger.info(f"[MasterRetriever] No GROQ_API_KEY available")

    def _is_valid_section(self, metadata: dict) -> bool:
        section = str(metadata.get("section", "")).lower()
        return section not in ["references", "bibliography"]
    def _is_reference_chunk(self, chunk) -> bool:
        """
        Content-aware reference detection — catches reference chunks that
        slipped through with wrong/missing section metadata.

        Layer 1: section name
        Layer 2: citation bracket pattern [1], [23], etc.
        Layer 3: comma density (citation lists like "Smith, J., Jones, K., ...")
        """
        if isinstance(chunk, dict):
            meta = chunk.get("metadata", {})
            text = chunk.get("text", "")
        else:
            meta = getattr(chunk, "metadata", {})
            text = getattr(chunk, "text", "")

        section = str(meta.get("section", "")).lower()

        # Layer 1: section name check
        if any(ref in section for ref in ["reference", "bibliography"]):
            return True

        # Layer 2: citation bracket pattern [1], [23]
        if re.search(r"\[\d+\]", text):
            return True

        # Layer 3: comma density (citation lists)
        if text.count(",") > 10:
            return True

        return False
    
    @staticmethod
    def _section_boost(metadata: dict) -> float:
        section = str(metadata.get("section", "")).lower()

        if any(kw in section for kw in ("method", "architect", "approach", "model")):
            return 1.4   
        elif any(kw in section for kw in ("experiment", "result", "evaluat", "finding")):
            return 1.3   # high: results are the evidence
        elif "abstract" in section:
            return 1.1   # medium: abstract is dense but broad
        elif "introduction" in section:
            return 1.05  # slight: intro has context
        elif "conclusion" in section:
            return 1.0   # neutral
        elif any(ref in section for ref in ("reference", "bibliography")):
            return 0.3   # soft penalty — survives if no better match exists
        else:
            return 1.0   # body/default

    # ── Query Expansion ────────────────────────────────────────────────────────
    def _call_groq(self, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self.groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _expand_and_hyde_parallel(self, query: str):
        queries = [query]
        hyde_text = None

        if not self.groq:
            return queries, hyde_text

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            if self.use_multi_query:
                futures["mq"] = executor.submit(
                    self._call_groq,
                    MULTI_QUERY_PROMPT.format(query=query), 0.7, 100
                )
            if self.use_hyde:
                futures["hyde"] = executor.submit(
                    self._call_groq,
                    HYDE_PROMPT.format(query=query), 0.5, 150
                )

        if "mq" in futures:
            try:
                raw = futures["mq"].result(timeout=5)
                alternatives = [q.strip() for q in raw.split("\n") if q.strip()][:2]
                queries = [query] + alternatives
                logger.info(f"[MultiQuery] Expanded to {len(queries)} queries: {alternatives}")
            except Exception as e:
                logger.info(f"[MultiQuery] Failed: {e}")

        if "hyde" in futures:
            try:
                hyde_text = futures["hyde"].result(timeout=5)
                logger.info(f"[HyDE] Generated: {hyde_text[:80]}...")
            except Exception as e:
                logger.info(f"[HyDE] Failed: {e}")

        return queries, hyde_text

    def _expand_queries(self, query: str) -> List[str]:
        return [query]

    def _generate_hyde(self, query: str) -> Optional[str]:
        return None

    def _classify_query(self, query: str) -> int:
        words = len(query.strip().split())
        if words <= 5:
            return 3
        if words >= 15:
            return 7
        return 5

    def _resolve_top_k(self, query: str, top_k: int) -> int:
        if top_k <= 0:
            top_k = self._classify_query(query)
            logger.info(f"[TopK] auto-selected top_k={top_k} for query length={len(query.split())}")
        return top_k

    def _choose_candidate_k(self, top_k: int) -> int:
        return min(top_k * 3, 30)

    def _rerank(self, query: str, candidates: list) -> list:
        if not self.reranker or not candidates:
            return candidates

        pairs = []
        for hit in candidates:
            text = hit.get("text") if isinstance(hit, dict) else getattr(hit, "text", "")
            pairs.append([query, text])

        try:
            scores = self.reranker.predict(pairs)
            # STEP 2: Attach rerank_score to each chunk
            for chunk, score in zip(candidates, scores):
                rerank_score = float(score)
                if isinstance(chunk, dict):
                    chunk["rerank_score"] = rerank_score
                else:
                    chunk.rerank_score = rerank_score
            
            # STEP 3: Sort by rerank_score
            ranked = sorted(
                candidates,
                key=lambda x: (x["rerank_score"] if isinstance(x, dict) else getattr(x, "rerank_score", 0)),
                reverse=True,
            )
            
            return ranked
        except Exception as exc:
            logger.info(f"[Rerank] failed: {exc}")
            return candidates

    # ── Core Retrieval ─────────────────────────────────────────────────────────
    def _dense_search(self, query: str, k: int) -> List[RetrievalResult]:
        return self.vector_store.search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            {
                "chunk_id": self._chunks[i]["chunk_id"],
                "text": self._chunks[i]["text"],
                "score": float(scores[i]),
                "doc_id": self._chunks[i]["doc_id"],
                "metadata": self._chunks[i].get("metadata", {}),
            }
            for i in top_indices
        ]
    def _rrf_fuse(
        self,
        all_dense_hits: List[List],
        all_bm25_hits: List[List],
        candidate_k: int,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List:
        """Thin wrapper around standalone rrf_fuse."""
        all_hits = all_dense_hits + all_bm25_hits
        weights  = [dense_weight] * len(all_dense_hits) + [bm25_weight] * len(all_bm25_hits)

        results = rrf_fuse(
            all_hits=all_hits,
            k=self.rrf_k,
            weights=weights,
            section_boost_fn=self._section_boost,
        )[:candidate_k]

        ref_penalized = sum(
            1 for c in results
            if self._section_boost(
                (c.get("metadata", {}) if isinstance(c, dict)
                 else getattr(c, "metadata", {})) or {}
            ) == 0.3
        )
        logger.info(
            f"[RRF_FILTER] candidates={len(results)} | "
            f"refs_penalized={ref_penalized} (boost=0.3, not blocked)"
        )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        t_search_start = time.time()
        top_k = self._resolve_top_k(query, top_k)
        plan = self._policy.initial_plan(query)
        self.use_multi_query = plan.allow_multi_query
        self.use_hyde = plan.allow_hyde
        logger.info(
            f"[POLICY] query_type={plan.query_type} | "
            f"multi_query={self.use_multi_query} | hyde={self.use_hyde}"
        )
        candidate_k = max(top_k, settings.candidate_k)  # 40–50 candidates for reranker

        all_dense_hits = []
        all_bm25_hits = []

        t_dense_start = time.time()
        all_dense_hits.append(self._dense_search(query, k=candidate_k))
        t_dense = time.time() - t_dense_start

        t_bm25_start = time.time()
        all_bm25_hits.append(self._bm25_search(query, k=candidate_k))
        t_bm25 = time.time() - t_bm25_start

        t_fuse_start = time.time()
        fused_results = self._rrf_fuse(all_dense_hits, all_bm25_hits, candidate_k, dense_weight, bm25_weight)

        # STEP 3: Filter low-quality chunks
        def is_low_quality(chunk):
            text = chunk["text"].strip() if isinstance(chunk, dict) else getattr(chunk, "text", "").strip()
            if len(text) < 20:
                return True
            if text.count(",") > 10 and len(text.split()) < 15:
                return True
            return False
        candidates = [c for c in fused_results if not is_low_quality(c)]
        # STEP 1: Always rerank
        reranked = self._rerank(query, candidates)

        # STEP 2: Logging AFTER rerank

        # STEP 5: QUALITY-BASED SELECTION (relevance + informativeness)
        # Filter 1: rerank_score threshold
        if reranked:
            scores = [
                c.get("rerank_score") if isinstance(c, dict) else getattr(c, "rerank_score", 0)
                for c in reranked
            ]
            scores_sorted = sorted(scores, reverse=True)
            gaps = [scores_sorted[i] - scores_sorted[i+1] for i in range(min(9, len(scores_sorted)-1))]
            dynamic_threshold = scores_sorted[min(top_k, len(scores_sorted) - 1)]
        else:
            dynamic_threshold = 0.0
    
        # Filter 2: text informativeness (must be substantial)
        def is_informative(chunk):
            text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", "")
            word_count = len(text.strip().split())
            return word_count > 20  # Substantial content (not just titles/headers)
    
        def safe_score(c):
            score = c.get("rerank_score") if isinstance(c, dict) else getattr(c, "rerank_score", None)
            return score if score is not None else float('-inf')

        filtered_chunks = [
            c for c in reranked
            if safe_score(c) > dynamic_threshold
            and is_informative(c)
        ]
        
        # Selection logic: prefer quality over quantity
        if filtered_chunks:
            final_chunks = filtered_chunks[:top_k]  # Take top-k from quality-filtered results
        else:
            final_chunks = []
        
        layer1 = [c for c in reranked if safe_score(c) > dynamic_threshold]
        layer2 = filtered_chunks  

        print("\n=== QUALITY-BASED SELECTION (3-LAYER FILTERING) ===")
        print(f"After rerank: {len(reranked)} chunks")
        print(f"Layer 1 (relevance > 0.0): {len(layer1)} chunks")
        print(f"Layer 2 (+ informative):   {len(layer2)} chunks ✓ FINAL")
        print(f"Selected: {len(final_chunks)} chunks")
        for c in final_chunks:
            cid = c["chunk_id"] if isinstance(c, dict) else getattr(c, "chunk_id", None)
            rerank_score = c.get("rerank_score") if isinstance(c, dict) else getattr(c, "rerank_score", None)
            text = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
            text_len = len(text.split())
            starts_ok = not text.strip().lower().startswith(("and ", "or ", "but "))
            ends_ok = text.strip().endswith((".", ":", ";", "?", ")"))
            print(f"  {cid}: score={rerank_score:.2f}, words={text_len}, complete=({starts_ok}&{ends_ok})")

        formatted = self._format_output(final_chunks, "hybrid_rrf_rerank")

        t_fuse = time.time() - t_fuse_start
        t_search_total = time.time() - t_search_start

        # Observability: reference contamination check on raw hits
        raw_hits = (all_dense_hits[0] if all_dense_hits else []) + (all_bm25_hits[0] if all_bm25_hits else [])
        ref_count = sum(1 for hit in raw_hits if self._is_reference_chunk(hit))

        logger.info(
            f"[RETRIEVAL_LATENCY] query='{query[:50]}' | "
            f"dense_ms={t_dense*1000:.1f} | bm25_ms={t_bm25*1000:.1f} | "
            f"fuse_ms={t_fuse*1000:.1f} | total_ms={t_search_total*1000:.1f} | "
            f"raw_refs={ref_count} | final={len(formatted)}"
        )

        return formatted

    def _format_output(self, results: list, method: str) -> List[Dict[str, Any]]:
        formatted = []
        for i, r in enumerate(results):
            if isinstance(r, dict):
                cid  = r.get("chunk_id")
                txt  = r.get("text")
                did  = r.get("doc_id")
                rerank_score = r.get("rerank_score") if isinstance(r, dict) else getattr(r, "rerank_score", None)
                rrf_score    = r.get("score", 0) if isinstance(r, dict) else getattr(r, "score", 0)
                meta = r.get("metadata", {})
            else:
                cid  = getattr(r, "chunk_id", "unknown")
                txt  = getattr(r, "text", "")
                did  = getattr(r, "doc_id", "unknown")
                rerank_score = getattr(r, "rerank_score", None)
                rrf_score    = getattr(r, "score", 0)
                meta = getattr(r, "metadata", {})

            formatted.append({
                "chunk_id":         cid,
                "text":             txt,
                "doc_id":           did,
                "rerank_score":     float(rerank_score) if rerank_score is not None else None,
                "retrieval_score":  float(rrf_score),
                "retrieval_rank":   i + 1,
                "retrieval_method": method,
                "metadata":         meta,
            })
        return formatted


# --- CLI TEST ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    store     = QdrantVectorStore()
    retriever = MasterHybridRetriever(store)

    test_queries = [
        "What is Retrieval-Augmented Generation?",
        "How does semantic chunking work?",
        "What are attention mechanisms?",
    ]

    logger.info("\n" + "=" * 80)
    logger.info("[PRODUCTION TEST] Running with TIERED SECTION BOOST + HARD REFERENCE FILTERING")
    logger.info("Target: references=0 in top results | methods/experiments ranked highest")
    logger.info("=" * 80)

    results_all = []

    for query in test_queries:
        t_start = time.time()
        results = retriever.search(query, top_k=10)
        latency_ms = round((time.time() - t_start) * 1000, 2)

        ref_in_top = sum(1 for r in results[:3] if retriever._is_reference_chunk(r))

        logger.info(f"\n[QUERY] {query}")
        logger.info(f"[TOTAL_LATENCY] {latency_ms}ms")
        logger.info(f"[REF_CHECK] references in top-3: {ref_in_top} (target: 0)")

        for r in results[:3]:
            section = (r.get("metadata") or {}).get("section", "unknown")
            boost   = MasterHybridRetriever._section_boost(r.get("metadata") or {})
            logger.info(
                f"  Rank {r['retrieval_rank']}: score={r['retrieval_score']:.4f} | "
                f"section={section} | boost={boost} | doc={r['doc_id']}"
            )
            logger.info(f"    Text: {r['text'][:100]}...")

        results_all.append({
            "query":       query,
            "latency_ms":  latency_ms,
            "top_chunks":  results[:3],
            "num_results": len(results),
            "refs_in_top3": ref_in_top,
        })
    logger.info("\n" + "=" * 80)
    avg_latency  = sum(r["latency_ms"] for r in results_all) / len(results_all)
    total_refs   = sum(r["refs_in_top3"] for r in results_all)
    logger.info(f"[SUMMARY] queries={len(results_all)} | avg_latency={avg_latency:.1f}ms | refs_in_top3={total_refs}")
    logger.info("=" * 80 + "\n")
    store.close()