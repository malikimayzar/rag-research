from __future__ import annotations

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.retrieval.qdrant_store import QdrantVectorStore, RetrievalResult
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MULTI_QUERY_PROMPT = """Generate 2 alternative search queries for the following question.
These should capture different aspects or phrasings of the same information need.
Return ONLY the queries, one per line, no numbering, no explanation.

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
        use_multi_query: bool = True,
        use_hyde: bool = True,
    ):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.use_multi_query = use_multi_query
        self.use_hyde = use_hyde
        self._chunks = []
        self._bm25 = None

        if Path(bm25_chunks_path).exists():
            with open(bm25_chunks_path, "r") as f:
                self._chunks = json.load(f)
            tokenized_corpus = [c["text"].lower().split() for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized_corpus)
            print(f"[MasterRetriever] BM25 loaded: {len(self._chunks)} chunks")

        print(f"[MasterRetriever] Loading Re-ranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)

        # Groq client untuk multi-query + HyDE
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.groq = Groq(api_key=api_key)
            print(f"[MasterRetriever] Groq ready (multi-query + HyDE)")
        else:
            self.groq = None
            print(f"[MasterRetriever] WARNING: No GROQ_API_KEY, multi-query + HyDE disabled")

    # ── Query Expansion ────────────────────────────────────────────────────────

    def _expand_queries(self, query: str) -> List[str]:
        """Generate 2 alternative queries via Groq. Returns [original] if fails."""
        if not self.groq or not self.use_multi_query:
            return [query]
        try:
            resp = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",  # pakai 8b biar cepat
                messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)}],
                temperature=0.7,
                max_tokens=100,
            )
            raw = resp.choices[0].message.content.strip()
            alternatives = [q.strip() for q in raw.split("\n") if q.strip()][:2]
            all_queries = [query] + alternatives
            print(f"[MultiQuery] Expanded to {len(all_queries)} queries: {alternatives}")
            return all_queries
        except Exception as e:
            print(f"[MultiQuery] Failed, using original: {e}")
            return [query]

    def _generate_hyde(self, query: str) -> Optional[str]:
        """Generate hypothetical answer for HyDE dense retrieval."""
        if not self.groq or not self.use_hyde:
            return None
        try:
            resp = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
                temperature=0.5,
                max_tokens=150,
            )
            hyde_text = resp.choices[0].message.content.strip()
            print(f"[HyDE] Generated: {hyde_text[:80]}...")
            return hyde_text
        except Exception as e:
            print(f"[HyDE] Failed: {e}")
            return None

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
                "doc_id": self._chunks[i]["doc_id"]
            }
            for i in top_indices if scores[i] > 0
        ]

    def _rrf_fuse(
        self,
        all_dense_hits: List[List],
        all_bm25_hits: List[List],
        candidate_k: int,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> List:
        """RRF fusion across multiple query results."""
        fused_scores = {}
        chunk_map = {}

        # Fuse semua dense hits dari semua queries
        for hits in all_dense_hits:
            for rank, hit in enumerate(hits):
                cid = hit.chunk_id
                fused_scores[cid] = fused_scores.get(cid, 0) + dense_weight / (self.rrf_k + rank + 1)
                chunk_map[cid] = hit

        # Fuse semua BM25 hits dari semua queries
        for hits in all_bm25_hits:
            for rank, hit in enumerate(hits):
                cid = hit["chunk_id"]
                fused_scores[cid] = fused_scores.get(cid, 0) + bm25_weight / (self.rrf_k + rank + 1)
                if cid not in chunk_map:
                    chunk_map[cid] = hit

        fused_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
        return [chunk_map[cid] for cid, _ in fused_ids]

    # ── Main Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_reranker: bool = True,
    ) -> List[Dict[str, Any]]:

        candidate_k = min(top_k * 4, 20)

        # Step 1: Query expansion (Multi-Query)
        queries = self._expand_queries(query)

        # Step 2: HyDE — tambah hypothetical answer sebagai query dense tambahan
        hyde_text = self._generate_hyde(query)

        # Step 3: Retrieve untuk setiap query
        all_dense_hits = []
        all_bm25_hits = []

        for q in queries:
            all_dense_hits.append(self._dense_search(q, k=candidate_k))
            all_bm25_hits.append(self._bm25_search(q, k=candidate_k))

        # Step 4: HyDE dense retrieval (pakai hypothetical text untuk embed)
        if hyde_text:
            hyde_hits = self._dense_search(hyde_text, k=candidate_k)
            all_dense_hits.append(hyde_hits)

        # Step 5: RRF fusion semua results
        fused_results = self._rrf_fuse(
            all_dense_hits, all_bm25_hits, candidate_k, dense_weight, bm25_weight
        )

        print(f"[DEBUG] Queries: {len(queries)} | HyDE: {'yes' if hyde_text else 'no'}")
        print(f"[DEBUG] Dense sources: {len(all_dense_hits)} | Fused candidates: {len(fused_results)}")

        # Step 6: Reranker pada original query (bukan expanded)
        if not use_reranker or not fused_results:
            return self._format_output(fused_results[:top_k], "hybrid_rrf_multiquery")

        rerank_candidates = fused_results[:5]
        pairs = [
            [query, getattr(hit, 'text') if hasattr(hit, 'text') else hit['text']]
            for hit in rerank_candidates
        ]
        rerank_scores = self.reranker.predict(pairs)

        for i, hit in enumerate(rerank_candidates):
            score = float(rerank_scores[i])
            if hasattr(hit, 'score'):
                hit.score = score
            else:
                hit['score'] = score

        final_results = sorted(
            rerank_candidates,
            key=lambda x: (x.score if hasattr(x, 'score') else x['score']),
            reverse=True
        )

        return self._format_output(final_results[:top_k], "hybrid_multiquery_hyde_rerank")

    # ── Output Formatter ───────────────────────────────────────────────────────

    def _format_output(self, results: list, method: str) -> List[Dict[str, Any]]:
        formatted = []
        for i, r in enumerate(results):
            if hasattr(r, 'chunk_id'):
                formatted.append({
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "doc_id": r.doc_id,
                    "retrieval_score": r.score,
                    "retrieval_rank": i + 1,
                    "retrieval_method": method
                })
            else:
                formatted.append({
                    **r,
                    "retrieval_score": r["score"],
                    "retrieval_rank": i + 1,
                    "retrieval_method": method
                })
        return formatted


# --- CLI TEST ---
if __name__ == "__main__":
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(store)
    query = "impact of semantic chunking on RAG performance"
    results = retriever.search(query, top_k=3)
    print(f"\n[OK] Top Results for: {query}")
    for r in results:
        print(f"[{r['retrieval_rank']}] Score: {r['retrieval_score']:.4f} | {r['doc_id']}")
        print(f"Text: {r['text'][:150]}...\n")
