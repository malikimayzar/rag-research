from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from src.retrieval.qdrant_store import QdrantVectorStore, RetrievalResult
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

class MasterHybridRetriever:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_chunks_path: str = "data/processed/chunks_semantic.json",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
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

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        dense_weight: float = 0.7, 
        bm25_weight: float = 0.3,
        use_reranker: bool = True
    ) -> List[Dict[str, Any]]:
        candidate_k = min(top_k * 4, 20)
        dense_hits = self.vector_store.search(query, k=candidate_k)
        bm25_hits = []
        if self._bm25:
            query_tokens = query.lower().split()
            scores = self._bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:candidate_k]
            for i in top_indices:
                if scores[i] > 0:
                    bm25_hits.append({
                        "chunk_id": self._chunks[i]["chunk_id"],
                        "text": self._chunks[i]["text"],
                        "score": float(scores[i]),
                        "doc_id": self._chunks[i]["doc_id"]
                    })

        fused_scores = {}
        chunk_map = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0) + dense_weight / (self.rrf_k + rank + 1)
            chunk_map[cid] = hit

        for rank, hit in enumerate(bm25_hits):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0) + bm25_weight / (self.rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = hit

        fused_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]
        fused_results = [chunk_map[cid] for cid, _ in fused_ids]

        if not use_reranker or not fused_results:
            return self._format_output(fused_results[:top_k], "hybrid_rrf")
        
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

        print(f"[DEBUG] Dense hits: {len(dense_hits)} | BM25 hits: {len(bm25_hits)}")
        print(f"[DEBUG] Fused candidates: {len(fused_results)}")

        return self._format_output(final_results[:top_k], "hybrid_rerank")

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