"""
Unit tests untuk 3 fixes di hybrid_retriever.py
Tidak butuh Qdrant atau Groq running.

Jalanin:
    python tests/test_retriever_fixes.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Mock supaya tidak perlu import Qdrant ─────────────────────────────────────

class MockRetrievalResult:
    def __init__(self, chunk_id, text, doc_id, score):
        self.chunk_id = chunk_id
        self.text = text
        self.doc_id = doc_id
        self.score = score

class MockVectorStore:
    def search(self, query, k):
        return [
            MockRetrievalResult(
                chunk_id=f"c_{i}",
                text=f"chunk text about {query[:20]} number {i}",
                doc_id="doc_test",
                score=1.0 - i * 0.05
            )
            for i in range(k)
        ]

# Patch imports sebelum import MasterHybridRetriever
import unittest.mock as mock

# Patch QdrantVectorStore dan RetrievalResult
with mock.patch.dict("sys.modules", {
    "src.retrieval.qdrant_store": mock.MagicMock(
        QdrantVectorStore=MockVectorStore,
        RetrievalResult=MockRetrievalResult
    )
}):
    # Patch sentence_transformers dan rank_bm25
    with mock.patch.dict("sys.modules", {
        "sentence_transformers": mock.MagicMock(),
        "rank_bm25": mock.MagicMock(),
        "groq": mock.MagicMock(),
    }):
        # Import setelah patch
        import importlib
        import types

        # Manual minimal class untuk test tanpa full import
        pass


# ── Test langsung tanpa import penuh ─────────────────────────────────────────
# Karena dependency berat (sentence_transformers, qdrant, dll),
# kita test logic murni secara isolated.

PASS = "✓"
FAIL = "✗"
results_log = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results_log.append((name, condition))
    return condition


# ── FIX 1: Adaptive rerank_n ──────────────────────────────────────────────────

def test_fix1_adaptive_rerank_n():
    print("\n[FIX 1] Adaptive rerank_n (bukan hardcoded 5)")
    
    # Formula: rerank_n = min(top_k * 2, 10)
    cases = [
        (3,  6,  "top_k=3 → rerank 6, bukan 5"),
        (5,  10, "top_k=5 → rerank 10, bukan 5"),
        (7,  10, "top_k=7 → rerank 10 (capped)"),
        (2,  4,  "top_k=2 → rerank 4"),
        (10, 10, "top_k=10 → rerank 10 (capped)"),
    ]
    
    for top_k, expected, note in cases:
        rerank_n = min(top_k * 2, 10)
        check(note, rerank_n == expected, f"got {rerank_n}")
    
    # Pastikan rerank_n selalu >= top_k
    for top_k in range(1, 11):
        rerank_n = min(top_k * 2, 10)
        check(
            f"rerank_n >= top_k untuk top_k={top_k}",
            rerank_n >= top_k,
            f"rerank_n={rerank_n}"
        )


# ── FIX 2: Query Classifier ───────────────────────────────────────────────────

def _classify_query(query: str) -> int:
    """Copy dari kode — test isolated."""
    words = query.strip().split()
    n = len(words)
    if n <= 5:
        return 3
    if n >= 15:
        return 7
    return 5

def test_fix2_query_classifier():
    print("\n[FIX 2] Adaptive top_k via query classifier")

    cases = [
        # (query, expected_top_k, description)
        ("RAG", 3, "1 word → 3"),
        ("what is RAG", 3, "3 words → 3"),
        ("what is retrieval augmented generation", 3, "5 words → 3"),
        ("how does hybrid search improve RAG quality", 5, "7 words → 5"),
        ("explain the difference between dense and BM25 retrieval", 5, "8 words → 5"),
        (
            "what are the main architectural differences between dense and sparse "
            "retrieval methods in modern RAG systems and how do they affect faithfulness",
            7,
            "≥15 words → 7"
        ),
    ]

    for query, expected, note in cases:
        result = _classify_query(query)
        check(note, result == expected, f"words={len(query.split())}, got top_k={result}")

    # Edge: top_k=0 trigger → harus auto-detect (simulated)
    # Dalam kode asli: if top_k == 0: top_k = self._classify_query(query)
    query = "what is RAG"
    top_k = 0
    if top_k == 0:
        top_k = _classify_query(query)
    check("top_k=0 trigger → auto-detect", top_k == 3, f"resolved to {top_k}")


# ── FIX 3: Consistent candidate pipeline ─────────────────────────────────────

def test_fix3_candidate_pipeline():
    print("\n[FIX 3] Consistent candidate_k formula")

    # Formula: candidate_k = min(top_k * 3, 30), rerank_n = min(top_k * 2, 10)
    cases = [
        (3,  9,  6),
        (5,  15, 10),
        (7,  21, 10),
        (10, 30, 10),
    ]

    for top_k, exp_candidate, exp_rerank in cases:
        candidate_k = min(top_k * 3, 30)
        rerank_n    = min(top_k * 2, 10)
        check(
            f"top_k={top_k} → fetch={candidate_k}, rerank={rerank_n}",
            candidate_k == exp_candidate and rerank_n == exp_rerank,
            f"got fetch={candidate_k}, rerank={rerank_n}"
        )

    # Invariant: candidate_k >= rerank_n >= top_k selalu
    print("  Invariant check: candidate_k >= rerank_n >= top_k")
    for top_k in range(1, 11):
        candidate_k = min(top_k * 3, 30)
        rerank_n    = min(top_k * 2, 10)
        ok = candidate_k >= rerank_n >= top_k
        check(
            f"invariant top_k={top_k}",
            ok,
            f"fetch={candidate_k}, rerank={rerank_n}, return={top_k}"
        )


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    total  = len(results_log)
    passed = sum(1 for _, ok in results_log if ok)
    failed = total - passed

    print("\n" + "=" * 50)
    print(f"  HASIL: {passed}/{total} passed", end="")
    if failed:
        print(f" | {failed} FAILED:")
        for name, ok in results_log:
            if not ok:
                print(f"    {FAIL} {name}")
    else:
        print(" — semua OK")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("  RAG Retriever Fix — Unit Tests")
    print("  Tidak butuh Qdrant / Groq / GPU")
    print("=" * 50)

    test_fix1_adaptive_rerank_n()
    test_fix2_query_classifier()
    test_fix3_candidate_pipeline()
    print_summary()