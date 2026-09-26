import argparse
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever

# Test queries — 3 kategori
TEST_QUERIES = [
    ("short",   "what is RAG"),
    ("medium",  "how does hybrid search improve retrieval quality in RAG"),
    ("long",    "what are the main architectural differences between dense and sparse "
                "retrieval methods in modern production RAG systems and how do they "
                "affect faithfulness scores"),
]

# ── Simulate formulas ─────────────────────────────────────────────────────────
def _classify_query(query: str) -> int:
    words = len(query.strip().split())
    if words <= 5:
        return 3
    if words >= 15:
        return 7
    return 5


def simulate_before(top_k_fixed: int = 5):
    candidate_k = min(top_k_fixed * 4, 20)
    rerank_n = 5
    return candidate_k, rerank_n, top_k_fixed


def simulate_after(query: str):
    top_k = _classify_query(query)
    candidate_k = min(top_k * 3, 30)
    rerank_n = min(top_k * 2, 10)
    return candidate_k, rerank_n, top_k


def print_comparison_table():
    print("\n" + "=" * 70)
    print("  BEFORE vs AFTER — Formula Comparison")
    print("=" * 70)
    print(f"  {'Query Type':<12} {'Words':>6} | {'BEFORE':^20} | {'AFTER':^20}")
    print(f"  {'':<12} {'':>6} | {'fetch/rerank/return':^20} | {'fetch/rerank/return':^20}")
    print("-" * 70)

    for qtype, query in TEST_QUERIES:
        words = len(query.split())
        bc, br, bt = simulate_before()
        ac, ar, at = simulate_after(query)
        print(
            f"  {qtype:<12} {words:>6} | "
            f"{f'{bc}/{br}/{bt}':^20} | "
            f"{f'{ac}/{ar}/{at}':^20}"
        )

    print("=" * 70)
    print("""
  Insight:
  - SHORT query: before reranks 5, after reranks 6 (+1 tapi top_k tetap 3)
  - MEDIUM query: before fetch 20 rerank 5, after fetch 15 rerank 10
  - LONG query: before sama, after fetch 21 rerank 10 return 7
""")


def run_live_benchmark():
    """Jalanin benchmark dengan Qdrant aktif."""
    print("\n" + "=" * 70)
    print("  LIVE BENCHMARK — Qdrant retrieval (auto top_k)")
    print("=" * 70)

    try:
        store = QdrantVectorStore()
    except Exception as e:
        print(f"\n  [SKIP] Qdrant tidak tersedia: {e}")
        print("  Jalanin dulu: docker compose -f docker-compose.yml up -d qdrant")
        return

    retriever = MasterHybridRetriever(
        store,
        use_multi_query=False,
        use_hyde=False,
    )

    print(f"\n  {'Query Type':<12} {'top_k':>6} {'Returned':>9} {'Latency':>10}")
    print("-" * 70)

    for qtype, query in TEST_QUERIES:
        t0 = time.time()
        results = retriever.search(query, top_k=0)
        elapsed = time.time() - t0
        print(f"  {qtype:<12} {_classify_query(query):>6} {len(results):>9} {elapsed:.3f}s")

    print("\n  Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval formulas")
    parser.add_argument("--live", action="store_true", help="Run live Qdrant benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("  RAG Retriever — Before vs After Benchmark")
    print("=" * 70)
    print_comparison_table()
    if args.live:
        run_live_benchmark()
    else:
        print("\n  Tip: jalankan dengan --live untuk eksekusi Qdrant nyata.")