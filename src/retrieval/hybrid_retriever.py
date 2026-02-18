import json
import sys
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "src/retrieval")
from embedder import load_index, dense_search
from bm25_retriever import load_bm25, bm25_search


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> list[dict]:
    
    scores = {} 
    chunk_map = {} 

    for result in dense_results:
        cid = result["chunk_id"]
        rank = result["retrieval_rank"]
        scores[cid] = scores.get(cid, 0) + dense_weight / (k + rank)
        chunk_map[cid] = result

    for result in bm25_results:
        cid = result["chunk_id"]
        rank = result["retrieval_rank"]
        scores[cid] = scores.get(cid, 0) + bm25_weight / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = result

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (cid, fused_score) in enumerate(ranked):
        chunk = chunk_map[cid].copy()
        chunk["retrieval_score"] = fused_score
        chunk["retrieval_rank"] = rank + 1
        chunk["retrieval_method"] = "hybrid_rrf"
        results.append(chunk)

    return results


def hybrid_search(
    query: str,
    emb_index,
    bm25,
    bm25_chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5
) -> dict:
    candidate_k = top_k * 3

    dense_results = dense_search(query, emb_index, model, top_k=candidate_k)
    bm25_results = bm25_search(query, bm25, bm25_chunks, top_k=candidate_k)

    fused = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight
    )[:top_k]

    return {
        "query": query,
        "dense": dense_results[:top_k],
        "bm25": bm25_results[:top_k],
        "hybrid": fused
    }


def compare_methods(results: dict):
    query = results["query"]
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    for method in ["dense", "bm25", "hybrid"]:
        print(f"\n--- {method.upper()} ---")
        for r in results[method][:3]:
            print(f"  Rank {r['retrieval_rank']} | Score: {r['retrieval_score']:.4f}")
            print(f"  {r['text'][:120].strip()}...")
            print()


if __name__ == "__main__":
    print("[LOAD] Loading indexes...")

    emb_index = load_index("data/processed/index_minilm")
    bm25, bm25_chunks = load_bm25("data/processed/index_bm25")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Test dengan 3 query berbeda untuk lihat perbedaan perilaku
    test_queries = [
        "what is chunking in RAG?",
        "how does hybrid retrieval work?",
        "what metrics are used to evaluate RAG systems?"
    ]

    all_results = []
    for query in test_queries:
        results = hybrid_search(
            query, emb_index, bm25, bm25_chunks, model, top_k=3
        )
        compare_methods(results)
        all_results.append(results)

    # Simpan untuk analisis nanti
    import json
    with open("results/logs/hybrid_search_test.json", 'w') as f:
        clean = []
        for r in all_results:
            clean.append({
                "query": r["query"],
                "dense": r["dense"],
                "bm25": r["bm25"],
                "hybrid": r["hybrid"]
            })
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print("\n[OK] Results saved → results/logs/hybrid_search_test.json")