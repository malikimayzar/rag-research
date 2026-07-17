from __future__ import annotations

import json
from typing import List, Dict, Any
from pathlib import Path

# METRICS
def normalize(text: str) -> str:
    return text.lower().strip()

def recall_at_k(retrieved_chunks, gold_chunk_id, k=5):
    ids = [get_chunk_id(c) for c in retrieved_chunks[:k]]
    return 1 if gold_chunk_id in ids else 0

def get_chunk_id(chunk):
    if isinstance(chunk, dict):
        return chunk.get("chunk_id") or chunk.get("id")

    if hasattr(chunk, "chunk_id"):
        return chunk.chunk_id

    if hasattr(chunk, "metadata"):
        return chunk.metadata.get("chunk_id")

    raise ValueError(f"Unknown chunk format: {type(chunk)}")

def get_score(chunk):
    if isinstance(chunk, dict):
        return chunk.get("rerank_score") or chunk.get("score")
    if hasattr(chunk, "rerank_score"):
        return chunk.rerank_score
    if hasattr(chunk, "score"):
        return chunk.score
    if hasattr(chunk, "metadata"):
        return chunk.metadata.get("rerank_score") or chunk.metadata.get("score")
    return None

def mrr(retrieved_chunks, gold_chunk_id):
    for i, chunk in enumerate(retrieved_chunks):
        if get_chunk_id(chunk) == gold_chunk_id:
            return 1.0 / (i + 1)
    return 0.0

# CORE EVALUATION
def evaluate_retrieval(
    dataset: List[Dict[str, Any]],
    retriever,
    top_k: int = 5,
    output_path: str = "results/retrieval_eval.json"
):
    results = []
    recall1_scores = []
    recall5_scores = []
    recall10_scores = []
    mrr_scores = []
    failures = []

    for i, item in enumerate(dataset):
        query = item["question"]
        gold_chunk_id = item["gold_chunk_id"]

        chunks = retriever.search(query, top_k=top_k)

        r1  = recall_at_k(chunks, gold_chunk_id, k=1)
        r5  = recall_at_k(chunks, gold_chunk_id, k=5)
        r10 = recall_at_k(chunks, gold_chunk_id, k=10)
        mrr_score = mrr(chunks, gold_chunk_id)

        recall1_scores.append(r1)
        recall5_scores.append(r5)
        recall10_scores.append(r10)
        mrr_scores.append(mrr_score)

        retrieved_ids = [get_chunk_id(c) for c in chunks]  
        retrieved_chunks = [
            {
                "chunk_id": get_chunk_id(c),
                "rerank_score": float(get_score(c)) if get_score(c) is not None else None
            }
            for c in chunks
        ]

        rank = next(
            (i+1 for i, c in enumerate(chunks) if get_chunk_id(c) == gold_chunk_id),
            None
        )

        if rank is None:
            failure_type = "retrieval_miss"
        elif rank > 5:
            failure_type = "ranking_failure"
        else:
            failure_type = "success"

        result = {
            "question": query,
            "gold_chunk_id": gold_chunk_id,
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "rank": rank,
            "mrr": mrr_score,
            "failure_type": failure_type,
            "retrieved_ids": retrieved_ids,
            "retrieved_chunks": retrieved_chunks
        }

        results.append(result)

        if failure_type != "success":
            failures.append(result)

    # SUMMARY
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    summary = {
        "total_samples": len(dataset),
        "recall@1": sum(recall1_scores)/len(dataset),
        "recall@5": sum(recall5_scores)/len(dataset),
        "recall@10": sum(recall10_scores)/len(dataset),
        "mrr": avg_mrr,
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(dataset)
    }

    print("\n" + "="*50)
    print("[RETRIEVAL EVAL RESULT]")
    print("="*50)
    print(f"Recall@1  : {summary['recall@1']:.4f}")
    print(f"Recall@5  : {summary['recall@5']:.4f}")
    print(f"Recall@10 : {summary['recall@10']:.4f}")
    print(f"MRR       : {avg_mrr:.4f}")
    print(f"Failures  : {len(failures)}/{len(dataset)}")
    print("="*50 + "\n")

    # SAVE
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": results,
            "failures": failures
        }, f, indent=2, ensure_ascii=False)
    print(f"[Saved] → {output_path}")
    return summary

# CLI TEST
if __name__ == "__main__":
    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(store)
    dataset_path = "data/processed/holdout_eval.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = [
        {"question": d["question"], "gold_chunk_id": d["gold_chunk_id"]}
        for d in dataset if d.get("gold_chunk_id")
    ]
    evaluate_retrieval(dataset, retriever, top_k=10)