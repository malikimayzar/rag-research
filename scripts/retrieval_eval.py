from __future__ import annotations

import json
from typing import List, Dict, Any
from pathlib import Path

# METRICS
def normalize(text: str) -> str:
    return text.lower().strip()

def recall_at_k(retrieved_chunks: List[Dict[str, Any]], ground_truth: str, k: int = 5) -> int:
    gt = normalize(ground_truth)
    for chunk in retrieved_chunks[:k]:
        text = normalize(chunk.get("text", ""))
        if gt in text:
            return 1
    return 0

def mrr(retrieved_chunks: List[Dict[str, Any]], ground_truth: str) -> float:
    gt = normalize(ground_truth)
    for i, chunk in enumerate(retrieved_chunks):
        text = normalize(chunk.get("text", ""))
        if gt in text:
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
    recall_scores = []
    mrr_scores = []
    failures = []

    for i, item in enumerate(dataset):
        query = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(dataset)}] {query[:60]}...")
        chunks = retriever.search(query, top_k=top_k)
        r_at_k = recall_at_k(chunks, ground_truth, k=top_k)
        mrr_score = mrr(chunks, ground_truth)
        recall_scores.append(r_at_k)
        mrr_scores.append(mrr_score)

        result = {
            "question": query,
            "ground_truth": ground_truth,
            "recall@{}".format(top_k): r_at_k,
            "mrr": mrr_score,
            "retrieved_chunks": chunks
        }

        if r_at_k == 0:
            failures.append(result)
        results.append(result)

    # SUMMARY
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    summary = {
        "total_samples": len(dataset),
        "recall@{}".format(top_k): avg_recall,
        "mrr": avg_mrr,
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(dataset) if dataset else 0.0
    }

    print("\n" + "="*50)
    print("[RETRIEVAL EVAL RESULT]")
    print("="*50)
    print(f"Recall@{top_k} : {avg_recall:.4f}")
    print(f"MRR            : {avg_mrr:.4f}")
    print(f"Failures       : {len(failures)}/{len(dataset)}")
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
    dataset_path = "data/processed/ground_truth_qa.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = [
        {
            "question": d["question"],
            "ground_truth": d["ground_truth"]
        }
        for d in dataset
        if d.get("ground_truth")
    ]
    evaluate_retrieval(dataset, retriever, top_k=5)