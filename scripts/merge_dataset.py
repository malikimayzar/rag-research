import json
from pathlib import Path

FILES = {
    "factual":    "data/processed/ground_truth_qa.json",
    "multihop":   "data/processed/multihop_queries.json",
    "paraphrase": "data/processed/paraphrase_queries.json",
    "adversarial":"data/processed/adversarial_queries.json",
}
OUTPUT = "data/processed/ground_truth_qa_v2.json"

def normalize(item: dict, fallback_type: str) -> dict:
    return {
        "query":          item.get("question") or item.get("query", ""),
        "gold_answer":    item.get("gold_answer") or item.get("ground_truth", ""),
        "gold_chunk_id":  item.get("gold_chunk_id") or item.get("source_chunk", ""),
        "question_type":  item.get("question_type") or item.get("query_type") or fallback_type,
        "should_abstain": item.get("should_abstain", False),
        "doc_id":         item.get("doc_id") or item.get("source_doc", ""),
    }

def main():
    merged = []
    for qtype, path in FILES.items():
        data = json.load(open(path))
        for item in data:
            merged.append(normalize(item, fallback_type=qtype))

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(merged, open(OUTPUT, "w"), indent=2, ensure_ascii=False)

    print(f"Total samples : {len(merged)}")
    
    from collections import Counter
    dist = Counter(r["question_type"] for r in merged)
    for k, v in dist.items():
        pct = round(v / len(merged) * 100, 1)
        print(f"  {k}: {v} samples ({pct}%)")

if __name__ == "__main__":
    main()