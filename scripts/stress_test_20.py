from __future__ import annotations
import argparse
import json
import random
import sys
import traceback

from collections import Counter
from pathlib import Path
from scripts.run_single_query import run_single_query

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Sumber query buat stratified sampling
SOURCES = {
    "ground_truth": "data/processed/ground_truth_qa_rebuilt.json",
    "adversarial":  "data/adversarial/adversarial_queries.json",
    "multihop":     "data/processed/multihop_queries.json",
    "paraphrase":   "data/processed/paraphrase_queries.json",
}

def load_queries(path: str, n: int, seed: int) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  [SKIP] {path} tidak ditemukan")
        return []
    with open(p) as f:
        data = json.load(f)
    if not data:
        return []
    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))
    return sample

def normalize_item(item: dict, source_label: str) -> dict:
    query = item.get("question") or item.get("query") or ""
    gt = item.get("ground_truth") or item.get("gold_answer") or item.get("answer")
    source_chunk = item.get("source_chunk") or item.get("gold_chunk_id")
    return {
        "query": query,
        "ground_truth": gt,
        "source_chunk_id": source_chunk,
        "source_label": source_label,
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 1 stress test — N query stratified, no-crash validation")
    parser.add_argument("--n-per-source", type=int, default=5, help="Jumlah query per kategori (default 5 x 4 = 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="full", choices=["baseline", "full"])
    args = parser.parse_args()

    all_queries: list[dict] = []
    for label, path in SOURCES.items():
        raw = load_queries(path, args.n_per_source, args.seed)
        normalized = [normalize_item(item, label) for item in raw]
        normalized = [q for q in normalized if q["query"]]
        print(f"  [{label}] loaded {len(normalized)} query")
        all_queries.extend(normalized)

    total = len(all_queries)
    print(f"\n{'='*60}")
    print(f"  STRESS TEST — {total} queries | mode={args.mode}")
    print(f"{'='*60}\n")

    results = []
    exceptions = []
    failure_types = Counter()
    statuses = Counter()
    latency_total = []
    answered_count = 0
    abstained_count = 0

    for i, item in enumerate(all_queries, 1):
        print(f"\n--- [{i}/{total}] ({item['source_label']}) {item['query'][:70]}...")
        try:
            output = run_single_query(
                query=item["query"],
                ground_truth=item["ground_truth"],
                source_chunk_id=item["source_chunk_id"],
                mode=args.mode,
                save_output=True,
            )
            results.append(output)
            statuses[output.get("status", "unknown")] += 1
            failure_types[output.get("failure_type", "none")] += 1
            total_ms = output.get("latency_breakdown", {}).get("total_ms", 0.0)
            if isinstance(total_ms, (int, float)):
                latency_total.append(float(total_ms))
            if output.get("status") == "ANSWERED":
                answered_count += 1
            if output.get("status") in {"INSUFFICIENT_CONTEXT", "FAILED"}:
                abstained_count += 1
        except Exception as e:
            print(f"  [CRASH] {type(e).__name__}: {e}")
            traceback.print_exc()
            exceptions.append({
                "index": i,
                "query": item["query"],
                "source": item["source_label"],
                "error_type": type(e).__name__,
                "error_message": str(e),
            })

    # SUMMARY
    print(f"\n\n{'='*60}")
    print(f"  STRESS TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Total queries     : {total}")
    print(f"  Completed         : {len(results)}")
    print(f"  Crashed           : {len(exceptions)}")
    print(f"\n  Status distribution:")
    for status, count in statuses.most_common():
        print(f"    {status:<25} {count}")
    print(f"\n  Failure type distribution:")
    for ftype, count in failure_types.most_common():
        print(f"    {ftype:<25} {count}")

    if latency_total:
        avg_latency = sum(latency_total) / len(latency_total)
        p95_latency = sorted(latency_total)[int(max(0, len(latency_total) * 0.95 - 1))]
        print(f"\n  Latency summary:")
        print(f"    avg_total_ms       : {avg_latency:.2f}")
        print(f"    p95_total_ms       : {p95_latency:.2f}")
    print(f"\n  Answerability summary:")
    print(f"    answered           : {answered_count}")
    print(f"    abstained          : {abstained_count}")
    print(f"    answer_rate        : {(answered_count / max(total, 1) * 100):.1f}%")

    if exceptions:
        print(f"\n  CRASHED QUERIES:")
        for exc in exceptions:
            print(f"    [{exc['index']}] ({exc['source']}) {exc['error_type']}: {exc['error_message']}")
            print(f"        query: {exc['query'][:80]}")

    # Save summary
    summary_path = Path("results/debug/stress_test_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "completed": len(results),
            "crashed": len(exceptions),
            "status_distribution": dict(statuses),
            "failure_type_distribution": dict(failure_types),
            "latency_ms": {
                "avg_total_ms": round(sum(latency_total) / len(latency_total), 2) if latency_total else 0.0,
                "p95_total_ms": round(sorted(latency_total)[int(max(0, len(latency_total) * 0.95 - 1))], 2) if latency_total else 0.0,
            },
            "answerability": {
                "answered": answered_count,
                "abstained": abstained_count,
                "answer_rate": round(answered_count / max(total, 1) * 100, 1),
            },
            "exceptions": exceptions,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  [Saved] {summary_path}")

    # Exit code — penting buat CI nanti
    if exceptions:
        print(f"\n  RESULT: FAIL — {len(exceptions)} crash(es) detected")
        sys.exit(1)
    else:
        print(f"\n  RESULT: PASS — 0 crashes across {total} queries")
        sys.exit(0)

if __name__ == "__main__":
    main()