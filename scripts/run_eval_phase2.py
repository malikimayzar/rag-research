import json
import time
import asyncio

from src.controller.agent import Agent
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime

# Config 
DATASET_PATH           = "data/processed/holdout_eval_v2.json"
OUTPUT_PATH            = f"results/eval_phase2_results.json"
TOP_K                  = 10
SLEEP                  = 0.3

# Metrics 
def normalize(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    return ' '.join(text.split())

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(normalize(pred).split())
    gold_tokens = set(normalize(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def recall_at_k(retrieved_ids: list, gold_chunk_ids: list, k: int = 10) -> float:
    for gold_chunk_id in gold_chunk_ids:
        if gold_chunk_id in retrieved_ids[:k]:
            return 1.0
    return 0.0

def reciprocal_rank(retrieved_ids: list, gold_chunk_ids: list) -> float:
    best_rr = 0.0
    for gold_chunk_id in gold_chunk_ids:
        for i, cid in enumerate(retrieved_ids):
            if cid == gold_chunk_id:
                rr = 1.0 / (i + 1)
                best_rr = max(best_rr, rr)
    return best_rr

# Main 
async def main():
    load_dotenv()
    agent = Agent()
    store = agent.store
    retriever = MasterHybridRetriever(vector_store=store)
    
    dataset = json.load(open(DATASET_PATH))
    print(f"[EVAL] Dataset: {len(dataset)} samples")

    results = []
    per_type = defaultdict(list)

    for i, item in enumerate(dataset):
        query = item.get("query") or item.get("question", "")
        gold_answer = item.get("answer") or item.get("gold_answer", "")
        gold_chunk_ids = item.get("gold_chunk_ids") or item.get("supporting_chunks") or [item.get("gold_chunk_id", "")]
        qtype = item.get("type") or item.get("question_type", "unknown")
        should_abstain = item.get("should_abstain", False)

        print(f"[{i+1}/{len(dataset)}] [{qtype}] {query[:60]}...")

        try:
            t0 = time.time()
            agent_resp = await agent.run(query)
            latency_ms = (time.time() - t0) * 1000

            raw_chunks = retriever.search(query, top_k=10)
            retrieved_ids = [c.get('chunk_id') if isinstance(c, dict) else getattr(c, 'chunk_id', '') for c in raw_chunks]
            system_abstained = agent_resp.status.value in ("abstained", "failed")
            pred_answer = agent_resp.answer or ""

            r_at_k = recall_at_k(retrieved_ids, gold_chunk_ids, k=TOP_K)
            rr     = reciprocal_rank(retrieved_ids, gold_chunk_ids)

            em = f1 = 0.0
            if not should_abstain and gold_answer:
                em = exact_match(pred_answer, gold_answer)
                f1 = token_f1(pred_answer, gold_answer)

            # Abstain accuracy
            abstain_correct = (system_abstained == should_abstain)

            # Failure type
            if should_abstain and not system_abstained:
                failure_type = "false_answer"       
            elif not should_abstain and system_abstained:
                failure_type = "false_abstain"       
            elif r_at_k == 0.0:
                failure_type = "retrieval_miss"
            elif rr < 0.33:
                failure_type = "ranking_error"
            else:
                failure_type = "none"

            record = {
                "query":           query,
                "type":            qtype,
                "should_abstain":  should_abstain,
                "system_abstained": system_abstained,
                "abstain_correct": abstain_correct,
                "gold_chunk_ids":  gold_chunk_ids,
                "retrieved_ids":   retrieved_ids[:5],
                "recall_at_10":    r_at_k,
                "reciprocal_rank": round(rr, 4),
                "exact_match":     em,
                "token_f1":        round(f1, 4),
                "failure_type":    failure_type,
                "latency_ms":      round(latency_ms, 1),
                "confidence":      agent_resp.state.confidence_score,
            }
            results.append(record)
            per_type[qtype].append(record)
            time.sleep(SLEEP)

        except Exception as e:
            import traceback
            traceback.print_exc()
            time.sleep(3)
            continue

    #  Aggregate metrics 
    total = len(results)
    answerable = [r for r in results if not r["should_abstain"]]
    abstain_labeled = [r for r in results if r["should_abstain"]]

    summary = {
        "total":           total,
        "recall_at_10":    round(sum(r["recall_at_10"] for r in results) / total, 4),
        "mrr_at_10":       round(sum(r["reciprocal_rank"] for r in results) / total, 4),
        "exact_match":     round(sum(r["exact_match"] for r in answerable) / max(len(answerable), 1), 4),
        "token_f1":        round(sum(r["token_f1"] for r in answerable) / max(len(answerable), 1), 4),
        "abstain_accuracy": round(sum(r["abstain_correct"] for r in results) / total, 4),
        "false_abstain_rate": round(sum(1 for r in answerable if r["system_abstained"]) / max(len(answerable), 1), 4),
        "false_answer_rate":  round(sum(1 for r in abstain_labeled if not r["system_abstained"]) / (len(abstain_labeled) if abstain_labeled else None), 4) if abstain_labeled else None,
        "failure_distribution": defaultdict(int),
        "per_type": {},
    }

    for r in results:
        summary["failure_distribution"][r["failure_type"]] += 1

    for qtype, recs in per_type.items():
        ans = [r for r in recs if not r["should_abstain"]]
        summary["per_type"][qtype] = {
            "count":       len(recs),
            "recall_at_10": round(sum(r["recall_at_10"] for r in recs) / len(recs), 4),
            "mrr":         round(sum(r["reciprocal_rank"] for r in recs) / len(recs), 4),
            "exact_match": round(sum(r["exact_match"] for r in ans) / max(len(ans), 1), 4),
        }

    summary["failure_distribution"] = dict(summary["failure_distribution"])

    #  Save ─
    output = {"summary": summary, "results": results}
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)

    # Versioned save
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_PATH_VERSIONED = f"results/metrics/eval_phase2_{_ts}.json"
    Path(OUTPUT_PATH_VERSIONED).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(OUTPUT_PATH_VERSIONED, "w"), indent=2, ensure_ascii=False)
    
    #  Print 
    print("\n" + "="*60)
    print("PHASE 2 EVAL RESULTS")
    print("="*60)
    print(f"Total samples     : {total}")
    print(f"Recall@10         : {summary['recall_at_10']}")
    print(f"MRR@10            : {summary['mrr_at_10']}")
    print(f"Exact Match       : {summary['exact_match']}")
    print(f"Token F1          : {summary['token_f1']}")
    print(f"Abstain Accuracy  : {summary['abstain_accuracy']}")
    print(f"False Abstain Rate: {summary['false_abstain_rate']}")
    far = summary['false_answer_rate']
    print(f'False Answer Rate : {far if far is not None else "N/A (no abstain samples)"}')
    print(f"\nPer Type:")
    for qtype, m in summary["per_type"].items():
        print(f"  [{qtype}] recall={m['recall_at_10']} mrr={m['mrr']} em={m['exact_match']}")
    print(f"\nFailure Distribution: {summary['failure_distribution']}")
    print(f"\nSaved -> {OUTPUT_PATH}")
    print(f"Versioned -> {OUTPUT_PATH_VERSIONED}")

if __name__ == "__main__":
    asyncio.run(main())