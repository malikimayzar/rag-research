import json
import sys
import requests
from dotenv import load_dotenv

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGSample

load_dotenv()
sys.path.insert(0, ".")

# Load ground truth
with open("data/processed/ground_truth_qa.json") as f:
    gt_data = json.load(f)

print(f"Total QA pairs: {len(gt_data)}")
samples_gt = gt_data[:30]
print("Retrieving answers from API...")
ragas_samples = []
rejected = 0

for i, item in enumerate(samples_gt):
    question = item.get("question", item.get("query", ""))
    ground_truth = item.get("answer", item.get("ground_truth", ""))
    try:
        resp = requests.post(
            "http://localhost:8003/generate",
            json={"query": question, "top_k": 5},
            timeout=30
        )
        data = resp.json()
        answer = data["answer"]
        contexts = data["contexts"]
        
        is_rejected = answer.startswith("The provided context does not")
        if is_rejected:
            rejected += 1
            
        ragas_samples.append(RAGSample(
            question=question,
            answer=answer,
            contexts=contexts if contexts else ["No context retrieved"],
            ground_truth=ground_truth,
        ))
        print(f"[{i+1}/30] {'REJECTED' if is_rejected else 'OK'} | {question[:60]}")
        
    except Exception as e:
        print(f"[{i+1}/30] ERROR: {e}")

print(f"\nReject rate: {rejected}/{len(ragas_samples)} ({rejected/len(ragas_samples):.0%})")
print("\nRunning RAGAS eval...")

evaluator = RAGASEvaluator(
    llm_model="llama-3.3-70b-versatile",
    include_answer_correctness=False,
)

result = evaluator.evaluate(ragas_samples, exp_id="phase2_30sample_70b")
evaluator.save_result(result, "results/metrics/ragas_phase2_postfix_8b.json")