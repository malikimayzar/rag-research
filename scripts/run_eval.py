import json
import time

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGSample
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.generation.generator import GroqGenerator

from dotenv import load_dotenv
load_dotenv()

def run_full_pipeline(limit=55):
    print(f"[Started] Full Pipeline Evaluation ({limit} samples)...")
    print(f"  Generator : llama-3.3-70b-versatile")
    print(f"  Judge     : llama-3.1-8b-instant + json_mode")
    print(f"  Retrieval : Hybrid RRF + BGE Reranker")
    print(f"  Chunks    : Rust Semantic (982 chunks)")

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator(model="llama-3.3-70b-versatile")
    evaluator = RAGASEvaluator(
        llm_model="llama-3.1-8b-instant",
        include_answer_correctness=True
    )

    with open('data/processed/ground_truth_qa.json', 'r') as f:
        gt_data = json.load(f)

    test_data = gt_data[:limit]
    samples = []

    for i, item in enumerate(test_data):
        print(f"[{i+1}/{limit}] {item['question'][:60]}...")
        try:
            chunks = retriever.search(item['question'], top_k=10)
            resp = generator.generate(item['question'], chunks)
            samples.append(RAGSample(
                question=item['question'],
                answer=resp.answer,
                contexts=[
                    c.get("text", "") if isinstance(c, dict) else c.text
                    for c in chunks
                ],
                ground_truth=item.get("ground_truth", item.get("gold_answer"))
            ))
            time.sleep(0.5)
        except Exception as e:
            print(f"  [Warning] {e}")
            time.sleep(3)
            continue

    if samples:
        print(f"\n[RAGAS] Evaluating {len(samples)} samples...")
        result = evaluator.evaluate(
            samples,
            exp_id="final_v1_70b_gen_8b_judge",
            config={
                "chunking": "rust_semantic",
                "retrieval": "hybrid_rerank",
                "generator": "llama-3.3-70b-versatile",
                "judge": "llama-3.1-8b-instant+json_mode",
                "dataset": "clean_55_pairs",
            }
        )
        evaluator.save_result(result, 'results/metrics/ragas_results.json')
        print("[OK] Done! Results saved.")

if __name__ == "__main__":
    run_full_pipeline(limit=55)