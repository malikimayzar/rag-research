import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_analysis")

def analyze_results(results_file: str = "results/eval_phase2_results.json"):
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    logger.info("\n" + "="*80)
    logger.info("[PHASE 2 EVALUATION ANALYSIS]")
    logger.info("="*80)
    evaluated = [r for r in results if r.get("is_answer_correct", -1) != -1]
    
    if not evaluated:
        logger.warning("No manually evaluated results found yet.")
        logger.info("Please fill in is_answer_correct and has_hallucination fields first.")
        logger.info(f"Edit: {results_file}")
        return
    
    # Calculate metrics
    total = len(evaluated)
    correct = sum(1 for r in evaluated if r["is_answer_correct"] == 2)
    partial = sum(1 for r in evaluated if r["is_answer_correct"] == 1)
    wrong = sum(1 for r in evaluated if r["is_answer_correct"] == 0)
    hallucinations = sum(1 for r in evaluated if r.get("has_hallucination") == 1)
    
    accuracy = correct / total if total > 0 else 0
    hallucination_rate = hallucinations / total if total > 0 else 0
    
    logger.info(f"\n[QUALITY METRICS]")
    logger.info(f"  Total evaluated: {total}")
    logger.info(f"  Correct: {correct} ({correct/total*100:.1f}%)")
    logger.info(f"  Partial: {partial} ({partial/total*100:.1f}%)")
    logger.info(f"  Wrong: {wrong} ({wrong/total*100:.1f}%)")
    logger.info(f"  Hallucination rate: {hallucination_rate*100:.1f}%")
    
    # By category
    logger.info(f"\n[BY CATEGORY]")
    categories = set(r.get("category") for r in evaluated)
    
    for cat in sorted(categories):
        cat_results = [r for r in evaluated if r.get("category") == cat]
        cat_correct = sum(1 for r in cat_results if r["is_answer_correct"] == 2)
        cat_accuracy = cat_correct / len(cat_results) if cat_results else 0
        cat_halluc = sum(1 for r in cat_results if r.get("has_hallucination") == 1)
        
        logger.info(f"  {cat.upper()}: {cat_correct}/{len(cat_results)} correct ({cat_accuracy*100:.1f}%), {cat_halluc} hallucinations")
    
    # Failure analysis
    logger.info(f"\n[FAILURE ANALYSIS]")
    failures = [r for r in evaluated if r["is_answer_correct"] in [0, 1]]
    
    if failures:
        logger.info(f"  Total failures: {len(failures)}")
        
        # Categorize failures
        retrieval_issues = sum(1 for f in failures if f.get("retrieval_score_top1", 0) < 0.5)
        context_issues = sum(1 for f in failures if f["num_chunks_retrieved"] < 3)
        generation_issues = len(failures) - retrieval_issues - context_issues
        
        logger.info(f"  Likely causes:")
        logger.info(f"    - Low retrieval score (<0.5): {retrieval_issues}")
        logger.info(f"    - Few chunks retrieved (<3): {context_issues}")
        logger.info(f"    - Other (generation/context quality): {max(0, generation_issues)}")
    else:
        logger.info("  No failures! All queries answered correctly.")
    
    # Latency
    avg_latency = sum(r["latency_total_ms"] for r in evaluated) / len(evaluated)
    logger.info(f"\n[PERFORMANCE]")
    logger.info(f"  Avg latency: {avg_latency:.1f}ms")
    logger.info(f"  Max latency: {max(r['latency_total_ms'] for r in evaluated):.1f}ms")
    logger.info(f"  Min latency: {min(r['latency_total_ms'] for r in evaluated):.1f}ms")
    
    # Recommendations
    logger.info(f"\n[RECOMMENDATIONS]")
    
    if accuracy >= 0.9:
        logger.info("✓ System is production-ready! Excellent retrieval + generation quality.")
    elif accuracy >= 0.75:
        logger.info("[WARN] System is mostly good. Some edge cases need attention.")
        if retrieval_issues > 0:
            logger.info("  → Consider: Better chunking strategy or hybrid search tuning")
        if hallucination_rate > 0.1:
            logger.info("  → Consider: Stricter confidence thresholds or prompt refinement")
    else:
        logger.info("[Fail] Quality issues detected. Need significant improvements.")
        logger.info("  → Investigate retrieval failures first (before generation)")
    
    logger.info("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_results()