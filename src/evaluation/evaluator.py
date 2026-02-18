import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import re

import ollama

FAILURE_MODES = {
    "retrieval_miss": "Chunk yang relevan tidak masuk top-k sama sekali",
    "partial_context": "Chunk relevan masuk tapi informasi tidak lengkap",
    "misleading_similarity": "Chunk high-score tapi tidak relevan untuk menjawab",
    "hallucination_with_context": "Konteks tersedia tapi LLM tetap hallucinate",
    "correct": "Retrieval dan generation benar",
    "honest_abstention": "LLM jujur bilang tidak tahu karena konteks tidak cukup",
    "context_ordering_bias": "Jawaban berubah bergantung posisi chunk dalam prompt"
}


@dataclass
class EvaluationResult:
    query: str
    answer: str
    ground_truth: Optional[str]
    retrieved_chunks: list[dict]
    retrieval_method: str

    # Metrics
    faithfulness_score: float       
    context_relevance_score: float  
    answer_relevance_score: float   

    # Error taxonomy
    failure_mode: str
    failure_explanation: str

    # Flags
    has_hallucination: bool
    is_honest_abstention: bool


def score_faithfulness_llm(
    answer: str,
    context: str,
    model: str = "mistral"
) -> tuple[float, str]:
    prompt = f"""You are evaluating an AI answer for faithfulness to its source context.

CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

Task: Rate how faithful the answer is to the context on a scale of 0.0 to 1.0.
- 1.0 = Every claim in the answer is directly supported by the context
- 0.5 = Some claims supported, some not verifiable from context  
- 0.0 = Answer contains claims not present in context (hallucination)

Respond in this exact format:
SCORE: [number between 0.0 and 1.0]
REASON: [one sentence explanation]"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )

    text = response["message"]["content"]

    # Parse score
    score_match = re.search(r'SCORE:\s*([\d.]+)', text)
    reason_match = re.search(r'REASON:\s*(.+)', text)

    score = float(score_match.group(1)) if score_match else 0.5
    reason = reason_match.group(1).strip() if reason_match else "Could not parse reason"

    # Clamp ke 0-1
    score = max(0.0, min(1.0, score))
    return score, reason


def score_context_relevance(
    query: str,
    chunks: list[dict],
    model: str = "mistral"
) -> tuple[float, str]:
    chunk_texts = "\n\n".join([
        f"[Chunk {i+1}]: {c['text'][:300]}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"""You are evaluating retrieved context chunks for their relevance to a query.

QUERY: {query}

RETRIEVED CHUNKS:
{chunk_texts}

Task: Rate how relevant these chunks are for answering the query on a scale of 0.0 to 1.0.
- 1.0 = Chunks directly contain information needed to answer the query
- 0.5 = Chunks partially relevant, some useful information present
- 0.0 = Chunks are completely irrelevant to the query

Respond in this exact format:
SCORE: [number between 0.0 and 1.0]
REASON: [one sentence explanation]"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )

    text = response["message"]["content"]
    score_match = re.search(r'SCORE:\s*([\d.]+)', text)
    reason_match = re.search(r'REASON:\s*(.+)', text)

    score = float(score_match.group(1)) if score_match else 0.5
    reason = reason_match.group(1).strip() if reason_match else "Could not parse reason"
    score = max(0.0, min(1.0, score))

    return score, reason


def classify_failure_mode(
    query: str,
    answer: str,
    chunks: list[dict],
    faithfulness: float,
    context_relevance: float
) -> tuple[str, str]:
    abstention_phrases = [
        "does not contain", "not in the context",
        "context does not", "no information", "cannot answer",
        "not provided", "insufficient"
    ]
    is_abstention = any(p in answer.lower() for p in abstention_phrases)

    if is_abstention:
        return "honest_abstention", "LLM correctly identified missing context"

    if context_relevance < 0.3:
        return "retrieval_miss", f"Retrieved chunks have low relevance ({context_relevance:.2f})"

    if 0.3 <= context_relevance < 0.6:
        return "partial_context", f"Context partially relevant ({context_relevance:.2f}), may cause incomplete answer"

    if context_relevance >= 0.6 and faithfulness < 0.5:
        return "hallucination_with_context", \
            f"Good context (rel={context_relevance:.2f}) but low faithfulness ({faithfulness:.2f})"

    top_chunk_score = chunks[0].get("retrieval_score", 0) if chunks else 0
    if context_relevance < 0.5 and top_chunk_score > 0.01:
        return "misleading_similarity", \
            f"High retrieval score but low context relevance — semantic gap"

    if context_relevance >= 0.6 and faithfulness >= 0.7:
        return "correct", f"Retrieval and generation both high quality"

    return "partial_context", f"Mixed signals — rel={context_relevance:.2f}, faith={faithfulness:.2f}"


def evaluate_response(
    query: str,
    answer: str,
    chunks: list[dict],
    retrieval_method: str,
    ground_truth: Optional[str] = None
) -> EvaluationResult:
    print(f"  [EVAL] Evaluating: {query[:50]}...")

    context = "\n\n".join([c["text"] for c in chunks])

    # faithfulness
    faithfulness, faith_reason = score_faithfulness_llm(answer, context)
    print(f"  → Faithfulness: {faithfulness:.2f} | {faith_reason[:60]}")

    # context relevance
    ctx_relevance, ctx_reason = score_context_relevance(query, chunks)
    print(f"  → Context Rel : {ctx_relevance:.2f} | {ctx_reason[:60]}")
    answer_relevance = 0.8 if len(answer) > 20 and not answer.startswith("I don't") else 0.3

    # Classify failure mode
    failure_mode, failure_explanation = classify_failure_mode(
        query, answer, chunks, faithfulness, ctx_relevance
    )
    print(f"  → Failure Mode: {failure_mode}")

    return EvaluationResult(
        query=query,
        answer=answer,
        ground_truth=ground_truth,
        retrieved_chunks=chunks,
        retrieval_method=retrieval_method,
        faithfulness_score=faithfulness,
        context_relevance_score=ctx_relevance,
        answer_relevance_score=answer_relevance,
        failure_mode=failure_mode,
        failure_explanation=failure_explanation,
        has_hallucination=(failure_mode == "hallucination_with_context"),
        is_honest_abstention=(failure_mode == "honest_abstention")
    )


def generate_report(results: list[EvaluationResult]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    failure_counts = {}
    for r in results:
        failure_counts[r.failure_mode] = failure_counts.get(r.failure_mode, 0) + 1

    return {
        "total_queries": total,
        "avg_faithfulness": round(sum(r.faithfulness_score for r in results) / total, 3),
        "avg_context_relevance": round(sum(r.context_relevance_score for r in results) / total, 3),
        "avg_answer_relevance": round(sum(r.answer_relevance_score for r in results) / total, 3),
        "failure_mode_distribution": failure_counts,
        "hallucination_rate": round(sum(1 for r in results if r.has_hallucination) / total, 3),
        "honest_abstention_rate": round(sum(1 for r in results if r.is_honest_abstention) / total, 3)
    }


if __name__ == "__main__":
    with open("results/logs/generation_test.json", 'r') as f:
        generation_results = json.load(f)

    print(f"[EVAL] Evaluating {len(generation_results)} responses...\n")

    eval_results = []
    for item in generation_results:
        result = evaluate_response(
            query=item["query"],
            answer=item["answer"],
            chunks=item["retrieved_chunks"],
            retrieval_method=item["retrieval_method"]
        )
        eval_results.append(result)
        print()

    # Report
    report = generate_report(eval_results)
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    for k, v in report.items():
        print(f"{k}: {v}")

    # save
    output = {
        "report": report,
        "individual_results": [asdict(r) for r in eval_results]
    }

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with open("results/metrics/evaluation_results.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    failure_cases = [
        asdict(r) for r in eval_results
        if r.failure_mode not in ["correct", "honest_abstention"]
    ]
    if failure_cases:
        with open("results/failure_cases/failure_analysis.json", 'w') as f:
            json.dump(failure_cases, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] {len(failure_cases)} failure cases saved → results/failure_cases/")

    print(f"[OK] Full report saved → results/metrics/evaluation_results.json")