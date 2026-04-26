from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGSample

THRESHOLDS = {
    "faithfulness": 0.88,
    "answer_relevancy": 0.88,
    "context_recall": 0.90,
    "hallucination_rate": 0.02,
}

DEFAULT_ADVERSARIAL_DIR = Path("data/adversarial")


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(path: Path) -> list[RAGSample]:
    if not path.exists():
        raise FileNotFoundError(f"Eval path not found: {path}")

    samples: list[RAGSample] = []
    if path.is_file():
        items = load_json_file(path)
        if isinstance(items, dict):
            items = [items]
    else:
        items = []
        for file_path in sorted(path.glob("*.json")):
            payload = load_json_file(file_path)
            if isinstance(payload, dict):
                items.extend(payload.get("samples", [payload]))
            elif isinstance(payload, list):
                items.extend(payload)

    for item in items:
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("query")
        answer = item.get("answer") or item.get("response")
        contexts = item.get("contexts") or item.get("context") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        if not question or not answer:
            continue
        samples.append(RAGSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=item.get("ground_truth"),
        ))

    return samples


def run_ci_eval(eval_dir: Path, output_path: Path) -> int:
    print(f"[CI EVAL] Loading adversarial data from: {eval_dir}")
    samples = load_samples(eval_dir)
    if not samples:
        print(f"No samples found in {eval_dir}")
        return 1

    evaluator = RAGASEvaluator(include_answer_correctness=False)
    result = evaluator.evaluate(samples, exp_id="ci_eval")
    summary = {
        "avg_faithfulness": result.avg_faithfulness,
        "avg_answer_relevancy": result.avg_answer_relevancy,
        "avg_context_recall": result.avg_context_recall,
        "hallucination_rate": result.hallucination_rate,
        "total_samples": result.total_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "config": THRESHOLDS}, f, indent=2)

    failures = []
    for metric, threshold in THRESHOLDS.items():
        actual = summary.get(metric)
        if metric == "hallucination_rate":
            if actual is not None and actual > threshold:
                failures.append(f"{metric}: {actual:.3f} > {threshold}")
        else:
            if actual is not None and actual < threshold:
                failures.append(f"{metric}: {actual:.3f} < {threshold}")

    print("\n[CI EVAL] Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if failures:
        print("\nEVAL GATE FAILED:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nAll eval gates passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CI evaluation gate")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_ADVERSARIAL_DIR,
        help="Path to adversarial evaluation samples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ci_eval.json"),
        help="Path to save the CI evaluation summary",
    )
    args = parser.parse_args()

    exit_code = run_ci_eval(args.eval_dir, args.output)
    sys.exit(exit_code)
