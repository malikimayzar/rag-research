import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "results" / "logs" / "single_query_full.json"
SCORE_DISTRIBUTION_PATH = ROOT / "data" / "debug" / "score_distribution.json"

def load_log_entries() -> list[dict]:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Required log file not found: {LOG_PATH}")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_score_distribution() -> dict:
    if not SCORE_DISTRIBUTION_PATH.exists():
        raise FileNotFoundError(f"Required score distribution file not found: {SCORE_DISTRIBUTION_PATH}")
    with open(SCORE_DISTRIBUTION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_author_query_answered_from_log():
    entries = load_log_entries()
    author_queries = [
        e for e in entries
        if "who are the authors of the paper" in e.get("query", "").lower()
    ]

    assert author_queries, "No author query records found in single_query_full.json"
    assert any(
        e.get("status") == "ANSWERED" and "Deng" in e.get("answer", "")
        for e in author_queries
    ), (
        "Expected a recorded author query to be answered with 'Deng' in the answer. "
        "This catches regressions in factual retrieval and reference handling."
    )

def test_no_low_confidence_answer_in_logs():
    entries = load_log_entries()
    low_confidence_answers = [
        e for e in entries
        if e.get("status") == "ANSWERED" and (e.get("confidence_v1") or 0) <= 0.3
    ]

    assert not low_confidence_answers, (
        "Found ANSWERED results with low retrieval confidence. "
        "The system should abstain rather than answer when retrieval signal is weak."
    )

def test_insufficient_context_factual_queries_do_not_answer():
    entries = load_log_entries()
    factual_author_queries = [
        e for e in entries
        if "author" in e.get("query", "").lower() and "paper" in e.get("query", "").lower()
    ]

    assert factual_author_queries, "No factual author query examples found in logs."
    assert any(e.get("status") == "INSUFFICIENT_CONTEXT" for e in factual_author_queries), (
        "Expected at least one factual author query to be labeled INSUFFICIENT_CONTEXT "
        "when retrieval signal was insufficient."
    )

def test_score_distribution_threshold_is_valid():
    distribution = load_score_distribution()
    mean = float(distribution.get("top1_mean", 0.0))
    std = float(distribution.get("top1_std", 0.0))
    threshold = max(mean - std, 0.0)

    assert threshold > 0.0, "Score calibration threshold should be positive."
    assert threshold < mean, "Calibration threshold must be lower than the mean top1 score."

if __name__ == "__main__":
    print("Running retrieval e2e regression tests...")
    test_author_query_answered_from_log()
    test_no_low_confidence_answer_in_logs()
    test_insufficient_context_factual_queries_do_not_answer()
    test_score_distribution_threshold_is_valid()
    print("All retrieval e2e checks passed.")