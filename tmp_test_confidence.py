# tmp_test_confidence.py
import asyncio
from src.controller.confidence_engine import ConfidenceEngine

engine = ConfidenceEngine()

# Case 1: chunks bagus — harusnya GENERATE
chunks_good = [
    {"retrieval_score": 0.9, "metadata": {"retrieval_method": "hybrid"}},
    {"retrieval_score": 0.85, "metadata": {"retrieval_method": "hybrid"}},
    {"retrieval_score": 0.4, "metadata": {"retrieval_method": "hybrid"}},
]

# Case 2: chunks jelek — harusnya REJECT
chunks_bad = [
    {"retrieval_score": 0.1, "metadata": {"retrieval_method": "hybrid"}},
    {"retrieval_score": 0.09, "metadata": {"retrieval_method": "bm25"}},
    {"retrieval_score": 0.08, "metadata": {"retrieval_method": "dense"}},
]

# Case 3: borderline — score sedang, agreement rendah
chunks_borderline = [
    {"retrieval_score": 0.55, "metadata": {"retrieval_method": "hybrid"}},
    {"retrieval_score": 0.52, "metadata": {"retrieval_method": "bm25"}},
    {"retrieval_score": 0.50, "metadata": {"retrieval_method": "dense"}},
]

# Case 4: noisy — top score tinggi tapi yang lain sangat rendah
chunks_noisy = [
    {"retrieval_score": 0.9, "metadata": {"retrieval_method": "hybrid"}},
    {"retrieval_score": 0.1, "metadata": {"retrieval_method": "hybrid"}},
    {"retrieval_score": 0.05, "metadata": {"retrieval_method": "hybrid"}},
]

# Case 3: empty — harusnya REJECT
chunks_empty = []

for label, chunks in [
    ("GOOD", chunks_good),
    ("BAD", chunks_bad),
    ("BORDERLINE", chunks_borderline),
    ("NOISY", chunks_noisy),
    ("EMPTY", chunks_empty)
]:
    result = engine.calculate_confidence(chunks)
    print(f"\n[{label}]")
    print(f" decision : {result['decision']}")
    print(f" score    : {result['confidence_score']}")
    print(f" signals  : {result['signals']}")