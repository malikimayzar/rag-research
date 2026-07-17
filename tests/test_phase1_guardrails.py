import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controller.confidence_engine import ConfidenceEngine
from scripts.run_single_query import should_skip_reranker
from src.generation.generator import (
    _make_abstain_response,
    sanity_check_answer,
    should_abort_before_generation,
)


def test_sanity_check_rejects_verbatim_reference_dump():
    answer = "This is a long answer that repeats the chunk content exactly and should not be accepted as a real answer because it is a copy of the retrieved text. " * 4
    chunk = {"text": "This is a long answer that repeats the chunk content exactly and should not be accepted as a real answer because it is a copy of the retrieved text."}

    ok, reason = sanity_check_answer(answer, [chunk])

    assert ok is False
    assert "reference_dump" in reason.lower()


def test_confidence_engine_rejects_non_finite_scores():
    engine = ConfidenceEngine()
    chunks = [{"retrieval_score": float("nan")}, {"retrieval_score": 0.1}]

    result = engine.calculate_confidence(chunks)

    assert result["decision"] == "REJECT"
    assert result["confidence_score"] == 0.0


def test_make_abstain_response_is_structured_and_safe():
    response = _make_abstain_response(
        query="What is the answer?",
        chunks=[{"chunk_id": "c1", "text": "context"}],
        model="test-model",
        reason="low confidence",
    )

    assert response.status == "INSUFFICIENT_CONTEXT"
    assert response.answer == "INSUFFICIENT_CONTEXT"
    assert response.confidence_score == 0.0
    assert response.supporting_sources == []


def test_should_abort_before_generation_only_for_truly_empty_or_very_low_confidence_context():
    assert should_abort_before_generation(confidence_score=0.2, decision="REJECT", has_chunks=True) is False
    assert should_abort_before_generation(confidence_score=0.05, decision="REJECT", has_chunks=True) is True
    assert should_abort_before_generation(confidence_score=0.0, decision="GENERATE", has_chunks=True) is False
    assert should_abort_before_generation(confidence_score=0.0, decision="REJECT", has_chunks=False) is True


def test_should_skip_reranker_for_small_or_weak_candidates():
    assert should_skip_reranker(candidate_count=2, top_gap=0.001, elapsed_ms=100.0) is True
    assert should_skip_reranker(candidate_count=6, top_gap=0.01, elapsed_ms=100.0) is True
    assert should_skip_reranker(candidate_count=6, top_gap=0.001, elapsed_ms=100.0) is False
