import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.controller.confidence_engine import ConfidenceEngine
from src.controller.policy_engine import PolicyEngine
from scripts import run_single_query as runner
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


def test_confidence_engine_prefers_reranker_score_when_available():
    engine = ConfidenceEngine()
    chunk = {"retrieval_score": 0.01, "rerank_score": 2.5}

    assert engine._get_chunk_score(chunk) == 2.5


def test_confidence_decision_uses_configured_thresholds():
    engine = ConfidenceEngine()

    assert engine._decision_for_score(0.4499) == "REJECT"
    assert engine._decision_for_score(0.45) == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.5555) == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.65) == "GENERATE"


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


def test_general_generation_policy_allows_json_response_budget():
    policy = PolicyEngine().generation_policy("general")

    assert policy["max_tokens"] >= 512


def test_factual_escalation_retries_once_and_emits_log_schema(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    context = "Attention lets a model focus on relevant tokens. " * 4
    initial_chunks = [{
        "chunk_id": "initial",
        "doc_id": "doc-1",
        "text": context,
        "score": 0.8,
        "retrieval_score": 0.8,
        "metadata": {"section": "body"},
    }]
    retry_chunks = [{
        **initial_chunks[0],
        "chunk_id": "retry",
    }]

    class FakeRetriever:
        def __init__(self):
            self.retry_calls = []
            self.vector_store = SimpleNamespace(
                search=lambda query, k: initial_chunks
            )

        def search(self, query, **kwargs):
            self.retry_calls.append(kwargs)
            return retry_chunks

    class FakeGenerator:
        def __init__(self):
            self.calls = 0
            self.responses = [
                SimpleNamespace(
                    status="INSUFFICIENT_CONTEXT",
                    answer="not enough context",
                    confidence_score=0.2,
                    supporting_sources=[],
                ),
                SimpleNamespace(
                    status="ANSWERED",
                    answer="Attention focuses on relevant tokens.",
                    confidence_score=0.8,
                    supporting_sources=["[Source 1]"],
                ),
            ]

        def generate(self, *args, **kwargs):
            self.calls += 1
            return self.responses.pop(0)

    class FakeConfidenceEngine:
        def calculate_confidence(self, chunks):
            return {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}}

    class FakePolicyEngine:
        def resolve(self, query):
            return {
                "query_type": "factual",
                "retrieval": {"top_k": 3, "use_hyde": False, "use_multi_query": False},
                "reference": {"allow_references": False, "max_ref_ratio": 0.0},
                "generation": {"max_tokens": 64, "temperature": 0.0},
            }

    retriever = FakeRetriever()
    generator = FakeGenerator()
    overlap_scores = iter([0.1, 0.8, 0.8])
    monkeypatch.setattr(runner, "QdrantVectorStore", lambda: object())
    monkeypatch.setattr(runner, "MasterHybridRetriever", lambda vector_store: retriever)
    monkeypatch.setattr(runner, "GroqGenerator", lambda model: generator)
    monkeypatch.setattr(runner, "Groq", lambda **kwargs: object())
    monkeypatch.setattr(runner, "ConfidenceEngine", FakeConfidenceEngine)
    monkeypatch.setattr(runner, "PolicyEngine", FakePolicyEngine)
    monkeypatch.setattr(
        runner,
        "compute_context_overlap",
        lambda answer, contexts: next(overlap_scores),
    )

    result = runner.run_single_query(
        query="What does attention do?",
        mode="baseline",
        save_output=False,
    )

    assert generator.calls == 2
    assert len(retriever.retry_calls) == 1
    assert retriever.retry_calls[0]["bm25_weight"] == 0.7
    assert retriever.retry_calls[0]["dense_weight"] == 0.3
    assert result["retry_triggered"] is True
    assert result["config"]["retries_used"] == 1

    required_log_fields = {
        "query_id",
        "query_type",
        "retrieval_scores",
        "reranker_scores",
        "chunks_used",
        "confidence_score",
        "failure_type",
        "answer_status",
        "latency_ms",
        "retry_triggered",
    }
    missing_fields = required_log_fields - result.keys()
    assert not missing_fields, f"Missing required log fields: {sorted(missing_fields)}"
    assert result["answer_status"] == "answered"
    assert result["chunks_used"] == ["retry"]
    assert result["retrieval_scores"] == [0.8]
    assert result["reranker_scores"] == [None]
    assert set(result["latency_ms"]) == {"retrieval", "reranker", "generation", "total"}
