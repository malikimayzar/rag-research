import asyncio

from src.api import main


class FakeRetriever:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeConfidenceEngine:
    def __init__(self, decisions):
        self.decisions = decisions

    def calculate_confidence(self, chunks):
        return self.decisions.pop(0)


class FakeGenerator:
    def generate(self, query, chunks):
        return type("Response", (), {
            "answer": "grounded answer",
            "retrieval_method": "hybrid_rrf_rerank",
            "model": "test-model",
        })()


async def _run_inline(func, *args, **kwargs):
    """Keep endpoint tests deterministic; threadpool behavior is FastAPI-owned."""
    return func(*args, **kwargs)


def test_api_retries_once_with_bm25_heavy_weights(monkeypatch):
    retriever = FakeRetriever([
        [],
        [{"chunk_id": "c1", "doc_id": "d1", "text": "complete grounded context.", "retrieval_score": 0.2, "metadata": {}}],
    ])
    confidence = FakeConfidenceEngine([
        {"confidence_score": 0.0, "decision": "REJECT", "signals": {}},
        {"confidence_score": 0.8, "decision": "GENERATE", "signals": {}},
    ])
    monkeypatch.setattr(main, "retriever", retriever)
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", confidence)
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.retry_triggered is True
    assert response.failure_type == "none"
    assert len(retriever.calls) == 2
    assert retriever.calls[1]["dense_weight"] == 0.3
    assert retriever.calls[1]["bm25_weight"] == 0.7
    assert len(events) == 1
    assert events[0]["query"] == "test query"
    assert events[0]["answer_status"] == "answered"
    assert events[0]["retry_triggered"] is True


def test_api_converts_pipeline_exception_to_safe_abstain(monkeypatch):
    monkeypatch.setattr(main, "retriever", FakeRetriever([RuntimeError("backend unavailable")]))
    monkeypatch.setattr(main, "generator", FakeGenerator())
    monkeypatch.setattr(main, "confidence_engine", FakeConfidenceEngine([]))
    monkeypatch.setattr(main, "run_in_threadpool", _run_inline)
    events = []
    monkeypatch.setattr(main, "_log_query_event", lambda **event: events.append(event))

    response = asyncio.run(main.generate_answer(main.GenerateRequest(query="test query")))

    assert response.decision == "REJECT"
    assert response.failure_type == "pipeline_error"
    assert response.retry_triggered is False
    assert events[0]["answer_status"] == "abstained"
    assert events[0]["failure_type"] == "pipeline_error"
