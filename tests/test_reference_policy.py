import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_single_query import limit_reference_chunks, enforce_reference_policy


def make_chunk(chunk_id, section, score=1.0):
    return {
        "chunk_id": chunk_id,
        "text": "dummy text",
        "score": score,
        "doc_id": "doc1",
        "metadata": {"section": section},
    }


def test_limit_reference_chunks_zero_budget():
    chunks = [
        make_chunk("c1", "references"),
        make_chunk("c2", "references"),
        make_chunk("c3", "body"),
    ]

    filtered, blocked = limit_reference_chunks(chunks, max_ref=0)
    assert all(chunk["metadata"]["section"] != "references" for chunk in filtered)
    assert len(blocked) == 2
    assert {"c1", "c2"} == blocked


def test_enforce_reference_policy_strict_block():
    chunks = [
        make_chunk("c1", "references", score=3.0),
        make_chunk("c2", "body", score=2.0),
        make_chunk("c3", "body", score=1.0),
    ]

    selected = enforce_reference_policy(chunks, allow_ref=False, max_ref_ratio=0.0, top_k=3)
    assert len(selected) == 2
    assert all(chunk["metadata"]["section"] != "references" for chunk in selected)
    assert [chunk["chunk_id"] for chunk in selected] == ["c2", "c3"]


def test_enforce_reference_policy_budget():
    chunks = [
        make_chunk("c1", "references", score=4.0),
        make_chunk("c2", "references", score=3.5),
        make_chunk("c3", "body", score=3.0),
        make_chunk("c4", "body", score=2.5),
        make_chunk("c5", "body", score=2.0),
    ]

    selected = enforce_reference_policy(chunks, allow_ref=True, max_ref_ratio=0.4, top_k=5)
    assert len(selected) == 5
    assert sum(1 for chunk in selected if chunk["metadata"]["section"] == "references") <= 2


def test_run_single_query_baseline_includes_max_ref_budget():
    import scripts.run_single_query as module

    class DummyVectorStore:
        def search(self, query, k):
            long_text = "dummy body text " * 10
            long_ref = "dummy reference text " * 10
            return [
                {
                    "chunk_id": "c1",
                    "text": long_text,
                    "retrieval_score": 0.9,
                    "doc_id": "doc1",
                    "metadata": {"section": "body"},
                },
                {
                    "chunk_id": "c2",
                    "text": long_ref,
                    "retrieval_score": 0.1,
                    "doc_id": "doc1",
                    "metadata": {"section": "references"},
                },
            ]

    class DummyMasterHybridRetriever:
        def __init__(self, vector_store, *args, **kwargs):
            self.vector_store = vector_store

    class DummyGenerator:
        def __init__(self, model):
            pass

        def generate(self, query, final_chunks_full, max_tokens, temperature, source_chunk_id, min_top1_score):
            class Response:
                answer = "dummy body text"
                status = "ANSWERED"
                confidence_score = 1.0
                supporting_sources = []

            return Response()

    class DummyGroq:
        def __init__(self, api_key=None):
            pass

    with patch.object(module, "QdrantVectorStore", DummyVectorStore), \
         patch.object(module, "MasterHybridRetriever", DummyMasterHybridRetriever), \
         patch.object(module, "GroqGenerator", DummyGenerator), \
         patch.object(module, "Groq", DummyGroq):
        output = module.run_single_query(
            "who wrote the paper?",
            top_k=5,
            mode="baseline",
            save_output=False,
        )

    assert output["config"]["max_ref_budget"] == 2