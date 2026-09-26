def _classify_query(query: str) -> int:
    words = len(query.strip().split())
    if words <= 5:
        return 3
    if words >= 15:
        return 7
    return 5

def test_query_classifier():
    assert _classify_query('what is RAG') == 3
    assert _classify_query('this example query contains significantly more than fifteen words to ensure long query behavior extra tokens') == 7
    assert _classify_query('explain the difference between dense and BM25 retrieval techniques in RAG systems') == 5


def test_search_returns_empty_results_when_no_candidates_remain(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None

    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 30)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: [])

    assert retriever.search("query with no candidates") == []


def test_search_abstains_instead_of_falling_back_to_fragment(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="general", allow_multi_query=False, allow_hyde=False,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None
    fragment = {
        "chunk_id": "fragment", "doc_id": "doc", "retrieval_score": 0.2,
        "text": "and generation must use this incomplete fragment",
        "metadata": {"section": "body"},
    }

    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 30)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [fragment])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: [fragment])

    assert retriever.search("query") == []

if __name__ == '__main__':
    test_query_classifier()
    print('OK')
