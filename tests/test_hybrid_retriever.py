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

if __name__ == '__main__':
    test_query_classifier()
    print('OK')