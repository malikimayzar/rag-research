import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    # Filter stop words sederhana
    stop_words = {
        'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to',
        'for', 'of', 'and', 'or', 'but', 'with', 'this', 'that',
        'are', 'was', 'be', 'as', 'by', 'from', 'we', 'they'
    }
    return [t for t in tokens if t not in stop_words and len(t) > 1]

def build_bm25_index(chunks: list[dict]) -> tuple:
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, chunks  

def save_bm25(bm25, chunks: list[dict], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "bm25.pkl", 'wb') as f:
        pickle.dump(bm25, f)

    with open(out / "bm25_chunks.json", 'w') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[OK] BM25 index saved → {output_dir}")


def load_bm25(index_dir: str) -> tuple:
    idx_path = Path(index_dir)

    with open(idx_path / "bm25.pkl", 'rb') as f:
        bm25 = pickle.load(f)

    with open(idx_path / "bm25_chunks.json", 'r') as f:
        chunks = json.load(f)

    return bm25, chunks

def bm25_search(
    query: str,
    bm25,
    chunks: list,
    top_k: int = 5
) -> list[dict]:
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] <= 0:
            continue

        # Konversi ke dict — handle semua kemungkinan tipe
        raw = chunks[idx]
        if isinstance(raw, dict):
            chunk = raw.copy()
        elif hasattr(raw, 'to_dict'):
            chunk = raw.to_dict()
        elif hasattr(raw, '__dataclass_fields__'):
            import dataclasses
            chunk = dataclasses.asdict(raw)
        else:
            # Last resort — manual extract field yang diketahui
            chunk = {
                "chunk_id": getattr(raw, "chunk_id", str(idx)),
                "doc_id": getattr(raw, "doc_id", ""),
                "text": getattr(raw, "text", ""),
                "tier": getattr(raw, "tier", 0),
                "chunk_index": getattr(raw, "chunk_index", idx),
                "total_chunks": getattr(raw, "total_chunks", 0),
                "start_char": getattr(raw, "start_char", 0),
                "end_char": getattr(raw, "end_char", 0),
                "metadata": getattr(raw, "metadata", {}),
            }

        chunk["retrieval_score"] = float(scores[idx])
        chunk["retrieval_rank"] = rank + 1
        chunk["retrieval_method"] = "bm25"
        results.append(chunk)

    return results

if __name__ == "__main__":
    with open("data/processed/chunks_512_64.json", 'r') as f:
        chunks = json.load(f)

    bm25, _ = build_bm25_index(chunks)
    save_bm25(bm25, chunks, "data/processed/index_bm25")

    # Test
    results = bm25_search("chunking overlap strategy", bm25, chunks, top_k=3)
    print("\n=== BM25 Search Results ===")
    for r in results:
        print(f"\nRank {r['retrieval_rank']} | Score: {r['retrieval_score']:.4f}")
        print(f"Text: {r['text'][:200]}...")