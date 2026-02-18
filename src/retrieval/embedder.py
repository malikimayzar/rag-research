import json
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import Optional
import time


@dataclass
class EmbeddingIndex:
    index: faiss.Index
    chunk_ids: list[str]
    chunks: list[dict]
    model_name: str
    dimension: int


def load_chunks(chunks_path: str) -> list[dict]:
    with open(chunks_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_dense_index(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> EmbeddingIndex:
    print(f"[EMBED] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print(f"[EMBED] Encoding {len(texts)} chunks...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True 
    )

    elapsed = time.time() - start
    print(f"[EMBED] Done in {elapsed:.1f}s — shape: {embeddings.shape}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    print(f"[FAISS] Index built — {index.ntotal} vectors, dim={dimension}")

    return EmbeddingIndex(
        index=index,
        chunk_ids=chunk_ids,
        chunks=chunks,
        model_name=model_name,
        dimension=dimension
    )


def save_index(emb_index: EmbeddingIndex, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Simpan FAISS index
    faiss.write_index(emb_index.index, str(out / "faiss.index"))

    # Simpan metadata
    meta = {
        "model_name": emb_index.model_name,
        "dimension": emb_index.dimension,
        "total_chunks": len(emb_index.chunk_ids),
        "chunk_ids": emb_index.chunk_ids,
        "chunks": emb_index.chunks
    }
    with open(out / "index_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Index saved → {output_dir}")


def load_index(index_dir: str) -> EmbeddingIndex:
    idx_path = Path(index_dir)

    faiss_index = faiss.read_index(str(idx_path / "faiss.index"))

    with open(idx_path / "index_meta.json", 'r') as f:
        meta = json.load(f)

    return EmbeddingIndex(
        index=faiss_index,
        chunk_ids=meta["chunk_ids"],
        chunks=meta["chunks"],
        model_name=meta["model_name"],
        dimension=meta["dimension"]
    )


def dense_search(
    query: str,
    emb_index: EmbeddingIndex,
    model: SentenceTransformer,
    top_k: int = 5
) -> list[dict]:
    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = emb_index.index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = emb_index.chunks[idx].copy()
        chunk["retrieval_score"] = float(score)
        chunk["retrieval_rank"] = len(results) + 1
        chunk["retrieval_method"] = "dense"
        results.append(chunk)

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src/ingestion")

    chunks = load_chunks("data/processed/chunks_512_64.json")
    print(f"Loaded {len(chunks)} chunks")

    # Build index
    emb_index = build_dense_index(
        chunks,
        model_name="all-MiniLM-L6-v2"
    )

    # Simpan
    save_index(emb_index, "data/processed/index_minilm")

    # Quick search test
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = dense_search("what is chunking in RAG?", emb_index, model, top_k=3)

    print(f"\n=== Search Results ===")
    for r in results:
        print(f"\nRank {r['retrieval_rank']} | Score: {r['retrieval_score']:.4f}")
        print(f"Chunk: {r['chunk_id']}")
        print(f"Text: {r['text'][:200]}...")