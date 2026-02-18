import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass

import ollama

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from embedder import load_index, dense_search
from bm25_retriever import load_bm25, bm25_search
from hybrid_retriever import hybrid_search

from sentence_transformers import SentenceTransformer


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved_chunks: list[dict]
    retrieval_method: str
    context_used: str
    latency_retrieval: float
    latency_generation: float
    model: str


def build_context(chunks: list[dict], max_chars: int = 2000) -> str:
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        text = chunk["text"].strip()
        header = f"[Source {i+1} | {chunk['chunk_id']}]"
        block = f"{header}\n{text}"

        if total_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    return f"""You are a precise research assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, explicitly state what is missing.
Do not hallucinate or add information beyond what is in the context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def generate(
    query: str,
    chunks: list[dict],
    model_name: str = "mistral",
    max_chars: int = 2000
) -> RAGResponse:
    t0 = time.time()
    context = build_context(chunks, max_chars)
    prompt = build_prompt(query, context)
    t1 = time.time()

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}  
    )

    t2 = time.time()
    answer = response["message"]["content"].strip()

    return RAGResponse(
        query=query,
        answer=answer,
        retrieved_chunks=chunks,
        retrieval_method=chunks[0].get("retrieval_method", "unknown") if chunks else "none",
        context_used=context,
        latency_retrieval=round(t1 - t0, 3),
        latency_generation=round(t2 - t1, 3),
        model=model_name
    )


def save_response(response: RAGResponse, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "query": response.query,
        "answer": response.answer,
        "retrieval_method": response.retrieval_method,
        "latency_retrieval": response.latency_retrieval,
        "latency_generation": response.latency_generation,
        "model": response.model,
        "retrieved_chunks": response.retrieved_chunks,
        "context_used": response.context_used
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("[LOAD] Loading retrieval components...")
    emb_index = load_index("data/processed/index_minilm")
    bm25, bm25_chunks = load_bm25("data/processed/index_bm25")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    test_queries = [
        "What is chunking and why does chunk size matter?",
        "How does hybrid retrieval combine BM25 and dense search?",
        "What metrics evaluate RAG system quality?",
        "What happens when the answer is not in the documents?"  # adversarial
    ]

    all_responses = []

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # Gunakan hybrid sebagai default
        results = hybrid_search(
            query, emb_index, bm25, bm25_chunks, model, top_k=3
        )
        chunks = results["hybrid"]

        if not chunks:
            print("[WARN] Tidak ada chunk yang di-retrieve")
            continue

        response = generate(query, chunks)

        print(f"Method : {response.retrieval_method}")
        print(f"Latency: retrieval={response.latency_retrieval}s | gen={response.latency_generation}s")
        print(f"\nAnswer:\n{response.answer}")

        all_responses.append({
            "query": response.query,
            "answer": response.answer,
            "retrieval_method": response.retrieval_method,
            "latency_retrieval": response.latency_retrieval,
            "latency_generation": response.latency_generation,
            "retrieved_chunks": response.retrieved_chunks
        })

    # Simpan semua
    with open("results/logs/generation_test.json", 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved → results/logs/generation_test.json")