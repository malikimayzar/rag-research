from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL     = "llama-3.3-70b-versatile"
DEFAULT_MAX_CHARS = 3000
DEFAULT_TEMP      = 0.1
DEFAULT_TOP_K     = 5

SYSTEM_PROMPT = (
    "You are a research assistant for academic NLP/AI papers.\n\n"
    "RULES:\n"
    "1. Answer using information stated or reasonably implied in the CONTEXT provided.\n"
    "2. If the context is clearly unrelated to the question, respond exactly: "
    "'The provided context does not contain enough information to answer this question.'\n"
    "3. Prefer context over prior knowledge, but you may use general knowledge "
    "to interpret or connect concepts mentioned in the context.\n"
    "4. Cite sources inline using [Source N] notation where applicable.\n"
    "5. Be concise — 1-3 sentences unless the question requires more detail."
)

@dataclass
class RAGResponse:
    query:              str
    answer:             str
    retrieved_chunks:   list
    retrieval_method:   str
    context_used:       str
    latency_context_build_ms: float
    latency_generation: float
    model:              str

def build_context(chunks: list, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text = chunk.strip()
            chunk_id = f"idx_{i}"
            doc_id = "manual"
        elif hasattr(chunk, "text"):
            text     = chunk.text.strip()
            chunk_id = getattr(chunk, "chunk_id", f"c_{i}")
            doc_id   = getattr(chunk, "doc_id", "unknown")
        elif isinstance(chunk, dict):
            text     = chunk.get("text", "").strip()
            chunk_id = chunk.get("chunk_id", f"c_{i}")
            doc_id   = chunk.get("doc_id", "unknown")
        else:
            continue

        header = f"[Source {i+1} | {doc_id} | {chunk_id}]"
        block  = f"{header}\n{text}"

        if total_chars + len(block) > max_chars:
            break

        context_parts.append(block)
        total_chars += len(block)
    return "\n\n".join(context_parts)

def filter_chunks_by_score(chunks: list, min_score: float = 0.0) -> list:
    """
    Gap-aware filtering:
    1. Kalau top score positif → ambil semua chunk sampai ada gap besar (> 2.0)
    2. Kalau top score negatif → fallback ke top-1 saja
    3. Minimum return: 1 chunk (tidak pernah return kosong)
    """
    if not chunks:
        return chunks

    scores = [
        c.get("retrieval_score", 0) if isinstance(c, dict)
        else getattr(c, "score", 0)
        for c in chunks
    ]

    # Kalau top score negatif → semua tidak relevan, return top-1
    if scores[0] <= 0:
        return [chunks[0]]

    # Top score positif → ambil sampai ketemu gap besar atau score drop ke negatif
    GAP_THRESHOLD = 2.0
    selected = [chunks[0]]
    for i in range(1, len(chunks)):
        gap = scores[i-1] - scores[i]
        if scores[i] < 0 or gap > GAP_THRESHOLD:
            break
        selected.append(chunks[i])

    return selected

def build_messages(query: str, context: str) -> list:
    user_content = "CONTEXT:\n" + context + "\n\nQUESTION:\n" + query + "\n\nANSWER (using ONLY the context above):"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

def build_prompt(query: str, context: str) -> str:
    return SYSTEM_PROMPT + "\n\nCONTEXT:\n" + context + "\n\nQUESTION:\n" + query + "\n\nANSWER:"

class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY tidak ditemukan.")
        self.client = Groq(api_key=key)
        self.model  = model
        print(f"[GroqGenerator] Ready. Model: {model}")

    def generate(
        self,
        query: str,
        chunks: list,
        max_chars: int = DEFAULT_MAX_CHARS,
        temperature: float = DEFAULT_TEMP,
    ) -> RAGResponse:
        t0 = time.time()
        before = len(chunks)
        chunks = filter_chunks_by_score(chunks)
        after = len(chunks)
        import logging; logging.getLogger(__name__).info(f'[Filter] before={before} after={after} dropped={before-after}')
        context  = build_context(chunks, max_chars)
        t1 = time.time()

        messages = build_messages(query, context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )

        t2 = time.time()
        answer = response.choices[0].message.content.strip()

        method = "hybrid_rerank"
        if chunks:
            first = chunks[0]
            if hasattr(first, "metadata"):
                method = first.metadata.get("retrieval_method", "hybrid_rerank")
            elif isinstance(first, dict):
                method = first.get("retrieval_method", "hybrid_rerank")

        formatted_chunks = []
        for c in chunks:
            if hasattr(c, "chunk_id"):
                formatted_chunks.append({
                    "chunk_id": c.chunk_id,
                    "doc_id":   c.doc_id,
                    "text":     c.text,
                    "score":    getattr(c, "score", 0.0)
                })
            elif isinstance(c, dict):
                formatted_chunks.append(c)
            else:
                formatted_chunks.append({"text": str(c)})

        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=formatted_chunks,
            retrieval_method=method,
            context_used=context,
            latency_context_build_ms=round((t1 - t0) * 1000, 3),
            latency_generation=round(t2 - t1, 3),
            model=self.model,
        )


def generate(query: str, chunks: list, **kwargs) -> RAGResponse:
    gen = GroqGenerator(model=kwargs.get("model_name", DEFAULT_MODEL))
    return gen.generate(query, chunks, max_chars=kwargs.get("max_chars", DEFAULT_MAX_CHARS))


def save_response(response: RAGResponse, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = asdict(response)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] -> {output_path}")

if __name__ == "__main__":
    from src.retrieval.qdrant_store import QdrantVectorStore, HybridRetriever

    store = QdrantVectorStore()
    retriever = HybridRetriever(vector_store=store)
    gen = GroqGenerator()

    queries = [
        "What is Retrieval-Augmented Generation?",
        "How does hybrid search improve RAG performance?",
        "What are the limitations of large language models?",
    ]

    for q in queries:
        chunks = retriever.search(q, k=5)
        resp = gen.generate(q, chunks)
        print(f"\nQ: {q}")
        print(f"A: {resp.answer}")
        print(f"Latency: {resp.latency_generation}s")
        print("-" * 60)