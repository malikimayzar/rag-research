from __future__ import annotations

import logging
import json
import os
import time
from src.api.config import settings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from src.controller.confidence_engine import ConfidenceEngine
from groq import Groq, AuthenticationError, GroqError

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("rag.generator")
load_dotenv()

DEFAULT_MODEL     = "llama-3.3-70b-versatile"
DEFAULT_MAX_CHARS = 3000
DEFAULT_TEMP      = 0.1
DEFAULT_TOP_K     = 10

SYSTEM_PROMPT = (
    "You are a Technical Research Assistant. Answer the query using ONLY the provided CONTEXT.\n\n"
    "STRICT RULES:\n"
    "1. Grounding: Every claim must be directly supported by at least one [Source N].\n"
    "2. Bounded Reasoning: You may infer only if multiple sources support the logic. Do NOT introduce external facts.\n"
    "3. Format: Respond ONLY in valid JSON. No preamble, no markdown, no backticks.\n\n"
    "JSON SCHEMA:\n"
    "{\n"
    "  \"answer\": \"your technical answer here\",\n"
    "  \"status\": \"ANSWERED\" | \"INSUFFICIENT_CONTEXT\",\n"
    "  \"confidence_score\": 0.0 to 1.0,\n"
    "  \"supporting_sources\": [\"[Source 1]\", \"[Source 2]\"]\n"
    "}"
)


@dataclass
class RAGResponse:
    query:                    str
    answer:                   str
    retrieved_chunks:         list
    retrieval_method:         str
    context_used:             str
    latency_context_build_ms: float
    latency_generation_ms:    float
    model:                    str
    num_chunks_retrieved:     int
    num_chunks_used:          int
    # FIX P2: Expose parsed LLM metadata so callers can use them
    status:                   str   = "UNKNOWN"
    confidence_score:         float = 0.0
    supporting_sources:       list  = None

    def __post_init__(self):
        if self.supporting_sources is None:
            self.supporting_sources = []


# ---------------------------------------------------------------------------
# Helper: safe attribute access for RetrievalResult OR dict
# ---------------------------------------------------------------------------

def _get_attr(chunk, key: str, default=None):
    """
    FIX P1 (CRITICAL): Unified accessor that works for both
    RetrievalResult objects and plain dicts.
    Never call .get() or ['key'] directly on chunks.
    """
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _get_score(chunk) -> float:
    """Single authoritative accessor. retrieval_score is canonical."""
    if isinstance(chunk, dict):
        return float(chunk.get("retrieval_score", chunk.get("score", 0.0)))
    return float(
        getattr(chunk, "retrieval_score", None)
        or getattr(chunk, "score", 0.0)
        or 0.0
    )


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def filter_chunks_by_diversity(chunks: list, max_per_doc: int = 2) -> list:
    if not chunks:
        return chunks

    # Guard: don't apply diversity to small result sets
    if len(chunks) <= 5:
        logger.info(
            f"[DIVERSITY_FILTER] Skipped (only {len(chunks)} chunks — below guard threshold)"
        )
        return chunks

    seen_docs: dict[str, int] = {}
    filtered = []

    for chunk in chunks:
        doc_id = _get_attr(chunk, "doc_id", "unknown")

        if doc_id not in seen_docs:
            seen_docs[doc_id] = 0

        if seen_docs[doc_id] < max_per_doc:
            filtered.append(chunk)
            seen_docs[doc_id] += 1

    logger.info(
        f"[DIVERSITY_FILTER] docs_count={len(seen_docs)} | "
        f"chunks_before={len(chunks)} | chunks_after={len(filtered)}"
    )
    return filtered

def build_context(chunks: list, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text     = chunk.strip()
            chunk_id = f"idx_{i}"
            doc_id   = "manual"
            section  = "General"
        else:
            # FIX P1: use _get_attr, never dict access
            text     = (_get_attr(chunk, "text", "") or "").strip()
            chunk_id = _get_attr(chunk, "chunk_id", f"c_{i}")
            doc_id   = _get_attr(chunk, "doc_id", "unknown")
            # FIX P4: restore metadata for traceability
            metadata = _get_attr(chunk, "metadata", {}) or {}
            section  = metadata.get("section", "General")

        # FIX P4: full header — doc_id + chunk_id + section all present
        header = f"[Source {i+1} | {doc_id} | {chunk_id} | Section: {section}]"
        block  = f"{header}\n{text}"

        if total_chars + len(block) > max_chars:
            logger.info(f"[CONTEXT_LIMIT] Reached max_chars={max_chars} at chunk {i+1}")
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n".join(context_parts)


def filter_chunks_by_score(chunks: list, min_score: float = 0.0) -> list:
    """FIX P5: Uses _get_score() — single consistent score source."""
    if not chunks:
        return chunks

    scores = [_get_score(c) for c in chunks]

    if scores[0] <= -3.0:
        return []

    GAP_THRESHOLD = settings.score_gap_threshold
    selected = [chunks[0]]
    for i in range(1, len(chunks)):
        gap = scores[i - 1] - scores[i]
        if scores[i] < -3.0 or gap > GAP_THRESHOLD:
            break
        selected.append(chunks[i])

    return selected


# ---------------------------------------------------------------------------
# LLM output parser
# ---------------------------------------------------------------------------
def _parse_llm_output(raw: str) -> tuple[str, str, float, list]:
    try:
        parsed = json.loads(raw)
        answer   = parsed.get("answer", "").strip()
        status   = parsed.get("status", "UNKNOWN")
        conf     = float(parsed.get("confidence_score", 0.0))
        sources  = parsed.get("supporting_sources", [])
        if not answer:
            raise ValueError("Empty answer field in LLM JSON")
        return answer, status, conf, sources
    except Exception as e:
        logger.warning(f"[PARSE_ERROR] LLM output is not valid JSON: {e} | raw='{raw[:200]}'")
        return raw, "PARSE_ERROR", 0.0, []


# ---------------------------------------------------------------------------
# TASK 2: Retrieval Hit Guard
# ---------------------------------------------------------------------------
def _make_abstain_response(
    query: str,
    chunks: list,
    model: str,
    reason: str,
) -> RAGResponse:
    logger.warning(f"[ABSTAIN] {reason} | query='{query[:60]}'")
    return RAGResponse(
        query=query,
        answer="INSUFFICIENT_CONTEXT",
        retrieved_chunks=chunks,
        retrieval_method="abstained",
        context_used="",
        latency_context_build_ms=0.0,
        latency_generation_ms=0.0,
        model=model,
        num_chunks_retrieved=len(chunks),
        num_chunks_used=0,
        status="INSUFFICIENT_CONTEXT",
        confidence_score=0.0,
        supporting_sources=[],
    )


def check_retrieval_confidence(
    chunks: list,
    source_chunk_id: Optional[str],
    top_k_scores: list[float],
    min_top1_score: float = 0.0,
) -> tuple[bool, str]:
    if not chunks:
        return False, "chunks kosong — tidak ada kandidat untuk generation"
    if source_chunk_id:
        retrieved_ids = [_get_attr(c, "chunk_id", "") for c in chunks]
        is_hit = any(
            rid == source_chunk_id or rid.startswith(source_chunk_id + "_sub")
            for rid in retrieved_ids
        )
        if not is_hit:
            return False, (
                f"retrieval_miss: source_chunk_id='{source_chunk_id}' "
                f"tidak ditemukan di {len(retrieved_ids)} retrieved chunks — "
                f"generation akan hallucinate, pipeline abstain"
            )

    # Kondisi 3: Top-1 score terlalu rendah → retriever tidak percaya diri
    if top_k_scores and top_k_scores[0] <= min_top1_score:
        return False, (
            f"top1_score={top_k_scores[0]:.4f} ≤ threshold={min_top1_score} — "
            f"retrieval tidak yakin, pipeline abstain"
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY tidak ditemukan.")
        self.client = Groq(api_key=key)
        self.model  = model
        logger.info(f"[GroqGenerator] Ready. Model: {model}")
        self._confidence = ConfidenceEngine()
        
    def generate(
        self,
        query:           str,
        chunks:          list,
        max_tokens:      Optional[int] = None,
        temperature:     float = 0.0,
        # TASK 2: parameter baru — opsional, tidak breaking existing callers
        source_chunk_id: Optional[str] = None,
        min_top1_score:  float         = -3.0,
    ) -> RAGResponse:
        t0_total = time.time()
        num_chunks_retrieved = len(chunks)
        top_k_scores = [_get_score(c) for c in chunks[:5]]

        should_proceed, reason = check_retrieval_confidence(
            chunks=chunks,
            source_chunk_id=source_chunk_id,
            top_k_scores=top_k_scores,
            min_top1_score=min_top1_score,
        )

        if not should_proceed:
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason=reason,
            )

        # ------------------------------------------------------------------
        # Ab sini normal path — retrieval dianggaif filtered_chunks:p cukup terpercaya
        # ------------------------------------------------------------------
        if not chunks:
            return _make_abstain_response(
                query=query, chunks=[], model=self.model,
                reason="chunks kosong setelah guard (defensive)",
            )

        # FIX P3: Guarded diversity filter
        score_filtered = filter_chunks_by_score(chunks)
        filtered_chunks = filter_chunks_by_diversity(score_filtered, max_per_doc=2)
        num_chunks_used = len(filtered_chunks)

        # FIX P5: Use _get_score() — no ambiguous field names
        logger.info(
            f"[RETRIEVAL_QUALITY] query='{query[:50]}...' | "
            f"num_retrieved={num_chunks_retrieved} | num_after_diversity={num_chunks_used} | "
            f"top_5_scores={[round(s, 4) for s in top_k_scores]}"
        )

        # FIX P1 + P4: Object-safe context building via build_context()
        context = build_context(filtered_chunks, max_chars=DEFAULT_MAX_CHARS)
        t1 = time.time()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
        ]

        effective_max_tokens = max_tokens if max_tokens is not None else settings.generation_max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )
        except AuthenticationError as exc:
            raise ValueError(
                "Invalid GROQ_API_KEY atau token tidak valid. "
                "Set environment variable GROQ_API_KEY dengan API key yang benar."
            ) from exc
        except GroqError as exc:
            raise RuntimeError(
                f"Groq generation failed: {exc}. "
                "Periksa koneksi API dan kunci Anda."
            ) from exc

        raw_answer = response.choices[0].message.content.strip()
        t2 = time.time()

        # FIX P2: Parse JSON output — enforce the schema we promised the LLM
        answer, status, confidence, sources = _parse_llm_output(raw_answer)

        # FIX P6: Use parsed status instead of fragile string matching
        if status == "INSUFFICIENT_CONTEXT":
            answer = "I'm sorry, I couldn't find the answer in the provided context."

        latency_context_ms = round((t1 - t0_total) * 1000, 2)
        latency_gen_ms     = round((t2 - t1) * 1000, 2)
        latency_total_ms   = round((t2 - t0_total) * 1000, 2)

        logger.info(
            f"[GENERATION_LATENCY] query='{query[:50]}...' | "
            f"context_build_ms={latency_context_ms} | "
            f"generation_ms={latency_gen_ms} | "
            f"total_ms={latency_total_ms} | "
            f"status={status} | confidence={confidence:.2f}"
        )

        retrieval_method = _get_attr(chunks[0], "retrieval_method", "hybrid_rrf_baseline")

        # CONFIDENCE: Compute based on actual retrieval strength
        confidence_result = self._confidence.calculate_confidence(filtered_chunks)
        final_confidence = confidence_result["confidence_score"]

        if confidence_result["decision"] == "REJECT":
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason=f"confidence_engine REJECT | score={final_confidence:.3f}",
            )
        
        logger.info(
            f"[CONFIDENCE_CALC] score={final_confidence:.2f} | "
            f"decision={confidence_result['decision']}"
            f"signals={confidence_result['signals']}"
        )
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=filtered_chunks,
            retrieval_method=retrieval_method,
            context_used=context,
            latency_context_build_ms=latency_context_ms,
            latency_generation_ms=latency_gen_ms,
            model=self.model,
            num_chunks_retrieved=num_chunks_retrieved,
            num_chunks_used=num_chunks_used,
            status=status,
            confidence_score=final_confidence,  # Use realistic confidence
            supporting_sources=sources,
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def generate(query: str, chunks: list, **kwargs) -> RAGResponse:
    gen = GroqGenerator(model=kwargs.get("model_name", DEFAULT_MODEL))
    return gen.generate(query, chunks)


def save_response(response: RAGResponse, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = asdict(response)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"[Saved] -> {output_path}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    store = QdrantVectorStore()
    try:
        retriever = MasterHybridRetriever(vector_store=store)
        gen = GroqGenerator()

        queries = [
            "What is Retrieval-Augmented Generation?",
            "How does hybrid search improve RAG performance?",
            "What are the limitations of large language models?",
        ]

        print("\n" + "=" * 80)
        print("[GENERATOR TEST] Object-safe + JSON-parsed generation")
        print("=" * 80)

        for q in queries:
            chunks = retriever.search(q, top_k=10)
            resp = gen.generate(q, chunks)
            print(f"\nQ: {q}")
            print(f"Chunks Retrieved: {resp.num_chunks_retrieved} | Used: {resp.num_chunks_used}")
            print(f"Status: {resp.status} | Confidence: {resp.confidence_score:.2f}")
            print(f"A: {resp.answer[:200]}...")
            print(f"Sources: {resp.supporting_sources}")
            print(f"Context Build: {resp.latency_context_build_ms}ms | Generation: {resp.latency_generation_ms}ms")
            print("-" * 80)
    finally:
        store.close()