#!/usr/bin/env python3
"""
EVALUATION & CONTROL SYSTEM
Metrik nyata untuk diagnosis failure mode RAG pipeline.

Jawaban dari: "Sistem gue gagal itu karena apa?"
  EM            = apakah jawaban tepat sama dengan ground truth?
  Retrieval Hit = apakah source chunk yang bener masuk ke retrieved set?
  FAR           = apakah model jawab tapi retrieval sebenernya miss? (hallucination proxy)
  Ref Ratio     = berapa banyak reference chunk mencemari context?

Target:
  EM            > 0.60
  Retrieval Hit > 0.75
  FAR           < 0.05   ← ini yang paling penting
  Avg Latency   < 3000ms
"""

import logging
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Test queries — diverse across categories
# Kalau mau pakai ground truth dari ground_truth_qa.json, set USE_GT_FILE = True
# ---------------------------------------------------------------------------
USE_GT_FILE    = True   # True = pakai data/processed/ground_truth_qa.json
GT_FILE_PATH   = "data/processed/ground_truth_qa.json"
MAX_GT_QUERIES = 20     # batasi agar tidak terlalu lama

# Fallback jika GT file tidak ada
FALLBACK_QUERIES = [
    # (query, category, ground_truth_or_None, source_chunk_id_or_None)
    ("What is Retrieval-Augmented Generation?",              "definition",  None, None),
    ("What is semantic chunking?",                           "definition",  None, None),
    ("What are attention mechanisms?",                       "definition",  None, None),
    ("What is RAG?",                                         "definition",  None, None),
    ("Define transformer architecture",                      "definition",  None, None),
    ("How does hybrid search work?",                         "how",         None, None),
    ("Why is semantic chunking important?",                  "why",         None, None),
    ("How does BM25 differ from dense retrieval?",           "how",         None, None),
    ("Why use vector databases for RAG?",                    "why",         None, None),
    ("How does HNSW indexing work?",                         "how",         None, None),
    ("Compare BM25 and dense retrieval",                     "comparison",  None, None),
    ("What's the difference between sparse and dense embeddings?", "comparison", None, None),
    ("How does semantic chunking compare to fixed-size chunking?", "comparison", None, None),
    ("What are pros and cons of RAG?",                       "comparison",  None, None),
    ("Difference between retrieval and generation?",         "comparison",  None, None),
    ("What are limitations of large language models?",       "edge_case",   None, None),
    ("How does hybrid retrieval improve RAG performance?",   "edge_case",   None, None),
    ("What is the relationship between chunking strategy and retrieval?", "edge_case", None, None),
    ("Explain the role of reranking in retrieval systems",   "edge_case",   None, None),
    ("How do you evaluate RAG system quality?",              "edge_case",   None, None),
]

# ---------------------------------------------------------------------------
# Constants — harus sinkron dengan run_single_query.py
# ---------------------------------------------------------------------------
BLOCKED_SECTIONS  = {"references", "bibliography"}
MIN_CHUNK_LENGTH  = 80


# ---------------------------------------------------------------------------
# Dataclass — structured result per query
# ---------------------------------------------------------------------------
@dataclass
class QueryEvaluation:
    query:                  str
    category:               str
    ground_truth:           Optional[str]
    source_chunk_id:        Optional[str]

    # --- Retrieval metrics ---
    retrieval_score_top1:   float
    retrieved_chunk_ids:    list = field(default_factory=list)
    num_chunks_retrieved:   int  = 0
    num_chunks_used:        int  = 0

    # --- Answer ---
    answer_text:            str  = ""
    answer_length:          int  = 0
    answer_status:          str  = ""

    # --- 4 Core Metrics ---
    exact_match:            Optional[bool] = None   # EM: pred == gt (string exact)
    retrieval_hit:          Optional[bool] = None   # source chunk masuk retrieved set?
    false_answer_rate:      Optional[bool] = None   # FAR: jawab tapi retrieval miss?
    ref_ratio:              float          = 0.0    # ratio reference di context

    # --- Latency ---
    latency_total_ms:       float = 0.0

    # --- Notes ---
    notes:                  str   = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_attr(chunk, key: str, default=None):
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)

def _get_section(chunk) -> str:
    meta = _get_attr(chunk, "metadata", {}) or {}
    return meta.get("section", "body").lower()

def _is_retrieval_hit(source_chunk_id: Optional[str], retrieved_ids: list) -> Optional[bool]:
    """
    Apakah source chunk yang benar masuk ke retrieved set?
    Juga check sub-chunk (uuid5 pattern: source_id + "_sub")
    """
    if not source_chunk_id:
        return None
    return any(
        rid == source_chunk_id or rid.startswith(source_chunk_id + "_sub")
        for rid in retrieved_ids
    )

def _compute_ref_ratio(chunks: list) -> float:
    """Berapa proporsi reference/bibliography di final context chunks."""
    if not chunks:
        return 0.0
    ref_count = sum(1 for c in chunks if _get_section(c) in BLOCKED_SECTIONS)
    return round(ref_count / len(chunks), 3)

def _exact_match(pred: str, gt: str) -> bool:
    """
    Strict exact match: strip + lowercase.
    Ini sengaja ketat — partial match bukan EM.
    Kalau EM rendah tapi answer masuk akal → pakai F1 di iteration berikutnya.
    """
    return pred.strip().lower() == gt.strip().lower()


# ---------------------------------------------------------------------------
# Core: evaluate single query
# ---------------------------------------------------------------------------
def evaluate_query(
    retriever,
    generator,
    query:           str,
    category:        str,
    ground_truth:    Optional[str] = None,
    source_chunk_id: Optional[str] = None,
    top_k:           int = 5,
) -> QueryEvaluation:
    """
    Pipeline:
      1. Retrieve top_k chunks
      2. Compute retrieval_hit (source chunk found?)
      3. Generate answer
      4. Compute EM, FAR, ref_ratio
    """
    t_start = time.time()

    try:
        # --- 1. Retrieval ---
        raw_chunks = retriever.search(query, top_k=top_k)
        chunks     = raw_chunks if raw_chunks else []

        if not chunks:
            return QueryEvaluation(
                query=query, category=category,
                ground_truth=ground_truth, source_chunk_id=source_chunk_id,
                retrieval_score_top1=0.0,
                answer_text="No chunks retrieved", answer_status="NO_RETRIEVAL",
                notes="Retrieval returned 0 chunks",
            )

        # Top-1 score (BM25/dense/RRF depends on retriever)
        top_score = _get_attr(chunks[0], "score", 0.0) or 0.0

        # Chunk IDs untuk hit detection
        retrieved_ids = [
            _get_attr(c, "chunk_id", "") for c in chunks
        ]

        # --- 2. Retrieval Hit ---
        r_hit = _is_retrieval_hit(source_chunk_id, retrieved_ids)

        # --- 3. Ref Ratio ---
        ref_ratio = _compute_ref_ratio(chunks)

        # --- 4. Generation ---
        resp = generator.generate(query, chunks)

        t_end      = time.time()
        latency_ms = round((t_end - t_start) * 1000, 2)

        # --- 5. Exact Match ---
        em = None
        if ground_truth:
            em = _exact_match(resp.answer, ground_truth)

        # --- 6. False Answer Rate (FAR) ---
        # Definisi: model menghasilkan jawaban (bukan INSUFFICIENT_CONTEXT)
        # TAPI retrieval miss terjadi → jawaban kemungkinan besar hallucination
        # FAR = True adalah sinyal paling berbahaya di RAG system
        answered = resp.status not in ("INSUFFICIENT_CONTEXT", "PARSE_ERROR", "ERROR")
        far: Optional[bool] = None
        if r_hit is not None:
            far = answered and (not r_hit)

        num_retrieved = len(chunks)
        num_used      = getattr(resp, "num_chunks_used", num_retrieved)

        return QueryEvaluation(
            query=query,
            category=category,
            ground_truth=ground_truth,
            source_chunk_id=source_chunk_id,
            retrieval_score_top1=round(float(top_score), 4),
            retrieved_chunk_ids=retrieved_ids,
            num_chunks_retrieved=num_retrieved,
            num_chunks_used=num_used,
            answer_text=resp.answer[:500],
            answer_length=len(resp.answer),
            answer_status=resp.status,
            exact_match=em,
            retrieval_hit=r_hit,
            false_answer_rate=far,
            ref_ratio=ref_ratio,
            latency_total_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"[ERROR] Query '{query[:60]}': {e}", exc_info=True)
        return QueryEvaluation(
            query=query, category=category,
            ground_truth=ground_truth, source_chunk_id=source_chunk_id,
            retrieval_score_top1=0.0,
            answer_text=f"ERROR: {e}",
            answer_status="ERROR",
            notes=f"Exception: {e}",
            latency_total_ms=round((time.time() - t_start) * 1000, 2),
        )


# ---------------------------------------------------------------------------
# Aggregate metrics computation
# ---------------------------------------------------------------------------
def compute_aggregate_metrics(results: list[QueryEvaluation]) -> dict:
    """
    Hitung 4 metrik utama dari semua hasil.
    Hanya menghitung dari query yang punya ground_truth / source_chunk_id.
    """
    n = len(results)
    if n == 0:
        return {}

    # EM — hanya query yang punya ground_truth
    em_results = [r for r in results if r.exact_match is not None]
    em_score   = sum(r.exact_match for r in em_results) / len(em_results) if em_results else None

    # Retrieval Hit — hanya query yang punya source_chunk_id
    rh_results = [r for r in results if r.retrieval_hit is not None]
    rh_score   = sum(r.retrieval_hit for r in rh_results) / len(rh_results) if rh_results else None

    # FAR — dari subset yang punya source_chunk_id
    far_results = [r for r in results if r.false_answer_rate is not None]
    far_score   = sum(r.false_answer_rate for r in far_results) / len(far_results) if far_results else None

    # Ref Ratio — semua query
    avg_ref_ratio = round(sum(r.ref_ratio for r in results) / n, 3)

    # Latency
    valid_latency = [r for r in results if r.latency_total_ms > 0]
    avg_latency   = round(sum(r.latency_total_ms for r in valid_latency) / len(valid_latency), 1) if valid_latency else 0

    # Kategori breakdown
    categories = sorted(set(r.category for r in results))
    cat_breakdown = {}
    for cat in categories:
        cat_rs = [r for r in results if r.category == cat]
        cat_breakdown[cat] = {
            "n":             len(cat_rs),
            "avg_score":     round(sum(r.retrieval_score_top1 for r in cat_rs) / len(cat_rs), 4),
            "avg_latency_ms":round(sum(r.latency_total_ms for r in cat_rs) / len(cat_rs), 1),
            "avg_ref_ratio": round(sum(r.ref_ratio for r in cat_rs) / len(cat_rs), 3),
            "em":            round(sum(r.exact_match for r in cat_rs if r.exact_match is not None)
                                   / max(1, len([r for r in cat_rs if r.exact_match is not None])), 3)
                             if any(r.exact_match is not None for r in cat_rs) else None,
        }

    return {
        "n_queries":       n,
        "em":              round(em_score, 3)  if em_score  is not None else None,
        "retrieval_hit":   round(rh_score, 3)  if rh_score  is not None else None,
        "far":             round(far_score, 3) if far_score is not None else None,
        "avg_ref_ratio":   avg_ref_ratio,
        "avg_latency_ms":  avg_latency,
        "em_n":            len(em_results),
        "rh_n":            len(rh_results),
        "far_n":           len(far_results),
        "category":        cat_breakdown,
    }


# ---------------------------------------------------------------------------
# Diagnosis: interpret aggregate metrics
# ---------------------------------------------------------------------------
def diagnose(metrics: dict) -> list[str]:
    """
    Baca metrik, output diagnosis actionable.
    Format: list of string yang langsung bisa di-print.
    """
    issues = []

    em  = metrics.get("em")
    rh  = metrics.get("retrieval_hit")
    far = metrics.get("far")
    ref = metrics.get("avg_ref_ratio", 0)
    lat = metrics.get("avg_latency_ms", 0)

    # EM
    if em is None:
        issues.append("⚠️  EM tidak bisa dihitung — ground_truth tidak tersedia di dataset")
    elif em < 0.5:
        issues.append(f"❌ EM={em:.2f} < 0.50 → jawaban sering salah")
        issues.append("   → Cek: apakah generator terlalu kreatif? temperature > 0.2?")
        issues.append("   → Cek: apakah context yang masuk memang mengandung jawaban?")
    elif em < 0.70:
        issues.append(f"⚠️  EM={em:.2f} — masih OK tapi ada ruang perbaikan")
    else:
        issues.append(f"✅ EM={em:.2f} — answer quality bagus")

    # Retrieval Hit
    if rh is None:
        issues.append("⚠️  Retrieval Hit tidak bisa dihitung — source_chunk_id tidak tersedia")
    elif rh < 0.70:
        issues.append(f"❌ Retrieval Hit={rh:.2f} < 0.70 → retriever sering miss source chunk")
        issues.append("   → Cek: apakah chunk ID di ground truth match dengan index?")
        issues.append("   → Cek: top_k terlalu kecil? Coba naikkan ke 10")
        issues.append("   → Cek: apakah BM25 weight perlu dinaikkan?")
    elif rh < 0.85:
        issues.append(f"⚠️  Retrieval Hit={rh:.2f} — bisa lebih baik")
    else:
        issues.append(f"✅ Retrieval Hit={rh:.2f} — retriever stabil")

    # FAR — ini yang paling critical
    if far is None:
        issues.append("⚠️  FAR tidak bisa dihitung — butuh source_chunk_id")
    elif far > 0.10:
        issues.append(f"🚨 FAR={far:.2f} > 0.10 → BAHAYA: model sering hallucinate")
        issues.append("   → Ini artinya: retrieval miss tapi model tetap jawab dengan percaya diri")
        issues.append("   → Fix: tambah confidence threshold — jika retrieval_hit=False, return INSUFFICIENT_CONTEXT")
        issues.append("   → Fix: perkuat self-reflection loop")
    elif far > 0.05:
        issues.append(f"⚠️  FAR={far:.2f} — ada hallucination risk, perlu ditekan")
    else:
        issues.append(f"✅ FAR={far:.2f} — hallucination rate terkontrol")

    # Ref Ratio
    if ref > 0.4:
        issues.append(f"⚠️  avg_ref_ratio={ref:.2f} > 0.40 — reference masih dominan di context")
        issues.append("   → Fix: turunkan MAX_REF_RATIO di run_single_query.py ke 0.3")
        issues.append("   → Fix: naikkan section penalty ke -3.0")
    elif ref > 0.2:
        issues.append(f"⚠️  avg_ref_ratio={ref:.2f} — masih acceptable, monitor terus")
    else:
        issues.append(f"✅ avg_ref_ratio={ref:.2f} — reference leakage terkontrol")

    # Latency
    if lat > 5000:
        issues.append(f"❌ Avg latency={lat:.0f}ms > 5000ms — terlalu lambat untuk production")
        issues.append("   → Fix: disable HyDE untuk factual queries")
        issues.append("   → Fix: reduce candidate_k dari 15 ke 10")
        issues.append("   → Fix: kurangi max_tokens generator untuk factual queries (100 → 80)")
    elif lat > 3000:
        issues.append(f"⚠️  Avg latency={lat:.0f}ms — borderline, perlu dioptimalkan")
    else:
        issues.append(f"✅ Avg latency={lat:.0f}ms — latency OK")

    return issues


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_evaluation(
    output_file: str = "results/eval_phase2_results.json",
    top_k: int = 5,
):
    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.generation.generator import GroqGenerator

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # --- Load test queries ---
    test_queries = []  # list of (query, category, ground_truth, source_chunk_id)

    if USE_GT_FILE and Path(GT_FILE_PATH).exists():
        logger.info(f"[EVAL] Loading ground truth dari {GT_FILE_PATH}")
        with open(GT_FILE_PATH) as f:
            gt_data = json.load(f)

        for item in gt_data[:MAX_GT_QUERIES]:
            test_queries.append((
                item["question"],
                item.get("category", "gt"),
                item.get("ground_truth") or item.get("answer"),
                item.get("source_chunk") or item.get("source_chunk_id"),
            ))
        logger.info(f"[EVAL] Loaded {len(test_queries)} queries dari GT file")
    else:
        logger.info("[EVAL] GT file tidak ditemukan — pakai FALLBACK_QUERIES")
        test_queries = [(q, cat, gt, src) for q, cat, gt, src in FALLBACK_QUERIES]

    n_total = len(test_queries)

    logger.info("\n" + "="*80)
    logger.info(f"[EVALUATION] {n_total} queries | top_k={top_k}")
    logger.info("="*80)

    store     = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator()

    results: list[QueryEvaluation] = []

    try:
        for i, (query, category, ground_truth, source_chunk_id) in enumerate(test_queries, 1):
            logger.info(f"\n[{i}/{n_total}] [{category.upper()}] {query[:65]}")
            if ground_truth:
                logger.info(f"   GT: {str(ground_truth)[:80]}")
            if source_chunk_id:
                logger.info(f"   Source: {source_chunk_id}")

            result = evaluate_query(
                retriever=retriever,
                generator=generator,
                query=query,
                category=category,
                ground_truth=ground_truth,
                source_chunk_id=source_chunk_id,
                top_k=top_k,
            )
            results.append(result)

            # Per-query log — langsung keliatan tanpa buka file
            em_str  = f"EM={'✅' if result.exact_match else ('❌' if result.exact_match is False else '—')}"
            rh_str  = f"RH={'✅' if result.retrieval_hit else ('❌' if result.retrieval_hit is False else '—')}"
            far_str = f"FAR={'🚨' if result.false_answer_rate else ('✅' if result.false_answer_rate is False else '—')}"
            logger.info(
                f"   {em_str} | {rh_str} | {far_str} | "
                f"ref_ratio={result.ref_ratio:.2f} | "
                f"score={result.retrieval_score_top1:.4f} | "
                f"latency={result.latency_total_ms:.0f}ms | "
                f"status={result.answer_status}"
            )
            logger.info(f"   Answer: {result.answer_text[:100]}...")

    finally:
        store.close()

    # --- Aggregate metrics ---
    metrics = compute_aggregate_metrics(results)
    issues  = diagnose(metrics)

    # --- Print summary ---
    logger.info("\n" + "="*80)
    logger.info("[AGGREGATE METRICS]")
    logger.info(f"  n_queries      = {metrics['n_queries']}")
    logger.info(f"  EM             = {metrics['em']}  (n={metrics['em_n']})")
    logger.info(f"  Retrieval Hit  = {metrics['retrieval_hit']}  (n={metrics['rh_n']})")
    logger.info(f"  FAR            = {metrics['far']}  (n={metrics['far_n']})")
    logger.info(f"  Avg Ref Ratio  = {metrics['avg_ref_ratio']}")
    logger.info(f"  Avg Latency    = {metrics['avg_latency_ms']}ms")
    logger.info("")
    logger.info("[TARGET]")
    logger.info("  EM            > 0.60")
    logger.info("  Retrieval Hit > 0.75")
    logger.info("  FAR           < 0.05  ← paling kritis")
    logger.info("  Avg Latency   < 3000ms")
    logger.info("")
    logger.info("[DIAGNOSIS]")
    for issue in issues:
        logger.info(f"  {issue}")
    logger.info("")

    # Category breakdown
    logger.info("[BREAKDOWN BY CATEGORY]")
    for cat, cstat in metrics.get("category", {}).items():
        logger.info(
            f"  {cat.upper():<12} n={cstat['n']} | "
            f"avg_score={cstat['avg_score']:.4f} | "
            f"avg_latency={cstat['avg_latency_ms']:.0f}ms | "
            f"avg_ref={cstat['avg_ref_ratio']:.2f} | "
            f"em={cstat['em']}"
        )
    logger.info("="*80)

    # --- Save ---
    output = {
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config":       {"top_k": top_k, "n_queries": n_total, "gt_file": GT_FILE_PATH if USE_GT_FILE else None},
        "metrics":      metrics,
        "diagnosis":    issues,
        "results": [
            {
                "query":                 r.query,
                "category":              r.category,
                "ground_truth":          r.ground_truth,
                "source_chunk_id":       r.source_chunk_id,
                "retrieval_score_top1":  r.retrieval_score_top1,
                "retrieved_chunk_ids":   r.retrieved_chunk_ids,
                "num_chunks_retrieved":  r.num_chunks_retrieved,
                "num_chunks_used":       r.num_chunks_used,
                "answer_text":           r.answer_text,
                "answer_length":         r.answer_length,
                "answer_status":         r.answer_status,
                # 4 core metrics
                "exact_match":           r.exact_match,
                "retrieval_hit":         r.retrieval_hit,
                "false_answer_rate":     r.false_answer_rate,
                "ref_ratio":             r.ref_ratio,
                "latency_total_ms":      r.latency_total_ms,
                "notes":                 r.notes,
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n[SAVED] → {output_file}")
    logger.info("[NEXT] Kalau FAR > 0.05:")
    logger.info("   1. Cek query mana yang FAR=True di results")
    logger.info("   2. Lihat retrieved_chunk_ids vs source_chunk_id")
    logger.info("   3. Itu retrieval miss — fix di BM25 weight atau top_k")
    logger.info("[NEXT] Kalau EM rendah tapi jawaban masuk akal:")
    logger.info("   → Tambah F1 token-level metric di iterasi berikutnya")
    logger.info("")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Evaluation — EM, Retrieval Hit, FAR, Ref Ratio")
    parser.add_argument("--output",  type=str, default="results/eval_phase2_results.json")
    parser.add_argument("--top_k",   type=int, default=5)
    parser.add_argument("--no-gt",   action="store_true", help="Force pakai FALLBACK_QUERIES")
    args = parser.parse_args()

    if args.no_gt:
        import builtins
        _orig = builtins.__dict__.get("USE_GT_FILE", True)
        USE_GT_FILE = False  # type: ignore[assignment]

    run_evaluation(output_file=args.output, top_k=args.top_k)