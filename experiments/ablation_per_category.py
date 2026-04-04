"""
ablation_per_category.py — Sprint 2 Per-Category Ablation
==========================================================
3 methods × 4 categories:
  Methods:  bm25_only | hybrid_rrf | hybrid_mq_hyde_rerank
  Category: paraphrase | multihop | adversarial | lexical

Output: results/metrics/ablation_per_category.json + leaderboard
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
METHODS = ["bm25_only", "hybrid_rrf", "hybrid_mq_hyde_rerank"]

CATEGORIES = {
    "paraphrase":  "data/processed/paraphrase_queries.json",
    "multihop":    "data/processed/multihop_queries.json",
    "adversarial": "data/processed/adversarial_queries.json",
    "lexical":     "data/processed/ground_truth_qa.json",
}

LEXICAL_LIMIT  = 20   # ambil 20 dari 55
OUTPUT_PATH    = "results/metrics/ablation_per_category.json"
TOP_K          = 10
MAX_RERANK_CHARS = 600

MULTI_QUERY_PROMPT = """Generate 2 alternative search queries for the following question.
Return ONLY the queries, one per line, no numbering, no explanation.

Original query: {query}

Alternative queries:"""

HYDE_PROMPT = """Write a short hypothetical passage (2-3 sentences) that would directly answer this question.
Be specific and technical. Write as if you are an expert answering from a research paper.

Question: {query}

Hypothetical passage:"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_context_overlap(answer: str, contexts: list[str]) -> float:
    if not answer or not contexts:
        return 0.0
    stopwords = {"the","a","an","is","are","was","were","in","on","at","to",
                 "for","of","and","or","but","it","this","that","with","as",
                 "by","from","be","has","have","not","does","do","its"}
    answer_words = set(re.sub(r'[^\w\s]', '', answer.lower()).split()) - stopwords
    context_words = set(re.sub(r'[^\w\s]', '', " ".join(contexts).lower()).split())
    if not answer_words:
        return 0.0
    return round(len(answer_words & context_words) / len(answer_words), 4)


def is_hit(retrieved: list, source_chunk_id: str) -> bool:
    """
    Hit jika:
    - exact match: cid == source_chunk_id
    - parent match: tier_rs_0002_sub0 matches tier_rs_0002
    - sibling match: tier_rs_0002_sub0 matches tier_rs_0002_sub1
    """
    base = source_chunk_id.rsplit("_sub", 1)[0]
    for r in retrieved:
        cid = r.get("chunk_id","") if isinstance(r,dict) else getattr(r,"chunk_id","")
        cid_base = cid.rsplit("_sub", 1)[0]
        if cid == source_chunk_id:          # exact
            return True
        if cid_base == base:                # sibling / parent
            return True
        if cid == base:                     # retrieved parent, source is sub
            return True
        if source_chunk_id == cid_base:     # retrieved sub, source is parent
            return True
    return False


def is_hit_multihop(retrieved: list, item: dict) -> bool:
    """Untuk multihop: hit jika SALAH SATU source chunk ditemukan."""
    hit1 = is_hit(retrieved, item.get("source_chunk", ""))
    hit2 = is_hit(retrieved, item.get("source_chunk_2", "")) if item.get("source_chunk_2") else True
    return hit1 or hit2


def both_hit_multihop(retrieved: list, item: dict) -> bool:
    """Strict: hit jika KEDUA source chunk ditemukan."""
    hit1 = is_hit(retrieved, item.get("source_chunk", ""))
    hit2 = is_hit(retrieved, item.get("source_chunk_2", "")) if item.get("source_chunk_2") else True
    return hit1 and hit2


def mrr_score(retrieved: list, source_chunk_id: str) -> float:
    base = source_chunk_id.rsplit("_sub", 1)[0]
    for i, r in enumerate(retrieved):
        cid = r.get("chunk_id","") if isinstance(r,dict) else getattr(r,"chunk_id","")
        cid_base = cid.rsplit("_sub", 1)[0]
        if cid == source_chunk_id or cid_base == base:
            return 1.0 / (i + 1)
    return 0.0


def best_window(text: str, query: str, window: int = MAX_RERANK_CHARS) -> str:
    if len(text) <= window:
        return text
    words = query.lower().split()
    best_start, best_score = 0, -1
    step = window // 2
    for start in range(0, max(1, len(text) - window + 1), step):
        snippet = text[start:start + window].lower()
        score = sum(snippet.count(w) for w in words)
        if score > best_score:
            best_score, best_start = score, start
    return text[best_start:best_start + window]


# ── Components ─────────────────────────────────────────────────────────────────

_components = None

def get_components():
    global _components
    if _components is None:
        from src.retrieval.model_registry import ModelRegistry
        from src.retrieval.qdrant_store import QdrantVectorStore
        from src.retrieval.hybrid_retriever import MasterHybridRetriever
        from src.generation.generator import GroqGenerator
        from groq import Groq
        import numpy as np

        registry  = ModelRegistry.get()
        store     = QdrantVectorStore()
        retriever = MasterHybridRetriever(vector_store=store)
        generator = GroqGenerator(model="llama-3.3-70b-versatile")
        groq      = Groq(api_key=os.getenv("GROQ_API_KEY"))
        _components = (store, retriever, generator, groq)
        print("[INIT] All components loaded.\n")
    return _components


# ── Retrieval per method ───────────────────────────────────────────────────────

def retrieve(query: str, method: str, store, retriever, groq, top_k: int) -> tuple[list, float]:
    import numpy as np

    t0 = time.time()

    if method == "bm25_only":
        scores  = retriever._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [{"chunk_id": retriever._chunks[i]["chunk_id"],
                    "doc_id":   retriever._chunks[i]["doc_id"],
                    "text":     retriever._chunks[i]["text"],
                    "score":    float(scores[i])}
                   for i in top_idx if scores[i] > 0]

    elif method == "hybrid_rrf":
        candidate_k = min(top_k * 4, 50)
        dense_hits  = [store.search(query, k=candidate_k)]
        bm25_scores = retriever._bm25.get_scores(query.lower().split())
        top_idx     = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_hits   = [[{"chunk_id": retriever._chunks[i]["chunk_id"],
                         "doc_id":   retriever._chunks[i]["doc_id"],
                         "text":     retriever._chunks[i]["text"],
                         "score":    float(bm25_scores[i])}
                        for i in top_idx if bm25_scores[i] > 0]]
        fused   = retriever._rrf_fuse(dense_hits, bm25_hits, candidate_k)
        results = [{"chunk_id": r.chunk_id if hasattr(r,"chunk_id") else r["chunk_id"],
                    "doc_id":   r.doc_id   if hasattr(r,"doc_id")   else r["doc_id"],
                    "text":     r.text     if hasattr(r,"text")     else r["text"],
                    "score":    r.score    if hasattr(r,"score")    else r["score"]}
                   for r in fused[:top_k]]

    elif method == "hybrid_mq_hyde_rerank":
        candidate_k = min(top_k * 4, 50)

        def call_groq(prompt, temp, max_tok):
            r = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                temperature=temp, max_tokens=max_tok,
            )
            return r.choices[0].message.content.strip()

        queries, hyde_text = [query], None
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_mq   = ex.submit(call_groq, MULTI_QUERY_PROMPT.format(query=query), 0.7, 100)
            f_hyde = ex.submit(call_groq, HYDE_PROMPT.format(query=query), 0.5, 150)
        try:
            alts    = [q.strip() for q in f_mq.result(timeout=5).split("\n") if q.strip()][:2]
            queries = [query] + alts
        except Exception:
            pass
        try:
            hyde_text = f_hyde.result(timeout=5)
        except Exception:
            pass

        all_dense, all_bm25 = [], []
        for q in queries:
            all_dense.append(store.search(q, k=candidate_k))
            sc  = retriever._bm25.get_scores(q.lower().split())
            ti  = np.argsort(sc)[::-1][:candidate_k]
            all_bm25.append([{"chunk_id": retriever._chunks[i]["chunk_id"],
                               "doc_id":   retriever._chunks[i]["doc_id"],
                               "text":     retriever._chunks[i]["text"],
                               "score":    float(sc[i])}
                              for i in ti if sc[i] > 0])
        if hyde_text:
            all_dense.append(store.search(hyde_text, k=candidate_k))

        fused      = retriever._rrf_fuse(all_dense, all_bm25, candidate_k)
        candidates = fused[:10]
        pairs      = [[query, best_window(
                            r.text if hasattr(r,"text") else r["text"], query)]
                      for r in candidates]
        scores_rk  = retriever.reranker.predict(pairs)
        for i, r in enumerate(candidates):
            s = float(scores_rk[i])
            if hasattr(r,"score"): r.score = s
            else: r["score"] = s
        candidates = sorted(candidates,
                            key=lambda x: x.score if hasattr(x,"score") else x["score"],
                            reverse=True)
        results = [{"chunk_id": r.chunk_id if hasattr(r,"chunk_id") else r["chunk_id"],
                    "doc_id":   r.doc_id   if hasattr(r,"doc_id")   else r["doc_id"],
                    "text":     r.text     if hasattr(r,"text")     else r["text"],
                    "score":    r.score    if hasattr(r,"score")    else r["score"]}
                   for r in candidates[:top_k]]
    else:
        raise ValueError(f"Unknown method: {method}")

    return results, round((time.time() - t0) * 1000, 2)


# ── Per-sample evaluation ──────────────────────────────────────────────────────

def evaluate_sample(item: dict, method: str, category: str,
                    store, retriever, generator, groq,
                    top_k: int = TOP_K) -> dict:
    from src.generation.generator import build_context

    query        = item["question"]
    source_chunk = item.get("source_chunk", "")
    is_multihop  = category == "multihop"

    try:
        chunks, latency_ms = retrieve(query, method, store, retriever, groq, top_k)
    except Exception as e:
        return {"error": str(e), "query": query, "method": method, "category": category}

    # Hit detection
    if is_multihop:
        hit      = is_hit_multihop(chunks, item)
        hit_both = both_hit_multihop(chunks, item)
        mrr_val  = mrr_score(chunks, source_chunk)
    else:
        hit      = is_hit(chunks, source_chunk)
        hit_both = hit
        mrr_val  = mrr_score(chunks, source_chunk)

    # Generate
    try:
        response    = generator.generate(query, chunks)
        answer      = response.answer
        gen_latency = response.latency_generation * 1000
    except Exception:
        answer, gen_latency = "", 0

    # Confidence
    if not answer or len(answer.strip()) < 10:
        return {
            "query": query, "method": method, "category": category,
            "source_chunk": source_chunk,
            "hit@k": False, "hit_both": False, "mrr": 0.0,
            "confidence_v1": 0.0, "answer_ok": False,
            "latency_retrieval_ms": latency_ms,
            "latency_generation_ms": 0,
            "latency_total_ms": latency_ms,
            "answer_preview": "[GENERATION FAILED]",
            "error": "empty_answer",
        }

    context_texts = [c["text"] for c in chunks]
    confidence    = compute_context_overlap(answer, context_texts)
    answer_ok     = confidence >= 0.5 and "does not contain" not in answer.lower()

    return {
        "query":        query,
        "method":       method,
        "category":     category,
        "source_chunk": source_chunk,
        "hit@k":        hit,
        "hit_both":     hit_both,
        "mrr":          mrr_val,
        "confidence_v1": confidence,
        "answer_ok":    answer_ok,
        "latency_retrieval_ms":  latency_ms,
        "latency_generation_ms": round(gen_latency, 2),
        "latency_total_ms":      round(latency_ms + gen_latency, 2),
        "answer_preview": answer[:150],
    }


# ── Category runner ────────────────────────────────────────────────────────────

def run_category(category: str, method: str,
                 store, retriever, generator, groq,
                 top_k: int = TOP_K) -> dict:

    path  = CATEGORIES[category]
    with open(path) as f:
        data = json.load(f)

    if category == "lexical":
        data = data[:LEXICAL_LIMIT]

    n = len(data)
    print(f"\n  [{method}] {category} ({n} samples)")

    results, t0 = [], time.time()

    for i, item in enumerate(data):
        result = evaluate_sample(item, method, category,
                                 store, retriever, generator, groq, top_k)
        results.append(result)

        hit  = result.get("hit@k", False)
        conf = result.get("confidence_v1", 0)
        err  = result.get("error", "")
        status = "✅" if hit else "❌"
        suffix = f" [{err}]" if err else ""
        print(f"    [{i+1:02d}/{n}] {status} conf={conf:.2f}{suffix} "
              f"| {item['question'][:45]}...")

        if method == "hybrid_mq_hyde_rerank":
            time.sleep(2.0)
        elif method == "hybrid_rrf":
            time.sleep(1.0)
        else:
            time.sleep(0.8)

    # Aggregate
    valid        = [r for r in results if "error" not in r or r.get("error") == ""]
    total        = len(results)
    precision_k  = round(sum(r.get("hit@k", False) for r in valid) / total, 4)
    both_hit     = round(sum(r.get("hit_both", False) for r in valid) / total, 4)
    avg_mrr      = round(sum(r.get("mrr", 0) for r in valid) / total, 4)
    avg_conf     = round(sum(r.get("confidence_v1", 0) for r in valid) / total, 4)
    answer_ok    = round(sum(r.get("answer_ok", False) for r in valid) / total, 4)
    avg_lat_ret  = round(sum(r.get("latency_retrieval_ms", 0) for r in valid) / total, 2)
    elapsed      = round(time.time() - t0, 1)

    summary = {
        "method":      method,
        "category":    category,
        "n_samples":   total,
        f"precision@{top_k}": precision_k,
        "both_hit":    both_hit,
        "mrr":         avg_mrr,
        "avg_confidence_v1": avg_conf,
        "answer_ok_rate":    answer_ok,
        "avg_latency_retrieval_ms": avg_lat_ret,
        "elapsed_s":   elapsed,
        "per_sample":  results,
    }

    print(f"    → P@{top_k}={precision_k} | MRR={avg_mrr} | "
          f"conf={avg_conf} | ok={answer_ok} | lat={avg_lat_ret}ms")

    return summary


# ── Main ablation ──────────────────────────────────────────────────────────────

def run_ablation_per_category(
    methods: list[str]    = METHODS,
    categories: list[str] = list(CATEGORIES.keys()),
    top_k: int            = TOP_K,
):
    print(f"\n{'='*65}")
    print(f"  ABLATION PER-CATEGORY — Sprint 2")
    print(f"  Methods    : {methods}")
    print(f"  Categories : {categories}")
    print(f"  top_k      : {top_k}")
    print(f"{'='*65}")

    store, retriever, generator, groq = get_components()

    all_results = {}
    t_total = time.time()

    for category in categories:
        print(f"\n{'━'*65}")
        print(f"  CATEGORY: {category.upper()}")
        print(f"{'━'*65}")
        all_results[category] = {}

        for method in methods:
            result = run_category(category, method,
                                  store, retriever, generator, groq, top_k)
            all_results[category][method] = result
            _save(all_results)

    _print_leaderboard(all_results, top_k)
    elapsed = round(time.time() - t_total, 1)
    print(f"\n[DONE] Total elapsed: {elapsed}s ({elapsed/60:.1f} min)")
    return all_results


# ── I/O ────────────────────────────────────────────────────────────────────────

def _save(results: dict):
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _print_leaderboard(results: dict, top_k: int):
    print(f"\n{'='*75}")
    print(f"  LEADERBOARD PER-CATEGORY")
    print(f"{'='*75}")
    print(f"  {'Category':<14} {'Method':<26} {'P@'+str(top_k):<7} {'MRR':<7} "
          f"{'Conf':<7} {'OK%':<7} {'Lat(ms)'}")
    print(f"  {'─'*70}")

    for category, method_results in results.items():
        for method, r in sorted(method_results.items(),
                                key=lambda x: x[1].get(f"precision@{top_k}", 0),
                                reverse=True):
            pk  = r.get(f"precision@{top_k}", 0)
            mrr = r.get("mrr", 0)
            cf  = r.get("avg_confidence_v1", 0)
            ok  = r.get("answer_ok_rate", 0)
            lat = r.get("avg_latency_retrieval_ms", 0)
            print(f"  {category:<14} {method:<26} {pk:<7.3f} {mrr:<7.3f} "
                  f"{cf:<7.3f} {ok:<7.3f} {lat:.1f}ms")
        print()

    # Summary insight
    print(f"  {'='*70}")
    print(f"  INSIGHT SUMMARY:")
    for category, method_results in results.items():
        sorted_m = sorted(method_results.items(),
                          key=lambda x: x[1].get(f"precision@{top_k}", 0),
                          reverse=True)
        best     = sorted_m[0]
        worst    = sorted_m[-1]
        print(f"  {category:<14}: best={best[0]} "
              f"(P@{top_k}={best[1].get(f'precision@{top_k}',0):.3f}) | "
              f"worst={worst[0]} "
              f"(P@{top_k}={worst[1].get(f'precision@{top_k}',0):.3f})")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",    nargs="+", default=METHODS)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()))
    parser.add_argument("--top_k",     type=int,  default=TOP_K)
    args = parser.parse_args()

    run_ablation_per_category(
        methods=args.methods,
        categories=args.categories,
        top_k=args.top_k,
    )
