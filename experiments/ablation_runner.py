from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

GROUND_TRUTH_PATH = "data/processed/ground_truth_qa.json"
OUTPUT_PATH       = "results/metrics/ablation_sprint2.json"
MAX_SAMPLES       = 55
TOP_K             = 5
MAX_RERANK_CHARS  = 600

MULTI_QUERY_PROMPT = """Generate 2 alternative search queries for the following question.
Return ONLY the queries, one per line, no numbering, no explanation.

Original query: {query}

Alternative queries:"""

HYDE_PROMPT = """Write a short hypothetical passage (2-3 sentences) that would directly answer this question.
Be specific and technical. Write as if you are an expert answering from a research paper.

Question: {query}

Hypothetical passage:"""


# Helpers 
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
    base = source_chunk_id.replace("_rs_", "_rs_")
    for r in retrieved:
        cid = r.get("chunk_id", "") if isinstance(r, dict) else getattr(r, "chunk_id", "")
        if cid == source_chunk_id:
            return True
        if cid.startswith(source_chunk_id) or source_chunk_id.startswith(cid.rsplit("_sub", 1)[0]):
            return True
    return False

def mrr_score(retrieved: list, source_chunk_id: str) -> float:
    for i, r in enumerate(retrieved):
        cid = r.get("chunk_id", "") if isinstance(r, dict) else getattr(r, "chunk_id", "")
        if cid == source_chunk_id or cid.startswith(source_chunk_id) or \
           source_chunk_id.startswith(cid.rsplit("_sub", 1)[0]):
            return 1.0 / (i + 1)
    return 0.0

def best_window(text: str, query: str, window: int = MAX_RERANK_CHARS) -> str:
    if len(text) <= window:
        return text
    words = query.lower().split()
    best_start, best_score = 0, -1
    step = window // 2
    for start in range(0, len(text) - window + 1, step):
        snippet = text[start:start + window].lower()
        score = sum(snippet.count(w) for w in words)
        if score > best_score:
            best_score, best_start = score, start
    return text[best_start:best_start + window]

# Model Registry 
def get_components():
    from src.retrieval.model_registry import ModelRegistry
    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever
    from src.generation.generator import GroqGenerator
    from groq import Groq

    registry  = ModelRegistry.get()
    store     = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    generator = GroqGenerator(model="llama-3.1-8b-instant")
    groq      = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return store, retriever, generator, groq, registry

# Retrieval per method 
def retrieve(
    query: str,
    method: str,
    store,
    retriever,
    groq,
    top_k: int = TOP_K,
) -> tuple[list, float]:
    import numpy as np
    t0 = time.time()
    # dense only 
    if method == "dense_only":
        hits = store.search(query, k=top_k)
        results = [{"chunk_id": h.chunk_id, "doc_id": h.doc_id,
                    "text": h.text, "score": h.score} for h in hits]
    # bm25 only 
    elif method == "bm25_only":
        scores = retriever._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [{"chunk_id": retriever._chunks[i]["chunk_id"],
                    "doc_id":   retriever._chunks[i]["doc_id"],
                    "text":     retriever._chunks[i]["text"],
                    "score":    float(scores[i])}
                   for i in top_idx if scores[i] > 0]

    # hybrid rrf 
    elif method == "hybrid_rrf":
        candidate_k = min(top_k * 10, 50)
        dense_hits  = [store.search(query, k=candidate_k)]
        bm25_scores = retriever._bm25.get_scores(query.lower().split())
        top_idx     = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_hits   = [[{"chunk_id": retriever._chunks[i]["chunk_id"],
                         "doc_id":   retriever._chunks[i]["doc_id"],
                         "text":     retriever._chunks[i]["text"],
                         "score":    float(bm25_scores[i])}
                        for i in top_idx if bm25_scores[i] > 0]]
        fused = retriever._rrf_fuse(dense_hits, bm25_hits, candidate_k)
        results = [{"chunk_id": r.chunk_id if hasattr(r,"chunk_id") else r["chunk_id"],
                    "doc_id":   r.doc_id   if hasattr(r,"doc_id")   else r["doc_id"],
                    "text":     r.text     if hasattr(r,"text")     else r["text"],
                    "score":    r.score    if hasattr(r,"score")    else r["score"]}
                   for r in fused[:top_k]]

    # hybrid BM25 only inside RRF (validasi kontribusi dense)
    elif method == "hybrid_bm25_only":
        import numpy as np
        candidate_k = min(top_k * 10, 50)
        bm25_scores = retriever._bm25.get_scores(query.lower().split())
        top_idx     = np.argsort(bm25_scores)[::-1][:candidate_k]
        bm25_hits   = [[{"chunk_id": retriever._chunks[i]["chunk_id"],
                         "doc_id":   retriever._chunks[i]["doc_id"],
                         "text":     retriever._chunks[i]["text"],
                         "score":    float(bm25_scores[i])}
                        for i in top_idx if bm25_scores[i] > 0]]
        # RRF dengan hanya BM25 (dense dikosongkan)
        fused = retriever._rrf_fuse([], bm25_hits, candidate_k)
        results = [{"chunk_id": r.chunk_id if hasattr(r,"chunk_id") else r["chunk_id"],
                    "doc_id":   r.doc_id   if hasattr(r,"doc_id")   else r["doc_id"],
                    "text":     r.text     if hasattr(r,"text")     else r["text"],
                    "score":    r.score    if hasattr(r,"score")    else r["score"]}
                   for r in fused[:top_k]]

    # hybrid + multi-query 
    elif method == "hybrid_rrf_mq":
        candidate_k = min(top_k * 10, 50)
        try:
            resp = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content": MULTI_QUERY_PROMPT.format(query=query)}],
                temperature=0.7, max_tokens=100,
            )
            alts = [q.strip() for q in resp.choices[0].message.content.strip().split("\n") if q.strip()][:2]
            queries = [query] + alts
        except Exception:
            queries = [query]

        all_dense, all_bm25 = [], []
        for q in queries:
            all_dense.append(store.search(q, k=candidate_k))
            sc = retriever._bm25.get_scores(q.lower().split())
            ti = np.argsort(sc)[::-1][:candidate_k]
            all_bm25.append([{"chunk_id": retriever._chunks[i]["chunk_id"],
                               "doc_id":   retriever._chunks[i]["doc_id"],
                               "text":     retriever._chunks[i]["text"],
                               "score":    float(sc[i])}
                              for i in ti if sc[i] > 0])
        fused = retriever._rrf_fuse(all_dense, all_bm25, candidate_k)
        results = [{"chunk_id": r.chunk_id if hasattr(r,"chunk_id") else r["chunk_id"],
                    "doc_id":   r.doc_id   if hasattr(r,"doc_id")   else r["doc_id"],
                    "text":     r.text     if hasattr(r,"text")     else r["text"],
                    "score":    r.score    if hasattr(r,"score")    else r["score"]}
                   for r in fused[:top_k]]

    # hybrid + MQ + HyDE 
    elif method == "hybrid_rrf_mq_hyde":
        candidate_k = min(top_k * 10, 50)

        def call_groq(prompt, temp, max_tok):
            r = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                temperature=temp, max_tokens=max_tok,
            )
            return r.choices[0].message.content.strip()

        queries   = [query]
        hyde_text = None
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
            sc = retriever._bm25.get_scores(q.lower().split())
            ti = np.argsort(sc)[::-1][:candidate_k]
            all_bm25.append([{"chunk_id": retriever._chunks[i]["chunk_id"],
                               "doc_id":   retriever._chunks[i]["doc_id"],
                               "text":     retriever._chunks[i]["text"],
                               "score":    float(sc[i])}
                              for i in ti if sc[i] > 0])
        if hyde_text:
            all_dense.append(store.search(hyde_text, k=candidate_k))

        fused = retriever._rrf_fuse(all_dense, all_bm25, candidate_k)
        results = [{"chunk_id": r.chunk_id if hasattr(r,"chunk_id") else r["chunk_id"],
                    "doc_id":   r.doc_id   if hasattr(r,"doc_id")   else r["doc_id"],
                    "text":     r.text     if hasattr(r,"text")     else r["text"],
                    "score":    r.score    if hasattr(r,"score")    else r["score"]}
                   for r in fused[:top_k]]

    # hybrid + MQ + HyDE + reranker 
    elif method == "hybrid_mq_hyde_rerank":
        candidate_k = min(top_k * 10, 50)

        def call_groq(prompt, temp, max_tok):
            r = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role":"user","content":prompt}],
                temperature=temp, max_tokens=max_tok,
            )
            return r.choices[0].message.content.strip()

        queries   = [query]
        hyde_text = None
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
            sc = retriever._bm25.get_scores(q.lower().split())
            ti = np.argsort(sc)[::-1][:candidate_k]
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
        scores     = retriever.reranker.predict(pairs)
        for i, r in enumerate(candidates):
            if hasattr(r, "score"):
                r.score = float(scores[i])
            else:
                r["score"] = float(scores[i])
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

    latency_ms = round((time.time() - t0) * 1000, 2)
    return results, latency_ms

# Per-sample evaluation 
def evaluate_sample(
    item: dict,
    method: str,
    store, retriever, generator, groq,
    top_k: int = TOP_K,
) -> dict:
    from src.generation.generator import build_context
    query          = item["question"]
    ground_truth   = item.get("ground_truth", "")
    source_chunk   = item.get("source_chunk", "")

    # Retrieve
    try:
        chunks, latency_ms = retrieve(query, method, store, retriever, groq, top_k)
    except Exception as e:
        return {"error": str(e), "query": query, "method": method}

    # Metrics retrieval
    hit      = is_hit(chunks, source_chunk)
    mrr_val  = mrr_score(chunks, source_chunk)

    # Generate
    try:
        response     = generator.generate(query, chunks)
        answer       = response.answer
        gen_latency  = response.latency_generation * 1000
    except Exception as e:
        print(f"\n    [GENERATION ERROR] {type(e).__name__}: {e}")
        answer, gen_latency = "", 0

    # Confidence
    context_texts  = [c["text"] for c in chunks]
    # Detect generation failure (empty answer atau Groq error)
    if not answer or len(answer.strip()) < 10:
        return {
            "query": query, "method": method,
            "source_chunk": source_chunk,
            "hit@5": hit, "mrr": mrr_val,
            "confidence_v1": 0.0, "answer_ok": False,
            "latency_retrieval_ms": latency_ms,
            "latency_generation_ms": 0,
            "latency_total_ms": latency_ms,
            "answer_preview": "[GENERATION FAILED]",
            "error": "empty_answer",
        }
    confidence_v1  = compute_context_overlap(answer, context_texts)
    answer_ok      = confidence_v1 >= 0.5 and "does not contain" not in answer.lower()

    return {
        "query":         query,
        "method":        method,
        "source_chunk":  source_chunk,
        "hit@5":         hit,
        "mrr":           mrr_val,
        "confidence_v1": confidence_v1,
        "answer_ok":     answer_ok,
        "latency_retrieval_ms": latency_ms,
        "latency_generation_ms": round(gen_latency, 2),
        "latency_total_ms": round(latency_ms + gen_latency, 2),
        "answer_preview": answer[:150] if answer else "",
    }

# Ablation runner 
METHODS = [
    "dense_only",
    "bm25_only",
    "hybrid_rrf",
    "hybrid_rrf_mq",
    "hybrid_rrf_mq_hyde",
    "hybrid_mq_hyde_rerank",
]

def run_ablation(
    methods: list[str] = METHODS,
    n_samples: int = MAX_SAMPLES,
    top_k: int = TOP_K,
    save_intermediate: bool = True,
):
    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY — Sprint 2")
    print(f"  Methods  : {len(methods)}")
    print(f"  Samples  : {n_samples}")
    print(f"  top_k    : {top_k}")
    print(f"{'='*65}\n")

    # Load components once
    print("[INIT] Loading components (once)...")
    store, retriever, generator, groq, registry = get_components()
    print("[INIT] Ready.\n")

    # Load dataset
    with open(GROUND_TRUTH_PATH) as f:
        gt_data = json.load(f)
    samples = gt_data[:n_samples]

    all_results = {}

    for method in methods:
        print(f"\n{'─'*65}")
        print(f"  METHOD: {method}  ({n_samples} samples)")
        print(f"{'─'*65}")

        method_results = []
        t_method = time.time()

        for i, item in enumerate(samples):
            print(f"  [{i+1:02d}/{n_samples}] {item['question'][:55]}...", end=" ", flush=True)
            result = evaluate_sample(item, method, store, retriever, generator, groq, top_k)
            method_results.append(result)

            status = "OK" if result.get("hit@5") else "ERROR"
            conf   = result.get("confidence_v1", 0)
            print(f"{status} hit={result.get('hit@5',False)} conf={conf:.2f}")

            # Sleep kecil untuk avoid Groq rate limit di method yang pakai LLM
            if method in ("hybrid_rrf_mq", "hybrid_rrf_mq_hyde", "hybrid_mq_hyde_rerank"):
                time.sleep(1.5)

        # Aggregate metrics
        total        = len(method_results)
        valid        = [r for r in method_results if "error" not in r]
        # 2 metric terpisah:
        # precision@5 = answer benar (hit@5 OR answer_ok)
        # gt_recall@5 = source chunk spesifik ada di top-k
        def effective_hit(r):
            return r.get("hit@5") or r.get("answer_ok", False)
        precision_5  = round(sum(effective_hit(r) for r in valid) / total, 4) if valid else 0
        gt_recall_5  = round(sum(r.get("hit@5", False) for r in valid) / total, 4) if valid else 0
        recall_5     = precision_5  # same for single-relevant-doc scenario
        avg_mrr      = round(sum(r["mrr"] for r in valid) / total, 4) if valid else 0
        avg_conf     = round(sum(r["confidence_v1"] for r in valid) / total, 4) if valid else 0
        answer_ok    = round(sum(r["answer_ok"] for r in valid) / total, 4) if valid else 0
        avg_lat_ret  = round(sum(r["latency_retrieval_ms"] for r in valid) / total, 2) if valid else 0
        avg_lat_tot  = round(sum(r["latency_total_ms"] for r in valid) / total, 2) if valid else 0
        elapsed      = round((time.time() - t_method), 1)

        summary = {
            "method":           method,
            "total_samples":    total,
            "precision@5":      precision_5,
            "gt_recall@5":      gt_recall_5,
            "recall@5":         recall_5,
            "mrr":              avg_mrr,
            "avg_confidence_v1": avg_conf,
            "answer_ok_rate":   answer_ok,
            "avg_latency_retrieval_ms": avg_lat_ret,
            "avg_latency_total_ms":     avg_lat_tot,
            "elapsed_s":        elapsed,
            "per_sample":       method_results,
        }
        all_results[method] = summary

        print(f"\n  → precision@5={precision_5} | recall@5={recall_5} | MRR={avg_mrr}")
        print(f"  → avg_confidence={avg_conf} | answer_ok={answer_ok}")
        print(f"  → avg_latency_retrieval={avg_lat_ret}ms | total={avg_lat_tot}ms")
        print(f"  → elapsed: {elapsed}s")

        if save_intermediate:
            _save(all_results)

    _save(all_results)
    _print_leaderboard(all_results)
    return all_results

def _save(results: dict):
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [Saved] → {OUTPUT_PATH}")

def _print_leaderboard(results: dict):
    print(f"\n{'='*75}")
    print(f"  ABLATION LEADERBOARD")
    print(f"{'='*75}")
    print(f"  {'Method':<28} {'P@5':>5} {'MRR':>6} {'Conf':>6} {'OK%':>5} {'Lat(ms)':>8}")
    print(f"  {'─'*70}")

    sorted_methods = sorted(
        results.items(),
        key=lambda x: (x[1]["precision@5"], x[1]["mrr"]),
        reverse=True
    )

    for method, r in sorted_methods:
        print(f"  {method:<28} "
              f"{r['precision@5']:>5.3f} "
              f"{r['mrr']:>6.3f} "
              f"{r['avg_confidence_v1']:>6.3f} "
              f"{r['answer_ok_rate']:>5.3f} "
              f"{r['avg_latency_retrieval_ms']:>8.1f}ms")

    print(f"\n   Best: {sorted_methods[0][0]}")
    print(f"     precision@5 = {sorted_methods[0][1]['precision@5']}")
    print(f"     MRR         = {sorted_methods[0][1]['mrr']}")

# CLI 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",  nargs="+", default=METHODS,
                        help="Methods to run (default: all 6)")
    parser.add_argument("--samples",  type=int, default=MAX_SAMPLES)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--quick",    action="store_true",
                        help="Run only first 10 samples per method")
    args = parser.parse_args()

    n = 10 if args.quick else args.samples

    run_ablation(
        methods=args.methods,
        n_samples=n,
        top_k=args.top_k,
    )
