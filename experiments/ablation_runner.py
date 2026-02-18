import json
import sys
import dataclasses
import itertools
import time
import threading
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from pathlib import Path
from dataclasses import dataclass, asdict
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "src/ingestion")
sys.path.insert(0, "src/retrieval")
sys.path.insert(0, "src/generation")
sys.path.insert(0, "src/evaluation")

from document_loader import load_documents
from chunker import chunk_documents
from embedder import build_dense_index, dense_search
from bm25_retriever import build_bm25_index, bm25_search
from hybrid_retriever import hybrid_search
from generator import generate
from evaluator import evaluate_response, generate_report

from sentence_transformers import SentenceTransformer


# Data classes
@dataclass
class ExperimentConfig:
    exp_id: str
    chunk_size: int
    overlap: int
    retrieval_method: str
    top_k: int
    embedding_model: str


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    avg_faithfulness: float
    avg_context_relevance: float
    avg_answer_relevance: float
    hallucination_rate: float
    honest_abstention_rate: float
    failure_mode_distribution: dict
    total_queries: int
    avg_latency: float


# Standard queries
STANDARD_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "How do attention mechanisms relate to retrieval?",
    "What is the training cost of GPT-4 according to these papers?",
]


# Cache layer
_model_cache: dict[str, SentenceTransformer] = {}
_model_lock = threading.Lock()

_chunk_cache: dict[tuple, list] = {}
_index_cache: dict[tuple, dict] = {}


def get_model(model_name: str) -> SentenceTransformer:
    with _model_lock:
        if model_name not in _model_cache:
            print(f"  [CACHE] Loading model: {model_name}")
            _model_cache[model_name] = SentenceTransformer(model_name)
        return _model_cache[model_name]


def get_chunks(docs, chunk_size: int, overlap: int) -> list:
    key = (chunk_size, overlap)
    if key not in _chunk_cache:
        print(f"\n  [CACHE] Chunking docs: chunk_size={chunk_size}, overlap={overlap}")
        chunks = chunk_documents(
            docs,
            strategy="fixed_size",
            chunk_size=chunk_size,
            overlap=overlap
        )
        _chunk_cache[key] = [to_dict(c) for c in chunks]
        print(f"  [CACHE] {len(_chunk_cache[key])} chunks stored for key={key}")
    else:
        print(f"  [CACHE HIT] chunks key={key} ({len(_chunk_cache[key])} chunks)")
    return _chunk_cache[key]


def get_indexes(chunk_dicts: list, chunk_size: int, overlap: int,
                model_name: str) -> dict:
    key = (chunk_size, overlap, model_name)
    if key not in _index_cache:
        print(f"  [CACHE] Building indexes for key={key}")
        t0 = time.time()
        emb_index = build_dense_index(chunk_dicts, model_name=model_name)
        bm25_index, bm25_chunks = build_bm25_index(chunk_dicts)
        elapsed = time.time() - t0
        _index_cache[key] = {
            "emb_index": emb_index,
            "bm25_index": bm25_index,
            "bm25_chunks": bm25_chunks,
        }
        print(f"  [CACHE] Indexes built in {elapsed:.1f}s — stored for key={key}")
    else:
        print(f"  [CACHE HIT] indexes key={key}")
    return _index_cache[key]


# Helper
def to_dict(c) -> dict:
    if isinstance(c, dict):
        return c.copy()
    elif hasattr(c, 'to_dict'):
        return c.to_dict()
    elif dataclasses.is_dataclass(c):
        return dataclasses.asdict(c)
    else:
        return {
            "chunk_id":     getattr(c, "chunk_id", ""),
            "doc_id":       getattr(c, "doc_id", ""),
            "text":         getattr(c, "text", ""),
            "tier":         getattr(c, "tier", 0),
            "chunk_index":  getattr(c, "chunk_index", 0),
            "total_chunks": getattr(c, "total_chunks", 0),
            "start_char":   getattr(c, "start_char", 0),
            "end_char":     getattr(c, "end_char", 0),
            "metadata":     getattr(c, "metadata", {}),
        }


# Query runner 
def run_single_query(
    query: str,
    config: ExperimentConfig,
    indexes: dict,
    emb_model: SentenceTransformer,
    llm_model: str
) -> dict | None:
    emb_index  = indexes["emb_index"]
    bm25_index = indexes["bm25_index"]
    bm25_chunks = indexes["bm25_chunks"]

    # Retrieve
    try:
        if config.retrieval_method == "dense":
            retrieved = dense_search(query, emb_index, emb_model, top_k=config.top_k)

        elif config.retrieval_method == "bm25":
            retrieved = bm25_search(query, bm25_index, bm25_chunks, top_k=config.top_k)

        elif config.retrieval_method == "hybrid":
            search_result = hybrid_search(
                query, emb_index, bm25_index, bm25_chunks, emb_model,
                top_k=config.top_k
            )
            retrieved = search_result["hybrid"]

        else:
            raise ValueError(f"Unknown method: {config.retrieval_method}")

    except Exception as e:
        print(f"    [ERROR] Retrieval failed for '{query[:50]}': {e}")
        return None

    if not retrieved:
        print(f"    [WARN] No results for: {query[:50]}")
        return None

    # Generate
    try:
        response = generate(query, retrieved, model_name=llm_model)
        latency  = response.latency_generation
    except Exception as e:
        print(f"    [ERROR] Generation failed for '{query[:50]}': {e}")
        return None

    # Evaluate
    try:
        eval_result = evaluate_response(
            query=query,
            answer=response.answer,
            chunks=retrieved,
            retrieval_method=config.retrieval_method
        )
    except Exception as e:
        print(f"    [ERROR] Evaluation failed for '{query[:50]}': {e}")
        return None

    return {"eval": eval_result, "latency": latency, "query": query}

# Single experiment
def run_single_experiment(
    config: ExperimentConfig,
    chunk_dicts: list,
    indexes: dict,
    emb_model: SentenceTransformer,
    llm_model: str = "mistral",
    max_workers: int = 1,
) -> ExperimentResult | None:

    print(f"\n{'─'*55}")
    print(f"  EXP  : {config.exp_id}")
    print(f"  Setup: chunk_size={config.chunk_size}, overlap={config.overlap}")
    print(f"  Run  : method={config.retrieval_method}, top_k={config.top_k}")

    t_start = time.time()
    all_eval_results = []
    latencies = []

    if max_workers > 1:
        print(f"  [PARALLEL] {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_query, q, config, indexes, emb_model, llm_model
                ): q
                for q in STANDARD_QUERIES
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_eval_results.append(result["eval"])
                    latencies.append(result["latency"])
                    print(f"    ✓ {result['query'][:55]}")
    else:
        # Sequential mode
        for query in STANDARD_QUERIES:
            print(f"  [EVAL] {query[:55]}...")
            result = run_single_query(query, config, indexes, emb_model, llm_model)
            if result:
                all_eval_results.append(result["eval"])
                latencies.append(result["latency"])

                ev = result["eval"]
                print(f"    → Faith: {ev.faithfulness_score:.2f}  "
                    f"CtxRel: {ev.context_relevance_score:.2f}  "
                    f"Mode: {ev.failure_mode}")

    if not all_eval_results:
        print("  [SKIP] No eval results")
        return None

    report   = generate_report(all_eval_results)
    elapsed  = time.time() - t_start
    avg_lat  = round(sum(latencies) / len(latencies), 2) if latencies else 0

    print(f"  [DONE] {elapsed:.1f}s — faith={report['avg_faithfulness']:.3f} "
          f"hall={report['hallucination_rate']:.3f}")

    return ExperimentResult(
        config=config,
        avg_faithfulness=report["avg_faithfulness"],
        avg_context_relevance=report["avg_context_relevance"],
        avg_answer_relevance=report["avg_answer_relevance"],
        hallucination_rate=report["hallucination_rate"],
        honest_abstention_rate=report["honest_abstention_rate"],
        failure_mode_distribution=report["failure_mode_distribution"],
        total_queries=report["total_queries"],
        avg_latency=avg_lat,
    )


# Config factory
def get_configs(quick_mode: bool) -> list[ExperimentConfig]:
    if quick_mode:
        return [
            ExperimentConfig("exp_001", 256,  0,  "dense",  3, "all-MiniLM-L6-v2"),
            ExperimentConfig("exp_002", 512,  64, "hybrid", 3, "all-MiniLM-L6-v2"),
            ExperimentConfig("exp_003", 512,  64, "bm25",   3, "all-MiniLM-L6-v2"),
        ]

    return [
        ExperimentConfig("exp_001", 512, 64, "dense",  3, "all-MiniLM-L6-v2"),
        ExperimentConfig("exp_002", 512, 64, "bm25",   3, "all-MiniLM-L6-v2"),
        ExperimentConfig("exp_003", 512, 64, "hybrid", 3, "all-MiniLM-L6-v2"),
        ExperimentConfig("exp_004", 256,  0, "dense",  3, "all-MiniLM-L6-v2"),
        ExperimentConfig("exp_005", 256,  0, "bm25",   3, "all-MiniLM-L6-v2"),
        ExperimentConfig("exp_006", 256,  0, "hybrid", 3, "all-MiniLM-L6-v2"),
    ]


# Ablation study — grouped execution 
def run_ablation_study(
    docs,
    quick_mode: bool = False,
    llm_model: str = "mistral",
    max_workers: int = 1,
) -> list[ExperimentResult]:

    configs = get_configs(quick_mode)

    # ── Sort by (chunk_size, overlap)  ──
    configs.sort(key=lambda c: (c.chunk_size, c.overlap, c.embedding_model))

    # Hitung berapa group unik
    unique_groups = len({(c.chunk_size, c.overlap) for c in configs})

    print(f"\n{'='*55}")
    print(f"  ABLATION STUDY (OPTIMIZED)")
    print(f"{'='*55}")
    print(f"  Total experiments   : {len(configs)}")
    print(f"  Unique index groups : {unique_groups}  "
          f"(vs {len(configs)} di versi lama)")
    print(f"  Queries per exp     : {len(STANDARD_QUERIES)}")
    print(f"  LLM model           : {llm_model}")
    print(f"  Query workers       : {max_workers}")

    est_idx = unique_groups * 3
    est_q   = len(configs) * len(STANDARD_QUERIES) * 2
    print(f"  Estimated time      : ~{est_idx + est_q // 60 + 1}+ minutes "
          f"(lama: {len(configs) * len(STANDARD_QUERIES) * 2 // 60}+ minutes)")
    print(f"{'='*55}\n")

    results = []

    #  Iterasi per group
    for (cs, ov), group_iter in groupby(
        configs, key=lambda c: (c.chunk_size, c.overlap)
    ):
        group = list(group_iter)
        model_name = group[0].embedding_model     

        print(f"\n{'━'*55}")
        print(f"  GROUP: chunk_size={cs}, overlap={ov}  "
              f"({len(group)} experiments)")
        print(f"{'━'*55}")

        chunk_dicts = get_chunks(docs, cs, ov)
        if not chunk_dicts:
            print("  [SKIP] No chunks produced")
            continue

        # Load model
        emb_model = get_model(model_name)

        # Build index
        indexes = get_indexes(chunk_dicts, cs, ov, model_name)

        # Jalankan semua experiment dalam group
        for config in group:
            result = run_single_experiment(
                config=config,
                chunk_dicts=chunk_dicts,
                indexes=indexes,
                emb_model=emb_model,
                llm_model=llm_model,
                max_workers=max_workers,
            )
            if result:
                results.append(result)
                save_results(results, "results/metrics/ablation_incremental.json")
    return results


# I/O
def save_results(results: list[ExperimentResult], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  [SAVE] {len(results)} results → {output_path}")


def print_leaderboard(results: list[ExperimentResult]):
    print(f"\n{'='*75}")
    print(f"{'ABLATION LEADERBOARD':^75}")
    print(f"{'='*75}")
    print(f"{'Rank':<5} {'Exp':<10} {'Method':<8} {'CS':<5} {'OV':<5} "
          f"{'Faith':<7} {'CtxRel':<8} {'Hall%':<7} {'Lat(s)':<7}")
    print(f"{'─'*75}")

    sorted_results = sorted(
        results,
        key=lambda r: (
            r.avg_faithfulness + r.avg_context_relevance - r.hallucination_rate
        ),
        reverse=True,
    )

    for rank, r in enumerate(sorted_results[:10], 1):
        print(
            f"{rank:<5} "
            f"{r.config.exp_id:<10} "
            f"{r.config.retrieval_method:<8} "
            f"{r.config.chunk_size:<5} "
            f"{r.config.overlap:<5} "
            f"{r.avg_faithfulness:<7.3f} "
            f"{r.avg_context_relevance:<8.3f} "
            f"{r.hallucination_rate:<7.3f} "
            f"{r.avg_latency:<7.2f}"
        )

    print(f"\n  Top config: {sorted_results[0].config.exp_id} — "
          f"method={sorted_results[0].config.retrieval_method}, "
          f"chunk_size={sorted_results[0].config.chunk_size}, "
          f"top_k={sorted_results[0].config.top_k}")


def print_cache_stats():
    print(f"\n  [CACHE STATS]")
    print(f"    Models loaded  : {len(_model_cache)}")
    print(f"    Chunk groups   : {len(_chunk_cache)}")
    print(f"    Index groups   : {len(_index_cache)}")

    total_chunks = sum(len(v) for v in _chunk_cache.values())
    print(f"    Total chunks   : {total_chunks} "
          f"(across {len(_chunk_cache)} unique (cs, ov) pairs)")


# Entry point

if __name__ == "__main__":
    docs_path = "data/processed/documents.json"
    if not Path(docs_path).exists():
        print("[ERROR] Jalankan document_loader.py dulu")
        sys.exit(1)

    docs = load_documents(docs_path)
    print(f"[LOAD] {len(docs)} documents loaded")

    # Parse args 
    quick    = "--full"    not in sys.argv
    workers  = 1
    llm      = "mistral"

    for arg in sys.argv[1:]:
        if arg.startswith("--workers="):
            workers = int(arg.split("=")[1])
        if arg.startswith("--llm="):
            llm = arg.split("=")[1]

    if workers > 1:
        print(f"[WARN] Parallel mode aktif (workers={workers}).")
        print(f"       Pastikan LLM '{llm}' thread-safe sebelum lanjut.")

    if quick:
        print("[MODE] Quick mode — 3 experiments")
    else:
        print("[MODE] Full ablation study — 6 experiments")

    # Run 
    t_total = time.time()
    results = run_ablation_study(
        docs,
        quick_mode=quick,
        llm_model=llm,
        max_workers=workers,
    )

    # Save & report 
    save_results(results, "results/metrics/ablation_final.json")
    print_leaderboard(results)
    print_cache_stats()

    elapsed = time.time() - t_total
    print(f"\n[OK] {len(results)} experiments selesai dalam "
          f"{elapsed/60:.1f} menit ({elapsed:.0f}s)")
    print(f"[OK] Results → results/metrics/ablation_final.json")