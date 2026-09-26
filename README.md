# RAG Research

> Production-grade RAG pipeline with Rust-powered chunking, hybrid retrieval, and RAGAS evaluation.  
> Part of the [AI Infrastructure Workspace](https://github.com/malikimayzar) — a polyglot research platform built with Python, Rust, and Go.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Rust](https://img.shields.io/badge/Rust-PyO3%20%2B%20Maturin-orange)
![Go](https://img.shields.io/badge/Go-Dashboard-cyan)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-green)

---

## What This Is

An ablation study and evaluation platform for Retrieval-Augmented Generation systems — comparing chunking strategies (Python semantic vs Rust semantic) and retrieval methods (Dense, BM25, Hybrid RRF + BGE Reranker) with standardized RAGAS metrics.

Built as the **Precision Engine** of a larger AI infrastructure stack. The system does not just answer questions — it measures how well it answers them.

---

## Architecture

```
PDF/TXT Documents
       │
       ▼
┌─────────────────────┐
│  document_loader.py │  PyMuPDF — layout-aware PDF parsing
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────────────────┐
│chunker │  │ semantic_chunker_rust │  Rust + PyO3 + Maturin
│(Python)│  │ (unicode-aware, fast) │  982 chunks from 7 papers
└────────┘  └──────────┬───────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────┐        ┌──────────────┐
    │  Qdrant     │        │  BM25Okapi   │
    │  HNSW       │        │  (sparse)    │
    │  (dense)    │        └──────┬───────┘
    └──────┬──────┘               │
           │                      │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Hybrid RRF (k=60)   │  Reciprocal Rank Fusion
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  BGE Reranker        │  BAAI/bge-reranker-base
           │  Cross-Encoder       │  Cross-encoder reranking
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  Groq Generator      │  llama-3.3-70b-versatile
           │  System Prompt Strict│  Citation rules, no hallucination
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  RAGAS Evaluator     │  5 metrics — Faithfulness,
           │  (Groq + HuggingFace)│  Answer Relevancy, Context
           └──────────────────────┘  Precision, Recall, Correctness
```

---

## Stack

| Layer | Tool | Role |
|---|---|---|
| Chunking | Rust + PyO3 + Maturin | Unicode-aware semantic segmentation, 10-50x faster than Python |
| Embedding | all-MiniLM-L6-v2 (384-dim) | Dense vector encoding |
| Vector Store | Qdrant HNSW (local persistent) | 982 vectors, filterable, scalable |
| Sparse Retrieval | BM25Okapi (rank-bm25) | Keyword matching |
| Hybrid Fusion | Reciprocal Rank Fusion k=60 | Combine dense + BM25 |
| Reranker | BAAI/bge-reranker-base | Cross-encoder reranking |
| LLM | Groq llama-3.3-70b-versatile | Sub-second generation |
| Evaluation | RAGAS + LangChain | 5 standardized metrics |
| Dashboard | Go + Fiber | Analytics leaderboard |

---

## Evaluation Results

All runs on 7 ArXiv papers (RAG, LLM, Attention domains). 55 clean QA pairs, RAGAS framework.

### RAGAS Metrics — Best Run (clean_dataset_v2, 70b judge, n=55)

| Metric | Score | Notes |
|---|---|---|
| **Faithfulness** | **0.8000** | Every claim traceable to source |
| **Context Precision** | 0.7500 | Relevant chunks retrieved |
| **Context Recall** | 0.9313 | Near-complete coverage |
| **Answer Relevancy** | 0.7438 | On-topic responses |
| **Answer Correctness** | 0.6846 | Factual accuracy vs ground truth |
| **Hallucination Rate** | 18.4% | Detected and logged |

### Progression Across Runs

| Experiment | Faithfulness | Samples | Judge | Notes |
|---|---|---|---|---|
| semantic_hybrid_rerank_v1 | 0.6754 | 20 | Manual | Corrupted dataset baseline |
| clean_dataset_v1 | 0.7987 | 33 | 8b | Clean dataset, first run |
| clean_dataset_v2 | **0.8000** | 55 | 70b | Full dataset, best result |
| final_v1 | 0.7297 | 55 | 8b+json | 70b generator, 8b judge |

**+18.4% Faithfulness improvement** from dataset cleanup + system prompt upgrade.

### Original Ablation Study (Phase 1, Ollama/Mistral)

6 experiments × 3 queries = 18 evaluations. Runtime: 241 minutes.

| Rank | Method | Chunk | Faithfulness | Hallucination |
|---|---|---|---|---|
| 🥇 1 | BM25 | 256 | 1.000 | 0% |
| 🥈 2 | Hybrid RRF | 512 | 1.000 | 0% |
| 🥉 3 | Dense | 512 | 0.933 | 0% |
| 4 | Dense | 256 | 0.917 | 0% |
| 5 | BM25 | 512 | 0.833 | 0% |
| 6 | Hybrid RRF | 256 | 0.633 | 0% |

**Zero hallucinations across all 18 queries** with Ollama/Mistral + context-grounding prompt.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/malikimayzar/rag-research
cd rag-research

python -m venv venv
source venv/bin/activate
pip install -e .

# Set API key
echo "GROQ_API_KEY=your_key_here" > .env

# Build Rust chunker
cd src/ingestion/semantic_chunker_rust && maturin develop && cd ../../..

# Run full pipeline
make all
```

### Manual Steps

```bash
# Ingest + chunk (Rust engine)
python src/ingestion/document_loader.py
python src/ingestion/semantic_chunker.py    # generates 982 chunks

# Index to Qdrant
python src/retrieval/qdrant_store.py

# Run evaluation (55 samples)
python scripts/run_eval.py

# Visualize
python scripts/visualize.py

# Start Go dashboard
go run cmd/server/main.go
```

### Live Demo

```bash
python demo.py
```

```
Q: What is Retrieval-Augmented Generation?
A: Retrieval-Augmented Generation (RAG) is a technique that combines
   information retrieval with language model generation [Source 1],
   enhancing LLMs by retrieving relevant document chunks from an
   external knowledge base [Source 2].
⚡ Latency: 1.471s
```

---

## Makefile

```bash
make rust-build   # Compile Rust chunker (PyO3 + Maturin)
make ingest       # Parse PDFs + semantic chunk
make index        # Index 982 vectors to Qdrant
make eval         # Run RAGAS evaluation (55 samples)
make viz          # Generate figures
make all          # Full pipeline: rust-build → ingest → index → eval → viz
make dashboard    # Start Go analytics dashboard
```

---

## Project Structure

```
rag-research/
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py        # PyMuPDF PDF parsing
│   │   ├── chunker.py                # Fixed-size chunker (regression baseline)
│   │   ├── semantic_chunker.py       # Python semantic chunker
│   │   └── semantic_chunker_rust/    # Rust chunker (PyO3 + Maturin)
│   ├── retrieval/
│   │   ├── qdrant_store.py           # Qdrant HNSW + HybridRetriever
│   │   ├── bm25_retriever.py         # BM25Okapi sparse retrieval
│   │   ├── embedder.py               # FAISS embedder (regression baseline)
│   │   └── hybrid_retriever.py       # Legacy RRF (pre-Qdrant)
│   ├── generation/
│   │   └── generator.py              # Groq generator, strict system prompt
│   ├── evaluation/
│   │   ├── ragas_evaluator.py        # RAGAS 5-metric evaluator
│   │   └── evaluator.py              # Legacy evaluator (regression baseline)
│   └── api/
│       └── main.py                   # FastAPI endpoint (WIP)
├── experiments/
│   └── ablation_runner.py            # Phase 1 ablation (legacy stack)
├── scripts/
│   ├── run_eval.py                   # Main evaluation script
│   └── visualize.py                  # Figure generation
├── data/
│   ├── raw/                          # 7 ArXiv PDFs + TXT
│   ├── processed/
│   │   ├── chunks_semantic.json      # 982 Rust semantic chunks
│   │   └── ground_truth_qa.json      # 55 clean QA pairs
│   └── qdrant_storage/               # Local Qdrant persistent storage
├── results/
│   ├── figures/                      # 6 visualization plots
│   ├── metrics/
│   │   └── ragas_results.json        # All RAGAS evaluation runs
│   └── failure_cases/
│       └── failure_analysis.json     # Failure mode breakdown
├── cmd/server/main.go                # Go dashboard server
├── demo.py                           # Live demo script
├── Makefile
├── pyproject.toml                    # Editable install (pip install -e .)
└── docker-compose.yml
```

---

## Key Engineering Decisions

**Rust Chunker over Python**  
Python semantic chunker → 652 chunks. Rust semantic chunker → 982 chunks. More granular, faster, unicode-aware. Rust handles sentence segmentation that regex-based Python misses.

**Qdrant over FAISS**  
FAISS is flat and memory-bound. Qdrant is HNSW-based, filterable by `doc_id`, persistent, and REST-accessible. Enables filtering for multi-document workloads.

**Groq over Ollama**  
Ollama latency: 30-60s per query. Groq latency: 0.3-1.5s. Same model quality, 40x faster. Critical for evaluation loops that call the LLM hundreds of times.

**Strict System Prompt**  
Generator uses `system` + `user` message separation with explicit rules: answer only from context, cite sources with [Source N], refuse if context insufficient. This is what drives Faithfulness toward 0.80+.

**Clean Ground Truth**  
Original 50 QA pairs contained ~30% noise (zip codes, anatomy, fiction). Regenerated with strict domain filtering → 55 clean pairs. Faithfulness jumped from 0.67 → 0.80 from dataset cleanup alone.

---

## Dataset

7 ArXiv papers across 3 tiers:

| Tier | Paper | Domain |
|---|---|---|
| 1 | 2005.11401 — RAG (Lewis et al.) | RAG |
| 1 | 2312.10997 — Advanced RAG Survey | RAG |
| 1 | test_intro.txt | RAG intro |
| 2 | 2210.11610 — Large Language Models | LLM |
| 2 | 2212.10560 — Self-Instruct | LLM |
| 3 | 1706.03762 — Attention Is All You Need | Transformers |
| 3 | 2307.09288 — Llama 2 | LLM |

55 clean QA pairs generated via Groq with strict domain filtering. Zero citation trivia, zero fictional content.

---

## Environment

```
Python  : 3.12
Rust    : stable (maturin + PyO3)
Go      : 1.22+
OS      : WSL2 Ubuntu
Hardware: CPU-only, 7.6GB RAM, 8 cores
Qdrant  : local persistent (data/qdrant_storage/)
Groq    : llama-3.3-70b-versatile (generator) + llama-3.1-8b-instant (judge)
Embed   : all-MiniLM-L6-v2 (384-dim, offline-capable)
```

---

## Part of AI Infrastructure Workspace

This repo is the **Precision Engine** in a 4-service AI research platform:

| Repo | Role | Stack |
|---|---|---|
| [mcp-gateway](https://github.com/malikimayzar/mcp-gateway) | Orchestration — routes queries, manages tool execution | Go |
| [arxiv-research-assistant](https://github.com/malikimayzar/arxiv-research-assistant) | Librarian — ArXiv ingestion, metadata, observability | Go + Python |
| **rag-research** | **Precision Engine — hybrid retrieval, RAGAS eval** | **Python + Rust + Go** |
| [llm-eval-framework](https://github.com/malikimayzar/llm-eval-framework) | Auditor — faithfulness without LLM judge | Python |

---

*Maliki Mayzar · 2025-2026 · [github.com/malikimayzar](https://github.com/malikimayzar)*