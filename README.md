# 🔍 RAG System Analysis
### Ablation Study on Chunking Strategies and Retrieval Methods

> **Maliki Mayzar** · February 2025  
> Complete Retrieval-Augmented Generation (RAG) pipeline built from scratch across 8 development phases.

---

## 📊 Key Results (TL;DR)

| Rank | Method | Chunk | Faithfulness | Hallucination | Latency |
|------|--------|-------|-------------|--------------|---------|
| 🥇 1 | BM25 | 256 | **1.000** | **0%** | 199s |
| 🥈 2 | Hybrid RRF | 512 | **1.000** | **0%** | 199s |
| 🥉 3 | Dense | 512 | 0.933 | **0%** | 316s |
| 4 | Dense | 256 | 0.917 | **0%** | 233s |
| 5 | BM25 | 512 | 0.833 | **0%** | 328s |
| 6 | Hybrid RRF | 256 | 0.633 | **0%** | 211s |

**Zero hallucinations across all 18 evaluated queries.**

---

## 🗂️ Project Structure

```
rag-research/
├── data/
│   ├── raw/                          # 7 ArXiv PDFs + TXT
│   │   ├── tier1_2005_11401.pdf      # RAG (Lewis et al.)
│   │   ├── tier1_2312_10997.pdf      # Advanced RAG
│   │   ├── tier1_test_intro.txt      # Custom intro doc
│   │   ├── tier2_2210_11610.pdf      # Retrieval methods
│   │   ├── tier2_2212_10560.pdf      # Dense retrieval
│   │   ├── tier3_1706_03762.pdf      # Attention Is All You Need
│   │   └── tier3_2307_09288.pdf      # LLM survey
│   ├── processed/
│   │   ├── documents.json            # Parsed documents
│   │   ├── chunks_512_64.json        # Chunks (size=512, overlap=64)
│   │   ├── dataset_meta.json         # Dataset metadata
│   │   ├── index_bm25/
│   │   │   ├── bm25_chunks.json      # BM25 chunk index
│   │   │   └── bm25.pkl              # BM25 serialized index
│   │   └── index_minilm/             # FAISS vector index
│   └── adversarial/
│       └── adversarial_queries.json  # Adversarial test queries
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py        # PDF/TXT parsing
│   │   ├── chunker.py                # Fixed-size chunking
│   │   └── dataset_builder.py        # Dataset construction
│   ├── retrieval/
│   │   ├── embedder.py               # all-MiniLM-L6-v2 encoding
│   │   ├── bm25_retriever.py         # BM25 sparse retrieval
│   │   └── hybrid_retriever.py       # RRF fusion
│   ├── generation/
│   │   └── generator.py              # Mistral LLM interface
│   └── evaluation/
│       └── evaluator.py              # Metrics: faithfulness, ctx, ans
│
├── experiments/
│   ├── ablation_runner.py            # Main ablation script
│   ├── chunking/                     # Chunking experiments
│   ├── embedding/                    # Embedding experiments
│   ├── hybrid/                       # Hybrid retrieval experiments
│   └── reranking/                    # Cross-encoder reranking
│
├── notebooks/
│   └── analysis.ipynb                # Visualization notebook
│
├── results/
│   ├── figures/                      # 6 visualization plots
│   │   ├── fig1_leaderboard.png
│   │   ├── fig2_chunksize.png
│   │   ├── fig3_heatmap.png
│   │   ├── fig4_failure_modes.png
│   │   ├── fig5_quality_latency.png
│   │   └── fig6_radar.png
│   ├── metrics/
│   │   ├── ablation_final.json       # Final 6-experiment results
│   │   ├── ablation_incremental.json # Per-experiment incremental
│   │   └── evaluation_results.json   # Phase 5–6 eval results
│   ├── logs/
│   │   ├── ablation_full.log         # Full ablation run log
│   │   ├── generation_test.json      # Generation test results
│   │   └── hybrid_search_test.json   # Hybrid search test
│   └── failure_cases/
│       └── failure_analysis.json     # Failure mode details
│
├── reports/
│   ├── FINAL_REPORT.md               # Mini-paper (Abstract→Conclusion)
│   └── RAG_Final_Report.pdf          # PDF version
│
├── requirements/
│   ├── base.txt                      # Core dependencies
│   ├── llm.txt                       # LLM dependencies
│   ├── api.txt                       # API dependencies
│   ├── dev.txt                       # Dev tools
│   └── research.txt                  # Research tools
│
├── visualize.py                      # Visualization script
├── requirements.txt                  # Main requirements
├── .gitignore
└── README.md
```

---

## 🏗️ System Architecture

```
PDF/TXT Documents
       │
       ▼
┌──────────────────┐
│ document_loader  │  PDF parsing + text extraction
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   chunker.py     │  fixed_size strategy
│  chunk=256/512   │  overlap=0/64
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ FAISS  │ │  BM25  │  embedder.py + bm25_retriever.py
│(dense) │ │(sparse)│  all-MiniLM-L6-v2 (384-dim)
└────────┘ └────────┘
    ▼         ▼
┌──────────────────┐
│ hybrid_retriever │  Reciprocal Rank Fusion (k=60)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  generator.py    │  Mistral via Ollama (local)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  evaluator.py    │  Faithfulness · Context Rel · Answer Rel
└──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.12
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama + Mistral (local LLM)
ollama pull mistral
```

### Run Pipeline
```bash
# Step 1 — Ingest & parse documents
python src/ingestion/document_loader.py

# Step 2 — Build chunks + indexes
python src/ingestion/chunker.py

# Step 3 — Quick ablation (3 experiments)
python experiments/ablation_runner.py

# Step 4 — Full ablation (6 experiments, ~4 hours)
python experiments/ablation_runner.py --full

# Step 5 — Generate visualizations
python visualize.py
```

### View Results Summary
```bash
python3 -c "
import json
data = json.load(open('results/metrics/ablation_final.json'))
for e in sorted(data, key=lambda x: -x['avg_faithfulness']):
    c = e['config']
    print(f\"{c['exp_id']} | {c['retrieval_method']:7} | chunk={c['chunk_size']} | \
F:{e['avg_faithfulness']:.3f} C:{e['avg_context_relevance']:.3f} | {e['avg_latency']:.1f}s\")
"
```

---

## 📈 Visualizations

| Figure | Description |
|--------|-------------|
| `fig1_leaderboard.png` | Faithfulness & context relevance ranking |
| `fig2_chunksize.png` | Chunk size 256 vs 512 per method |
| `fig3_heatmap.png` | Full metrics heatmap across all experiments |
| `fig4_failure_modes.png` | Correct / partial / abstention breakdown |
| `fig5_quality_latency.png` | Quality vs speed trade-off bubble chart |
| `fig6_radar.png` | Method comparison radar chart |

---

## 🔬 Ablation Study

6 experiments × 3 queries = **18 total evaluations** · Runtime: **241.2 minutes**

| Exp | Chunk | Overlap | Method | Faithfulness | Ctx Rel | Latency |
|-----|-------|---------|--------|-------------|---------|---------|
| exp_001 | 512 | 64 | Dense | 0.933 | 0.667 | 316s |
| exp_002 | 512 | 64 | BM25 | 0.833 | 0.500 | 328s |
| exp_003 | 512 | 64 | Hybrid RRF | 1.000 | 0.667 | 199s |
| exp_004 | 256 | 0 | Dense | 0.917 | 0.667 | 233s |
| exp_005 | 256 | 0 | **BM25** | **1.000** | 0.667 | 199s |
| exp_006 | 256 | 0 | Hybrid RRF | 0.633 | 0.500 | 211s |

---

## 💡 Key Findings

**1. Zero Hallucinations**  
Hallucination rate = 0.000 across all 18 queries. Context-grounding prompting + honest abstention works.

**2. BM25 Wins with Small Chunks**  
BM25 + chunk=256 → faithfulness 1.000. Exact-match scoring excels on precise technical queries with distinctive keywords.

**3. Hybrid RRF Needs Large Chunks**  
chunk=512 + overlap=64 → faithfulness 1.000. chunk=256 + no overlap → faithfulness 0.633 (worst).
> ⚠️ Do not use Hybrid RRF with chunks smaller than ~384 tokens on technical corpora.

**4. Honest Abstention ≠ Failure**  
56% of queries triggered abstention — LLM correctly declining when answer isn't in corpus. This is desired behavior for a trustworthy system.

**5. Generation is Stable**  
Answer relevance = 0.800 uniformly across all configs. The bottleneck is retrieval, not generation.

---

## 🛠️ Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment setup | ✅ |
| 1 | Ingestion + Chunking (`src/ingestion/`) | ✅ |
| 2 | Dense + Sparse Retrieval (`src/retrieval/`) | ✅ |
| 3 | Hybrid RRF (`hybrid_retriever.py`) | ✅ |
| 4 | LLM Generation (`src/generation/`) | ✅ |
| 5 | Evaluation Pipeline (`src/evaluation/`) | ✅ |
| 6 | Real Dataset (7 ArXiv papers) | ✅ |
| 7 | Quick Ablation (3 experiments) | ✅ |
| 7b | Full Ablation (6 experiments, 241 min) | ✅ |
| 8 | Visualization (`visualize.py`, `notebooks/`) | ✅ |
| 9 | Error Analysis (`results/failure_cases/`) | ✅ |
| 10 | Final Report (`reports/`) | ✅ |

---

## 📝 Failure Mode Analysis

| Mode | Count | % | Meaning |
|------|-------|---|---------|
| `correct` | 6 | 33% | Perfect retrieval + generation |
| `honest_abstention` | 10 | 56% | Answer not in corpus (correct behavior ✅) |
| `partial_context` | 2 | 11% | Truncated chunk retrieved (chunking artifact) |
| `hallucination` | 0 | 0% | Never occurred 🎉 |

---

## 🔮 Future Work

- [ ] Sentence-boundary-aware chunking → eliminate `partial_context` failures
- [ ] Cross-encoder reranking (`experiments/reranking/`) → push ctx_relevance above 0.667
- [ ] Ground truth QA pairs → enable precision/recall metrics
- [ ] Larger query set (10+ per experiment) → statistical significance
- [ ] Test with Llama 3 / Mixtral → compare hallucination rates

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.12 |
| Vector Store | FAISS |
| Sparse Retrieval | BM25 (custom) |
| Embedding | all-MiniLM-L6-v2 (384-dim) |
| LLM | Mistral (Ollama, local) |
| Visualization | matplotlib, seaborn |
| Environment | WSL2 Ubuntu + venv |

---

## 📄 Full Report

- [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) — Mini-paper format (Abstract → Conclusion)
- [`reports/RAG_Final_Report.pdf`](reports/RAG_Final_Report.pdf) — PDF version

---

*RAG Research Project · February 2025 · [@malikimayzar](https://github.com/malikimayzar)*
