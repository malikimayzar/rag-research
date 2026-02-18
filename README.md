# 🔍 RAG System Analysis
### Ablation Study on Chunking Strategies and Retrieval Methods

> **Maliki Mayzar** · February 2025  
> A complete Retrieval-Augmented Generation (RAG) pipeline built and evaluated from scratch across 8 development phases.

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
│   ├── raw/                    # 7 ArXiv PDFs (Tier 1–3)
│   ├── processed/
│   │   ├── index_bm25/         # BM25 inverted index
│   │   └── index_minilm/       # FAISS vector index
│   └── adversarial/            # Adversarial query set
├── src/
│   ├── ingestion/              # PDF parsing, chunking
│   ├── retrieval/              # Dense + sparse retrieval
│   ├── generation/             # Mistral LLM interface
│   └── evaluation/             # Metrics & scoring
├── experiments/
│   ├── reranking/              # Cross-encoder reranking
│   └── ablation_runner.py      # Main ablation script
├── notebooks/
│   └── analysis.ipynb          # Visualization notebook
├── results/
│   ├── figures/                # 6 visualization plots
│   ├── logs/                   # ablation_full.log
│   ├── metrics/                # JSON results
│   └── failure_cases/          # Failure analysis
└── reports/
    ├── README.md               # This file
    ├── FINAL_REPORT.md         # Full mini-paper
    └── RAG_Final_Report.pdf    # PDF version
```

---

## 🏗️ System Architecture

```
PDF Documents
     │
     ▼
┌─────────────┐
│   Chunker   │  fixed_size strategy, configurable chunk_size + overlap
└──────┬──────┘
       │
  ┌────┴────┐
  ▼         ▼
┌──────┐  ┌──────┐
│FAISS │  │ BM25 │  all-MiniLM-L6-v2 (384-dim) + JSON inverted index
└──────┘  └──────┘
  ▼         ▼
┌─────────────┐
│  RRF Fusion │  Reciprocal Rank Fusion (k=60)
└──────┬──────┘
       ▼
┌─────────────┐
│  Mistral    │  Local LLM via Ollama, context-grounded prompting
└──────┬──────┘
       ▼
    Answer
```

---

## 📚 Dataset

7 ArXiv papers organized in 3 tiers:

| Tier | Paper ID | Topic |
|------|----------|-------|
| 1 | 2005.11401 | RAG (Lewis et al.) |
| 1 | 2312.10997 | Advanced RAG techniques |
| 1 | tier1_test_intro | Custom intro document |
| 2 | 2210.11610 | Retrieval methods |
| 2 | 2212.10560 | Dense retrieval |
| 3 | 1706.03762 | Attention Is All You Need |
| 3 | 2307.09288 | LLM survey |

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama + Mistral (local LLM)
ollama pull mistral
```

### Run Pipeline
```bash
# Step 1 — Ingest documents
python src/ingestion/document_loader.py

# Step 2 — Build indexes
python src/ingestion/chunker.py

# Step 3 — Quick ablation (3 experiments)
python experiments/ablation_runner.py

# Step 4 — Full ablation (6 experiments, ~4 hours)
python experiments/ablation_runner.py --full

# Step 5 — Visualize results
python visualize.py
```

### View Results
```bash
# Quick summary
python3 -c "
import json
data = json.load(open('results/metrics/ablation_final.json'))
for e in sorted(data, key=lambda x: -x['avg_faithfulness']):
    print(f\"{e['config']['exp_id']} | {e['config']['retrieval_method']:7} | \
chunk={e['config']['chunk_size']} | F:{e['avg_faithfulness']:.3f} \
C:{e['avg_context_relevance']:.3f}\")
"
```

---

## 📈 Visualizations

All figures are in `results/figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_leaderboard.png` | Faithfulness & context relevance ranking |
| `fig2_chunksize.png` | Chunk size 256 vs 512 per method |
| `fig3_heatmap.png` | Full metrics heatmap |
| `fig4_failure_modes.png` | Correct / partial / abstention breakdown |
| `fig5_quality_latency.png` | Quality vs speed trade-off bubble chart |
| `fig6_radar.png` | Method comparison radar chart |

---

## 🔬 Ablation Configuration

6 experiments × 3 queries = **18 total evaluations**

| Exp | Chunk | Overlap | Method | Total Runtime |
|-----|-------|---------|--------|--------------|
| exp_001 | 512 | 64 | Dense | — |
| exp_002 | 512 | 64 | BM25 | — |
| exp_003 | 512 | 64 | Hybrid RRF | — |
| exp_004 | 256 | 0 | Dense | — |
| exp_005 | 256 | 0 | BM25 | — |
| exp_006 | 256 | 0 | Hybrid RRF | — |
| **Total** | | | | **241.2 minutes** |

---

## 💡 Key Findings

### 1. Zero Hallucinations
Across all 18 queries and 6 configurations, hallucination rate = **0.000**. Context-grounding prompting with honest abstention works.

### 2. BM25 Wins with Small Chunks
BM25 + chunk=256 achieves perfect faithfulness (1.000). For precise technical queries with distinctive keywords, exact-match scoring outperforms semantic retrieval.

### 3. Hybrid RRF Requires Large Chunks
Hybrid RRF with chunk=512+overlap=64 → faithfulness=1.000.  
Hybrid RRF with chunk=256+no overlap → faithfulness=0.633 (worst).

> **Rule of thumb: Do not use Hybrid RRF with chunks smaller than ~384 tokens.**

### 4. Honest Abstention ≠ Failure
56% of queries triggered "honest abstention" — the LLM correctly declining to answer because the fact wasn't in the corpus. This is the **desired behavior** for a trustworthy system.

### 5. Answer Relevance is Always 0.800
Generation quality is stable regardless of retrieval config. The bottleneck is retrieval, not generation.

---

## 🛠️ Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment setup | ✅ |
| 1 | Ingestion + Chunking | ✅ |
| 2 | Dense + Sparse Retrieval | ✅ |
| 3 | Hybrid RRF | ✅ |
| 4 | LLM Generation (Mistral) | ✅ |
| 5 | Evaluation Pipeline | ✅ |
| 6 | Real Dataset (7 ArXiv papers) | ✅ |
| 7 | Quick Ablation (3 experiments) | ✅ |
| 7b | Full Ablation (6 experiments) | ✅ |
| 8 | Visualization Notebook | ✅ |
| 9 | Error Analysis | ✅ |
| 10 | Final Report | ✅ |

---

## 📝 Failure Mode Analysis

| Mode | Count | % | Meaning |
|------|-------|---|---------|
| `correct` | 6 | 33% | Perfect retrieval + generation |
| `honest_abstention` | 10 | 56% | Answer not in corpus (correct behavior) |
| `partial_context` | 2 | 11% | Chunking artifact (truncated sentence retrieved) |
| `hallucination` | 0 | 0% | Never occurred |

Root causes:
- **partial_context** → Fixed-size chunking splits sentences mid-word. Fix: sentence-boundary-aware chunking.
- **honest_abstention** → Out-of-corpus queries. Not a bug — this is the system working correctly.

---

## 🔮 Future Work

- [ ] Sentence-boundary-aware chunking to eliminate partial_context failures
- [ ] Cross-encoder reranking (`experiments/reranking/`) to push ctx_relevance above 0.667
- [ ] Ground truth QA pairs for precision/recall evaluation
- [ ] Larger query set (10+ per experiment) for statistical significance
- [ ] Test with larger LLMs (Llama 3, Mixtral)

---

## 📄 Report

Full analysis available in:
- [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) — Mini-paper format (Abstract → Conclusion)
- [`reports/RAG_Final_Report.pdf`](reports/RAG_Final_Report.pdf) — PDF version

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.12 |
| Vector Store | FAISS |
| Sparse Retrieval | BM25 (custom) |
| Embedding Model | all-MiniLM-L6-v2 |
| LLM | Mistral (Ollama) |
| Visualization | matplotlib, seaborn |
| PDF Generation | reportlab |
| Environment | WSL2 Ubuntu + venv |

---

*RAG Research Project · February 2025*