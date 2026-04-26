# PROJECT: RAG Research System

## GOAL

Build production-grade RAG system with:

* high retrieval precision
* reliable grounding (no hallucination)
* confidence-aware responses

## CURRENT ARCHITECTURE

Pipeline:
Query → Hybrid Retrieval (Dense + BM25) → RRF → Reranker → Filtering → LLM

## COMPONENTS

### Retrieval

* Dense: sentence-transformers (MiniLM)
* Sparse: BM25
* Fusion: RRF (k=60)
* Candidate pool: ~40

### Reranker

* Model: cross-encoder/ms-marco-MiniLM-L-6-v2
* Output: raw relevance score (can be negative)

### Filtering

* Layer 1: relevance > 0
* Layer 2: informative (heuristic)
* Layer 3: completeness (start/end check)
* Fallback: add lower-quality chunks if <3

## KNOWN PROBLEMS

* Title/author chunks still pass filtering
* Fragmented chunks ("and generation...") used
* Filtering logic inconsistent (layer mismatch)
* Confidence misaligned with reranker
* Chunking is not sentence-aware

## CURRENT BEHAVIOR

* Reranker correctly separates signal vs noise
* Only 1 truly good chunk per query
* System still injects noisy fallback chunks

## CONSTRAINTS

* Must remain fast (<2s total latency)
* No heavy models beyond current reranker
* Prefer data-quality fixes over model changes
