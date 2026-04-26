# Coverage Metric Report  ·  v2
_Generated: 2026-04-07 15:50 UTC  ·  faithfulness_mode: `hybrid`_

---
## Core Metrics

| Metric | Value | Bar |
|---|---|---|
| **Coverage** (answered/answerable) | `82.1%` | [████████████████░░░░] 82.1% |
| **Effective Coverage** (answered∧faithful/answerable) | `1.8%` | [░░░░░░░░░░░░░░░░░░░░] 1.8% |
| Gap Rate (hard_gap/total) | `13.9%` | [██░░░░░░░░░░░░░░░░░░] 13.9% |
| Retrieval Fail Rate | `9.2%` | [█░░░░░░░░░░░░░░░░░░░] 9.2% |
| Abstention Rate (abstained/answerable) | `7.1%` | [█░░░░░░░░░░░░░░░░░░░] 7.1% |

**Health:** 🟡 Needs attention

### Count Breakdown

| Class | Count | Meaning |
|---|---|---|
| answered | 46 | Retrieved + LLM answered |
| answered ∧ faithful | 1 | Answered + grounded in context |
| abstained | 4 | Retrieved but LLM refused |
| retrieval_fail | 6 | Low score (0–0.2), pipeline problem |
| hard_gap | 9 | Score ≤ 0, corpus gap |
| **total** | **65** | |
| faithfulness scored | 14 | queries with faith data |

### Latency

| Stage | Avg ms |
|---|---|
| Retrieval | `2993.1` |
| Generation | `876.4` |

---
## Actionable Insights

> ⚠️ **Coverage-Faithfulness Gap** = `80.3%` → kemungkinan hallucination.
> Action: audit per_query section, tighten SYSTEM_PROMPT.

---
## By Source

| Source | Total | Answered | Faithful | Abstained | Ret.Fail | Gap | Coverage | Eff.Cov |
|---|---|---|---|---|---|---|---|---|
| primary | 2 | 0 | 0 | 0 | 2 | 0 | 0.0% | 0.0% |
| logs | 11 | 0 | 0 | 0 | 4 | 7 | 0.0% | 0.0% |
| live | 52 | 46 | 1 | 4 | 0 | 2 | 92.0% | 2.0% |

---
## Non-Answered Queries

| Class | Score | Faith | Query | Source |
|---|---|---|---|---|
| retrieval_fail | 0.016261 | 1.00 | What is chunking and why does chunk size matter?… | primary |
| retrieval_fail | 0.016261 | 1.00 | What metrics evaluate RAG system quality?… | primary |
| retrieval_fail | 0.016261 | 1.00 | What is chunking and why does chunk size matter?… | log:generation_test.json |
| retrieval_fail | 0.016393 | 1.00 | How does hybrid retrieval combine BM25 and dense search?… | log:generation_test.json |
| retrieval_fail | 0.016261 | 1.00 | What metrics evaluate RAG system quality?… | log:generation_test.json |
| retrieval_fail | 0.016261 | 1.00 | What happens when the answer is not in the documents?… | log:generation_test.json |
| hard_gap | 0.0 | 1.00 | what is chunking in RAG?… | log:hybrid_search_test.json |
| hard_gap | 0.0 | 1.00 | how does hybrid retrieval work?… | log:hybrid_search_test.json |
| hard_gap | 0.0 | 1.00 | what metrics are used to evaluate RAG systems?… | log:hybrid_search_test.json |
| hard_gap | 0.0 | 1.00 | What is attention mechanism in transformer models?… | log:single_query_debug.json |
| hard_gap | 0.0 | 0.00 | What statistic was used to measure inter-rater reliability (IRR) … | log:single_query_debug.json |
| hard_gap | 0.0 | 1.00 | What is a notable issue that large language models (LLMs) face wh… | log:single_query_debug.json |
| hard_gap | 0.0 | — | How are FEVER class labels mapped for training in the context of … | log:single_query_debug.json |
| abstained | 0.5 | — | How does RAG compare to Fine-tuning (FT) in terms of its suitabil… | live_api |
| hard_gap | 0.0 | — | What characteristic of RAG allows it to achieve strong results wi… | live_api |
| abstained | 0.5 | — | What task is a large language model found to be good at, accordin… | live_api |
| abstained | 0.5 | — | How can RAG be fine-tuned for sequence-to-sequence tasks?… | live_api |
| hard_gap | 0.0 | — | Where can the illustration of the model's consistency be found?… | live_api |
| abstained | 0.5 | — | What type of language models are learned in the Zemi approach?… | live_api |

---
## Definition

```
coverage           = answered / answerable
effective_coverage = (answered AND faithful) / answerable
answerable         = total - hard_gap
hard_gap           = top_score <= 0.0   (data problem)
retrieval_fail     = 0.0 < top_score < 0.2  (pipeline problem)
abstained          = top_score >= 0.2 but LLM refused
faithful           = faithfulness_score >= 0.5
```

**hard_gap vs retrieval_fail:**
hard_gap = exclude dari denominator (bukan salah pipeline).
retrieval_fail = masuk denominator — pipeline harusnya bisa handle ini.

**coverage vs effective_coverage:**
Coverage bisa tinggi walau model hallucinating.
effective_coverage adalah metric yang lebih jujur untuk production readiness.

---
_Script: `scripts/coverage_metric.py` · rag-research v0.2.0_