from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("coverage_metric")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_FILE = PROJECT_ROOT / "results/failure_cases/failure_analysis.json"
LOG_DIR      = PROJECT_ROOT / "results/logs"
RAGAS_DIR    = PROJECT_ROOT / "results/metrics"
GT_FILE      = PROJECT_ROOT / "data/processed/ground_truth_qa.json"
OUTPUT_DIR   = PROJECT_ROOT / "results/debug"
OUTPUT_JSON  = OUTPUT_DIR / "coverage_report.json"
OUTPUT_MD    = OUTPUT_DIR / "coverage_report.md"

# ── Thresholds ────────────────────────────────────────────────────────────────
HARD_GAP_THRESHOLD       = 0.0   
RETRIEVAL_FAIL_THRESHOLD = 0.2  
FAITHFULNESS_THRESHOLD   = 0.5   
DEFAULT_API_URL          = "http://localhost:8003"
DEFAULT_MAX_WORKERS      = 3

ABSTENTION_PHRASES = [
    "does not contain enough information",
    "cannot find",
    "not mentioned in",
    "no information",
    "context does not",
    "not found in",
    "cannot answer",
    "unable to find",
    "not available in",
    "no relevant information",
]

FAITHFULNESS_JUDGE_PROMPT = """You are evaluating a RAG system answer.

QUESTION: {query}

CONTEXT:
{context}

ANSWER: {answer}

Task: Is the answer faithful to the context?
- Faithful = answer only uses information present in context, no hallucination
- Score 1.0 = fully faithful
- Score 0.5 = partially faithful (some hallucination)
- Score 0.0 = not faithful (mostly hallucinated)

Respond with ONLY a JSON object, nothing else:
{{"faithful": true/false, "score": 0.0-1.0, "reason": "one sentence"}}"""


# 1. CLASSIFIERS

def get_top_score(record: dict) -> float:
    chunks = record.get("retrieved_chunks", [])
    if not chunks:
        return 0.0
    return max(
        c.get("retrieval_score", c.get("score", 0.0))
        for c in chunks
    )


def is_abstention(answer: str) -> bool:
    if not answer or len(answer.strip()) < 10:
        return True
    a = answer.lower().strip()
    if any(phrase in a for phrase in ABSTENTION_PHRASES):
        return True
    if len(a.split()) < 5:
        return True
    return False


def classify_retrieval(top_score: float) -> str:
    if top_score <= HARD_GAP_THRESHOLD:
        return "hard_gap"
    elif top_score < RETRIEVAL_FAIL_THRESHOLD:
        return "retrieval_fail"
    return "retrieved"


def classify_record(record: dict) -> str:
    top_score       = get_top_score(record)
    retrieval_class = classify_retrieval(top_score)

    if retrieval_class in ("hard_gap", "retrieval_fail"):
        return retrieval_class

    if is_abstention(record.get("answer", "")):
        return "abstained"

    return "answered"


# 2. FAITHFULNESS EVALUATOR
class FaithfulnessEvaluator:
    def __init__(self, mode: str = "offline", sample_size: int = 20):
        self.mode         = mode
        self.sample_size  = sample_size
        self._cache: dict[str, float] = {}
        self._groq        = None
        self._judge_calls = 0
        self._judge_429   = 0

        if mode in ("offline", "hybrid"):
            self._load_ragas_cache()
        if mode in ("inline", "hybrid"):
            self._init_groq()

    def _load_ragas_cache(self):
        ragas_files = list(RAGAS_DIR.glob("ragas*.json"))
        if not ragas_files:
            logger.warning("No ragas_*.json found in %s", RAGAS_DIR)
            return

        loaded = 0
        for rf in ragas_files:
            try:
                with open(rf) as f:
                    data = json.load(f)

                items = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = (
                        data.get("results")
                        or data.get("scores")
                        or data.get("data")
                        or []
                    )

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    q = item.get("question") or item.get("query") or item.get("user_input")
                    f = item.get("faithfulness") or item.get("faithfulness_score")
                    if q and f is not None:
                        self._cache[q.strip()] = float(f)
                        loaded += 1

            except Exception as e:
                logger.warning("Failed loading %s: %s", rf.name, e)
        logger.info("Faithfulness cache: %d entries from %d files", loaded, len(ragas_files))

    def _init_groq(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — inline judge disabled")
            return
        try:
            from groq import Groq
            self._groq = Groq(api_key=api_key)
            logger.info("Inline judge ready (llama-3.1-8b-instant, sample_size=%d)", self.sample_size)
        except ImportError:
            logger.warning("groq not installed — inline judge disabled")

    def _judge_inline(self, query: str, answer: str, contexts: list[str]) -> Optional[float]:
        if not self._groq:
            return None

        context_text = "\n\n".join(contexts[:3])[:2000]
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(
            query=query, context=context_text, answer=answer
        )

        try:
            self._judge_calls += 1
            resp = self._groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            score  = float(parsed.get("score", 0.0))
            logger.info(
                "Judge [%d/%d]: %.2f | %s | %s",
                self._judge_calls, self.sample_size,
                score, parsed.get("reason", "")[:50], query[:40],
            )
            return score

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                self._judge_429 += 1
                logger.warning("Rate limited (429 #%d) — sleep 2s", self._judge_429)
                time.sleep(2)
            else:
                logger.warning("Judge error: %s", e)
            return None

    def score(self, record: dict) -> Optional[float]:
        query    = record.get("query", "").strip()
        answer   = record.get("answer", "")
        contexts = [c.get("text", "") for c in record.get("retrieved_chunks", [])]

        if self.mode == "offline":
            return self._cache.get(query)

        if self.mode == "inline":
            if self._judge_calls >= self.sample_size:
                return None
            return self._judge_inline(query, answer, contexts)

        if self.mode == "hybrid":
            cached = self._cache.get(query)
            if cached is not None:
                return cached
            if self._judge_calls < self.sample_size:
                return self._judge_inline(query, answer, contexts)
            return None
        return None

    def score_batch(self, records: list[dict]) -> dict[str, Optional[float]]:
        results: dict[str, Optional[float]] = {}
        for r in records:
            q = r.get("query", "")
            if q not in results:
                results[q] = self.score(r)

        scored = sum(1 for v in results.values() if v is not None)
        logger.info(
            "Faithfulness: %d/%d scored (mode=%s, judge_calls=%d, 429s=%d)",
            scored, len(results), self.mode, self._judge_calls, self._judge_429,
        )
        return results


# 3. LOADERS
def load_primary(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("Primary not found: %s", path)
        return []
    with open(path) as f:
        data = json.load(f)
    records = data if isinstance(data, list) else [data]
    for r in records:
        r["_source"] = "primary"
    logger.info("Primary  : %d records ← %s", len(records), path.name)
    return records


def load_secondary(log_dir: Path) -> list[dict]:
    records   = []
    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        logger.warning("No log files in %s", log_dir)
        return []

    seen = set()
    for lf in log_files:
        try:
            with open(lf) as f:
                data = json.load(f)
            items  = data if isinstance(data, list) else [data]
            loaded = 0
            for item in items:
                q = item.get("query", "")
                if q in seen:
                    continue
                seen.add(q)
                item["_source"] = f"log:{lf.name}"
                records.append(item)
                loaded += 1
            logger.info("Secondary: %d records ← %s", loaded, lf.name)
        except Exception as e:
            logger.warning("Skipping %s: %s", lf.name, e)
    return records


def load_queries_for_live(gt_path: Path) -> list[str]:
    if not gt_path.exists():
        logger.warning("Query file not found: %s", gt_path)
        return []
    with open(gt_path) as f:
        data = json.load(f)
    queries = []
    for item in data:
        if isinstance(item, str):
            queries.append(item)
        elif isinstance(item, dict):
            q = item.get("question") or item.get("query") or item.get("q")
            if q:
                queries.append(q)
    logger.info("Loaded %d queries ← %s", len(queries), gt_path.name)
    return queries

# 4. LIVE API RUNNER  (concurrent + observability)
def call_generate_api(query: str, api_url: str, timeout: int = 30) -> Optional[dict]:
    url     = f"{api_url.rstrip('/')}/generate"
    payload = json.dumps({"query": query, "top_k": 5, "use_reranker": True}).encode()
    req     = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        logger.warning("HTTP %d: %s", e.code, query[:50])
        return None
    except Exception as e:
        logger.warning("API error: %s", e)
        return None

def _normalize_api_response(query: str, resp: dict, elapsed_ms: float) -> dict:
    method   = resp.get("retrieval_method", "")
    contexts = resp.get("contexts", [])

    if method == "rejected_low_score":
        chunks = []
    elif not contexts:
        chunks = []
    else:
        chunks = [{"text": ctx, "retrieval_score": 0.5} for ctx in contexts]

    return {
        "_source":               "live_api",
        "query":                 query,
        "answer":                resp.get("answer", ""),
        "retrieval_method":      method,
        "latency_retrieval_ms":  resp.get("latency_retrieval_ms", elapsed_ms),
        "latency_generation_ms": resp.get("latency_generation_ms", 0),
        "retrieved_chunks":      chunks,
    }

def run_live(queries: list[str], api_url: str, max_workers: int = DEFAULT_MAX_WORKERS) -> list[dict]:
    if not queries:
        logger.warning("No queries for live run")
        return []

    # Health check
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=5) as r:
            status = json.loads(r.read())
        logger.info("API health OK — vectors=%s", status.get("vectors", "?"))
    except Exception as e:
        logger.error("API unreachable at %s: %s", api_url, e)
        logger.error("Start API: uvicorn src.api.main:app --host 0.0.0.0 --port 8003")
        sys.exit(1)

    logger.info("Live run: %d queries | max_workers=%d", len(queries), max_workers)

    records   = []
    success   = 0
    failed    = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {
            executor.submit(call_generate_api, q, api_url): q
            for q in queries
        }

        for i, future in enumerate(as_completed(future_to_query), 1):
            query = future_to_query[future]
            try:
                t0   = time.time()
                resp = future.result()
                elapsed = round((time.time() - t0) * 1000, 1)

                if resp is None:
                    failed += 1
                    logger.warning("[%d/%d] FAILED: %s", i, len(queries), query[:50])
                    continue

                record = _normalize_api_response(query, resp, elapsed)
                cls    = classify_record(record)
                success += 1
                logger.info(
                    "[%d/%d] %-14s | %5.0fms | %s",
                    i, len(queries), cls.upper(),
                    record["latency_retrieval_ms"], query[:55],
                )
                records.append(record)

            except Exception as e:
                failed += 1
                logger.warning("[%d/%d] Exception: %s — %s", i, len(queries), e, query[:40])

    total_ms = round((time.time() - start_all) * 1000)
    logger.info(
        "Live done: success=%d failed=%d total=%dms avg=%.0fms/query",
        success, failed, total_ms, total_ms / max(success, 1),
    )
    return records

# 5. AGGREGATION
def compute_coverage(
    records: list[dict],
    faith_scores: Optional[dict[str, Optional[float]]] = None,
) -> dict:
    if not records:
        return _empty_coverage()

    counts    = defaultdict(int)
    per_query = []
    lat_ret   = []
    lat_gen   = []
    answered_and_faithful = 0

    for r in records:
        cls       = classify_record(r)
        query     = r.get("query", "")
        top_score = get_top_score(r)
        counts[cls] += 1

        faith = faith_scores.get(query.strip()) if faith_scores else None
        is_faithful = faith is not None and faith >= FAITHFULNESS_THRESHOLD

        if cls == "answered" and is_faithful:
            answered_and_faithful += 1

        lr = r.get("latency_retrieval_ms")
        lg = r.get("latency_generation_ms") or r.get("latency_generation")
        if lr:
            lat_ret.append(float(lr))
        if lg:
            lat_gen.append(float(lg))

        per_query.append({
            "query":        query[:100],
            "class":        cls,
            "top_score":    round(top_score, 6),
            "abstention":   is_abstention(r.get("answer", "")),
            "faithfulness": round(faith, 3) if faith is not None else None,
            "faithful":     is_faithful if faith is not None else None,
            "source":       r.get("_source", "unknown"),
        })

    total      = len(records)
    hard_gap   = counts["hard_gap"]
    ret_fail   = counts["retrieval_fail"]
    abstained  = counts["abstained"]
    answered   = counts["answered"]
    answerable = total - hard_gap  

    faith_available = sum(1 for q in per_query if q["faithfulness"] is not None)

    return {
        "total":                    total,
        "answered":                 answered,
        "abstained":                abstained,
        "retrieval_fail":           ret_fail,
        "hard_gap":                 hard_gap,
        "answerable":               answerable,
        "answered_and_faithful":    answered_and_faithful,
        "coverage":                 round(answered / answerable, 4) if answerable > 0 else None,
        "effective_coverage":       round(answered_and_faithful / answerable, 4)
                                    if answerable > 0 and faith_available > 0 else None,
        "gap_rate":                 round(hard_gap / total, 4) if total > 0 else None,
        "retrieval_fail_rate":      round(ret_fail / total, 4) if total > 0 else None,
        "abstention_rate":          round(abstained / answerable, 4) if answerable > 0 else None,
        "faithfulness_available":   faith_available,
        "avg_latency_retrieval_ms": round(sum(lat_ret) / len(lat_ret), 1) if lat_ret else None,
        "avg_latency_generation_ms": round(sum(lat_gen) / len(lat_gen), 1) if lat_gen else None,
        "per_query":                per_query,
    }

def _empty_coverage() -> dict:
    return {
        "total": 0, "answered": 0, "abstained": 0,
        "retrieval_fail": 0, "hard_gap": 0, "answerable": 0,
        "answered_and_faithful": 0, "coverage": None, "effective_coverage": None,
        "gap_rate": None, "retrieval_fail_rate": None, "abstention_rate": None,
        "faithfulness_available": 0,
        "avg_latency_retrieval_ms": None, "avg_latency_generation_ms": None,
        "per_query": [],
    }

def split_by_source(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        src = r.get("_source", "unknown")
        key = "primary" if src == "primary" else \
              "logs"    if src.startswith("log:") else \
              "live"    if src == "live_api" else "other"
        groups[key].append(r)
    return dict(groups)


# 6. RENDERERS
def _bar(value: Optional[float], width: int = 20) -> str:
    if value is None:
        return "N/A"
    filled = int(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value * 100:.1f}%"

def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"

def render_json(overall: dict, by_source: dict, ts: str, faith_mode: str) -> dict:
    return {
        "generated_at":      ts,
        "faithfulness_mode": faith_mode,
        "thresholds": {
            "hard_gap":          HARD_GAP_THRESHOLD,
            "retrieval_fail":    RETRIEVAL_FAIL_THRESHOLD,
            "faithfulness_min":  FAITHFULNESS_THRESHOLD,
        },
        "definition": {
            "coverage":           "answered / answerable",
            "effective_coverage": "(answered AND faithful) / answerable",
            "answerable":         "total - hard_gap",
            "hard_gap":           "top_score <= 0.0",
            "retrieval_fail":     "0.0 < top_score < 0.2",
            "abstained":          "top_score >= 0.2 but LLM abstained",
        },
        "overall":   {k: v for k, v in overall.items() if k != "per_query"},
        "by_source": {
            src: {k: v for k, v in m.items() if k != "per_query"}
            for src, m in by_source.items()
        },
        "per_query": overall.get("per_query", []),
    }

def render_md(overall: dict, by_source: dict, ts: str, faith_mode: str) -> str:
    cov     = overall["coverage"]
    eff_cov = overall["effective_coverage"]
    gap     = overall["gap_rate"]
    ret_f   = overall["retrieval_fail_rate"]
    abst    = overall["abstention_rate"]
    faith_n = overall["faithfulness_available"]

    if cov is None:
        health = "[WARN]  No answerable data"
    elif cov >= 0.90 and (eff_cov is None or eff_cov >= 0.80):
        health = "OK Healthy"
    elif cov >= 0.75:
        health = "[OK] Needs attention"
    else:
        health = "[OK] Critical"

    lines = [
        "# Coverage Metric Report  ·  v2",
        f"_Generated: {ts}  ·  faithfulness_mode: `{faith_mode}`_",
        "", "---", "## Core Metrics", "",
        "| Metric | Value | Bar |",
        "|---|---|---|",
        f"| **Coverage** (answered/answerable) | `{_pct(cov)}` | {_bar(cov)} |",
        f"| **Effective Coverage** (answered∧faithful/answerable) | `{_pct(eff_cov)}` | {_bar(eff_cov)} |",
        f"| Gap Rate (hard_gap/total) | `{_pct(gap)}` | {_bar(gap)} |",
        f"| Retrieval Fail Rate | `{_pct(ret_f)}` | {_bar(ret_f)} |",
        f"| Abstention Rate (abstained/answerable) | `{_pct(abst)}` | {_bar(abst)} |",
        "", f"**Health:** {health}", "",
        "### Count Breakdown", "",
        "| Class | Count | Meaning |",
        "|---|---|---|",
        f"| answered | {overall['answered']} | Retrieved + LLM answered |",
        f"| answered ∧ faithful | {overall['answered_and_faithful']} | Answered + grounded in context |",
        f"| abstained | {overall['abstained']} | Retrieved but LLM refused |",
        f"| retrieval_fail | {overall['retrieval_fail']} | Low score (0–0.2), pipeline problem |",
        f"| hard_gap | {overall['hard_gap']} | Score ≤ 0, corpus gap |",
        f"| **total** | **{overall['total']}** | |",
        f"| faithfulness scored | {faith_n} | queries with faith data |",
        "",
    ]

    lr = overall["avg_latency_retrieval_ms"]
    lg = overall["avg_latency_generation_ms"]
    if lr or lg:
        lines += [
            "### Latency", "",
            "| Stage | Avg ms |", "|---|---|",
            f"| Retrieval | `{lr}` |",
            f"| Generation | `{lg}` |", "",
        ]

    lines += ["---", "## Actionable Insights", ""]

    if gap and gap > 0.15:
        lines += [
            "> ⚠️ **High Gap Rate** → corpus tidak cover topik yang ditanya.",
            "> Action: jalanin `corpus_audit.py`, tambah dokumen yang missing.", "",
        ]
    if ret_f and ret_f > 0.10:
        lines += [
            "> ⚠️ **High Retrieval Fail Rate** → chunk ada tapi retrieval miss.",
            "> Action: tune `default_top_k`, chunk overlap, reranker threshold.", "",
        ]
    if abst and abst > 0.10:
        lines += [
            "> ⚠️ **High Abstention Rate** → model terlalu konservatif.",
            "> Action: review `SYSTEM_PROMPT` di generator.py, cek early reject threshold.", "",
        ]
    if eff_cov is not None and cov is not None and (cov - eff_cov) > 0.15:
        lines += [
            f"> ⚠️ **Coverage-Faithfulness Gap** = `{_pct(cov - eff_cov)}` → kemungkinan hallucination.",
            "> Action: audit per_query section, tighten SYSTEM_PROMPT.", "",
        ]
    if faith_n == 0:
        lines += [
            f"> ℹ️  Effective coverage tidak tersedia — tidak ada faithfulness data.",
            f"> Action: run dengan `--faithfulness-mode offline` atau `inline --sample 20`", "",
        ]

    if by_source:
        lines += [
            "---", "## By Source", "",
            "| Source | Total | Answered | Faithful | Abstained | Ret.Fail | Gap | Coverage | Eff.Cov |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for src, m in by_source.items():
            lines.append(
                f"| {src} | {m['total']} | {m['answered']} | {m['answered_and_faithful']} "
                f"| {m['abstained']} | {m['retrieval_fail']} | {m['hard_gap']} "
                f"| {_pct(m['coverage'])} | {_pct(m['effective_coverage'])} |"
            )
        lines.append("")

    non_answered = [q for q in overall.get("per_query", []) if q["class"] != "answered"]
    if non_answered:
        lines += [
            "---", "## Non-Answered Queries", "",
            "| Class | Score | Faith | Query | Source |",
            "|---|---|---|---|---|",
        ]
        for q in non_answered[:20]:
            f_str = f"{q['faithfulness']:.2f}" if q["faithfulness"] is not None else "—"
            lines.append(
                f"| {q['class']} | {q['top_score']} | {f_str} "
                f"| {q['query'][:65]}… | {q['source']} |"
            )
        lines.append("")

    lines += [
        "---", "## Definition", "",
        "```",
        "coverage           = answered / answerable",
        "effective_coverage = (answered AND faithful) / answerable",
        "answerable         = total - hard_gap",
        "hard_gap           = top_score <= 0.0   (data problem)",
        "retrieval_fail     = 0.0 < top_score < 0.2  (pipeline problem)",
        "abstained          = top_score >= 0.2 but LLM refused",
        "faithful           = faithfulness_score >= 0.5",
        "```", "",
        "**hard_gap vs retrieval_fail:**",
        "hard_gap = exclude dari denominator (bukan salah pipeline).",
        "retrieval_fail = masuk denominator — pipeline harusnya bisa handle ini.", "",
        "**coverage vs effective_coverage:**",
        "Coverage bisa tinggi walau model hallucinating.",
        "effective_coverage adalah metric yang lebih jujur untuk production readiness.", "",
        "---",
        "_Script: `scripts/coverage_metric.py` · rag-research v0.2.0_",
    ]

    return "\n".join(lines)


# 7. MAIN
def main():
    parser = argparse.ArgumentParser(
        description="RAG Coverage Metric v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--live", action="store_true",
                        help="Hit live API untuk setiap query")
    parser.add_argument("--queries", type=Path, default=GT_FILE,
                        help="Query file untuk live test")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Concurrent workers (tune ke rate limit Groq)")
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument(
        "--faithfulness-mode", choices=["offline", "inline", "hybrid"], default="offline",
        help="offline=ragas files | inline=LLM judge (sampling) | hybrid=offline+inline fallback",
    )
    parser.add_argument("--sample", type=int, default=20,
                        help="Max inline LLM judge calls")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    primary_records   = load_primary(PRIMARY_FILE)
    secondary_records = [] if args.primary_only else load_secondary(LOG_DIR)
    live_records      = run_live(load_queries_for_live(args.queries), args.api_url, args.max_workers) \
                        if args.live else []

    all_records = primary_records + secondary_records + live_records
    if not all_records:
        logger.error("No records found. Check paths or use --live.")
        sys.exit(1)

    logger.info(
        "Total: %d records (%d primary, %d logs, %d live)",
        len(all_records), len(primary_records), len(secondary_records), len(live_records),
    )

    # Faithfulness
    evaluator    = FaithfulnessEvaluator(mode=args.faithfulness_mode, sample_size=args.sample)
    faith_scores = evaluator.score_batch(all_records)

    # Compute
    overall   = compute_coverage(all_records, faith_scores)
    by_source = {
        src: compute_coverage(recs, faith_scores)
        for src, recs in split_by_source(all_records).items()
    }

    # Write
    ts       = datetime.now(timezone.utc).isoformat()
    ts_human = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(render_json(overall, by_source, ts, args.faithfulness_mode), f, indent=2)
    logger.info("Wrote → %s", OUTPUT_JSON)

    with open(OUTPUT_MD, "w") as f:
        f.write(render_md(overall, by_source, ts_human, args.faithfulness_mode))
    logger.info("Wrote → %s", OUTPUT_MD)

    # Summary
    logger.info("─" * 60)
    logger.info("COVERAGE REPORT  (mode=%s)", args.faithfulness_mode)
    logger.info("─" * 60)
    logger.info("  Coverage           %s", _bar(overall["coverage"]))
    logger.info("  Effective Coverage %s", _bar(overall["effective_coverage"]))
    logger.info("  Gap Rate           %s", _bar(overall["gap_rate"]))
    logger.info("  Retrieval Fail     %s", _bar(overall["retrieval_fail_rate"]))
    logger.info("  Abstention Rate    %s", _bar(overall["abstention_rate"]))
    logger.info("─" * 60)
    logger.info(
        "  total=%-4d  answered=%-4d  faithful=%-4d  abstained=%-4d  ret_fail=%-4d  gap=%-4d",
        overall["total"], overall["answered"], overall["answered_and_faithful"],
        overall["abstained"], overall["retrieval_fail"], overall["hard_gap"],
    )

    if args.verbose:
        logger.info("PER-QUERY BREAKDOWN:")
        for q in overall["per_query"]:
            f_s = f"faith={q['faithfulness']:.2f}" if q["faithfulness"] is not None else "faith=N/A"
            logger.info("  [%-14s] score=%.4f  %s  %s", q["class"], q["top_score"], f_s, q["query"][:55])


if __name__ == "__main__":
    main()