import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger("rag.confidence_engine")

class ConfidenceEngine:
    def __init__(self):
        self.reject_threshold = 0.25
        self.partial_threshold = 0.45

    def _normalize_scores(self, scores) -> np.ndarray:
        arr = np.array(scores)
        min_v = arr.min()
        max_v = arr.max()
        if max_v == min_v:
            return np.ones_like(arr)
        return (arr - min_v) / (max_v - min_v)
    
    def _sigmoid(self, x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def _calculate_entropy(self, scores: List[float]) -> float:
        if scores is None or len(scores) < 2:
            return 1.0
        arr = np.array(scores)
        exp_scores = np.exp(arr - np.max(arr))
        probs = exp_scores / exp_scores.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        return float(entropy / np.log(len(probs)))

    def _get_chunk_score(self, chunk: Any) -> float:
        if isinstance(chunk, dict):
            score = chunk.get("retrieval_score")
            if score is None:
                score = chunk.get("rerank_score", 0.0)
        else:
            score = getattr(chunk, "retrieval_score", None)
            if score is None:
                score = getattr(chunk, "rerank_score", 0.0)
        return float(score)

    def _source_agreement(self, chunks) -> float:
        sources = []
        for c in chunks[:5]:
            meta = c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {})
            sources.append(meta.get("retrieval_method", "unknown"))
        if not sources:
            return 0.0
        unique = len(set(sources))
        total = len(sources)
        return 1.0 - ((unique - 1) / total)
    
    def _detect_patterns(self, top_score: float, mean_top3: float, gap: float) -> dict:
        return {
            "spurious_match": top_score > 0.8 and mean_top3 < 0.5,
            "unstable_retrieval": gap > 0.8 and mean_top3 < 0.5,
        }
    
    def calculate_confidence(self, chunks: list[Any]) -> Dict[str, Any]:
        if not chunks:
            return {
                "confidence_score": 0.0,
                "decision": "REJECT",
                "signals": {
                    "top_score": 0.0,
                    "gap": 0.0,
                    "mean_top3": 0.0,
                    "score_level": 0.0,
                    "entropy": 1.0,
                    "agreement": 0.0,
                    "spurious_match": False,
                    "unstable_retrieval": False,
                }
            }
        raw_scores = [
            s for s in (self._get_chunk_score(c) for c in chunks)
            if isinstance(s, (int, float))
        ]

        if not raw_scores:
            return {
                "confidence_score": 0.0,
                "decision": "REJECT",
                "signals": {"reason": "no_valid_scores"}
            }
        
        normalized_scores = list(self._normalize_scores(raw_scores))

        top_score = raw_scores[0]
        mean_top3 = float(np.mean(raw_scores[:3])) if len(raw_scores) >= 3 else top_score
        top_score_norm = self._sigmoid(top_score)
        mean_top3_norm = self._sigmoid(mean_top3)
        gap = (normalized_scores[0] - normalized_scores[1]) / (abs(normalized_scores[0]) + 1e-6) if len(normalized_scores) > 1 else 0.0
        gap = max(0.0, min(gap, 1.0))
        score_level = min(self._sigmoid(mean_top3) / 0.8, 1.0)
        entropy = self._calculate_entropy(normalized_scores)
        agreement = self._source_agreement(chunks)

        confidence_score = (
            0.35 * top_score_norm +
            0.25 * mean_top3_norm +
            0.15 * gap +
            0.10 * agreement +
            0.15 * score_level
        )

        logger.info(
            f"[CONFIDENCE_BREAKDOWN] top_score={top_score:.4f} | mean_top3={mean_top3:.4f} | "
            f"gap={gap:.4f} | entropy={entropy:.4f} | agreement={agreement:.4f} | score_level={score_level:.4f}"
        )
        logger.info(
            f"[DEBUG_CONFIDENCE] raw top_score={top_score} | mean_top3={mean_top3} | "
            f"confidence_score={confidence_score} | reject_threshold={self.reject_threshold}"
        )
        logger.info(
            f"[DEBUG_NORM] top_norm={top_score_norm:.4f} | mean_norm={mean_top3_norm:.4f}"
        )
        patterns = self._detect_patterns(top_score, mean_top3, gap)

        if patterns["spurious_match"] and patterns["unstable_retrieval"]:
            decision = "REJECT"
        elif patterns["spurious_match"]:
            decision = "PARTIAL_TRUST"
        else:
            if confidence_score < self.reject_threshold:
                decision = "REJECT"
            elif confidence_score < self.partial_threshold:
                decision = "PARTIAL_TRUST"
            else:
                decision = "GENERATE"

        logger.info(f"[CONFIDENCE_RESULT] confidence_score={confidence_score:.4f} | decision={decision}")

        return {
            "confidence_score": round(float(confidence_score), 4),
            "decision": decision,
            "signals": {
                "top_score": round(top_score, 4),
                "gap": round(gap, 4),
                "mean_top3": round(mean_top3, 4),
                "score_level": round(score_level, 4),
                "entropy": round(entropy, 4),
                "agreement": round(agreement, 4),
                "spurious_match": patterns["spurious_match"],
                "unstable_retrieval": patterns["unstable_retrieval"],
            }
        }