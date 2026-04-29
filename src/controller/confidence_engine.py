import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger("rag.confidence_engine")

class ConfidenceEngine:
    def __init__(self):
        self.reject_threshold = 0.25
        self.partial_threshold = 0.45

    def _normalize_scores(self, scores: List[float], score_type: str = "cosine") -> np.ndarray:
        arr = np.array(scores, dtype=float)
        if score_type == "cross_encoder":
            return 1 / (1 + np.exp(-arr))
        elif score_type == "rrf":
            return np.clip(arr / (arr.max() + 1e-9), 0.0, 1.0)
        else:  
            return np.clip(arr, 0.0, 1.0)
    
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
            score = chunk.get("rerank_score")
            if score is None:
                score = chunk.get("retrieval_score")
            if score is None:
                score = chunk.get("score")
            if score is None:
                return None
        else:
            score = getattr(chunk, "rerank_score", None)
            if score is None:
                score = getattr(chunk, "retrieval_score", None)
            if score is None:
                score = getattr(chunk, "score", None)
        return float(score) if score is not None else None
    def _source_agreement(self, chunks) -> float:
        doc_ids = []
        for c in chunks[:5]:
            meta = c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {})
            doc_id = meta.get("doc_id") or meta.get("source") or meta.get("file_name")
            if doc_id:
                doc_ids.append(doc_id)
        if not doc_ids:
            return 0.5
        return min(len(set(doc_ids)) / 3.0, 1.0)
    
    def _detect_patterns(self, top_norm: float, mean_top3_norm: float, gap: float) -> dict:
        return {
            "spurious_match": bool(top_norm > 0.85 and mean_top3_norm < 0.45),
            "unstable_retrieval": bool(gap < 0.05 and mean_top3_norm < 0.4),
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

        normalized_scores = list(self._normalize_scores(raw_scores, score_type="cross_encoder"))
        top_score = raw_scores[0]
        mean_top3 = float(np.mean(raw_scores[:3])) if len(raw_scores) >= 3 else top_score
        top_score_norm = normalized_scores[0]

        mean_top3_norm = float(np.mean(normalized_scores[:3])) if len(normalized_scores) >= 3 else normalized_scores[0]
        if len(normalized_scores) > 1:
            gap = normalized_scores[0] - normalized_scores[1]
        else:
            gap = 0.0
        score_level = float(np.mean(normalized_scores[:3])) if len(normalized_scores) >= 3 else normalized_scores[0]
        entropy = self._calculate_entropy(normalized_scores)
        agreement = self._source_agreement(chunks)

        base_confidence = (
            0.40 * top_score_norm +
            0.35 * mean_top3_norm +
            0.15 * agreement +
            0.10 * score_level
        )

        if gap > 0.5:
            gap_modifier = +0.10 if mean_top3_norm > 0.6 else -0.15
        else:
            gap_modifier = 0.0

        confidence_score = max(0.0, min(1.0, base_confidence + gap_modifier))

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
        patterns = self._detect_patterns(top_score_norm, mean_top3_norm, gap)

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