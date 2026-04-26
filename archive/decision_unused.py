from dataclasses import dataclass
from typing import Literal

@dataclass
class RetrievalDecision:
    action: Literal["generate", "abstain", "retry"]
    chunks: list
    reason: str
    confidence: float

class DecisionController:
    def __init__(self, settings):
        self.min_score = settings.min_score_threshold
        self.gap_threshold = settings.score_gap_threshold

    def decide(self, chunks: list, query: str) -> RetrievalDecision:
        if not chunks:
            return RetrievalDecision("abstain", [], "no chunks retrieved", 0.0)

        top_score = chunks[0].get("retrieval_score", 0)

        if top_score <= self.min_score:
            return RetrievalDecision("abstain", [], f"top score {top_score:.3f} below threshold", top_score)

        # Filter berdasarkan gap — satu tempat, bukan tersebar
        filtered = [chunks[0]]
        for i in range(1, len(chunks)):
            gap = chunks[i-1].get("retrieval_score", 0) - chunks[i].get("retrieval_score", 0)
            if gap > self.gap_threshold:
                break
            filtered.append(chunks[i])

        confidence = top_score  # bisa diganti kalkulasi yang lebih canggih nanti
        return RetrievalDecision("generate", filtered, "ok", confidence)

# Archived: not wired into API path, superseded by ConfidenceEngine
