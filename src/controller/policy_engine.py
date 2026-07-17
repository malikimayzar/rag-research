from __future__ import annotations
from dataclasses import dataclass

QueryType = str   

@dataclass
class PlanHints:
    query_type: QueryType
    preferred_strategy: str
    expected_complexity: str
    allow_multi_query: bool
    allow_hyde: bool
    allow_references: bool
    max_ref_ratio: float
    top_k_hint: int
    generation_hint: dict[str, float]


class PolicyEngine:
    def decide_query_type(self, query: str) -> QueryType:
        if not query or not query.strip():
            return "general"
        q = query.lower().strip()
        FACTUAL_SIGNALS = {"who", "when", "where", "which", "whom"}
        if any(word in q.split() or q.startswith(word) for word in FACTUAL_SIGNALS):
            return "factual"
        REASONING_SIGNALS = {"how", "why", "explain", "describe", "what causes", "what makes"}
        if any(signal in q for signal in REASONING_SIGNALS):
            return "reasoning"
        return "general"

    def retrieval_strategy(self, query_type: QueryType) -> dict:
        _STRATEGIES: dict[QueryType, dict] = {
            "factual": {
                "use_hyde":        False,
                "use_multi_query": False,
                "top_k":           5,
            },
            "general": {
                "use_hyde":        False,
                "use_multi_query": False,
                "top_k":           8,
            },
            "reasoning": {
                "use_hyde":        False,
                "use_multi_query": False,
                "top_k":           10,
            },
        }

        strategy = _STRATEGIES.get(query_type)
        if strategy is None:
            return {"use_hyde": False, "use_multi_query": False, "top_k": 8}
        return strategy

    # 3. Reference Policy

    def reference_policy(self, query: str, query_type: QueryType) -> dict:
        CITATION_SIGNALS = {
            "author", "authors", "who wrote", "published by",
            "citation", "cite", "reference", "bibliography",
            "journal", "proceedings", "paper by",
        }
        q_lower = query.lower()
        has_citation_signal = any(signal in q_lower for signal in CITATION_SIGNALS)

        if has_citation_signal:
            return {
                "allow_references": True,
                "max_ref_ratio":    0.4,
            }

        _POLICIES: dict[QueryType, dict] = {
            "factual": {
                "allow_references": True,
                "max_ref_ratio":    0.4,
            },
            "reasoning": {
                "allow_references": False,
                "max_ref_ratio":    0.0,
            },
            "general": {
                "allow_references": False,
                "max_ref_ratio":    0.0,
            },
        }

        policy = _POLICIES.get(query_type)
        if policy is None:
            return {"allow_references": False, "max_ref_ratio": 0.0}
        return policy

    # 4. Generation Policy
    def generation_policy(self, query_type: QueryType) -> dict:
        _POLICIES: dict[QueryType, dict] = {
            "factual": {
                "max_tokens":  100,
                "temperature": 0.0,
            },
            "general": {
                "max_tokens":  200,
                "temperature": 0.2,
            },
            "reasoning": {
                "max_tokens":  300,
                "temperature": 0.3,
            },
        }

        policy = _POLICIES.get(query_type)
        if policy is None:
            return {"max_tokens": 200, "temperature": 0.2}
        return policy

    # Convenience: get all policies in one call
    def resolve(self, query: str) -> dict:
        q_type = self.decide_query_type(query)
        return {
            "query_type": q_type,
            "retrieval":  self.retrieval_strategy(q_type),
            "reference":  self.reference_policy(query, q_type),
            "generation": self.generation_policy(q_type),
        }

    def initial_plan(self, query: str) -> PlanHints:
        q_type = self.decide_query_type(query)
        retrieval = self.retrieval_strategy(q_type)
        reference = self.reference_policy(query, q_type)
        generation = self.generation_policy(q_type)

        preferred_strategy = "hybrid" if q_type != "factual" else "dense"
        expected_complexity = (
            "high" if q_type == "reasoning" else "medium" if q_type == "general" else "low"
        )

        return PlanHints(
            query_type=q_type,
            preferred_strategy=preferred_strategy,
            expected_complexity=expected_complexity,
            allow_multi_query=retrieval["use_multi_query"],
            allow_hyde=retrieval["use_hyde"],
            allow_references=reference["allow_references"],
            max_ref_ratio=reference["max_ref_ratio"],
            top_k_hint=retrieval["top_k"],
            generation_hint=generation,
        )
