from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.controller.agent_state import AgentState, AgentStatus
from src.controller.policy_engine import PolicyEngine, PlanHints
from src.controller.confidence_engine import ConfidenceEngine
from src.api.config import settings
from src.generation.generator import GroqGenerator
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.retrieval.tools import RetrievalTools

@dataclass
class AgentResponse:
    answer: Optional[str]
    status: AgentStatus
    state: AgentState
    response: Any = None
    retrieved_ids: list = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)

class Agent:
    def __init__(self, max_steps: int = 8):
        self.policy_engine = PolicyEngine()
        self.confidence_engine = ConfidenceEngine()
        self.generator = GroqGenerator(model=settings.groq_model or self._default_model())
        self.store = QdrantVectorStore()
        self.retriever = MasterHybridRetriever(vector_store=self.store)
        self.tools = RetrievalTools(
            vector_store=self.store,
            retriever=self.retriever,
            confidence_engine=self.confidence_engine,
            generator=self.generator,
        )
        self.max_steps = max_steps

    def _default_model(self) -> str:
        return "llama-3.3-70b-versatile"
    
    # EXECUTION
    async def _execute_retrieval(self, state: AgentState, params: dict):
        result = await self.tools.call_tool("search_hybrid", params)
        state.add_observation("search_hybrid", result)

        if "chunks" in result and result["chunks"]:
            state.extend_chunks(result["chunks"])
            state.retrieved_chunks = sorted(
                state.retrieved_chunks,
                key=lambda x: x.get("retrieval_score", 0),
                reverse=True
            )[:20]
            seen = set()
            deduped = []
            for c in state.retrieved_chunks:
                cid = c.get("chunk_id")
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(c)
            state.retrieved_chunks = deduped

    async def _execute_generation(self, state: AgentState, plan: PlanHints) -> Optional[AgentResponse]:
        result = await self.tools.call_tool("generate_answer", {
            "query": state.query,
            "chunks": state.retrieved_chunks,
            "max_tokens": plan.generation_hint["max_tokens"],
            "temperature": plan.generation_hint["temperature"],
            "source_chunk_id": state.source_chunk_id,
            "min_top1_score": -999.0,
        })

        state.add_observation("generate_answer", result)
        response = result.get("response")

        if response and response.status == "ANSWERED":
            state.status = AgentStatus.ANSWERED
            return AgentResponse(
                answer=response.answer,
                status=state.status,
                state=state,
                response=response,
                retrieved_ids=[
                    c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                    for c in state.retrieved_chunks
                ],
                latency_ms={}
                )
        return AgentResponse(
            answer=response.answer,
            status=state.status,
            state=state,
            response=response,
            retrieved_ids=[
                c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                for c in state.retrieved_chunks
            ],
        )

    # STRATEGY
    def _build_retrieval_params(self, state: AgentState, plan: PlanHints) -> dict:
        params = {
            "query": state.query,
            "k": plan.top_k_hint,
            "use_multi_query": plan.allow_multi_query,
            "use_hyde": plan.allow_hyde,
        }

        if getattr(state, 'force_expand', False) or (state.step_count > 0 and state.confidence_score < 0.4):
            params["k"] = min(plan.top_k_hint * 2, 50)
            params["use_multi_query"] = True
            state.force_expand = False
        return params
    
    def _decide_next_action(self, signals: dict, score: float) ->  str:
        if signals.get("unstable_retrieval"):
            return "RETRIEVE_AGAIN"
        if signals.get("spurious_match"):
            return "PARTIAL_TRUST"
        if score < self.confidence_engine.reject_threshold:
            return "REJECT"
        if score < self.confidence_engine.partial_threshold:
            return "PARTIAL_TRUST"
        return "GENERATE"
    
    # MAIN LOOP
    async def run(self, query: str, source_chunk_id: str | None = None) -> AgentResponse:
        state = AgentState(query=query, source_chunk_id=source_chunk_id)
        plan = self.policy_engine.initial_plan(query)
        MAX_ITERATIONS = 2
        iteration_count = 0
        response = None
        import time
        t_start = time.time()
        while state.status == AgentStatus.RUNNING and not state.is_budget_exhausted():
            if iteration_count >= MAX_ITERATIONS:
                state.status = AgentStatus.ABSTAINED
                break
            iteration_count += 1
            params = self._build_retrieval_params(state, plan)

            # Retrieval
            await self._execute_retrieval(state, params)
            print(f"[DEBUG] chunks going to confidence: {len(state.retrieved_chunks)}")
            for c in state.retrieved_chunks[:3]:
                print(f"  chunk_id={c.get('chunk_id')} retrieval_score={c.get('retrieval_score')}")


            # Confidence
            quality = await self.tools.assess_retrieval_quality(
                state.retrieved_chunks, state.query
            )

            confidence = quality["confidence"]
            state.confidence_score = confidence["confidence_score"]
            signals = confidence["signals"]

            current_top_score = state.retrieved_chunks[0].get("retrieval_score", 0.0) if state.retrieved_chunks else 0.0
            if abs(current_top_score - state.prev_top_score) < 0.1:
                state.stagnation_count += 1
            else:
                state.stagnation_count = 0
            state.prev_top_score  = current_top_score

            if state.stagnation_count >= 2:
                state.status = AgentStatus.ABSTAINED
                break

            action = self._decide_next_action(signals, state.confidence_score)
            if action == "REJECT":
                state.status = AgentStatus.ABSTAINED
                break
            if action == "RETRIEVE_AGAIN":
                state.force_expand = True
                params = self._build_retrieval_params(state, plan)
                await self._execute_retrieval(state, params)
                continue

            if action in ("PARTIAL_TRUST", "GENERATE"):
                response = await self._execute_generation(state, plan)
                if response:
                    response.latency_ms = {"total": round((time.time() - t_start) * 1000, 1)}
                    return response
                
        # fallback
        return AgentResponse(
            answer=response.answer if response else None,
            status=state.status,
            state=state,
            response=None,
            retrieved_ids=[
                c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", "")
                for c in state.retrieved_chunks
            ],
            latency_ms={"total": round((time.time() - t_start) * 1000, 1)},
        )