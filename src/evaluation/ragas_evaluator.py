from __future__ import annotations

import numpy as np
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig  
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

# LangChain LLM + Embeddings
from langchain_groq import ChatGroq
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
from langchain_huggingface import HuggingFaceEmbeddings

# Config 
DEFAULT_LLM_MODEL   = "llama-3.1-8b-instant"   
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class RAGSample:
    question:          str
    answer:            str
    contexts:          List[str]   
    ground_truth:      Optional[str] = None   

@dataclass
class EvalResult:
    question:           str
    faithfulness:       float
    answer_relevancy:   float
    context_precision:  Optional[float]   
    context_recall:     Optional[float]   
    answer_correctness: Optional[float]
    latency_ms:         float
    metadata:           dict = field(default_factory=dict)

@dataclass
class AblationResult:
    exp_id:             str
    config:             dict
    samples:            List[EvalResult]
    avg_faithfulness:       float
    avg_answer_relevancy:   float
    avg_context_precision:  float
    avg_context_recall:     float
    avg_answer_correctness: Optional[float]
    avg_latency_ms:         float
    hallucination_rate:     float   
    total_samples:          int

# RAGAS Evaluator 
class RAGASEvaluator:
    def __init__(
        self,
        groq_api_key=None,
        llm_model="llama-3.3-70b-versatile",
        embed_model=DEFAULT_EMBED_MODEL,
        include_answer_correctness=False,
        llm_provider="groq",
    ):
        print(f"[RAGASEvaluator] Init Evaluation LLM: {llm_model} (provider: {llm_provider})")

        if llm_provider == "gemini" and GEMINI_AVAILABLE:
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                raise ValueError("GOOGLE_API_KEY tidak ditemukan.")
            self.llm = ChatGoogleGenerativeAI(
                model=llm_model,
                google_api_key=gemini_key,
                temperature=0,
            )
        else:
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY tidak ditemukan.")
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=llm_model,
                temperature=0,
                max_retries=10,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

        print(f"[RAGASEvaluator] Init Embeddings: {embed_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.include_answer_correctness = include_answer_correctness
        
        # Bind metrics
        self._faithfulness        = Faithfulness(llm=self.llm)
        self._answer_relevancy    = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self._context_precision   = ContextPrecision(llm=self.llm)
        self._context_recall      = ContextRecall(llm=self.llm)
        self._answer_correctness  = AnswerCorrectness(llm=self.llm)
    
        # CONFIG SUPER SABAR
        self.run_config = RunConfig(
            max_retries=20,    
            timeout=180,       
            max_workers=1     
        )
        self.metrics_no_ref = [self._faithfulness, self._answer_relevancy]
        self.metrics_with_ref = [
            self._faithfulness,
            self._answer_relevancy,
            self._context_precision,
            self._context_recall,
        ]
        if include_answer_correctness:
            self.metrics_with_ref.append(self._answer_correctness)
            
    def evaluate(
        self,
        samples: List[RAGSample],
        exp_id: str = "exp_001",
        config: Optional[dict] = None,
    ) -> AblationResult:
        print(f"\n[RAGASEvaluator] Evaluating {len(samples)} samples for {exp_id}...")
        t0 = time.time()
        has_reference = all(s.ground_truth for s in samples)
        active_metrics = self.metrics_with_ref if has_reference else self.metrics_no_ref

        data = {
            "question": [s.question for s in samples],
            "answer":   [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        if has_reference:
            data["ground_truth"] = [s.ground_truth for s in samples]
        
        dataset = Dataset.from_dict(data)
        ragas_result = evaluate(
            dataset=dataset,
            metrics=active_metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=self.run_config
        )
        
        ragas_df = ragas_result.to_pandas()
        elapsed_ms = (time.time() - t0) * 1000
        per_sample_ms = elapsed_ms / len(samples)

        eval_results = []
        for i, sample in enumerate(samples):
            row = ragas_df.iloc[i]
            eval_results.append(EvalResult(
                question=sample.question,
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)) if "context_precision" in row else None,
                context_recall=float(row.get("context_recall", 0.0)) if "context_recall" in row else None,
                answer_correctness=float(row.get("answer_correctness", 0.0)) if "answer_correctness" in row else None,
                latency_ms=per_sample_ms,
                metadata={"exp_id": exp_id},
            ))

        def avg(key):
            vals = [
                getattr(r, key) for r in eval_results
                if getattr(r, key) is not None and not np.isnan(getattr(r, key))
            ]
            if not vals:
                return 0.0
            return sum(vals) / len(vals)
        
        successful_faithfulness = [
            r.faithfulness for r in eval_results
            if not np.isnan(r.faithfulness)
        ]

        if successful_faithfulness:
            hallucination_count = sum(1 for v in successful_faithfulness if v < 0.5)
            hallucination_rate = hallucination_count / len(successful_faithfulness)
        else:
            hallucination_rate = 0.0
            
        result = AblationResult(
            exp_id=exp_id,
            config=config or {},
            samples=eval_results,
            avg_faithfulness=avg("faithfulness"),
            avg_answer_relevancy=avg("answer_relevancy"),
            avg_context_precision=avg("context_precision"),
            avg_context_recall=avg("context_recall"),
            avg_answer_correctness=avg("answer_correctness") if self.include_answer_correctness else None,
            avg_latency_ms=avg("latency_ms"),
            hallucination_rate=hallucination_rate,
            total_samples=len(samples),
        )

        self._print_summary(result)
        return result

    def _print_summary(self, result: AblationResult) -> None:
        print(f"\n{'='*55}")
        print(f"  [OK] RAGAS Results — {result.exp_id}")
        print(f"{'='*55}")
        print(f"  Faithfulness       : {result.avg_faithfulness:.4f}")
        print(f"  Answer Relevancy   : {result.avg_answer_relevancy:.4f}")
        print(f"  Context Precision  : {result.avg_context_precision:.4f}")
        print(f"  Context Recall     : {result.avg_context_recall:.4f}")
        if result.avg_answer_correctness is not None:
            print(f"  Answer Correctness : {result.avg_answer_correctness:.4f}")
        print(f"  Hallucination Rate : {result.hallucination_rate:.1%}")
        print(f"  Samples            : {result.total_samples}")
        print(f"{'='*55}\n")

    def save_result(self, result: AblationResult, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if Path(output_path).exists():
            with open(output_path) as f:
                existing = json.load(f)
        existing.append(asdict(result))
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print(f"[Saved] Results → {output_path}")

# CLI Smoke Test
if __name__ == "__main__":
    samples = [
        RAGSample(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation.",
            contexts=["RAG combines retrieval with generation."],
            ground_truth="RAG stands for Retrieval-Augmented Generation."
        )
    ]
    if os.getenv("GROQ_API_KEY"):
        evaluator = RAGASEvaluator(include_answer_correctness=True)
        evaluator.evaluate(samples, exp_id="smoke_test")
    else:
        print("Set GROQ_API_KEY dulu!")