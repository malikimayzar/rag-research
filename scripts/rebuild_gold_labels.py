import json
import time
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from src.retrieval.qdrant_store import QdrantVectorStore
from src.retrieval.hybrid_retriever import MasterHybridRetriever

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_PATH = "data/processed/ground_truth_qa.json"
OUTPUT_PATH = "data/processed/ground_truth_qa_rebuilt.json"
RETRIEVAL_TOP_K = 20
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 200
SLEEP_BETWEEN_SAMPLES = 0.5
MAX_RETRIES = 3
RETRY_SLEEP = 10
PROGRESS_INTERVAL = 10

# ── Helper functions ────────────────────────────────────────────────────────
def _get_attr(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def call_groq_with_retry(groq_client: Groq, prompt: str) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                if attempt < MAX_RETRIES:
                    print(f"    [Rate limit] Attempt {attempt}/{MAX_RETRIES}, sleep {RETRY_SLEEP}s...")
                    time.sleep(RETRY_SLEEP)
                else:
                    print(f"    [Rate limit] Max retries reached")
                    return None
            else:
                print(f"    [Groq Error] {err[:80]}")
                return None
    return None

def build_retrieval_prompt(question: str, chunks: list) -> str:
    candidate_list = ""
    for i, chunk in enumerate(chunks, 1):
        chunk_id = _get_attr(chunk, "chunk_id", f"unknown_{i}")
        text = _get_attr(chunk, "text", "")
        text_preview = (text[:300] + "...") if len(text) > 300 else text
        candidate_list += f"[{i}] {chunk_id}: {text_preview}\n"
        
    prompt = f"""You are evaluating retrieval candidates for a question.

Question: {question}

Candidate chunks:
{candidate_list}

Task: Select ALL chunk numbers that contain enough information to answer the question.
Return ONLY a JSON array of chunk numbers. Example: [1, 5, 12]
If none are relevant, return: []"""
    return prompt

def parse_chunk_selection(response_text: str, num_candidates: int) -> list[int]:
    if not response_text:
        return []
    try:
        response_text = response_text.strip()
        if response_text.startswith("[") and response_text.endswith("]"):
            selected = json.loads(response_text)
        else:
            import re
            match = re.search(r'\[[\d,\s]*\]', response_text)
            if match:
                selected = json.loads(match.group())
            else:
                return []

        if not isinstance(selected, list):
            return []

        valid_selected = [s for s in selected if isinstance(s, int) and 1 <= s <= num_candidates]
        return valid_selected
    except Exception as e:
        print(f"    [Parse Error] {e}")
        return []

def map_indices_to_chunk_ids(selected_indices: list[int], chunks: list) -> list[str]:
    chunk_ids = []
    for idx in selected_indices:
        chunk_idx = idx - 1
        if 0 <= chunk_idx < len(chunks):
            chunk_id = _get_attr(chunks[chunk_idx], "chunk_id", None)
            if chunk_id:
                chunk_ids.append(chunk_id)
    return chunk_ids

def main():
    print("=" * 80)
    print("REBUILD GOLD LABELS")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[Loading] {INPUT_PATH}...")
    with open(INPUT_PATH, "r") as f:
        ground_truth = json.load(f)
    print(f"  → Loaded {len(ground_truth)} samples") 

    # ── Initialize components ────────────────────────────────────────────────
    print(f"\n[Init] QdrantVectorStore, MasterHybridRetriever, Groq...")
    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(vector_store=store)
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("  → OK")
    print(f"\n[Processing] {len(ground_truth)} samples...\n")
    rebuilt_samples = []
    zero_gold_count = 0
    total_gold_chunks = 0

    for i, sample in enumerate(ground_truth):
        question = sample.get("question", "")
        gold_answer = sample.get("gold_answer", "")
        question_type = sample.get("question_type", "unknown")
        original_gold_chunk_id = sample.get("gold_chunk_id", "")
        should_abstain = sample.get("should_abstain", False)

        # Progress
        if (i + 1) % PROGRESS_INTERVAL == 0:
            avg_gold = total_gold_chunks / (i + 1) if (i + 1) > 0 else 0
            print(
                f"[{i + 1}/{len(ground_truth)}] Progress: "
                f"avg_gold={avg_gold:.2f}, zero_gold={zero_gold_count}"
            )

        try:
            retrieved_chunks = retriever.search(question, top_k=RETRIEVAL_TOP_K)
            retrieved_chunks = [c for c in retrieved_chunks if 'test_intro' not in (_get_attr(c, 'chunk_id', '') or '')]
            num_candidates = len(retrieved_chunks)

            if num_candidates == 0:
                print(f"  [{i + 1}] WARNING: No chunks retrieved for: {question[:50]}...")
                rebuilt_samples.append({
                    "question": question,
                    "gold_answer": gold_answer,
                    "question_type": question_type,
                    "gold_chunk_ids": [],
                    "original_gold_chunk_id": original_gold_chunk_id,
                    "should_abstain": True,  
                })
                zero_gold_count += 1
                time.sleep(SLEEP_BETWEEN_SAMPLES)
                continue

            prompt = build_retrieval_prompt(question, retrieved_chunks)
            response = call_groq_with_retry(groq, prompt)

            if not response:
                print(f"  [{i + 1}] ERROR: Groq failed for: {question[:50]}...")
                rebuilt_samples.append({
                    "question": question,
                    "gold_answer": gold_answer,
                    "question_type": question_type,
                    "gold_chunk_ids": [],
                    "original_gold_chunk_id": original_gold_chunk_id,
                    "should_abstain": True,
                })
                zero_gold_count += 1
                time.sleep(SLEEP_BETWEEN_SAMPLES)
                continue
            selected_indices = parse_chunk_selection(response, num_candidates)
            gold_chunk_ids = map_indices_to_chunk_ids(selected_indices, retrieved_chunks)
            result_sample = {
                "question": question,
                "gold_answer": gold_answer,
                "question_type": question_type,
                "gold_chunk_ids": gold_chunk_ids,
                "original_gold_chunk_id": original_gold_chunk_id,
                "should_abstain": len(gold_chunk_ids) == 0,
            }
            rebuilt_samples.append(result_sample)

            if len(gold_chunk_ids) == 0:
                zero_gold_count += 1
            else:
                total_gold_chunks += len(gold_chunk_ids)

        except Exception as e:
            print(f"  [{i + 1}] EXCEPTION: {e}")
            rebuilt_samples.append({
                "question": question,
                "gold_answer": gold_answer,
                "question_type": question_type,
                "gold_chunk_ids": [],
                "original_gold_chunk_id": original_gold_chunk_id,
                "should_abstain": True,
            })
            zero_gold_count += 1
        time.sleep(SLEEP_BETWEEN_SAMPLES)

    # Save results 
    print(f"\n[Saving] {OUTPUT_PATH}...")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rebuilt_samples, f, indent=2, ensure_ascii=False)
    print(f"  → Saved {len(rebuilt_samples)} samples")

    # Final stats 
    avg_gold_chunks = (
        total_gold_chunks / (len(rebuilt_samples) - zero_gold_count)
        if (len(rebuilt_samples) - zero_gold_count) > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("FINAL STATS")
    print("=" * 80)
    print(f"Total samples              : {len(rebuilt_samples)}")
    print(f"Avg gold chunks per query  : {avg_gold_chunks:.2f}")
    print(f"Zero-gold count            : {zero_gold_count}")
    print(f"Samples with gold chunks   : {len(rebuilt_samples) - zero_gold_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()