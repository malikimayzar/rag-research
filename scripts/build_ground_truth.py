import json
import random
import re
import time
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
CHUNKS_PATH          = "data/processed/chunks_semantic.json"
OUTPUT_PATH          = "data/processed/ground_truth_qa.json"
SLEEP_BETWEEN_CALLS  = 3
MAX_RETRIES          = 3
MIN_CHUNK_LENGTH     = 100
MAX_DIGIT_RATIO      = 0.4
TARGET_PER_TIER      = {1: 100, 2: 40, 3: 50}

# ── Filters ──────────────────────────────────────────────────────────────────
def is_low_quality(text: str) -> bool:
    if len(text.split()) < 30:
        return True
    if text.count("\n") > 20:
        return True
    if text.count("[") > 5:
        return True
    return False


def is_reference_like(text: str) -> bool:
    text_lower = text.lower().strip()
    if re.match(r"^\[\d+\]", text.strip()):
        return True

    strong_signals = sum([
        "proceedings of"  in text_lower,
        "conference on"   in text_lower,
        "journal of"      in text_lower,
        "vol."            in text_lower,
        "pp."             in text_lower,
        "doi"             in text_lower,
        "et al."          in text_lower,
    ])

    if "arxiv.org" in text_lower and strong_signals >= 1:
        return True

    if strong_signals >= 2:
        return True

    if text.count(",") > 8 and strong_signals >= 1:
        return True
    return False


def is_eligible_chunk(chunk: dict) -> bool:
    text    = chunk["text"]
    section = chunk["metadata"]["section"]

    if section == "references":
        return False
    if len(text) < MIN_CHUNK_LENGTH:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > MAX_DIGIT_RATIO:
        return False
    if is_low_quality(text):
        return False
    if is_reference_like(text):
        return False
    return True

# ── Sampling ─────────────────────────────────────────────────────────────────
def sample_chunks(all_chunks: list) -> list:
    sampled = []
    for tier, target in TARGET_PER_TIER.items():
        tier_chunks = [c for c in all_chunks if c["tier"] == tier]
        by_doc = defaultdict(list)
        for c in tier_chunks:
            by_doc[c["doc_id"]].append(c)

        doc_ids = list(by_doc.keys())
        random.shuffle(doc_ids)
        num_docs = len(doc_ids)
        if num_docs == 0:
            continue

        quota       = max(1, target // num_docs)
        tier_sample = []

        for doc_id in doc_ids:
            candidates = by_doc[doc_id]
            k = min(len(candidates), quota)
            tier_sample.extend(random.sample(candidates, k))

        if len(tier_sample) < target:
            already = set(id(c) for c in tier_sample)
            remaining = [c for c in tier_chunks if id(c) not in already]
            needed    = target - len(tier_sample)
            tier_sample.extend(random.sample(remaining, min(len(remaining), needed)))

        sampled.extend(tier_sample[:target])
    return sampled

# ── Prompt ───────────────────────────────────────────────────────────────────
def build_prompt(chunk: dict) -> str:
    text = chunk["text"][:1000]
    return f"""You are a dataset builder for a RAG evaluation system.

CHUNK ID: {chunk["chunk_id"]}
CHUNK TEXT:
{text}

Generate exactly 2 questions in valid JSON format like this:
[
  {{
    "question": "...",
    "answer": "...",
    "question_type": "factual",
    "answer_span": "exact substring from chunk text"
  }},
  {{
    "question": "...",
    "answer": "...",
    "question_type": "paraphrase",
    "answer_span": "exact substring from chunk text"
  }}
]

Rules:
- answer must come verbatim or near-verbatim from the chunk text
- answer_span must be an exact substring found in the chunk text above
- do not generate yes/no questions
- do not add any explanation outside the JSON array
- output only the JSON array, nothing else"""

# ── Groq call ────────────────────────────────────────────────────────────────
def call_groq_with_retry(client: Groq, prompt: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                print(f"  [Rate limit] attempt {attempt}, sleep 10s...")
                time.sleep(10)
            else:
                print(f"  [Error] {err[:80]}")
                return None
    return None

# ── Parser ───────────────────────────────────────────────────────────────────
def parse_response(response_text: str, chunk: dict) -> list[dict]:
    if not response_text:
        return []
    try:
        cleaned = response_text.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            for key in ["questions", "data", "items", "results"]:
                if key in parsed:
                    parsed = parsed[key]
                    break

        if not isinstance(parsed, list):
            return []

        results = []
        for item in parsed:
            if not all(k in item for k in ["question", "answer", "question_type", "answer_span"]):
                continue
            results.append({
                "question":      item["question"],
                "gold_answer":   item["answer"],
                "question_type": item["question_type"],
                "answer_span":   item["answer_span"],
                "gold_chunk_id": chunk["chunk_id"],
                "doc_id":        chunk["doc_id"],
                "tier":          chunk["tier"],
                "section":       chunk["metadata"]["section"],
                "should_abstain": False,
            })
        return results
    except Exception:
        return []

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    with open(CHUNKS_PATH) as f:
        data = json.load(f)

    print(f"Total chunks    : {len(data)}")

    eligible = [c for c in data if is_eligible_chunk(c)]
    print(f"Eligible chunks : {len(eligible)}")

    tier_count = defaultdict(int)
    for c in eligible:
        tier_count[c["tier"]] += 1
    print(f"Eligible per tier: {dict(tier_count)}")

    sampled = sample_chunks(eligible)
    print(f"Sampled chunks  : {len(sampled)}")

    sampled_tier = defaultdict(int)
    for c in sampled:
        sampled_tier[c["tier"]] += 1
    print(f"Sampled per tier: {dict(sampled_tier)}")

    # ── DRY RUN (hapus baris ini setelah test OK) ──
    print(f"\n[FULL RUN] Processing {len(sampled)} chunks...\n")

    all_samples = []
    for i, chunk in enumerate(sampled):
        print(f"[{i+1}/{len(sampled)}] {chunk['chunk_id']}")
        prompt   = build_prompt(chunk)
        response = call_groq_with_retry(client, prompt)
        samples  = parse_response(response, chunk)
        print(f"  → {len(samples)} samples generated")
        all_samples.extend(samples)
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nTotal samples generated: {len(all_samples)}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT_PATH}")

    # Preview
    print("\n── Preview 2 samples ──")
    for s in all_samples[:2]:
        print(json.dumps(s, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()