import json, os, re, time, random
from pathlib import Path
from collections import defaultdict, Counter
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = "data/processed/chunks_semantic.json"
OUTPUT_PATH = "data/processed/ground_truth_qa_v2.json"
SLEEP = 2
MAX_RETRIES = 3

TARGET = {"lexical": 60, "paraphrase": 60, "multihop": 60, "adversarial": 50}

NOISE_CHUNK_IDS = {
    "tier3_2307_09288_s0461","tier3_2307_09288_s0306","tier3_2307_09288_s0418",
    "tier3_2307_09288_s0406","tier3_2307_09288_s0408","tier3_2307_09288_s0407",
    "tier3_2307_09288_s0419","tier3_2307_09288_s0291","tier3_2307_09288_s0294",
    "tier3_2307_09288_s0405","tier3_2307_09288_s0295","tier3_2307_09288_s0404",
    "tier3_2307_09288_s0296","tier3_2307_09288_s0307","tier3_2307_09288_s0337",
    "tier3_2307_09288_s0292","tier3_2307_09288_s0309","tier3_2307_09288_s0420",
    "tier3_2307_09288_s0308","tier3_2307_09288_s0293","tier3_2307_09288_s0421",
    "tier3_2307_09288_s0402"
}

def get_tier(doc_id: str) -> str:
    if doc_id.startswith("tier1"): return "tier1"
    if doc_id.startswith("tier2"): return "tier2"
    if doc_id.startswith("tier3"): return "tier3"
    return "unknown"

def is_reference_chunk(text: str) -> bool:
    signals = sum([
        "proceedings of" in text.lower(),
        "journal of" in text.lower(),
        "et al." in text.lower(),
        "doi" in text.lower(),
        bool(re.search(r"\[\d+\]", text)),
        text.count(",") > 8 and len(text.split()) < 60,
    ])
    return signals >= 2

def is_eligible(chunk: dict) -> bool:
    text = chunk.get("text", "")
    if chunk.get("chunk_id") in NOISE_CHUNK_IDS: return False
    if len(text.split()) < 40: return False
    if is_reference_chunk(text): return False
    return True

PROMPT_LEXICAL = """Given this chunk, generate 1 factual question with a direct answer from the text.
Output valid JSON only:
{{"question": "...", "answer": "exact phrase from text", "answer_span": "exact phrase from text"}}

CHUNK: {text}"""

PROMPT_PARAPHRASE = """Given this chunk, generate 1 question that asks the same thing as a factual question but in different words.
Output valid JSON only:
{{"question": "...", "answer": "exact phrase from text", "answer_span": "exact phrase from text"}}

CHUNK: {text}"""

PROMPT_MULTIHOP = """Given these TWO chunks, generate 1 question that requires BOTH chunks to answer.
STRICT RULES:
- The question MUST NOT be answerable from Chunk 1 alone
- The question MUST NOT be answerable from Chunk 2 alone  
- The answer MUST combine specific facts from BOTH chunks
- Do NOT ask comparison questions between concepts

Output valid JSON only:
{{"question": "...", "answer": "synthesized from both chunks", 
  "answer_span": "key phrase from chunk 1 or 2",
  "requires_chunk1": "what fact from chunk1 is needed",
  "requires_chunk2": "what fact from chunk2 is needed"}}

CHUNK 1: {text1}

CHUNK 2: {text2}"""

PROMPT_ADVERSARIAL = """Given this chunk, generate 1 question that CANNOT be answered from this text.
STRICT RULES:
- Do NOT ask about ages, personal info, or obviously missing data
- The question must LOOK relevant to the topic but the specific answer is absent
- Example of BAD adversarial: "What is the author's age?" (too obvious)
- Example of GOOD adversarial: "What was the F1 score on dataset X?" (plausible but not in text)

Output valid JSON only:
{{"question": "...", "answer": null, "answer_span": null}}

CHUNK: {text}"""

PROMPT_VALIDATE_MULTIHOP = """Can this question be answered from Chunk 1 alone, without Chunk 2?
Answer with JSON only: {{"answerable_from_chunk1_alone": true/false, "reason": "..."}}

QUESTION: {question}
CHUNK 1: {text1}"""

def call_groq(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e): time.sleep(10)
            else: return None
    return None

def is_true_multihop(client, question, text1):
    raw = call_groq(client, PROMPT_VALIDATE_MULTIHOP.format(
        question=question, text1=text1[:600]
    ))
    if not raw: return True 
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        return not parsed.get("answerable_from_chunk1_alone", True)
    except:
        return True
    
def parse_single(raw, chunk_id, doc_id, qtype, chunk_id_2=None, should_abstain=False):
    if not raw: return None
    try:
        cleaned = raw.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        return {
            "id": f"{qtype}_{chunk_id}",
            "query": parsed["question"],
            "answer": parsed.get("answer"),
            "type": qtype,
            "should_abstain": should_abstain,
            "supporting_chunks": [chunk_id] if not chunk_id_2 else [chunk_id, chunk_id_2],
            "doc_id": doc_id,
            "gold_chunk_id": chunk_id,
        }
    except:
        return None
    
def main():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chunks = json.load(open(CHUNKS_PATH))
    eligible = [c for c in chunks if is_eligible(c)]
    print(f"Eligible chunks: {len(eligible)} / {len(chunks)}")

    # Group by doc
    by_doc = defaultdict(list)
    for c in eligible:
        by_doc[c["doc_id"]].append(c)

    results = []

    # LEXICAL + PARAPHRASE
    for qtype, prompt_tpl in [("lexical", PROMPT_LEXICAL), ("paraphrase", PROMPT_PARAPHRASE)]:
        target = TARGET[qtype]
        pool = random.sample(eligible, min(target, len(eligible)))
        for chunk in pool:
            raw = call_groq(client, prompt_tpl.format(text=chunk["text"][:800]))
            item = parse_single(raw, chunk["chunk_id"], chunk["doc_id"], qtype)
            if item: results.append(item)
            time.sleep(SLEEP)
        print(f"[{qtype}] generated: {sum(1 for r in results if r['type']==qtype)}")

    # MULTIHOP — ambil 2 chunks dari doc berbeda
    target_mh = TARGET["multihop"]
    doc_ids = list(by_doc.keys())
    mh_count = 0
    attempts = 0
    while mh_count < target_mh and attempts < target_mh * 3:
        attempts += 1
        d1, d2 = random.sample(doc_ids, 2)
        c1 = random.choice(by_doc[d1])
        c2 = random.choice(by_doc[d2])
        raw = call_groq(client, PROMPT_MULTIHOP.format(
            text1=c1["text"][:600], text2=c2["text"][:600]
        ))
        item = parse_single(raw, c1["chunk_id"], c1["doc_id"], "multihop", c2["chunk_id"])
        if item and is_true_multihop(client, item["query"], c1["text"]):
            results.append(item)
            mh_count += 1
        else:
            print(f"  [skip] fake multihop dropped")
        time.sleep(SLEEP)
    print(f"[multihop] generated: {mh_count}")

    # ADVERSARIAL — should_abstain=True
    target_adv = TARGET["adversarial"]
    pool_adv = random.sample(eligible, min(target_adv, len(eligible)))
    for chunk in pool_adv:
        raw = call_groq(client, PROMPT_ADVERSARIAL.format(text=chunk["text"][:800]))
        item = parse_single(raw, chunk["chunk_id"], chunk["doc_id"], "adversarial",
                        should_abstain=True)
        if item: results.append(item)
        time.sleep(SLEEP)
    print(f"[adversarial] generated: {sum(1 for r in results if r['type']=='adversarial')}")

    # Save
    random.shuffle(results)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTPUT_PATH, "w"), indent=2, ensure_ascii=False)
    print(f"\nTotal: {len(results)} samples → {OUTPUT_PATH}")
    dist = Counter(r["type"] for r in results)
    abstain = sum(1 for r in results if r["should_abstain"])
    print(f"Distribution: {dict(dist)}")
    print(f"Should abstain: {abstain}")

if __name__ == "__main__":
    main()