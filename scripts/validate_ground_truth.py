import json
from collections import defaultdict

GT_PATH      = "data/processed/ground_truth_qa.json"
CHUNKS_PATH  = "data/processed/chunks_semantic.json"
TRAIN_PATH   = "data/processed/train_eval.json"
HOLDOUT_PATH = "data/processed/holdout_eval.json"

# ── Loader ───────────────────────────────────────────────────────────────────
def load_json(path: str) -> list | dict:
    with open(path) as f:
        return json.load(f)

# ── Checks ───────────────────────────────────────────────────────────────────
def check_required_fields(samples: list) -> list[str]:
    required = {"question", "gold_answer", "question_type",
                "answer_span", "gold_chunk_id", "doc_id", "tier", "section"}
    errors = []
    for i, s in enumerate(samples):
        missing = required - set(s.keys())
        if missing:
            errors.append(f"  [#{i}] Missing fields: {missing}")
    return errors

def check_answer_span_in_chunk(samples: list, chunk_map: dict) -> list[str]:
    def norm(s):
        return s.replace('\n', ' ').replace('  ', ' ').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"').lower()
    errors = []
    for i, s in enumerate(samples):
        cid   = s.get("gold_chunk_id", "")
        span  = s.get("answer_span", "")
        chunk = chunk_map.get(cid)
        if not chunk:
            errors.append(f"  [#{i}] chunk_id '{cid}' tidak ditemukan di chunks_semantic.json")
            continue
        if span and norm(span) not in norm(chunk["text"]):
            errors.append(f"  [#{i}] answer_span tidak ada di chunk text | chunk={cid} | span='{span[:60]}'")
    return errors

def check_empty_fields(samples: list) -> list[str]:
    errors = []
    for i, s in enumerate(samples):
        for field in ["question", "gold_answer", "answer_span"]:
            if not s.get(field, "").strip():
                errors.append(f"  [#{i}] Field '{field}' kosong")
    return errors

def check_question_types(samples: list) -> list[str]:
    valid_types = {"factual", "paraphrase", "multihop", "adversarial"}
    errors = []
    for i, s in enumerate(samples):
        qt = s.get("question_type", "")
        if qt not in valid_types:
            errors.append(f"  [#{i}] question_type invalid: '{qt}'")
    return errors

def check_no_overlap(train: list, holdout: list) -> list[str]:
    train_ids   = {s["gold_chunk_id"] + s["question"] for s in train}
    holdout_ids = {s["gold_chunk_id"] + s["question"] for s in holdout}
    overlap = train_ids & holdout_ids
    if overlap:
        return [f"  OVERLAP ditemukan: {len(overlap)} samples ada di train DAN holdout"]
    return []

def check_split_ratio(train: list, holdout: list) -> list[str]:
    total = len(train) + len(holdout)
    ratio = len(train) / total
    errors = []
    if not (0.78 <= ratio <= 0.82):
        errors.append(f"  Split ratio off: train={ratio:.2%} (expected ~80%)")
    return []

def check_tier_distribution(samples: list, label: str) -> None:
    dist = defaultdict(int)
    for s in samples:
        dist[s.get("tier", "?")] += 1
    print(f"  Tier distribution ({label}): { {k: dist[k] for k in sorted(dist)} }")

# ── Report ────────────────────────────────────────────────────────────────────
def run_validation():
    print("=" * 60)
    print("VALIDATE GROUND TRUTH — VALIDITY REPORT")
    print("=" * 60)

    # Load data
    samples = load_json(GT_PATH)
    chunks  = load_json(CHUNKS_PATH)
    train   = load_json(TRAIN_PATH)
    holdout = load_json(HOLDOUT_PATH)

    chunk_map = {c["chunk_id"]: c for c in chunks}

    print(f"\nDataset loaded")
    print(f"  ground_truth_qa : {len(samples)} samples")
    print(f"  train_eval      : {len(train)} samples")
    print(f"  holdout_eval    : {len(holdout)} samples")
    print(f"  chunks          : {len(chunks)} total")

    all_errors = {}

    # Run checks
    checks = [
        ("Required fields",       check_required_fields(samples)),
        ("Empty fields",          check_empty_fields(samples)),
        ("Question type valid",   check_question_types(samples)),
        ("Answer span in chunk",  check_answer_span_in_chunk(samples, chunk_map)),
        ("Train/holdout overlap", check_no_overlap(train, holdout)),
        ("Split ratio",           check_split_ratio(train, holdout)),
    ]

    print(f"\n{'─'*60}")
    print("CHECKS")
    print(f"{'─'*60}")

    total_errors = 0
    for name, errors in checks:
        status = "[OK] PASS" if not errors else f" FAIL ({len(errors)} issues)"
        print(f"\n[{status}] {name}")
        for e in errors[:5]: 
            print(e)
        if len(errors) > 5:
            print(f"  ... dan {len(errors)-5} lainnya")
        total_errors += len(errors)

    # Distributions
    print(f"\n{'─'*60}")
    print("DISTRIBUTIONS")
    print(f"{'─'*60}")
    check_tier_distribution(samples, "full")
    check_tier_distribution(train,   "train")
    check_tier_distribution(holdout, "holdout")

    qt_dist = defaultdict(int)
    for s in samples:
        qt_dist[s.get("question_type", "?")] += 1
    print(f"  Question types: {dict(qt_dist)}")

    # Final verdict
    print(f"\n{'='*60}")
    if total_errors == 0:
        print("[OK] VALID — Dataset siap dipakai untuk eval")
    else:
        print(f"[WARNING]  {total_errors} issues ditemukan — review sebelum eval")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_validation()