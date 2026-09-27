from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")

def load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data

def tokenize(text: str) -> Counter:
    return Counter(TOKEN_RE.findall(text.lower()))

def overlap_score(old_text: str, new_text: str) -> float:
    old_tokens = tokenize(old_text)
    new_tokens = tokenize(new_text)

    if not old_tokens:
        return 0.0

    covered = sum((old_tokens & new_tokens).values())
    return covered / sum(old_tokens.values())


def best_match(old_chunk: dict, candidates: list[dict]) -> tuple[dict | None, float]:
    old_text = old_chunk.get("text", "")
    normalized_old = " ".join(old_text.lower().split())

    best_chunk = None
    best_score = -1.0

    for candidate in candidates:
        new_text = candidate.get("text", "")
        normalized_new = " ".join(new_text.lower().split())

        if normalized_old and normalized_old in normalized_new:
            score = 1.0
        else:
            score = overlap_score(old_text, new_text)

        if score > best_score:
            best_chunk = candidate
            best_score = score

    return best_chunk, best_score


def remap_dataset(
    dataset_path: Path,
    old_by_id: dict[str, dict],
    new_by_doc: dict[str, list[dict]],
    min_score: float,
) -> tuple[list[dict], list[dict]]:
    remapped = []
    unresolved = []

    for sample in load_json(dataset_path):
        result = dict(sample)

        if sample.get("should_abstain", False):
            result["original_gold_chunk_ids"] = []
            result["gold_chunk_id"] = ""
            result["supporting_chunks"] = []
            result["remap_status"] = "abstain_preserved"
            remapped.append(result)
            continue

        old_ids = sample.get("supporting_chunks") or [sample.get("gold_chunk_id", "")]
        old_ids = [chunk_id for chunk_id in old_ids if chunk_id]

        new_ids = []
        details = []

        for old_id in old_ids:
            old_chunk = old_by_id.get(old_id)

            if old_chunk is None:
                details.append({
                    "old_chunk_id": old_id,
                    "new_chunk_id": None,
                    "score": 0.0,
                    "reason": "old_chunk_missing",
                })
                continue

            candidates = new_by_doc.get(old_chunk["doc_id"], [])
            candidate, score = best_match(old_chunk, candidates)

            if candidate is None or score < min_score:
                details.append({
                    "old_chunk_id": old_id,
                    "new_chunk_id": None,
                    "score": round(score, 4),
                    "reason": "low_overlap",
                })
                continue

            new_id = candidate["chunk_id"]
            if new_id not in new_ids:
                new_ids.append(new_id)

            details.append({
                "old_chunk_id": old_id,
                "new_chunk_id": new_id,
                "score": round(score, 4),
                "reason": "mapped",
            })

        result["original_gold_chunk_ids"] = old_ids
        result["gold_chunk_id"] = new_ids[0] if new_ids else ""
        result["supporting_chunks"] = new_ids
        result["remap_details"] = details
        result["remap_status"] = "mapped" if len(new_ids) == len(old_ids) else "needs_review"

        if result["remap_status"] == "needs_review":
            unresolved.append({
                "sample_id": sample.get("id"),
                "query": sample.get("query"),
                "details": details,
            })

        remapped.append(result)

    return remapped, unresolved


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be between 0 and 1")

    old_chunks = load_json(Path("data/processed/chunks_semantic.json"))
    new_chunks = load_json(Path("data/processed/chunks_semantic_v2.json"))

    old_by_id = {chunk["chunk_id"]: chunk for chunk in old_chunks}
    new_by_doc = defaultdict(list)

    for chunk in new_chunks:
        new_by_doc[chunk["doc_id"]].append(chunk)

    train, train_unresolved = remap_dataset(
        Path("data/processed/train_eval_v2.json"),
        old_by_id,
        new_by_doc,
        args.min_score,
    )
    holdout, holdout_unresolved = remap_dataset(
        Path("data/processed/holdout_eval_v2.json"),
        old_by_id,
        new_by_doc,
        args.min_score,
    )

    unresolved = train_unresolved + holdout_unresolved

    print(f"Train samples: {len(train)}")
    print(f"Holdout samples: {len(holdout)}")
    print(f"Needs review: {len(unresolved)}")

    if unresolved:
        review_path = Path("results/remap_v2_to_v3_review.json")
        write_json(review_path, unresolved)
        print(f"Review file: {review_path}")
        print("v3 dataset not written; review unresolved mappings first.")
        return 2

    if not args.write:
        print("Dry run passed. Re-run with --write to create v3 datasets.")
        return 0

    write_json(Path("data/processed/train_eval_v3.json"), train)
    write_json(Path("data/processed/holdout_eval_v3.json"), holdout)

    print("Created data/processed/train_eval_v3.json")
    print("Created data/processed/holdout_eval_v3.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())