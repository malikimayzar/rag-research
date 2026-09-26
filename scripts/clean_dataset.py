import json
from pathlib import Path
from collections import defaultdict


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent / "data" / "processed"
    qa_file = base_dir / "ground_truth_qa.json"
    chunks_file = base_dir / "chunks_semantic.json"
    output_file = base_dir / "ground_truth_qa_clean.json"

    # Load data
    print("Loading data...")
    qa_samples = load_json(qa_file)
    chunks_data = load_json(chunks_file)

    # Build chunk lookup by chunk_id
    chunks_by_id = {chunk["chunk_id"]: chunk["text"] for chunk in chunks_data}
    print(f"Loaded {len(qa_samples)} QA samples and {len(chunks_by_id)} chunks")

    # Process samples
    clean_samples = []
    dropped_reasons = defaultdict(int)
    kept = 0
    flagged_mislabeled = 0

    for idx, sample in enumerate(qa_samples):
        gold_chunk_id = sample.get("gold_chunk_id", "")
        gold_answer = sample.get("gold_answer", "")

        # Rule 1: Drop if gold_chunk_id contains 'test_intro'
        if "test_intro" in gold_chunk_id:
            dropped_reasons["contains_test_intro"] += 1
            continue

        # Rule 2: Check if gold_answer is substring of chunk text (case-insensitive)
        chunk_text = chunks_by_id.get(gold_chunk_id, "")

        if not chunk_text:
            dropped_reasons["chunk_not_found"] += 1
            continue

        # Case-insensitive substring check
        if gold_answer.lower() in chunk_text.lower():
            # Answer is found in chunk, keep it
            sample["mislabeled"] = False
            clean_samples.append(sample)
            kept += 1
        else:
            # Answer is NOT found in chunk, flag as mislabeled but keep it
            sample["mislabeled"] = True
            clean_samples.append(sample)
            flagged_mislabeled += 1
            kept += 1

    # Save results
    save_json(clean_samples, output_file)

    # Print statistics
    print("\n" + "=" * 60)
    print("DATASET CLEANING SUMMARY")
    print("=" * 60)
    print(f"Original samples: {len(qa_samples)}")
    print(f"Kept samples: {kept}")
    print(f"Dropped samples: {len(qa_samples) - kept}")
    print(f"  - Contains 'test_intro': {dropped_reasons['contains_test_intro']}")
    print(f"  - Chunk not found: {dropped_reasons['chunk_not_found']}")
    print(f"\nQuality flags:")
    print(f"  - Flagged as mislabeled: {flagged_mislabeled}")
    print(f"  - Valid (answer in chunk): {kept - flagged_mislabeled}")
    print("=" * 60)


if __name__ == "__main__":
    main()
