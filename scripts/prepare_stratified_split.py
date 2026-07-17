import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {filepath}")

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent / "data" / "processed"
    clean_file = base_dir / "ground_truth_qa_clean.json"
    train_file = base_dir / "train_eval.json"
    holdout_file = base_dir / "holdout_eval.json"

    # Load clean data
    print("Loading clean dataset...")
    samples = load_json(clean_file)
    print(f"Loaded {len(samples)} samples")

    # Remove mislabeled field
    print("\nRemoving 'mislabeled' field from all samples...")
    for sample in samples:
        if "mislabeled" in sample:
            del sample["mislabeled"]

    # Extract question types for stratification
    question_types = [sample.get("question_type", "unknown") for sample in samples]

    # Stratified 80/20 split
    print("\nPerforming stratified 80/20 split by question_type...")
    train_samples, holdout_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
        stratify=question_types
    )

    print(f"Train set: {len(train_samples)} samples")
    print(f"Holdout set: {len(holdout_samples)} samples")

    # Save datasets
    save_json(train_samples, train_file)
    save_json(holdout_samples, holdout_file)

    # Print statistics
    print("\n" + "=" * 70)
    print("STRATIFIED SPLIT SUMMARY (by question_type)")
    print("=" * 70)

    # Count by question type for train
    train_types = defaultdict(int)
    for sample in train_samples:
        qtype = sample.get("question_type", "unknown")
        train_types[qtype] += 1

    # Count by question type for holdout
    holdout_types = defaultdict(int)
    for sample in holdout_samples:
        qtype = sample.get("question_type", "unknown")
        holdout_types[qtype] += 1

    # Count by question type for original
    original_types = defaultdict(int)
    for sample in samples:
        qtype = sample.get("question_type", "unknown")
        original_types[qtype] += 1

    # Print table
    all_types = sorted(set(original_types.keys()))
    print(f"{'Question Type':<20} {'Original':<12} {'Train':<12} {'Holdout':<12}")
    print("-" * 70)

    for qtype in all_types:
        orig_count = original_types[qtype]
        train_count = train_types[qtype]
        holdout_count = holdout_types[qtype]
        print(f"{qtype:<20} {orig_count:<12} {train_count:<12} {holdout_count:<12}")

    print("-" * 70)
    print(f"{'TOTAL':<20} {len(samples):<12} {len(train_samples):<12} {len(holdout_samples):<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
