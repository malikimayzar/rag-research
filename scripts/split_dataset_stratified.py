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
    print(f"Saved to {filepath}")

def main():
    base_dir = Path(__file__).parent.parent / "data" / "processed"
    clean_file = base_dir / "ground_truth_qa_clean.json"
    train_file = base_dir / "train_eval.json"
    holdout_file = base_dir / "holdout_eval.json"

    # Load data
    print("Loading cleaned dataset...")
    samples = load_json(clean_file)
    print(f"Loaded {len(samples)} samples")

    # Remove mislabeled field from all samples
    print("\nRemoving 'mislabeled' field...")
    for sample in samples:
        if "mislabeled" in sample:
            del sample["mislabeled"]

    # Extract question types for stratification
    question_types = [sample.get("question_type", "unknown") for sample in samples]

    # Stratified split 80/20
    print("Performing stratified 80/20 split by question_type...")
    train_indices, holdout_indices = train_test_split(
        range(len(samples)),
        test_size=0.2,
        stratify=question_types,
        random_state=42
    )

    train_samples = [samples[i] for i in train_indices]
    holdout_samples = [samples[i] for i in holdout_indices]

    # Save splits
    save_json(train_samples, train_file)
    save_json(holdout_samples, holdout_file)

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET SPLIT SUMMARY")
    print("=" * 70)

    # Count by question type for original
    type_counts_all = defaultdict(int)
    for sample in samples:
        qtype = sample.get("question_type", "unknown")
        type_counts_all[qtype] += 1

    # Count by question type for train
    type_counts_train = defaultdict(int)
    for sample in train_samples:
        qtype = sample.get("question_type", "unknown")
        type_counts_train[qtype] += 1

    # Count by question type for holdout
    type_counts_holdout = defaultdict(int)
    for sample in holdout_samples:
        qtype = sample.get("question_type", "unknown")
        type_counts_holdout[qtype] += 1

    # Print detailed breakdown
    all_types = sorted(set(type_counts_all.keys()))
    
    print(f"\nTotal samples: {len(samples)}")
    print(f"Training samples: {len(train_samples)} (80%)")
    print(f"Holdout samples: {len(holdout_samples)} (20%)")
    
    print("\n" + "-" * 70)
    print(f"{'Question Type':<20} {'Total':>10} {'Train':>10} {'Holdout':>10}")
    print("-" * 70)
    
    for qtype in all_types:
        total = type_counts_all[qtype]
        train_count = type_counts_train[qtype]
        holdout_count = type_counts_holdout[qtype]
        print(f"{qtype:<20} {total:>10} {train_count:>10} {holdout_count:>10}")
    
    print("-" * 70)
    total_all = sum(type_counts_all.values())
    total_train = sum(type_counts_train.values())
    total_holdout = sum(type_counts_holdout.values())
    print(f"{'TOTAL':<20} {total_all:>10} {total_train:>10} {total_holdout:>10}")
    print("=" * 70)

if __name__ == "__main__":
    main()