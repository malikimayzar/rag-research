"""
Split rebuilt ground truth dataset into train (80%) and holdout (20%) stratified by question_type.
"""

import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_PATH = "data/processed/ground_truth_qa_rebuilt.json"
TRAIN_OUTPUT_PATH = "data/processed/train_eval_v2.json"
HOLDOUT_OUTPUT_PATH = "data/processed/holdout_eval_v2.json"
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


def main():
    print("=" * 80)
    print("SPLIT REBUILT DATASET")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\n[Loading] {INPUT_PATH}...")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    print(f"  → Loaded {len(data)} samples")

    # ── Extract stratification column ────────────────────────────────────────
    question_types = [sample.get("question_type", "unknown") for sample in data]

    # ── Stratified split ─────────────────────────────────────────────────────
    print(f"\n[Splitting] 80/20 stratified by question_type...")
    train_indices, holdout_indices = train_test_split(
        range(len(data)),
        train_size=TRAIN_RATIO,
        stratify=question_types,
        random_state=RANDOM_STATE,
    )

    train_data = [data[i] for i in train_indices]
    holdout_data = [data[i] for i in holdout_indices]

    print(f"  → Train: {len(train_data)} samples")
    print(f"  → Holdout: {len(holdout_data)} samples")

    # ── Count per question_type ──────────────────────────────────────────────
    def count_by_type(samples):
        counts = defaultdict(int)
        for sample in samples:
            q_type = sample.get("question_type", "unknown")
            counts[q_type] += 1
        return dict(sorted(counts.items()))

    train_counts = count_by_type(train_data)
    holdout_counts = count_by_type(holdout_data)

    print("\n[Train split by question_type]")
    for q_type, count in train_counts.items():
        print(f"  {q_type:20s}: {count:3d} samples")

    print("\n[Holdout split by question_type]")
    for q_type, count in holdout_counts.items():
        print(f"  {q_type:20s}: {count:3d} samples")

    # ── Count zero_gold in holdout ───────────────────────────────────────────
    zero_gold_holdout = sum(
        1 for sample in holdout_data if len(sample.get("gold_chunk_ids", [])) == 0
    )
    print(f"\n[Holdout] Zero-gold samples: {zero_gold_holdout}")

    # ── Save splits ──────────────────────────────────────────────────────────
    print(f"\n[Saving] Train split → {TRAIN_OUTPUT_PATH}")
    Path(TRAIN_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_OUTPUT_PATH, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"  → Saved {len(train_data)} samples")

    print(f"\n[Saving] Holdout split → {HOLDOUT_OUTPUT_PATH}")
    Path(HOLDOUT_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HOLDOUT_OUTPUT_PATH, "w") as f:
        json.dump(holdout_data, f, indent=2, ensure_ascii=False)
    print(f"  → Saved {len(holdout_data)} samples")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"Total samples           : {len(data)}")
    print(f"Train samples (80%)     : {len(train_data)}")
    print(f"Holdout samples (20%)   : {len(holdout_data)}")
    print(f"Holdout zero-gold       : {zero_gold_holdout}")
    print("=" * 80)


if __name__ == "__main__":
    main()
