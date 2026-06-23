"""
Build ONE combined dataset for the 200M foundation model:
  - Math + QA + word facts (from create_large_dataset)
  - TinyStories (English)

Everything is stored as plain text lines. The training script turns each line
into next-token prediction (same as language training), so one model learns
math, facts, AND English together - no forgetting, no separate models.

Output: combined_dataset.json  (a list of text strings)
"""

import sys
import os
import json

sys.path.append(os.getcwd())

from create_large_dataset import create_comprehensive_dataset


def main():
    print("=" * 70)
    print("BUILDING COMBINED DATASET (math + QA + facts + stories)")
    print("=" * 70)
    print()

    texts = []

    # 1. Math / QA / word facts -> render as "question answer" text
    print("[1] Adding math / QA / word-fact tasks...")
    tasks = create_comprehensive_dataset()
    for t in tasks:
        # Format as a single line the model learns to complete
        texts.append(f"{t['question']} {t['answer']}")
    # Repeat the structured tasks a few times so they aren't drowned by 20k stories
    structured = list(texts)
    for _ in range(20):  # heavy repetition: math/facts matter and are few
        texts.extend(structured)
    print(f"  {len(tasks)} unique tasks -> {len(texts)} lines after repetition")

    # 2. TinyStories
    print("[2] Adding TinyStories...")
    if os.path.exists("tinystories_prepared.json"):
        with open("tinystories_prepared.json", "r", encoding="utf-8") as f:
            stories = json.load(f)
        texts.extend(stories)
        print(f"  Added {len(stories)} stories")
    else:
        print("  WARNING: tinystories_prepared.json not found - run prepare_tinystories.py first")

    # Shuffle so math and stories interleave (prevents the model from learning
    # them in separate phases)
    import random
    random.shuffle(texts)

    with open("combined_dataset.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)

    total_chars = sum(len(t) for t in texts)
    print()
    print(f"✓ combined_dataset.json: {len(texts):,} lines, {total_chars:,} chars")
    print("  Next: train with scripts/train_combined.py")


if __name__ == "__main__":
    main()
