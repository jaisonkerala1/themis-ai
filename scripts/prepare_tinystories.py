"""
Download TinyStories and prepare it as next-token training data for Themis.

TinyStories is a dataset of simple English short stories written with a small
vocabulary - specifically designed to train SMALL language models to produce
coherent English. Perfect for our ~15M param model.

Output: tinystories_prepared.json  ->  a list of {"question": context, "answer": next_chunk}
We frame language modeling as: given a chunk of text, predict the continuation.
"""

import sys
import os
import json

sys.path.append(os.getcwd())

from datasets import load_dataset


def main():
    print("=" * 70)
    print("PREPARING TINYSTORIES DATASET")
    print("=" * 70)
    print()

    # How many stories to use (keep modest for a small model + laptop)
    NUM_STORIES = 2000
    # Max characters per story to keep sequences manageable
    MAX_CHARS = 200

    print(f"Downloading TinyStories (streaming, first {NUM_STORIES} stories)...")
    # Stream so we don't download the whole multi-GB dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    stories = []
    for i, row in enumerate(ds):
        text = row["text"].strip().replace("\n", " ")
        # Collapse multiple spaces
        text = " ".join(text.split())
        if len(text) < 20:
            continue
        # Truncate to keep it short and end at a sentence if possible
        if len(text) > MAX_CHARS:
            cut = text[:MAX_CHARS]
            # try to end at a period
            last_period = cut.rfind(".")
            if last_period > 50:
                cut = cut[:last_period + 1]
            text = cut
        stories.append(text)
        if len(stories) >= NUM_STORIES:
            break
        if (i + 1) % 500 == 0:
            print(f"  Collected {len(stories)} stories...")

    print(f"\nCollected {len(stories)} stories.")
    print()

    # Show a few examples
    print("Example stories:")
    for s in stories[:3]:
        print(f"  - {s[:120]}...")
    print()

    # Save raw stories - the training script will turn them into next-token sequences
    with open("tinystories_prepared.json", "w", encoding="utf-8") as f:
        json.dump(stories, f, ensure_ascii=False)

    # Stats
    total_chars = sum(len(s) for s in stories)
    print(f"Saved {len(stories)} stories to tinystories_prepared.json")
    print(f"Total characters: {total_chars:,}")
    print(f"Avg length: {total_chars // len(stories)} chars")
    print()
    print("Next: train with scripts/train_language.py")


if __name__ == "__main__":
    main()
