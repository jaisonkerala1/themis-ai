"""
Create Comprehensive Training Dataset for Real AI Learning
200+ diverse reasoning tasks for proper generalization
"""

import json
import random

def create_comprehensive_dataset():
    """Generate 200+ diverse reasoning tasks"""
    
    dataset = []
    
    # =================================================================
    # CATEGORY 1: ADDITION (60 tasks)
    # =================================================================
    print("Creating addition tasks...")
    
    # Single digit addition (20 tasks)
    for a in range(0, 10):
        for b in range(0, 10):
            if len(dataset) < 30:  # Limit to 30
                result = a + b
                dataset.append({
                    "question": f"If A = {a} and B = {b}, then A + B =",
                    "answer": str(result),
                    "category": "addition_single"
                })
    
    # Alternative formats (15 tasks)
    additions = [
        (1, 1, 2), (2, 2, 4), (3, 3, 6), (4, 4, 8), (5, 5, 10),
        (1, 2, 3), (2, 3, 5), (3, 4, 7), (4, 5, 9), (5, 6, 11),
        (1, 9, 10), (2, 8, 10), (3, 7, 10), (4, 6, 10), (5, 4, 9)
    ]
    for a, b, result in additions:
        dataset.append({
            "question": f"What is {a} + {b} =",
            "answer": str(result),
            "category": "addition_format2"
        })
        dataset.append({
            "question": f"Add {a} and {b} =",
            "answer": str(result),
            "category": "addition_format3"
        })

    # Compact formats people actually type: "a+b=" and "a + b ="
    # Cover every single-digit combination so the model recognizes the notation.
    for a in range(0, 10):
        for b in range(0, 10):
            result = a + b
            dataset.append({
                "question": f"{a}+{b}=",
                "answer": str(result),
                "category": "addition_compact"
            })
            dataset.append({
                "question": f"{a} + {b} =",
                "answer": str(result),
                "category": "addition_compact_spaced"
            })
    
    # =================================================================
    # CATEGORY 2: NUMBER SEQUENCES (50 tasks)
    # =================================================================
    print("Creating number sequence tasks...")
    
    # Increment by 1
    for start in range(1, 11):
        seq = [start, start+1, start+2, start+3]
        dataset.append({
            "question": f"Write the next number: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
            "answer": str(seq[3] + 1),
            "category": "sequence_plus1"
        })
    
    # Increment by 2
    for start in [1, 2, 3, 5, 10]:
        seq = [start, start+2, start+4, start+6]
        dataset.append({
            "question": f"Write the next number in sequence: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
            "answer": str(seq[3] + 2),
            "category": "sequence_plus2"
        })
    
    # Increment by 5
    for start in [0, 5, 10]:
        seq = [start, start+5, start+10, start+15]
        dataset.append({
            "question": f"Continue the pattern: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
            "answer": str(seq[3] + 5),
            "category": "sequence_plus5"
        })
    
    # Increment by 10
    for start in [0, 10, 20]:
        seq = [start, start+10, start+20, start+30]
        dataset.append({
            "question": f"What comes next: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
            "answer": str(seq[3] + 10),
            "category": "sequence_plus10"
        })
    
    # Multiply by 2
    for start in [1, 2, 3]:
        seq = [start, start*2, start*4, start*8]
        dataset.append({
            "question": f"Next number: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
            "answer": str(seq[3] * 2),
            "category": "sequence_mult2"
        })
    
    # =================================================================
    # CATEGORY 3: LETTER SEQUENCES (30 tasks)
    # =================================================================
    print("Creating letter sequence tasks...")
    
    # Consecutive letters
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(0, 20):
        seq = [alphabet[i], alphabet[i+1], alphabet[i+2], alphabet[i+3]]
        dataset.append({
            "question": f"Write the next letter: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
            "answer": alphabet[i+4],
            "category": "letter_consecutive"
        })
    
    # Skip by 2 (A, C, E, G...)
    for i in range(0, 10):
        idx = i * 2
        if idx + 8 < len(alphabet):
            seq = [alphabet[idx], alphabet[idx+2], alphabet[idx+4], alphabet[idx+6]]
            dataset.append({
                "question": f"Write the next letters in sequence: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},",
                "answer": alphabet[idx+8],
                "category": "letter_skip2"
            })
    
    # =================================================================
    # CATEGORY 4: COUNTING (40 tasks)
    # =================================================================
    print("Creating counting tasks...")
    
    # Count different letters
    letters = ['A', 'B', 'X', 'O', 'Y', 'Z']
    patterns = [
        "AAABBA",  # 4 A's
        "XXYXX",   # 4 X's
        "OOXXOO",  # 4 O's
        "ABABAB",  # 3 A's
        "BBBABB",  # 4 B's
        "XOXOXO",  # 3 X's
        "YYYZYY",  # 5 Y's
        "ZZAZZA",  # 3 Z's
    ]
    
    for letter in letters:
        for _ in range(5):
            # Generate random pattern
            pattern = ''.join(random.choices(['A', 'B', 'X', 'O', 'Y', 'Z'], k=6))
            count = pattern.count(letter)
            dataset.append({
                "question": f"If count of {letter} in {pattern} is",
                "answer": str(count),
                "category": "counting"
            })
    
    # =================================================================
    # CATEGORY 5: SIMPLE WORDS (Single tokens only for now)
    # =================================================================
    print("Creating word tasks...")
    
    # Numbers as words (single digit)
    for i in range(10):
        dataset.append({
            "question": f"Write the number {i} in digits:",
            "answer": str(i),
            "category": "number_word"
        })
    
    # Simple yes/no
    dataset.extend([
        {"question": "Is 5 greater than 3?", "answer": "Y", "category": "logic"},
        {"question": "Is 2 less than 1?", "answer": "N", "category": "logic"},
        {"question": "Is 10 equal to 10?", "answer": "Y", "category": "logic"},
    ])

    # =================================================================
    # CATEGORY 6: WORD-ANSWER FACTS (multi-token answers)
    # These exercise multi-token generation (answers are whole words).
    # Duplicated a few times so they get enough training exposure
    # relative to the large addition set.
    # =================================================================
    print("Creating word-fact tasks...")

    word_facts = [
        ("Capital of France is", "Paris"),
        ("Capital of Japan is", "Tokyo"),
        ("Capital of Italy is", "Rome"),
        ("The antonym of hot is", "cold"),
        ("The antonym of big is", "small"),
        ("The antonym of up is", "down"),
        ("The antonym of fast is", "slow"),
        ("Complete the logic: sky is blue, grass is", "green"),
        ("The color of the sun is", "yellow"),
        ("The color of blood is", "red"),
        ("Two plus two in words is", "four"),
        ("The first month is", "January"),
        ("A baby dog is called a", "puppy"),
        ("A baby cat is called a", "kitten"),
        ("Water freezes into", "ice"),
    ]
    # Repeat each fact several times to boost exposure during sampling
    for _ in range(8):
        for q, a in word_facts:
            dataset.append({
                "question": q,
                "answer": a,
                "category": "word_fact"
            })

    return dataset


def save_dataset(dataset, filename):
    """Save dataset to JSON file"""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"DATASET CREATED: {filename}")
    print(f"{'='*60}")
    print(f"Total tasks: {len(dataset)}")
    
    # Count by category
    categories = {}
    for item in dataset:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nBreakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print(f"\n{'='*60}")
    print("✓ Dataset ready for training!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("="*60)
    print("CREATING COMPREHENSIVE TRAINING DATASET")
    print("="*60)
    print()
    
    dataset = create_comprehensive_dataset()
    save_dataset(dataset, "training_dataset_large.json")
    
    # Show examples
    print("\nExample tasks:")
    print("-" * 60)
    for i in range(min(5, len(dataset))):
        print(f"{i+1}. Q: {dataset[i]['question']}")
        print(f"   A: {dataset[i]['answer']}")
        print()
