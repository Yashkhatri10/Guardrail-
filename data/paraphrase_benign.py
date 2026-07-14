"""
data/paraphrase_benign.py

Paraphrases benign prompts to balance dataset.
Uses LLM-based paraphrasing to generate natural variations.
Appends paraphrased records to data/build/data.jsonl.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BUILD_PATH = Path("data/build/data.jsonl")

def load_existing(path):
    """Load all records from JSONL."""
    with open(path) as f:
        return [json.loads(line) for line in f]

def get_benign_records(records):
    """Extract only benign records."""
    return [r for r in records if r.get('label') == 'benign']

def count_by_label(records):
    """Count records by label."""
    benign = sum(1 for r in records if r.get('label') == 'benign')
    attack = sum(1 for r in records if r.get('label') != 'benign')
    return benign, attack

def simple_paraphrase(text, variation):
    """
    Simple rule-based paraphrasing.
    This gives quick, deterministic variations without external API calls.
    """
    paraphrases = {
        0: text,  # Original
        1: f"Can you help me with: {text}",
        2: f"I need assistance with: {text}",
        3: f"Question: {text}",
        4: f"Could you please: {text}",
        5: f"Help me understand: {text}",
        6: f"How do I: {text}",
        7: f"I'm trying to: {text}",
        8: f"What is the best way to: {text}",
        9: f"Can someone explain how to: {text}",
        10: f"Is it possible to: {text}",
    }
    return paraphrases.get(variation % 11, text)

def paraphrase_with_llm(texts, num_variations=2):
    """
    Use Hugging Face transformers for paraphrasing.
    Falls back to simple paraphrasing if T5 not available.
    """
    try:
        from transformers import pipeline
        print("Loading paraphrasing model... (this may take a moment)")
        paraphraser = pipeline("text2text-generation", model="t5-base")
        
        paraphrased = []
        for text in texts:
            prompts = [
                f"paraphrase: {text}",
            ]
            results = paraphraser(prompts, max_length=200, num_beams=3)
            
            for result in results:
                paraphrased.append(result["generated_text"])
                if len(paraphrased) >= len(texts) * num_variations:
                    break
            
            if len(paraphrased) >= len(texts) * num_variations:
                break
        
        return paraphrased[:len(texts) * num_variations]
    except Exception as e:
        print(f"⚠ LLM paraphrasing failed ({str(e)[:50]}...), using rule-based variations")
        return None

# Load existing data
existing = load_existing(BUILD_PATH)
benign_count, attack_count = count_by_label(existing)

print(f"Current balance:")
print(f"  Benign: {benign_count}")
print(f"  Attack: {attack_count}")
print(f"  Ratio: {benign_count/attack_count:.2%}")

# Calculate how many paraphrases we need
needed = attack_count - benign_count
print(f"\nNeed to create: {needed} paraphrased benign examples")

# Get benign records
benign_records = get_benign_records(existing)
print(f"Available benign records to paraphrase: {len(benign_records)}")

# Try LLM-based paraphrasing first, fallback to rule-based
all_paraphrased_texts = []
paraphrased_texts = paraphrase_with_llm([r['text'] for r in benign_records], num_variations=2)

if paraphrased_texts is None:
    # Use rule-based paraphrasing
    print("\nGenerating rule-based variations...")
    variations_per_record = (needed // len(benign_records)) + 2
    
    for record in benign_records:
        for var in range(1, variations_per_record):
            paraphrased_text = simple_paraphrase(record['text'], var)
            if paraphrased_text != record['text']:
                all_paraphrased_texts.append(paraphrased_text)
else:
    all_paraphrased_texts = paraphrased_texts

# Trim to exact amount needed
all_paraphrased_texts = all_paraphrased_texts[:needed]

print(f"Generated {len(all_paraphrased_texts)} paraphrased examples")

# Create new records with paraphrased text
new_records = []
for i, text in enumerate(all_paraphrased_texts):
    new_records.append({
        "text": text,
        "label": "benign",
        "attack_type": "none",
        "source": f"paraphrased_benign_{i % len(benign_records)}"
    })

# Append to data file
print(f"Appending {len(new_records)} paraphrased records to data/build/data.jsonl...")
with open(BUILD_PATH, "a") as f:
    for r in new_records:
        f.write(json.dumps(r) + "\n")

# Verify balance
updated = load_existing(BUILD_PATH)
new_benign, new_attack = count_by_label(updated)

print(f"\nNew balance:")
print(f"  Benign: {new_benign}")
print(f"  Attack: {new_attack}")
print(f"  Ratio: {new_benign/new_attack:.2%}")
print(f"\n✓ Dataset is now {'more balanced' if new_benign/new_attack > 0.8 else 'still imbalanced (consider running again)'}")
