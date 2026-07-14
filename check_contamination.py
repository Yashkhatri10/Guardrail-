"""
check_contamination.py
"""
import json
from difflib import SequenceMatcher
from datasets import load_dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from layer1_rules.language_filter import is_likely_english

def load_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
    return texts

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

existing_texts = load_texts("data/build/data.jsonl") + load_texts("data/holdout/data.jsonl")

ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
stress_texts = []
for row in ds:
    text = row.get("Prompt", "")
    if text.strip() and is_likely_english(text):
        stress_texts.append(text)

results = []
for stress_text in stress_texts:
    best_score = 0
    best_match = ""
    for existing_text in existing_texts:
        score = similarity(stress_text[:200], existing_text[:200])
        if score > best_score:
            best_score = score
            best_match = existing_text
    results.append({
        "stress_prompt": stress_text[:80],
        "best_match_score": round(best_score, 3),
        "best_match_preview": best_match[:80],
        "likely_contaminated": best_score > 0.7
    })

contaminated = [r for r in results if r["likely_contaminated"]]
clean = [r for r in results if not r["likely_contaminated"]]

print(f"Contaminated: {len(contaminated)} / {len(results)}")
print(f"Clean: {len(clean)} / {len(results)}")

with open("results/contamination_check.json", "w") as f:
    json.dump(results, f, indent=2)

print(len(existing_texts))
print(results[0]['best_match_score'], results[0]['best_match_preview'])
print(results[5]['best_match_score'], results[5]['best_match_preview'])