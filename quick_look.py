import json
from transformers import pipeline

classifier = pipeline("text-classification", model="deepset/deberta-v3-base-injection")

with open("data/holdout/data.jsonl") as f:
    records = [json.loads(l) for l in f]

false_positives = []
for r in records:
    if r["label"] == "benign":
        result = classifier(r["text"])[0]
        if result["label"] == "INJECTION":
            false_positives.append((r["text"], result["score"]))

print(f"Total benign misclassified as INJECTION: {len(false_positives)}")
for text, score in false_positives[:15]:
    print(f"[{score:.3f}] {text[:100]}")