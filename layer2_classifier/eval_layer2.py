# Placeholder for Layer 2 evaluation script.
"""
layer2_classifier/eval_layer2.py

Scores the trained classifier against data/build/data.jsonl (the same population
Layer 1 was scored on), so you can compare Layer 1 vs Layer 2 fairly.

This is still a build-set number, not the final claim — Phase 4 does that on holdout.

Requires: layer2_classifier/model/ folder (downloaded from Colab)
"""
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from datasets import load_dataset # Removed this import
import sys
sys.path.append(str(Path(__file__).parent.parent))
from layer1_rules.language_filter import is_likely_english

MODEL_PATH = "layer2_classifier/model/layer2_model_final"
BUILD_PATH = Path("walla.jsonl") # Changed to use walla.jsonl
RESULTS_PATH = Path("results/layer2_scores.json")
RESULTS_PATH.parent.mkdir(exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    return pred, confidence  # pred: 0 = benign, 1 = attack

def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    print(f"Loading records from {BUILD_PATH}...")
    raw_records = load_records(BUILD_PATH)
    records = []
    filtered_out_non_english = 0
    for r in raw_records:
        if not is_likely_english(r["text"]):
            filtered_out_non_english += 1
            continue
        records.append(r)

    print(f"Loaded {len(records)} English records for evaluation.")
    if filtered_out_non_english > 0:
        print(f"Filtered out {filtered_out_non_english} non-English records.")

    tp = fp = tn = fn = 0
    latencies = []

    for r in records:
        is_attack = r["label"] != "benign"
        start = time.perf_counter()
        pred, conf = predict(r["text"])
        latencies.append((time.perf_counter() - start) * 1000)

        if is_attack and pred == 1: tp += 1
        elif is_attack and pred == 0: fn += 1
        elif not is_attack and pred == 1: fp += 1
        else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    latencies.sort()

    total_records_processed = len(records)
    if total_records_processed == 0:
        p50 = 0
        p95 = 0
    else:
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]


    scores = {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "latency_p50_ms": round(p50, 4), "latency_p95_ms": round(p95, 4),
        "total_examples": total_records_processed,
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
    }
    print(json.dumps(scores, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    main()