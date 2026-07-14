"""
eval/run_full_pipeline_eval.py
Runs the full 3-layer pipeline against a dataset file, tracks which layer
resolved each case, and computes precision/recall/FPR/latency.

Usage:
  python eval/run_full_pipeline_eval.py data/build/data.jsonl
  python eval/run_full_pipeline_eval.py data/holdout/data.jsonl
"""
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.router import route

def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval/run_full_pipeline_eval.py <path_to_dataset.jsonl>")
        sys.exit(1)

    data_path = sys.argv[1]
    records = load_records(data_path)

    tp = fp = tn = fn = 0
    latencies = []
    layer_counts = {"layer1": 0, "layer2": 0, "layer3": 0, "none": 0}

    for r in records:
        is_attack = r["label"] != "benign"
        start = time.perf_counter()
        result = route(r["text"])
        latencies.append((time.perf_counter() - start) * 1000)

        layer_counts[result["layer"]] = layer_counts.get(result["layer"], 0) + 1
        predicted_attack = result["blocked"]

        if is_attack and predicted_attack: tp += 1
        elif is_attack and not predicted_attack: fn += 1
        elif not is_attack and predicted_attack: fp += 1
        else: tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    scores = {
        "dataset": data_path,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "latency_p50_ms": round(p50, 4),
        "latency_p95_ms": round(p95, 4),
        "total_examples": len(records),
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "layer_breakdown": layer_counts,
    }
    print(json.dumps(scores, indent=2))

    out_name = "holdout" if "holdout" in data_path else "build"
    with open(f"results/full_pipeline_{out_name}_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

if __name__ == "__main__":
    main()