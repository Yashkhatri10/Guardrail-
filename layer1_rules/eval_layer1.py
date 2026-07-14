# Placeholder for Layer 1 evaluation script.
"""
Runs Layer 1 alone against data/build/data.jsonl.
Writes results/layer1_scores.json AND data/build/layer1_misses.jsonl.
The misses file is Phase 2's real input — not the raw build set — because
Phase 2 should focus on what Layer 1 got wrong, not relearn what it already catches.
"""
import json
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from layer1_rules.normalize import normalize
from layer1_rules.patterns import check_layer1
from layer1_rules.language_filter import is_likely_english

BUILD_PATH = Path("data/build/data.jsonl")
RESULTS_PATH = Path("results/layer1_scores.json")
RESULTS_PATH.parent.mkdir(exist_ok=True)

def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    records = load_records(BUILD_PATH)
    non_english_count = sum(1 for r in records if not is_likely_english(r["text"]))
    records = [r for r in records if is_likely_english(r["text"])]
    print(f"Filtered out {non_english_count} non-English records. Scoring on {len(records)} remaining.")
    
    tp = fp = tn = fn = 0
    latencies = []
    misses = []

    for r in records:
        is_attack = r["label"] != "benign"
        start = time.perf_counter()
        norm_text = normalize(r["text"])
        result = check_layer1(norm_text)
        latencies.append((time.perf_counter() - start) * 1000)

        predicted_attack = result["blocked"]
        if is_attack and predicted_attack:
            tp += 1
        elif is_attack and not predicted_attack:
            fn += 1
            misses.append(r)
        elif not is_attack and predicted_attack:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    scores = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "latency_p50_ms": round(p50, 4),
        "latency_p95_ms": round(p95, 4),
        "total_examples": len(records),
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
    }

    print(json.dumps(scores, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    with open("data/build/layer1_misses.jsonl", "w") as f:
        for m in misses:
            f.write(json.dumps(m) + "\n")

    print(f"\n{len(misses)} attacks missed by Layer 1 — saved for Phase 2.")

if __name__ == "__main__":
    main()