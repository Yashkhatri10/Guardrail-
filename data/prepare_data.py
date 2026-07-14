"""
data/prepare_data.py
Pulls jailbreak + prompt-injection datasets, splits 70/30 into build/holdout.
Run this ONCE. Do not re-run after Phase 1 starts — it will reshuffle the split
and silently invalidate every number you produce after that.
"""
import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)  # fixed seed so the split is reproducible, not different every run

OUT_DIR = Path("data")
(OUT_DIR / "build").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "holdout").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "raw").mkdir(parents=True, exist_ok=True)

def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def stratified_split(records, holdout_frac=0.3):
    # split separately per label so build and holdout both have a realistic mix
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    build, holdout = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        cut = int(len(items) * holdout_frac)
        holdout.extend(items[:cut])
        build.extend(items[cut:])
    random.shuffle(build)
    random.shuffle(holdout)
    return build, holdout

all_records = []

# --- Dataset 1: jailbreak vs benign ---
jb = load_dataset("jackhhao/jailbreak-classification")
for split in ["train", "test"]:
    for row in jb[split]:
        all_records.append({
            "text": row["prompt"],
            "label": "jailbreak" if row["type"] == "jailbreak" else "benign",
            "attack_type": "jailbreak" if row["type"] == "jailbreak" else "none",
            "source": "jackhhao"
        })

# --- Dataset 2: prompt injection vs legit ---
inj = load_dataset("deepset/prompt-injections")
for split in inj.keys():
    for row in inj[split]:
        all_records.append({
            "text": row["text"],
            "label": "injection" if row["label"] == 1 else "benign",
            "attack_type": "injection" if row["label"] == 1 else "none",
            "source": "deepset"
        })

print(f"Total records pulled: {len(all_records)}")
save_jsonl(all_records, OUT_DIR / "raw" / "all_records.jsonl")

build, holdout = stratified_split(all_records)

print(f"Build set: {len(build)} | Holdout set: {len(holdout)}")
save_jsonl(build, OUT_DIR / "build" / "data.jsonl")
save_jsonl(holdout, OUT_DIR / "holdout" / "data.jsonl")

print("Done. Do not touch data/holdout/ again until Phase 4.")