"""
data/expand_dataset.py

Pulls additional external datasets and appends them to data/build/data.jsonl ONLY.
Does NOT touch data/holdout/ — that set stays frozen from Phase 0.
Run this locally (not in this sandbox) since it needs huggingface.co access.
"""
import json
import os
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Authenticate with Hugging Face using the token from .env
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env file")

BUILD_PATH = Path("data/build/data.jsonl")

def load_existing(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def append_records(path, new_records):
    with open(path, "a") as f:
        for r in new_records:
            f.write(json.dumps(r) + "\n")

new_records = []

# --- Dataset 3: Qualifire prompt injections benchmark ---
# NOTE: This is a gated dataset. You must visit https://huggingface.co/datasets/qualifire/prompt-injections-benchmark
# and request access first. The token alone isn't enough; manual approval is required.
try:
    q = load_dataset("qualifire/prompt-injections-benchmark")
    for split in q.keys():
        for row in q[split]:
            label = row.get("label", row.get("type", ""))
            is_attack = str(label).lower() not in ("0", "benign", "legit")
            new_records.append({
                "text": row["text"],
                "label": "injection" if is_attack else "benign",
                "attack_type": "injection" if is_attack else "none",
                "source": "qualifire"
            })
    print("✓ Loaded Qualifire dataset")
except Exception as e:
    print(f"⚠ Skipping Qualifire dataset: {str(e)}")
    print("  → Visit https://huggingface.co/datasets/qualifire/prompt-injections-benchmark to request access")

# --- Dataset 4: larger in-the-wild jailbreak set ---
try:
    wild = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split="train")
    for row in wild:
        new_records.append({
            "text": row["prompt"],
            "label": "jailbreak",
            "attack_type": "jailbreak",
            "source": "trustairlab"
        })
    print("✓ Loaded TrustAIRLab dataset")
except Exception as e:
    print(f"⚠ Skipping TrustAIRLab dataset: {str(e)}")

# --- Dataset 4b: Regular prompts from TrustAIRLab ---
try:
    
    print("✓ Loaded TrustAIRLab regular prompts dataset")
except Exception as e:
    print(f"⚠ Skipping TrustAIRLab regular dataset: {str(e)}")

if not new_records:
    print("\n⚠ WARNING: No datasets were loaded successfully.")
    print("Please check your Hugging Face token and dataset access permissions.")
    exit(1)

print(f"\nNew records pulled: {len(new_records)}")

existing = load_existing(BUILD_PATH)
print(f"Existing build set: {len(existing)}")

append_records(BUILD_PATH, new_records)

print(f"New build set total: {len(existing) + len(new_records)}")
print("data/holdout/ was NOT modified — still the original Phase 0 split.")