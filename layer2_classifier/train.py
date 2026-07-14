import json
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "build" / "data.jsonl"
MODEL_NAME = "microsoft/deberta-v3-small"
OUTPUT_DIR = ROOT / "layer2_classifier" / "model" / "layer2_model_final"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        records.append({
            "text": r["text"],
            "label": 0 if r["label"] == "benign" else 1
        })

print(f"Loaded {len(records)} records")
print(f"Benign : {sum(x['label']==0 for x in records)}")
print(f"Attack : {sum(x['label']==1 for x in records)}")

train_records, val_records = train_test_split(
    records,
    test_size=0.15,
    stratify=[r["label"] for r in records],
    random_state=42,
)

train_ds = Dataset.from_list(train_records)
val_ds = Dataset.from_list(val_records)

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=str(ROOT / "results_training"),
    overwrite_output_dir=True,
    num_train_epochs=4,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("\\nStarting training...")
trainer.train()

print("\\nSaving model...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

metrics = trainer.evaluate()
print("\\nValidation Metrics")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\\nModel saved to: {OUTPUT_DIR}")
print("You can now run:")
print("python layer2_classifier\\\\eval_layer2.py")
