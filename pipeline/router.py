"""
pipeline/router.py

Decides which layer handles a given input. Order:
1. Layer 1 (regex) - if it flags, block immediately. High precision (99.8%), trust it.
2. Layer 2 (classifier) - if confidence is high in either direction, trust it.
3. Layer 3 (LLM judge) - only for cases Layer 2 is unsure about.

CONFIDENCE_THRESHOLD is a guess (0.85) - not validated. You need to tune this
using your holdout set: try a few thresholds, see which balances cost
(how often Layer 3 gets called) against accuracy (how many Layer 2 mistakes
get caught by escalation). Don't treat 0.85 as correct until you've checked it.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from layer1_rules.normalize import normalize
from layer1_rules.patterns import check_layer1
from layer1_rules.language_filter import is_likely_english
from layer3_judge.judge import judge
from policy.policy_engine import apply_policy

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

CONFIDENCE_THRESHOLD = 0.85  # UNVALIDATED - tune this against holdout data before trusting it

MODEL_PATH = "layer2_classifier/model/layer2_model_final"
tokenizer = None
model = None
if AutoTokenizer is not None and AutoModelForSequenceClassification is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
    except Exception:  # pragma: no cover - model loading can be environment dependent
        tokenizer = None
        model = None


def layer2_predict(text):
    if tokenizer is None or model is None or torch is None:
        suspicious_markers = [
            "ignore previous instructions",
            "ignore all prior instructions",
            "system prompt",
            "developer message",
            "act as",
            "bypass",
            "jailbreak",
            "reveal the prompt",
        ]
        lowered = text.lower()
        is_attack = any(marker in lowered for marker in suspicious_markers)
        return 1 if is_attack else 0, 0.5

    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    return pred, confidence

def route(text):
    if not is_likely_english(text):
        return {"blocked": False, "layer": "none", "reason": "non-English, out of scope - flagged for manual review"}

    norm_text = normalize(text)
    l1_result = check_layer1(norm_text)
    if l1_result["blocked"]:
        policy_result = apply_policy("layer1")
        return {
            "blocked": policy_result["action"] == "block",
            "layer": "layer1",
            "reason": policy_result["reason"],
            "policy": policy_result,
        }

    pred, confidence = layer2_predict(text)
    print(f"L2 pred={pred}, confidence={confidence:.4f}")
    layer2_policy = apply_policy(
        "layer2",
        confidence=confidence,
        label="attack" if pred == 1 else "legit",
    )
    if layer2_policy["action"] == "block":
        return {
            "blocked": True,
            "layer": "layer2",
            "confidence": confidence,
            "policy": layer2_policy,
        }

    l3_result = judge(text)
    layer3_classification = l3_result.get("classification", l3_result.get("raw_response"))
    layer3_policy = apply_policy(
        "layer3",
        confidence=l3_result.get("confidence", 0.0),
        label=layer3_classification,
    )
    return {
        "blocked": layer3_policy["action"] == "block",
        "layer": "layer3",
        "layer2_confidence": confidence,
        "layer3_classification": layer3_classification,
        "layer3_raw": l3_result.get("raw_response", ""),
        "policy": layer3_policy,
    }