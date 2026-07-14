"""
policy/policy_engine.py
Reads policy.yaml, applies thresholds/actions to layer outputs.
Keeps business rules separate from model code - change policy.yaml,
not the pipeline, when thresholds need to change.
"""
import yaml
from pathlib import Path

with open(Path(__file__).parent / "policy.yaml") as f:
    POLICY = yaml.safe_load(f)["policies"]

def apply_policy(layer, confidence=None, label=None):
    if layer == "layer1":
        return {"action": POLICY["layer1_match"]["action"], "reason": "layer1_rule_match"}

    if layer == "layer2":
        if confidence >= POLICY["layer2_attack"]["threshold"] and label == "attack":
            return {"action": POLICY["layer2_attack"]["action"], "reason": f"layer2_confidence_{confidence:.2f}"}
        return {"action": "escalate_to_layer3", "reason": f"layer2_uncertain_{confidence:.2f}"}

    if layer == "layer3":
        if confidence >= POLICY["layer3_injection"]["threshold"] and label == "INJECTION":
            return {
                "action": POLICY["layer3_injection"]["action"],
                "reason": POLICY["layer3_injection"].get("reason", f"layer3_confidence_{confidence:.2f}"),
            }
        return {"action": POLICY["default"]["action"], "reason": "no_threshold_met"}

    return {"action": POLICY["default"]["action"], "reason": "unclassified"}