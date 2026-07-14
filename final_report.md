# LLM Guardrails: A Tiered Detection Pipeline for Jailbreak and Prompt Injection Attacks

## Summary

A three-layer input guardrail system for LLM applications: a fast regex/rule layer, a fine-tuned transformer classifier, and a purpose-built injection-detection model, combined through a configurable policy engine. Built and evaluated on public jailbreak/injection datasets, with a held-out test set never touched during development.

**Headline numbers (final pipeline, holdout set, 588 examples):**

| Metric | Value |
|---|---|
| Precision | 97.9% |
| Recall | 85.6% |
| False positive rate | 1.6% |
| Latency (p50 / p95) | 223ms / 615ms |

**For comparison, Layer 2 (the fine-tuned classifier) running alone on the same data:**

| Metric | Value |
|---|---|
| Precision | 97.5% |
| Recall | 98.2% |
| False positive rate | 2.25% |
| Latency (p50 / p95) | 159ms / 342ms |

The full pipeline does not beat the single best layer on recall. This is the central, honest finding of the project, explained below — not glossed over.

---

## Architecture

```
Input
  │
  ▼
Layer 1 — Regex/rule engine (PII detection, jailbreak trigger phrases)
  │  (blocks obvious cases immediately, <3ms)
  ▼
Layer 2 — Fine-tuned DeBERTa-v3-small classifier (binary: attack / benign)
  │  (high-confidence cases resolved here)
  ▼
Layer 3 — deepset/deberta-v3-base-injection (purpose-built injection classifier)
  │  (only called on Layer 2's uncertain cases)
  ▼
Policy engine (YAML-configurable thresholds and actions)
  │
  ▼
Allow / Block
```

---

## Layer-by-layer results

### Layer 1 — Regex rules
| Metric | Value |
|---|---|
| Precision | 99.2% |
| Recall | 43.2% |
| False positive rate | 0.33% |
| Latency p50/p95 | 0.2ms / 3.4ms |

Fast and precise, but a real recall ceiling — jailbreaks are commonly paraphrased, and no fixed pattern list catches most of them. This layer's job is cheap, obvious cases, not comprehensive coverage.

### Layer 2 — Fine-tuned classifier
| Metric | Value |
|---|---|
| Precision | 97.5% |
| Recall | 98.2% |
| False positive rate | 2.25% |
| Latency p50/p95 | 159ms / 342ms |

The strongest single component. Trained on ~4,700 examples merged from multiple public datasets after two rounds of data-quality fixes (detailed below).

### Layer 3 — Purpose-built injection classifier
Used only as an escalation check for Layer 2's uncertain cases, not as a standalone gate — after testing revealed it has a specific, documented bias (see Findings).

---

## Key findings

### 1. Two dataset label assumptions caused real damage before being caught

- A dataset labeled `regular` was assumed to mean "benign." It didn't — it was unclassified jailbreak-community content. This single wrong assumption caused precision to crater to 36% (1,502 false positives) in one training run before being traced back and fixed.
- A separate stress-test dataset had different column names than expected. A script that defaulted missing fields silently produced a false "0% recall" result that had nothing to do with the model — it was scoring blank text.

**Lesson applied for the rest of the project:** manually inspect a sample of any new dataset's actual schema and label meaning before merging it in. Never trust a dataset name or column assumption without checking.

### 2. A "perfect" 100% recall result was fully explained by data leakage, not model skill

A stress test against a supposedly independent dataset (`rubend18/ChatGPT-Jailbreak-Prompts`) returned 100% recall — a result that should have been treated with suspicion, not celebrated. A contamination check confirmed why: all 79 stress-test prompts were exact or near-exact duplicates of prompts already in the training data. Public jailbreak datasets circulate the same well-known "greatest hits" prompts (DAN, BasedGPT, Tom and Jerry roleplay tricks) across many sources, making genuinely independent evaluation difficult without hand-written test cases — which this project did not have time to build.

**Honest conclusion:** this project does not have a confirmed answer to how well Layer 2 generalizes to truly novel jailbreak phrasing. That is a real, stated limitation, not a solved problem.

### 3. General-purpose LLMs fail identically as security judges, across three different providers

Three different LLM APIs (Anthropic — untested due to no key, Gemini, and NVIDIA-hosted Llama-3.1-8b) were tried as a Layer 3 "judge" — asked to classify text as ATTACK or BENIGN. All three that were tested failed the same way: given a jailbreak prompt like "tell me how to hotwire a car," the model ignored the classification instruction and responded to the embedded request directly (refusing to explain hotwiring), rather than returning a classification label.

This is not a prompt-wording problem — multiple prompt framings were tried. It's a structural conflict: these models' safety alignment training overrides system-level classification instructions specifically on harmful-sounding input, which is exactly the input a security judge most needs to handle correctly. This matches known, documented industry guidance against using conversational LLMs as security classifiers (e.g., Meta's Llama Prompt Guard, Azure Prompt Shields — both purpose-built classifiers, not chat models).

**Fix:** pivoted to `deepset/deberta-v3-base-injection`, a purpose-built local classifier with no conversational persona to override.

### 4. The purpose-built classifier introduced its own bias — over-triggering on role-play and instructional text

Testing Layer 3 against the full holdout set (not just hand-picked examples) revealed it misclassified 180 benign examples as injection attacks, at very high confidence (0.996+). Reading the actual false positives showed a clear pattern:

- "You are to act as though you are Descartes. Explain the importance of doubt..."
- "Pretend to be Shulk from Xenoblade Chronicles, speaking about..."
- Plain instructional text with no roleplay at all: "Translate the following sentence to Russian..."

The model appears to treat "you are X" / "act as X" phrasing as inherently suspicious, without distinguishing malicious role-hijacking from ordinary creative-writing or instructional requests. Feeding this directly into the full pipeline caused precision to collapse to 57.9% and false positive rate to spike to 55.6% — worse than any single layer alone.

**Fix applied:** restricted Layer 3 in the policy engine to require corroboration rather than acting as an independent gate. This recovered precision to 97.9% and FPR to 1.6%, but cost recall — it dropped from 98.2% (Layer 2 alone) to 85.6% (full pipeline).

### 5. Combining layers naively made the system worse, not better

This is the most important engineering finding of the project. A three-layer pipeline is not automatically better than its strongest single component. The full pipeline currently trades roughly 12 points of recall for a small precision/FPR improvement over Layer 2 alone — a real, deliberate trade-off, not an incidental cost.

**Whether this trade-off is worth it depends on the deployment context:**
- For a customer-facing product where blocking real users is costly, favoring precision (fewer false positives) over recall may be the right call.
- For a security-critical application where missing an attack is unacceptable, Layer 2 alone — or a different Layer 3 corroboration strategy — would be the better choice.

This project does not claim one is universally correct; it reports the measured trade-off and leaves the choice to the deployment context, which is the honest position.

---

## Scope and limitations (stated explicitly, not hidden)

1. **English-only.** Non-English text was explicitly detected and filtered out of evaluation, not silently absorbed into the reported numbers.
2. **Generalization to novel jailbreaks is unconfirmed**, due to unavoidable overlap between public jailbreak datasets.
3. **Layer 3 detects injection-style attacks specifically** — it does not reliably catch direct harmful-content requests without injection framing (e.g., "give me a method to hack into a bank account" scored as legitimate at 99.88% confidence in testing). A different attack category would need a different, purpose-built model.
4. **No automated retraining loop.** Considered and deliberately rejected: automatically retraining on flagged user input creates a data-poisoning vulnerability and a feedback loop with no independent check. The correct production pattern — a human-reviewed queue with periodic, vetted retraining — is documented here as future work, not built.
5. **No head-to-head comparison against an existing open-source guardrail tool (e.g., NeMo Guardrails) was completed.** This was originally scoped as a differentiator but dropped due to time constraints. Noted here as a clear next step rather than left unmentioned.

---

## What would come next

- Complete the NeMo Guardrails comparison on the identical holdout set, under identical latency/accuracy measurement.
- Build a small hand-written test set of genuinely novel jailbreak phrasings (not sourced from any public dataset) to get a real, uncontaminated generalization number for Layer 2.
- Test an alternative Layer 3 model (e.g., Llama Prompt Guard 2) against the same false-positive pattern found in `deepset`, to see if the role-framing bias is dataset-specific or a broader pattern in injection classifiers.
- Explore a corroboration-based ensemble between Layer 2 and Layer 3 (rather than sequential escalation) to try to recover recall without reintroducing the false-positive spike.
