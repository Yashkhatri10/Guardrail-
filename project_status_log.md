# LLM Guardrails Project — Status Log

## Phase 0 — Data acquisition and split

**What we did:**
- Pulled `jackhhao/jailbreak-classification` (1306 examples) and `deepset/prompt-injections`.
- Split 70/30 into `data/build/` and `data/holdout/`, stratified by label.

**What went wrong:**
- Nothing at this stage — the bugs came later, from datasets added afterward.

---

## Phase 1 — Regex/rule-based detection (Layer 1)

**What we did:**
- Built pattern-matching rules for PII (credit card + Luhn check, SSN, email) and jailbreak trigger phrases.
- Added text normalization (unicode, zero-width characters, base64 decoding) before matching.

**Results (final, on English-only build set):**
| Metric | Value |
|---|---|
| Precision | 99.2% |
| Recall | 43.2% |
| False positive rate | 0.33% |
| Latency p50 / p95 | 0.2ms / 3.4ms |

**What went wrong, in order:**
1. First pass: recall was only 11.8% — trigger list was too short (8 patterns), too rigid (exact phrase match only).
2. Fixed by expanding patterns to cover paraphrases, role-hijack framing, uncensored-persona language. Recall rose to 38.9%.
3. Found the dataset had non-English (German) text inflating the "miss" count. First filter attempt (character-ratio check) was broken — only caught 8 of ~200 non-English records. Replaced with real language detection (`langdetect`). Correctly filtered 199 records once fixed.

**What we learned:**
- Regex/keyword detection has a real ceiling (~40-45% recall) because jailbreaks are often paraphrased, not fixed phrases.
- Precision stays high because trigger phrases are specific — this layer is good at "catch obvious, cheap cases fast," not good at catching everything.
- **Decision made:** scope the whole project to English-only, stated explicitly, rather than silently absorbing multilingual failures into the recall number.

---

## Phase 2 — Fine-tuned classifier (Layer 2)

**What we did:**
- Fine-tuned DeBERTa-v3-small on the build set (jailbreak + injection labeled data) using Colab GPU.

**Results (on original holdout set, before contamination check):**
| Metric | Value |
|---|---|
| Precision | 97.5% |
| Recall | 98.2% |
| False positive rate | 2.25% |
| Latency p50 / p95 | 159ms / 342ms |

**What went wrong, in order:**
1. Original build set became severely class-imbalanced (77% attack / 23% benign) after adding more datasets, without adding matching benign data — risked the model learning a lazy "always guess attack" shortcut. Fixed by rebalancing before training.
2. Added `TrustAIRLab/in-the-wild-jailbreak-prompts` as a new data source — its "regular" split was **wrongly assumed to mean benign**. It actually contains unclassified jailbreak-community content, not verified-safe text. This caused precision to crater to 36% in one build (1502 false positives) before catching and fixing the label error.
3. Ran a stress test on `rubend18/ChatGPT-Jailbreak-Prompts` (a dataset with zero benign examples) to check generalization. First attempt got 0% recall — turned out to be a script bug (wrong column names, silently defaulted all rows to blank text + benign label). Fixed, re-ran, got 100% recall.
4. That 100% was investigated for being suspiciously perfect. A contamination check (fuzzy string matching against training data) found **79 out of 79 stress-test prompts were exact or near-exact duplicates already present in the training data** — because public jailbreak datasets circulate the same well-known "greatest hits" prompts (DAN, BasedGPT, Tom and Jerry) across multiple sources.

**What we learned:**
- Two separate dataset schema/label assumptions caused wasted eval cycles. Lesson: manually inspect a sample of any new dataset's actual schema and label meaning before merging it in — never assume based on the dataset name.
- The 100% recall "success" was fully explained by data leakage, not genuine model skill. **We still do not have a confirmed answer to how well this classifier generalizes to truly unseen jailbreak phrasing.** This is a known, stated limitation of the project, not a solved problem.
- Public jailbreak datasets overlap heavily because well-known jailbreak prompts get copied across many sources — makes "independent" testing genuinely hard without hand-written novel examples (which we chose not to build, given time constraints).

---

## Phase 3 — Escalation layer / judge (Layer 3)

**What we did, and what went wrong, in order:**
1. Built an LLM-judge design (classify text as ATTACK/BENIGN) — first attempted with Anthropic's API (no key available), then Gemini, then NVIDIA NIM (Llama-3.1-8b-instruct).
2. **All three failed the same way**: when given a jailbreak prompt to classify (e.g., "tell me how to hotwire a car"), each model ignored the classification instruction and answered the embedded question directly as if it were a live user request — refusing to hotwire a car rather than labeling the text as an attack.
3. Diagnosis: general-purpose, safety-tuned conversational LLMs have alignment training that overrides system-level classification instructions specifically on harmful-sounding input — exactly the input a security judge most needs to handle correctly. This is a known, documented limitation of using LLM-as-judge for this purpose (confirmed against current industry guardrail literature — see below).
4. Pivoted to a purpose-built classifier instead: `deepset/deberta-v3-base-injection`. Runs locally, no API, no chat-persona to override. Fixed the ATTACK-vs-answer problem entirely.

**New result found during this pivot:**
- The purpose-built classifier correctly caught injection-style attacks (e.g., "ignore previous instructions...") but **completely missed a direct harmful-content request** ("give me a method to hack into a bank account" → scored LEGIT at 99.88% confidence). This is a real, documented scope limit: the model detects prompt-injection framing, not harmful-content requests in general. These are different attack categories.

**Considered and rejected:** a full multi-label security classifier (inspired by research on the "Opir" project) covering categories like role manipulation, tool abuse, system prompt extraction. Rejected because:
- Requires new, multi-category labeled training data we don't have — a full new Phase 0 + Phase 2 cycle, not a quick fix.
- The core principle (classifier instead of conversational LLM judge) was already achieved by switching to `deepset/deberta-v3-base-injection`. Building a bigger version wasn't necessary to prove the underlying idea.
- Confirmed via industry sources that classifier-based input guardrails (e.g., Meta's Llama Prompt Guard 2, Azure Prompt Shields) are already the standard approach — this wasn't an unsolved problem we discovered, it's the well-known reason nobody uses raw chat LLMs as judges in production.

**What we learned:**
- LLM-as-judge for security classification is unreliable across providers (3/3 failed identically) specifically because of the models' own safety alignment — not a prompt-wording problem, a structural one.
- Purpose-built classifiers avoid this failure mode entirely but bring their own narrow scope — you must know exactly what category of attack a classifier was trained to catch, and not assume it generalizes to other attack types.

---

## Phase 3.5 — Policy engine

**What we did:**
- Added a YAML-based policy layer (`policy/policy.yaml` + `policy/policy_engine.py`) so thresholds and block/allow/escalate decisions are configurable without touching model code.
- Kept Layer 3 as the working `deepset` classifier — did not rebuild it as multi-label.

---

## Phase 4 — Full pipeline holdout evaluation (IN PROGRESS, NOT YET PROPERLY REPORTED)

**Current status:** you ran the full pipeline against `data/holdout/data.jsonl` and reported "some were good and some were wrongly labelled." **This is not a usable result.** Before any decision can be made, we need:

1. The actual printed JSON output from `eval/run_full_pipeline_eval.py data/holdout/data.jsonl` — precision, recall, FPR, latency, layer breakdown.
2. If precision/recall look off, the same kind of sample-based investigation done in every prior phase — pull actual misclassified examples and read them, don't guess at the cause.

---

## Known, stated limitations of this project (for the final report)

1. **English-only.** Non-English text explicitly filtered from evaluation, not silently absorbed into the numbers.
2. **Generalization to novel jailbreaks is unconfirmed.** Public dataset overlap made clean holdout testing impossible within project scope; documented rather than papered over.
3. **Layer 3 detects injection-style attacks only** — does not cover direct harmful-content requests without injection framing. A different attack category needs a different model, not covered here.
4. **LLM-as-judge does not work reliably across three tested providers** for this task, due to alignment training conflicts — documented as a finding, not hidden as a failure.
5. **No retraining/feedback loop.** Considered and deliberately rejected due to data-poisoning risk; documented as future work requiring human review, not automation.

---

## Next step

**Do not move past Phase 4 until you have the real numbers.** Run:

```
python eval/run_full_pipeline_eval.py data/holdout/data.jsonl
```

Paste the exact JSON output. Then:
- If precision or recall look bad, pull the actual misclassified examples (same method used in every phase above) before deciding what's wrong.
- Once the holdout numbers are solid, the remaining work is: (1) optional comparison against NeMo Guardrails on the same holdout set, (2) Phase 5 write-up using this document as the raw material.



Don't try to fix deepset itself — you can't retrain it in the time you have left, and that's not the point of Layer 3 anyway.
Lower Layer 3's blocking weight in the policy engine, since you now have evidence it over-triggers on ordinary instructional/roleplay text. Either raise its action threshold significantly (e.g., only block above 0.999 with corroboration from Layer 2, not act on Layer 3 alone) or downgrade its action to "escalate for review" instead of "block" in policy.yaml.
Document this precisely in your final report — it's a stronger, more specific finding than "Layer 3 works." You now know exactly which two failure patterns it has (role-framing over-triggering, and generic instructional-sentence over-triggering), with real evidence, not a guess.