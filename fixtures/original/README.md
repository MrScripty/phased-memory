# Semantic memory: starter test set

Version 0.1 · 26 October 2026 · Fictional, hand-authored development fixtures

## Purpose and status

Test whether the standalone prototype distinguishes incompatible claims from changes over story time, then whether those distinctions improve retrieval. The suite contains **24 labeled claim-pair cases**, a separate **10-memory retrieval corpus**, and **8 labeled retrieval queries**.

These are proposed test fixtures, not measured Kev or Gemma results, not a representative benchmark, and not a held-out test set. Review the labels before freezing a later benchmark. No model inference has been run as part of preparing this pack. The scorer's internal checks verify fixture and scoring integrity only.

All model inference must go through Pumas Library. This pack deliberately specifies no unverified Pumas endpoint, provider-specific request, or model-loading implementation. The local agent working on Pumas owns those integration changes. Model selections, quantization and projector configuration are run metadata, not assumptions baked into the fixtures.

## Files

| File | Purpose |
| --- | --- |
| `pairs.jsonl` | 24 claim pairs, normalized reference claims, expected labels, rationales and extraction checks |
| `retrieval_memories.jsonl` | 10 independent retrieval memories with story time, scope and ingestion order |
| `retrieval_queries.jsonl` | 8 queries with relevant IDs, hard negatives and one chronological-order target |
| `score.py` | Dependency-free Python 3 scorer for pair predictions and retrieval rankings |
| `manifest.json` | SHA-256 hashes and fixture counts for reproducible handoff |

## Label contract

The comparison is **B relative to A**, using the primary claims selected by `comparison_targets`. Do not collapse the following three axes into a single label.

### 1. `relation`

| Value | Meaning |
| --- | --- |
| `supports` | B directly repeats, corroborates or entails A in the same entity, perspective, continuity and time scope. This is directional. |
| `contradicts` | The primary proposition contents cannot both be true in the same relevant scope and time, including stated uniqueness or exclusivity constraints. |
| `compatible` | The stated scopes or propositions allow both claims to hold without B establishing A. Different times, subjects, belief holders or continuities can make opposing wording compatible. |
| `insufficient_information` | Missing material scope prevents choosing among the other labels. For example, an unspecified time could turn apparent opposition into either a contradiction or a state change. |

Apply `supports` or `contradicts` when established. Otherwise use `compatible` when the supplied context resolves the distinction, and `insufficient_information` when material scope is unresolved. This is an operational annotation policy: the last label does not mean that simultaneous truth is logically impossible.

An authoritative correction still has `relation=contradicts` when its replacement content conflicts with the old content. Its `update_type` records that this is a correction, not a physical change over time. Do not treat both corrected and replacement claims as currently canonical.

### 2. `temporal_order`

`same_time`, `b_later`, `b_earlier`, or `unknown`.

Compare the valid story time of the selected claims, not source order, ingestion order or the date the author edited the script. `story_day` and `minute_of_day` form the fixture's fictional clock; they are not real-world dates. For a plan, the time the plan is held is distinct from the target time of its intended action.

### 3. `update_type`

| Value | Meaning |
| --- | --- |
| `none` | No state transition or authoritative correction is established. |
| `state_change` | Observations establish different states at distinct story times. This can hold even when B describes an earlier event. It does not supply an exact transition time. |
| `explicit_correction` | B explicitly and authoritatively replaces the earlier record about the same event or time. Preserve the old record and the correction link. |
| `undetermined` | Missing scope prevents deciding whether a state change, correction or no update occurred. |

A changed state is normally `compatible`, not `contradicts`. A changed record can be an `explicit_correction`. Mere ingestion recency establishes neither.

## Fixture coverage

| IDs | Count | What they test |
| --- | ---: | --- |
| C01–C04 | 4 | Same-time negation, exclusive locations, exact counts, directional trust |
| T01–T04 | 4 | Gate states, lost trust, movement and a late-ingested flashback |
| P01–P04 | 4 | Explicit correction, lying dialogue, mistaken belief and separate continuities |
| N01–N04 | 4 | Reversed roles, different subjects, lack of trust versus distrust, unexecuted plans |
| U01–U04 | 4 | Missing time, ambiguous identity, missing speaker/narration scope, unknown continuity |
| S01–S04 | 4 | Repetition, paraphrase and support with an additional independently negated claim |

The relation distribution is 4 supports, 5 contradictions, 11 compatible and 4 insufficient-information cases. Always predicting compatible would score 11/24 on relation accuracy; accuracy alone is therefore inadequate.

Narrator-established facts are authoritative within their stated continuity unless the source explicitly says otherwise. Dialogue establishes what a character said, not automatically what is true. Belief, intention and quoted statements retain their scopes. Do not infer that 'does not trust' means 'actively distrusts'.

The reference claims are an inspection target, not an exhaustive proposition parser. Some scenes include additional evidence such as a narrator identifying a lie. Preserve that evidence and check the case-specific `extraction_checks`; do not discard it simply because the main comparison selects one claim.

## Run the experiment in three stages

### Stage A — isolate decision quality

Give Kev the case context and the two human-authored normalized primary claims, using the label definitions above. Do not send `gold`, `rationale`, or `extraction_checks` to the model. The application assigns the case ID; it must not depend on the model inventing one.

Request separate decisions for relation, temporal order and update type. Where available, retain the native candidate probabilities for the relation decision. Do not manufacture probabilities from a text explanation. Unsupported probability output is recorded as unavailable, not replaced with zero or a uniform distribution.

An additional raw-text-only Kev run is a useful optional diagnostic. Keep it separate from the normalized-claim and end-to-end runs rather than averaging them.

### Stage B — isolate extraction errors

Give Gemma the raw A/B text and context, without the reference claims or labels. Require structured claims with subject, predicate, object, polarity, modality, holder, continuity, story time and an exact supporting source quote. Allow unknown fields.

Inspect the extraction against the reference claims and `extraction_checks`, then run Kev on Gemma's output with the same decision instructions used in Stage A. Save both outputs. Report when a wrong final decision came from extraction rather than the decision model.

Do not automatically repair a wrong extraction using the gold data before evaluating the end-to-end run. The provided scorer evaluates the final decisions; extraction inspection remains a separate recorded review in this starter pack.

### Stage C — retrieval comparison

Index `retrieval_memories.jsonl`, then run every query in `retrieval_queries.jsonl` against all 10 memories. Use exact scoring at this size. The candidate set must not contain the gold relevance labels.

Compare:

- **A: embedding baseline** — the same fixed embedding model, unaugmented text and cosine similarity.
- **B: structured retrieval** — the same embeddings plus extracted scope, time and relationship information; include Kev reranking when available.
- **C: optional phase experiment** — identical semantic information to B, with a documented phase representation. It is not a prerequisite for the first working test runner.

Do not turn Kev confidence directly into phase angles without a defined representation and objective. To attribute an improvement to phase, B and C must have identical information access. Include an equivalent real-valued representation as a later control. Do not describe a metadata improvement as proof that wave representations are superior.

In this tiny corpus, retrieve from all 10 records. When later using candidate preselection, separately measure whether relevant records reached the reranker.

For R03, keep `ranked_ids` for retrieval evaluation and `chronological_ids` for story-time presentation as distinct outputs. A relevance ranking is not necessarily a timeline.

## Prediction files and scoring

Python 3 and the standard library are sufficient. The scorer makes no inference calls.

### Pair prediction format

Write one JSON object per line. Example structure below is illustrative, not a measured model output:

```json
{"id":"C01","relation":"contradicts","temporal_order":"same_time","update_type":"none","relation_probabilities":{"supports":0.02,"contradicts":0.94,"compatible":0.02,"insufficient_information":0.02}}
```

`relation_probabilities` is optional. When supplied, it must contain the four relation labels and a finite probability distribution summing to one. For a failed model call or unparsable output, preserve the expected ID and use `{"id":"C01","status":"invalid_output"}`. Do not remove failed cases from the denominator.

```bash
python score.py pairs pair-predictions.jsonl > pair-scores.json
```

The output includes field accuracies, relation macro-F1, a confusion matrix, joint three-field accuracy, and per-case errors. When native probabilities are available, it also reports multiclass Brier score with its coverage count. Brier is the mean sum of squared probability errors across the four classes; lower is better. It is not a standalone measure of calibration, and 24 examples cannot establish reliable calibration.

### Retrieval prediction format

```json
{"id":"R01","ranked_ids":["M02","M01","M06","M10","M08"]}
```

For R03 additionally supply `chronological_ids` containing the retrieved historical evidence in chronological order.

```bash
python score.py retrieval retrieval-predictions.jsonl > retrieval-scores.json
```

Metrics are top-1 relevance, mean reciprocal rank, recall@5, the fraction of queries with any hard negative in the top five, and exact timeline order for R03. A hard negative is a deliberately misleading but wrong match for that query. It is not a globally invalid memory. Non-hard-negative distractors are still irrelevant; the hard-negative metric supplements rather than replaces ordinary relevance metrics.

Missing predictions score as failures or empty rankings. Duplicate or unexpected IDs and invalid probability distributions stop scoring with a clear error so malformed runs cannot silently appear complete.

## Reproducibility record

For each run save the fixture manifest hash, app commit, Pumas version/commit, provider and runtime version, exact model IDs and artifact hashes, quantization, embedding configuration, prompt/schema version, decoding options, context limits, hardware, stage latencies and all raw outputs. Distinguish loading latency from warm inference. Record unsupported capabilities and retries explicitly.

Never overwrite original source memories during an evaluation. Store derived claims and judgments with versioned provenance. Corrections, changed states and alternate drafts must be inspectable independently.

## Acceptance and the next dataset

The software passes its initial acceptance when it can run every case, preserve outputs and failures, score the two stages independently, show retrieval side by side, and reproduce metrics from saved predictions. It must be able to report a negative research result honestly. Beating a baseline is not a prerequisite for declaring the test harness functional.

Before choosing a model or claiming an improvement, extend toward roughly 200 carefully reviewed cases, with paraphrases and harder natural script excerpts. Split by scene/scenario family, not by individual near-duplicate rows. Keep the final evaluation partition hidden from prompt tuning, training and threshold selection. Compare systems on the same frozen cases, report category-level errors and uncertainty, and never use Kev's own judgments as its independent ground truth.

The first research question is whether reliable semantic information helps retrieval. The second is whether phase adds anything over conventional use of the same information.
