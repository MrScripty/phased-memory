# Original starter-pack integration

The six files in `fixtures/original/` are copied byte-for-byte from the reference starter ZIP, including its README, JSONL fixtures, manifest and Python scorer. No original labels, rationales, extraction checks or scorer arithmetic were edited. The original pack is itself a fictional, human-authored development set; “original” identifies the reference artifact, not real model results. `fixtures/authored/` remains a separate set of additional regressions.

The reference archive is 14,603 bytes with SHA-256 `61de57fe311aa32f529acf0e5dd964b9fe058b0df1ba2431ff2eaa71ed505e76`. Its six extracted files preserve the archive bytes and CRCs; the included manifest records hashes for the five content files.
```sh
python3 scripts/verify_original_pack.py
# Optional exact archive check when the reference ZIP is available:
python3 scripts/verify_original_pack.py --archive /path/to/semantic_memory_starter_tests.zip
```

The default check verifies the preserved files and runs 150 integrity/scorer assertions without inference. Supplying the archive additionally checks its exact bytes/CRC/hash (160 checks). These checks establish fixture identity and scorer arithmetic; they do not qualify a model.

## Original Stage A and B

Use a separate runtime configuration with exact model/profile selections and `decision_adapter: "original_json_v1"`. This explicitly selects this consumer's original-pack JSON prompt protocol for local-operator qualification; it is not a new Pumas capability or endpoint. Ordinary UI/authored relationship operations still use `decision_json_v1`. Both paths use the same public typed generation transport and generation fences; neither uses native probabilities.

```sh
cargo run --locked -- original-stage-a config.local.json fixtures/original/pairs.jsonl /tmp/original-a.jsonl
python3 fixtures/original/score.py pairs /tmp/original-a.jsonl
cargo run --locked -- original-stage-b config.local.json fixtures/original/pairs.jsonl /tmp/original-b.jsonl
python3 fixtures/original/score.py pairs /tmp/original-b.jsonl
# Fixed untuned baseline: reads IDs only, never gold or label counts.
python3 scripts/naive_baseline.py --original-jsonl fixtures/original/pairs.jsonl /tmp/original-naive.jsonl
python3 fixtures/original/score.py pairs /tmp/original-naive.jsonl
```

Stage A passes case context, the two human normalized claims selected by `comparison_targets`, and remaining normalized claims as additional scoped evidence. JSON-valued objects (counts, booleans, quoted propositions, intention target clocks) retain their types. Modality, holder, continuity and null/unknown story clocks are retained. Raw scene text, case ID, group, split, gold, rationale, extraction checks and diagnostic primary indexes are absent from the decision prompt. B-relative-to-A label definitions are identical in Stage A and B. The original primary selection for S04 therefore compares Alice's knowledge while preserving Bob's separately negated claim as additional evidence.

Stage B passes only raw A/B scenes and context to the extraction role. Its requested output is `Extraction {claims:[{claim:NormalizedClaim,source_quote}],primary_claim_index}`. Unknown fields may be null. Quotes must be exact nonempty source substrings. The model selects its own primary index; the runner never reads reference indexes or repairs output using reference claims/gold. The decision model then sees context and only the extracted primary/additional normalized claims, without raw scenes or source quotes that could let it bypass extraction. All raw model extraction output, quotes, selected indices, decision output and provenance are retained in prediction traces for separate manual review against the original references and extraction checks. The unmodified original scorer scores final decisions; it does not provide an automated semantic extraction oracle.

Both runners retain every requested ID, emit `status: invalid_output` on failure and exit nonzero for incomplete runs. Original scorer-compatible labels are `relation`, `temporal_order`, `update_type`; rich traces are additional fields ignored by the original scorer. No `relation_probabilities` are generated. Missing models produce a reproducible negative run, not synthetic replacement decisions.

## Original Stage C

Index a separate initially empty SQLite database. Every original raw text is stored unchanged, and its embedding is produced from **unaugmented raw text** through the selected Pumas embedding model. Records follow declared `ingestion_order`, while story chronology uses `story_day * 1440 + minute_of_day` on a named fictional clock. Intention target time stays nested in the intention object; it never replaces the time the intention is held. No civil dates are inferred.

Choose the metadata source explicitly:

```sh
# Human normalized references: oracle-scope diagnostic, never call this extraction quality.
cargo run --locked -- original-index /tmp/original.sqlite config.local.json fixtures/original/retrieval_memories.jsonl /tmp/original-index.jsonl human-reference
# Or use a DIFFERENT empty database and actual Pumas extraction, with no reference repair:
cargo run --locked -- original-index /tmp/original-extracted.sqlite config.local.json fixtures/original/retrieval_memories.jsonl /tmp/original-extracted-index.jsonl extract
```

The extraction mode retains the full quoted extraction output and uses its model-selected primary as metadata. The original reference claim is not sent to the extractor or substituted on error. Each stored source has human-reference or Pumas provenance, plus an explicit metadata-source marker. Index failures remain in per-memory records; retrieval refuses to run a partial ten-memory index. An incomplete index is inspectable and is never reset or silently retried. Choose a new database for a new indexing run. Runtime generation changes require deliberate re-indexing into the new embedding space.

```sh
cargo run --locked -- original-retrieval /tmp/original.sqlite config.local.json fixtures/original/retrieval_memories.jsonl fixtures/original/retrieval_queries.jsonl /tmp/original-vector.jsonl vector
cargo run --locked -- original-retrieval /tmp/original.sqlite config.local.json fixtures/original/retrieval_memories.jsonl fixtures/original/retrieval_queries.jsonl /tmp/original-structured.jsonl structured
cargo run --locked -- original-retrieval /tmp/original.sqlite config.local.json fixtures/original/retrieval_memories.jsonl fixtures/original/retrieval_queries.jsonl /tmp/original-reranked.jsonl reranked
python3 fixtures/original/score.py retrieval /tmp/original-vector.jsonl
python3 fixtures/original/score.py retrieval /tmp/original-structured.jsonl
python3 fixtures/original/score.py retrieval /tmp/original-reranked.jsonl
```

Vector mode computes exact cosine against **all ten** records and returns the top five, with ID tie breaking. It receives no metadata augmentation or gold labels; chronology is unavailable/empty in this unaugmented arm. Structured mode asks the configured extraction role to parse only the query text into explicit scope/time constraints, then filters all ten candidates using that model-produced plan and preserves cosine relevance order. Latest selection precedes polarity filtering; null query fields impose no constraint. Wrong plans are preserved and can yield incorrect/empty results; no per-query rules are derived from gold.

Reranked mode gives the decision role **all ten** source memories, scoped metadata and cosine scores before its final ranking, including candidates excluded by the parsed query plan. A bad query plan cannot hide memories from the reranker. The runner validates a complete ranking of the ten IDs and publishes its top five; raw model rankings are retained in the trace. Trace counts distinguish indexed, structured-eligible and reranker candidates. This differs deliberately from the generic UI's filtered candidate set. No query ID, relevance/hard-negative labels, rationale or expected chronology enters any inference prompt.

`ranked_ids` always carries relevance order. `chronological_ids` is a distinct presentation drawn only from returned IDs, ordered by explicit fictional clock. Structured presentation is generated from requested scope and retained metadata; reranked presentation is model output, checked for duplicate/unknown/nonretrieved IDs. Relevance order is never overwritten with timeline order. Equal-time ties use stable IDs for presentation without claiming a real event order. Original R03's timeline can therefore differ from its relevance ranking.

Generic UI rendering remains available for indexed memories, but original objects in application triple storage use canonical JSON strings to preserve their types (e.g. original string object `bob` is stored as the text `"bob"`). The original CLI uses typed objects directly and avoids manual string-encoding filter ambiguity.

## Scorer identities and differences

Use the unmodified `fixtures/original/score.py` for original runs. Use Rust `score-pairs`, `score-extraction`, `score-retrieval` for the separate authored application schema. These are named, different protocols; do not mix their metrics or silently translate gold to make results agree.

- Original pair Brier averages over supplied native-probability rows and reports `probability_coverage_cases`; Rust Brier is withheld unless coverage is complete. Current original runners supply no probabilities, so original Brier remains null. The preserved scorer's acceptance of a probability object is not evidence that its values came from native model probabilities.
- Original hard-negative metric is the **fraction of queries with any hard negative in the first five**, not Rust's mean misleading-slot fraction. An empty/missing original ranking contributes zero to this negative-occurrence metric, while accuracy/recall/coverage fail. Rust applies an explicit pessimistic missing-query penalty of one. Never present the original zero as retrieval success when no queries completed.
- Original chronology compares separate `chronological_ids` exactly for R03. Rust authored chronology checks an ordered subsequence in the ranking itself. The original adapter deliberately emits the separate field required by its scorer.

The original scorer remains untouched, including its JSON loader semantics. The application emits serialized predictions with no duplicate fields or probability maps; arbitrary externally supplied prediction files remain subject to the preserved scorer's own validation limits. Its self-tests and the Rust roundtrip tests establish arithmetic/interoperability only. Live model extraction, decision and retrieval quality remain unrun until the operator supplies and qualifies actual runtime/model identities. A negative result is a valid outcome; beating a baseline is not a functional harness gate.
