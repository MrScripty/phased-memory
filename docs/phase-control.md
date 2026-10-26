# Optional phase expected-equivalence experiment

This is an opt-in CLI experiment, implemented independently of the ordinary UI and retrieval pipelines. It introduces no model, provider, training, database dependency or ResonanceDB code. It tests a numerical identity and provides a separately gated learned embedding comparator. Neither a passing identity check nor the authored fixture proves a phase advantage or semantic retrieval quality.

## Commands and evidence

```sh
cargo run --locked -- phase-control fixtures/phase/control.json /tmp/phase-control.json
cargo run --locked -- phase-compare config.example.json fixtures/phase/control.json /tmp/phase-compare.json
```

Outputs must be new paths; existing evidence is never overwritten. `phase-control` requires no runtime and leaves `pumas_embeddings` null. It freezes the input projection/vocabulary/time scale and reports all candidate scores and rankings for complex and real-concatenated arms, per-query/global absolute score error, the `1e-12` tolerance and raw ranking-equality diagnostics.

`phase-compare CONFIG INPUT OUTPUT` computes the same control, then requests embeddings through the existing public Pumas typed capability/fence transport, using only the configured exact embedding model/profile. No decision/extraction model is needed. Each embedding receives exactly the corresponding `frozen.corpus[].canonical_semantics` or `frozen.queries[].canonical_semantics` string. Validated model vectors, provenance, exact embedding space and scored rankings are retained for review/recomputation. No source prose, ID, confidence, original ingestion order or gold label is included in this projection.

An absent embedding configuration yields explicit `unrun` arm/corpus/query records and performs no network inference. A configured observation or operation failure retains its actual error, including an unknown outcome when transport cannot establish completion. If any corpus embedding fails, query records remain `unrun`: the runner does not shrink the candidate pool or request query embeddings. Partial evidence is saved and the command exits nonzero. There are no inference retries or synthetic fallbacks. Generation-scoped spaces must match across the entire learned arm; changed spaces fail. A completed learned arm means the operation path returned vectors, not that the model has passed a quality benchmark.

## Explicit semantic input

The versioned input is:

```json
{
  "schema_version": 1,
  "data_kind": "authored_development",
  "candidate_count": 1,
  "corpus": [{"id":"memory-1","semantics":{
    "subject":"alice","predicate":"trusts","object":"bob","polarity":"affirmed",
    "perspective":{"holder":"narrator","modality":"fact"},"continuity":"main",
    "entities":["alice","bob"],"qualifiers":[],"state_key":null,
    "story_time":{"timeline":"fictional-minute","position":540}
  }}],
  "queries": [{"id":"query-1","semantics":{
    "subject":"alice","predicate":"trusts","object":"bob","polarity":"unknown",
    "perspective":{"holder":"narrator","modality":"fact"},"continuity":"main",
    "entities":["bob","alice"],"qualifiers":[],"state_key":null,"story_time":null
  }}]
}
```

`data_kind` is `authored_development` or `operator_supplied`; it is a declared origin label, not attestation. `candidate_count` must equal the full corpus length. Each arm scores every candidate for every query; there is no preselection or top-five truncation. Inputs support 1–128 memories and 1–128 queries, with at most 8,192 frozen categorical slots. Corpus and query IDs are unique within their separate namespaces and appear only in reporting and tie-breaking.

All semantic fields must be present. Subject, predicate, continuity and state key may be explicit null; no identity/continuity default is invented. Object is typed JSON, retaining nested quoted claims, numeric/bool values and intended target clocks. Perspective is a string, null, or exactly `{holder,modality}` with each category a string or null. Polarity is explicitly `affirmed`, `negated` or `unknown`. Entity and qualifier lists are sets: both numerical features and Pumas JSON use deduplicated, sorted values. Other JSON arrays retain order. Object keys are sorted recursively; serde_json owns escaping and number spelling. This is a reproducible local canonical projection, not a claim of RFC 8785 conformance.

Story time is null or a named timeline plus an explicit integer position. It is never inferred from source/ingestion order or the target time of an intention. Operator preparation must preserve the source clock and unknowns. Positions outside the exactly representable f64 integer range `[-(2^53-1),2^53-1]` are unsupported, avoiding silent integer conversion ambiguity; `i64::MIN` is rejected safely. No original fixture/gold conversion is performed automatically. If an operator prepares data for a separate quality evaluation, labels stay in a separate scorer-only artifact.

## Frozen mapping and comparison

`semantic-phase-control-v1` freezes the following source rules:

1. Build a lexicographically sorted vocabulary over corpus **and queries**, without labels. The directed tuple key is canonical JSON of `[subject,predicate,object,perspective,continuity]`; swapping roles changes its slot. Nulls and literal unknown strings retain distinct categorical representations.
2. At that tuple slot set affirmed to `(real=1,imaginary=0)`, negated to `(-1,0)`, unknown to `(0,1)`. Separate prefixed entity, qualifier, state-key and named-timeline presence slots have `(1,0)`. An unknown clock has its own presence slot. This is explicit polarity encoding, never confidence-to-angle conversion. Shared metadata may dilute the aggregate effect of tuple negation; no rule forces negated memories to receive a particular overall retrieval rank.
3. Known story position occupies a separate real channel for its named timeline, divided by `T=max(1,max(abs(position)))` over the frozen corpus and queries. Queries therefore participate in this declared transductive vocabulary/scale freeze. Preserve the original integer/timeline in the canonical semantic input. Time is not converted into an angle.
4. Normalize each nonzero complex vector by Euclidean norm. Score with `Re(sum(conj(query_j)*memory_j))`. The same-information ordinary numerical control concatenates `[Re(z),Im(z)]` and computes cosine independently in f64. The learned Pumas arm receives exactly the same semantic inputs as canonical JSON; it is a distinct learned operation, not a numerical implementation of this identity.
5. Every arm has the same full candidate budget. Sort descending by the raw computed score and then ascending ID for an **exact** score tie. Numerical equivalence applies to pairwise scores within `1e-12` only. Raw ranks are not rounded, repaired or forced to agree.

Floating-point near ties can produce different raw rankings despite passing score equivalence. A focused regression has `T=245`, a query at position49, same-tuple candidates at positions0 and100, plus an unrelated candidate setting the scale. On the tested Rust/Linux build the phase scores differ by one ulp while the real scores tie, so the ID tie rule reverses the two top ranks. `rankings_equal` honestly reports false, while `equivalence_passed` remains true because pairwise errors are below tolerance. This is a numerical tie effect, not evidence for phase superiority. The output's `equivalence_scope` makes that distinction explicit.

Different kernels, feature weights, training, inferred metadata or nonlinear phase interactions require a new protocol and independently frozen experiment. Quality evaluation must hold semantic information, corpus/query partition, candidate budget and scorer-only labels constant. Existing original Stage C raw-text embedding results cannot establish a representation effect by comparison with enriched phase features.
