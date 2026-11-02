# Phased Memory

[MrScripty/phased-memory](https://github.com/MrScripty/phased-memory) is a standalone local experiment.

Standalone Rust experiment: SQLite stores immutable source text, normalized directed claims, embeddings and relationship history. A small loopback web UI compares exact cosine retrieval, explicit structured filtering and optional Pumas relationship reranking. No vector database or Pumas/Eidetic modifications.

**Real-model qualification is unrun.** Configure served model aliases, a compatible Pumas v0.8 endpoint and an explicitly qualified decision adapter before evaluating model quality. Authored development fixtures and synthetic vectors demonstrate plumbing only. The original starter archive is byte-verified and its six files are preserved in `fixtures/original/`; [original integration](docs/original-fixtures.md) documents its JSONL interfaces and unchanged scorer.

## Build and launch

Clone this repository and use Rust 1.99.1 with Cargo, rustfmt and clippy (`rust-toolchain.toml`). Cargo fetches application dependencies; the app never downloads model weights or installs providers. Bundled SQLite builds with a C compiler.

```sh
git clone https://github.com/MrScripty/phased-memory.git
cd phased-memory
cargo build --locked
cargo run --locked -- serve /tmp/phased-memory.sqlite config.example.json 8789
# Open http://127.0.0.1:8789
```

The example config deliberately has unconfigured model roles. Human claim storage and explicitly synthetic queries work offline; model requests fail visibly. Use the provided claim JSON form or paste `fixtures/authored/corpus.json` into it, save without embedding, select “Use explicitly synthetic query vector”, and compare with appropriate structured filters. Synthetic query space is `synthetic-smoke-v1`; all ten authored corpus vectors are intentionally identical. Result ties are stable by claim ID and carry no semantic quality claim.

For real inference, copy [`config.models.example.json`](config.models.example.json) to the ignored `config.local.json` and name each role's **served alias** exactly as `GET /v1/served-models` (or `probe`) lists it, not a filename or inferred nickname:

```json
{
  "endpoint": "http://127.0.0.1:8080",
  "extraction": {"model": "SERVED_CHAT_ALIAS"},
  "embedding": {
    "model": "embeddinggemma-2",
    "query_prefix": "task: search result | query: ",
    "document_prefix": "title: none | text: "
  },
  "decision": {"model": "kev-4b"},
  "decision_adapter": null
}
```

Each role is `{"model", "profile"?, "capability"?}`; the embedding role also accepts `query_prefix` and `document_prefix`. The profile is discovered from `/v1/served-models` and sent only when Pumas reports `profile_required` (the alias is served by more than one profile) or when the operator pins `profile`. An alias served by several profiles without a pin fails visibly and lists them; nothing is guessed. Without a `capability` pin, a role selects by name from what the served model advertises as available: extraction prefers `chat_generation` then `text_generation`, embedding uses `text_embedding`, and decision follows the adapter rules below. Legacy configs with explicit `profile`/`capability` remain valid pins.

**Embedding prefixes.** Some embedding models expect task prefixes in the text. For EmbeddingGemma 2 (`unsloth/embeddinggemma-2-GGUF`), queries use `task: search result | query: ` and stored documents use `title: none | text: `. Prefixes are explicit configuration (default: none). Every stored claim, original-pack memory and phase corpus item is embedded with `document_prefix`, and every search, Stage C, original-retrieval and phase query with `query_prefix`, so both sides land in one space. The prefix pair is part of the embedding-space identity: changing either requires deliberate re-embedding.

**Decision adapters.** `decision_adapter` is an explicit **operator qualification**; keep it null until confirmed. The decision path is chosen from what the served decision model advertises, and the qualification must name that same path:

- `decision_native_v1`: the model advertises an available Pumas `decision` capability (llama.cpp native decision models such as Kev, `ggml-org/Kev-4B-GGUF`). The directed claim pair is sent as the decision `state` with three `choice` questions over this app's labels (`relationship`, `temporal_order`, `update_type`). The chosen options become the labels; the relationship answer's confidence and per-question option probabilities are retained as evidence.
- `decision_json_v1`: a chat/text generation model, prompted for labels JSON by this consumer. Pumas' ordinary generation capability does not establish that a model can implement relationship decisions.

A mismatch (for example `decision_json_v1` with a model that advertises `decision`, or `decision_native_v1` with a chat model) fails before any operation; there is no fallback between adapters. Native probabilities are temperature-scaled by the model and **not calibrated** for this data; they are never presented as calibrated probability.

```sh
cargo run --locked -- probe config.local.json
# Advertised capability is not model qualification.
```

“Extract, embed and save through Pumas” performs the server-owned pipeline. Extracted local claim IDs are namespaced per source, preserving `model_local_id`. Preview alone does not store. Browser-submitted normalized claims are labeled human; browser-submitted vectors must be explicitly synthetic. Failed extraction/embedding leaves no partially stored source.

## Semantics and persistence

A claim contains `id`, `normalized`, `triple` (`subject`, `predicate`, `object`, `polarity`), `entities`, `source`, `perspective`, `continuity`, nullable `story_time` (`timeline`, integer `position`), and `qualifiers`. Optional `state_key` explicitly groups mutually exclusive state values; absent means no guessed exclusivity. Optional `model_local_id` retains extraction-local identity. Positions are author-defined within one named timeline. Unknown chronology stays null. `sources.ingested_at` records ingestion separately and never determines story ordering.

Qualifiers and perspective retain beliefs, lies, intentions and direction. The system does not equate an utterance with world truth. Directed relationships store independent `relationship`, `temporal_order` and `update_type` labels, confidence and provenance; confidence is never interpreted as calibrated probability. New claims and relations append history; duplicate IDs and unsupported schema versions fail. SQLite source/claim/embedding insertion is one transaction; relationships use foreign keys. Version 1 is the only persisted schema; future versions are rejected without resetting data. Back up the database before any future migration. There is no destructive reset route.

Structured filters: `subject`, `predicate`, `object`, `polarity`, `entity`, `qualifier`, `perspective`, `continuity`, `relationship`, `timeline`, `at_or_before`, `latest`, `chronological`. Text matching is exact and case-sensitive; normalization is author/model responsibility. A relationship filter selects endpoints of stored matching annotations; it does not infer a relation to the query. A cutoff requires a timeline. Latest retains the maximum known position per directed triple, perspective, continuity and timeline, or per explicitly supplied `state_key` instead of object. Latest is computed across authoritative claims before filtering state values/polarity, including later claims without vectors. Independent objects remain distinct; equal-time claims remain tied. Unknown-time claims cannot qualify as latest. `chronological` requires a timeline and orders known story positions ascending, breaking ties by claim ID without inferring real order within a tie.

Embeddings rank only within the same exact space and dimension. Real spaces include endpoint, served alias, resolved profile, both runtime generations and the configured query/document text prefixes; restart, replacement or a prefix change requires deliberate re-embedding into the new space. Exact cosine uses f64 accumulation over finite nonzero f32 vectors. Results use deterministic ID tie breaking. Both vector and structured columns show at most five hits. Reranking classifies **all structured-eligible candidates before taking five**, then orders supports, compatible, insufficient information, contradicts, cosine and ID. The original-pack reranker separately receives all ten memories even when query-plan filters exclude some. An empty set is `unrun`, never a decision completion. Failure leaves that column unavailable.

## Evaluation

Original recovered JSONL fixtures use [their dedicated Stage A/B/C commands and unchanged Python scorer](docs/original-fixtures.md). The following commands use the separate authored regression schema and Rust scorer.

See [evaluation protocol](docs/evaluation.md). All commands read JSON arrays and write new output paths without overwriting existing evidence:

```sh
# Human/offline storage. Optional CONFIG creates real vectors only when input embeddings is empty.
cargo run --locked -- ingest /tmp/memory.sqlite fixtures/authored/corpus.json
cargo run --locked -- stage-a config.local.json fixtures/authored/pairs.json /tmp/pairs-run.json
cargo run --locked -- score-pairs fixtures/authored/pairs.json /tmp/pairs-run.json
# Predefined naive baseline: compatible/unknown/none, reads IDs only, never gold.
python3 scripts/naive_baseline.py fixtures/authored/pairs.json /tmp/naive-pairs.json
cargo run --locked -- score-pairs fixtures/authored/pairs.json /tmp/naive-pairs.json
cargo run --locked -- stage-b config.local.json fixtures/authored/raw-pairs.json /tmp/extraction-run.json
cargo run --locked -- score-extraction fixtures/authored/raw-pairs.json /tmp/extraction-run.json
cargo run --locked -- stage-c /tmp/memory.sqlite config.local.json fixtures/authored/queries.json /tmp/structured-run.json structured
cargo run --locked -- score-retrieval /tmp/memory.sqlite fixtures/authored/queries.json /tmp/structured-run.json
```

With `decision_native_v1`, Stage A/B predictions retain the native relationship choice probabilities (when they form a complete normalized vector) as `native_probabilities`, so `score-pairs` can report Brier; these are uncalibrated model outputs and Brier measures exactly that. Prompted adapters never produce probabilities. Run Stage C separately for `vector`, `structured`, `reranked`. For live retrieval, import a claims-only corpus with the optional config argument; the synthetic corpus intentionally has a different space. Missing models/outputs remain in denominators. Runners retain per-case errors and exit nonzero for incomplete runs. Offline scoring remains runnable. No command silently switches to synthetic inference.

## Optional phase control

The separate [phase experiment](docs/phase-control.md) now implements the frozen score-equivalence control and same-information Pumas embedding arm:

```sh
cargo run --locked -- phase-control fixtures/phase/control.json /tmp/phase-control.json
cargo run --locked -- phase-compare config.example.json fixtures/phase/control.json /tmp/phase-compare.json
```

The authored fixture has ten semantic candidates and four queries, with no quality gold or precomputed vectors. All arms use the complete candidate set and the same canonical semantic fields. Numeric phase/real score equivalence is checked to 1e-12; raw near-tie rankings can differ and are reported honestly. The example Pumas config records the embedding arm explicitly unrun, retains every row and exits nonzero. No numeric/synthetic fallback is used for real inference, and no phase advantage is claimed.

## Boundary and lifecycle

Only numeric loopback HTTP base URLs are accepted. The UI binds `127.0.0.1`, enforces exact Host/Origin and fetch-site metadata, a per-launch request header, 1 MiB JSON body limit, same-origin assets and restrictive CSP. SQLite API work uses an owned synchronous blocking region on the multithread runtime; lock contention fails after a two-second busy budget. One API operation is admitted; overlapping operations receive 429 without a work queue. This is a local tool for one operator, not isolation from hostile same-user processes. Browser state is in memory; SQLite stores explicitly submitted data. Do not submit credentials in prose. No bearer secret is collected or forwarded by this prototype.

All actual inference goes through public `/.well-known/pumas`, `/v1/served-models`, `/v1/model-operations` (Pumas model-operations contract 1, v0.8 typed slice). Both generation headers are pinned to the first compatible runtime observation and sent to discovery/operations. Generation changes fail and require an explicit consumer restart. No redirects, protocol retries, alternate routes, direct inference, serving lifecycle changes or downloads. Observation/embedding requests have 30-second budgets and connecting has 10 seconds. Generation has **no total/read/idle timeout**. Ctrl-C closes admission and cancels active owned HTTP futures before server drain; dropping transport is not proof that GPU/provider work stopped. Browser loss does not provide provider-cessation evidence; never replay an uncertain operation. Transport loss, malformed responses and correlation mismatches preserve an unknown outcome.

Requests are closed and exact. Responses are read tolerantly, as the Pumas compatibility rules require: unknown response fields, capability descriptors, availability states and reasons, formats, option names, error codes and extra list entries are ignored, and capabilities are selected by name, never by position or count. Only the fields this consumer uses are validated. A typed error is examined before correlation. `outcome: "not_admitted"` (no provider effect), including a `null` `request_id` for an unreadable body and empty-body 400/412/503 admission-fence refusals, is reported as `not_admitted`, not as an uncertain outcome; `outcome: "unknown"` or any unrecognized outcome is reported as `unknown` and never replayed. Pumas' `error.message` is always included in the surfaced error.

## Verification and remaining gates

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
python3 scripts/verify_original_pack.py
# Optional existing Python Playwright + installed Chromium only; installs nothing:
python3 scripts/browser_smoke.py
```

CI runs deterministic formatting/lint/tests with read-only permissions and pinned Rust. Test-only HTTP simulators prove consumer sequencing and rejection behavior, not runtime compatibility or model quality. Browser smoke uses labeled synthetic fixtures through the real UI. See [verification evidence](docs/verification.md), [source and standards decisions](docs/sources-and-standards.md), and the [source-content mapping](docs/source-map.md). Original fixture identity is verified; live model qualification and learned phase-comparison quality remain blocked/unrun; the deterministic phase control is implemented.
