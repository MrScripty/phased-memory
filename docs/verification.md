# Verification scope

The deterministic suite contains **41 tests**: 15 domain/storage/scorer tests, 11 original-pack integration tests, 11 phase/math/boundary tests and four public Pumas HTTP tests. The supported verification environment for this prototype is Linux x86_64 with Rust 1.99.1.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --locked
python3 scripts/verify_original_pack.py
# Optional installed Python Playwright and Chromium; installs nothing:
python3 scripts/browser_smoke.py
```

CI defines the deterministic format/lint/test and 150-check original integrity/scorer gates with read-only repository permissions. Browser verification is separate and uses a temporary database plus explicitly synthetic vectors. Its nine checks cover loading/unrun status, transactional human-claim ingestion, latest/historical/directed retrieval, retained relation annotations, unavailable empty-set reranking, foreign-origin rejection and browser exceptions.

## Claims covered

Storage/math checks cover finite nonzero vectors, exact cosine, rollback, raw text retention, future-schema preservation, explicit story-time history, directed/perspective scopes, latest-state filtering, explicit exclusive state keys and namespaced extraction IDs. Scorer tests retain missing-result denominators and reject malformed/duplicate/unknown IDs and inconsistent extraction successes.

Test-only loopback Pumas servers exercise actual HTTP discovery, exact identities, fences, stale-generation rejection, unavailable capability admission, correlation/result decoding, unknown outcomes and no retries. They are never exposed as an application inference fallback and never qualify real models.

Original integration checks preserve every original row, selected primary and typed/null fields; isolate gold/diagnostics/raw scenes from decision prompts; retain model-selected extraction primaries without reference repair; expose all ten memories to original reranking; keep chronology separate; and roundtrip saved failures through the unchanged scorer. `verify_original_pack.py` performs 150 identity/arithmetic checks, or 160 when given the exact reference archive. See [archive identity](original-fixtures.md).

Phase checks cover the explicit polarity channels, ordered tuples, JSON types, named/unknown clocks, query-inclusive scale, integer bounds, canonical sets/keys, full budgets and strict fields. They check the exact canonical payloads sent through the test-only HTTP boundary and ensure unavailable models do not trigger retries, partial candidate scoring or synthetic fallback. A near-tie regression preserves raw ranking divergence while score errors satisfy the tolerance. See [phase evidence limits](phase-control.md).

## Unqualified claims

Real extraction, embedding, relationship decisions and retrieval quality remain **unrun**. A compatible endpoint, exact model/profile identities and operator qualification are required. An advertised capability alone is insufficient. Native probability output is unavailable in the implemented inference runners. No phase advantage is claimed by an arithmetic identity or synthetic development set.

Functional acceptance can include an honest negative result: runners retain all requested IDs and errors rather than pretending unavailable inference succeeded. The retained phase command can pass its numeric control while returning an explicitly unrun learned arm and a nonzero exit code. No model weights are downloaded by the application or deterministic checks.
