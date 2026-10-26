# Sources and design guidance

Phased Memory is an independently implemented bounded consumer of the public Pumas HTTP API. Pumas owns runtime admission, provider execution and model lifecycle; this repository does not import Pumas runtime code or modify its projects.

## Public contract authority

The consumer was implemented against Pumas Library revision [`ad31e391dcd94c204b3fcd748cc98d27b3281e65`](https://github.com/MrScripty/Pumas-Library/tree/ad31e391dcd94c204b3fcd748cc98d27b3281e65):

- [Model operations v1](https://github.com/MrScripty/Pumas-Library/blob/ad31e391dcd94c204b3fcd748cc98d27b3281e65/docs/contracts/model-operations-v1.md).
- [Operation DTOs](https://github.com/MrScripty/Pumas-Library/blob/ad31e391dcd94c204b3fcd748cc98d27b3281e65/rust/crates/pumas-rpc/src/handlers/model_operations/types.rs).
- [HTTP discovery and endpoint rules](https://github.com/MrScripty/Pumas-Library/blob/ad31e391dcd94c204b3fcd748cc98d27b3281e65/rust/crates/pumas-core/src/discovery/http.rs).
- [HTTP generation admission](https://github.com/MrScripty/Pumas-Library/blob/ad31e391dcd94c204b3fcd748cc98d27b3281e65/rust/crates/pumas-rpc/src/discovery.rs).

[Pumas Playground](https://github.com/MrScripty/Pumas-Playground) provides a related public consumer example. No code from it is bundled here. Public source contracts establish route, DTO and fence meaning; they do not establish availability of an installed runtime or quality of a selected model. Model names and plugin manifests cannot qualify behavior. `decision_json_v1` and `original_json_v1` are consumer prompting protocols requiring explicit runtime-operator qualification, not new native Pumas capabilities.

## Responsibility boundaries

- `domain.rs` owns typed claim and relationship semantics, provenance and vector validity.
- `store.rs` owns transactional append-only SQLite history, schema admission and exact cosine retrieval.
- `pumas.rs` owns public HTTP transport, capability inspection and exact model/profile/generation fences.
- `eval.rs` owns the authored regression schema and Rust scoring rules.
- `original.rs` owns lossless original JSONL projections and compatibility with its preserved Python scorer.
- `phase.rs` owns the optional feature projection, frozen vocabulary/scale and expected-equivalence control. The learned arm uses the unchanged Pumas transport.
- `main.rs` owns CLI selection, local listener lifecycle and admission. The browser uses native controls and renders returned text as text.

The original starter pack retains its own labels, manifest and scorer unchanged. The authored regressions and phase feature fixture are separate development data. See [original interfaces](original-fixtures.md) and [phase protocol](phase-control.md). No scorer arithmetic or synthetic control is a model-quality oracle.

## Development guidance

[Core Standards](https://github.com/MrScripty/Coding-Standards/blob/main/CORE-STANDARDS.md) and [Standards Router](https://github.com/MrScripty/Coding-Standards/blob/main/STANDARDS-ROUTER.md) informed the implementation, alongside applicable implementation, verification/oracles/GUI, planning, build, tooling, documentation; Rust API/dependencies/tooling/security; persistence, IPC and frontend; architecture, contracts, dependency, resilience, concurrency, diagnostics and accessibility guidance. These references track a mutable branch and are not a blanket compliance certification.

The resulting decisions include narrow responsibility ownership, boundary validation, append-only transactions, future-schema rejection, exact generation fencing, no replay after uncertain inference, same-origin browser controls and separate deterministic/browser/live evidence. No external accessibility certification, dependency security audit or multi-platform qualification is claimed.

## Dependencies

Application dependencies are lockfile-resolved: serde/serde_json for JSON and typed data (MIT/Apache-2.0), reqwest for loopback HTTP/URL handling (MIT/Apache-2.0, no TLS feature), axum/tokio for HTTP and owned asynchronous work (MIT), rusqlite for transactions (MIT, bundled SQLite public domain), uuid for operation/source identities (MIT/Apache-2.0), and anyhow for application errors (MIT/Apache-2.0). tempfile is used for test isolation (MIT/Apache-2.0). Cargo.lock records exact versions and checksums.

These libraries own HTTP, URL, JSON and SQL mechanics. The application owns its declared domain-specific cosine/scoring and phase-control rules. Maintainers own dependency updates, license/security triage and regression checks; passing tests are not a dependency audit.
