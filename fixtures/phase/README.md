# Authored phase-control inputs

`control.json` is a new synthetic development **feature-input** fixture: ten candidate memories and four semantic queries. It contains no gold relevance labels, inference results, trained vectors or claimed model performance. It is separate from both the byte-preserved original starter pack and the authored Stage A/B/C regression schema.

Its only purpose is to exercise the frozen optional phase/real expected-equivalence experiment: direction, polarity, scoped belief/intention, named fictional clocks, nested typed objects, unknowns and full candidate budgets. IDs are reporting/tie-breaking fields, never semantic features. No original gold was translated into this file's features.

```sh
cargo run --locked -- phase-control fixtures/phase/control.json /tmp/phase-control.json
# Explicit unrun negative observation with the unconfigured example config:
cargo run --locked -- phase-compare config.example.json fixtures/phase/control.json /tmp/phase-compare.json
```

The first command performs arithmetic only. The second writes complete evidence and exits nonzero because its Pumas embedding model is unconfigured; it never substitutes numeric vectors. Real model comparison requires an exact qualified Pumas embedding identity and public capability/fence support. See [protocol and limits](../../docs/phase-control.md).
