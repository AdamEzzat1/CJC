# Locke Testing and Verification

## Test bucket layout

```text
crates/cjc-locke/src/*.rs        (unit tests inline via #[cfg(test)])
tests/locke/
├── mod.rs
├── validation_tests.rs          integration: validators over DataFrames
├── drift_tests.rs               integration: drift module
├── lineage_tests.rs             integration: lineage builder + audit
├── belief_tests.rs              integration: belief score + report
├── causal_tests.rs              integration: causal guardrails
├── determinism_tests.rs         GATE: bit-identical across runs
├── locke_proptest.rs            proptest property tests
└── locke_fuzz.rs                Bolero structural fuzz
```

CLI tests live in `crates/cjc-cli/src/commands/locke.rs` (unit tests for argument parsing and JSON escaping).

## Headline counts

| Bucket                       | Tests | Status |
|------------------------------|-------|--------|
| `cargo test -p cjc-locke --lib` | 65    | all passing |
| `cargo test --test locke`       | 44    | all passing (includes 6 proptest + 5 bolero) |
| `cargo test -p cjc-cli locke`   | 6     | all passing |
| **Total Locke-owned**           | **115** | **0 failures** |

## What the proptest properties guarantee

| Property                                                | Cases |
|--------------------------------------------------------|-------|
| validate is bit-identical across repeated runs          | 256   |
| missingness count never exceeds row count               | 256   |
| belief overall score ∈ [0, 1]                           | 256   |
| more missingness never improves missingness score       | 256   |
| drift comparison is deterministic                       | 256   |
| sample-score curve stays in [0, 1]                      | 256   |

## What Bolero fuzz exercises

- Arbitrary `Vec<f64>` → `validate(...)` does not panic
- Arbitrary `Vec<i64>` → `validate(...)` does not panic
- Arbitrary `(Vec<f64>, Vec<f64>)` → `compare(...)` does not panic
- Arbitrary UTF-8 byte sequences as `label_text` → `causal_guardrail` does not panic
- Arbitrary lineage operations → graph remains acyclic

Bolero auto-falls-back to proptest on Windows/macOS; on Linux CI it can be promoted to libfuzzer-driven coverage-guided fuzz via `cargo bolero test`.

## Regression gates

The Locke regression command:

```bash
cargo test -p cjc-locke --lib
cargo test --test locke
cargo test -p cjc-cli locke
```

CI should run all three before merging any change to `crates/cjc-locke/`, `crates/cjc-cli/src/commands/locke.rs`, or `tests/locke/`.

## Determinism contract — how it's enforced

| Mechanism | Enforced by |
|-----------|-------------|
| `BTreeMap`/`BTreeSet` everywhere | code review + clippy lint candidate |
| Kahan summation in all float reductions | `cjc_repro::KahanAccumulatorF64` only |
| No FMA / `mul_add` | not used anywhere in `cjc-locke` |
| No HashMap iteration | grep gates in CI (suggested) |
| Bit-identical reports across runs | `tests/locke/determinism_tests.rs` (5 tests) |
| Repeated-build text emit stability | `lineage_tests::graph_round_trip_bytes_are_stable` |

## CLI snapshot testing (suggested for v0.2)

The text emit from `cjcl locke validate` is already byte-stable across runs. A future v0.2 task is to add a snapshot-test crate (e.g., `insta`) over the canonical CLI output so any cosmetic change to the format is reviewed deliberately. For v0, the determinism tests in Rust cover the same invariant at a lower level.

## Known testing gaps

- No multi-process determinism test yet — Locke is verified deterministic *within* a process. Across processes, the smoke test in this conversation (`diff -s a.txt b.txt`) demonstrates the property, but a permanent test would require driving the binary from a build-script.
- No coverage-guided fuzz on Linux CI yet — Bolero supports it but the workspace hasn't enabled libfuzzer for Locke targets.
- `CategoricalAdaptive` is conservatively treated as opaque; a deeper-dict integration test is deferred.
