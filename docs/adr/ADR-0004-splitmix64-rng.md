# ADR-0004: SplitMix64 as the Canonical CJC RNG

**Status:** Accepted
**Date:** 2024-01-15
**Deciders:** Applied Scientist, Technical Lead
**Supersedes:** none

## Context

CJC requires a deterministic random number generator for:
- `Tensor::randn` (seeded Gaussian noise for ML tests)
- Shuffle operations
- Reproducible test fixtures

Requirements:
- **Deterministic**: Same seed → same sequence, always.
- **Cross-platform**: Bit-identical output on x86, ARM, WASM.
- **Fast**: Suitable for generating millions of tensor elements.
- **Zero dependencies**: CJC has a zero-external-dependency constraint.

## Decision

Use **SplitMix64** as the canonical CJC RNG, implemented in `cjc-repro/src/lib.rs` as `Rng::seeded(seed: u64)`.

SplitMix64 is a bijective function: `state -> (state', value)`. It passes BigCrush statistical tests and has no internal state beyond a single `u64`.

Box-Muller transform converts uniform `u64` output to standard normal samples for `Tensor::randn`.

## Rationale

- **Simplicity**: The entire implementation is 8 lines of Rust.
- **Determinism proof**: Bijective function on `u64` → identical output given identical `state`. Platform-independent because it uses only `u64` arithmetic (no float rounding in state update).
- **Speed**: ~1 ns/sample on modern hardware. Adequate for 100k-element tensor generation in test suites.
- **Precedent**: Used in Java's `java.util.SplittableRandom`, Rust's `splitmix64` crate ecosystem.

## Consequences

**Positive:**
- `test_tensor_randn_deterministic` verifies bit-identical output from the same seed.
- No external dependency required.

**Known limitations:**
- SplitMix64 has a period of 2^64 — sufficient for ML testing but not for cryptographic applications (CJC makes no cryptographic guarantees).
- Box-Muller introduces 2 float operations per normal sample; Ziggurat would be faster but more complex.

## Implementation Notes

- Crates affected: `cjc-repro`, `cjc-runtime`
- Files: `crates/cjc-repro/src/lib.rs` (`Rng`, `SplitMix64`), `crates/cjc-runtime/src/lib.rs` (`Tensor::randn`)
- Regression gate: `cargo test --workspace` must pass with 0 failures
