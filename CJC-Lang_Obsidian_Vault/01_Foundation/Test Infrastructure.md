---
title: Test Infrastructure
tags: [foundation, tests]
status: Implemented; counts need verification
---

# Test Infrastructure

How CJC-Lang's tests are organized, what they enforce, and where they live.

## Layout

```
tests/
  milestone_2_4/          — Stage 2.4 acceptance: parity, NoGC, optimizer, shape
    parity/               — G-8 and G-10 parity gates
    nogc/                 — @nogc verifier proof programs
    optimizer/            — MIR optimizer correctness
  chess_rl_project/       — Core chess RL suite (~49 tests per CLAUDE.md)
  chess_rl_advanced/      — Advanced chess RL (66 tests)
  chess_rl_hardening/     — Chess RL robustness
  chess_rl_playability/   — End-to-end gameplay
  hardening_tests/        — Beta-hardening phase tests (H1..H4)
  audit_tests/            — Audit tests (COW, parallelism, ML types, SSA, shape, numeric, collections)
  prop_tests/             — Property-based tests (proptest) — S3-P0-06
  fixtures/               — End-to-end golden-file runner — S3-P0-07
  bench_50q.rs            — 50-qubit quantum benchmark
  ...                     — ~55 test binaries total
```

## Counts (verified 2026-04-09)

CJC-Lang has **two different test-count numbers** and both are meaningful. Understanding which one applies depends on what question you're asking.

### Raw markers in the source tree (6,715+)

| Source | Count |
|---|---|
| Unit tests — `#[test]` markers in `crates/*/src/` | **1,913** |
| Integration tests — `#[test]` markers in `tests/` | **4,802** |
| **Total `#[test]` markers** | **6,715** |
| `proptest!` macros (each expands to many property-test cases at runtime) | **22** |
| `bolero::check` fuzz targets | **38** |

This is the "how many tests has the team written" number. A `proptest!` invocation typically generates 256–1000 random cases per run, so the effective test surface is far larger than the marker count.

### Tests actually executed in the default build (5,353)

Authoritative count via `cargo test --workspace --release` (118 test binaries, 0 failures):

| Metric | Value |
|---|---|
| **Total tests passing** | **5,353** |
| Test binaries | **118** |
| Failures | **0** |
| Ignored | ~20 (intentionally skipped fixtures) |
| Chess RL suites total | **319** (69 project + 84 advanced + 104 hardening + 62 playability) |
| Dedicated parity files | **13** files, **179** parity tests (see [[Parity Coverage Matrix]]) |
| Stage 2.4 baseline | **535** (must only increase per roadmap) |

The ~1,360-test gap between 6,715 markers and 5,353 executed tests is **feature-gated tests** that don't compile in the vanilla workspace build: `cjc-runtime/parallel` (rayon paths from [[ADR-0011 Parallel Matmul]]), experimental quantum backends, and some optimizer-audit tests that require explicit feature flags.

### Historical claims vs reality

| Source | Claim | Reality |
|---|---|---|
| `README.md` | 3,700+ | **stale — actual 5,353 default / 6,715 markers** |
| `CLAUDE.md` memory (2026-03-21) | 5,320 | close to default build — 33 tests added since |
| Previous [[Chess RL Demo]] note | 216 | **stale — actual 319** |

## Regression gates

All gates that must pass before any roadmap task is marked complete (from `docs/spec/stage3_roadmap.md`):

```bash
cargo test --workspace                     # full gate
cargo test milestone_2_4 -- parity         # [[Parity Gates]]
cargo test milestone_2_4 -- nogc           # [[NoGC Verifier]]
cargo test milestone_2_4 -- optimizer      # [[MIR Optimizer]]
cargo test fixtures                         # after S3-P0-07
cargo test prop_tests                       # after S3-P0-06
cargo test --workspace --features cjc-runtime/parallel  # after S3-P1-02
```

## Test categories

| Category | Purpose | Example |
|---|---|---|
| **Unit** | Feature-level behavior | `test_lexer_tokenize_float` |
| **Integration** | Cross-feature interaction | `test_closure_in_match_arm` |
| **Compile-fail** | Expected error codes | `test_e0150_immutable_assign` |
| **Determinism** | Same seed → same bits | `test_chess_rl_determinism` |
| **Parity** | AST-eval ≡ MIR-exec | G-8 and G-10 fixtures |
| **Property** | Parser/type-checker never panic | `prop_parser_no_panic` |
| **Audit** | Specific invariants (COW, SSA, …) | `test_audit_cow_array` |
| **Fixture** | Golden `.cjcl`/`.stdout` pairs | `tests/fixtures/basic/` |

## What "parity gate" means

The gate is a test binary that runs the same program through both [[cjc-eval]] and [[cjc-mir-exec]] and asserts byte-identical output. If the two backends diverge, the gate fails, and the offending change cannot be merged. See [[Parity Gates]].

## Related

- [[Parity Gates]]
- [[NoGC Verifier]]
- [[MIR Optimizer]]
- [[Roadmap]]
- [[Open Questions]]
