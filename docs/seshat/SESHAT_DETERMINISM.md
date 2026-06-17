# Seshat — Determinism Contract

Seshat profiles *time*, which is nondeterministic, inside CJC-Lang, which
guarantees bit-identical output. This document is the gate that reconciles them.
Modeled on `docs/cana/DETERMINISM_CONTRACT.md`.

## The core decision

> **Analysis is a pure, byte-identical function of a recorded `.seshat` trace.**
> Nondeterminism is quarantined in the collection layer, which only *produces*
> traces.

A recorded trace is the unit of reproducibility. Re-recording the same program
live is *not* expected to be byte-identical — that drift is exactly what the
variance analyzer (feature 9) measures.

## What is GATED (in the content hash)

These are bit-identical for a fixed trace, asserted by the determinism gate and
the pinned golden hash (`tests/integration.rs::golden_report_hash_is_stable`):

1. **Structure** — the merged flamegraph tree, frame labels, child ordering.
2. **Count-based attribution** — sample counts, per-frame self/total, boundary
   share, contention split, pipeline per-stage counts.
3. **Byte accounting** — per-domain alloc/free/live, copy bytes, peak bytes.
4. **Causal aggregates** — crossings, wakeups, resumes, stall ticks, transfers.
5. **Recommendation codes + evidence** — derived deterministically from 1–4.

## What is ADVISORY (excluded from the hash)

Recorded for display, never gated, never used for ordering:

- `Trace::wall_ns_total` — the only wall-clock value in the model, a single
  scalar.
- Narrative/evidence prose, thermal/peak human strings (derived from gated
  numbers).
- Live run-to-run timing (the collectors' domain).

## The 10 invariants

1. **Logical clock, not wall-clock.** Event order = index in `Trace::events`.
   No analysis reads a timestamp for ordering or hashing.
2. **Counts, not durations, for reproducible attribution.** "38% of the profile"
   is a sample-count ratio.
3. **Integer math in the hashed path.** Percentages are milli-percent
   (`100% = 100_000`); no float is ever hashed. Byte/count sums are exact
   integers (no Kahan needed because there is no float summation).
4. **`BTreeMap`/`BTreeSet` everywhere.** No `HashMap` iteration-order dependence.
5. **Stable tie-breaking.** Equal counts → sort by frame label, then key. Never
   insertion or hash order.
6. **Interning-id independence.** Hashes resolve frames to `"kind:name
   (file:line)"` labels; report hashes do not depend on id *values*.
7. **FNV-1a content addressing.** Fixed little-endian byte order, length-prefixed
   strings; bit-identical on every platform. Never `DefaultHasher` (random seed).
8. **Round-trip identity.** `replay(serialize(t)).content_hash() ==
   t.content_hash()`.
9. **No panics at the boundary.** `replay` of arbitrary bytes and `dispatch_seshat`
   of arbitrary args return `Err`/`DecodeError`, never panic (bolero-enforced).
10. **Collectors are quarantined.** All wall-clock / `Instant::now` / OS-thread /
    perf-counter access lives behind `collect-live`; the default build is pure.

## How it's enforced (tests)

| Invariant | Test |
|---|---|
| Determinism (1–7) | `integration.rs::report_is_deterministic_across_runs`, `::golden_report_hash_is_stable` |
| Order-invariance (5,6) | `prop.rs::p3_flamegraph_order_invariant` |
| Partition/conservation (3) | `prop.rs::p1_sample_conservation`, `::p2_ownership_partition` |
| Copy soundness | `prop.rs::p4_copy_soundness` |
| Round-trip (8) | `prop.rs::p6_round_trip_content_hash`, `serialize.rs` unit tests |
| No-panic boundary (9) | `fuzz.rs::fuzz_decode_never_panics`, `::fuzz_tamper_never_panics`, `::fuzz_builtin_args_never_panic` |
| Bounded outputs (3) | `prop.rs::p7_bounded_outputs`, `::p8_pct_milli_in_range` |
| Executor parity | `tests/seshat/parity.rs` (AST-eval == MIR-exec) |

## Honesty clause

Report what is reproducible (structure, counts, bytes) as **gated**, and what is
not (wall-clock ms, live variance) as **advisory**. A feature that can only be
done by hashing wall-clock is a hard stop — redesign around sample counts or
logical sequence.
