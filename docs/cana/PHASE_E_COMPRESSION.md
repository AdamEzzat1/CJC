# Phase E — Compression Prototypes: Measured Bytes at Bounded Error

**Date:** 2026-06-12 (follows Phase D on `claude/stupefied-liskov-83b258`)
**Crate:** `bench/cana_compress_probe` (lib + bin, publish = false)
**Spec source:** `docs/cana/NEXT_ARC_RESEARCH.md` §2 (the memory track's
"three prototype-ready wins"), roadmap §6 row E.
**Exit criterion:** before/after bytes at bounded reconstruction error.
**Artifacts:** `bench_results/cana_compress_probe/REPORT.md` (regenerable
by `cargo run --release -p cana-compress-probe`).

## 0. Scope discipline

All prototype code (the trace serializations, the rank search, the
artifact walkers) lives in the bench crate. `crates/` is untouched:
transforms graduate to `cjc-cana-compress` only with measured numbers
behind them — which this phase now provides. The codecs themselves are
the committed Phase-6 ones (`lossless_compress_bytes` byte-RLE,
`compress_motif_dictionary`, `compress_low_rank`).

## 1. Prototype 1 — trace-stream compression: 35–43×, lossless, proven

`MirTraceEvent` streams from instrumented runs of six hash-pinned
Phase-D subjects (seed 42), serialized two ways and fed to both
lossless codecs. Every measurement decodes back and compares
**bit-exactly** (NaN payloads included) — reconstruction error is zero
by proof. Two representations:

- **canonical** — row-major fixed-width records (45 B/event);
- **delta-columnar** — whole columns, integers as wrapping deltas,
  floats as XOR-of-consecutive-bit-patterns. Loop-dominated traces
  become long zero runs. This transform IS the research doc's
  "trace-stream RLE" idea made concrete.

| subject | events | canonical: RLE / motif | delta: RLE / motif |
|---|---|---|---|
| mem_grad_a1 | 258 | 2.24× / 4.63× | 5.20× / **35.11×** |
| mem_grad_a5 | 65,538 | 2.05× / 5.29× | 5.32× / **43.28×** |
| fp_hot | 3,402 | 2.16× / 8.93× | 8.24× / **42.28×** |
| grad_f90_d1_n1024 | 1,026 | 1.68× / 8.86× | 9.67× / **40.09×** |
| tensor_ew_n32_i200 | 2,251 | 2.12× / 8.92× | 9.73× / **41.40×** |
| holdout_alloc_pulse | 5,002 | 2.06× / 5.25× | 5.32× / **42.73×** |

**Verdict: the research doc's "plausible 5–28×" band is EXCEEDED** —
delta-columnar + motif lands at 35–43× on every stream, uniformly
across program families (alloc churn, scalar FP, tensor). The
decomposition matters: raw bytes through RLE alone is only ~2×; the
representation transform contributes more than the codec choice.
mem_grad_a5's full 65k-event stream (2.9 MB serialized) compresses to
67 KB — trace archiving for the training corpus is now obviously
practical.

## 2. Prototype 2 — checkpoint low-rank: 1.38×, honest miss of the 2–3× band

The REAL chess-RL v2.1 checkpoint (`checkpoint_ep60.bin`, 1,101,744
bytes, 31 tensors — main-worktree artifact, not in git). Each 2-D
tensor is compressed to the SMALLEST rank within a 5% relative
Frobenius tolerance (binary search; truncation error is monotone in
rank); matrices that can't beat raw storage are kept raw; non-2-D
tensors (biases etc.) pass through raw and are counted.

- **Format correction (THE RULE, again):** the research doc said
  "checkpoint compression (cjc-snap)". The artifact's magic is `CJCT`
  — it is a `cjc_runtime::tensor_snap` flat tensor list, not a
  `cjc-snap` value file. The probe handles both.
- Result: **1,100,976 → 800,072 tensor-payload bytes (1.38×)** — below
  the doc's "plausible 2–3×". The per-matrix texture explains it: the
  checkpoint holds three 774×48/48×48/48×64 matrix groups, and the
  first group (weight matrices near their random init — v2.1 trained
  only 60 episodes with near-zero reward signal) is effectively
  FULL-RANK, so the search correctly keeps them raw. The later groups
  (optimizer-state-like) compress to ranks 13–33 (best single matrix:
  774×48 at rank 15 = 3.0×). Random-init-like weights are the
  worst case for low-rank by construction; a converged model would
  compress better, but that claim awaits a converged artifact.
- Bound held everywhere: every compressed matrix reports
  rel-Frobenius ≤ 0.0500. Diagnostic checkpoints only — never
  training-resumption paths.

## 3. Prototype 3 — disk artifacts: motif 8.3×, and byte-RLE actively HURTS

Committed files, byte-level, roundtrip-verified:

| artifact | RLE | motif |
|---|---|---|
| `profiles.cpdb` (1,088,931 B) | **0.96× (EXPANDS)** | **8.34×** |
| `phases.csv` (21,341 B) | 0.97× (expands) | 2.54× |

**Finding:** byte-RLE is the wrong codec for structured disk artifacts
— run lengths are short, so the escape overhead net-expands the file.
The motif (back-reference) codec is the right tool: 8.34× on the
corpus puts the doc's 2–5× band at the optimistic end and beyond.
Pairing rule, now measured: **RLE earns its keep only AFTER the
delta/XOR transform manufactures runs; motif is the general-purpose
disk codec.**

## 4. Verification (test-discipline contract)

`bench/cana_compress_probe/tests/test_compress_probe.rs` — 14 tests:
wiring (real instrumented stream end-to-end; rank search on an exact
rank-2 matrix; incompressible-noise behavior; tensor walker on nested
values; committed-corpus measurement), unit (empty/single-event
roundtrips, NaN-payload + extreme-integer bit-exactness, decoder
rejection of malformed input incl. reserved flag bits, bookkeeping),
proptest (both encodings roundtrip arbitrary event vectors bit-exactly;
codecs roundtrip arbitrary bytes), bolero (both decoders never panic on
arbitrary bytes).

## 5. What graduates, what doesn't (recommended next)

- **Graduate:** the delta-columnar transform → `cjc-cana-compress` as a
  first-class `CompressionKind` (lossless, hash-embedded like its
  siblings), so the ablation/training pipeline can archive full
  instrumented traces at ~40×. The corpus regen currently keeps no
  traces at all — at 43× it could.
- **Hold:** checkpoint low-rank stays a probe until a converged model
  exists to measure (the honest 1.38× on a near-init checkpoint neither
  confirms nor refutes the 2–3× band for real models).
- **Adopt operationally:** motif-compress `profiles.cpdb` (8.3×) if/when
  corpus size becomes a friction point; it is 1 MB today, so this is
  recorded, not urgent.
