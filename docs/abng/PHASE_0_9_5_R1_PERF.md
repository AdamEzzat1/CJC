# ABNG Phase 0.9.5 — Research Phase R1: post-R0 result-path speed + memory

**Date:** 2026-05-17
**Branch:** `claude/abng-phase-0-9-5` (worktree `claude/zealous-nobel-66abc9`)
**Status:** COMPLETE — profiled; Option 1 (lane-parallel x8 Kahan) shipped
as R1-1; Option 2 (triangular storage) measured and rejected. §6-§8 record
the measured results.
**Continues** [`PHASE_0_9_5_R0_PROFILE.md`](PHASE_0_9_5_R0_PROFILE.md) — R0
cut the per-row `state_hash`; R1 looks at what is now dominant.

---

## 1. Scope

After R0 (streaming `state_hash` + periodic BLR audit checkpoints) the
result path is ~2.6 ms/row and its dominant cost is the **2× O(d²)
rank-1 NIG `update`** the harness runs per training row (leaf via
`train_step`, root via `blr_update`). R1 profiles *inside* that update
and asks the same question R0 asked of `state_hash`: where does the
time actually go, and what can be done while holding determinism,
accuracy, auditability, and every other ABNG guarantee invariant?

## 2. Method

`bench/abng_result_profile` was extended (Research Phase R1 section)
with three directly-measurable update sub-costs: a raw `Vec<f64>` clone
of d² elements (`≈ precision.to_vec()`), `chol_rank1_update`, and
`cholesky_solve`. The remaining O(d²) passes — `matvec_plus_xty_kahan`
and the two `quadratic_form` calls — are `pub(crate)`/private and not
separately callable, so they are estimated by subtraction. Median over
9 trials; same machine-contention caveat as R0 (same-run ratios are the
robust read).

## 3. Measured — inside the 1.04 ms rank-1 update (d=247)

| Sub-cost | Median | Share of the update |
|---|---:|---:|
| `vec clone d×d` (≈ `precision.to_vec()`) | 14.7 µs | **1.4 %** |
| `chol_rank1_update` — O(d²) Givens | 64 µs | 6 % |
| `cholesky_solve` — O(d²) fwd+back subst (Kahan) | 160 µs | 15 % |
| **remainder** — `matvec` + 2× `quadratic_form` + Λ+φφᵀ build | **~804 µs** | **77 %** |
| **total `update` (n=1)** | **1.04 ms** | 100 % |

## 4. Finding — the bottleneck is the serial Kahan chain, not memory

The R0 design doc and the first R1 sketch both *assumed* the per-update
477 KB `precision.to_vec()` clone was a meaningful cost — the handoff
even lists "memory churn" (§R0.4 #3) as a lever. **The measurement
refutes it: the clone is 1.4 % of the update.** A `Vec<f64>` clone of
61,009 elements is a memcpy at memory-bandwidth speed (~14 µs).
Allocation and copy are not the bottleneck — the big `Tensor → Vec<f64>`
storage refactor that would eliminate the clone is **not worth its
~40-site churn for ~1.4 %**, and is dropped.

The bottleneck is **compute**: ~92 % of the update (`matvec`, two
`quadratic_form`s, `cholesky_solve`) is `f64` arithmetic threaded
through a **`KahanAccumulatorF64` serial dependency chain**. Each Kahan
`add` is `y = x − c; t = sum + y; c = (t − sum) − y; sum = t` — four
flops where every one depends on the previous add's `sum` and
compensation register `c`. A `quadratic_form` is d² such adds in one
unbroken chain; it is latency-bound on that recurrence, not on memory
traffic (confirmed: a full d² *memory* read is the ~14 µs clone, vs
~250-400 µs for a d² *Kahan-summed* pass).

**Consequence: a byte-identical speedup of the update is essentially
impossible.** Any faster summation — lane-parallel accumulation,
tree reduction, blocking — reorders the d² adds and changes the f64
low bits. The ~1.4 % clone is the *only* byte-identical lever, and it
is not worth taking.

## 5. The lever — lane-parallel Kahan (`KahanAccumulatorF64x4`)

The serial chain is broken by accumulating into **8 independent lanes**
and horizontally reducing once at the end: `KahanAccumulatorF64x8` from
`cjc-repro`. Eight independent Kahan recurrences run with
instruction-level parallelism (and opportunistic SIMD on AVX/NEON
release builds) instead of one serial chain.

**Measured** (`abng-result-profile`, within-run so contention cancels):
on a d²-element reduction `KahanAccumulatorF64x8` is **5.69×** faster
than scalar Kahan (`x4`: 4.70× — so x8 is the choice). But the
*realized* gain on the full `update` is far smaller, **~1.25×**
(~1.05 ms → ~0.85 ms): the reduction is only one part of the update —
the d² product computations feeding it, the non-Kahan `Λ+φφᵀ` build,
`cholesky_solve`, and `chol_rank1_update` are all unchanged. The first
R1-design pass projected ~1.5× on the row; the honest measured figure
is **~1.2×** (per-row ~2.6 → ~2.1 ms). Modest — but real, and free of
any accuracy or determinism cost.

This is **not** a new or exotic primitive: Phase 0.8c **Item D2b**
already replaced the batch (`n>1`) `update`'s accumulators with
`KahanAccumulatorF64x4`, and D3's `matvec_plus_xty_kahan` already has a
lane-parallel path — *gated at `d ≥ 8 && d % 4 == 0`*, which the
Diabetes-130 width d=247 (d%4=3) does not satisfy, so the n=1 hot path
never reaches it. R1 would extend lane-parallel Kahan to the n=1
`update_rank1` reductions for **arbitrary d** (process `d − d%4` lanes
×4, fold the `d%4` tail scalar).

### Determinism / accuracy / auditability

- **Determinism — preserved.** `KahanAccumulatorF64x4` has a fixed lane
  assignment and a fixed horizontal-reduce order; it is bit-stable
  across runs and platforms, uses no FMA, and is single-threaded (lanes
  are within one thread). "Same seed + same data + same build →
  byte-identical" holds. This is exactly the property D2b relies on.
- **Accuracy — preserved.** Output stays Kahan-compensated (four
  compensated lanes, not an uncompensated sum). No precision
  regression; if anything the per-lane chains are shorter.
- **Auditability — preserved.** The audit chain, Merkle roots, and
  replay are untouched in *structure*; only the f64 values flowing
  into them change.

### What it costs — a re-lock, confined to the training path

Lane-parallel Kahan reorders the adds, so it is **bit-different** from
today's scalar path — the same class of change as R0's Option C: a
deliberate canary re-lock, **no wire-format bump**. If the change is
confined to `update_rank1` (the n=1 hot path) via dedicated
lane-parallel reductions — leaving the *shared* `quadratic_form` /
`matvec_plus_xty_kahan` (used by `combine`, `kl_divergence`, the n>1
`update`) on the scalar path — then the blast radius is the n=1
training path only. By the same orthogonality R0-3 found, the 28
SHA-256 canaries and the `decide_step` canary do **not** train via
`train_step` / n=1 `blr_update` (the `decide_step` canary's Merge goes
through the *shared* `combine` → unchanged), so **~0 canaries
re-lock**. The Wisconsin BC baseline does train via `train_step`; its
tests are relative (run-to-run, seed-distinctness) and hold, and it
has no absolute chain-head pin — only its `chain_heads.txt` artifact
shifts. Expected churn: a handful of round-trip tests, R0-3-sized.

## 6. Memory efficiency — measured

The handoff (§R0.4 #3) and the user both asked for memory efficiency.
Measured, it is a non-finding:

- **The accumulator is never the memory cost.** Scalar / x4 / x8 Kahan
  accumulators are 16 / 64 / 128-byte *stack* structs — switching to x8
  costs ~112 B per call site, immeasurable. The memory in the update is
  the d×d precision matrix.
- **Per-update churn is already small** — the `precision.to_vec()`
  clone is 14.7 µs (1.4 % of the update); `Tensor::from_vec` *moves*
  its `Vec`. No hotspot, and the handoff recorded "peak ~700 MB, no
  memory pressure."
- **Triangular precision storage — measured and rejected.** Storing
  each symmetric d×d precision as its lower triangle halves the
  footprint (476 → 239 KB/node, ~40 → ~20 MB across the graph). A
  micro-benchmark (`abng-result-profile`) timed a matvec over a full-d²
  layout vs a triangular d(d+1)/2 layout: triangular is **0.76× — 24 %
  *slower***. Halving the storage forces scattered `acc[b] += …`
  writes, `tri(a,b)` index arithmetic, and an `if a≠b` branch — costing
  more than the cache benefit of touching half the bytes. Triangular
  storage trades 24 % speed for memory under no pressure. **Rejected by
  measurement** — no production refactor written; the micro-bench
  settled it.

## 7. Option 1 vs Option 2 — the comparison

| Axis | **Option 1** — lane-parallel x8 Kahan | **Option 2** — + triangular storage |
|---|---|---|
| **Speed** | **~1.2× on the row** (update ~1.05→~0.85 ms; reduction kernel 5.69×) | **0.76× on matvec — slower**; adds nothing over Option 1 |
| **Memory** | accumulator +112 B/call (immeasurable); matrix unchanged | matrix −50 % (476→239 KB/node) — but no memory pressure exists |
| **Determinism** | preserved — `KahanAccumulatorF64x8` bit-stable, no FMA, single-thread; a re-lock vs the scalar order | preserved (byte-identical if `canonical_bytes` reconstructs full d²) |
| **Auditability** | preserved — chain / Merkle / replay intact; re-lock confined to the n=1 train path → **0 of 28 canaries** | preserved |
| **ABNG features** | all preserved — `cargo test --test abng` 624/0 (1 known wall-clock flake passes in isolation) | would touch every precision-indexing site — large, risky surface, for no speed and no needed memory |
| **Verdict** | **SHIP** (R1-1) — modest but real, zero accuracy/determinism cost | **REJECT** — measured slower, relieves no pressure |

## 8. Verdict

- The post-R0 result path is latency-bound on the serial Kahan
  reduction chain inside the rank-1 BLR update; **byte-identical
  speedups are marginal (~1.4 %)**.
- **Option 1 (lane-parallel x8 Kahan) — shipped as R1-1.** ~1.2× on the
  row, honest measured (the design pass over-projected ~1.5× — the
  reduction is only part of the update). Determinism, accuracy, and
  auditability all preserved; **0 of the 28 canaries re-locked** — the
  change is confined to the n=1 training path, which the canaries do
  not exercise. `cargo test --test abng` 624/0; +7 lane-parallel tests.
- **Option 2 (triangular storage) — rejected by measurement.** The
  triangular-layout matvec is 24 % *slower*; it relieves no memory
  pressure. Not implemented — the micro-bench was the test.
- **Cumulative R0 + R1:** per-row ~6.9 → ~2.1 ms (**~3.3×**); the
  residual is the inherent O(d²) rank-1 NIG arithmetic, which cannot be
  cut further without a model approximation (rejected — it would
  change the epistemic-uncertainty output) or an audit-granularity
  change (a separate signed-off decision).
