# ABNG Phase 0.9.5 — Research Phase R1: post-R0 result-path speed + memory

**Date:** 2026-05-17
**Branch:** `claude/abng-phase-0-9-5` (worktree `claude/zealous-nobel-66abc9`)
**Status:** profiled; speedup proposed; the one real lever awaits sign-off.
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

The serial chain is broken by accumulating into **4 (or 8) independent
lanes** and horizontally reducing once at the end:
`KahanAccumulatorF64x4` / `x8` from `cjc-repro`. Four independent Kahan
recurrences run with instruction-level parallelism (and opportunistic
SIMD on AVX/NEON release builds) instead of one serial chain —
realistically **~2-4×** on the reduction passes, taking the ~804 µs
of Kahan reductions to roughly ~250-400 µs and the whole update from
~1.04 ms toward ~0.5-0.6 ms (**~1.7-2× on the update**, **~1.5× on the
row**: 2.6 → ~1.7 ms).

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

## 6. Memory efficiency

The handoff (§R0.4 #3) and the user both asked for memory efficiency.
The measured picture:

- **Per-update churn is already small** — the 14 µs clone, plus O(d)
  scratch vectors. `Tensor::from_vec` *moves* its `Vec` (no copy). The
  audit log grows ~100 B/event; `transform` allocates small per-row
  vectors. None is a hotspot, and the handoff itself recorded "peak
  ~700 MB, no memory pressure."
- **The one real footprint item: triangular precision storage.** Each
  node's BLR precision is a *symmetric* d×d matrix stored in full —
  477 KB at d=247, ~40 MB across an 85-node graph. Storing only the
  lower triangle (d(d+1)/2) halves that to ~20 MB. It can be made
  byte-identical (`canonical_bytes` reconstructs the full d² for the
  hash, exactly as snapshot decoding already mirrors the v14
  lower-triangle layout). But it touches every precision-indexing site
  (`cholesky`, `chol_rank1_update`, the solves, `matvec`,
  `quadratic_form`), it does **not** speed the compute (the d²
  multiply-adds are unchanged), and there is no memory *pressure* to
  relieve. **Noted; deferred** — high churn, no speed gain, no pressure.

## 7. Recommendation

- **Byte-identical R1 — nothing worth shipping.** The only byte-
  identical lever (the clone) is 1.4 % and costs a 40-site refactor.
  Skip it.
- **The R1 lever is lane-parallel Kahan in `update_rank1`** — ~1.5× on
  the row, determinism + accuracy + auditability all preserved, a
  re-lock confined to the training path (~0 canaries, like Option C).
  It needs the same Lead-Architect sign-off Option C did; surfaced
  before implementation.
- **Triangular precision storage** (memory) — deferred: invasive, no
  speed gain, no memory pressure.

## 8. If the lever is taken — commit plan

| Commit | Content | Gate |
|---|---|---|
| **R1-1** | lane-parallel Kahan reductions in `update_rank1` (arbitrary d: ×4 lanes + scalar tail), dedicated so shared helpers stay scalar | full `cargo test --test abng`; canaries expected unchanged; round-trip tests re-checked |

A single focused commit — the change is one hot function plus its
dedicated reduction helpers and their tests.
