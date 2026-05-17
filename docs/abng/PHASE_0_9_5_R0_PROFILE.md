# ABNG Phase 0.9.5 — Research Phase R0: Result-Path Profile + Speedup Plan

**Date:** 2026-05-17
**Branch:** `claude/abng-phase-0-9-5`
**Status:** profiling complete; speedup plan proposed; Tier 2 awaits sign-off.
**Supersedes nothing** — this is the R0 deliverable required by
[`PHASE_0_9_5_HANDOFF_V2.md`](PHASE_0_9_5_HANDOFF_V2.md) §R0.6 ("a design
doc / ADR ranking the hotspots and proposing the changes").

---

## 1. Scope

Research Phase R0's question (handoff §R0.1): make ABNG's **result
path** — training (`train_step`) and inference (`predict`) —
dramatically faster and more memory-efficient on wide-categorical
workloads, while holding **every** ABNG guarantee invariant
(determinism, numerical accuracy, auditability, no exotic hardware).

This document profiles the result path with real measurements, ranks
the hotspots empirically, and proposes a tiered, commit-by-commit
speedup plan. It does **not** itself change behaviour.

## 2. Method

A new bench crate, [`bench/abng_result_profile`](../../bench/abng_result_profile/main.rs),
loads the real Diabetes-130 CSV, fits the real `CategoricalTransform`,
builds the real ABNG graph (85 nodes, K=3 routing tree), and times each
result-path segment through the public APIs the path is composed of.
Two synthetic isolations — `sha256(d²·8 bytes)` and `cholesky(d)` — pin
where the per-row cost actually goes. No production code is
instrumented.

**Fitted shape:** `phi` width **d = 247**; the per-node BLR precision
matrix is 247×247 = 61 009 `f64` = **477 KB**.

**Contention caveat.** The development box is oversubscribed (handoff
§R0.4 #4 — "averaged ~0.8 cores"). Absolute wall-clock therefore
carries a noise factor; the profiler reports **median over 9 trials**
(robust to high-side outliers) and every conclusion below leans on
**same-run ratios**, which cancel the contention factor. A consistency
cross-check (`train_step` ≈ `update` + `state_hash`) passed at +2.5 %,
confirming the segment numbers compose.

## 3. Measured result path (d = 247, median over trials)

| Segment | Cost | Notes |
|---|---:|---|
| `transform` (raw row → x, φ, y) | 3.44 µs | per row |
| `encode_prefix` | 89 ns | per row |
| `descend` (radix walk) | 108 ns | per row |
| `graph.observe` (≈ `append_event`) | 2.12 µs | audit append is cheap |
| BLR `update` — n=1 rank-1 NIG, O(d²) | **1.036 ms** | steady state |
| `BlrState::state_hash` | **2.318 ms** | `sha256(canonical_bytes)` |
|  ↳ `canonical_bytes` (alloc + serialize) | 97 µs | 4.2 % of `state_hash` |
|  ↳ `sha256(d²·8 bytes)` (pure compression) | 2.117 ms | **91.3 % of `state_hash`** |
| `cholesky(d)` — O(d³), `chol_factor` miss | 4.513 ms | once per leaf, then cached |
| `blr_predict` — cache hit | 60 µs | D1 Cholesky cache |
| `blr_predict` — cache miss | 5.372 ms | first predict per node only |
| `train_step` (steady, end-to-end) | **3.437 ms** | leaves pre-warmed |

### Per-row breakdown

The COMMIT-5 harness pays, per training row: `transform` +
`encode_prefix` + `descend` (once each) + `train_step` (leaf) +
`blr_update` (root). `train_step` and the root `blr_update` are the
same shape of work — one O(d²) BLR update + one `state_hash` + one
audit append.

```
  transform              3.44 us/row    0.05 %
  encode_prefix          0.09 us/row    0.00 %
  descend                0.11 us/row    0.00 %
  train_step  (leaf)     3.437 ms/row   50.0 %
  blr_update  (root)     3.437 ms/row   50.0 %
  ----------------------------------------------
  TOTAL                  6.877 ms/row

  state_hash   4.636 ms/row   67.4 %   (2x SHA-256 over the d*d precision)
  BLR update   2.072 ms/row   30.1 %   (2x O(d²) rank-1 conjugate update)
  everything else          < 0.6 %
```

### Extrapolation (single-threaded, no contention)

| Workload | CPU time |
|---|---:|
| 16 000 train rows (the ~20K sub-sample) | **110 s (1.8 min)** |
| 81 412 train rows (full 101 766) | **560 s (9.3 min)** |

## 4. Hotspot ranking & the contention reconciliation

1. **`state_hash` — 67.4 % of the row.** Per row the harness SHA-256s
   the **full 247×247 precision matrix twice** (leaf + root). The
   isolation proves this is **91.3 % SHA-256 compression** — only
   4.2 % is the `canonical_bytes` allocation. cjc-snap's portable
   (no-SHA-NI) SHA-256 runs ~225 MB/s here; 477 KB × 2/row is
   unavoidable cost *as long as the per-row digest hashes the d×d
   matrix*.
2. **BLR `update` — 30.1 % of the row.** The n=1 rank-1 NIG conjugate
   update is ~7 O(d²) Kahan passes over the 477 KB matrix — memory-
   bound. Phase 0.9.5's `08a4a6b` already cut this from O(d³) to
   O(d²); the profiler confirms the steady update is 1.0 ms and the
   one-off O(d³) `cholesky` miss (4.5 ms) is now amortised to nothing
   over the thousands of rows each leaf sees.
3. **Everything else — < 0.6 % combined.** `transform`,
   `encode_prefix`, `descend`, `observe`/`append_event` are
   microseconds against milliseconds. The audit append is **not** a
   hotspot. The transform's per-row allocations are real but
   immaterial.
4. **`predict`** — the D1 cache makes a hit 60 µs; only the first
   predict per node pays the 5.4 ms Cholesky. Over an eval pass this
   amortises (handoff §R0.5 — "fine if amortised"). Not a hotspot.

**The reconciliation that reframes the phase.** 16 000 rows = 110 s of
*CPU*. The handoff's COMMIT-5 observation — "a 20 000-row sub-sample
took ~1-2 hours of wall-clock" — is that same ~2 min of CPU stretched
by the documented ~0.8-effective-core oversubscription (handoff §R0.4
#4 already attributes the 87-minute stall to contention, "not the
algorithm"). **The result path was always minutes of CPU.** R0's job
is to cut those minutes; it cannot and need not rescue an
hours-long algorithm, because there was never one.

## 5. Speedup plan — three tiers

### Tier 1 — byte-identical (no sign-off; the 28 canaries hold)

Every Tier-1 change produces **bit-identical** `state_hash`,
`chain_head`, `merkle_root`, snapshots, and predictions. Determinism,
accuracy, and auditability are untouched by construction; the
determinism gate is a regression check, not a re-lock.

- **T1.1 — streaming `state_hash`.** Feed the canonical byte sequence
  directly into `cjc_snap::hash::Sha256` instead of materialising a
  477 KB `Vec` first. Streaming SHA-256 == one-shot SHA-256, so the
  digest is identical (the precedent is Phase 0.7 C `compute_new_hash`
  and Phase 0.8 B2 `serialize_into`). Removes the per-call 477 KB
  allocation + the d² `extend_from_slice` churn.
- **T1.2 — eliminate the redundant `precision.to_vec()` clones.**
  `state_hash`→`canonical_bytes` and `update_rank1` each clone the
  477 KB precision `Tensor` into a fresh `Vec`. Where the borrow model
  allows, hash / read straight from the tensor buffer.
- **Honest ceiling: ~3-5 % of the row.** The allocation is only 4.2 %
  of `state_hash`; Tier 1 cannot touch the 91.3 % that is SHA-256
  *compression*. Tier 1 is worth shipping — it is free and removes
  ~1 MB/row of allocator traffic (handoff §R0.4 #3, memory churn) —
  but it is not the win.

### Tier 2 — the order-of-magnitude lever (requires explicit sign-off)

`state_hash` is 67 % of the row and 91 % of *it* is SHA-256 compressing
**d² bytes**. SHA-256 is a sequential hash with no incremental-edit
property: producing `sha256(buffer)` is irreducibly O(buffer). The
**only** way to make the per-row digest O(d) is to stop committing the
d×d matrix into the per-row audit witness.

This changes what the `TrainStep` / `BlrUpdated` event's 32-byte
`state_hash` field *contains*. The field's **byte layout is
unchanged** (still 32 bytes) ⇒ **no snapshot wire-format bump** (v14
stays v14, decoders unchanged). But the field's *value* changes ⇒ the
chain hashes change ⇒ the **28 SHA-256 canaries re-lock**, plus the
Wisconsin BC `chain_heads.txt`. `serialize.rs:686` already frames a
change to the hash *input* as "v15 work" — this is exactly the
"explicit sign-off" gate the handoff names (§R0.2).

Three sub-options, all O(d) per row:

| Option | Per-row witness | Replay model | Notes |
|---|---|---|---|
| **A — commit-to-inputs** | `sha256(φ ‖ y)` | re-derive state by applying (φ,y); cross-check final reconstructed state vs the snapshot (already stored in full) | standard event-sourcing audit; cleanest |
| **B — incremental digest** | rolling per-node digest updated O(d) per rank-1 update | digest recomputed during replay | new per-node state field |
| **C — periodic snapshot** | full `state_hash` every *k* rows; cheap link in between | unchanged, sparser | weakest tamper-localisation |

**Recommendation: Option A.** It is a well-understood model, the
snapshot *already* stores every node's full BLR state so the
end-of-replay cross-check is free, and it localises tamper to the
exact `(φ, y)` row. Per-row hashing drops O(d²) → O(d): `state_hash`
2.32 ms → `sha256(φ‖y)` ≈ 9 µs.

**Projected impact (Option A):** per-row 6.88 ms → **~2.26 ms (≈3.0×)**;
16 000 rows 110 s → **~36 s CPU**. The row is then dominated by the
2× O(d²) rank-1 update (94 % of the new total).

**Why not more than ~3×.** After Tier 2 the residual cost is the
rank-1 NIG conjugate update. A rank-1 update of a d×d precision matrix
is *inherently* O(d²) — `φφᵀ` touches all d² entries and the
triangular solves are O(d²). Going below O(d²)/row would require
either (a) a diagonal-precision approximation — rejected, it changes
the epistemic-uncertainty output and so violates "accuracy-preserving";
or (b) batched updates — which amortise the cost but change the audit
*granularity* (one event per batch, not per row), a separate
audit-model decision with its own sign-off. Both are out of R0's
byte-identical-or-cheap-sign-off scope and are noted for the team.

### Tier 3 — deferred / out of scope for R0

- **Deterministic parallelism.** Training is a sequential SHA-256 hash
  chain — events must be totally ordered — so naïve row-parallelism
  breaks the chain. A deterministic batch formulation may exist but is
  high-risk and Determinism-Auditor-gated; defer.
- **A faster portable SHA-256 in cjc-snap.** ~225 MB/s is slow even
  for no-intrinsics Rust; a faster *byte-identical* implementation
  would speed every hash in the workspace. But it is a different
  crate, outside R0's "result path" scope, and risky to touch under a
  determinism contract. Note and defer.
- **`predict` cache-miss O(d³).** Already mitigated by the D1 cache;
  amortises over an eval pass. No action.

## 6. Commit plan

| Commit | Content | Gate | Sign-off |
|---|---|---|---|
| **R0-1** | profiler bench crate + this doc | builds | none |
| **R0-2** | T1.1 streaming `state_hash` | `cargo test --test abng` + canaries byte-identical | none |
| **R0-3** | T1.2 clone elimination in the update / hash path | same | none |
| **R0-4+** | Tier 2 Option A (commit-to-inputs) | full gate **+ deliberate 28-canary re-lock** | **required** |

Every commit re-runs the determinism gate (`cargo test --test abng`,
the 28 SHA-256 canaries, the Wisconsin BC baseline, AST↔MIR parity).
R0-2 and R0-3 must show the canaries **unchanged**; R0-4 re-locks them
as a deliberate, reviewed step.

## 7. Determinism / accuracy / auditability impact

| Change | Determinism | Accuracy | Auditability |
|---|---|---|---|
| T1.1 streaming hash | identical digest | no math change | identical chain |
| T1.2 clone elimination | identical digest | no math change | identical chain |
| T2 Option A | deterministic post-relock (same seed+data+build → byte-identical) | no math change — same posterior, same predictions | preserved: per-row `(φ,y)` witness localises tamper; replay re-derives + cross-checks state vs snapshot |

Tier 2 preserves auditability under a *different but equally sound*
model: the chain commits to **inputs** and replay re-derives state,
versus committing to **post-state** directly. Both are tamper-evident;
the input-commit model is the standard event-sourcing form. The change
is not a *weakening* of auditability — it is a re-expression of it —
but because it re-locks the canaries it is a Lead-Language-Architect
decision, not an implementation detail.

## 8. Verdict

- The result path is **6.88 ms/row**; `state_hash` is **67 %** of it
  and is **91 % irreducible SHA-256 compression** over the d×d
  precision matrix.
- The handoff's "1-2 hours" was **machine contention**, not the
  algorithm: the true cost is ~110 s CPU for 16 000 rows.
- **Tier 1** (byte-identical, no sign-off) ships now: ~3-5 %, and it
  drains the per-row allocator churn.
- **Tier 2 Option A** (commit-to-inputs) is the real lever: ~3.0×, to
  ~36 s CPU for 16 000 rows. It needs an explicit, reviewed
  28-canary re-lock — **no wire-format bump** — and is therefore
  surfaced for sign-off before implementation.
