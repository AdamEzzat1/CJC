# ABNG Phase 0.9.5 — Research Phase R0: Result-Path Profile + Speedup Plan

**Date:** 2026-05-17
**Branch:** `claude/abng-phase-0-9-5`
**Status:** COMPLETE — profiled; Tier 1 (R0-2) + Tier 2 Option C (R0-3)
shipped and determinism-gated. §5-§8 record what was shipped.
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

- **T1.1 — streaming `state_hash` (SHIPPED, commit R0-2).** Feeds the
  canonical byte sequence directly into `cjc_snap::hash::Sha256` in
  4 KiB chunks (`hash_f64_slice_be`) instead of materialising a
  ~477 KB `Vec` first. Streaming SHA-256 == one-shot SHA-256, so the
  digest is byte-identical (precedent: Phase 0.7 C `compute_new_hash`,
  Phase 0.8 B2 `serialize_into`); the 28 canaries verified unchanged.
  Removes the per-call ~477 KB allocation + the d² `extend_from_slice`
  churn.
- **T1.2 — `precision.to_vec()` clone elimination — subsumed by
  Tier 2.** Option C removes `state_hash` from the per-row hot path
  entirely (a checkpoint fires only every 64th row), so the redundant
  per-row precision clone it would have targeted no longer exists on
  the hot path. Not shipped as a separate commit.
- **Honest ceiling of Tier 1: ~3-5 % of the row.** The allocation is
  only 4.2 % of `state_hash`; Tier 1 cannot touch the 91.3 % that is
  SHA-256 *compression*. Worth shipping — free, and it drains
  ~1 MB/row of allocator traffic (handoff §R0.4 #3) — but it is not
  the win.

### Tier 2 — the order-of-magnitude lever — SHIPPED as Option C (commit R0-3)

`state_hash` is 67 % of the row and 91 % of *it* is SHA-256 compressing
**d² bytes**. SHA-256 is a sequential hash with no incremental-edit
property, so `sha256(buffer)` is irreducibly O(buffer): the **only**
way to an O(d) per-row digest is to stop committing the d×d matrix
into *every* per-row audit witness.

#### A wire-format correction

The first design pass listed three O(d) options and mis-stated
**commit-to-inputs (A)** as wire-format-neutral. That was wrong, and
was caught and corrected before any implementation:

> For replay to *re-derive* state from inputs it must *have* the
> inputs. Today `φ` is stored nowhere (only `y`, in `TrainStep.value`).
> Commit-to-inputs therefore requires storing the d-vector `φ` in each
> `TrainStep` / `BlrUpdated` event — the payload grows from a fixed
> 40 bytes to a variable ~2 KB. That **is** a genuine v14→v15 snapshot
> wire-format bump (plus a ~50× larger audit log). A witness that
> commits to inputs is only verifiable if replay has the inputs.

The corrected option set:

| Option | Per-row witness | Wire format | Speedup | Tamper localization |
|---|---|---|---|---|
| **A — commit-to-inputs** | `sha256(φ‖y)`; `φ` stored in the event | **v14→v15 bump** | ~3.0× | exact row |
| **C — periodic checkpoint** | full `state_hash` every *k* rows; zero sentinel between | none | ~3.0× | k-row window |
| **L — lower-triangle hash** | `sha256` of the symmetric matrix's lower triangle only | none | ~1.5× | exact row |

**Option C was chosen** (sign-off granted): it recovers essentially
all of A's speedup with **no wire-format bump**, and the
tamper-localization tradeoff (a k-row window instead of the exact row)
is acceptable for a training audit — replay still fully reconstructs
and verifies the final state.

#### Option C as shipped

- **Periodic witness.** `periodic_blr_witness` (graph.rs) gives a BLR
  update its full `state_hash` only when the post-update `n_seen` is a
  multiple of `BLR_CHECKPOINT_INTERVAL` (**= 64**); intermediate rows
  carry `BLR_INTERMEDIATE_WITNESS` — an all-zero sentinel. The row is
  still fully chain-bound by the outer `new_hash = sha256(prev ‖
  payload)`; the sentinel only signals "no independent d×d witness
  here — the nearest checkpoint carries it." Applied to `train_step`
  and the n=1 `blr_update` hot path. Batched (n>1) `blr_update` and
  every structural-op `BlrUpdated` (`combine`/Merge) keep the full
  witness.
- **Flush contract.** `AdaptiveBeliefGraph::checkpoint_blr()` emits one
  full-witness `BlrUpdated` for every node left mid-interval. Replay's
  end-of-replay verifier checks each node's *latest* BLR witness
  against the reconstructed state, so a trained graph **must** call
  `checkpoint_blr` once before `serialize` — the flush-before-serialize
  contract. Forgetting it is a *loud* `DecodeError::BlrStateHashMismatch`,
  never silent corruption. The verifier itself needed **no change**:
  after the flush every node's latest BLR event carries a real hash.
- **No wire-format change.** `TrainStep` / `BlrUpdated` keep their
  exact byte shapes — only the *value* in the 32-byte witness field
  changes (a sentinel on intermediate rows). v14 stays v14; decoders
  are untouched.

#### Impact

A checkpoint fires on 1/64 of rows; the rest pay a ~free sentinel.
Per-row `state_hash` cost (4.64 ms — two updates/row) collapses ~64×.

**Measured** by re-running `abng-result-profile` post-R0-3: per-row
**6.88 ms → 2.38 ms** (median; **≈2.9×**), 16 000 rows **110 s → 38 s
CPU**, 101 766 rows **560 s → 194 s**. The dev box's documented
contention adds noise to absolute wall-clock; the **within-run**
consistency check is the contention-robust read — `train_step`
measured **82 % below** the cost of an every-row full hash
(`update + state_hash`), which is exactly the 63/64 of `state_hash`
that Option C elides. The residual row is dominated by the 2× O(d²)
rank-1 NIG update.

#### The canary outcome

Sign-off anticipated re-locking "the 28 SHA-256 canaries." In the
event **zero canaries re-locked.** The `decide_step` chain-head
canaries and the Wisconsin BC baseline never train through
`train_step` / n=1 `blr_update` — they use `observe`, `decide_step`,
and structural ops, none of which Option C touches — so their chain
heads are byte-identical. (`combine`'s `BlrUpdated` deliberately keeps
the full witness, so the Merge-firing canary is unaffected.) The only
test churn was **6 round-trip tests** that train then
`serialize`+`replay`: each gained one `checkpoint_blr()` call — the
contract, not a re-lock. 9 new tests pin Option C (7 integration in
`blr_checkpoint_tests.rs` + 2 in-crate).

#### Why not more than ~3×

After Option C the row is the 2× O(d²) rank-1 NIG conjugate update. A
rank-1 update of a d×d precision matrix is *inherently* O(d²) — `φφᵀ`
touches all d² entries, the triangular solves are O(d²). Sub-O(d²)/row
would need a diagonal-precision approximation (rejected — it changes
the epistemic-uncertainty output, violating "accuracy-preserving") or
batched updates (which amortise the cost but change the audit
*granularity* to per-batch — a separate signed-off decision). Both are
out of R0 scope and noted for the team.

### Tier 3 — deferred / out of scope for R0

- **Deterministic parallelism.** Training is a sequential SHA-256 hash
  chain; naïve row-parallelism breaks it. High-risk,
  Determinism-Auditor-gated; defer.
- **A faster portable SHA-256 in cjc-snap.** ~225 MB/s is slow even
  for no-intrinsics Rust; a faster *byte-identical* implementation
  would speed every hash in the workspace — but it is a different
  crate, outside R0's "result path" scope. Note and defer.
- **`predict` cache-miss O(d³).** Already mitigated by the D1 cache;
  amortises over an eval pass. No action.

## 6. Commit sequence (as shipped)

| Commit | Content | Gate | Sign-off |
|---|---|---|---|
| **R0-1** | `abng-result-profile` bench crate + this doc | builds | none |
| **R0-2** | T1.1 streaming `state_hash` — byte-identical | `cargo test --test abng` 612/0; canaries unchanged | none |
| **R0-3** | Tier 2 Option C — periodic BLR checkpoints + `checkpoint_blr` | full gate 624/0 (1 known wall-clock flake passes in isolation); 0 canaries re-locked; 6 round-trip tests gain the contract call | Lead Architect (audit model) — granted |

Every commit re-runs the determinism gate (`cargo test --test abng`,
the 28 SHA-256 canaries, the Wisconsin BC baseline, AST↔MIR parity).

## 7. Determinism / accuracy / auditability impact

| Change | Determinism | Accuracy | Auditability |
|---|---|---|---|
| R0-2 streaming hash | identical digest | no math change | identical chain |
| R0-3 Option C | deterministic — same seed+data+build → byte-identical; `checkpoint_blr` visits nodes in `NodeId` order | no math change — same posterior, same predictions; the BLR `update` math is untouched | preserved — see below |

Option C does **not weaken** auditability. Every audit event is still
chain-bound and tamper-evident via the outer `new_hash`; every node's
final d×d state is still cryptographically verified at replay (against
the `checkpoint_blr` flush witness); the d×d state is independently
witnessed every 64 rows. What changes is *granularity*: a tampered
mid-interval state is still **detected**, but **localized** to a
64-row window rather than the exact row. That is the tradeoff the Lead
Architect signed off — a re-expression of the audit model, not a
reduction of it.

## 8. Verdict

- The result path was **6.88 ms/row**; `state_hash` was **67 %** of it
  and **91 %** of that is irreducible SHA-256 compression over the d×d
  precision matrix.
- The handoff's "1-2 hours" was **machine contention**, not the
  algorithm: the true cost was ~110 s CPU for 16 000 rows.
- **R0-2** (Tier 1, byte-identical) drained the per-row allocator
  churn; the 28 canaries verified unchanged.
- **R0-3** (Tier 2 Option C) is the lever: per-row hashing O(d²) →
  amortized O(d²/64), the row **~2.9× faster** measured (6.88 →
  2.38 ms/row; ~110 s → ~38 s CPU for 16 000 rows), **no wire-format
  bump**, **zero canaries re-locked**.
- The residual post-R0 row is dominated by the inherent O(d²) rank-1
  NIG update; going further needs a model or audit-granularity change
  outside R0's scope.
