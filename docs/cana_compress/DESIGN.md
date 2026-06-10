# CANA Compression Layer + Quantum-Inspired Energy Ranking — Design Doc

**Status:** Shipped (Phase 6a, 2026-06-09)
**Crates:** `cjc-cana-compress` (new), `cjc-nss::density` (new module)
**Prior art:** `docs/cana/HANDOFF_NEXT_SESSION_v4.md`, `docs/nss/HANDOFF_PHASE_5_COMPILER_INTEGRATION.md`, `docs/QUANTUM_SIMULATION.md`

---

## 1. Why this is quantum-inspired, not quantum-dependent

We borrow the *discipline* of CJC-Lang's deterministic quantum simulator
([`cjc-quantum`](../../crates/cjc-quantum/src/lib.rs)) — sign-stabilized SVD, Kahan
accumulation, fixed iteration order, no FMA, stable iteration over
collections — and the *shape* of two algorithms from quantum
computing literature:

- **MPS / TT decomposition** for compressing high-dimensional advisory
  feature tensors;
- **Ising/QAOA-style energy minimization** for ranking compiler plans
  against a multi-objective scalar.

**We do not** require quantum hardware, simulate quantum circuits at
runtime, or claim the layer benefits from quantum speedup. The
terminology is honest only because:

1. The Ising energy decomposition is **fully exposed** in the
   [`EnergyComponents`](../../crates/cjc-cana-compress/src/energy.rs) struct, so an
   auditor can see exactly which cost terms and reward terms drove a
   ranking. No hidden weights.
2. The tensor-train compressor literally calls into
   [`cjc_quantum::mps::svd_sign_stabilized`](../../crates/cjc-quantum/src/mps.rs)
   — the same deterministic SVD primitive that powers the QAOA / DMRG
   modules.
3. Every step in the compression pipeline is a deterministic, classical
   transformation. The "quantum" framing is a *naming convention* and a
   *reuse pathway*, not a runtime dependency.

If a future Phase 7 layer wants to swap in real quantum hardware (e.g.,
to solve the energy minimization on a small QAOA device), the
candidate-shape and energy-decomposition surfaces are already
expressed in a form a Hamiltonian compiler could ingest. But that's
out of scope here.

---

## 2. CANA Compression Layer — mechanics

### 2.1 Four-piece type system

```rust
pub enum CompressionKind {
    LosslessTrace,        // lossless
    MotifDictionary,      // lossless
    LowRankAdvisory,      // lossy (advisory only)
    TensorTrainAdvisory,  // lossy (advisory only)
}

pub enum Criticality {
    SemanticCritical,
    AdvisoryOnly { tolerance_f: f64 },
}

pub struct CompressionCandidate { /* id, kind, criticality, payload, label */ }
pub struct CompressionPlan { /* sorted-by-id list of candidates */ }
pub struct CompressionReport { /* per-entry status + canonical bytes + hash */ }
```

The hard rule — **semantic-critical facts must use a lossless kind** —
is enforced at construction time:

```rust
if kind.is_lossy() && matches!(criticality, Criticality::SemanticCritical) {
    return Err(CompressionError::LossyOnCritical { kind });
}
```

So "compress a no-GC fact with low-rank" is a compile-time-shaped error
rather than a soundness bug discovered at audit time.

### 2.2 Lossless schemes

- **`LosslessTrace`** — byte-level RLE with magic header `"CLT0"`, original
  length stamp, and FNV-1a hash of the input bytes. The encoder is one-
  pass greedy: at each position, emit the longest run of equal bytes if
  ≥ 2, else emit a literal block ending where the next run would begin.
  Round-trips exactly; the decoder verifies the length stamp and input
  hash before accepting.

- **`MotifDictionary`** — LZ77-style back-references with magic header
  `"CMD0"`. Encoder is also one-pass greedy: longest match in the
  sliding window of `WINDOW_SIZE = 32768` bytes wins; on length ties,
  the **smallest offset** (= most recent occurrence) wins. `MIN_MATCH =
  3`, `MAX_MATCH = 130`. Literals batched into blocks of up to 128.

Both schemes wrap canonical pass-history bytes via
[`compress_pass_history`](../../crates/cjc-cana-compress/src/lossless_trace.rs) /
[`decompress_pass_history`](../../crates/cjc-cana-compress/src/lossless_trace.rs).
The PassHistory canonical encoding is documented inline in
[`lossless_trace.rs`](../../crates/cjc-cana-compress/src/lossless_trace.rs).

### 2.3 Advisory lossy schemes

- **`LowRankAdvisory`** — deterministic truncated SVD via power
  iteration with deflation. For a feature matrix `M ∈ ℝ^(m×n)`, we
  compute the rank-`K` approximation `M ≈ Σ σ_k u_k v_k^T`. Initial
  eigenvectors are seeded with a deterministic alternating-±1 pattern
  (no RNG). Sign stabilization: largest-magnitude `v_k` entry forced
  positive; on ties the smallest index wins.

- **`TensorTrainAdvisory`** — TT-SVD that reuses
  [`cjc_quantum::mps::svd_sign_stabilized`](../../crates/cjc-quantum/src/mps.rs).
  We lift real `f64` matrices into `ComplexF64` with `im = 0.0`, run
  the existing SVD, and drop the (numerically zero) imaginary parts on
  the way out. **No special-casing on `m < n`**: the wide-matrix bug
  in `svd_sign_stabilized` that this crate originally discovered (one-
  sided Jacobi can leave dominant singular vectors outside the first
  `min(m, n)` columns of the converged `work` matrix when `m < n`) is
  fixed upstream — `svd_sign_stabilized` now routes wide matrices
  through the tall path via the conjugate-transpose SVD identity. See
  that function's docstring for the routing logic and its regression
  tests for the property coverage.

Both lossy schemes report observed Frobenius error and compare it
against the declared `tolerance_f` from `Criticality::AdvisoryOnly`.
Exceeding tolerance produces `EntryStatus::ToleranceExceeded` in the
report — never a panic, never a silent acceptance.

### 2.4 What gets compressed

| Compressible | Not compressible (verifier-critical) |
|---|---|
| Pass histories | MIR semantic ops |
| Pass diagnostics | Alias/effect facts |
| Repeated MIR motifs | No-GC facts |
| Shape/profile summaries | Verifier state |
| Runtime pressure trajectories | Audit chains |
| Kernel decision histories | Exact shape facts |
| NSS pressure traces | User-visible values |
| CANA feature vectors (advisory) | |

---

## 3. NSS Pressure Correlation — mechanics

### 3.1 `PressureDensityState`

A density-matrix-inspired summary with:

- **Diagonal** `D[k]` = per-`PressureKind` magnitude (from the last
  observation in a trajectory, or Kahan-summed mean across the
  trajectory).
- **Thresholds** `T[k]` = instability thresholds for normalization.
- **Off-diagonal** `C[i, j] = cov(m_i, m_j) / (σ_i σ_j)` ∈ `[-1, 1]`:
  Pearson correlation between pressure dimensions `i` and `j` over the
  trajectory.

Stored in `BTreeMap` for total iteration order. Canonical bytes carry
a `"NDS0"` magic header + diagonal + thresholds + correlations in
fixed kind-declaration order.

### 3.2 `PressureCorrelationSummary`

Derived view exposing:

- `saturation_score` — average per-kind saturation across all 9
  `PressureKind`s, normalized to `[0, 1]`. Answers "how loaded is the
  system overall?"
- `collapse_risk` — max per-kind saturation. Answers "which dimension
  is closest to its instability threshold?"
- `dominant_coupling` — the off-diagonal cell with the largest
  absolute value. Answers "which two pressures are moving together?"
- `dominant_kind_for_risk` — which `PressureKind` produced
  `collapse_risk`.
- `stable_hash` — FNV-1a over canonical bytes.

JSON serializer is hand-written (same philosophy as
[`cjc_cana::report`](../../crates/cjc-cana/src/report.rs)): no serde, deterministic key
order, ~50 LOC.

### 3.3 Why "density-matrix-inspired"

A quantum density matrix `ρ ∈ ℂ^{N×N}` has:

- diagonal entries = probabilities of outcomes,
- off-diagonal entries = coherences (interference between outcomes).

Our `PressureDensityState` has the same shape:

- diagonal entries = per-kind pressure magnitudes,
- off-diagonal entries = pairwise correlations (interference between
  pressure dimensions).

We don't claim Hermiticity (the matrix is symmetric, real, but not
"complex Hermitian"). We don't claim a tracial constraint (the
diagonal doesn't sum to 1). The analogy is for *naming and intuition*,
not for borrowing quantum theorems.

---

## 4. CANA ↔ NSS bridge

The bridge translates a `CompressionReport` into per-`PressureKind`
deltas applied to a baseline `PressureDensityState`:

| Compression event | Pressure effect |
|---|---|
| Validated lossless entry with original > compressed | Memory pressure ↓ by `memory_reward_scale · (1 - ratio)` |
| Validated lossless entry | Throughput pressure ↓ by `throughput_reward_per_validated` |
| Validated advisory entry with observed_error > 0 | Thermal pressure ↑ by `thermal_advisory_scale · observed_error` |
| `MalformedRoundTrip` | Memory pressure ↑ by `memory_malformed_penalty` |
| `ToleranceExceeded` | Thermal pressure ↑ by `thermal_tolerance_exceeded_penalty` |
| `DecodeFailed` | No pressure movement; logged as `penalised_entries` |

Coefficients are tunable via
[`BridgeCoefficients`](../../crates/cjc-cana-compress/src/bridge.rs); the defaults
are hand-picked conservative weights. `BridgeCoefficients::zero()`
makes the bridge a no-op (useful for tests that want to isolate
specific effect paths).

The bridge returns a [`CompressionPressureDelta`](../../crates/cjc-cana-compress/src/bridge.rs)
carrying the updated state, its derived summary, signed aggregate
deltas, and counts of rewarded vs penalised entries. The energy ranker
consumes these to produce a re-ranked plan list — see §6.

---

## 5. Quantum-inspired energy ranking

The energy objective:

```
energy =
  + runtime_cost
  + memory_pressure
  + thermal_pressure
  + code_size_cost
  + reconstruction_risk
  + verifier_risk_penalty
  - fusion_reward
  - reuse_reward
  - compression_reward
```

`EnergyComponents` validates every term is finite and non-negative.
`EnergyScore::from_components` Kahan-sums the terms in declaration
order. `EnergyRanker::rank` sorts by `(total ASC, candidate_id ASC)`,
dropping non-finite components into `metadata.dropped` (never
panicking). Standard f64 `total_cmp` is deterministic across
platforms.

The decomposition is exposed in every `RankedCandidate.score.components`
— an auditor can see exactly which cost term made a plan lose, or
which reward term carried it to the top. No hidden scalarization
weights.

---

## 6. End-to-end flow

```text
   CANA features / pass history
       │
       ▼
   CompressionCandidate (advisory or semantic-critical)
       │
       ▼
   CompressionPlan ─► chosen CompressionKind
       │              (LosslessTrace / MotifDict / LowRank / TT)
       ▼
   CompressionReport (validated entries + observed errors +
                      content-addressed report_hash)
       │
       ├─► bridge::compression_pressure_delta
       │      └─► PressureDensityState' (memory ↓, thermal ↑ for advisory)
       │           └─► PressureCorrelationSummary (saturation, collapse risk)
       │
       └─► EnergyRanker::rank
              └─► [(candidate_id, score), ...] sorted ascending
                    └─► best plan = arg min energy

   ⮡ cjc_cana::LegalityGate / cjc-mir verifier retain final
     authority over MIR. The compression layer never mutates a
     MirProgram and has no `cjc-mir-exec` dependency.
```

---

## 7. Determinism preservation

Identical input → byte-identical output at every layer:

- [`BTreeMap`](https://doc.rust-lang.org/std/collections/struct.BTreeMap.html)
  everywhere; no `HashMap`/`HashSet` in decision paths.
- All reductions via [`cjc_repro::KahanAccumulatorF64`](../../crates/cjc-repro/src/kahan.rs).
- All RNG via deterministic seed pattern (initial eigenvector seeded
  by a fixed alternating-±1 pattern; no `Instant::now`, no
  thread-local randomness).
- All hashing via [`cjc_cana::CanaHasher`](../../crates/cjc-cana/src/hash.rs)
  (FNV-1a, 64-bit, no per-process key).
- All canonical-bytes serializers use little-endian byte order with
  explicit `.to_bits().to_le_bytes()` for `f64`.
- `f64::total_cmp` for any sorting that touches floats.
- Stable tie-break by `CandidateId` in every ranking output.

The wiring tests in
[`tests/wiring.rs`](../../crates/cjc-cana-compress/tests/wiring.rs) +
[`tests/determinism.rs`](../../crates/cjc-cana-compress/tests/determinism.rs)
double-run the entire stack to assert byte-identity.

---

## 8. Thermal-control story

The compression layer connects to NSS's existing thermal model in two
places:

1. **Bridge**: advisory entries with non-zero observed reconstruction
   error add `thermal_advisory_scale · observed_error` to the post-
   delta thermal pressure. The reconstruction has to happen *at
   runtime*, so it taxes thermal headroom.

2. **Energy ranker**: `thermal_pressure` is one of the 9 cost terms.
   A plan that lowers memory pressure (compression reward) but raises
   thermal pressure (reconstruction cost) gets a *net* energy that
   reflects both effects. The ranker can't be tricked into picking a
   thermally-disastrous compression just because it scored well on
   memory.

This composes with the existing `ThermalAwareCostModel` in
[`cjc-cana`](../../crates/cjc-cana/src/thermal_cost_model.rs): the
post-bridge `PressureCorrelationSummary` can be fed into
`predict_thermal()` to get the per-function thermal forecast the
existing cost model uses, with the compression-induced delta already
baked in.

---

## 9. What facts may / may not be compressed

**May be compressed losslessly** (round-trip exact):

- Pass histories (`cjc_cana::PassHistory`)
- Pass diagnostics
- Audit chains
- Verifier state snapshots

**May be compressed lossily** (advisory only, with declared tolerance):

- Cost-model feature histograms
- Scratch feature vectors (CANA's per-function metrics summarized)
- Runtime pressure trajectories
- Kernel-decision histories
- NSS pressure traces (for archival, not for replay)

**Must NEVER be lossily compressed**:

- MIR semantic ops (control flow, type info, arithmetic)
- Alias/effect facts
- No-GC facts
- Exact shape facts
- User-visible values
- Anything that feeds directly into legality gates

The type system enforces this. There is **no API** for
`SemanticCritical + LowRankAdvisory` — the constructor returns
`Err(LossyOnCritical)`.

---

## 10. Reading order for a reviewer

1. [`crates/cjc-cana-compress/src/lib.rs`](../../crates/cjc-cana-compress/src/lib.rs) — module list + crate-level docs
2. [`crates/cjc-cana-compress/src/candidate.rs`](../../crates/cjc-cana-compress/src/candidate.rs) — the hard rule
3. [`crates/cjc-cana-compress/src/plan.rs`](../../crates/cjc-cana-compress/src/plan.rs) — the executor
4. [`crates/cjc-cana-compress/src/report.rs`](../../crates/cjc-cana-compress/src/report.rs) — canonical bytes + JSON
5. [`crates/cjc-cana-compress/src/energy.rs`](../../crates/cjc-cana-compress/src/energy.rs) — Ising-style ranker
6. [`crates/cjc-nss/src/density.rs`](../../crates/cjc-nss/src/density.rs) — pressure-density structures
7. [`crates/cjc-cana-compress/src/bridge.rs`](../../crates/cjc-cana-compress/src/bridge.rs) — CANA ↔ NSS plumbing
8. [`crates/cjc-cana-compress/tests/wiring.rs`](../../crates/cjc-cana-compress/tests/wiring.rs) — end-to-end properties
9. [`crates/cjc-cana-compress/tests/determinism.rs`](../../crates/cjc-cana-compress/tests/determinism.rs) — bit-equality canaries
