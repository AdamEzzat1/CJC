# Blog Notes — Quantum-Inspired CANA Compression + Energy Ranking

Source material for a future technical blog post on the
`cjc-cana-compress` Phase-6 layer. Sections are roughly in the order a
blog post would present them.

---

## 1. Problem statement — what we wanted

CJC-Lang's compiler intelligence layer (CANA) had a clean Phase 1
"passive observer" + Phase 2-5 advisory stack that worked, but it
suffered from three quietly-growing problems:

1. **Pass histories were keeping every compilation's full diagnostic
   trail in memory**. Audit chains for a multi-stage compilation could
   reach megabytes — not catastrophic, but a smell.
2. **The cost-model layer was a weighted sum of hand-tuned
   coefficients**. Every time a new optimization pass landed, the
   weights had to be re-trained or re-justified. The objective wasn't
   *structured* — it was a single scalar with no decomposition you
   could show an auditor.
3. **CANA and NSS talked through `PressurePredictor`**, but the
   conversation was one-way. CANA asked NSS "what's the thermal
   forecast?" NSS had no way to ask CANA "did your last compression
   decision *change* the thermal forecast?"

We wanted a Phase 6 layer that:

- Compressed audit-trail data losslessly (so no semantic information
  is lost).
- Compressed advisory feature data lossily (so we could trade
  precision for memory).
- Gave us an **auditable energy decomposition** for ranking compiler
  plans, instead of a single opaque scalar.
- Closed the CANA↔NSS loop: compression decisions move pressure;
  pressure feeds back into ranking.

And we wanted it without compromising CJC-Lang's central invariants:
*determinism*, *no hidden allocations in `@nogc`*, *legality and
verifier authority over MIR*.

---

## 2. Classical compiler baseline

Before this layer, CANA had:

- A passive observer that extracted features from MIR.
- A linear cost model (`LinearCostModel`) producing scalar `CostEstimate`s.
- A `LegalityGate` trait + `DefaultLegalityGate` / `PerPassLegalityGate`
  enforcing safety classifications on proposed pass reorders.
- A `ThermalAwareCostModel` that wrapped a base cost model with
  per-function thermal-pressure penalties.
- A bridge crate `cjc-cana-nss` implementing `PressurePredictor` via
  NSS's synthetic-trace projection.

Everything was deterministic, everything was correctly partitioned
into "advisory" vs "authoritative", everything used `BTreeMap` and
`KahanAccumulatorF64`. But pass histories were stored uncompressed,
and the cost model's scalar wasn't decomposed.

---

## 3. Why MPS / tensor-train ideas are useful for compression

CANA's feature substrate is *naturally tensor-shaped*:

- Per-function feature vectors of length ~10 (CFG metrics, memory
  proxy, reduction axes).
- Stacked across N functions → matrix `(N, 10)`.
- Tracked over T optimization rounds → tensor `(T, N, 10)`.
- Tracked across pass-ordering variants V → tensor `(V, T, N, 10)`.

For audit purposes we want this stored *exactly* — but for advisory
queries ("which functions have the most similar cost profile?",
"does this pass ordering increase pressure on a redundant axis?") we
only need a low-rank summary.

**Matrix Product States (MPS) — also known as Tensor-Train (TT)
decomposition** — is a quantum-computing primitive (it represents
many-body quantum states with bounded entanglement entropy in
O(N · χ²) memory instead of O(2^N)). The decomposition is:

```
T[i_1, ..., i_n]  ≈  G_1[1, i_1, :] · G_2[:, i_2, :] · ... · G_n[:, i_n, 1]
```

where each "core" `G_k` has shape `(r_k × d_k × r_{k+1})` and the bond
dimensions `r_k` control accuracy. For feature tensors with low rank
along most axes, this is a major compression win.

We reuse [`cjc_quantum::mps::svd_sign_stabilized`](../../crates/cjc-quantum/src/mps.rs)
— the same deterministic SVD primitive the QAOA module uses for
quantum circuit simulation — to compute the chain. The discipline is
inherited: one-sided Jacobi rotations (no random pivoting),
deterministic sign stabilization (largest-magnitude entry of each U
column is positive), Kahan accumulation, no FMA.

What that gives us: **a real-valued compiler-feature compression
codec whose determinism story is the same as the quantum simulator's
determinism story**. If `cjc_quantum::mps` produces byte-identical
output for byte-identical input across runs and platforms, so does
our TT compression — by reuse.

(One footnote: while building this, we discovered an upstream bug in
`svd_sign_stabilized` for wide matrices (`m < n`) where dominant
singular vectors can land in arbitrary columns of the converged
`work` matrix but only the first `min(m, n)` are extracted. We
landed the upstream fix in the same session: `svd_sign_stabilized`
now routes wide matrices through the tall path via the conjugate-
transpose SVD identity `A^H = V Σ U^H`, with six new regression
tests covering wide real, wide complex, the boundary 1×N case,
determinism, and the conjugate-transpose identity itself.)

---

## 4. Why QAOA / Ising-style scoring is useful for compiler choices

The cost model in a deterministic compiler has to balance ~9
competing objectives at once: runtime cost, memory pressure, thermal
pressure, code size, reconstruction risk, verifier risk, fusion
reward, reuse reward, compression reward. Squashing those into a
single scalar means you've already chosen one set of weights — and
you have to defend those weights to every reviewer.

**Ising-model objectives in physics** have a similar shape:
weighted sum of contributions where some terms are "ferromagnetic"
(pull toward alignment) and others "antiferromagnetic" (push apart).
Quantum Approximate Optimization Algorithm (QAOA) is a heuristic for
minimizing such objectives by parameterizing them as the expectation
value of a Hamiltonian and searching parameter space.

We don't run QAOA. But we **borrow the shape**:

```
energy = costs - rewards
       = (runtime + memory + thermal + code_size + recon_risk + verifier_risk)
       - (fusion_reward + reuse_reward + compression_reward)
```

The 9 components are *exposed* in [`EnergyComponents`](../../crates/cjc-cana-compress/src/energy.rs).
Each `RankedCandidate` carries its component decomposition. A
reviewer asking "why did this plan win?" gets a complete answer:
"runtime cost was 2.5, but fusion reward was 3.1, so total energy was
-0.6."

That's the *transparency win* the quantum-inspired framing buys us.
No hidden scalarization weights.

---

## 5. Why NSS pressure correlations matter

NSS already had:

- per-tick `PressureField` (one `Pressure` value per `PressureKind`),
- `PressureGraph` for propagation,
- `PressureTrajectory` for replay.

What it didn't have was a **density-matrix-shaped summary** that
captured *which pressures move together*. Without that, you couldn't
ask:

- *"When compression reduces memory pressure, does thermal pressure
   follow (because we're trading bytes for compute cycles)?"*
- *"Are there two pressures that, when both saturated, drive a
   global collapse — even though individually neither is at its
   instability threshold?"*

The new [`PressureDensityState`](../../crates/cjc-nss/src/density.rs) adds:

- Diagonal: per-`PressureKind` magnitude (instantaneous or
  Kahan-mean over a trajectory).
- Off-diagonal: pairwise Pearson correlation of pressures' time
  series.

And `PressureCorrelationSummary` derives:

- `collapse_risk`: max per-kind saturation.
- `saturation_score`: average saturation across kinds.
- `dominant_coupling`: largest absolute pairwise correlation.

This is enough to answer the "do pressures move together?" question
without committing to a full quantum-style mixed-state representation.

---

## 6. Determinism story

The first thing a reviewer asks about a quantum-flavored numerical
layer is: **"are you sure this is deterministic?"**

We are. Every component:

- Uses `BTreeMap` / `BTreeSet` everywhere; no `HashMap` iteration in
  decision paths.
- Uses `KahanAccumulatorF64` for every floating-point reduction.
- Uses FNV-1a (via `cjc_cana::CanaHasher`) for all content addressing.
- Uses `f64::total_cmp` for any sort involving floats.
- Avoids FMA contraction (`a + b` and `a * b` are written separately).
- Avoids `Instant::now`, `rand::thread_rng`, and any other
  nondeterministic source.

The wiring tests double-run the entire stack and assert byte-identical
output:

```rust
#[test]
fn end_to_end_pipeline_double_run() {
    let run = || { /* build plan -> execute -> bridge -> summary */ };
    let a = run();
    let b = run();
    assert_eq!(a.report_hash, b.report_hash);
    assert_eq!(a.density_bytes, b.density_bytes);
    assert_eq!(a.summary, b.summary);
    // ... etc
}
```

Eight such canary tests live in
[`tests/determinism.rs`](../../crates/cjc-cana-compress/tests/determinism.rs).

---

## 7. Thermal-control story

CJC-Lang's design philosophy treats thermal pressure as a first-class
resource. Compression decisions can shift the thermal budget:

- A lossless compression that saves memory bytes also saves the bytes
  that would have been traversed by downstream code → small
  throughput reward, no thermal cost.
- A lossy compression that produces a low-rank summary requires the
  consumer to *reconstruct* the original data → reconstruction work
  → thermal cost.

The bridge translates this directly: validated advisory entries add
`thermal_advisory_scale · observed_error` to the post-delta thermal
magnitude. The energy ranker then sees the full picture: a plan that
saves memory but heats up the chip gets *net* energy reflecting both.

You can't game it by choosing a compression that "looks free" — the
thermal cost will surface in the energy total.

---

## 8. Test / verification summary

| Layer | Test type | Count | Pass status |
|---|---|---:|---|
| candidate.rs | unit | 16 | ✅ |
| lossless_trace.rs | unit | 17 | ✅ |
| motif_dictionary.rs | unit | 13 | ✅ |
| lowrank.rs | unit | 14 | ✅ |
| tensor_train.rs | unit | 13 | ✅ |
| plan.rs | unit | 13 | ✅ |
| report.rs | unit | 9 | ✅ |
| energy.rs | unit | 13 | ✅ |
| bridge.rs | unit | 8 | ✅ |
| density.rs (in cjc-nss) | unit | 18 | ✅ |
| **`tests/wiring.rs`** | integration | 8 | ✅ |
| **`tests/proptest_compress.rs`** | proptest | 11 (× 64 cases each) | ✅ |
| **`tests/bolero_fuzz.rs`** | bolero fuzz | 7 (× 500–1000 iter each) | ✅ |
| **`tests/determinism.rs`** | double-run regression | 8 | ✅ |
| **Total** | | **168 + ~700 fuzz iter + ~700 proptest cases** | ✅ |

Verification commands:

```bash
cargo fmt --all --check
cargo test -p cjc-cana-compress --release
cargo test -p cjc-nss --release
cargo test -p cjc-quantum --release    # untouched
cargo test --workspace --release       # parity gate + everything else
```

---

## 9. What's deliberately deferred

- **User-facing CLI for compression reports**: the layer ships as a
  library; the CLI integration into `cjcl` is queued.
- **Real QAOA on hardware**: the energy decomposition is shaped so
  a future Hamiltonian-compiler could ingest it, but Phase 6 ships
  pure classical ranking.
- **Lossless schemes beyond RLE + LZ77**: we considered LZ4 / zstd
  but rejected pulling in C dependencies. A pure-Rust zstd would be
  a future addition.
- **Streaming compression**: the API is batch-oriented (compress a
  full payload, decompress a full payload). Streaming would let the
  pass-history adapter compress incrementally as records land.
- **`cjcl run --compression-report` flag**: integrating the report
  into the CLI sidecar artefact set is a small follow-up.

---

## 10. Anticipated blog-post structure

1. **Hook**: "We added a compression layer to our compiler that
   thinks in tensor trains."
2. **Problem framing**: pass histories were unbounded, cost model
   was opaque, CANA↔NSS conversation was one-way.
3. **Why "quantum-inspired" is honest here**: borrowed discipline,
   not borrowed mysticism. Exposed decomposition vs hidden weights.
4. **The four-piece architecture**: `Candidate → Plan → Report →
   Bridge/Ranker`.
5. **The hard rule** (semantic-critical + lossy → compile-time
   error): a type-system trick that pays off.
6. **The bug we found in our own quantum stack**: `svd_sign_stabilized`'s
   wide-matrix failure mode + the transpose adapter that works around
   it. Honest war-story material.
7. **The auditable energy decomposition**: why exposing `score.components`
   to reviewers is better than tuning hidden weights.
8. **Determinism canaries**: how we know we got it right.
9. **What's next**: real QAOA hardware ingest, streaming, CLI sidecar.
