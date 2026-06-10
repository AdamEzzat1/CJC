# Phase A Handoff — Quantum-Inspired Compression + Energy Ranking + What's Next

**Date:** 2026-06-09
**Branch:** `claude/objective-davinci-62975a`
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\objective-davinci-62975a`
**Author:** session that shipped the quantum-inspired CANA/NSS compression layer + the SVD upstream fix + Phase A1 (EnergyAwarePassRanker) + Phase A3 (CompressedCanaSidecar)

This document is a deliberate handoff: the next session is adding a
**physics-informed neural (PINN) layer** to CANA and NSS, and the
classical infrastructure (A4-A6 + training) will be finished after
that. This note tells the next session:

1. What's already on disk and tested (so you don't redo work).
2. What's left in Phase A and beyond.
3. How the PINN layer fits architecturally — without removing the
   quantum-inspired pieces.
4. Suggested ordering.

---

## 0. TL;DR — what shipped this session

**11 commits' worth of work** landed across one focused session:

```text
Phase 6 — quantum-inspired CANA/NSS + compression layer (the original prompt):
  cjc-cana-compress (new crate)           ~3,500 LOC src + 1,700 LOC tests
  cjc-nss::density (new module)           ~500 LOC src + tests
  workspace Cargo.toml                    add member + workspace dep
  docs/cana_compress/                     DESIGN, ARCHITECTURE, BLOG_NOTES,
                                            VERIFICATION_REPORT

cjc-quantum SVD bug fix (discovered during TT-SVD):
  cjc-quantum/src/mps.rs                   ~70 LOC wide-matrix routing
                                            + 6 regression tests
  cjc-cana-compress/src/tensor_train.rs    workaround removed

Phase A1 — EnergyAwarePassRanker:
  cjc-cana-compress/src/energy_pass_ranker.rs   ~600 LOC + 16 unit tests
  cjc-cana-compress/tests/energy_pass_ranker_wiring.rs   ~360 LOC + 9 wiring tests

Phase A3 — CompressedCanaSidecar persistence:
  cjc-cana-compress/src/sidecar.rs         ~550 LOC + 18 unit tests
```

**Test totals across the session:**

| Crate | Tests | Pass | Fail |
|---|---:|---:|---:|
| cjc-cana-compress (lib) | 155 | 155 | 0 |
| cjc-cana-compress (wiring, proptest, bolero, determinism, energy_pass_ranker_wiring) | 52 | 52 | 0 |
| cjc-nss (incl. new density module) | 217+ | 217+ | 0 |
| cjc-quantum (incl. 6 new wide-matrix SVD tests) | 280 | 280 | 0 |
| cjc-cana (untouched, regression check) | 144 | 144 | 0 |
| cjc-cana-nss (untouched) | 11+ | 11+ | 0 |
| AST/MIR parity gate (tests/fixtures) | 1 (over N fixtures) | 1 | 0 |

**Zero regressions to any pre-existing suite. Zero failures.**

---

## 1. What's done from the original quantum-inspired prompt

The "Quantum-Inspired CANA/NSS + Compression Layer" prompt from
`claude-quantum-cana-nss-prompt.md`:

| Prompt section | Status | Where |
|---|---|---|
| §1. CANA Compression Layer (4 kinds, semantic-critical guard) | ✅ shipped | `crates/cjc-cana-compress/src/{candidate,lossless_trace,motif_dictionary,lowrank,tensor_train,plan,report}.rs` |
| §2. Quantum-Inspired CANA Scoring (Ising/QAOA-style) | ✅ shipped | `crates/cjc-cana-compress/src/energy.rs` |
| §3. NSS Quantum-Inspired Pressure Summaries | ✅ shipped | `crates/cjc-nss/src/density.rs` |
| §4. CANA/NSS Bridge | ✅ shipped | `crates/cjc-cana-compress/src/bridge.rs` |
| Tests (unit, wiring, proptest, bolero, determinism) | ✅ shipped | `crates/cjc-cana-compress/{src/**/*.rs::tests,tests/}` + `crates/cjc-nss/src/density.rs::tests` |
| Verification loop documented | ✅ shipped | `docs/cana_compress/VERIFICATION_REPORT.md` |
| Documentation | ✅ shipped | `docs/cana_compress/{DESIGN,ARCHITECTURE,BLOG_NOTES}.md` |

**The "all four pieces" of the original prompt are done.** What's
*not* done is the **classical integration** that makes those
primitives actually improve the compiler — Phase A in the breakdown
below.

---

## 2. Phase A status — Classical Infrastructure

The Phase A breakdown was 6 items (A1-A6). Status:

| # | Item | Status | LOC | Tests |
|---|---|---|---:|---:|
| **A1** | Wire `EnergyRanker` into `PassRanker` (opt-in parallel ranker) | ✅ shipped | ~600 | 25 |
| **A2** | `derive_energy_components` pure function | ✅ shipped (lives inside A1's file) | folded into A1 | folded into A1 |
| **A3** | Hook `compress_pass_history` into `CanaReport` sidecar persistence | ✅ shipped | ~550 | 18 |
| **A4** | "Expected vs Actual" profile DB | ❌ pending | est. ~700 | est. 25 |
| **A5** | Wire `compression_pressure_delta` into `NssPressurePredictor` | ❌ pending | est. ~250 | est. 12 |
| **A6** | Benchmark harness emits training records | ❌ pending | est. ~400 | est. 15 |

### Detailed status — what A1 and A3 actually deliver

**A1 — `EnergyAwarePassRanker`** ([crates/cjc-cana-compress/src/energy_pass_ranker.rs](../../crates/cjc-cana-compress/src/energy_pass_ranker.rs)):

- Wraps any `cjc_cana::PassRanker<M, G>` + any `Box<dyn PressurePredictor>`.
- `rank()` runs the base ranker, then re-orders each function's
  `recommended` list by ascending energy via `EnergyRanker`.
- **Structural safety**: cannot drop a recommendation, cannot add
  one, cannot alter the legality verdict. The
  `debug_assert_eq!(reordered.len(), total)` is the canary.
- `derive_energy_components` is the pure mapping from
  `(PassRecommendation + FnFeatures + NSS pressure)` → 9-component
  `EnergyComponents`. Routes per-pass via `pass_benefit_channel` (dce/cse/licm
  → reuse_reward; everything else → fusion_reward) and `pass_code_size_factor`
  (loop_unroll=0.3, vectorize=0.15, etc.).
- 8 scaling knobs in `EnergyComponentsConfig`; the defaults are
  hand-tuned conservative weights.
- `audit()` method returns `(report, BTreeMap<fn_name, EnergyAuditEntry>)`
  for per-function energy-total inspection.
- **Not yet wired into `cjcl`.** It's a library; callers wrap their
  `PassRanker` explicitly. A future CLI flag `--energy-aware` is the
  obvious integration point.

**A3 — `CompressedCanaSidecar`** ([crates/cjc-cana-compress/src/sidecar.rs](../../crates/cjc-cana-compress/src/sidecar.rs)):

- Bundles `(CanaReport JSON bytes + compressed PassHistory + stable hashes)`.
- Magic header `"CCS0"` + `schema_version` field for forward compat.
- `build(&report, &history)` constructs.
- `to_bytes()` / `from_bytes()` for serialization.
- `write_to_path()` / `read_from_path()` for convenience.
- `verify()` re-checks every embedded hash in memory.
- `bundle_hash` covers the entire body and is appended last — single
  FNV-1a roundtrip verifies integrity at load.
- **Used as the at-rest format for training data.** A4 (profile DB)
  will reference the `bundle_hash` as a stable join key.

### What A4-A6 still need

**A4 — Expected vs Actual profile DB:**
- A new module `crates/cjc-cana/src/profile_db.rs` (or its own crate
  if it grows).
- Schema: `CompilationProfile { mir_hash, program_hash, sidecar_bundle_hash,
  energy_predicted, plan_chosen, runtime_observed, memory_peak_observed,
  thermal_observed, energy_actual }`.
- Storage: append-only file per benchmark, one
  `CompilationProfile` per row, each row compressed via
  `LosslessTracePayload`.
- The `bundle_hash` from A3 is the join key — given a profile row,
  you can look up the full `CompressedCanaSidecar` on disk and recover
  the `PassHistory` that produced the prediction.

**A5 — Wire compression-pressure delta into NssPressurePredictor:**
- Today `crates/cjc-cana-nss/src/lib.rs::NssPressurePredictor` synthesizes
  traces from features but never sees compression decisions. The bridge
  `compression_pressure_delta` exists but isn't called from this path.
- Change: extend `NssPressurePredictor::predict_thermal/memory/cpu` to
  optionally take a `CompressionReport`, and apply the bridge's
  pre-prediction memory/thermal delta.
- Why: makes the energy ranker's "thermal pressure" term sensitive to
  what compression has *already* relaxed.

**A6 — Benchmark harness emits training records:**
- Extend `bench/cana_train_cost_model/main.rs` (or a new
  `bench/cana_train_energy_components/`).
- For every (program × pass plan × measured cost) triple in the corpus,
  emit a `CompilationProfile` row.
- This becomes the labelled corpus for Phase B (training).

---

## 3. What's left for training (Phase B / C / D)

Once Phase A is done (A4-A6 land), the training pipeline becomes
straightforward:

**Phase B — Train the EnergyRanker:**
1. **B1**: Per-component regression on the profile DB. One linear
   model per `EnergyComponents` term: `actual_runtime_cost ←
   f(CanaFeatures)`, etc. Same shape as
   `cjc-cana::linear_cost_model::default_pass_coefficients`. Output: 9
   trained coefficient tables.
2. **B2**: Calibration — Pareto-scale the 9 weights against held-out
   benchmark ordering.
3. **B3**: Determinism canary — same corpus, same coefficients,
   byte-identical predictions across two runs.
4. **B4**: Bake the trained weights into `EnergyAwarePassRanker`
   defaults.

**Phase C — Train the NSS density predictions:**
1. **C1**: Multi-tick pressure trajectories — extend NSS to capture
   `PressureField` snapshots across compilation phases.
2. **C2**: Fit Pearson correlations from real workloads (the
   `PressureDensityState::from_trajectory` API already exists).
3. **C3**: Tune `collapse_risk` threshold against observed
   throttling/OOM events.
4. **C4**: New `NssDensityPredictor` impl of `PressurePredictor` that
   uses the trained correlations.

**Phase D — Integration:**
- `EnergyAwarePassRanker` becomes the default `PassRanker`.
- `NssDensityPredictor` replaces `NssPressurePredictor`.
- `compression_pressure_delta` is called automatically.

---

## 4. The Physics-Informed Neural Layer

The next session is adding a PINN layer to CANA and NSS. Read this
carefully because the architecture matters: **the PINN layer is
*additive*, not a replacement for the quantum-inspired pieces**.

### 4.1 Why PINN composes cleanly here

The existing primitives expose **traits** that take any implementor:

- `cjc_cana::CostModel` — used by `PassRanker`.
- `cjc_cana::PressurePredictor` — used by `EnergyAwarePassRanker` +
  `NssPressurePredictor`.
- `cjc_cana::LegalityGate` — final authority.

So a PINN layer doesn't have to touch any existing code — it ships as
**new trait implementations** that plug into the existing call sites.
Specifically:

```text
cjc-cana-pinn (new crate, parallel to cjc-cana-compress)
  ├── PinnCostModel: CostModel
  │     • a small MLP over CanaFeatures
  │     • physics-residual loss term: total-work conservation,
  │       arithmetic intensity laws, IEEE-754 reduction-order
  │       invariants
  │     • trained via cjc-ad::GradGraph + Adam (already in workspace)
  │
  └── PinnPressurePredictor: PressurePredictor
        • a small MLP over (CanaFeatures, structural hot kernels)
        • physics-residual loss: pressure conservation (already a
          first-class concept in cjc-nss::PressureGraph), thermal
          diffusion (Fourier law residual), queue dynamics
          (Little's law residual)
        • trained via cjc-ad::GradGraph + Adam
```

Both new implementors **coexist with** the quantum-inspired pieces:

- The existing `EnergyAwarePassRanker` takes
  `Box<dyn PressurePredictor>` — it can wrap `PinnPressurePredictor`
  as easily as `NssPressurePredictor` or `NullPressurePredictor`.
- The existing `LinearCostModel` can be the *fallback* for
  `PinnCostModel` — when the PINN's confidence is below threshold,
  fall back to the deterministic linear coefficients.
- The existing `PressureCorrelationSummary` is fed by *any* pressure
  trajectory, including PINN-generated ones.

### 4.2 Why "physics-informed" pays off here specifically

The compiler's domain has real physics:

- **Total work conservation**: a program's FLOPs are conserved across
  semantically-equivalent transformations. CSE doesn't reduce FLOPs,
  it reduces redundant computation. Constant-folding eliminates FLOPs
  by lifting them to compile time. A cost model whose loss includes
  `|FLOPs_predicted - FLOPs_actual|^2` plus
  `|FLOPs_before_pass - FLOPs_after_pass - FLOPs_eliminated|^2`
  enforces the conservation law structurally.
- **Arithmetic intensity laws** (Williams-Patterson roofline model):
  `peak_perf = min(peak_flops, peak_bandwidth * arithmetic_intensity)`.
  A predictor that respects this as a residual loss term cannot
  predict above-roofline performance for any pass.
- **Pressure conservation** (NSS-side): in
  `cjc_nss::PressureGraph`, every edge transforms `weight *
  source.magnitude` of pressure from source to target, with a small
  dissipation tax. A PINN pressure predictor should respect this
  global balance.
- **Thermal diffusion (Fourier law)**: the rate of thermal pressure
  change is proportional to the laplacian of the
  surrounding-functions' thermal pressure. A residual loss enforces
  this — preventing the predictor from learning spurious spatial
  spikes.
- **Queue dynamics (Little's law)**: `queue_length = arrival_rate *
  service_time`. A predictor whose loss includes Little's-law
  residuals cannot learn impossible queue scenarios.

These are real physical constraints — using them as soft losses (PINN
style) means the predictor's outputs respect the constraints by
construction, which makes the *non-physics* signal it learns more
trustworthy.

### 4.3 What NOT to do

- **Do not remove the quantum-inspired pieces.** The
  density-matrix-inspired pressure summary, the Ising-style energy
  ranker, the tensor-train compression — all of these are independent
  primitives that compose with PINN. Removing them would lose
  determinism guarantees (the energy ranker's tie-break, the density
  state's stable hash) that PINN doesn't replace.
- **Do not put PINN weights into the determinism contract.** The
  weights are *trained*; they change between releases. The compiler's
  output for a given (source, MIR, pass plan) must remain
  byte-identical. So PINN-generated `CostEstimate`s must be wrapped
  with a `model_version` field that's part of the report hash —
  identical only if the model is identical.
- **Do not train during normal compilation.** Training happens
  *offline* in the bench harness (Phase A6 / Phase B). The compiler
  itself only does inference. This preserves the "no nondeterministic
  runtime behavior" contract.
- **Do not make PINN authoritative.** Like every advisory layer, PINN
  outputs are *suggestions*. `cjc_cana::LegalityGate` and
  `cjc_mir::verifier` retain final authority.

### 4.4 Suggested PINN layer architecture

```text
crates/cjc-cana-pinn/    ← NEW satellite crate (mirrors cjc-cana-compress pattern)
  Cargo.toml             deps: cjc-cana + cjc-mir + cjc-nss + cjc-ad + cjc-repro
  src/
    lib.rs               crate docs + re-exports
    cost_model.rs        PinnCostModel: CostModel
    pressure.rs          PinnPressurePredictor: PressurePredictor
    residuals.rs         physics-residual loss components
    network.rs           the MLP architecture (built on cjc-ad::GradGraph)
    training.rs          offline training driver
    serialize.rs         trained-weight on-disk format (deterministic)
  tests/
    cost_model_wiring.rs PinnCostModel + PassRanker round-trip
    pressure_wiring.rs   PinnPressurePredictor + EnergyAwarePassRanker
    residuals.rs         each physical law's residual is finite + bounded
    determinism.rs       same weights + same input → byte-identical predictions
```

The crate is structured so it can be developed in parallel with
Phase A4-A6 — the trait implementations slot in immediately.

---

## 5. Reading order for the next session

In rough order of usefulness:

1. **This doc** (you're reading it).
2. [docs/cana_compress/DESIGN.md](DESIGN.md) — Phase 6 design rationale,
   especially the "quantum-inspired vs quantum-dependent" framing.
3. [docs/cana_compress/ARCHITECTURE.md](ARCHITECTURE.md) — the
   end-to-end data flow + crate dep graph.
4. [crates/cjc-cana-compress/src/energy_pass_ranker.rs](../../crates/cjc-cana-compress/src/energy_pass_ranker.rs) —
   the seam where any new `PressurePredictor` (including
   `PinnPressurePredictor`) plugs in.
5. [crates/cjc-cana-compress/src/sidecar.rs](../../crates/cjc-cana-compress/src/sidecar.rs) —
   the persistence format you'll persist PINN training data into.
6. [crates/cjc-cana/src/pressure.rs](../../crates/cjc-cana/src/pressure.rs) —
   the `PressurePredictor` trait surface you'll implement.
7. [crates/cjc-cana/src/cost_model.rs](../../crates/cjc-cana/src/cost_model.rs) —
   the `CostModel` trait surface you'll implement.
8. [crates/cjc-nss/src/pressure.rs](../../crates/cjc-nss/src/pressure.rs) —
   the `PressureField`/`PressureGraph` primitives the
   physics-residual loss terms operate over.
9. [crates/cjc-nss/src/density.rs](../../crates/cjc-nss/src/density.rs) —
   the existing density-matrix-inspired summary that PINN's
   trajectory outputs can feed into.
10. [crates/cjc-ad/src/](../../crates/cjc-ad/src/) — the GradGraph +
    Adam optimizer for offline training. Specifically:
    `dispatch.rs` for the `grad_graph_*` surface,
    `lib.rs` for `GradGraph::backward_collect` + `fit_with_adam`.
11. **CLAUDE.md** at the repo root — every prior session's invariants
    (determinism contract, no-FMA discipline, BTreeMap everywhere).

---

## 6. Suggested ordering for the next session

Given the user's plan is "PINN layer, then finish classical infra":

**Phase PINN (next session):**
1. Read the files in §5.
2. Create the `cjc-cana-pinn` crate scaffold (Cargo.toml + lib.rs +
   module stubs).
3. Implement `PinnCostModel` + `PinnPressurePredictor` as no-op
   implementations first (so the wiring tests pass before any actual
   network is trained).
4. Build the physics residuals one at a time:
   - Pressure conservation (NSS-side, easiest — already a first-class
     concept in `PressureGraph`).
   - Total work conservation (CANA-side, needs FLOPs estimation
     hooked into `CanaFeatures`).
   - Arithmetic intensity / roofline (CANA-side).
   - Thermal diffusion (NSS-side).
   - Little's law (NSS-side).
5. Build the MLP via `cjc-ad::GradGraph` + the existing
   `grad_graph_*` builtins.
6. Offline training driver (reads `CompressedCanaSidecar` files from
   A3, fits weights, writes a trained-weight bundle).
7. Wire `PinnCostModel` + `PinnPressurePredictor` into
   `EnergyAwarePassRanker` (just `Box::new(PinnPressurePredictor)` as
   the predictor).
8. Determinism canaries — same training data + same weights produce
   byte-identical predictions.
9. Docs (`docs/cana_pinn/{DESIGN,ARCHITECTURE,BLOG_NOTES}.md`).

**Phase A finish (after PINN):**
1. A4 — Profile DB (the schema where PINN training data lives).
2. A5 — `compression_pressure_delta` wired into
   `NssPressurePredictor`.
3. A6 — Benchmark harness emits training records (uses A3's sidecar
   format + A4's DB).
4. B1-B4 — Train the EnergyRanker per-component weights.
5. C1-C4 — Train the NSS density predictions.
6. D — Integration into production.

---

## 7. Open questions / decisions for the next session

These are not blockers, but they're decisions the next session will
have to make explicitly:

1. **Where does `cjc-cana-pinn` live in the workspace?** Satellite
   crate like `cjc-cana-compress`, or a feature flag inside an
   existing crate? **Recommendation: satellite crate**, same isolation
   reasoning as before (PINN's training dep on `cjc-ad` stays out of
   the compiler driver).

2. **Trained-weight persistence format?** Reuse
   `LosslessTracePayload`? Build a dedicated `PinnWeightsBundle`?
   **Recommendation: build on `LosslessTracePayload`** for the byte
   layer, add a `PinnWeightsBundle` type alias with `magic = "CPB0"`
   (CANA PINN Bundle v0). Match the `CompressedCanaSidecar` pattern.

3. **Model versioning in the report hash?** Today
   `CompressionReport.report_hash` is content-addressed over byte-
   identical reports. When PINN-trained predictions become part of
   the report, the model version must be in the hash so two
   compilations with different trained weights produce different
   hashes. **Recommendation: add `model_id: String` to
   `EnergyComponents` (defaulting to `"manual_weights_v1"`), include
   it in the canonical bytes.**

4. **Physics residuals: hard constraints (Lagrangian) or soft losses
   (weighted MSE)?** The PINN literature uses soft losses by default.
   For a compiler we might want hard constraints on some laws (total
   work conservation must hold exactly) and soft on others (thermal
   diffusion can be approximate). **Recommendation: soft losses for
   v1**, defer the Lagrangian formulation to v2.

5. **How much of `cjc-ad` to reuse vs build fresh?**
   The `grad_graph_*` builtins were designed for the PINN-in-pure-CJC
   demos. They support the full forward/backward needed. **Recommendation:
   reuse `cjc-ad::GradGraph` end-to-end** — it's
   already the workspace's PINN substrate.

---

## 8. Files written this session

Source:
```
crates/cjc-cana-compress/
  Cargo.toml                            crate manifest (deps: cjc-cana, cjc-mir, cjc-nss, cjc-quantum, cjc-runtime, cjc-repro)
  src/
    lib.rs                              re-exports + crate-level docs
    candidate.rs                        CompressionCandidate + CompressionKind + Criticality
    lossless_trace.rs                   byte-RLE + PassHistory adapter
    motif_dictionary.rs                 deterministic LZ77
    lowrank.rs                          power-iteration truncated SVD (advisory)
    tensor_train.rs                     TT-SVD via cjc-quantum::mps
    plan.rs                             CompressionPlan + executor + lossy payload codecs
    report.rs                           CompressionReport + ReportHash + JSON
    energy.rs                           Ising-style EnergyRanker + EnergyComponents
    bridge.rs                           CANA → NSS pressure delta
    energy_pass_ranker.rs               EnergyAwarePassRanker (A1)
    sidecar.rs                          CompressedCanaSidecar (A3)
  tests/
    wiring.rs                           end-to-end CANA compression pipeline
    proptest_compress.rs                proptest properties
    bolero_fuzz.rs                      bolero fuzz targets
    determinism.rs                      byte-identical double-run canaries
    energy_pass_ranker_wiring.rs        A1 wiring tests

crates/cjc-nss/src/density.rs           PressureDensityState + PressureCorrelationSummary
crates/cjc-quantum/src/mps.rs           wide-matrix routing + 6 regression tests (the upstream SVD fix)

Workspace:
  Cargo.toml                            workspace member + workspace dep entries
```

Docs:
```
docs/cana_compress/
  DESIGN.md                             10-section design doc
  ARCHITECTURE.md                       ASCII flow diagram + crate dep graph
  BLOG_NOTES.md                         blog source material
  VERIFICATION_REPORT.md                test counts + verification command list
  PHASE_A_HANDOFF.md                    this doc
```

Memory:
```
~/.claude/projects/C--Users-adame-CJC/memory/
  project_cana_compress.md              project memory note (created, then updated 3x)
  MEMORY.md                             index entry under "CANA Phase 6"
```

---

## 9. Determinism invariants the next session must preserve

The whole house of cards depends on these. Don't violate them; PINN
training does NOT exempt you from them:

1. **`BTreeMap` / `BTreeSet` / sorted `Vec` everywhere in decision paths.**
2. **All FP reductions via `cjc_repro::KahanAccumulatorF64` or
   `BinnedAccumulator`.** Naive `iter().sum::<f64>()` is forbidden.
3. **No FMA contraction.** Write `a * b + c` as `let t = a * b; t + c;`.
4. **All RNG via `cjc_repro::Rng` (SplitMix64) with explicit seed
   threading.** Never `rand::thread_rng()`, `Instant::now()`, or any
   OS entropy.
5. **All hashing via `cjc_cana::CanaHasher` (FNV-1a) or
   `cjc_repro::hash_bytes`.** Never `std::collections::hash_map::DefaultHasher`.
6. **f64 sorting via `f64::total_cmp`.** Never `partial_cmp().unwrap()`.
7. **No wall-clock or OS thermal sensor in decision outputs.**
8. **The AST/MIR parity gate (`tests/fixtures`) MUST pass after every
   change.** This is the load-bearing CI gate.

---

*Generated 2026-06-09 as the closing artifact of the session that
shipped the Phase 6 quantum-inspired CANA/NSS compression layer +
SVD fix + Phase A1 + Phase A3. Next session: add PINN.*
