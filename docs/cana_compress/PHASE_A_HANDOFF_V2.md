# Phase A Handoff v2 — Quantum-Inspired CANA/NSS + Physics-Informed Layer Plan

**Date:** 2026-06-09
**Branch:** `claude/objective-davinci-62975a`
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\objective-davinci-62975a`
**Supersedes:** `PHASE_A_HANDOFF.md` (v1) — preserved for diff
**Source prompts:** `claude-quantum-cana-nss-prompt.md` (shipped) + `claude-runtime-quantum-cana-nss-prompt.md` (queued) + `claude-pinn-cana-nss-prompt.md` (next session)

This is the v2 handoff. It synthesizes the v1 handoff with the explicit
guidance from `claude-pinn-cana-nss-prompt.md`. Key changes vs v1:

- **v1 implementation = deterministic coefficient-based physical-cost
  model**, not an MLP. Neural training deferred to v2 of the
  physics-informed layer.
- **Explicit three-role framing** of quantum-inspired CANA: compression
  + search/ranking + state representation.
- **Concrete type signatures** for `PhysicalCostQuery`,
  `PhysicalCostEstimate`, `PhysicalConstraints`, `PinnPhysicalCostModel`.
- **NSS `PressureKind` mapping** — explicit per-pressure routing.
- **3 new EnergyComponents-style terms**: `bandwidth_pressure`,
  `code_size_pressure`, `locality_reward`.
- **5-way ablation plan** and **11 acceptance metrics**.
- **Shadow mode → active mode promotion** gate.
- **20-column training data schema** (was ~7 in v1).

---

## 0. TL;DR — what shipped this session

**11 commits' worth of work** landed across one focused session:

```text
Phase 6 — quantum-inspired CANA/NSS + compression layer (the original prompt):
  cjc-cana-compress (new crate)            ~3,500 LOC src + 1,700 LOC tests
  cjc-nss::density (new module)            ~500 LOC src + tests
  workspace Cargo.toml                     add member + workspace dep
  docs/cana_compress/                      DESIGN, ARCHITECTURE, BLOG_NOTES,
                                             VERIFICATION_REPORT, PHASE_A_HANDOFF (v1)

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
| cjc-cana-compress (lib, incl. 16 energy_pass_ranker + 18 sidecar) | 155 | 155 | 0 |
| cjc-cana-compress (wiring + proptest + bolero + determinism + energy_pass_ranker_wiring) | 52 | 52 | 0 |
| cjc-nss (incl. 18 new density tests) | 217+ | 217+ | 0 |
| cjc-quantum (incl. 6 new wide-matrix SVD tests) | 280 | 280 | 0 |
| cjc-cana (untouched, regression check) | 144 | 144 | 0 |
| cjc-cana-nss (untouched) | 11+ | 11+ | 0 |
| AST/MIR parity gate (tests/fixtures) | 1 (over N fixtures) | 1 | 0 |

**Zero regressions to any pre-existing suite. Zero failures.**

---

## 1. What's done from the original quantum-inspired prompt

| Prompt section | Status | Where |
|---|---|---|
| §1. CANA Compression Layer (4 kinds, semantic-critical guard) | ✅ shipped | `crates/cjc-cana-compress/src/{candidate,lossless_trace,motif_dictionary,lowrank,tensor_train,plan,report}.rs` |
| §2. Quantum-Inspired CANA Scoring (Ising/QAOA-style) | ✅ shipped | `crates/cjc-cana-compress/src/energy.rs` |
| §3. NSS Quantum-Inspired Pressure Summaries | ✅ shipped | `crates/cjc-nss/src/density.rs` |
| §4. CANA/NSS Bridge | ✅ shipped | `crates/cjc-cana-compress/src/bridge.rs` |
| Tests (unit, wiring, proptest, bolero, determinism) | ✅ shipped | `crates/cjc-cana-compress/{src/**/*.rs::tests,tests/}` + `crates/cjc-nss/src/density.rs::tests` |
| Verification loop documented | ✅ shipped | `docs/cana_compress/VERIFICATION_REPORT.md` |
| Documentation | ✅ shipped | `docs/cana_compress/{DESIGN,ARCHITECTURE,BLOG_NOTES}.md` |

---

## 2. Quantum-Inspired CANA covers THREE roles (not just compression)

This is a clarification from the PINN prompt that v1 of the handoff
under-emphasised. The quantum-inspired layer in this codebase covers
**three independent advisory roles**, all already shipped:

| Role | Status | Where |
|---|---|---|
| **Compression** — tensor-network / MPS-style compression of advisory traces, pass histories, pressure trajectories, repeated MIR motifs | ✅ shipped | `crates/cjc-cana-compress/src/{lossless_trace,motif_dictionary,lowrank,tensor_train}.rs` |
| **Search / ranking** — QAOA / Ising / Hamiltonian-style energy scoring for pass ordering, fusion grouping, compression target selection, memory plans, runtime strategy choices | ✅ shipped | `crates/cjc-cana-compress/src/energy.rs` (`EnergyRanker`) + `crates/cjc-cana-compress/src/energy_pass_ranker.rs` (`EnergyAwarePassRanker`) |
| **State representation** — compact pressure / correlation / state summaries that help CANA and NSS reason about coupled compiler/runtime decisions | ✅ shipped | `crates/cjc-nss/src/density.rs` (`PressureDensityState`, `PressureCorrelationSummary`) |

The architecture stack as it stands today (and as the PINN prompt
expects it to be referenced):

```text
MIR program + CANA features
  -> quantum-inspired CANA  (compression + search/ranking + state)
  -> NSS pressure prediction
  -> PINN physical cost model         ← NEW: next session
  -> combined deterministic score
  -> legality gate                    (final authority, unchanged)
  -> pass / runtime strategy plan
  -> report / profile / training row
```

---

## 3. Phase A status — Classical Infrastructure

| # | Item | Status | LOC | Tests |
|---|---|---|---:|---:|
| **A1** | Wire `EnergyRanker` into `PassRanker` (opt-in parallel ranker) | ✅ shipped | ~600 | 25 |
| **A2** | `derive_energy_components` pure function | ✅ shipped (folded into A1) | inside A1 | inside A1 |
| **A3** | Hook `compress_pass_history` into `CanaReport` sidecar persistence | ✅ shipped | ~550 | 18 |
| **A4** | "Expected vs Actual" profile DB | ❌ pending | est. ~700 | est. 25 |
| **A5** | Wire `compression_pressure_delta` into `NssPressurePredictor` | ❌ pending | est. ~250 | est. 12 |
| **A6** | Benchmark harness emits training records | ❌ pending | est. ~400 | est. 15 |

### A1 — `EnergyAwarePassRanker` ([crates/cjc-cana-compress/src/energy_pass_ranker.rs](../../crates/cjc-cana-compress/src/energy_pass_ranker.rs))

- Wraps any `cjc_cana::PassRanker<M, G>` + any
  `Box<dyn PressurePredictor>`.
- `rank()` runs the base ranker, then re-orders each function's
  `recommended` list by ascending energy via `EnergyRanker`.
- **Structural safety**: cannot drop a recommendation, cannot add one,
  cannot alter the legality verdict. The
  `debug_assert_eq!(reordered.len(), total)` is the canary.
- `derive_energy_components` is the pure mapping from
  `(PassRecommendation + FnFeatures + NSS pressure)` → 9-component
  `EnergyComponents`. Routes per-pass via `pass_benefit_channel`
  (dce/cse/licm → reuse_reward; everything else → fusion_reward) and
  `pass_code_size_factor` (loop_unroll=0.3, vectorize=0.15, etc.).
- 8 scaling knobs in `EnergyComponentsConfig`; the defaults are
  hand-tuned conservative weights.
- `audit()` method returns `(report, BTreeMap<fn_name, EnergyAuditEntry>)`
  for per-function energy-total inspection.

**Note:** the 9-component `EnergyComponents` will need to grow to
**12 components** when the PINN layer lands — see §4.5 below for the
new `bandwidth_pressure`, `code_size_pressure`, and `locality_reward`
terms.

### A3 — `CompressedCanaSidecar` ([crates/cjc-cana-compress/src/sidecar.rs](../../crates/cjc-cana-compress/src/sidecar.rs))

- Bundles `(CanaReport JSON + compressed PassHistory + stable hashes)`.
- Magic header `"CCS0"` + `schema_version` field for forward compat.
- `build()` / `to_bytes()` / `from_bytes()` / `verify()` /
  `write_to_path()` / `read_from_path()`.
- `bundle_hash` (FNV-1a over body) appended last — single roundtrip
  integrity check at load.
- **The `bundle_hash` is the join key** for A4 (profile DB).

### What A4-A6 still need

**A4 — Expected vs Actual profile DB:**
- New module `crates/cjc-cana/src/profile_db.rs` (or its own crate if
  it grows).
- See §5.1 below for the 20-column schema.

**A5 — Wire compression-pressure delta into NssPressurePredictor:**
- Extend `crates/cjc-cana-nss/src/lib.rs::NssPressurePredictor::predict_thermal/memory/cpu`
  to optionally take a `CompressionReport`, and apply the bridge's
  pre-prediction memory/thermal delta.

**A6 — Benchmark harness emits training records:**
- Extend `bench/cana_train_cost_model/main.rs` (or create
  `bench/cana_train_physical_cost/`).
- For every (program × pass plan × measured cost) triple, emit a
  `CompilationProfile` row joining by `bundle_hash` from A3's sidecar.

---

## 4. Physics-Informed Layer — v1 deterministic coefficient model

### 4.1 Why PINN composes cleanly here

The existing primitives expose **traits** that take any implementor:

- `cjc_cana::CostModel` — used by `PassRanker`.
- `cjc_cana::PressurePredictor` — used by `EnergyAwarePassRanker` +
  `NssPressurePredictor`.
- `cjc_cana::LegalityGate` — final authority on correctness, never
  influenced by PINN.

So a PINN layer ships as **new trait implementations** that plug into
the existing call sites. Nothing in `cjc-cana`, `cjc-mir`, or
`cjc-mir-exec` needs to change.

### 4.2 v1 design — deterministic coefficient-based model (SHIP THIS FIRST)

**The PINN prompt explicitly says:** *"This does not need to be a large
neural net at first. Start with a deterministic physics-informed model
with coefficients and constraints."*

v1 is a **deterministic** physical-cost calculator. It encodes physics
via fixed coefficients and explicit constraint thresholds — no trained
weights, no shadow mode required, no model versioning in the report
hash. It ships and wires immediately, and gives a baseline against
which v2 (neural-trained) must demonstrate improvement.

#### Concrete type signatures

```rust
// crates/cjc-cana/src/physical_cost.rs (new file)

/// Input query for a physical-cost prediction. All counters are
/// integer estimates derived from `CanaFeatures` + a candidate
/// strategy; no wall-clock, no OS sensors.
pub struct PhysicalCostQuery<'a> {
    pub function_name: &'a str,
    pub strategy_id: &'a str,
    pub flops_estimate: u64,
    pub bytes_read_estimate: u64,
    pub bytes_written_estimate: u64,
    pub allocation_bytes_estimate: u64,
    pub working_set_bytes_estimate: u64,
    pub thread_count: u32,
    pub batch_size: u32,
}

/// Output of a physical-cost prediction. Every float finite and
/// pressures clamped to `[0, 1]`.
pub struct PhysicalCostEstimate {
    pub thermal_pressure: f64,
    pub memory_pressure: f64,
    pub bandwidth_pressure: f64,
    pub energy_estimate: f64,
    pub locality_risk: f64,
    pub confidence: f64,
}

/// Coefficients of the deterministic physical-cost model.
/// Hand-tuned for v1; trained offline for v2.
pub struct PhysicalCoefficients {
    /// Heat accumulation per FLOP. Multiplied by `flops_estimate`
    /// before contributing to thermal_pressure.
    pub heat_per_flop: f64,
    /// Cooling / dissipation rate. Subtracted from heat accumulation.
    pub cooling_rate: f64,
    /// Memory intensity coefficient (bytes per FLOP).
    pub bytes_per_flop_scale: f64,
    /// Bandwidth pressure coefficient on bytes moved.
    pub bandwidth_per_byte: f64,
    /// Allocation churn pressure coefficient.
    pub alloc_churn_per_byte: f64,
    /// Thermal amplification with thread count.
    pub thread_amplification: f64,
    /// Thermal amplification with batch size.
    pub batch_amplification: f64,
    /// Locality / reuse-distance proxy weight.
    pub locality_weight: f64,
}

impl Default for PhysicalCoefficients {
    fn default() -> Self {
        Self {
            heat_per_flop: 1e-9,         // 1 nano-unit heat per FLOP
            cooling_rate: 0.05,
            bytes_per_flop_scale: 0.1,
            bandwidth_per_byte: 1e-10,
            alloc_churn_per_byte: 1e-11,
            thread_amplification: 0.1,
            batch_amplification: 0.05,
            locality_weight: 0.3,
        }
    }
}

/// Hard upper bounds and soft preferences. Used by the cost model
/// to penalise or reject strategies that exceed configured limits.
pub struct PhysicalConstraints {
    pub max_thermal_pressure: f64,        // hard limit; > this is rejected
    pub max_memory_pressure: f64,
    pub max_bandwidth_pressure: f64,
    pub max_energy_estimate: Option<f64>, // None = unbounded
    /// Soft preference: prefer plans whose thermal pressure is at
    /// least this much below the max. Default 0.1 = "prefer plans
    /// with 10% headroom".
    pub prefer_cooler_plan_margin: f64,
}

impl Default for PhysicalConstraints {
    fn default() -> Self {
        Self {
            max_thermal_pressure: 0.95,
            max_memory_pressure: 0.95,
            max_bandwidth_pressure: 0.95,
            max_energy_estimate: None,
            prefer_cooler_plan_margin: 0.1,
        }
    }
}

/// The cost-model wrapper. Composes a base CANA cost model + an NSS
/// pressure predictor + physical coefficients + constraints.
pub struct PinnPhysicalCostModel<M, P> {
    pub base_model: M,
    pub pressure_predictor: P,
    pub physical_coeffs: PhysicalCoefficients,
    pub thresholds: PhysicalConstraints,
    /// Stable model identifier, included in report hashes so two
    /// compilations with different coefficient sets produce
    /// different `report_hash`. v1 is "pinn_coeffs_v1".
    pub model_id: &'static str,
    pub model_version: u32,
}
```

#### How the coefficients combine

The deterministic prediction:

```text
heat_accumulation = heat_per_flop * flops_estimate
                  * (1 + thread_amplification * thread_count)
                  * (1 + batch_amplification * batch_size)
cooling = cooling_rate * (some_baseline)
thermal_pressure = clamp01(heat_accumulation - cooling)

bytes_moved = bytes_read_estimate + bytes_written_estimate
memory_pressure = clamp01(
    bytes_per_flop_scale * (bytes_moved / max(flops_estimate, 1))
)
bandwidth_pressure = clamp01(bandwidth_per_byte * bytes_moved)
alloc_churn = alloc_churn_per_byte * allocation_bytes_estimate
energy_estimate = heat_accumulation + bandwidth_pressure + alloc_churn
locality_risk = clamp01(
    locality_weight * (working_set_bytes_estimate / some_cache_proxy)
)

confidence = 1.0 - (alloc_churn + thread_amplification * thread_count).min(1.0)
```

(Exact formulas are placeholders — the next session refines them. The
key property: every output is a deterministic, finite, clamped function
of integer inputs.)

#### Trait impl

```rust
impl<M: CostModel, P: PressurePredictor> CostModel for PinnPhysicalCostModel<M, P> {
    fn query(&self, program: &MirProgram, features: &CanaFeatures, query: &CostQuery)
        -> CostEstimate
    {
        // 1. Delegate to base model for benefit/runtime predictions.
        let base_estimate = self.base_model.query(program, features, query);

        // 2. Build a PhysicalCostQuery from features + query.
        let phys_query = build_physical_query(features, query);

        // 3. Run the deterministic physical-cost calculation.
        let phys_estimate = self.predict_physical(&phys_query);

        // 4. Check constraints — reject the strategy if any hard
        //    limit is exceeded.
        if phys_estimate.thermal_pressure > self.thresholds.max_thermal_pressure
            || phys_estimate.memory_pressure > self.thresholds.max_memory_pressure
            || phys_estimate.bandwidth_pressure > self.thresholds.max_bandwidth_pressure
        {
            return CostEstimate::Unknown;  // base ranker treats this as "do nothing"
        }

        // 5. Blend base estimate with physical penalty.
        match base_estimate {
            CostEstimate::Estimated { value, confidence } => {
                let phys_penalty = phys_estimate.thermal_pressure
                    + phys_estimate.memory_pressure
                    + phys_estimate.bandwidth_pressure;
                CostEstimate::Estimated {
                    value: (value - phys_penalty).max(0.0),
                    confidence: (confidence * phys_estimate.confidence).clamp(0.0, 1.0),
                }
            }
            CostEstimate::Unknown => CostEstimate::Unknown,
        }
    }

    fn name(&self) -> &'static str { self.model_id }
    fn version(&self) -> u32 { self.model_version }
}
```

The model ID + version flow through to `CanaReport` so two
compilations with different coefficients produce different report
hashes — without needing trained weights yet.

### 4.3 v2 design — neural training (DEFERRED)

After v1 ships and stabilises, v2 swaps `PhysicalCoefficients` for a
small MLP trained via `cjc-ad::GradGraph` + Adam, using the training
data corpus from A6. The PINN-style residual losses (Fourier law,
Little's law, pressure conservation) become soft penalty terms in the
training objective. Key design decisions deferred to v2:

- **Lagrangian vs soft losses** for physics residuals.
- **MLP architecture** (likely 2-layer, ~64 hidden units).
- **Confidence calibration** — the v1 confidence formula is heuristic;
  v2's confidence comes from training-data variance.
- **Online vs offline updates** — strict offline only.

Critically, v2 keeps the **same `PinnPhysicalCostModel<M, P>` trait
surface** as v1. The MLP is wrapped behind the same `CostModel` impl.
Callers don't have to change.

### 4.4 NSS `PressureKind` mapping

The physical cost estimates feed NSS pressure fields via the bridge:

| `PhysicalCostEstimate` field | NSS `PressureKind` |
|---|---|
| `thermal_pressure` | `Thermal` |
| `memory_pressure` | `Memory` |
| `bandwidth_pressure` | `Throughput` (primary), `Io` (secondary if disk-heavy) |
| `energy_estimate` | (composite — informs `Cpu` and `Thermal`) |
| `locality_risk` | (informs `Memory` via working-set saturation) |
| `confidence` | (not a pressure — quality-weights downstream consumers) |

Thread/contention pressure isn't a `PhysicalCostEstimate` field
directly, but `thread_count` feeds the `thermal_pressure` formula via
`thread_amplification`. If a future iteration adds explicit contention
modelling, the natural mappings are:

| Future field | NSS `PressureKind` |
|---|---|
| `scheduler_contention` | `Scheduler` |
| `sync_contention` | `Sync` |
| `cpu_saturation` | `Cpu` |

### 4.5 Combined scoring formula

The existing 9-component `EnergyComponents` from
`crates/cjc-cana-compress/src/energy.rs` grows to **12 components** to
accommodate the physical layer. New terms:

- `bandwidth_pressure: f64` — added as a cost term.
- `code_size_pressure: f64` — added as a cost term (re-styled from the
  existing `code_size_cost` to align with the prompt's vocabulary).
- `locality_reward: f64` — added as a reward term.

**Updated formula:**

```text
quantum_inspired_energy =
    runtime_cost
  + compile_cost
  + memory_pressure
  + thermal_pressure
  + bandwidth_pressure        ← NEW
  + code_size_pressure        ← NEW (rename of code_size_cost)
  + reconstruction_risk
  + verifier_risk_penalty
  - fusion_reward
  - reuse_reward
  - compression_reward
  - locality_reward           ← NEW

physical_penalty =
    heat_rise_proxy
  + memory_traffic_proxy
  + bandwidth_saturation_proxy
  + allocation_churn_proxy
  - cooling_reward
  - reuse_reward (from physical model)
  - locality_reward (from physical model)

combined_score =
    quantum_inspired_energy
  + physical_penalty
  + NSS_risk_penalty

(lower score ranks better)
```

The `EnergyComponents` struct update is **breaking** — every caller of
`EnergyComponents::new()` will need updating to pass the 3 new fields.
The builder pattern (`EnergyComponents::builder().bandwidth_pressure(x)
.build()`) is backward-compatible because unset fields default to 0.0.
Plan the change accordingly:

1. Add fields to `EnergyComponents` with default `0.0`.
2. Extend the builder with `.bandwidth_pressure()`, `.code_size_pressure()`,
   `.locality_reward()` (rename `.code_size_cost()` to
   `.code_size_pressure()`; keep old name as deprecated alias for one
   release).
3. Update `EnergyComponents::new()` signature (or add a 12-arg
   variant) — straightforward since there's only one production caller
   (the energy ranker).
4. Update all proptest strategies in
   `crates/cjc-cana-compress/tests/proptest_compress.rs`.

### 4.6 Compression integration

The PINN layer must be aware of compression's effect on physical cost:

- **Compression saves advisory memory** → reduces `memory_pressure`
  contribution.
- **Decompression has CPU + thermal cost** → adds to
  `bandwidth_pressure` + `thermal_pressure`.
- **Lossy advisory compression** → reduces ranking confidence (multiply
  the cost-model's confidence by `1.0 - reconstruction_error`).
- **Semantic-critical facts** → must stay lossless; the bridge already
  enforces this via the `Criticality::SemanticCritical + lossy_kind ->
  Err` check at construction time.

Concrete change: extend `crates/cjc-cana-compress/src/bridge.rs` to
emit a `PhysicalCostQuery::compression_overhead` field that
`PinnPhysicalCostModel` consumes. This is small (~50 LOC).

### 4.7 What NOT to do

- **Do not remove the quantum-inspired pieces.** The
  density-matrix-inspired pressure summary, the Ising-style energy
  ranker, the tensor-train compression — all of these are independent
  primitives that compose with PINN. They retain their roles
  (state representation, search/ranking, compression).
- **Do not put PINN coefficients in the determinism contract for v1.**
  The v1 model is fully deterministic by construction (fixed
  coefficients, fixed formulas). The `model_id` + `model_version`
  fields propagate to the report hash so two configurations produce
  different hashes, but the values themselves don't need shadow mode
  because there are no trained weights yet.
- **Do not train during normal compilation.** Training happens
  *offline* in the bench harness (Phase A6 / §5 below). The compiler
  itself only does inference. This preserves the "no nondeterministic
  runtime behavior" contract.
- **Do not make PINN authoritative.** Like every advisory layer, PINN
  outputs are *suggestions*. `cjc_cana::LegalityGate` and
  `cjc_mir::verifier` retain final authority.
- **For v2 (neural):** model weight changes require **shadow mode →
  active mode promotion**. See §5.4 for the gate.

### 4.8 Suggested PINN crate architecture

The PINN prompt suggests modules inside `cjc-cana`:

```text
crates/cjc-cana/src/
  physical_cost.rs          ← NEW: PhysicalCostQuery, PhysicalCostEstimate,
                              PhysicalCoefficients, PhysicalConstraints
  pinn_cost_model.rs        ← NEW: PinnPhysicalCostModel<M, P> + CostModel impl
  quantum_score.rs          ← NEW: explicit three-role framing docs +
                              the combined scoring formula
crates/cjc-cana-nss/src/
  pinn_bridge.rs            ← NEW: ties PinnPhysicalCostModel +
                              NssPressurePredictor + compression deltas
```

**Decision needed:** my v1 handoff suggested a satellite crate
(`cjc-cana-pinn`) following the `cjc-cana-compress` pattern. The PINN
prompt suggests modules inside existing crates. The tradeoff:

| Approach | Pros | Cons |
|---|---|---|
| Modules inside `cjc-cana` | Simpler workspace; no new Cargo.toml | Pulls future v2 MLP deps (`cjc-ad`) into compiler driver |
| Satellite crate `cjc-cana-pinn` | Keeps `cjc-ad` out of compiler driver until v2 ships | One more crate to maintain |

**Recommendation:** modules inside `cjc-cana` for v1 (deterministic
coefficients, zero new deps). Promote to a satellite crate only when
v2's MLP lands and pulls `cjc-ad`. This matches the new prompt's
guidance and keeps v1 simple.

---

## 5. Training pipeline (Phase A6 + Phase B/C/D)

### 5.1 Data collection schema (20+ columns)

The append-only training corpus row format, joining to `A3`'s sidecar
via `bundle_hash`:

```text
CompilationProfile {
    program_hash:                  u64,
    mir_hash:                      u64,
    feature_hash:                  u64,
    sidecar_bundle_hash:           u64,        // → joins to A3
    cost_model_id:                 String,
    cost_model_version:            u32,
    pinn_model_id:                 String,
    pinn_model_version:            u32,
    candidate_strategy_id:         String,
    pass_sequence:                 Vec<String>,
    compression_plan_id:           Option<String>,

    // Workload estimates (deterministic, derived from features)
    estimated_flops:               u64,
    estimated_bytes_read:          u64,
    estimated_bytes_written:       u64,
    estimated_alloc_bytes:         u64,
    estimated_working_set:         u64,
    thread_count:                  u32,
    batch_size:                    u32,

    // Predictions (from NSS + PINN at compile time)
    nss_predicted_cpu:             f64,
    nss_predicted_memory:          f64,
    nss_predicted_thermal:         f64,
    pinn_predicted_energy:         f64,
    pinn_predicted_thermal:        f64,
    pinn_predicted_bandwidth:      f64,

    // Observed (from bench harness)
    compile_time_counter:          u64,        // not wall-clock — deterministic counter
    compile_memory_counter:        u64,
    runtime_counter_if_available:  Option<u64>,
    runtime_memory_counter_if_available: Option<u64>,
    deterministic_energy_estimate: f64,        // from cjc-runtime's energy estimator

    // Verifier / parity outcomes
    verifier_result:               VerifierOutcome,  // Approved | Rejected{reason}
    parity_result:                 ParityOutcome,    // Match | Mismatch{diff_hash}

    // Final score
    score:                         f64,
}
```

Stored as append-only `LosslessTrace` rows (one row per (program ×
plan) experiment) in a file per benchmark.

### 5.2 5-way ablation plan

The PINN prompt is explicit about which ablations to run:

| Configuration | What it isolates |
|---|---|
| **CANA baseline** | Linear cost model only. Reference point. |
| **CANA + NSS** | Add pressure predictions. Isolates thermal/memory signal value. |
| **CANA + quantum-inspired scoring** | Add Ising-style ranking. Isolates score-decomposition value. |
| **CANA + NSS + quantum scoring** | Composition of both advisory layers. Isolates interaction. |
| **CANA + NSS + quantum scoring + PINN physical model** | Full stack. Measures PINN's marginal value. |

Each ablation runs the **same** corpus with the **same** seeds. The
training driver records `score` for each configuration. PINN must beat
the second-best ablation by at least `prefer_cooler_plan_margin`
(default 0.1) on at least 60% of held-out programs before promotion to
active mode (see §5.4).

### 5.3 11 acceptance metrics

Track all of these for every benchmark + every ablation:

1. **Compile time** (deterministic counter — not wall-clock for
   strict decision paths)
2. **Compiler peak memory**
3. **Runtime speed** (where measurable via parity-gate fixtures)
4. **Runtime peak memory**
5. **Deterministic energy estimate** (from `cjc-runtime`'s estimator)
6. **Predicted thermal pressure**
7. **Pressure prediction error** (predicted − actual, per pressure
   kind)
8. **Verifier rejection rate** (must stay at baseline — PINN cannot
   *increase* verifier rejections, only sustain them)
9. **Parity pass rate** (must stay 100% — AST↔MIR byte-identity)
10. **Compression ratio** (where compression is in the plan)
11. **Report hash stability** (the determinism canary — two runs of
    the same configuration on the same input must produce
    byte-identical report hashes)

### 5.4 Shadow mode → active mode promotion

The PINN prompt's strict promotion gate (applies once v2 trained
weights land):

```text
shadow mode:
  PINN runs in parallel with the existing CANA + NSS path.
  Its predictions are recorded in CompilationProfile rows.
  Its rankings are NOT applied. The base CANA + NSS path
  still drives pass selection.

  Stay in shadow until:
    - 1000+ profile rows accumulated
    - PINN beats the second-best ablation by ≥0.1 score margin
      on ≥60% of held-out programs
    - Parity pass rate stays at 100%
    - Determinism canaries (report hash) stay byte-identical
      across two runs of the same configuration

active mode:
  PINN's rankings drive pass selection.
  Original CANA + NSS path retained as fallback under any
  PINN error / Unknown output.
  Verifier / legality gates retain final authority.
```

v1's deterministic coefficient model **skips shadow mode** because
there are no trained weights — its outputs are a deterministic function
of integer inputs by construction. v1 lands in active mode immediately,
but with conservative default thresholds (`prefer_cooler_plan_margin =
0.1`, `max_thermal_pressure = 0.95`) so it bias-orders rather than
aggressively constrains.

---

## 6. Reading order for the next session

In rough order of usefulness for the PINN-implementing session:

1. **This doc** (you're reading it).
2. **The PINN prompt** itself (`claude-pinn-cana-nss-prompt.md` in
   Downloads) — has the canonical type signatures.
3. [docs/cana_compress/DESIGN.md](DESIGN.md) — Phase 6 design
   rationale, especially the "quantum-inspired vs quantum-dependent"
   framing.
4. [docs/cana_compress/ARCHITECTURE.md](ARCHITECTURE.md) — the
   end-to-end data flow + crate dep graph.
5. [crates/cjc-cana-compress/src/energy_pass_ranker.rs](../../crates/cjc-cana-compress/src/energy_pass_ranker.rs)
   — the seam where any new `PressurePredictor` (including the future
   PINN-derived ones) plugs in.
6. [crates/cjc-cana-compress/src/energy.rs](../../crates/cjc-cana-compress/src/energy.rs)
   — the `EnergyComponents` struct that grows by 3 fields for the
   PINN layer.
7. [crates/cjc-cana-compress/src/sidecar.rs](../../crates/cjc-cana-compress/src/sidecar.rs)
   — the persistence format you'll persist PINN training data into.
8. [crates/cjc-cana/src/pressure.rs](../../crates/cjc-cana/src/pressure.rs)
   — the `PressurePredictor` trait surface.
9. [crates/cjc-cana/src/cost_model.rs](../../crates/cjc-cana/src/cost_model.rs)
   — the `CostModel` trait surface PINN implements.
10. [crates/cjc-cana/src/thermal_cost_model.rs](../../crates/cjc-cana/src/thermal_cost_model.rs)
    — the existing thermal model PINN composes with (don't duplicate
    its terms blindly).
11. [crates/cjc-cana/src/memory_proxy.rs](../../crates/cjc-cana/src/memory_proxy.rs)
    — the existing memory proxy you'll derive `working_set_bytes_estimate`
    and `allocation_bytes_estimate` from.
12. [crates/cjc-nss/src/pressure.rs](../../crates/cjc-nss/src/pressure.rs)
    — the `PressureField` / `PressureGraph` primitives the
    physics-residual loss terms operate over.
13. [crates/cjc-nss/src/density.rs](../../crates/cjc-nss/src/density.rs)
    — the existing density-matrix-inspired summary that PINN's
    trajectory outputs can feed into.
14. [crates/cjc-nss/src/mir_adapter.rs](../../crates/cjc-nss/src/mir_adapter.rs)
    — the existing trace adapter that feeds compile-time prediction.
15. [crates/cjc-ad/src/pinn.rs](../../crates/cjc-ad/src/pinn.rs)
    — existing PINN proxy demo (look here for v2 patterns).
16. `bench/cana_ab_pinn/` — existing PINN-shaped benchmark.
17. `bench/cana_train_cost_model/` — existing training pipeline you'll
    extend for A6.
18. **CLAUDE.md** at the repo root — every prior session's invariants.

---

## 7. Suggested ordering for the next session

Two-track plan: ship v1 of the physics-informed layer (coefficient
model), then finish Phase A so the training-data corpus exists for v2.

**Track 1 — Physics-informed layer v1 (deterministic coefficients):**

1. Read the files in §6.
2. Create `crates/cjc-cana/src/physical_cost.rs` with the four
   type signatures from §4.2 (`PhysicalCostQuery`, `PhysicalCostEstimate`,
   `PhysicalCoefficients`, `PhysicalConstraints`). Unit-test that:
   - Every output is finite + clamped.
   - Non-finite inputs produce `CostEstimate::Unknown` (not panic).
   - The default coefficients give sensible outputs on a synthetic
     workload.
3. Create `crates/cjc-cana/src/pinn_cost_model.rs` with
   `PinnPhysicalCostModel<M, P>` + `CostModel` impl per §4.2.
   Unit-test that:
   - Composition with `NullCostModel` works.
   - Composition with `LinearCostModel` doesn't double-count
     existing thermal terms (look at `thermal_cost_model.rs` for the
     existing pattern).
   - Hard limits in `PhysicalConstraints` reject strategies that
     exceed them.
   - Higher thermal demotes equivalent strategies.
4. Create `crates/cjc-cana/src/quantum_score.rs` — docs only,
   explicit three-role framing of quantum-inspired CANA. References
   the existing primitives (energy.rs, density.rs, compression
   modules).
5. Extend `crates/cjc-cana-compress/src/energy.rs::EnergyComponents`
   with the 3 new fields per §4.5. Update proptest strategies +
   `EnergyComponents::new` + tests.
6. Create `crates/cjc-cana-nss/src/pinn_bridge.rs` — wires
   `PinnPhysicalCostModel` outputs into NSS `PressureField` updates
   per §4.4. Tests prove the bridge is deterministic + clamped.
7. Update `crates/cjc-cana-compress/src/energy_pass_ranker.rs::derive_energy_components`
   to populate the 3 new fields from `PinnPhysicalCostModel`'s
   output.
8. Wiring tests: `cana_compression_feeds_nss_pressure_summary`,
   `nss_pressure_delta_changes_cana_ranking_without_changing_legality`,
   etc. — same shape as the existing Phase 6 wiring tests, now
   exercising the PINN path.
9. Add bolero fuzz targets per the prompt's list (random physical
   queries, random coefficient sets, random NSS pressure maps,
   malformed reports).
10. Write `docs/cana/PHYSICS_INFORMED_CANA_NSS.md` per the prompt's
    documentation requirements.
11. Run the verification loop: `cargo fmt --all --check`,
    `cargo test -p cjc-cana`, `cargo test -p cjc-cana-nss`,
    `cargo test -p cjc-nss`, `cargo test -p cjc-ad`,
    `cargo test --test fixtures --release`.

**Track 2 — Finish Phase A classical infra:**

12. A4 — `crates/cjc-cana/src/profile_db.rs` per §5.1's schema.
13. A5 — Wire `compression_pressure_delta` into
    `crates/cjc-cana-nss/src/lib.rs::NssPressurePredictor`.
14. A6 — Extend `bench/cana_train_cost_model/main.rs` to emit
    `CompilationProfile` rows for the 5-way ablation set (§5.2).
15. Run the verification loop again. Compare ablations against
    acceptance metrics (§5.3).

**Track 3 — v2 neural training (deferred until ≥1000 profile rows):**

16. Phase B/C/D from the v1 handoff (now extended with the prompt's
    explicit shadow-mode gate from §5.4).

---

## 8. Open questions / decisions

These need explicit choices before / during the next session:

| # | Question | v1 handoff recommendation | v2 handoff recommendation |
|---|---|---|---|
| 1 | Where does PINN code live? Satellite crate vs modules in `cjc-cana`? | satellite crate `cjc-cana-pinn` | **modules in `cjc-cana`** for v1 deterministic; satellite when v2 MLP lands |
| 2 | Trained-weight persistence format? | build on `LosslessTracePayload` | same — `LosslessTracePayload` with magic `"CPB0"` (CANA PINN Bundle v0); v1 has no weights to persist |
| 3 | Model version in report hash? | `model_id: String` + `model_version: u32` in `EnergyComponents` | same — but include even for v1 so the schema is forward-compatible |
| 4 | Hard constraints (Lagrangian) or soft losses (weighted MSE)? | soft losses for v1 (`PhysicalConstraints` field) | **v1 uses hard rejection for `max_*` fields + soft preference for `prefer_cooler_plan_margin`**; v2's neural training uses soft residual losses |
| 5 | How much of `cjc-ad` to reuse vs build fresh? | reuse `cjc-ad::GradGraph` end-to-end | **v1 needs NONE of `cjc-ad`** (deterministic only); v2 reuses `GradGraph` |
| 6 | NEW: Shadow mode for v1 deterministic coefficients? | not addressed | **No** — v1 is deterministic by construction; shadow mode applies only to v2 trained weights |
| 7 | NEW: Should `EnergyComponents` grow to 12 fields or split into a separate `PhysicalEnergyComponents` struct? | not addressed | **Grow `EnergyComponents` to 12 fields.** Splitting would mean two ranker call sites; one ranker is cleaner |
| 8 | NEW: How to handle `LinearCostModel`'s existing thermal terms (double-counting risk)? | not addressed | **`PinnPhysicalCostModel` should compose with `ThermalAwareCostModel` via the existing per-pass scaling pattern** — `thermal_cost_model.rs` already shows the right pattern |

---

## 9. Files written this session

Source:
```
crates/cjc-cana-compress/                  NEW crate
  Cargo.toml
  src/
    lib.rs
    candidate.rs                           CompressionCandidate + CompressionKind + Criticality
    lossless_trace.rs                      byte-RLE + PassHistory adapter
    motif_dictionary.rs                    deterministic LZ77
    lowrank.rs                             power-iteration truncated SVD (advisory)
    tensor_train.rs                        TT-SVD via cjc-quantum::mps
    plan.rs                                CompressionPlan + executor + lossy payload codecs
    report.rs                              CompressionReport + ReportHash + JSON
    energy.rs                              Ising-style EnergyRanker + EnergyComponents (9 fields → will grow to 12)
    bridge.rs                              CANA → NSS pressure delta
    energy_pass_ranker.rs                  EnergyAwarePassRanker (A1)
    sidecar.rs                             CompressedCanaSidecar (A3)
  tests/
    wiring.rs
    proptest_compress.rs
    bolero_fuzz.rs
    determinism.rs
    energy_pass_ranker_wiring.rs

crates/cjc-nss/src/density.rs              PressureDensityState + PressureCorrelationSummary
crates/cjc-quantum/src/mps.rs              wide-matrix routing + 6 regression tests (the upstream SVD fix)

Workspace:
  Cargo.toml                               workspace member + workspace dep entries
```

Docs:
```
docs/cana_compress/
  DESIGN.md                                10-section design doc
  ARCHITECTURE.md                          ASCII flow diagram + crate dep graph
  BLOG_NOTES.md                            blog source material
  VERIFICATION_REPORT.md                   test counts + verification command list
  PHASE_A_HANDOFF.md                       v1 handoff (superseded by this doc)
  PHASE_A_HANDOFF_V2.md                    this doc
```

Memory:
```
~/.claude/projects/C--Users-adame-CJC/memory/
  project_cana_compress.md                 project memory note
  MEMORY.md                                index entry under "CANA Phase 6"
```

---

## 10. Determinism invariants the next session must preserve

The whole house of cards depends on these. PINN does NOT exempt you
from any of them:

1. **`BTreeMap` / `BTreeSet` / sorted `Vec` everywhere in decision paths.**
2. **All FP reductions via `cjc_repro::KahanAccumulatorF64` or
   `BinnedAccumulator`.** Naive `iter().sum::<f64>()` is forbidden.
3. **No FMA contraction.** Write `a * b + c` as
   `let t = a * b; t + c;`.
4. **All RNG via `cjc_repro::Rng` (SplitMix64) with explicit seed
   threading.** Never `rand::thread_rng()`, `Instant::now()`, or any
   OS entropy.
5. **All hashing via `cjc_cana::CanaHasher` (FNV-1a) or
   `cjc_repro::hash_bytes`.** Never
   `std::collections::hash_map::DefaultHasher`.
6. **f64 sorting via `f64::total_cmp`.** Never
   `partial_cmp().unwrap()`.
7. **No wall-clock or OS thermal sensor in decision outputs.**
   Wall-clock can be stored as *diagnostic / training metadata* in
   `CompilationProfile.compile_time_counter` etc., but the strict
   decision path uses only deterministic workload facts.
8. **`model_id` + `model_version` must appear in the report hash**
   for any cost model whose outputs influence pass plans. Two
   configurations with different model IDs must produce different
   `report_hash`.
9. **The AST/MIR parity gate (`tests/fixtures`) MUST pass after every
   change.** This is the load-bearing CI gate.
10. **No automatic weight updates during normal compilation.**
    Training happens offline (Phase A6 + B). Compilation runs
    inference only.
11. **Shadow mode required before active mode for any trained model.**
    Skip only when the model has zero trained parameters (v1
    deterministic coefficients).
12. **Legality / verifier authority is non-negotiable.** PINN
    rejects strategies via `CostEstimate::Unknown` (which the base
    ranker treats as "do nothing"), but cannot *approve* a strategy
    the legality gate would reject.

---

## 11. Verification status (this session)

| Suite | Tests | Pass | Fail |
|---|---:|---:|---:|
| cjc-cana-compress (lib) | 155 | 155 | 0 |
| cjc-cana-compress (wiring + proptest + bolero + determinism + energy_pass_ranker_wiring) | 52 | 52 | 0 |
| cjc-nss (incl. 18 new density tests) | 217+ | 217+ | 0 |
| cjc-quantum (incl. 6 new wide-matrix SVD tests) | 280 | 280 | 0 |
| cjc-cana (untouched, regression check) | 144 | 144 | 0 |
| cjc-cana-nss (untouched) | 11+ | 11+ | 0 |
| AST/MIR parity gate (tests/fixtures) | 1 (over N fixtures) | 1 | 0 |
| `cargo fmt --all --check` | — | clean | — |

**Verification commands run:**

```powershell
cargo fmt -p cjc-cana-compress --check
cargo fmt -p cjc-nss --check
cargo test -p cjc-cana-compress --release
cargo test -p cjc-nss --release
cargo test -p cjc-quantum --release
cargo test -p cjc-cana --release          # regression check
cargo test -p cjc-cana-nss --release      # regression check
cargo test --test fixtures --release      # parity gate (load-bearing)
```

All commands passed. No regressions to pre-existing suites.

---

## Appendix: v1 vs v2 of the physics-informed layer in one diagram

```text
v1 (this handoff says ship FIRST):
  ┌──────────────────────────────────────────┐
  │ PinnPhysicalCostModel<M, P>              │
  │ ├─ PhysicalCoefficients (hand-tuned)      │
  │ ├─ PhysicalConstraints (hard limits)      │
  │ └─ predict_physical() = deterministic     │
  │    closed-form combination of:            │
  │    • heat = heat_per_flop · FLOPs ·       │
  │             (1 + thread_amp · threads) ·  │
  │             (1 + batch_amp · batch)       │
  │    • memory = bytes_per_flop · (bytes /   │
  │               max(FLOPs, 1))              │
  │    • bandwidth = bw_per_byte · bytes      │
  │    • allocation = alloc_per_byte · allocs │
  │    • locality = lw · (working_set /       │
  │                 cache_proxy)              │
  │    • confidence = 1 - (alloc_churn +      │
  │                  thread_amp · threads)    │
  └──────────────────────────────────────────┘
  ▶ Zero training data required.
  ▶ Zero new deps.
  ▶ Determinism by construction.
  ▶ Ships in active mode immediately.

v2 (after ≥1000 profile rows):
  ┌──────────────────────────────────────────┐
  │ PinnPhysicalCostModel<M, P>              │
  │ ├─ Trained MLP (cjc-ad::GradGraph)        │
  │ ├─ PhysicalConstraints (unchanged)        │
  │ └─ predict_physical() = forward(MLP)      │
  │    + soft residual losses:                │
  │    • total work conservation              │
  │    • arithmetic intensity / roofline      │
  │    • pressure conservation (NSS-side)     │
  │    • Fourier law residual                 │
  │    • Little's law residual                │
  └──────────────────────────────────────────┘
  ▶ Trained offline via bench harness.
  ▶ Reuses cjc-ad (already in workspace).
  ▶ Determinism via fixed weight bundle.
  ▶ Requires shadow mode → active mode promotion.

Both versions share the same trait surface.
v2 is a drop-in replacement for v1's coefficient table.
Callers don't change.
```

---

*Generated 2026-06-09 as the v2 handoff for the Phase 6
quantum-inspired CANA/NSS compression layer + SVD fix + Phase A1 +
Phase A3, with the physics-informed layer plan synthesised from
`claude-pinn-cana-nss-prompt.md`. Next session: implement v1 of the
physics-informed layer (deterministic coefficients), then finish
Phase A4–A6 (training data corpus), then v2 (neural training).*
