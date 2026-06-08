# CANA Phase 4 — NSS Integration Design

**Date:** 2026-06-06
**Status:** Design only — NSS is not yet in the CJC-Lang workspace
**Phase:** 4 (after Phase 2 wiring shipped, Phase 5 caching shipped, Phase 3 fusion-identification shipped)

---

## TL;DR

Phase 4 integrates NSS — the **Neural Systems Simulator** from a separate fork — with CANA's compiler-side recommendation engine. NSS predicts *runtime pressure* (CPU, memory, thermal, throughput) using a deterministic SSM-based neural architecture; CANA Phase 4 consumes those predictions to make compile-time decisions that depend on runtime dynamics.

The two systems are architecturally compatible: NSS already models `PressureKind × NodeId` (cluster nodes); CANA Phase 4 maps that substrate onto `PressureKind × MirFunctionId / MirBlockId`. Same neural architecture, different topology.

**Phase 4 is design-doc-only in this session** because NSS isn't in the workspace yet. The doc below specifies what the integration looks like, what changes each crate, and what the new surfaces are — so when NSS lands, Phase 4 implementation is mechanical.

---

## 1. Background — what NSS provides

From the NSS architecture doc (in `~/Downloads/NSS_ARCHITECTURE_AND_COMPILER_IMPROVEMENTS.md`):

> NSS is a **deterministic infrastructure-dynamics modeling architecture** that learns and predicts how *pressure* propagates through interacting subsystems.

Key NSS primitives Phase 4 will consume:

| NSS Type | What it predicts | Phase 4 mapping |
|---|---|---|
| `PressureField` (9 kinds) | CPU / Memory / Io / Network / Queue / Scheduler / Sync / Thermal / Throughput | Per-function compiler-side pressure |
| `MultiTimescaleEngine` (α ∈ {0.5, 0.85, 0.95, 0.99}) | Short / medium / long / structural timescale signals | "Is this kernel a transient blip or a structural hot path?" |
| `ClusterFailureHead` | P(collapse), P(degraded) | "Will this code path likely OOM / overheat / starve under load?" |
| `ClusterCausalAttribution` | Per-node, per-pressure-kind contribution | "Which functions dominate which pressure axes?" |
| `SchedulerAdvisor` | Ranked counterfactual actions | Maps to compiler actions (inline, fuse, specialize) |
| `NssRunId` (content-addressed) | Audit trail | Composable with CANA's `ProgramHash` |

NSS is **production-ready** at the cluster level (240 tests passing, Phase 1–5 shipped). Phase 4's task is the projection layer that re-routes those tools toward the compiler.

---

## 2. Why this integration is architecturally clean

The genius (and the reason this design is feasible at all) is that NSS's substrate is **already general** — `PressureKind × NodeId`. NSS doesn't know or care what a "node" is. It models propagation dynamics over an arbitrary topology.

CJC-Lang's MIR program is naturally a graph:

| NSS concept | CJC-Lang mapping |
|---|---|
| `NodeId` | `MirFnId` (function-granular) or `BlockId` (basic-block-granular) |
| `NetworkLink` between nodes | Call edges in the call graph, OR CFG edges within a function |
| `PressureKind::Cpu` | Estimated cycles per function/block |
| `PressureKind::Memory` | Estimated peak working-set per function |
| `PressureKind::Sync` | Cross-function call frequency |
| `PressureKind::Thermal` | Sustained-execution thermal footprint |
| `PressureKind::Queue` | Pending work in compile-time analysis queues |

The MIR-adapter pattern from NSS's Phase 5a (`adapt_mir_trace_to_cluster_trajectory`) was designed for exactly this — converting MIR-execution traces into NSS trajectories. CANA Phase 4 builds the complementary direction: feed CANA's compile-time `CanaFeatures` *into* NSS as a static pressure topology, get back per-function pressure predictions.

---

## 3. New surface in CANA after Phase 4

### 3.1 New traits in `cjc-cana`

```rust
/// A predictor that uses runtime pressure dynamics to inform compile-time
/// decisions. Implementations wrap NSS to expose its predictions in the
/// vocabulary CANA's other components use.
pub trait PressurePredictor {
    /// Predict the per-function thermal pressure under sustained execution
    /// of the given program. Returns a value in [0, 1] where 1.0 ≈ "thermal
    /// trip imminent."
    fn predict_thermal(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64>;

    /// Predict the per-function peak memory pressure.
    fn predict_memory_peak(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64>;

    /// Predict the per-function CPU saturation under sustained execution.
    fn predict_cpu_saturation(&self, program: &MirProgram, features: &CanaFeatures)
        -> BTreeMap<String, f64>;

    /// Identify "structural" hot kernels — functions whose pressure
    /// trajectory persists at the longest NSS timescale (α=0.99). These
    /// are the canonical Phase 3 fusion / Phase 6 native-lowering targets.
    fn identify_structural_hot_kernels(
        &self,
        program: &MirProgram,
        features: &CanaFeatures,
    ) -> Vec<String>;

    /// Model name + version for audit. Phase 4 ships `nss_v1`.
    fn name(&self) -> &'static str;
    fn version(&self) -> u32;
}
```

### 3.2 New crate `crates/cjc-cana-nss`

NSS lives in a separate crate (`cjc-nss`) to keep its 240-test substrate isolated from CANA's compile-time-only logic. The bridge crate `cjc-cana-nss`:

- Depends on both `cjc-cana` and `cjc-nss`
- Implements `cjc_cana::PressurePredictor` for an `NssPressurePredictor` type
- Internally:
  - Calls CANA's featurizer for static structure
  - Builds a `cjc_nss::ClusterTrajectory` from those features
  - Runs the cluster NSS predictor over a synthetic "compile-time pressure" trajectory
  - Maps NSS's `ClusterPrediction` back into per-function pressure maps

```rust
// crates/cjc-cana-nss/src/lib.rs
pub struct NssPressurePredictor {
    nss: cjc_nss::ClusterNeuralSystemsSimulator,
    seed: cjc_nss::NssSeed,
}

impl NssPressurePredictor {
    pub fn from_seed(seed: u64) -> Result<Self, NssBridgeError> {
        let nss_cfg = cjc_nss::ClusterNssConfig {
            temporal_mode: cjc_nss::TemporalMode::MultiAll,
            ..Default::default()
        };
        let nss = cjc_nss::ClusterNeuralSystemsSimulator::from_seed(
            nss_cfg, cjc_nss::NssSeed(seed))?;
        Ok(Self { nss, seed: cjc_nss::NssSeed(seed) })
    }
}

impl cjc_cana::PressurePredictor for NssPressurePredictor { /* ... */ }
```

### 3.3 New cost-model variant — `ThermalAwareCostModel`

CANA's existing `CostModel` trait gains conceptual companion methods (or a new trait `ThermalAwareCostModel: CostModel`) that consume `PressurePredictor` predictions:

```rust
pub struct ThermalAwareCostModel<P: PressurePredictor> {
    pub base_model: Box<dyn CostModel>,  // e.g. LinearCostModel
    pub pressure: P,
    pub thermal_target_celsius: f64,     // user-settable; default 75.0
}

impl<P: PressurePredictor> CostModel for ThermalAwareCostModel<P> {
    fn query(&self, program: &MirProgram, features: &CanaFeatures, query: &CostQuery)
        -> CostEstimate {
        let base = self.base_model.query(program, features, query);
        // Adjust: penalize benefit estimates for thermally-aggressive passes
        // (e.g. loop unrolling) when predicted thermal trajectory exceeds target.
        match query {
            CostQuery::PassBenefit { function_name, pass_name } => {
                let thermal_map = self.pressure.predict_thermal(program, features);
                let predicted_thermal = thermal_map
                    .get(*function_name).copied().unwrap_or(0.0);
                let thermal_penalty = if predicted_thermal > 0.8 {
                    // Hot kernel — penalize aggressive optimizations
                    if matches!(*pass_name,
                        "loop_unroll" | "vectorize" | "specialize") {
                        0.5  // halve the predicted benefit
                    } else { 1.0 }
                } else { 1.0 };
                // ...
            }
            _ => base,
        }
    }
}
```

### 3.4 New runtime variants — hot/warm/cool kernels

Phase 4 introduces the kernel-variant infrastructure described in `CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` §6.4 (Method T3):

```rust
pub enum KernelVariant {
    Hot,    // fully fused, peak speed, peak heat
    Warm,   // partially fused, moderate
    Cool,   // MIR-walked, slow but cool
}
```

For each fused kernel (from Phase 3's `FusionPlan`), Phase 4 generates THREE variants and emits all of them. At runtime, NSS picks which to call based on observed thermal pressure. All three produce byte-identical output (Phase 1's determinism contract).

---

## 4. The unified Phase 4 architecture

```
                          ┌─────────────────────────┐
                          │   CJC-Lang source       │
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │  AST → HIR → MIR        │
                          └────────────┬────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │  CANA featurize         │ (Phase 1)
                          └────────────┬────────────┘
                                       ▼
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │ Linear cost model        │    │ NssPressurePredictor     │
        │ (Phase 2; pass benefits) │    │ (Phase 4; thermal etc.)  │
        └─────────────┬────────────┘    └─────────────┬────────────┘
                      ▼                               ▼
                      └──────────────┬────────────────┘
                                     ▼
                       ┌───────────────────────────┐
                       │  ThermalAwareCostModel    │ (Phase 4)
                       │  combines base + pressure │
                       └────────────┬──────────────┘
                                    ▼
                       ┌───────────────────────────┐
                       │  PassRanker               │ (Phase 2 + caching)
                       │  → RankingReport          │
                       └────────────┬──────────────┘
                                    ▼
                       ┌───────────────────────────┐
                       │  LegalityGate             │ (Phase 1 + extended)
                       └────────────┬──────────────┘
                                    ▼
                       ┌───────────────────────────┐
                       │  optimize_program_with_   │
                       │  plan + kernel variants   │
                       └───────────────────────────┘
```

The new edges are the NSS predictor and the thermal-aware cost model. Everything else is unchanged from Phase 2/3/5.

---

## 5. Implementation effort estimate

| Step | Effort | Dependencies |
|---|---|---|
| 1. Add NSS as a workspace member | ~1 day | Requires `cjc-nss` crate to land in this workspace |
| 2. Build `cjc-cana-nss` bridge crate | ~3 days | Steps 1; defines `NssPressurePredictor` |
| 3. Add `PressurePredictor` trait to `cjc-cana` | ~1 day | None; pure surface change |
| 4. Build `ThermalAwareCostModel` | ~2 days | Steps 2 + 3 |
| 5. Wire `ThermalAwareCostModel` into `default_ranker` (opt-in) | ~1 day | Step 4 |
| 6. Hot/warm/cool kernel variant generation | ~5 days | Phase 3 fusion codegen (out of scope here) |
| 7. Runtime variant selector | ~3 days | Step 6 + NSS runtime sampling |
| 8. Benchmark on real ML workloads | ~3 days | All previous steps |
| **Total** | **~19 days** | |

For a single engineer: **~4 weeks of focused work** once NSS is in the workspace. For a small team: **~2 weeks** with parallel work.

---

## 6. What NSS needs to expose for Phase 4

For the bridge crate to work, NSS's existing surface (per the architecture doc) is *almost* sufficient. The gaps:

1. **`cjc-nss` must publish a single-program `predict_thermal_for_program` entry point.** NSS currently predicts on `ClusterTrajectory` (a time series). CANA at compile time has a static MIR program, not a trajectory. The bridge can synthesize a trivial trajectory (one tick, no events), but a convenience function would be cleaner.

2. **NSS needs an "infer node count from program" mode.** Currently `ClusterNssConfig` expects a fixed node count. The bridge passes `program.functions.len()` as the node count.

3. **`PressureKind::Thermal` runtime feedback.** NSS currently has `PressureKind::Thermal` in its enum but no concrete runtime feedback loop (Phases 1–5 focused on cluster prediction). Phase 4 needs the runtime variant selector to call NSS's prediction with *measured* CPU temperature as input. This requires NSS to accept a `runtime_observation: ThermalObservation` input on its predict surface.

None of these are blockers — they're all small additions to NSS's API. The 240-test substrate doesn't need to change.

---

## 7. Determinism contract preservation

Phase 4 introduces *runtime feedback*, which creates a determinism risk. Two safeguards:

1. **Compile time stays deterministic.** The `NssPressurePredictor` predictions at compile time depend only on `(features, NssSeed)`. Same MIR + same seed → same prediction. Phase 1's `ProgramHash` content-addressing extends cleanly.

2. **Runtime variant selection is replayable.** Every runtime decision (hot/warm/cool switch) is logged with its `NssRunId`. A subsequent run with the same source + same observed temperatures produces byte-identical execution. The audit chain keeps growing without forcing the *compile* to be non-deterministic.

3. **The three kernel variants produce byte-identical output.** This is the load-bearing invariant — Phase 4's runtime selector can switch variants mid-execution without changing observable program behavior. Verified by parity tests during Phase 3 codegen (not Phase 4).

---

## 8. Phase 4 milestones (suggested)

| Milestone | Deliverable | Duration |
|---|---|---|
| **M1: NSS in workspace** | `cjc-nss` crate added, builds cleanly, all 240 NSS tests pass | 1 day |
| **M2: Bridge crate** | `cjc-cana-nss::NssPressurePredictor` implements `PressurePredictor`; 20+ unit tests | 3 days |
| **M3: Thermal cost model** | `ThermalAwareCostModel` wraps base; benchmarked vs Phase 2 baseline | 2 days |
| **M4: Opt-in CLI flag** | `cjcl run --thermal-aware` enables the thermal cost model | 1 day |
| **M5: Kernel variant infrastructure** | Hot/warm/cool variant types defined + parity-tested | 5 days |
| **M6: Runtime variant selector** | Reads NSS predictions, switches variants based on observed thermal | 3 days |
| **M7: Real-workload benchmark** | Chess RL training + PINN demo show measured thermal/runtime improvement | 3 days |
| **Total** | | **~18 days** |

---

## 9. What this design does NOT solve

Honest list of what's still out of scope for Phase 4:

1. **Cross-process pressure sharing.** NSS was built for *cluster* pressure. Multiple CJC-Lang processes running on the same machine should share thermal predictions. Phase 4 ships per-process; Phase 5 can add inter-process coordination via a shared NSS instance.

2. **Pre-training the NSS model on real CJC-Lang workloads.** NSS's prediction accuracy depends on its training data. The NSS handoff doc says "calibrated on synthetic simulators." Real CJC-Lang programs may need a different distribution. Phase 4 ships the bridge; Phase 5 collects training data.

3. **NSS's neural complexity.** NSS has ~600 learnable parameters. At inference time per-program, that's tiny. But if Phase 4 needs to retrain NSS during compile (it shouldn't — but if it did), that's not in scope.

4. **OS-level thermal probe APIs.** Phase 4 assumes the runtime can read CPU temperature via OS APIs (`/sys/class/thermal/thermal_zone*` on Linux, IOKit on macOS, Windows ACPI). Implementing those is straightforward but platform-dependent — not in this design doc.

---

## 10. Decision points for whoever implements Phase 4

Three architectural choices the implementer needs to make:

1. **Pre-train or lazy-train NSS for compiler workloads?**
   - Pre-train: ship the trained weights with `cjc-cana-nss`; fast inference, but stale predictions for unusual workloads.
   - Lazy-train: train on first N compilations, then freeze; fresh predictions, but cold-start cost.
   - **Recommendation:** start lazy-train with N=100, then make the trained weights cacheable on disk for distribution. Best of both.

2. **Per-process vs shared NSS?**
   - Per-process: simpler; no IPC; predictions don't share insights across CJC-Lang programs.
   - Shared (e.g. a daemon): cross-program learning, but adds an IPC dependency and a separate failure mode.
   - **Recommendation:** start per-process; revisit when there's measured value in sharing.

3. **How aggressive is the default thermal target?**
   - Conservative (60°C): rare to overheat; small speed cost.
   - Balanced (75°C): occasional throttling; ~10% speed cost over sustained workloads.
   - Aggressive (90°C): rarely throttles; significant heat for nominal workloads.
   - **Recommendation:** ship `--thermal-target=75` as default. Power users get `--thermal-target=balanced/conservative/aggressive` aliases.

---

## 11. Relationship to other Phase 4-adjacent work

Phase 4 is **independent of** but **complementary to**:

- **Phase 5 caching (SHIPPED in commit `522520b`):** the cached recommendations would be expanded to include thermal predictions. Cache key extends to `(ProgramHash, CostModel.version(), NssPressurePredictor.version())`.

- **Phase 3 fusion-identification (SHIPPED in commit `ece3a81`):** the `FusionPlan` becomes input to the kernel-variant generator. Each fusion candidate gets three variants (hot/warm/cool).

- **Phase 6 native lowering:** the kernel variants are a stepping stone. Phase 6 replaces the MIR-walked "cool" variant with an LLVM-compiled native variant (using the deterministic LLVM flag subset from `CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` §5.4).

---

## 12. Why this is the right time to write this design

Three reasons:

1. **The architectural seams are clean *now*.** Phase 1's `CostModel` trait was deliberately open. Phase 2's `PassRanker` is generic over `M: CostModel`. Phase 3's `FusionPlan` is a separate data structure. Phase 5's `CachingPassRanker` wraps anything. *All* of CANA's components compose with a `PressurePredictor` without changing their existing surfaces.

2. **The roadmap dependency chain is clear.** Phase 4 needs (1) NSS in the workspace, (2) Phase 3 fusion-identification (DONE), and (3) hot/warm/cool kernel codegen (which has its own dependencies on `cjc-runtime`). Writing this design now lets the team plan parallel work.

3. **NSS is "ready to integrate" per its handoff doc.** The architecture is shipped; the substrate is tested; only the bridge crate is missing. This design specifies the bridge.

---

## 13. Cross-references

- `docs/cana/CANA_NSS_COMPILER_IMPROVEMENT_PLAN.md` — the multi-phase plan that this document operationalizes for Phase 4
- `docs/cana/CANA_PHASE_2_DESIGN.md` — Phase 2 ADRs that establish CostModel + PassRanker conventions
- `docs/cana/CANA_PHASE_2_BENCHMARK_FINDINGS.md` — the empirical data Phase 4 builds on
- `~/Downloads/NSS_ARCHITECTURE_AND_COMPILER_IMPROVEMENTS.md` — the NSS architecture this design wraps
- `~/Downloads/NSS_HANDOFF_PHASE_5_COMPILER_INTEGRATION.md` — the NSS handoff that describes the MIR-trace adapter

---

*Generated alongside Phase 3 fusion-identification shipping. To be acted on when `cjc-nss` lands in the CJC-Lang workspace.*
