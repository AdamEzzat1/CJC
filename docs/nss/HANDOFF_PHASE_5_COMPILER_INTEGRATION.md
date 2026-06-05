# NSS Handoff — Compiler Integration & Companion Network

**Status:** NSS Phases 1 → 5b complete. Ready for CJC-Lang compiler-fork integration in the next session.

**Crate:** `crates/cjc-nss/` — `publish = false`, in-workspace, 2.4k LOC + ~1.6k LOC tests.

**Test count:** 240 tests passing across all phases, 0 failures, 0 ignored. Determinism verified end-to-end.

**Latest demo run:** `cargo run --example autonomous_closed_loop -p cjc-nss --release` — 4-node cluster recovers from scripted failure with 1 applied action, 29/32 nominal ticks.

---

## 1. What's shipped — phase-by-phase map

| Phase | File(s) | What it gives you |
|---|---|---|
| **1** — Foundation | `pressure.rs`, `system.rs`, `simulator.rs`, `propagation.rs`, `encoder.rs`, `temporal.rs`, `heads.rs`, `nss.rs`, `replay.rs`, `seed.rs`, `failure.rs`, `scheduler.rs`, `error.rs` | Single-tier queue simulator, pressure-field primitives, deterministic-seed contract, count-based-calibrated NSS predictor with exact causal attribution, replay validator |
| **2** — Cluster | `cluster.rs`, `cluster_simulator.rs`, `cluster_nss.rs` | N-node cluster simulator with network edges, scripted interventions (FailNode/RecoverNode), cluster-aware predictor with sum-pool aggregation, per-node + cluster-summary attribution, cluster replay validator |
| **2b** — GPU training | `gpu_training.rs` | Lockstep data-parallel training simulator (jitter, allreduce, memory fragmentation, OOM), emits the same `ClusterTrajectory` so the cluster NSS predicts on it unchanged |
| **2c** — Pipeline | `gpu_training.rs` (extended) | `pipeline_stages` + `microbatches_per_iteration`, GPipe bubble formula, stage-dependent activation memory |
| **2d** — GradGraph training | `cluster_grad.rs` | `Optimizer::Adam` over `cjc-ad::GradGraph` — actual gradient descent on cluster head, replaces count-based calibration when configured |
| **3a** — Counterfactual | `counterfactual.rs` | `ClusterSimulator::snapshot()` + `ClusterSnapshot::fork(extra_ivs)`, `CounterfactualComparison`, `run_cluster_counterfactual` |
| **3b** — Multi-timescale | `multi_timescale.rs` + `cluster_nss.rs` (extended) | `Timescale::{Short,Medium,Long,Structural}` enum (α=0.5/0.85/0.95/0.99), `MultiTimescaleEngine` with N parallel SSM buffers, `ClusterNssConfig::temporal_mode = TemporalMode::MultiAll` |
| **3c** — Advisory head | `advisory.rs` | `SchedulerAdvisor::recommend(snapshot, nss) → AdvisoryRanking`, ranks `DoNothing` / `FailNode` / `RecoverNode` by counterfactual P(collapse) |
| **3d** — Pipeline ext | `gpu_training.rs` (extended) | `PipelineSchedule::{GPipe,OneForwardOneBackward,Interleaved{factor}}`, activation checkpointing (`activation_checkpointing`, `checkpoint_memory_factor`, `checkpoint_recompute_overhead`) |
| **3e** — Advisor ext | `advisory.rs` (extended), `cluster.rs` (extended), `cluster_simulator.rs` (extended) | `NodeHealth::Absent`, autoscaling interventions (`AddNode`/`RemoveNode`), `ShedLoadOverride { intensity }`, `AdvisoryAction::{ShedLoad,AddNode,RemoveNode}`, `SchedulerAdvisor::recommend_per_node` |
| **4** — Autonomous engine | `autonomous.rs` | `AutonomousOptimizer` closed-loop controller with `SafetyMode::{Conservative,Moderate,Aggressive}`, per-node cooldowns, improvement floor, action budget, `DecisionRecord` audit trail |
| **5a** — MIR adapter | `mir_adapter.rs` | `MirTraceEvent` → `ClusterTrajectory` adapter. The compiler-integration bridge. |
| **5b** — Legality verifier | `legality.rs` | `LegalityVerifier::verify(script, topology) → LegalityReport`. Static analysis: oscillation, cooldown, empty-cluster, unknown-node, invalid-intensity, aggressive-forbidden |

## 2. Demos that work today

```bash
# Phase 2 — cluster failure-cascade
cargo run --example cluster_failure_cascade -p cjc-nss --release

# Phase 2b — GPU training with OOM
cargo run --example gpu_training_imbalance -p cjc-nss --release

# Phase 3a — counterfactual node-failure analysis
cargo run --example counterfactual_node_failure -p cjc-nss --release

# Phase 3b — multi-timescale impulse decay
cargo run --example multi_timescale_decay -p cjc-nss --release

# Phase 3c — scheduler advisor on a failed-node cluster
cargo run --example scheduler_advisor -p cjc-nss --release

# Phase 3d — pipeline-schedule comparison (GPipe/1F1B/Interleaved/+ckpt)
cargo run --example pipeline_schedule_comparison -p cjc-nss --release

# Phase 3e — per-node advisor with NodeHealth::Absent
cargo run --example advisor_per_node -p cjc-nss --release

# Phase 4 — autonomous closed-loop
cargo run --example autonomous_closed_loop -p cjc-nss --release
```

---

## 3. The compiler-integration recipe

The eventual goal: a **forked CJC-Lang compiler** that uses NSS to drive optimisation decisions. Here's the complete recipe from a Phase 5 MIR trace to a compiler decision:

### 3.1 Instrument the MIR executor

Add a callback / event-emit hook to `crates/cjc-mir-exec/src/lib.rs` that emits `MirTraceEvent`s per basic block. Minimum surface to instrument:

| Hook | Field on `MirTraceEvent` | Notes |
|---|---|---|
| Per-instruction count | `instruction_count` | The executor already tracks this for `--mir-opt` |
| Register allocator state | `register_pressure` ∈ [0, 1] | Fraction of MIR registers currently spilled |
| Heap allocator state | `heap_bytes_in_use` | Sum of live allocations |
| Call frame depth | `call_depth` | The executor's call-stack count |
| Branch taken? | `branch_taken` | Bool per `MirInst::Jump` / `JumpIf` execution |
| Syscall? | `io_event` | True on `MirInst::CallBuiltin` for IO-like builtins |
| GC trigger? | `gc_event` | True when the GC fires |

Each emitted event maps a basic block (or instruction window) to one `MirTraceEvent`. Aggregation into NSS ticks happens automatically via `events_per_tick`.

### 3.2 Wire the trace through the adapter

```rust
use cjc_nss::{
    adapt_mir_trace_to_cluster_trajectory, MirAdapterConfig, MirTraceEvent,
    ClusterNeuralSystemsSimulator, ClusterNssConfig, TemporalMode, NssSeed,
    SchedulerAdvisor, AdvisorConfig, SafetyMode,
};

// 1. Build the MIR trace.
let events: Vec<MirTraceEvent> = run_instrumented_mir_executor(program);

// 2. Configure the adapter. n_blocks = how many distinct basic blocks
//    the program has; events_per_tick = aggregation granularity.
let adapter_cfg = MirAdapterConfig {
    n_blocks: program.basic_block_count() as u32,
    events_per_tick: 16,
    heap_capacity_bytes: 1024 * 1024 * 1024,  // 1 GiB
    call_depth_threshold: 32,
    link_capacity: 8,
    link_weight: 0.25,
    initial_tick: 0,
};

// 3. Convert to NSS trajectory.
let out = adapt_mir_trace_to_cluster_trajectory(&events, &adapter_cfg).unwrap();

// 4. Build the cluster NSS — use multi-timescale memory for
//    compiler-side decisions because they need both short-term
//    (per-instruction) and structural (across-program) signals.
let nss_cfg = ClusterNssConfig {
    temporal: cjc_nss::TemporalStateConfig {
        state_dim: 4,
        input_dim: 16,
        ..Default::default()
    },
    head: cjc_nss::HeadConfig { state_dim: 16, ..Default::default() }, // 4 timescales * 4 per-scale
    temporal_mode: TemporalMode::MultiAll,
    ..Default::default()
};
let nss = ClusterNeuralSystemsSimulator::from_seed(nss_cfg, NssSeed(42)).unwrap();
```

### 3.3 Drive compiler-side decisions through the advisor

Each MIR optimisation pass (inline / unroll / vectorise / etc.) becomes an `AdvisoryAction` candidate. For Phase 5 the candidate space is the existing one (DoNothing / FailNode / RecoverNode / ShedLoad / AddNode / RemoveNode), so you'll need an interpretation:

| Compiler concept | Maps to AdvisoryAction |
|---|---|
| Inline function F at site A | `FailNode { node: site_A }` (drains the site's "queue" of work into the caller) |
| Don't inline (keep call) | `DoNothing` |
| Unroll loop L | `ShedLoad { node: loop_L, intensity }` (the intensity = unroll factor / max_unroll) |
| Specialise hot path | `AddNode { node: cold_path_slot }` (the cold slot exists in the topology as Absent until specialisation activates it) |
| Inline back / un-specialise | `RemoveNode` |

For Phase 5 this is a *projection* — you'll likely want a domain-specific `CompilerAction` enum in a future Phase 6 that maps directly to compiler semantics. The advisor's substrate works without modification because the counterfactual ranking is action-agnostic.

### 3.4 Apply the legality verifier first

Before the compiler commits to an optimisation script, run it through the legality verifier:

```rust
use cjc_nss::{LegalityConfig, LegalityVerifier};

let verifier = LegalityVerifier::new(LegalityConfig {
    min_action_cooldown: 1,        // back-to-back ops on the same block are fine for compilation
    min_active_nodes: 1,           // at least one block must remain active
    allow_aggressive_actions: true, // inline / drain are normal compiler ops
    max_reversals_per_node: 1,     // catch "inline X, un-inline X" oscillations
    initial_active_nodes: out.topology.node_count() as u32,
}).unwrap();

let report = verifier.verify(&proposed_script, &out.topology);
if !report.passed() {
    return Err(CompilerError::IllegalOptimizationScript(report));
}
```

The verifier catches **whole classes of optimisation bugs** before the compiler commits: oscillations (inlining then un-inlining), capacity violations (removing all basic blocks), invalid intensities (unroll factor > 1.0).

### 3.5 Audit the closed loop

Every `DecisionRecord` carries an `NssRunId` content-addressing the input state. This means the compiler can:

1. Save the audit log to a `.cjcl.audit.json` sidecar.
2. Reproducibly verify "did the optimiser apply this transformation?" by re-running NSS on the same MIR trace.
3. A regulator / reviewer can replay the entire optimisation sequence and verify byte-identically.

This is the *infrastructure-grade-auditability* that the spec called for, now hooked end-to-end into the compiler.

---

## 4. Architectural invariants you must preserve

Things that **must not change** when wiring NSS to the compiler:

1. **Deterministic substreaming.** Every RNG draw goes through `NssSeed::substream("named.domain")`. If you add new randomness on the compiler side, give it its own substream salt — never reuse an existing salt.
2. **BTreeMap iteration.** Never replace a `BTreeMap` with a `HashMap` in any of the NSS code paths. Same for `BTreeSet`. The iteration determinism is structural.
3. **Kahan reductions.** All cross-feature / cross-tick sums use `cjc_repro::KahanAccumulatorF64`. Don't replace with naive `.sum()`.
4. **No FMA.** Phase 1 forbids fused-multiply-add in numerical paths to preserve cross-platform bit-identity. The compiler may want FMA for *compiled code*, but NOT for NSS prediction paths.
5. **Closed enums.** `PressureKind`, `NodeHealth`, `Intervention`, `AdvisoryAction`, `PipelineSchedule`, `Timescale`, `TemporalMode`, `SafetyMode`, `FailureKind`, `SchedulerKind`. New variants are fine; replacing with an open enum or string-based dispatch breaks the audit-trace canonical-byte system.
6. **`Intervention` ordering.** When adding new variants, extend `Intervention::kind_byte()` with a new discriminator and ensure `canonical_bytes()` and `Ord` are consistent.
7. **Trajectory tick monotonicity, but allow any starting tick.** Phase 3a fix — needed so a forked simulator can resume from snapshot tick.

---

## 5. Companion network — what the next session should know

You mentioned working on **another neural network** that will improve the CJC-Lang compiler alongside NSS. A few integration points NSS leaves available:

### 5.1 Two-NN handoff via the audit trace

The cleanest way to compose two networks: NSS produces predictions + audit traces; the second NN reads the audit traces as a *feature stream*. The `DecisionRecord` already carries everything a downstream model needs:
- `run_id` — content-addressed input fingerprint
- `recommended` action + `recommended_collapse` probability
- `baseline_collapse` (DoNothing) for delta computation
- `confidence_margin`
- `outcome` (Applied / Skipped / NoOp)
- `skip_reason` (interpretable text reason for skips)
- `applied_interventions` (the actual ones injected)

A second network can be trained to **predict the safety-layer's decision** (which is the *meta-decision*) — useful if you want to learn when to trust NSS's recommendation.

### 5.2 Two-NN composition via the pressure field

The `PressureField` is open at the *value level* but closed at the *kind level* (no `PressureKind::Custom`). If your second network produces pressure-domain features (e.g., learned compiler-state representations), you can:

1. Use the pressure-field abstraction as a shared interlingua — your second NN's output projects onto `PressureKind`s the same way the MIR adapter does.
2. **Or** add a `LearnedPressureField` wrapper (Phase 6) that carries a small fixed-size additional learned vector alongside the canonical pressure fields. The NSS encoder concatenates this with the standard feature vector. This way two networks compose via the encoder layer without changing the trajectory shape.

### 5.3 Cross-NN determinism

The biggest pitfall: if the second NN is non-deterministic (e.g., uses `HashMap` iteration, system RNG, or FMA), it'll break NSS's audit-trace contract. Either:
- Make the second NN deterministic (recommended — use the same `NssSeed::substream` pattern).
- Or, treat the second NN's outputs as *inputs* to NSS (so the second NN can be non-deterministic but its output is captured in the `input_hash`).

---

## 6. Known gaps and good places to extend

Honest list of where NSS still leaves work for the compiler integration:

### 6.1 The MIR adapter is **synthetic-shaped**

`MirTraceEvent` is the *shape* of what an instrumented MIR executor should emit, but the actual instrumentation in `cjc-mir-exec/src/lib.rs` isn't wired up. Next session needs to:
1. Add an `MirTraceCollector` trait + `Vec<MirTraceEvent>` collector implementation to `cjc-mir-exec`.
2. Emit events from the executor's instruction-dispatch loop.
3. Test that `adapt_mir_trace_to_cluster_trajectory(collector.finalize(), &cfg)` returns a non-empty trajectory.

This is ~200-400 LOC of instrumentation work, no architectural challenges.

### 6.2 The advisor's `AdvisoryAction` set is **infrastructure-shaped**

For real compiler integration you'll want compiler-domain actions: `Inline { call_site }`, `Unroll { loop_id, factor }`, `Vectorize { loop_id }`, `Specialize { fn_id }`. Options:

- **Option A**: Map compiler actions onto existing `AdvisoryAction` variants (as outlined in §3.3). Quick and dirty.
- **Option B (recommended for production)**: Add a `CompilerAction` enum in a new `compiler_advisor.rs` module that wraps the existing advisor and exposes domain-specific recommendations. The wrapper does the projection internally.

### 6.3 NSS is **trained on synthetic simulators**

The cluster + GPU-training + queue simulators are *plausible* but not calibrated to real compiler workloads. Once the MIR instrumentation lands, you'll want to:
1. Collect MIR traces from real CJC-Lang programs.
2. Use those as the training trajectories for `ClusterNeuralSystemsSimulator::fit_with_adam(...)`.
3. Re-evaluate predictions against held-out programs.

This is the most important post-Phase-5 ML work. The substrate (Adam, GradGraph, deterministic fit) is all there.

### 6.4 The legality verifier is **structural-only**

`LegalityVerifier` catches structural violations (oscillation, cooldown, capacity) but doesn't verify *semantic* compiler properties like "this optimisation preserves observable behaviour". For the eventual compiler integration you'll likely want a complementary `SemanticLegalityVerifier` that checks compiler-side correctness invariants. This is genuinely new code, not an NSS extension.

### 6.5 Phase 5 deferred items still on the list

- **`cjc-locke` detector composition.** `nss_*` custom detectors that emit `E9500+` codespace findings for instability-propagation issues. The substrate (`cjc-locke::CustomDetector`) exists; the wiring is ~150 LOC.

---

## 7. Quick-reference: file map for the next session

```
crates/cjc-nss/
├── Cargo.toml                                    deps: cjc-repro, cjc-locke, cjc-ad, cjc-runtime
├── src/
│   ├── lib.rs                                    Crate root + phase docs + re-exports
│   ├── error.rs                                  NssError enum
│   ├── seed.rs                                   NssSeed substream, NssRunId, InputHash
│   ├── pressure.rs                               PressureKind enum (9 variants), PressureField, PressureGraph
│   ├── system.rs                                 SystemState, SystemEvent, SystemTrajectory
│   ├── scheduler.rs                              SchedulerAction, SchedulerKind
│   ├── failure.rs                                FailureState, FailureKind, FailurePrediction
│   ├── simulator.rs                              Phase-1 queue simulator
│   ├── propagation.rs                            PressurePropagator (intra-node)
│   ├── encoder.rs                                SystemEncoder (saturating_add patch from earlier session)
│   ├── temporal.rs                               TemporalStateEngine (single-α SSM)
│   ├── heads.rs                                  FailurePredictionHead, CausalAttributionHead
│   ├── nss.rs                                    NeuralSystemsSimulator (Phase 1)
│   ├── replay.rs                                 PredictionTrace, ReplayValidator
│   ├── cluster.rs                                NodeId, NetworkLink, ClusterTopology, NodeHealth (incl. Absent)
│   ├── cluster_simulator.rs                      ClusterSimulator, Intervention (5 variants), RoutingPolicy
│   ├── cluster_nss.rs                            ClusterNeuralSystemsSimulator, TemporalMode (Single / MultiAll)
│   ├── cluster_grad.rs                           Phase 2d Adam over GradGraph
│   ├── counterfactual.rs                         Snapshot / Fork / CounterfactualComparison
│   ├── multi_timescale.rs                        Timescale, MultiTimescaleEngine
│   ├── advisory.rs                               SchedulerAdvisor (recommend + recommend_per_node)
│   ├── autonomous.rs                             AutonomousOptimizer (Phase 4 closed-loop)
│   ├── gpu_training.rs                           GpuTrainingSimulator + PipelineSchedule (3d) + Absent (3e)
│   ├── mir_adapter.rs                            ⭐ Phase 5a — compiler-integration bridge
│   └── legality.rs                               ⭐ Phase 5b — static script analysis
├── tests/                                        9 + 6 + 6 + 5 = 26 integration/property tests
├── examples/                                     8 runnable demos covering Phases 2 → 4
└── docs/nss/HANDOFF_PHASE_5_COMPILER_INTEGRATION.md     ⭐ This file
```

⭐ = key files for next-session compiler-fork work.

---

## 8. First-session goals for next time

A reasonable next-session plan:

1. **Instrument cjc-mir-exec** — add `MirTraceCollector` trait, wire into the executor's dispatch loop. ~half-session of work, no architectural changes to NSS needed.
2. **Wire a smoke-test end-to-end** — run a real CJC-Lang program through the instrumented MIR executor, feed the trace to the adapter, ask the cluster NSS to predict. Confirm the round-trip works.
3. **Sketch the `CompilerAction` enum** in a new `compiler_advisor.rs` module. Map two or three optimisations (inline, unroll) to the existing `AdvisoryAction` substrate. Test on a small CJC-Lang program.
4. **Decide where the second NN slots in** — feature-extractor (its output → MirTraceEvent), policy head (its output → AdvisoryAction ranking), or audit-trace consumer (post-NSS).

If you have ~3 hours, items 1+2 are the natural anchor; items 3+4 are good to start scoping with the architecture in front of you.

---

## 9. Verification you can run anytime

```bash
# Full test sweep (should be 240 tests, 0 failures)
cargo test -p cjc-nss --release

# Lint-clean workspace build
cargo build --workspace --release

# Re-run any demo
cargo run --example <name> -p cjc-nss --release
```

Determinism check is built into every demo's last line.

---

*Generated by NSS Phase 5 session — full git log is in this branch.*
