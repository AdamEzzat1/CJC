//! # cjc-nss — Neural Systems Simulator (Phase 1)
//!
//! NSS is **not** a generic forecasting model. It is a deterministic
//! infrastructure-dynamics modeling architecture that learns:
//!
//! * system dynamics
//! * resource contention
//! * temporal pressure propagation
//! * scheduler behavior
//! * cascading failure emergence
//! * causal operational states
//!
//! The architecture models infrastructure as a dynamic evolving pressure
//! system rather than a static metrics prediction problem.
//!
//! ## Positioning
//!
//! NSS is positioned as **Deterministic Infrastructure Dynamics Modeling /
//! Learned Systems Dynamics Architecture / Causal Resource Pressure
//! Simulation** — *not* "AI for infrastructure". Differentiation against:
//!
//! | Class                  | What it does                                  | What NSS adds                            |
//! |------------------------|-----------------------------------------------|------------------------------------------|
//! | Anomaly detection      | flags outliers in metrics                     | causal propagation chain + lineage       |
//! | Observability dashboards| visualises raw metrics                       | learned pressure-field dynamics          |
//! | Generic forecasters     | extrapolates a univariate series             | scheduler-aware multi-field interactions |
//! | AIOps                   | rule-based alert correlation                 | deterministic replay + audit trace       |
//!
//! ## Core philosophy
//!
//! Traditional ML predicts outputs. NSS models the transition:
//!
//! ```text
//! SystemState[t]
//!     → ResourcePressure[t]
//!     → TemporalPropagation[t]
//!     → SchedulerInteraction[t]
//!     → SystemState[t+1]
//! ```
//!
//! Pressure is a **first-class computational primitive** with magnitude,
//! accumulation, dissipation, propagation, amplification, instability
//! thresholds, and temporal persistence — see [`Pressure`].
//!
//! ## Determinism contract
//!
//! 1. All RNG via [`cjc_repro::Rng`] (SplitMix64), seeded from
//!    [`NssSeed::substream`] with a domain-named salt.
//! 2. All reductions (encoder dot-products, state updates, head logits) use
//!    [`cjc_repro::KahanAccumulatorF64`].
//! 3. All maps use [`std::collections::BTreeMap`] / [`std::collections::BTreeSet`].
//!    No `HashMap` iteration anywhere.
//! 4. Pressure-graph propagation visits edges in lexicographic
//!    `(source, target)` order.
//! 5. No FMA, no thread-parallel reductions that would alter accumulation
//!    order, no hidden randomness.
//! 6. Stability is **structural**: encoder/state weights are constrained
//!    such that the temporal-state matrix has spectral norm `≤ α < 1` by
//!    construction, so rollouts cannot blow up.
//! 7. Every [`PredictionTrace`] carries an [`NssRunId`] fingerprint that
//!    binds (seed, config, input hash, model version). Two runs with
//!    identical inputs produce the same `NssRunId`.
//!
//! ## Architectural primitives (Phase 1)
//!
//! - Pressure: [`PressureKind`], [`Pressure`], [`PressureField`],
//!   [`PressureGraph`], [`PressureEdge`], [`PressureFlow`].
//! - System state: [`SystemState`], [`SystemEvent`] (SIR record),
//!   [`SystemTrajectory`].
//! - Scheduler: [`SchedulerAction`], [`SchedulerKind`].
//! - Failure: [`FailureState`], [`FailureKind`], [`FailurePrediction`].
//! - Determinism: [`NssSeed`], [`NssRunId`], [`InputHash`].
//! - Simulator: [`QueueSimulator`], [`QueueConfig`], [`QueueSnapshot`].
//! - Propagation: [`PressurePropagator`], [`PropagationConfig`].
//! - Neural: [`SystemEncoder`], [`TemporalStateEngine`],
//!   [`FailurePredictionHead`], [`CausalAttributionHead`],
//!   [`NeuralSystemsSimulator`].
//! - Replay / audit: [`PredictionTrace`], [`ReplayValidator`].
//! - Errors: [`NssError`].
//!
//! ## Phase 2 — distributed cluster simulator (SHIPPED)
//!
//! Phase 2 adds the [`ClusterSimulator`], [`ClusterTopology`],
//! [`ClusterSystemState`], [`ClusterTrajectory`], scripted
//! [`Intervention`]s (deterministic node failures + recoveries), and
//! the [`ClusterNeuralSystemsSimulator`] — a cluster-aware predictor
//! that sum-pools per-node encodings, runs the Phase 1 temporal engine
//! on the pooled latent, and produces a cluster-level failure
//! prediction plus per-node + cluster-summary causal attribution. The
//! [`ClusterReplayValidator`] reproduces fitted cluster predictions
//! byte-identically from the bundled `(seed, config, topology,
//! intervention_script, training_trajectory)` tuple.
//!
//! ## Phase 2b — GPU training simulator (SHIPPED)
//!
//! Phase 2b adds [`GpuTrainingSimulator`] + [`GpuTrainingConfig`], a
//! lockstep data-parallel training simulator that emits the same
//! [`ClusterTrajectory`] the Phase 2 cluster simulator does — so the
//! existing [`ClusterNeuralSystemsSimulator`] predicts on GPU-training
//! data with no changes. Models:
//!
//! - Per-GPU per-microbatch service-time jitter → drives `Sync` pressure
//!   via straggler-induced idle time.
//! - Allreduce overhead → drives `Network` pressure.
//! - Per-iteration memory growth + periodic GC + residual
//!   fragmentation → drives `Memory` pressure.
//! - OOM crashes via the same `Intervention::FailNode` script
//!   primitives used by Phase 2.
//!
//! Pipeline parallelism is shipped in **Phase 2c** (see below).
//!
//! ## Phase 2c — pipeline parallelism in GPU training (SHIPPED)
//!
//! `GpuTrainingConfig` now carries `pipeline_stages` and
//! `microbatches_per_iteration`. With `pipeline_stages > 1`:
//!
//! - GPUs are partitioned into stages (each stage = data-parallel
//!   replicas group).
//! - The canonical **GPipe bubble fraction**
//!   `(stages - 1) / (stages - 1 + microbatches)` is applied to every
//!   tick as a floor on sync pressure.
//! - **Early stages accumulate more activation memory** (stage 0
//!   holds ~`microbatches` worth of forward activations until backward
//!   starts; final stage holds 1).
//! - Allreduce is restricted to within-stage replicas, reducing
//!   per-tick network pressure proportionally.
//!
//! Pure data-parallel behaviour (Phase 2b) is preserved by default
//! (`pipeline_stages: 1`).
//!
//! ## Phase 2d — GradGraph cluster-head training (SHIPPED)
//!
//! Adds [`cluster_grad::fit_with_adam`] — actual gradient descent via
//! [`cjc_ad::GradGraph`] over the cluster failure head. Replaces the
//! Phase 2 count-based calibration with a per-batch BCE loss minimised
//! by Adam (`β1=0.9, β2=0.999, ε=1e-8`). Encoder + temporal stay at
//! their seeded init; only the head is trained. Loss is fully
//! deterministic — two runs with identical inputs produce
//! bit-identical loss curves.
//!
//! ## Phase 3a — counterfactual forking (SHIPPED)
//!
//! Adds [`ClusterSimulator::snapshot`] +
//! [`GpuTrainingSimulator::snapshot`] returning [`ClusterSnapshot`] /
//! [`GpuTrainingSnapshot`] handles. Forks via `snapshot.fork(extras)`
//! create independent simulator instances that continue from the
//! snapshot tick with merged interventions.
//! [`run_cluster_counterfactual`] runs a paired "A vs B" experiment
//! and returns a [`CounterfactualComparison`] with collapse-probability
//! delta, dominant-node flip, label flip, and per-node health
//! disagreements.
//!
//! ## Phase 3b — multi-timescale memory (SHIPPED)
//!
//! [`MultiTimescaleEngine`] runs N parallel SSM buffers, each with its
//! own decay rate α. The canonical [`Timescale`] enum gives four named
//! scales: Short (α=0.5), Medium (α=0.85), Long (α=0.95), Structural
//! (α=0.99). Wired into [`ClusterNeuralSystemsSimulator`] via the
//! [`cluster_nss::TemporalMode`] selector — set
//! `ClusterNssConfig::temporal_mode = TemporalMode::MultiAll` to swap
//! the single-timescale engine for the four-buffer one. Backward-
//! compatible: `Single` is the default.
//!
//! ## Phase 3c — scheduler advisory head (SHIPPED)
//!
//! [`SchedulerAdvisor`] enumerates candidate actions
//! ([`AdvisoryAction::DoNothing`], `FailNode`, `RecoverNode`),
//! counterfactually evaluates each via Phase 3a's snapshot/fork
//! machinery, and ranks them by post-action collapse probability.
//! Every recommendation comes with an auditable rationale: the
//! supporting counterfactual trajectory's collapse-tick count. Same
//! snapshot + same NSS produces a bit-identical recommendation
//! (see the `recommend_is_deterministic` test).
//!
//! ## Phase 3d — pipeline-parallelism extensions (SHIPPED)
//!
//! [`PipelineSchedule`] enum on [`GpuTrainingConfig`]:
//! - `GPipe` (default) — Phase 2c baseline
//! - `OneForwardOneBackward` — same bubble fraction as GPipe but
//!   activation memory scales with `stages / microbatches` instead of
//!   1.0
//! - `Interleaved { factor }` — bubble divided by `factor` at the
//!   cost of `factor`× pipeline-shift comms
//!
//! Plus `activation_checkpointing` + `checkpoint_memory_factor` +
//! `checkpoint_recompute_overhead` for the time-vs-memory trade-off.
//! All four schedule + checkpointing combinations are tested for
//! determinism. Default behaviour preserves Phase 2c.
//!
//! ## Phase 3e — advisor extensions (SHIPPED)
//!
//! Added [`AdvisoryAction::ShedLoad { intensity }`],
//! [`AdvisoryAction::AddNode`], [`AdvisoryAction::RemoveNode`]. Added
//! [`NodeHealth::Absent`] for capacity-aware autoscaling (a node slot
//! that exists in the topology but isn't currently part of the active
//! cluster — different from `Failed` in that the failure rollup treats
//! it as Nominal, not Collapse). Added
//! [`SchedulerAdvisor::recommend_per_node`] for per-node ranking so an
//! operator gets a separate recommendation for every node in the
//! cluster.
//!
//! ## What is deferred to later phases
//!
//! - **Phase 4 — Autonomous optimisation engine.** Apply advisor
//!   recommendations automatically; closed-loop policy iteration;
//!   adaptive runtime behaviour with safety bounds.
//! - **Phase 5 — Runtime integration.** MIR trace → NSS pressure
//!   modeling → adaptive-optimization recommendations against the
//!   CJC-Lang MIR executor.
//! - **Phase 5 — Pressure legality verifier.** Static checks that
//!   scheduler actions never create illegal oscillation, that mitigation
//!   actions never violate deterministic replay semantics.
//! - **Phase 5 — cjc-locke detector composition.** `nss_*` custom
//!   detectors (E9500+ codespace) for instability-propagation findings.
//!
//! ## Quick start
//!
//! ```
//! use cjc_nss::{
//!     NeuralSystemsSimulator, NssConfig, NssSeed,
//!     QueueConfig, QueueSimulator,
//! };
//!
//! let seed = NssSeed(42);
//!
//! // 1. Build a deterministic infrastructure trace.
//! let sim_cfg = QueueConfig::default();
//! let mut sim = QueueSimulator::new(sim_cfg, seed).unwrap();
//! let traj = sim.run(64).unwrap();
//!
//! // 2. Train NSS on the trace (Phase 1: single-step prediction).
//! let nss_cfg = NssConfig::default();
//! let mut nss = NeuralSystemsSimulator::from_seed(nss_cfg, seed).unwrap();
//! nss.fit(&traj).unwrap();
//!
//! // 3. Predict next-step instability + attribute dominant cause.
//! let last = traj.last_state().expect("non-empty trajectory");
//! let prediction = nss.predict_next(last).unwrap();
//! assert!(prediction.failure.collapse_probability >= 0.0);
//! assert!(prediction.failure.collapse_probability <= 1.0);
//!
//! // 4. The same (seed, trace, config) must produce a byte-identical run-id.
//! let id_a = prediction.run_id;
//! let mut nss2 = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
//! nss2.fit(&traj).unwrap();
//! let p2 = nss2.predict_next(last).unwrap();
//! assert_eq!(id_a, p2.run_id);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod encoder;
pub mod error;
pub mod failure;
pub mod heads;
pub mod nss;
pub mod pressure;
pub mod propagation;
pub mod replay;
pub mod scheduler;
pub mod seed;
pub mod simulator;
pub mod system;
pub mod temporal;

// Phase 2 — distributed cluster simulator + cluster-aware NSS.
pub mod cluster;
pub mod cluster_nss;
pub mod cluster_simulator;

// Phase 2b — GPU training simulator (data-parallel, lockstep microbatches).
pub mod gpu_training;

// Phase 2d — GradGraph-based cluster-head training.
pub mod cluster_grad;

// Phase 3a — counterfactual forking (snapshot + fork API + comparison).
pub mod counterfactual;

// Phase 3b — multi-timescale pressure memory.
pub mod multi_timescale;

// Phase 3c — counterfactual-based scheduler advisory head.
pub mod advisory;

// Phase 4 — autonomous closed-loop optimisation engine.
pub mod autonomous;

// Phase 5a — MIR-trace → ClusterTrajectory adapter (the compiler-integration bridge).
pub mod mir_adapter;

// Phase 5b — pressure-legality verifier (static analysis on intervention scripts).
pub mod legality;

// Phase 6 — density-matrix-inspired pressure correlation summary.
// Used by `cjc-cana-compress` for the CANA↔NSS bridge: a compression
// decision shifts the diagonal magnitudes (memory pressure down, with a
// reconstruction-risk uptick on the advisory axis) and the resulting
// summary feeds back into CANA's energy ranking. The module itself is
// purely structural — it produces deterministic summaries from
// pressure trajectories and never reaches back into NSS's simulators
// or schedulers.
pub mod density;

pub use encoder::{EncoderConfig, SystemEncoder};
pub use error::NssError;
pub use failure::{FailureKind, FailurePrediction, FailureState};
pub use heads::{
    CausalAttribution, CausalAttributionHead, FailurePredictionHead, HeadConfig,
    PressureContribution,
};
pub use nss::{NeuralSystemsSimulator, NssConfig, NssPrediction};
pub use pressure::{
    Pressure, PressureEdge, PressureField, PressureFlow, PressureGraph, PressureKind,
};
pub use propagation::{PressurePropagator, PropagationConfig};
pub use replay::{PredictionTrace, ReplayValidator, TransitionRecord};
pub use scheduler::{SchedulerAction, SchedulerKind};
pub use seed::{InputHash, NssRunId, NssSeed};
pub use simulator::{QueueConfig, QueueSimulator, QueueSnapshot};
pub use system::{SystemEvent, SystemState, SystemTrajectory};
pub use temporal::{TemporalStateConfig, TemporalStateEngine};

// Phase 2 re-exports.
pub use cluster::{
    ClusterEvent, ClusterSystemState, ClusterTopology, ClusterTrajectory, NetworkLink, NodeHealth,
    NodeId,
};
pub use cluster_nss::{
    cluster_summary_label, ClusterCausalAttribution, ClusterFailurePredictionHead,
    ClusterNeuralSystemsSimulator, ClusterNssConfig, ClusterPrediction, ClusterReplayValidator,
    ClusterTrace, CLUSTER_SUMMARY_FEATURES,
};
pub use cluster_simulator::{ClusterConfig, ClusterSimulator, Intervention, RoutingPolicy};

// Phase 2b re-exports.
pub use gpu_training::{GpuTrainingConfig, GpuTrainingSimulator};

// Phase 3d re-exports — pipeline schedule.
pub use gpu_training::PipelineSchedule;

// Phase 2d re-exports.
pub use cluster_grad::{fit_with_adam, EpochLoss, Optimizer, TrainingHistory};

// Phase 3a re-exports.
pub use counterfactual::{
    run_cluster_counterfactual, ClusterSnapshot, CounterfactualComparison, CounterfactualOutcome,
    GpuTrainingSnapshot,
};

// Phase 3b re-exports.
pub use cluster_nss::TemporalMode;
pub use multi_timescale::{MultiTimescaleConfig, MultiTimescaleEngine, Timescale};

// Phase 3c re-exports.
pub use advisory::{
    AdvisorConfig, AdvisoryAction, AdvisoryCandidate, AdvisoryRanking, SchedulerAdvisor,
};

// Phase 4 re-exports.
pub use autonomous::{
    AutonomousOptimizer, ClosedLoopReport, DecisionOutcome, DecisionRecord, OptimizerConfig,
    SafetyMode,
};

// Phase 5a re-exports.
pub use mir_adapter::{
    adapt_mir_trace_to_cluster_trajectory, build_topology as build_mir_topology, MirAdapterConfig,
    MirAdapterOutput, MirTraceEvent,
};

// Phase 5b re-exports.
pub use legality::{LegalityConfig, LegalityReport, LegalityVerifier, LegalityViolation};

// Phase 6 re-exports — density-matrix-inspired pressure correlation
// summary, consumed by the CANA compression bridge.
pub use density::{PressureCorrelationSummary, PressureDensityState};

/// Crate version stamped into every [`PredictionTrace`]. Phase 1 is
/// `0.1.0` of the NSS surface; bump on every non-backward-compatible
/// change to a primitive, simulator, or neural component.
pub const NSS_MODEL_VERSION: &str = "nss-0.1.0";
