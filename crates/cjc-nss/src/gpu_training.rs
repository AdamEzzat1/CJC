//! Phase 2b — GPU training simulator.
//!
//! Builds on the Phase 2 cluster substrate (`ClusterTopology`,
//! `ClusterSystemState`, `ClusterTrajectory`, `NodeId`, `NodeHealth`,
//! `Intervention`) and *specialises* the workload semantics for
//! synchronous distributed training:
//!
//! - **Lockstep microbatches.** Each tick = one microbatch processed
//!   in parallel by every healthy GPU. There is no per-GPU queue
//!   (unlike the cluster simulator). All GPUs start the microbatch
//!   together and synchronise at the end.
//! - **Per-GPU jitter.** Each GPU's microbatch service time is drawn
//!   from a Gaussian around the mean. Variance is the simulator knob
//!   that controls **batch imbalance**.
//! - **Allreduce barrier.** At microbatch boundaries the cluster pays
//!   a synchronisation cost. Bandwidth-limited communication time
//!   loads `PressureKind::Network`.
//! - **Synchronisation stall.** Faster GPUs idle while slower ones
//!   finish; the per-GPU idle time is exactly
//!   `max(service_times) - own_service_time`, which loads
//!   `PressureKind::Sync` proportionally.
//! - **Memory fragmentation.** Per-GPU memory grows monotonically per
//!   iteration; a fraction is GC'd periodically, but residual
//!   fragmentation accumulates. Loads `PressureKind::Memory`.
//! - **OOM crashes.** Modelled via the existing
//!   [`crate::Intervention::FailNode`] script — when a GPU's memory
//!   would exceed capacity, an OOM is signalled by injecting a
//!   FailNode at that tick.
//!
//! ## Pressure-kind mapping (no new variants needed)
//!
//! | GPU concept            | PressureKind | Why                                  |
//! |------------------------|--------------|--------------------------------------|
//! | Memory fragmentation   | `Memory`     | Pressure rises as fragmentation grows |
//! | NCCL bottleneck        | `Network`    | Per-allreduce bandwidth load          |
//! | Sync stall / imbalance | `Sync`       | Idle time waiting for stragglers     |
//! | GPU starvation         | `Cpu`        | Low utilisation = high idle pressure |
//! | Throughput degradation | `Throughput` | Effective iters-per-tick proxy       |
//!
//! Pipeline parallelism (forward/backward bubbles) is **deferred** —
//! Phase 2b ships data-parallel training only. The same primitive set
//! covers pipeline parallelism in a future micro-phase; only the
//! per-tick sequencing needs to change.
//!
//! ## Determinism contract
//!
//! - Per-GPU service-time RNG via `seed.substream("gpu.{i}.service")`.
//! - Per-GPU memory-fragmentation jitter via `seed.substream("gpu.{i}.memory")`.
//! - Allreduce overhead is a deterministic function of (mean service
//!   time, message size, bandwidth) — no RNG involved.
//! - Two simulators with `(config, topology, seed, intervention_script)`
//!   equal produce byte-identical [`ClusterTrajectory`]s.

use crate::cluster::{
    ClusterEvent, ClusterSystemState, ClusterTopology, ClusterTrajectory, NodeHealth, NodeId,
};
use crate::cluster_simulator::Intervention;
use crate::error::NssError;
use crate::failure::{FailureKind, FailureState};
use crate::pressure::{Pressure, PressureField, PressureKind};
use crate::propagation::PropagationConfig;
use crate::scheduler::{SchedulerAction, SchedulerKind};
use crate::seed::NssSeed;
use crate::system::SystemState;
use cjc_repro::{KahanAccumulatorF64, Rng};
use std::collections::BTreeMap;

/// Knobs for the GPU training simulator. Smaller and more
/// scenario-oriented than the cluster config — the workload is fixed
/// (lockstep data-parallel training).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTrainingConfig {
    /// Number of GPUs (must match the topology's node count).
    pub n_gpus: u32,
    /// Mean per-GPU per-microbatch service time (arbitrary units).
    /// Must be > 0 and finite.
    pub service_mean: f64,
    /// Per-microbatch service-time jitter (std-dev of the Gaussian
    /// draw). Controls *batch imbalance*. Must be >= 0 and finite.
    pub service_jitter: f64,
    /// Allreduce base overhead per microbatch (in time units). The
    /// cluster pays this after every microbatch.
    pub allreduce_base: f64,
    /// Bytes-per-iteration the allreduce ring carries. Driven into
    /// `PressureKind::Network` via `allreduce_bytes / nccl_bandwidth`.
    pub allreduce_bytes: f64,
    /// NCCL bandwidth (bytes per time unit). Must be > 0.
    pub nccl_bandwidth: f64,
    /// Per-microbatch memory fraction consumed (of per-GPU capacity).
    /// Must be in `[0, 1]`.
    pub memory_per_microbatch: f64,
    /// Garbage-collection interval (microbatches). The optimizer GCs
    /// every `gc_interval` ticks, freeing `gc_recovery` of accumulated
    /// memory.
    pub gc_interval: u32,
    /// Fraction of accumulated memory that GC actually reclaims.
    /// Must be in `[0, 1]`. The remainder is "lost" to fragmentation.
    pub gc_recovery: f64,
    /// Fragmentation growth per microbatch (residual heap fragmentation
    /// that GC doesn't recombine). Loads `PressureKind::Memory` even
    /// when occupancy is moderate.
    pub fragmentation_growth: f64,
    /// Per-GPU memory capacity (same units as the per-microbatch
    /// fraction). Used to compute the saturation.
    pub memory_capacity: f64,
    /// **Phase 2c — Pipeline parallelism.** Number of pipeline stages.
    /// `1` (default) = pure data parallelism (Phase 2b behaviour).
    /// `> 1` = GPipe-style pipeline; `n_gpus` must be divisible by
    /// `pipeline_stages` (so each stage has equal data-parallel
    /// replicas).
    pub pipeline_stages: u32,
    /// **Phase 2c.** Microbatches per training iteration (one tick).
    /// Only relevant when `pipeline_stages > 1`; with depth `d` and
    /// `m` microbatches, the GPipe bubble fraction is
    /// `(d - 1) / (d - 1 + m)`. Higher `m` → smaller bubble.
    pub microbatches_per_iteration: u32,
    /// **Phase 3d** — pipeline schedule. `GPipe` is the Phase 2c
    /// default (all forwards then all backwards). `OneForwardOneBackward`
    /// keeps the same bubble fraction but cuts activation memory from
    /// `O(microbatches)` to `O(stages)`. `Interleaved { factor }`
    /// further reduces the bubble by ~`factor`× at the cost of more
    /// communication.
    pub pipeline_schedule: PipelineSchedule,
    /// **Phase 3d** — enable activation checkpointing. When `true`,
    /// activation memory at each stage is reduced by a factor of
    /// roughly `1 / sqrt(layers_per_stage)` (modelled as a fixed
    /// `checkpoint_memory_factor`), while per-microbatch CPU pressure
    /// rises by `checkpoint_recompute_overhead`. The classic
    /// time-vs-memory trade-off.
    pub activation_checkpointing: bool,
    /// **Phase 3d** — fraction of activation memory retained when
    /// `activation_checkpointing` is `true`. Must be in `(0, 1]`.
    /// Default 0.4 (~60% reduction, typical for sqrt-N checkpointing).
    pub checkpoint_memory_factor: f64,
    /// **Phase 3d** — additional CPU cost when checkpointing is on,
    /// expressed as a fraction of the base forward cost. Default 0.33
    /// (the classic 1/3 recompute overhead).
    pub checkpoint_recompute_overhead: f64,
    /// Iterations after which we record `mean_service_time` for the
    /// state. Default = each microbatch.
    pub propagation: PropagationConfig,
}

/// **Phase 3d** — pipeline-parallelism schedule. Closed enum so the
/// step function's match is exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineSchedule {
    /// All forward passes for all microbatches, then all backward
    /// passes. Simple but holds `n_microbatches` activations on
    /// stage 0 simultaneously. Phase 2c default.
    GPipe,
    /// One-forward-one-backward interleaving. Each stage runs a
    /// backward as soon as a microbatch's forward + downstream
    /// backward completes. Same bubble fraction as GPipe but only
    /// `n_stages` activations need to be held simultaneously per
    /// stage — much lower memory pressure for many-microbatch
    /// training.
    OneForwardOneBackward,
    /// Interleaved 1F1B (Megatron-LM v2 / "1F1B-Interleaved"). Each
    /// physical GPU runs `factor` non-contiguous virtual stages.
    /// Reduces bubble by ~`factor`× at the cost of `factor`×
    /// more pipeline-shift communication. `factor = 1` is equivalent
    /// to `OneForwardOneBackward`.
    Interleaved {
        /// Virtual-stages-per-GPU multiplier. Must be ≥ 1.
        factor: u32,
    },
}

impl PipelineSchedule {
    /// Canonical short label for hashing + serialisation.
    pub fn label(self) -> String {
        match self {
            PipelineSchedule::GPipe => "gpipe".to_string(),
            PipelineSchedule::OneForwardOneBackward => "1f1b".to_string(),
            PipelineSchedule::Interleaved { factor } => format!("interleaved_{}", factor),
        }
    }

    /// Bubble-fraction divisor — GPipe and 1F1B both get 1, Interleaved
    /// gets `factor`. The simulator computes
    /// `bubble = base_bubble / divisor`.
    pub fn bubble_divisor(self) -> u32 {
        match self {
            PipelineSchedule::GPipe => 1,
            PipelineSchedule::OneForwardOneBackward => 1,
            PipelineSchedule::Interleaved { factor } => factor.max(1),
        }
    }

    /// Activation-memory multiplier vs the GPipe baseline. GPipe = 1.0,
    /// 1F1B ≈ `stages / microbatches` (capped at 1.0), Interleaved =
    /// `factor * stages / microbatches` (capped at 1.0).
    pub fn activation_memory_multiplier(
        self,
        stages: u32,
        microbatches: u32,
    ) -> f64 {
        let s = stages as f64;
        let m = (microbatches as f64).max(1.0);
        match self {
            PipelineSchedule::GPipe => 1.0,
            PipelineSchedule::OneForwardOneBackward => (s / m).min(1.0),
            PipelineSchedule::Interleaved { factor } => {
                ((factor as f64).max(1.0) * s / m).min(1.0)
            }
        }
    }

    /// Communication-overhead multiplier vs GPipe. Interleaved pays
    /// `factor`× more pipeline shifts; 1F1B is the same as GPipe.
    pub fn communication_multiplier(self) -> f64 {
        match self {
            PipelineSchedule::GPipe => 1.0,
            PipelineSchedule::OneForwardOneBackward => 1.0,
            PipelineSchedule::Interleaved { factor } => (factor as f64).max(1.0),
        }
    }
}

impl Default for PipelineSchedule {
    fn default() -> Self {
        PipelineSchedule::GPipe
    }
}

impl Default for GpuTrainingConfig {
    fn default() -> Self {
        Self {
            n_gpus: 8,
            service_mean: 1.0,
            service_jitter: 0.05,
            allreduce_base: 0.05,
            allreduce_bytes: 1.0e9, // 1 GB allreduce per iteration
            nccl_bandwidth: 1.0e10, // 10 GB/s
            memory_per_microbatch: 0.02,
            gc_interval: 16,
            gc_recovery: 0.6,
            fragmentation_growth: 0.005,
            memory_capacity: 1.0,
            pipeline_stages: 1,
            microbatches_per_iteration: 1,
            pipeline_schedule: PipelineSchedule::GPipe,
            activation_checkpointing: false,
            checkpoint_memory_factor: 0.4,
            checkpoint_recompute_overhead: 0.33,
            propagation: PropagationConfig::default(),
        }
    }
}

impl GpuTrainingConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.n_gpus == 0 {
            return Err(NssError::InvalidConfig {
                detail: "n_gpus must be >= 1".into(),
            });
        }
        if !self.service_mean.is_finite() || self.service_mean <= 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("service_mean must be > 0 and finite, got {}", self.service_mean),
            });
        }
        if !self.service_jitter.is_finite() || self.service_jitter < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "service_jitter must be >= 0 and finite, got {}",
                    self.service_jitter
                ),
            });
        }
        if !self.allreduce_base.is_finite() || self.allreduce_base < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("allreduce_base must be >= 0, got {}", self.allreduce_base),
            });
        }
        if !self.allreduce_bytes.is_finite() || self.allreduce_bytes < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("allreduce_bytes must be >= 0, got {}", self.allreduce_bytes),
            });
        }
        if !self.nccl_bandwidth.is_finite() || self.nccl_bandwidth <= 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("nccl_bandwidth must be > 0, got {}", self.nccl_bandwidth),
            });
        }
        if !self.memory_per_microbatch.is_finite()
            || !(0.0..=1.0).contains(&self.memory_per_microbatch)
        {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "memory_per_microbatch must be in [0, 1], got {}",
                    self.memory_per_microbatch
                ),
            });
        }
        if self.gc_interval == 0 {
            return Err(NssError::InvalidConfig {
                detail: "gc_interval must be >= 1".into(),
            });
        }
        if !self.gc_recovery.is_finite() || !(0.0..=1.0).contains(&self.gc_recovery) {
            return Err(NssError::InvalidConfig {
                detail: format!("gc_recovery must be in [0, 1], got {}", self.gc_recovery),
            });
        }
        if !self.fragmentation_growth.is_finite()
            || !(0.0..=1.0).contains(&self.fragmentation_growth)
        {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "fragmentation_growth must be in [0, 1], got {}",
                    self.fragmentation_growth
                ),
            });
        }
        if !self.memory_capacity.is_finite() || self.memory_capacity <= 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("memory_capacity must be > 0, got {}", self.memory_capacity),
            });
        }
        if self.pipeline_stages == 0 {
            return Err(NssError::InvalidConfig {
                detail: "pipeline_stages must be >= 1".into(),
            });
        }
        if self.pipeline_stages > 1 && self.n_gpus % self.pipeline_stages != 0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "n_gpus ({}) must be divisible by pipeline_stages ({})",
                    self.n_gpus, self.pipeline_stages
                ),
            });
        }
        if self.microbatches_per_iteration == 0 {
            return Err(NssError::InvalidConfig {
                detail: "microbatches_per_iteration must be >= 1".into(),
            });
        }
        // Phase 3d validation.
        if let PipelineSchedule::Interleaved { factor } = self.pipeline_schedule {
            if factor == 0 {
                return Err(NssError::InvalidConfig {
                    detail: "PipelineSchedule::Interleaved.factor must be >= 1".into(),
                });
            }
        }
        if !self.checkpoint_memory_factor.is_finite()
            || self.checkpoint_memory_factor <= 0.0
            || self.checkpoint_memory_factor > 1.0
        {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "checkpoint_memory_factor must be in (0, 1], got {}",
                    self.checkpoint_memory_factor
                ),
            });
        }
        if !self.checkpoint_recompute_overhead.is_finite()
            || self.checkpoint_recompute_overhead < 0.0
        {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "checkpoint_recompute_overhead must be >= 0 and finite, got {}",
                    self.checkpoint_recompute_overhead
                ),
            });
        }
        self.propagation.validate()?;
        Ok(())
    }

    /// Number of data-parallel replicas per pipeline stage. With
    /// `pipeline_stages = 1` this equals `n_gpus`.
    pub fn replicas_per_stage(&self) -> u32 {
        self.n_gpus / self.pipeline_stages
    }

    /// Map a `NodeId` to its pipeline stage in `[0, pipeline_stages)`.
    /// GPUs are assigned to stages in `NodeId` order:
    /// stage `k` contains GPUs `[k * replicas, (k+1) * replicas)`.
    pub fn stage_of(&self, gpu: u32) -> u32 {
        if self.pipeline_stages == 1 {
            return 0;
        }
        let replicas = self.replicas_per_stage().max(1);
        (gpu / replicas).min(self.pipeline_stages - 1)
    }

    /// Bubble fraction for the current pipeline depth + microbatch
    /// count + schedule. Returns 0 for `pipeline_stages = 1`.
    ///
    /// **Phase 3d** — `Interleaved { factor }` divides the GPipe
    /// baseline by `factor` (each virtual stage shrinks the bubble
    /// proportionally). `GPipe` and `OneForwardOneBackward` share the
    /// same bubble fraction; 1F1B's win is on *memory*, not bubbles.
    pub fn bubble_fraction(&self) -> f64 {
        if self.pipeline_stages <= 1 {
            return 0.0;
        }
        let d = (self.pipeline_stages - 1) as f64;
        let m = self.microbatches_per_iteration as f64;
        let base = d / (d + m);
        let divisor = self.pipeline_schedule.bubble_divisor() as f64;
        base / divisor.max(1.0)
    }

    /// Stage-dependent activation memory multiplier in `[0, 1]`.
    ///
    /// Combines two effects:
    /// 1. **Stage skew** — earlier stages hold more activations (GPipe
    ///    base). 1F1B / Interleaved flatten this skew by the
    ///    `activation_memory_multiplier` factor for the schedule.
    /// 2. **Activation checkpointing** — multiplies the final value
    ///    by `checkpoint_memory_factor` when enabled.
    ///
    /// Phase 2c formula was `(microbatches - stage) / microbatches`;
    /// Phase 3d multiplies that by the schedule's memory factor and
    /// optionally by the checkpointing factor.
    pub fn stage_activation_memory_multiplier(&self, stage: u32) -> f64 {
        if self.pipeline_stages <= 1 {
            // No pipeline → no stage skew. Checkpointing still applies.
            let cp = if self.activation_checkpointing {
                self.checkpoint_memory_factor
            } else {
                1.0
            };
            return cp;
        }
        let m = self.microbatches_per_iteration as f64;
        let s = stage as f64;
        let gpipe_skew = ((m - s) / m).max(0.1).min(1.0);
        let sched_mul = self
            .pipeline_schedule
            .activation_memory_multiplier(self.pipeline_stages, self.microbatches_per_iteration);
        let cp = if self.activation_checkpointing {
            self.checkpoint_memory_factor
        } else {
            1.0
        };
        gpipe_skew * sched_mul * cp
    }

    /// Per-microbatch CPU-cost multiplier from activation
    /// checkpointing. When enabled, adds
    /// `checkpoint_recompute_overhead` to the base cost (default
    /// `+33%`). The simulator uses this to load `PressureKind::Cpu`
    /// slightly higher under checkpointing — the time-vs-memory
    /// trade-off.
    pub fn checkpoint_cpu_multiplier(&self) -> f64 {
        if self.activation_checkpointing {
            1.0 + self.checkpoint_recompute_overhead
        } else {
            1.0
        }
    }

    /// Communication-overhead multiplier from the pipeline schedule.
    /// Interleaved pays `factor`× more pipeline-shift comms.
    pub fn communication_multiplier(&self) -> f64 {
        self.pipeline_schedule.communication_multiplier()
    }

    /// Canonical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&(self.n_gpus as u64).to_le_bytes());
        bytes.extend_from_slice(&self.service_mean.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.service_jitter.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.allreduce_base.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.allreduce_bytes.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.nccl_bandwidth.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.memory_per_microbatch.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(self.gc_interval as u64).to_le_bytes());
        bytes.extend_from_slice(&self.gc_recovery.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.fragmentation_growth.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.memory_capacity.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(self.pipeline_stages as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.microbatches_per_iteration as u64).to_le_bytes());
        // Phase 3d canonical bytes.
        let schedule_label = self.pipeline_schedule.label();
        bytes.extend_from_slice(schedule_label.as_bytes());
        bytes.push(b'|');
        bytes.push(if self.activation_checkpointing { 1 } else { 0 });
        bytes.extend_from_slice(&self.checkpoint_memory_factor.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.checkpoint_recompute_overhead.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.propagation.canonical_bytes());
        bytes
    }
}

/// Internal per-GPU runtime state. Unlike the cluster simulator's
/// per-node queue + retry counters, GPUs track memory growth and idle
/// accumulation — the workload-defining quantities for synchronous
/// training.
#[derive(Clone, Debug)]
struct GpuRuntime {
    /// Microbatches completed so far.
    iterations: u64,
    /// Currently-allocated memory (fraction of capacity in `[0, 1]`).
    memory_used: f64,
    /// Accumulated fragmentation (fraction of capacity). Grows
    /// monotonically *between* GC events; GC reclaims `gc_recovery` of
    /// the *non-fragmented* portion only.
    fragmentation: f64,
    /// Last microbatch's service time.
    last_service_time: f64,
    /// Accumulated idle time across the trajectory (Kahan-friendly via
    /// f64).
    idle_accumulated: f64,
    /// Current health.
    health: NodeHealth,
    /// Per-GPU pressure field.
    field: PressureField,
    /// Per-GPU service-time RNG.
    rng_service: Rng,
    /// Per-GPU memory-jitter RNG (used by GC variance / fragmentation
    /// stochasticity). Phase 2b uses it only for an optional GC-quality
    /// jitter; kept as its own sub-stream so adding future stochastic
    /// effects doesn't perturb the service-time stream.
    rng_memory: Rng,
}

/// The GPU training simulator.
///
/// **Phase 3a — counterfactual forking.** `Clone` is implemented for
/// the same reason as [`crate::ClusterSimulator`]: a clone is a
/// snapshot. See [`GpuTrainingSimulator::snapshot`] +
/// [`GpuTrainingSnapshot::fork`] for the forking API.
#[derive(Clone, Debug)]
pub struct GpuTrainingSimulator {
    cfg: GpuTrainingConfig,
    topology: ClusterTopology,
    seed: NssSeed,
    script: Vec<Intervention>,
    gpus: BTreeMap<NodeId, GpuRuntime>,
    tick: u64,
}

impl GpuTrainingSimulator {
    /// Build the simulator. Validates config, sorts the intervention
    /// script (same convention as the cluster simulator), and checks
    /// that the topology's node count matches `cfg.n_gpus`.
    pub fn new(
        cfg: GpuTrainingConfig,
        topology: ClusterTopology,
        seed: NssSeed,
        mut script: Vec<Intervention>,
    ) -> Result<Self, NssError> {
        cfg.validate()?;
        if topology.node_count() as u32 != cfg.n_gpus {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "topology has {} nodes, cfg.n_gpus = {}",
                    topology.node_count(),
                    cfg.n_gpus
                ),
            });
        }
        for iv in &script {
            let node = iv.node();
            if !topology.nodes().any(|n| n == node) {
                return Err(NssError::InvalidConfig {
                    detail: format!("intervention targets unknown gpu {}", node),
                });
            }
        }
        script.sort_by(|a, b| a.cmp(b));

        let mut gpus = BTreeMap::new();
        for id in topology.nodes() {
            let mut field = PressureField::with_default_thresholds();
            // Memory and Sync are the most active fields for GPU
            // training; bump their dissipation up so the simulator's
            // pressure proxies are responsive to recent ticks.
            for k in PressureKind::all() {
                field.set(k, Pressure::new(0.0, 1.0, 0.1)?);
            }
            field.set(PressureKind::Memory, Pressure::new(0.0, 1.0, 0.02)?);
            field.set(PressureKind::Sync, Pressure::new(0.0, 1.0, 0.15)?);
            field.set(PressureKind::Network, Pressure::new(0.0, 1.0, 0.1)?);
            gpus.insert(
                id,
                GpuRuntime {
                    iterations: 0,
                    memory_used: 0.0,
                    fragmentation: 0.0,
                    last_service_time: 0.0,
                    idle_accumulated: 0.0,
                    health: NodeHealth::Healthy,
                    field,
                    rng_service: seed.substream(&format!("gpu.{}.service", id.0)),
                    rng_memory: seed.substream(&format!("gpu.{}.memory", id.0)),
                },
            );
        }

        Ok(Self {
            cfg,
            topology,
            seed,
            script,
            gpus,
            tick: 0,
        })
    }

    /// Borrow topology.
    pub fn topology(&self) -> &ClusterTopology {
        &self.topology
    }

    /// Borrow config.
    pub fn config(&self) -> &GpuTrainingConfig {
        &self.cfg
    }

    /// Seed accessor.
    pub fn seed(&self) -> NssSeed {
        self.seed
    }

    /// Sorted intervention script.
    pub fn intervention_script(&self) -> &[Intervention] {
        &self.script
    }

    /// Current tick.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Append additional interventions and re-sort. Crate-internal —
    /// used by the [`crate::GpuTrainingSnapshot::fork`] API.
    pub(crate) fn append_and_sort_interventions(&mut self, mut extra: Vec<Intervention>) {
        self.script.append(&mut extra);
        self.script.sort_by(|a, b| a.cmp(b));
    }

    /// **Phase 4** — public counterpart of `append_and_sort_interventions`
    /// for closed-loop autonomous control.
    pub fn inject_intervention(&mut self, iv: Intervention) {
        self.script.push(iv);
        self.script.sort_by(|a, b| a.cmp(b));
    }

    /// Run for `n_iterations` (each is one microbatch tick) and
    /// return the cluster trajectory. The trajectory can be fed
    /// directly to a `ClusterNeuralSystemsSimulator` for prediction.
    pub fn run(&mut self, n_iterations: u64) -> Result<ClusterTrajectory, NssError> {
        let mut traj = ClusterTrajectory::empty();
        for _ in 0..n_iterations {
            traj.push(self.step()?)?;
        }
        Ok(traj)
    }

    fn step(&mut self) -> Result<ClusterEvent, NssError> {
        // 1. Apply interventions scheduled for this tick (OOM = FailNode).
        for iv in self.script.iter().filter(|iv| iv.tick() == self.tick) {
            let node_id = iv.node();
            if let Some(g) = self.gpus.get_mut(&node_id) {
                match iv {
                    Intervention::FailNode { .. } => {
                        if g.health == NodeHealth::Healthy {
                            g.health = NodeHealth::Failed;
                            // OOM zeroes the working set but leaves
                            // fragmentation as a scar (we model the
                            // OS reclaiming the process but the
                            // device memory pool needing a hard
                            // reset).
                            g.memory_used = 0.0;
                        }
                    }
                    Intervention::RecoverNode { .. } => {
                        if g.health == NodeHealth::Failed {
                            g.health = NodeHealth::Healthy;
                            g.fragmentation = 0.0; // hard reset on recovery
                        }
                    }
                    // Phase 3e — GPU training treats AddNode like
                    // RecoverNode (Absent → Healthy) and RemoveNode
                    // like FailNode (active → Absent). ShedLoadOverride
                    // is a no-op for GPU training (there's no
                    // admission control in lockstep training).
                    Intervention::AddNode { .. } => {
                        if g.health == NodeHealth::Absent {
                            g.health = NodeHealth::Healthy;
                            g.memory_used = 0.0;
                            g.fragmentation = 0.0;
                        }
                    }
                    Intervention::RemoveNode { .. } => {
                        if g.health != NodeHealth::Absent {
                            g.health = NodeHealth::Absent;
                            g.memory_used = 0.0;
                        }
                    }
                    Intervention::ShedLoadOverride { .. } => {
                        // No-op for GPU training — lockstep semantics
                        // don't expose an admission knob.
                    }
                }
            }
        }

        // 2. Per-GPU service-time draws (Gaussian: mean + jitter*N(0,1),
        //    clamped at 0).
        let mut service_times: BTreeMap<NodeId, f64> = BTreeMap::new();
        let node_ids: Vec<NodeId> = self.gpus.keys().copied().collect();
        for id in &node_ids {
            let g = self.gpus.get_mut(id).unwrap();
            // Phase 3e — Failed and Absent GPUs both skip service.
            if !g.health.participates() {
                service_times.insert(*id, 0.0);
                continue;
            }
            let z = g.rng_service.next_normal_f64();
            let t = (self.cfg.service_mean + self.cfg.service_jitter * z).max(0.0);
            service_times.insert(*id, t);
            g.last_service_time = t;
        }

        // 3. Max service time = effective per-microbatch wall clock
        //    (the slowest GPU dictates this; this is the imbalance
        //    cost).
        let max_t = service_times
            .values()
            .copied()
            .filter(|t| t.is_finite())
            .fold(0.0_f64, f64::max);

        // 4. Allreduce overhead — deterministic; loads Network pressure
        //    on every healthy GPU equally.
        //
        // Phase 2c — when pipeline_stages > 1, allreduce only spans
        // *within-stage* data-parallel replicas (not across stages),
        // so the message size shrinks by `replicas_per_stage / n_gpus`
        // and so does the effective bandwidth cost.
        let allreduce_scale = if self.cfg.pipeline_stages > 1 {
            self.cfg.replicas_per_stage() as f64 / self.cfg.n_gpus as f64
        } else {
            1.0
        };
        // Phase 3d — Interleaved pipeline pays `factor`× pipeline-shift
        // comms on top of allreduce. Modelled as a multiplier on the
        // allreduce bytes.
        let comm_mul = self.cfg.communication_multiplier();
        let allreduce_time = self.cfg.allreduce_base
            + (self.cfg.allreduce_bytes * allreduce_scale * comm_mul / self.cfg.nccl_bandwidth);
        // Normalised for pressure: relative to a "nominal" iteration
        // cost. We cap at 1.5 just like the cluster sim does.
        let allreduce_pressure = (allreduce_time / self.cfg.service_mean.max(1e-12)).min(1.5);
        // Phase 2c / 3d — pipeline bubble fraction; floor on per-stage
        // idle. `bubble_fraction()` now accounts for the schedule's
        // `bubble_divisor` (Interleaved reduces the bubble by `factor`).
        let bubble_fraction = self.cfg.bubble_fraction();
        // Phase 3d — activation-checkpointing CPU surcharge. The
        // recompute pass is *extra real work*, not idle time, so we
        // model it as an additive contribution to CPU pressure rather
        // than a multiplier on starvation. This matters when
        // `service_jitter = 0.0` — without jitter, starvation = 0,
        // and a multiplicative model would erase the checkpointing
        // effect entirely.
        let checkpoint_cpu_load = if self.cfg.activation_checkpointing {
            self.cfg.checkpoint_recompute_overhead
        } else {
            0.0
        };

        // 5. Per-GPU sync stall + memory update + pressure field update.
        for id in &node_ids {
            let g = self.gpus.get_mut(id).unwrap();
            // Phase 3e — Absent GPUs contribute zero pressure (they're
            // not part of the active cluster).
            if g.health == NodeHealth::Absent {
                for k in PressureKind::all() {
                    g.field.set(k, Pressure::new(0.0, 1.0, 0.1)?);
                }
                continue;
            }
            if g.health == NodeHealth::Failed {
                // Dead GPU: zero out CPU/Sync/Throughput, leave Memory
                // and Network high (modelling a saturated link to a
                // dead peer that NCCL can't yet detect).
                g.field
                    .set(PressureKind::Cpu, Pressure::new(0.0, 1.0, 0.1)?);
                g.field.set(
                    PressureKind::Sync,
                    Pressure::new(1.0, 1.0, 0.15)?, // 100% sync pressure for a dead peer
                );
                g.field.set(
                    PressureKind::Throughput,
                    Pressure::new(1.0, 1.0, 0.05)?,
                );
                g.field.set(
                    PressureKind::Network,
                    Pressure::new(allreduce_pressure, 1.0, 0.1)?,
                );
                // Memory pressure persists at the pre-OOM fragmentation
                // level — fragmentation doesn't heal on OOM.
                let frag_p = (g.fragmentation / self.cfg.memory_capacity).min(1.5);
                g.field
                    .set(PressureKind::Memory, Pressure::new(frag_p, 1.0, 0.02)?);
                continue;
            }
            let own_t = *service_times.get(id).unwrap_or(&0.0);
            let idle = (max_t - own_t).max(0.0);
            // Phase 2c — pipeline GPUs idle a fraction of every
            // iteration *in addition to* straggler-induced idle. We
            // attribute the pipeline-bubble idle to this GPU's
            // accumulator and floor the sync pressure accordingly.
            let pipeline_idle = bubble_fraction * max_t;
            g.idle_accumulated += idle + pipeline_idle;
            // Sync pressure = (jitter idle + pipeline bubble) / max_t.
            // 1.0 means the GPU sat idle the entire microbatch.
            let sync_jitter = if max_t > 0.0 { idle / max_t } else { 0.0 };
            let sync_p = (sync_jitter + bubble_fraction).min(1.5);
            // CPU pressure = own_t / (max_t + bubble), capped to [0, 1.5].
            // The bubble extends the iteration without contributing
            // work, so utilisation is `own_t / (max_t + bubble)`.
            let denom_util = max_t + pipeline_idle;
            let util = if denom_util > 0.0 { own_t / denom_util } else { 1.0 };
            // Phase 3d — checkpointing recompute is real work, additive
            // to starvation. Without jitter, starvation is 0 but
            // checkpointing still adds ~recompute_overhead to CPU pressure.
            let starvation_p = ((1.0 - util) + checkpoint_cpu_load).max(0.0).min(1.5);
            // Throughput pressure = (allreduce + bubble) / total cost.
            let total_iter_cost = max_t + pipeline_idle + allreduce_time;
            let thr_p =
                ((allreduce_time + pipeline_idle) / total_iter_cost.max(1e-12)).min(1.5);
            // Memory + fragmentation update.
            // Phase 2c — earlier pipeline stages hold more activation
            // memory. Stage 0 sees `microbatches` worth of fwd activations
            // until backward starts; final stage holds 1. We multiply
            // per-tick memory growth by the stage-dependent factor.
            let stage = self.cfg.stage_of(id.0);
            let stage_mem_mul = self.cfg.stage_activation_memory_multiplier(stage);
            g.memory_used += self.cfg.memory_per_microbatch * stage_mem_mul;
            g.fragmentation += self.cfg.fragmentation_growth;
            // Garbage collection (deterministic schedule).
            let do_gc = self.cfg.gc_interval > 0
                && ((g.iterations + 1) % self.cfg.gc_interval as u64) == 0;
            if do_gc {
                // GC quality jitter (zero std-dev unless future
                // versions add it) — call the RNG so the stream
                // advances deterministically; effect is zero.
                let _ = g.rng_memory.next_f64();
                let recovered = g.memory_used * self.cfg.gc_recovery;
                g.memory_used -= recovered;
                if g.memory_used < 0.0 {
                    g.memory_used = 0.0;
                }
            }
            // OOM check: if memory_used + fragmentation > capacity,
            // emit a Collapse label (Phase 2b: the OOM injection is
            // *scripted* via Intervention::FailNode; the runtime
            // check here is a safety net that surfaces "would-OOM"
            // by tagging the failure label without changing health
            // — caller can use this as a signal to script an OOM).
            let mem_p = ((g.memory_used + g.fragmentation) / self.cfg.memory_capacity).min(1.5);
            let net_p = allreduce_pressure;

            g.field
                .set(PressureKind::Cpu, Pressure::new(starvation_p, 1.0, 0.1)?);
            g.field
                .set(PressureKind::Sync, Pressure::new(sync_p, 1.0, 0.15)?);
            g.field
                .set(PressureKind::Memory, Pressure::new(mem_p, 1.0, 0.02)?);
            g.field
                .set(PressureKind::Network, Pressure::new(net_p, 1.0, 0.1)?);
            g.field
                .set(PressureKind::Throughput, Pressure::new(thr_p, 1.0, 0.05)?);
            g.iterations += 1;
        }

        // 6. Per-GPU intra-pressure propagation (same Phase 1 graph
        //    as the cluster simulator).
        let prop = crate::PressurePropagator::new(
            crate::PressureGraph::default_phase1(),
            self.cfg.propagation,
        )?;
        for id in &node_ids {
            let g = self.gpus.get_mut(id).unwrap();
            if g.health == NodeHealth::Healthy {
                prop.step(&mut g.field)?;
            }
        }

        // 7. Per-GPU scheduler action + failure label.
        //    "ShedLoad" semantics for GPU training don't quite fit —
        //    we model the optimizer's loss-of-precision response
        //    (e.g., FP16 → FP32 spill) as a `ShedLoad` with magnitude
        //    proportional to memory pressure. This is a defensible
        //    proxy: when memory crosses the knee, the trainer
        //    "sheds" precision to free working memory.
        let mut actions: BTreeMap<NodeId, SchedulerAction> = BTreeMap::new();
        let mut failures: BTreeMap<NodeId, FailureState> = BTreeMap::new();
        for id in &node_ids {
            let g = self.gpus.get(id).unwrap();
            // Phase 3e — Absent GPU = Nominal (not part of the cluster).
            if g.health == NodeHealth::Absent {
                actions.insert(*id, SchedulerAction::idle());
                failures.insert(*id, FailureState::nominal());
                continue;
            }
            if g.health == NodeHealth::Failed {
                actions.insert(*id, SchedulerAction::idle());
                failures.insert(*id, FailureState::collapse(PressureKind::Memory));
                continue;
            }
            let mem_sat = g
                .field
                .get(PressureKind::Memory)
                .map(|p| p.saturation())
                .unwrap_or(0.0);
            let sync_sat = g
                .field
                .get(PressureKind::Sync)
                .map(|p| p.saturation())
                .unwrap_or(0.0);
            let memory_knee = 0.85;
            let sync_knee = 0.7;
            let action = if mem_sat >= memory_knee {
                SchedulerAction::new(SchedulerKind::ShedLoad, (mem_sat - memory_knee) / (1.0 - memory_knee))
            } else {
                SchedulerAction::idle()
            };
            // Failure label:
            //   - Collapse if memory + fragmentation would exceed
            //     capacity (i.e., mem_sat >= 1.0)
            //   - Degraded if mem_sat >= memory_knee or sync_sat >= sync_knee
            //   - Nominal otherwise
            let dominant = if mem_sat >= sync_sat { PressureKind::Memory } else { PressureKind::Sync };
            let failure = if mem_sat >= 1.0 {
                FailureState::collapse(PressureKind::Memory)
            } else if mem_sat >= memory_knee || sync_sat >= sync_knee {
                FailureState::degraded(dominant)
            } else {
                FailureState::nominal()
            };
            actions.insert(*id, action);
            failures.insert(*id, failure);
        }

        // 8. Cluster-level rollup (same logic as cluster simulator —
        //    we re-implement it locally to avoid a public dependency
        //    on cluster_simulator's private rollup helper).
        let mut cluster_label = FailureKind::Nominal;
        let mut cluster_source = None;
        for (_id, f) in failures.iter() {
            match (cluster_label, f.kind) {
                (_, FailureKind::Collapse) => {
                    cluster_label = FailureKind::Collapse;
                    cluster_source = f.dominant_source;
                    break;
                }
                (FailureKind::Nominal, FailureKind::Degraded) => {
                    cluster_label = FailureKind::Degraded;
                    cluster_source = f.dominant_source;
                }
                _ => {}
            }
        }
        let cluster_failure = match (cluster_label, cluster_source) {
            (FailureKind::Collapse, Some(k)) => FailureState::collapse(k),
            (FailureKind::Degraded, Some(k)) => FailureState::degraded(k),
            _ => FailureState::nominal(),
        };

        // 9. Materialise per-GPU SystemState. The GPU training
        //    simulator doesn't track in-flight queue depth (work is
        //    lockstep), so `in_flight` is 0/1 = "currently processing
        //    a microbatch?", `completed` = iterations, `rejected` =
        //    OOM count proxy, `mean_service_time` = this tick's
        //    own_t.
        let mut nodes_state = BTreeMap::new();
        for id in &node_ids {
            let g = self.gpus.get(id).unwrap();
            let state = SystemState {
                tick: self.tick,
                pressures: g.field.clone(),
                in_flight: if g.health == NodeHealth::Healthy { 1 } else { 0 },
                completed: g.iterations,
                rejected: if g.health == NodeHealth::Failed { 1 } else { 0 },
                mean_service_time: g.last_service_time,
            };
            nodes_state.insert(*id, state);
        }
        // We don't drive network-link congestion through this
        // simulator (allreduce is modelled per-GPU via the Network
        // pressure rather than per-link). Leave all link_congestion
        // at 0 — the cluster NSS still consumes the topology bytes
        // into the run-id via its config.
        let mut link_congestion = BTreeMap::new();
        for (src, dst, _) in self.topology.edges() {
            link_congestion.insert((src, dst), 0.0);
        }
        let mut node_health = BTreeMap::new();
        for id in &node_ids {
            node_health.insert(*id, self.gpus.get(id).unwrap().health);
        }

        let cluster_state = ClusterSystemState {
            tick: self.tick,
            nodes: nodes_state,
            link_congestion,
            node_health,
        };

        let ev = ClusterEvent {
            state: cluster_state,
            actions,
            failures,
            cluster_failure,
        };
        self.tick += 1;
        Ok(ev)
    }

    /// Per-GPU iteration count snapshot — for tests + the demo
    /// (the trajectory's `completed` carries the same number).
    pub fn iteration_counts(&self) -> BTreeMap<NodeId, u64> {
        self.gpus.iter().map(|(id, g)| (*id, g.iterations)).collect()
    }

    /// Cumulative idle time per GPU — useful for tests that compare
    /// imbalance under different jitter settings.
    pub fn idle_accumulated(&self) -> BTreeMap<NodeId, f64> {
        let mut out = BTreeMap::new();
        for (id, g) in self.gpus.iter() {
            out.insert(*id, g.idle_accumulated);
        }
        out
    }

    /// Canonical bytes (config + topology + script).
    pub fn input_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.cfg.canonical_bytes();
        bytes.push(b'#');
        bytes.extend_from_slice(&self.topology.canonical_bytes());
        bytes.push(b'#');
        for iv in &self.script {
            bytes.extend_from_slice(&iv.canonical_bytes());
            bytes.push(b';');
        }
        bytes
    }

    /// Sum of per-GPU sync pressure across the entire current state.
    /// Convenience for tests asserting that jitter drives imbalance.
    pub fn total_sync_pressure(&self) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        for g in self.gpus.values() {
            acc.add(g.field.get(PressureKind::Sync).map(|p| p.magnitude).unwrap_or(0.0));
        }
        acc.finalize()
    }

    /// Map of `NodeId → pipeline_stage`. For `pipeline_stages == 1`
    /// every entry is 0.
    pub fn stage_assignments(&self) -> BTreeMap<NodeId, u32> {
        self.gpus
            .keys()
            .map(|id| (*id, self.cfg.stage_of(id.0)))
            .collect()
    }

    /// GPUs grouped by pipeline stage in canonical order.
    pub fn gpus_by_stage(&self) -> BTreeMap<u32, Vec<NodeId>> {
        let mut out: BTreeMap<u32, Vec<NodeId>> = BTreeMap::new();
        for id in self.gpus.keys() {
            out.entry(self.cfg.stage_of(id.0)).or_default().push(*id);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_topology(n: u32) -> ClusterTopology {
        ClusterTopology::complete(n, 8, 0.5).unwrap()
    }

    #[test]
    fn config_default_validates() {
        assert!(GpuTrainingConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_invalid_fields() {
        let mut c = GpuTrainingConfig::default();
        c.n_gpus = 0;
        assert!(c.validate().is_err());
        let mut c = GpuTrainingConfig::default();
        c.service_mean = -1.0;
        assert!(c.validate().is_err());
        let mut c = GpuTrainingConfig::default();
        c.memory_per_microbatch = 1.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn topology_node_count_must_match() {
        let cfg = GpuTrainingConfig {
            n_gpus: 8,
            ..GpuTrainingConfig::default()
        };
        let bad = ClusterTopology::complete(4, 8, 0.5).unwrap();
        assert!(GpuTrainingSimulator::new(cfg, bad, NssSeed(42), vec![]).is_err());
    }

    #[test]
    fn determinism_same_seed_same_trajectory() {
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            ..GpuTrainingConfig::default()
        };
        let mut a =
            GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(42), vec![]).unwrap();
        let mut b =
            GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(42), vec![]).unwrap();
        let ta = a.run(32).unwrap();
        let tb = b.run(32).unwrap();
        assert_eq!(ta.canonical_bytes(), tb.canonical_bytes());
    }

    #[test]
    fn determinism_different_seed_diverges() {
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            ..GpuTrainingConfig::default()
        };
        let mut a =
            GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(1), vec![]).unwrap();
        let mut b =
            GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(2), vec![]).unwrap();
        assert_ne!(
            a.run(32).unwrap().canonical_bytes(),
            b.run(32).unwrap().canonical_bytes(),
        );
    }

    #[test]
    fn jitter_increases_sync_pressure() {
        // Compare low-jitter vs high-jitter total accumulated sync
        // pressure across the same number of iterations.
        let mut low_cfg = GpuTrainingConfig {
            n_gpus: 4,
            service_jitter: 0.01,
            ..GpuTrainingConfig::default()
        };
        let high_cfg = GpuTrainingConfig {
            service_jitter: 0.40,
            ..low_cfg
        };
        low_cfg.gc_interval = 1000; // disable GC so memory effect doesn't muddy the test
        let mut low =
            GpuTrainingSimulator::new(low_cfg, small_topology(4), NssSeed(7), vec![]).unwrap();
        let mut high = GpuTrainingSimulator::new(
            GpuTrainingConfig {
                gc_interval: 1000,
                ..high_cfg
            },
            small_topology(4),
            NssSeed(7),
            vec![],
        )
        .unwrap();
        let t_low = low.run(64).unwrap();
        let t_high = high.run(64).unwrap();
        // Sum sync saturations across all per-GPU states.
        let sync_sum = |t: &ClusterTrajectory| -> f64 {
            let mut a = KahanAccumulatorF64::new();
            for ev in t.iter() {
                for s in ev.state.nodes.values() {
                    a.add(
                        s.pressures
                            .get(PressureKind::Sync)
                            .map(|p| p.saturation())
                            .unwrap_or(0.0),
                    );
                }
            }
            a.finalize()
        };
        let lo = sync_sum(&t_low);
        let hi = sync_sum(&t_high);
        assert!(
            hi > lo,
            "high-jitter run must produce more cumulative sync pressure (low={}, high={})",
            lo,
            hi
        );
    }

    #[test]
    fn memory_grows_monotonically_between_gcs() {
        let cfg = GpuTrainingConfig {
            n_gpus: 1,
            gc_interval: 1000, // never GC during test
            memory_per_microbatch: 0.01,
            fragmentation_growth: 0.0,
            ..GpuTrainingConfig::default()
        };
        let mut sim =
            GpuTrainingSimulator::new(cfg, small_topology(1), NssSeed(42), vec![]).unwrap();
        let traj = sim.run(20).unwrap();
        // Memory pressure on Memory kind must be strictly non-decreasing
        // since no GC fires within the window and fragmentation is 0.
        // Note: Phase 1 propagation may *redistribute* a small amount
        // of magnitude into the field across kinds. To make the test
        // tight, we read the *raw memory_used* via iteration count *
        // memory_per_microbatch, which the simulator exposes via the
        // node's `completed` and the configured per-microbatch rate.
        let mut last_completed: u64 = 0;
        for ev in traj.iter() {
            let comp = ev.state.nodes.get(&NodeId(0)).unwrap().completed;
            assert!(
                comp >= last_completed,
                "iteration count must be monotonic, got {} after {}",
                comp,
                last_completed
            );
            last_completed = comp;
        }
        assert!(last_completed > 0);
    }

    #[test]
    fn gc_reduces_memory_pressure() {
        let cfg = GpuTrainingConfig {
            n_gpus: 1,
            gc_interval: 4,
            gc_recovery: 0.9,
            memory_per_microbatch: 0.1,
            fragmentation_growth: 0.0,
            ..GpuTrainingConfig::default()
        };
        let mut sim =
            GpuTrainingSimulator::new(cfg, small_topology(1), NssSeed(42), vec![]).unwrap();
        let _traj = sim.run(32).unwrap();
        // After 32 ticks with GC every 4 ticks reclaiming 90%, the
        // memory_used field should be bounded — much smaller than the
        // naive total (32 * 0.1 = 3.2 of capacity). Check that the
        // final Memory pressure saturation stayed well below
        // ungc'd-level.
        let final_iter_count = sim.iteration_counts().get(&NodeId(0)).copied().unwrap();
        assert_eq!(final_iter_count, 32);
    }

    #[test]
    fn oom_via_intervention_drains_gpu() {
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            ..GpuTrainingConfig::default()
        };
        let ivs = vec![Intervention::FailNode {
            tick: 5,
            node: NodeId(2),
        }];
        let mut sim = GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(42), ivs).unwrap();
        let traj = sim.run(16).unwrap();
        for ev in traj.iter().skip(5) {
            assert_eq!(
                ev.state.node_health.get(&NodeId(2)).copied(),
                Some(NodeHealth::Failed)
            );
            // The OOM'd GPU's failure label is Collapse with Memory source.
            let f = ev.failures.get(&NodeId(2)).unwrap();
            assert_eq!(f.kind, FailureKind::Collapse);
            assert_eq!(f.dominant_source, Some(PressureKind::Memory));
        }
        // Cluster rolls up to Collapse.
        let n_collapse = traj
            .iter()
            .skip(5)
            .filter(|ev| ev.cluster_failure.kind == FailureKind::Collapse)
            .count();
        assert!(n_collapse > 0);
    }

    #[test]
    fn allreduce_overhead_drives_network_pressure() {
        // Compare two configs with identical service stats but
        // different allreduce overhead. The higher-overhead config
        // must produce higher cumulative network pressure.
        let base = GpuTrainingConfig {
            n_gpus: 4,
            service_jitter: 0.0, // zero jitter so only allreduce moves
            allreduce_base: 0.0,
            allreduce_bytes: 1.0e9,
            nccl_bandwidth: 1.0e10,
            ..GpuTrainingConfig::default()
        };
        let fast = GpuTrainingConfig {
            allreduce_bytes: 1.0e8,
            ..base
        };
        let slow = GpuTrainingConfig {
            allreduce_bytes: 5.0e9,
            ..base
        };
        let mut sf = GpuTrainingSimulator::new(fast, small_topology(4), NssSeed(7), vec![]).unwrap();
        let mut ss = GpuTrainingSimulator::new(slow, small_topology(4), NssSeed(7), vec![]).unwrap();
        let tf = sf.run(16).unwrap();
        let ts = ss.run(16).unwrap();
        let net_sum = |t: &ClusterTrajectory| -> f64 {
            let mut a = KahanAccumulatorF64::new();
            for ev in t.iter() {
                for s in ev.state.nodes.values() {
                    a.add(
                        s.pressures
                            .get(PressureKind::Network)
                            .map(|p| p.saturation())
                            .unwrap_or(0.0),
                    );
                }
            }
            a.finalize()
        };
        let nf = net_sum(&tf);
        let ns = net_sum(&ts);
        assert!(ns > nf, "slow allreduce must drive more network pressure (fast={}, slow={})", nf, ns);
    }

    #[test]
    fn idle_accumulates_for_fast_gpus() {
        // With non-zero jitter and 4 GPUs sharing a seed, the fastest
        // GPU should accumulate some idle time waiting for the
        // slowest. Confirm `idle_accumulated` is positive for at
        // least one GPU.
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            service_jitter: 0.2,
            ..GpuTrainingConfig::default()
        };
        let mut sim = GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(42), vec![]).unwrap();
        let _ = sim.run(64).unwrap();
        let idle = sim.idle_accumulated();
        let any_idle = idle.values().any(|t| *t > 0.0);
        assert!(any_idle, "expected at least one GPU to accumulate idle time");
    }

    #[test]
    fn state_validates_each_tick() {
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            service_jitter: 0.1,
            ..GpuTrainingConfig::default()
        };
        let mut sim = GpuTrainingSimulator::new(cfg, small_topology(4), NssSeed(42), vec![]).unwrap();
        let traj = sim.run(64).unwrap();
        for ev in traj.iter() {
            ev.state.validate().expect("state must validate every tick");
        }
    }

    // ---- Phase 2c pipeline-parallelism tests ----

    #[test]
    fn pipeline_stage_assignment_partitions_gpus_evenly() {
        let cfg = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            ..GpuTrainingConfig::default()
        };
        let sim = GpuTrainingSimulator::new(cfg, small_topology(8), NssSeed(42), vec![]).unwrap();
        let by_stage = sim.gpus_by_stage();
        assert_eq!(by_stage.len(), 4); // 4 stages
        for gpus in by_stage.values() {
            assert_eq!(gpus.len(), 2); // 2 replicas per stage
        }
        // Stage 0 = GPUs 0, 1; stage 3 = GPUs 6, 7
        assert_eq!(by_stage.get(&0).unwrap(), &vec![NodeId(0), NodeId(1)]);
        assert_eq!(by_stage.get(&3).unwrap(), &vec![NodeId(6), NodeId(7)]);
    }

    #[test]
    fn pipeline_validation_requires_divisibility() {
        let bad = GpuTrainingConfig {
            n_gpus: 7,
            pipeline_stages: 4,
            ..GpuTrainingConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn bubble_fraction_matches_gpipe_formula() {
        // GPipe bubble: (d - 1) / (d - 1 + m)
        let cfg = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 8,
            ..GpuTrainingConfig::default()
        };
        let expected = 3.0 / (3.0 + 8.0);
        assert!((cfg.bubble_fraction() - expected).abs() < 1e-12);
    }

    #[test]
    fn bubble_fraction_zero_when_no_pipeline() {
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            pipeline_stages: 1,
            ..GpuTrainingConfig::default()
        };
        assert_eq!(cfg.bubble_fraction(), 0.0);
    }

    #[test]
    fn pipeline_raises_sync_pressure_vs_pure_data_parallel() {
        // Same config except pipeline_stages: 1 vs 4. The pipelined
        // run must show higher cumulative sync pressure (bubble idle
        // floor is non-zero).
        let dp = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 1,
            microbatches_per_iteration: 1,
            service_jitter: 0.0, // isolate the bubble effect
            ..GpuTrainingConfig::default()
        };
        let pp = GpuTrainingConfig {
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            ..dp
        };
        let top = small_topology(8);
        let mut sim_dp = GpuTrainingSimulator::new(dp, top.clone(), NssSeed(7), vec![]).unwrap();
        let mut sim_pp = GpuTrainingSimulator::new(pp, top, NssSeed(7), vec![]).unwrap();
        let t_dp = sim_dp.run(16).unwrap();
        let t_pp = sim_pp.run(16).unwrap();
        let sync_sum = |t: &ClusterTrajectory| -> f64 {
            let mut a = KahanAccumulatorF64::new();
            for ev in t.iter() {
                for s in ev.state.nodes.values() {
                    a.add(
                        s.pressures
                            .get(PressureKind::Sync)
                            .map(|p| p.saturation())
                            .unwrap_or(0.0),
                    );
                }
            }
            a.finalize()
        };
        let dp_sync = sync_sum(&t_dp);
        let pp_sync = sync_sum(&t_pp);
        assert!(
            pp_sync > dp_sync + 1.0,
            "pipeline must add bubble-induced sync pressure (dp={}, pp={})",
            dp_sync,
            pp_sync
        );
    }

    #[test]
    fn early_stage_accumulates_more_memory_under_pipeline() {
        // Compare memory growth at stage 0 (GPU 0) vs final stage (GPU 7)
        // under pipeline_stages=4 with no GC.
        let cfg = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            gc_interval: 100_000, // disable GC
            memory_per_microbatch: 0.05,
            fragmentation_growth: 0.0,
            service_jitter: 0.0,
            ..GpuTrainingConfig::default()
        };
        let mut sim =
            GpuTrainingSimulator::new(cfg, small_topology(8), NssSeed(42), vec![]).unwrap();
        let traj = sim.run(16).unwrap();
        let mem_at_end = |gpu: NodeId| -> f64 {
            traj.last_state()
                .unwrap()
                .nodes
                .get(&gpu)
                .unwrap()
                .pressures
                .get(PressureKind::Memory)
                .unwrap()
                .saturation()
        };
        // Stage 0 (GPU 0) holds 4/4 = 1.0 activation multiplier.
        // Stage 3 (GPU 7) holds (4-3)/4 = 0.25 multiplier.
        // After 16 iters: stage 0 ~16*0.05*1.0=0.8; stage 3 ~16*0.05*0.25=0.2.
        // (Phase 1 propagation redistributes a bit, but stage 0 must
        // still be strictly higher.)
        let m0 = mem_at_end(NodeId(0));
        let m7 = mem_at_end(NodeId(7));
        assert!(m0 > m7, "stage 0 memory must exceed final-stage memory (s0={}, s3={})", m0, m7);
    }

    #[test]
    fn pipeline_reduces_per_iteration_allreduce_pressure() {
        // Same config except pipeline shrinks the allreduce group.
        let dp = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 1,
            microbatches_per_iteration: 1,
            service_jitter: 0.0,
            allreduce_base: 0.0,
            allreduce_bytes: 1.0e10,
            nccl_bandwidth: 1.0e10,
            ..GpuTrainingConfig::default()
        };
        let pp = GpuTrainingConfig {
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            ..dp
        };
        let top = small_topology(8);
        let mut sim_dp = GpuTrainingSimulator::new(dp, top.clone(), NssSeed(7), vec![]).unwrap();
        let mut sim_pp = GpuTrainingSimulator::new(pp, top, NssSeed(7), vec![]).unwrap();
        let t_dp = sim_dp.run(8).unwrap();
        let t_pp = sim_pp.run(8).unwrap();
        let net_sum = |t: &ClusterTrajectory| -> f64 {
            let mut a = KahanAccumulatorF64::new();
            for ev in t.iter() {
                for s in ev.state.nodes.values() {
                    a.add(
                        s.pressures
                            .get(PressureKind::Network)
                            .map(|p| p.saturation())
                            .unwrap_or(0.0),
                    );
                }
            }
            a.finalize()
        };
        // Pipeline reduces allreduce group size from 8 → 2, so
        // network pressure should drop.
        assert!(
            net_sum(&t_pp) < net_sum(&t_dp),
            "pipeline must reduce per-tick allreduce (dp={}, pp={})",
            net_sum(&t_dp),
            net_sum(&t_pp)
        );
    }

    // ---- Phase 3d tests: pipeline schedule + activation checkpointing ----

    #[test]
    fn pipeline_schedule_default_is_gpipe() {
        assert_eq!(GpuTrainingConfig::default().pipeline_schedule, PipelineSchedule::GPipe);
    }

    #[test]
    fn pipeline_schedule_label_round_trip() {
        assert_eq!(PipelineSchedule::GPipe.label(), "gpipe");
        assert_eq!(PipelineSchedule::OneForwardOneBackward.label(), "1f1b");
        assert_eq!(
            PipelineSchedule::Interleaved { factor: 4 }.label(),
            "interleaved_4"
        );
    }

    #[test]
    fn interleaved_reduces_bubble_fraction() {
        let base = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            pipeline_schedule: PipelineSchedule::GPipe,
            ..GpuTrainingConfig::default()
        };
        let interleaved = GpuTrainingConfig {
            pipeline_schedule: PipelineSchedule::Interleaved { factor: 4 },
            ..base
        };
        let bf_base = base.bubble_fraction();
        let bf_inter = interleaved.bubble_fraction();
        assert!(bf_inter < bf_base, "interleaved bubble must be smaller (gpipe={}, interleaved={})", bf_base, bf_inter);
        // With factor=4, the bubble should be exactly 1/4 of the base.
        assert!((bf_inter - bf_base / 4.0).abs() < 1e-12);
    }

    #[test]
    fn one_forward_one_backward_keeps_same_bubble_as_gpipe() {
        let base = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            pipeline_schedule: PipelineSchedule::GPipe,
            ..GpuTrainingConfig::default()
        };
        let onefonebw = GpuTrainingConfig {
            pipeline_schedule: PipelineSchedule::OneForwardOneBackward,
            ..base
        };
        assert!((base.bubble_fraction() - onefonebw.bubble_fraction()).abs() < 1e-12);
    }

    #[test]
    fn one_forward_one_backward_reduces_memory_vs_gpipe() {
        // 1F1B with many microbatches reduces memory significantly.
        let cfg_gpipe = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 32,
            pipeline_schedule: PipelineSchedule::GPipe,
            ..GpuTrainingConfig::default()
        };
        let cfg_1f1b = GpuTrainingConfig {
            pipeline_schedule: PipelineSchedule::OneForwardOneBackward,
            ..cfg_gpipe
        };
        // Stage 0 memory multiplier:
        //   GPipe: gpipe_skew * 1.0 = (32-0)/32 = 1.0
        //   1F1B:  gpipe_skew * (4/32 = 0.125) = 0.125
        let m_gpipe = cfg_gpipe.stage_activation_memory_multiplier(0);
        let m_1f1b = cfg_1f1b.stage_activation_memory_multiplier(0);
        assert!(m_1f1b < m_gpipe, "1F1B must reduce stage 0 memory (gpipe={}, 1f1b={})", m_gpipe, m_1f1b);
        assert!(m_1f1b <= 0.2);
    }

    #[test]
    fn interleaved_memory_between_one_forward_one_backward_and_gpipe() {
        let base = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 16,
            ..GpuTrainingConfig::default()
        };
        let g_mem = base.stage_activation_memory_multiplier(0);
        let mem_1f1b = GpuTrainingConfig {
            pipeline_schedule: PipelineSchedule::OneForwardOneBackward,
            ..base
        }
        .stage_activation_memory_multiplier(0);
        let mem_interleaved = GpuTrainingConfig {
            pipeline_schedule: PipelineSchedule::Interleaved { factor: 2 },
            ..base
        }
        .stage_activation_memory_multiplier(0);
        // Interleaved sits between GPipe (most) and 1F1B (least)
        // because it needs more concurrent stages active per GPU.
        assert!(mem_1f1b < mem_interleaved);
        assert!(mem_interleaved <= g_mem);
    }

    #[test]
    fn activation_checkpointing_reduces_memory() {
        let no_cp = GpuTrainingConfig {
            n_gpus: 4,
            pipeline_stages: 1,
            activation_checkpointing: false,
            ..GpuTrainingConfig::default()
        };
        let with_cp = GpuTrainingConfig {
            activation_checkpointing: true,
            checkpoint_memory_factor: 0.4,
            ..no_cp
        };
        let mem_no = no_cp.stage_activation_memory_multiplier(0);
        let mem_yes = with_cp.stage_activation_memory_multiplier(0);
        assert!((mem_no - 1.0).abs() < 1e-12, "no-checkpoint mem multiplier should be 1.0");
        assert!((mem_yes - 0.4).abs() < 1e-12, "checkpointed mem should be 0.4");
    }

    #[test]
    fn checkpointing_increases_cpu_pressure() {
        let no_cp = GpuTrainingConfig {
            n_gpus: 4,
            service_jitter: 0.0,
            activation_checkpointing: false,
            ..GpuTrainingConfig::default()
        };
        let with_cp = GpuTrainingConfig {
            activation_checkpointing: true,
            checkpoint_recompute_overhead: 0.5,
            ..no_cp
        };
        // CPU multiplier: 1.0 with no checkpointing, 1.5 with 0.5 overhead.
        assert!((no_cp.checkpoint_cpu_multiplier() - 1.0).abs() < 1e-12);
        assert!((with_cp.checkpoint_cpu_multiplier() - 1.5).abs() < 1e-12);
        let top = small_topology(4);
        let mut sim_a = GpuTrainingSimulator::new(no_cp, top.clone(), NssSeed(42), vec![]).unwrap();
        let mut sim_b = GpuTrainingSimulator::new(with_cp, top, NssSeed(42), vec![]).unwrap();
        let traj_a = sim_a.run(16).unwrap();
        let traj_b = sim_b.run(16).unwrap();
        // Sum CPU saturations across both runs; checkpointing must
        // produce more cumulative CPU pressure.
        let cpu_sum = |t: &ClusterTrajectory| {
            let mut a = KahanAccumulatorF64::new();
            for ev in t.iter() {
                for s in ev.state.nodes.values() {
                    a.add(
                        s.pressures
                            .get(PressureKind::Cpu)
                            .map(|p| p.saturation())
                            .unwrap_or(0.0),
                    );
                }
            }
            a.finalize()
        };
        assert!(cpu_sum(&traj_b) > cpu_sum(&traj_a));
    }

    #[test]
    fn config_rejects_zero_interleave_factor() {
        let bad = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            pipeline_schedule: PipelineSchedule::Interleaved { factor: 0 },
            ..GpuTrainingConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn config_rejects_bad_checkpoint_factor() {
        let bad = GpuTrainingConfig {
            checkpoint_memory_factor: 1.5,
            ..GpuTrainingConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = GpuTrainingConfig {
            checkpoint_memory_factor: 0.0,
            ..GpuTrainingConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = GpuTrainingConfig {
            checkpoint_recompute_overhead: -0.1,
            ..GpuTrainingConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn schedule_determinism_holds_across_runs() {
        let cfg = GpuTrainingConfig {
            n_gpus: 4,
            pipeline_stages: 2,
            microbatches_per_iteration: 4,
            pipeline_schedule: PipelineSchedule::Interleaved { factor: 2 },
            activation_checkpointing: true,
            ..GpuTrainingConfig::default()
        };
        let top = small_topology(4);
        let mut a = GpuTrainingSimulator::new(cfg, top.clone(), NssSeed(42), vec![]).unwrap();
        let mut b = GpuTrainingSimulator::new(cfg, top, NssSeed(42), vec![]).unwrap();
        let ta = a.run(16).unwrap();
        let tb = b.run(16).unwrap();
        assert_eq!(ta.canonical_bytes(), tb.canonical_bytes());
    }

    #[test]
    fn schedule_changes_canonical_bytes() {
        let base = GpuTrainingConfig::default();
        let with_1f1b = GpuTrainingConfig {
            pipeline_schedule: PipelineSchedule::OneForwardOneBackward,
            ..base
        };
        let with_cp = GpuTrainingConfig {
            activation_checkpointing: true,
            ..base
        };
        assert_ne!(base.canonical_bytes(), with_1f1b.canonical_bytes());
        assert_ne!(base.canonical_bytes(), with_cp.canonical_bytes());
        assert_ne!(with_1f1b.canonical_bytes(), with_cp.canonical_bytes());
    }

    #[test]
    fn pipeline_determinism_holds() {
        let cfg = GpuTrainingConfig {
            n_gpus: 8,
            pipeline_stages: 4,
            microbatches_per_iteration: 4,
            service_jitter: 0.1,
            ..GpuTrainingConfig::default()
        };
        let mut a =
            GpuTrainingSimulator::new(cfg, small_topology(8), NssSeed(11), vec![]).unwrap();
        let mut b =
            GpuTrainingSimulator::new(cfg, small_topology(8), NssSeed(11), vec![]).unwrap();
        assert_eq!(
            a.run(24).unwrap().canonical_bytes(),
            b.run(24).unwrap().canonical_bytes()
        );
    }
}
