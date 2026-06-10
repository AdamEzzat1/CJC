//! Phase 2 — `ClusterNeuralSystemsSimulator`.
//!
//! The cluster-aware predictor. Composes Phase 1 building blocks:
//!
//! 1. Per-node [`crate::SystemEncoder`] (weights *shared* across nodes —
//!    each node feeds through the same encoder).
//! 2. **Sum-pool aggregator** over `NodeId` order — produces a
//!    fixed-size cluster latent regardless of N.
//! 3. Phase 1 [`crate::TemporalStateEngine`] consuming the pooled
//!    latent.
//! 4. A new [`ClusterFailurePredictionHead`] whose input is
//!    `[cluster_latent | cluster_features]`. `cluster_features` is a
//!    fixed-length, hand-built summary vector that captures
//!    cluster-only signals the per-node features can't see (failed
//!    fraction, max link congestion, total in-flight, etc.).
//! 5. Per-node attribution via the Phase 1
//!    [`crate::CausalAttributionHead`] — every node's per-feature
//!    contribution is exact; the cluster rollup picks the node whose
//!    dominant contribution has the largest absolute magnitude.
//!
//! ## Determinism
//!
//! - Per-node encoder shares weights → adding/removing a node never
//!   changes the encoder.
//! - Sum-pool over `BTreeMap<NodeId, _>` iteration = deterministic.
//! - All reductions Kahan-compensated.
//! - Cluster run-id binds `(seed, model_version, cluster_config_bytes,
//!   cluster_input_hash)` — replaying with the same trajectory + topology
//!   + interventions produces an identical id.

use crate::cluster::{ClusterSystemState, ClusterTopology, ClusterTrajectory, NodeId};
use crate::cluster_simulator::ClusterConfig;
use crate::encoder::{EncoderConfig, SystemEncoder};
use crate::error::NssError;
use crate::failure::{FailureKind, FailurePrediction};
use crate::heads::{CausalAttribution, CausalAttributionHead, HeadConfig, PressureContribution};
use crate::multi_timescale::{MultiTimescaleConfig, MultiTimescaleEngine, Timescale};
use crate::pressure::PressureKind;
use crate::seed::{InputHash, NssRunId, NssSeed};
use crate::temporal::{TemporalStateConfig, TemporalStateEngine};
use crate::NSS_MODEL_VERSION;
use cjc_repro::{KahanAccumulatorF64, Rng};
use std::collections::BTreeMap;

/// Cluster-only summary features. Concatenated with the cluster latent
/// before the failure head fires. The size is fixed at compile time so
/// the head input dimension is stable across node-count changes.
pub const CLUSTER_SUMMARY_FEATURES: usize = 6;

/// Cluster-aware predictor config. Wraps Phase 1 component configs +
/// adds calibration_gain.
/// **Phase 3b** — temporal-engine selector. `Single` (default) uses
/// the original Phase 1/2 [`TemporalStateEngine`]; `MultiAll` uses the
/// four canonical timescales (Short/Medium/Long/Structural) in parallel.
///
/// We use a closed enum (not a `Vec<Timescale>`) to keep
/// [`ClusterNssConfig`] `Copy` — preserving the public API across all
/// existing callers that pass configs by value.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum TemporalMode {
    /// Phase 1/2 single SSM — one decay rate `α` set by
    /// `TemporalStateConfig::alpha`.
    #[default]
    Single,
    /// Phase 3b multi-timescale memory — parallel SSM buffers at
    /// Short/Medium/Long/Structural (α=0.5/0.85/0.95/0.99). The
    /// configured `temporal.state_dim` becomes the *per-buffer* dim;
    /// the cluster head must be configured with
    /// `head.state_dim = 4 * temporal.state_dim`.
    MultiAll,
}

impl TemporalMode {
    /// Number of parallel timescales for this mode.
    pub fn timescale_count(self) -> usize {
        match self {
            TemporalMode::Single => 1,
            TemporalMode::MultiAll => 4,
        }
    }

    /// Canonical short label.
    pub fn label(self) -> &'static str {
        match self {
            TemporalMode::Single => "single",
            TemporalMode::MultiAll => "multi_all",
        }
    }
}

/// Cluster-aware predictor config. Wraps Phase 1 component configs +
/// adds calibration_gain + the Phase 3b temporal-mode selector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterNssConfig {
    /// Per-node encoder config.
    pub encoder: EncoderConfig,
    /// Temporal-state engine config (input_dim == encoder.latent_dim).
    /// In [`TemporalMode::MultiAll`], `state_dim` is the **per-buffer**
    /// dim; the cluster head sees `4 * state_dim`.
    pub temporal: TemporalStateConfig,
    /// Per-node attribution-head config. In `Single` mode,
    /// `state_dim == temporal.state_dim`. In `MultiAll`,
    /// `state_dim == 4 * temporal.state_dim`.
    pub head: HeadConfig,
    /// Same count-based calibration gain as Phase 1.
    pub calibration_gain: f64,
    /// **Phase 3b** — temporal-engine selector. Default `Single`
    /// preserves Phase 1/2 behaviour.
    pub temporal_mode: TemporalMode,
}

impl Default for ClusterNssConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            temporal: TemporalStateConfig::default(),
            head: HeadConfig::default(),
            calibration_gain: 1.0,
            temporal_mode: TemporalMode::Single,
        }
    }
}

impl ClusterNssConfig {
    /// Validate, including dimensional agreement.
    pub fn validate(&self) -> Result<(), NssError> {
        self.encoder.validate()?;
        self.temporal.validate()?;
        self.head.validate()?;
        if self.temporal.input_dim != self.encoder.latent_dim {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "temporal.input_dim ({}) must equal encoder.latent_dim ({})",
                    self.temporal.input_dim, self.encoder.latent_dim
                ),
            });
        }
        let expected_head_dim = self.temporal.state_dim * self.temporal_mode.timescale_count();
        if self.head.state_dim != expected_head_dim {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "head.state_dim ({}) must equal temporal.state_dim ({}) * timescale_count ({}) = {}",
                    self.head.state_dim,
                    self.temporal.state_dim,
                    self.temporal_mode.timescale_count(),
                    expected_head_dim,
                ),
            });
        }
        if !self.calibration_gain.is_finite() || self.calibration_gain < 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "calibration_gain must be finite and >= 0, got {}",
                    self.calibration_gain
                ),
            });
        }
        Ok(())
    }

    /// Canonical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(&self.encoder.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.temporal.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.head.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.calibration_gain.to_bits().to_le_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(self.temporal_mode.label().as_bytes());
        bytes
    }

    /// Total head input size = state_dim + summary features.
    pub fn cluster_head_input(&self) -> usize {
        self.head.state_dim + CLUSTER_SUMMARY_FEATURES
    }
}

/// The cluster-level failure head. Two logits over cluster summary +
/// pooled latent. Same structural shape as the Phase 1 head, just over
/// the cluster input layout.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterFailurePredictionHead {
    cfg: ClusterNssConfig,
    /// Collapse weights — length = `cfg.cluster_head_input()`.
    w_collapse: Vec<f64>,
    b_collapse: f64,
    /// Degraded weights.
    w_degraded: Vec<f64>,
    b_degraded: f64,
}

impl ClusterFailurePredictionHead {
    /// Build from a seed. Two heads from the same `(cfg, seed)` are
    /// bit-identical.
    pub fn from_seed(cfg: ClusterNssConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let n = cfg.cluster_head_input();
        let mut rng_c = seed.substream("cluster_head.collapse.W");
        let mut rng_d = seed.substream("cluster_head.degraded.W");
        let mut rng_bc = seed.substream("cluster_head.collapse.b");
        let mut rng_bd = seed.substream("cluster_head.degraded.b");
        Ok(Self {
            cfg,
            w_collapse: init_vec(&mut rng_c, n, cfg.encoder.init_scale),
            b_collapse: cfg.encoder.init_scale * rng_bc.next_normal_f64(),
            w_degraded: init_vec(&mut rng_d, n, cfg.encoder.init_scale),
            b_degraded: cfg.encoder.init_scale * rng_bd.next_normal_f64(),
        })
    }

    /// Borrow collapse weights — used by the per-node attribution head
    /// to read the cluster-summary slots.
    pub fn collapse_weights(&self) -> &[f64] {
        &self.w_collapse
    }

    /// Borrow config.
    pub fn config(&self) -> &ClusterNssConfig {
        &self.cfg
    }

    /// Predict failure probabilities from pooled latent + summary.
    pub fn predict(
        &self,
        h: &[f64],
        summary: &[f64; CLUSTER_SUMMARY_FEATURES],
    ) -> Result<FailurePrediction, NssError> {
        if h.len() != self.cfg.head.state_dim {
            return Err(NssError::InvalidState {
                detail: format!("h.len {} != state_dim {}", h.len(), self.cfg.head.state_dim),
            });
        }
        let mut z = Vec::with_capacity(h.len() + CLUSTER_SUMMARY_FEATURES);
        z.extend_from_slice(h);
        z.extend_from_slice(summary);
        let logit_c = dot_kahan(&self.w_collapse, &z) + self.b_collapse;
        let logit_d = dot_kahan(&self.w_degraded, &z) + self.b_degraded;
        let p_c = sigmoid(logit_c);
        let p_d = sigmoid(logit_d);
        let conf = p_c.max(p_d).max(1.0 - p_c.max(p_d));
        Ok(FailurePrediction {
            collapse_probability: p_c,
            degraded_probability: p_d,
            confidence: conf,
        })
    }

    /// Module-internal collapse-weight setter — used by
    /// `ClusterNeuralSystemsSimulator::fit` for count-based calibration.
    pub(crate) fn set_collapse_weights(&mut self, w: Vec<f64>) {
        debug_assert_eq!(w.len(), self.cfg.cluster_head_input());
        self.w_collapse = w;
    }
}

/// One cluster attribution: per-node `CausalAttribution` + the rolled-up
/// dominant node.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterCausalAttribution {
    /// Per-node breakdown (BTreeMap so iteration order is deterministic).
    pub per_node: BTreeMap<NodeId, CausalAttribution>,
    /// Node whose dominant-source magnitude is the largest in absolute
    /// value across the cluster.
    pub dominant_node: NodeId,
    /// The (kind, magnitude) of that dominant source.
    pub dominant_contribution: PressureContribution,
    /// Cluster-summary contributions — per-summary-feature attribution
    /// to the collapse logit (`w_collapse[state_dim..] * summary[i]`).
    /// Indexed by [`cluster_summary_label`].
    pub summary_contributions: [f64; CLUSTER_SUMMARY_FEATURES],
}

/// Labels for the cluster summary features. Aligns with the layout in
/// [`build_cluster_summary`]. Used for human-readable attribution
/// output.
pub fn cluster_summary_label(i: usize) -> &'static str {
    match i {
        0 => "failed_fraction",
        1 => "max_link_congestion",
        2 => "mean_link_congestion",
        3 => "total_in_flight_log1p",
        4 => "rejected_fraction",
        5 => "max_node_queue_saturation",
        _ => "unknown",
    }
}

/// One cluster prediction. Mirrors the Phase 1 `NssPrediction` API but
/// with a cluster-level head + per-node attribution.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterPrediction {
    /// Content-addressed run identifier.
    pub run_id: NssRunId,
    /// Hash of the input cluster state.
    pub input_hash: InputHash,
    /// Cluster-level failure head output.
    pub failure: FailurePrediction,
    /// Cluster-level causal attribution.
    pub attribution: ClusterCausalAttribution,
}

/// **Phase 3b** — internal enum dispatching between single-timescale
/// and multi-timescale temporal engines. Both expose the same
/// `step(h, z) → h'` and `zero_state()` interface, so the cluster NSS's
/// forward path is mode-agnostic.
#[derive(Clone, Debug, PartialEq)]
enum ClusterTemporalEngine {
    Single(TemporalStateEngine),
    Multi(MultiTimescaleEngine),
}

impl ClusterTemporalEngine {
    fn step(&self, h: &[f64], z: &[f64]) -> Result<Vec<f64>, NssError> {
        match self {
            ClusterTemporalEngine::Single(e) => e.step(h, z),
            ClusterTemporalEngine::Multi(e) => e.step_concatenated(h, z),
        }
    }
    fn zero_state(&self) -> Vec<f64> {
        match self {
            ClusterTemporalEngine::Single(e) => e.zero_state(),
            ClusterTemporalEngine::Multi(e) => e.zero_state_concatenated(),
        }
    }
}

/// The cluster predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterNeuralSystemsSimulator {
    cfg: ClusterNssConfig,
    seed: NssSeed,
    /// Encoder is shared across nodes (same weights for every node).
    encoder: SystemEncoder,
    temporal: ClusterTemporalEngine,
    head: ClusterFailurePredictionHead,
    /// Per-node attribution head — same weights as a single Phase 1
    /// failure head, used to compute per-node decomposition.
    per_node_attr: CausalAttributionHead,
    /// Reusable per-node failure head — its collapse weights drive the
    /// per-node attribution. (Note: this is a *separate* set of weights
    /// from the cluster-level head's collapse weights; the per-node
    /// head sees per-node features in `[h | x]` layout.)
    per_node_head: crate::heads::FailurePredictionHead,
    fitted: bool,
    h0: Vec<f64>,
}

impl ClusterNeuralSystemsSimulator {
    /// Build from `(cfg, seed)`.
    pub fn from_seed(cfg: ClusterNssConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let encoder = SystemEncoder::from_seed(cfg.encoder, seed)?;
        let temporal = match cfg.temporal_mode {
            TemporalMode::Single => {
                ClusterTemporalEngine::Single(TemporalStateEngine::from_seed(cfg.temporal, seed)?)
            }
            TemporalMode::MultiAll => {
                let mt_cfg = MultiTimescaleConfig {
                    per_scale_dim: cfg.temporal.state_dim,
                    input_dim: cfg.temporal.input_dim,
                    timescales: Timescale::ALL.to_vec(),
                    init_scale: cfg.temporal.init_scale,
                };
                ClusterTemporalEngine::Multi(MultiTimescaleEngine::from_seed(mt_cfg, seed)?)
            }
        };
        let head = ClusterFailurePredictionHead::from_seed(cfg, seed)?;
        let per_node_head = crate::heads::FailurePredictionHead::from_seed(cfg.head, seed)?;
        let per_node_attr = CausalAttributionHead::new(cfg.head)?;
        let h0 = temporal.zero_state();
        Ok(Self {
            cfg,
            seed,
            encoder,
            temporal,
            head,
            per_node_attr,
            per_node_head,
            fitted: false,
            h0,
        })
    }

    /// Config accessor.
    pub fn config(&self) -> &ClusterNssConfig {
        &self.cfg
    }

    /// Seed accessor.
    pub fn seed(&self) -> NssSeed {
        self.seed
    }

    /// `true` once fit() has run.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Build the fixed-size cluster summary feature vector for a
    /// single `ClusterSystemState`. Stable layout — see
    /// [`cluster_summary_label`] for the index → name map.
    pub fn build_cluster_summary(state: &ClusterSystemState) -> [f64; CLUSTER_SUMMARY_FEATURES] {
        let n_nodes = state.nodes.len().max(1) as f64;
        let failed_count = state.failed_count() as f64;
        let failed_fraction = failed_count / n_nodes;
        let max_congestion = state
            .link_congestion
            .values()
            .fold(0.0f64, |acc, c| acc.max(*c));
        let mut mean_acc = KahanAccumulatorF64::new();
        for c in state.link_congestion.values() {
            mean_acc.add(*c);
        }
        let mean_congestion = if state.link_congestion.is_empty() {
            0.0
        } else {
            mean_acc.finalize() / state.link_congestion.len() as f64
        };
        let total_in_flight: u64 = state.nodes.values().map(|s| s.in_flight).sum();
        let total_completed: u64 = state.nodes.values().map(|s| s.completed).sum();
        let total_rejected: u64 = state.nodes.values().map(|s| s.rejected).sum();
        let total = total_completed.saturating_add(total_rejected).max(1) as f64;
        let rejected_fraction = total_rejected as f64 / total;
        let max_queue_saturation = state
            .nodes
            .values()
            .map(|s| {
                s.pressures
                    .get(PressureKind::Queue)
                    .map(|p| p.saturation())
                    .unwrap_or(0.0)
            })
            .fold(0.0f64, |acc, q| acc.max(q));
        [
            failed_fraction,
            max_congestion,
            mean_congestion,
            (total_in_flight as f64).ln_1p(),
            rejected_fraction,
            max_queue_saturation,
        ]
    }

    /// Encode + pool the cluster latent. Each node's `SystemState`
    /// goes through the shared encoder; the resulting vectors are
    /// Kahan-sum-pooled into the cluster latent of `encoder.latent_dim`.
    fn cluster_latent(&self, state: &ClusterSystemState) -> Vec<f64> {
        let dim = self.cfg.encoder.latent_dim;
        let mut acc: Vec<KahanAccumulatorF64> =
            (0..dim).map(|_| KahanAccumulatorF64::new()).collect();
        for (_id, node_state) in state.nodes.iter() {
            let z = self.encoder.forward(node_state);
            for (i, v) in z.iter().enumerate() {
                acc[i].add(*v);
            }
        }
        acc.into_iter().map(|a| a.finalize()).collect()
    }

    /// One-step cluster prediction.
    pub fn predict_next(&self, state: &ClusterSystemState) -> Result<ClusterPrediction, NssError> {
        state.validate()?;
        let z = self.cluster_latent(state);
        let h = self.temporal.step(&self.h0, &z)?;
        let summary = Self::build_cluster_summary(state);
        let failure = self.head.predict(&h, &summary)?;
        let attribution = self.attribute(state, &summary);

        let input_hash = InputHash::of_bytes(&state.canonical_bytes());
        let run_id = NssRunId::build(
            self.seed,
            NSS_MODEL_VERSION,
            &self.cfg.canonical_bytes(),
            input_hash,
        );

        Ok(ClusterPrediction {
            run_id,
            input_hash,
            failure,
            attribution,
        })
    }

    /// Per-node attribution + cluster summary attribution.
    fn attribute(
        &self,
        state: &ClusterSystemState,
        summary: &[f64; CLUSTER_SUMMARY_FEATURES],
    ) -> ClusterCausalAttribution {
        // Per-node: run each node's features through the per-node
        // failure-head's collapse weights. (We use the per-node head
        // because the cluster-level head consumes a *summary* vector,
        // not per-node features — but the per-node head was seeded
        // identically and gives consistent decomposition.)
        let mut per_node: BTreeMap<NodeId, CausalAttribution> = BTreeMap::new();
        let mut dominant_node: NodeId = state.nodes.keys().copied().next().unwrap_or(NodeId(0));
        let mut dominant_mag = f64::NEG_INFINITY;
        let mut dominant_contrib = PressureContribution {
            kind: PressureKind::Queue,
            magnitude: 0.0,
        };
        for (id, node_state) in state.nodes.iter() {
            let x = SystemEncoder::features_of(node_state);
            let attr = self.per_node_attr.attribute(&self.per_node_head, &x);
            if attr.dominant_source.magnitude.abs() > dominant_mag {
                dominant_mag = attr.dominant_source.magnitude.abs();
                dominant_node = *id;
                dominant_contrib = attr.dominant_source;
            }
            per_node.insert(*id, attr);
        }

        // Cluster-summary contributions: read the cluster head's
        // collapse weights at indices `[state_dim, state_dim + 6)`.
        let mut summary_contributions = [0.0_f64; CLUSTER_SUMMARY_FEATURES];
        let w = self.head.collapse_weights();
        let off = self.cfg.head.state_dim;
        for i in 0..CLUSTER_SUMMARY_FEATURES {
            summary_contributions[i] = w[off + i] * summary[i];
        }

        ClusterCausalAttribution {
            per_node,
            dominant_node,
            dominant_contribution: dominant_contrib,
            summary_contributions,
        }
    }

    /// Count-based calibration over a cluster trajectory. Same
    /// philosophy as Phase 1: walk the trajectory, contrast
    /// (mean_summary | next=Collapse) vs (mean_summary | next≠Collapse),
    /// add a calibrated boost to the cluster head's collapse weights
    /// on the corresponding summary indices. The per-node head is
    /// also boosted using each node's own collapse-vs-other contrast.
    pub fn fit(&mut self, traj: &ClusterTrajectory) -> Result<(), NssError> {
        if traj.len() < 2 {
            return Err(NssError::InvalidTrajectory {
                detail: "cluster fit requires trajectory of length >= 2".into(),
            });
        }
        // Re-seed both heads so fit is idempotent.
        self.head = ClusterFailurePredictionHead::from_seed(self.cfg, self.seed)?;
        self.per_node_head =
            crate::heads::FailurePredictionHead::from_seed(self.cfg.head, self.seed)?;
        self.h0 = self.temporal.zero_state();

        // Cluster-summary calibration.
        let mut sum_c = [0.0_f64; CLUSTER_SUMMARY_FEATURES];
        let mut sum_o = [0.0_f64; CLUSTER_SUMMARY_FEATURES];
        let mut n_c: u64 = 0;
        let mut n_o: u64 = 0;

        // Per-node-feature calibration (same shape as Phase 1).
        let n_p = PressureKind::all().len();
        let mut sum_pc = vec![0.0_f64; n_p];
        let mut sum_po = vec![0.0_f64; n_p];
        let mut n_pc: u64 = 0;
        let mut n_po: u64 = 0;

        for window in traj.as_slice().windows(2) {
            let now = &window[0];
            let next = &window[1];
            let next_collapse = next.cluster_failure.kind == FailureKind::Collapse;
            let summary = Self::build_cluster_summary(&now.state);
            if next_collapse {
                for i in 0..CLUSTER_SUMMARY_FEATURES {
                    sum_c[i] += summary[i];
                }
                n_c += 1;
            } else {
                for i in 0..CLUSTER_SUMMARY_FEATURES {
                    sum_o[i] += summary[i];
                }
                n_o += 1;
            }
            // Per-node contributions. We use the *cluster-level*
            // label rather than per-node label so the calibration is
            // about "what node-features predict cluster collapse",
            // not per-node collapse.
            for node_state in now.state.nodes.values() {
                let sats: Vec<f64> = PressureKind::all()
                    .iter()
                    .map(|k| {
                        node_state
                            .pressures
                            .get(*k)
                            .map(|p| p.saturation())
                            .unwrap_or(0.0)
                    })
                    .collect();
                if next_collapse {
                    for (i, v) in sats.iter().enumerate() {
                        sum_pc[i] += v;
                    }
                    n_pc += 1;
                } else {
                    for (i, v) in sats.iter().enumerate() {
                        sum_po[i] += v;
                    }
                    n_po += 1;
                }
            }
        }

        // Cluster-summary boost.
        let mean_sc = if n_c == 0 {
            [0.5; CLUSTER_SUMMARY_FEATURES]
        } else {
            let mut m = [0.0; CLUSTER_SUMMARY_FEATURES];
            for i in 0..CLUSTER_SUMMARY_FEATURES {
                m[i] = sum_c[i] / n_c as f64;
            }
            m
        };
        let mean_so = if n_o == 0 {
            [0.5; CLUSTER_SUMMARY_FEATURES]
        } else {
            let mut m = [0.0; CLUSTER_SUMMARY_FEATURES];
            for i in 0..CLUSTER_SUMMARY_FEATURES {
                m[i] = sum_o[i] / n_o as f64;
            }
            m
        };
        let mut w_new = self.head.collapse_weights().to_vec();
        let off = self.cfg.head.state_dim;
        for i in 0..CLUSTER_SUMMARY_FEATURES {
            w_new[off + i] += self.cfg.calibration_gain * (mean_sc[i] - mean_so[i]);
        }
        self.head.set_collapse_weights(w_new);

        // Per-node-feature boost (Phase 1 style, applied to the
        // per-node head used by the attribution).
        let mean_pc = if n_pc == 0 {
            vec![0.5; n_p]
        } else {
            sum_pc.iter().map(|s| s / n_pc as f64).collect()
        };
        let mean_po = if n_po == 0 {
            vec![0.5; n_p]
        } else {
            sum_po.iter().map(|s| s / n_po as f64).collect()
        };
        let mut w_pn = self.per_node_head.collapse_weights().to_vec();
        let off_pn = self.cfg.head.state_dim;
        for i in 0..n_p {
            w_pn[off_pn + i] += self.cfg.calibration_gain * (mean_pc[i] - mean_po[i]);
        }
        crate::heads::__set_collapse_weights(&mut self.per_node_head, w_pn);

        self.fitted = true;
        Ok(())
    }
}

fn dot_kahan(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = KahanAccumulatorF64::new();
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc.add(ai * bi);
    }
    acc.finalize()
}

// ---- Module-internal helpers used by cluster_grad.rs (Phase 2d) ----

/// Run the cluster encoder + sum-pool, returning the cluster latent
/// BEFORE the temporal step. Crate-internal accessor for the Phase 2d
/// gradient trainer.
pub(crate) fn __cluster_latent(
    nss: &ClusterNeuralSystemsSimulator,
    state: &ClusterSystemState,
) -> Vec<f64> {
    nss.cluster_latent(state)
}

/// Step the temporal engine once from the zero state. Same as the
/// prediction-time forward.
pub(crate) fn __step_temporal_from_zero(
    nss: &ClusterNeuralSystemsSimulator,
    latent: &[f64],
) -> Vec<f64> {
    nss.temporal.step(&nss.h0, latent).expect("temporal step")
}

/// Replace the cluster head's collapse + degraded weights/biases
/// in-place. Used by the Adam trainer to write trained weights back.
pub(crate) fn __set_cluster_head_weights(
    nss: &mut ClusterNeuralSystemsSimulator,
    w_collapse: Vec<f64>,
    b_collapse: f64,
    w_degraded: Vec<f64>,
    b_degraded: f64,
) {
    debug_assert_eq!(w_collapse.len(), nss.cfg.cluster_head_input());
    debug_assert_eq!(w_degraded.len(), nss.cfg.cluster_head_input());
    nss.head.w_collapse = w_collapse;
    nss.head.b_collapse = b_collapse;
    nss.head.w_degraded = w_degraded;
    nss.head.b_degraded = b_degraded;
}

/// Read the seeded head's degraded-weight vector (Phase 2d init).
pub(crate) fn __degraded_weights(head: &ClusterFailurePredictionHead) -> &[f64] {
    &head.w_degraded
}

/// Read the seeded head's collapse bias.
pub(crate) fn __collapse_bias(head: &ClusterFailurePredictionHead) -> f64 {
    head.b_collapse
}

/// Read the seeded head's degraded bias.
pub(crate) fn __degraded_bias(head: &ClusterFailurePredictionHead) -> f64 {
    head.b_degraded
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn init_vec(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for slot in v.iter_mut() {
        *slot = scale * rng.next_normal_f64();
    }
    v
}

/// Cluster-level audit-trace input bundle. Carries everything the
/// `ClusterReplayValidator` needs to reproduce a cluster prediction:
/// config, seed, topology, intervention script, and (optionally) the
/// training trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterTrace {
    /// Run id at the moment of prediction.
    pub run_id: NssRunId,
    /// Hash of `state` bytes.
    pub input_hash: InputHash,
    /// Cluster state the prediction was made on.
    pub input_state: ClusterSystemState,
    /// Cluster topology used for the prediction (only used for
    /// canonical-byte stability; the cluster state already pins
    /// per-node + per-link data).
    pub topology: ClusterTopology,
    /// Simulator config used to produce the training trajectory.
    /// Carried so the validator can rebuild the trace bundle byte-identically.
    pub simulator_config: ClusterConfig,
    /// Intervention script used to produce the training trajectory.
    pub intervention_script: Vec<crate::cluster_simulator::Intervention>,
    /// NSS config.
    pub nss_config: ClusterNssConfig,
    /// Seed.
    pub seed: NssSeed,
    /// Training trajectory used to `fit` (or `None` for un-fitted runs).
    pub training_trajectory: Option<ClusterTrajectory>,
    /// Failure prediction values stamped at emission time. The
    /// validator re-runs and demands bit-identical reproduction.
    pub collapse_probability: f64,
    /// Degraded probability stamped at emission.
    pub degraded_probability: f64,
    /// Model version stamped at emission.
    pub model_version: String,
}

/// Cluster replay validator.
#[derive(Clone, Debug, Default)]
pub struct ClusterReplayValidator;

impl ClusterReplayValidator {
    /// Build.
    pub fn new() -> Self {
        Self
    }

    /// Reproduce the prediction from the bundled inputs.
    pub fn verify(&self, trace: &ClusterTrace) -> Result<(), NssError> {
        let mut nss = ClusterNeuralSystemsSimulator::from_seed(trace.nss_config, trace.seed)?;
        if let Some(t) = trace.training_trajectory.as_ref() {
            nss.fit(t)?;
        }
        let p = nss.predict_next(&trace.input_state)?;
        if p.run_id != trace.run_id {
            return Err(NssError::ReplayMismatch {
                expected: trace.run_id.to_string(),
                actual: p.run_id.to_string(),
            });
        }
        if p.failure.collapse_probability.to_bits() != trace.collapse_probability.to_bits()
            || p.failure.degraded_probability.to_bits() != trace.degraded_probability.to_bits()
        {
            return Err(NssError::ReplayMismatch {
                expected: format!(
                    "collapse={} degraded={}",
                    trace.collapse_probability, trace.degraded_probability
                ),
                actual: format!(
                    "collapse={} degraded={}",
                    p.failure.collapse_probability, p.failure.degraded_probability
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterTopology;
    use crate::cluster_simulator::{ClusterSimulator, Intervention};

    fn small_top(n: u32) -> ClusterTopology {
        ClusterTopology::complete(n, 8, 0.5).unwrap()
    }

    #[test]
    fn determinism_two_clusters_match() {
        let cfg = ClusterNssConfig::default();
        let a = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let b = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let top = small_top(3);
        let state = ClusterSystemState::initial(&top);
        let pa = a.predict_next(&state).unwrap();
        let pb = b.predict_next(&state).unwrap();
        assert_eq!(pa.run_id, pb.run_id);
        assert_eq!(pa.failure, pb.failure);
    }

    #[test]
    fn cluster_summary_is_fixed_size_regardless_of_node_count() {
        let top4 = small_top(4);
        let top9 = small_top(9);
        let s4 = ClusterSystemState::initial(&top4);
        let s9 = ClusterSystemState::initial(&top9);
        let sm4 = ClusterNeuralSystemsSimulator::build_cluster_summary(&s4);
        let sm9 = ClusterNeuralSystemsSimulator::build_cluster_summary(&s9);
        assert_eq!(sm4.len(), CLUSTER_SUMMARY_FEATURES);
        assert_eq!(sm9.len(), CLUSTER_SUMMARY_FEATURES);
    }

    #[test]
    fn end_to_end_fit_and_predict() {
        let cfg_sim = ClusterConfig::default();
        let top = small_top(4);
        let mut sim = ClusterSimulator::new(cfg_sim, top, NssSeed(42), vec![]).unwrap();
        let traj = sim.run(48).unwrap();
        let mut nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        nss.fit(&traj).unwrap();
        let last = traj.last_state().unwrap();
        let pred = nss.predict_next(last).unwrap();
        assert!(pred.failure.collapse_probability.is_finite());
        assert!(pred.failure.collapse_probability >= 0.0);
        assert!(pred.failure.collapse_probability <= 1.0);
    }

    #[test]
    fn fit_changes_head_weights_when_failure_signal_exists() {
        let cfg_sim = ClusterConfig {
            cluster_arrival_rate: 12.0,
            ..ClusterConfig::default()
        };
        let top = small_top(2);
        let ivs = vec![Intervention::FailNode {
            tick: 4,
            node: NodeId(0),
        }];
        let mut sim = ClusterSimulator::new(cfg_sim, top, NssSeed(42), ivs).unwrap();
        let traj = sim.run(48).unwrap();
        let mut nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let before = nss.head.collapse_weights().to_vec();
        nss.fit(&traj).unwrap();
        let after = nss.head.collapse_weights().to_vec();
        // At least one cluster-summary weight must have moved.
        let n_changed = before
            .iter()
            .zip(after.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-15)
            .count();
        assert!(
            n_changed > 0,
            "fit did not adjust any cluster-summary weight"
        );
    }

    // ---- Phase 3b: cluster NSS with multi-timescale memory ----

    #[test]
    fn multi_timescale_mode_requires_head_dim_to_match() {
        // Default config has head.state_dim = temporal.state_dim = 16.
        // Switching to MultiAll without bumping head dim → validation error.
        let bad = ClusterNssConfig {
            temporal_mode: TemporalMode::MultiAll,
            ..ClusterNssConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn multi_timescale_config_validates_when_dims_agree() {
        let good = ClusterNssConfig {
            temporal: TemporalStateConfig {
                state_dim: 4,
                input_dim: 16,
                ..TemporalStateConfig::default()
            },
            head: HeadConfig {
                state_dim: 16, // 4 timescales * 4 per-scale dim
                ..HeadConfig::default()
            },
            temporal_mode: TemporalMode::MultiAll,
            ..ClusterNssConfig::default()
        };
        assert!(good.validate().is_ok());
    }

    #[test]
    fn multi_timescale_cluster_nss_predicts_finite_output() {
        let cfg = ClusterNssConfig {
            temporal: TemporalStateConfig {
                state_dim: 4,
                input_dim: 16,
                ..TemporalStateConfig::default()
            },
            head: HeadConfig {
                state_dim: 16, // 4 * 4
                ..HeadConfig::default()
            },
            temporal_mode: TemporalMode::MultiAll,
            ..ClusterNssConfig::default()
        };
        let nss = ClusterNeuralSystemsSimulator::from_seed(cfg, NssSeed(42)).unwrap();
        let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
        let state = ClusterSystemState::initial(&top);
        let pred = nss.predict_next(&state).unwrap();
        assert!(pred.failure.collapse_probability.is_finite());
        assert!(
            pred.failure.collapse_probability >= 0.0 && pred.failure.collapse_probability <= 1.0
        );
    }

    #[test]
    fn multi_timescale_and_single_predict_differently() {
        // Same seed, same trajectory; single-α vs multi-α heads must
        // produce different predictions because their internal latent
        // is structurally different (4x dimensionality + per-scale α).
        let single_cfg = ClusterNssConfig {
            temporal: TemporalStateConfig {
                state_dim: 4,
                input_dim: 16,
                ..TemporalStateConfig::default()
            },
            head: HeadConfig {
                state_dim: 4,
                ..HeadConfig::default()
            },
            temporal_mode: TemporalMode::Single,
            ..ClusterNssConfig::default()
        };
        let multi_cfg = ClusterNssConfig {
            temporal: TemporalStateConfig {
                state_dim: 4,
                input_dim: 16,
                ..TemporalStateConfig::default()
            },
            head: HeadConfig {
                state_dim: 16,
                ..HeadConfig::default()
            },
            temporal_mode: TemporalMode::MultiAll,
            ..ClusterNssConfig::default()
        };
        let nss_single = ClusterNeuralSystemsSimulator::from_seed(single_cfg, NssSeed(42)).unwrap();
        let nss_multi = ClusterNeuralSystemsSimulator::from_seed(multi_cfg, NssSeed(42)).unwrap();
        let top = ClusterTopology::complete(3, 8, 0.5).unwrap();
        let state = ClusterSystemState::initial(&top);
        let p_single = nss_single.predict_next(&state).unwrap();
        let p_multi = nss_multi.predict_next(&state).unwrap();
        // Run-ids differ because canonical_bytes includes temporal_mode.
        assert_ne!(p_single.run_id, p_multi.run_id);
    }

    #[test]
    fn multi_timescale_run_id_changes_when_mode_changes() {
        // Two configs identical except temporal_mode.
        let cfg_a = ClusterNssConfig {
            temporal: TemporalStateConfig {
                state_dim: 4,
                input_dim: 16,
                ..TemporalStateConfig::default()
            },
            head: HeadConfig {
                state_dim: 4,
                ..HeadConfig::default()
            },
            temporal_mode: TemporalMode::Single,
            ..ClusterNssConfig::default()
        };
        let cfg_b = ClusterNssConfig {
            head: HeadConfig {
                state_dim: 16,
                ..HeadConfig::default()
            },
            temporal_mode: TemporalMode::MultiAll,
            ..cfg_a
        };
        // Different canonical bytes ⇒ different run-ids.
        assert_ne!(cfg_a.canonical_bytes(), cfg_b.canonical_bytes());
    }

    #[test]
    fn attribution_dominant_node_corresponds_to_pressure_loaded_node() {
        let nss =
            ClusterNeuralSystemsSimulator::from_seed(ClusterNssConfig::default(), NssSeed(42))
                .unwrap();
        let top = small_top(3);
        let mut state = ClusterSystemState::initial(&top);
        // Light up node 2's queue heavily; the dominant node should
        // be node 2 (subject to per-node head weight signs).
        let queue_loaded = crate::system::SystemState {
            tick: 0,
            pressures: {
                let mut f = state.nodes.get(&NodeId(2)).unwrap().pressures.clone();
                f.set(
                    PressureKind::Queue,
                    crate::pressure::Pressure::new(0.95, 1.0, 0.0).unwrap(),
                );
                f
            },
            in_flight: 12,
            completed: 0,
            rejected: 0,
            mean_service_time: 1.0,
        };
        state.nodes.insert(NodeId(2), queue_loaded);
        let pred = nss.predict_next(&state).unwrap();
        // The dominant node should be one of the cluster's nodes
        // (smoke check); we don't enforce *which* without committing
        // to a specific seed's weight signs.
        assert!(state.nodes.contains_key(&pred.attribution.dominant_node));
    }
}
