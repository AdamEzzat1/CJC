//! `NeuralSystemsSimulator` — the Phase 1 orchestrator that wires
//! encoder → temporal-state engine → failure head → attribution head
//! into a single predictor.
//!
//! ## Training (Phase 1)
//!
//! Phase 1 ships a **deterministic count-based calibration**, not
//! gradient descent. The motivation is that the architectural primitives
//! (pressure fields, propagation, encoder, temporal state, heads) need
//! to be on stable ground before we layer autodiff in Phase 2. Calibration:
//!
//! 1. Walk the trajectory.
//! 2. For each transition `state[t] → state[t+1]`, compute the encoder
//!    output and step the temporal state forward.
//! 3. Record per-pressure-kind saturations at `t`, conditioned on the
//!    label at `t+1`.
//! 4. Use the resulting (P(saturation | Collapse) - P(saturation | Nominal))
//!    contrast as a per-feature *logit boost* added to the head's
//!    collapse weights on the corresponding features.
//!
//! This produces a head whose attribution is interpretable from the
//! start (high queue-saturation contribution = head learned to associate
//! queue-saturation with collapse) without any non-determinism from
//! gradient noise. Phase 2 wires this into `cjc-ad::GradGraph` for
//! end-to-end backprop while keeping the Phase 1 calibration as a
//! warm-start initialiser.
//!
//! ## Prediction
//!
//! `predict_next` consumes a single [`SystemState`] (typically the last
//! state of a trajectory) and returns an [`NssPrediction`] containing
//! the head outputs, the causal attribution, *and* a
//! [`TransitionRecord`] — the audit trace is part of the API, not an
//! opt-in.
//!
//! ## Determinism contract
//!
//! - Every component (`SystemEncoder`, `TemporalStateEngine`, head) is
//!   seeded from a distinct sub-stream of `NssSeed`. Two NSS instances
//!   from the same `(NssConfig, NssSeed)` are bit-identical.
//! - `fit(traj)` mutates the head's weights deterministically — same
//!   trajectory in, same head out.
//! - `predict_next` is pure with respect to NSS state; reset-on-fit
//!   semantics make `fit(traj); predict_next(state)` reproducible.

use crate::encoder::{EncoderConfig, SystemEncoder};
use crate::error::NssError;
use crate::failure::{FailureKind, FailurePrediction};
use crate::heads::{CausalAttribution, CausalAttributionHead, FailurePredictionHead, HeadConfig};
use crate::pressure::PressureKind;
use crate::propagation::PropagationConfig;
use crate::replay::TransitionRecord;
use crate::seed::{InputHash, NssRunId, NssSeed};
use crate::system::{SystemState, SystemTrajectory};
use crate::temporal::{TemporalStateConfig, TemporalStateEngine};
use crate::NSS_MODEL_VERSION;

/// Knobs for the full NSS predictor. Wires `EncoderConfig`,
/// `TemporalStateConfig`, `HeadConfig`, and `PropagationConfig` so the
/// caller controls one knob bag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NssConfig {
    /// Encoder config.
    pub encoder: EncoderConfig,
    /// Temporal-state config (`input_dim` must equal `encoder.latent_dim`).
    pub temporal: TemporalStateConfig,
    /// Head config (`state_dim` must equal `temporal.state_dim`).
    pub head: HeadConfig,
    /// Propagation config (used when the predictor needs to roll a
    /// hypothetical SystemState forward; Phase 1 doesn't roll the
    /// simulator from inside `predict_next`, but the config is bound
    /// into the run-id so audit traces capture it).
    pub propagation: PropagationConfig,
    /// Calibration weight on the count-based logit boost. Default 1.0.
    /// Must be ≥ 0 and finite.
    pub calibration_gain: f64,
}

impl Default for NssConfig {
    fn default() -> Self {
        Self {
            encoder: EncoderConfig::default(),
            temporal: TemporalStateConfig::default(),
            head: HeadConfig::default(),
            propagation: PropagationConfig::default(),
            calibration_gain: 1.0,
        }
    }
}

impl NssConfig {
    /// Validate all the sub-configs *and* their dimension agreement.
    pub fn validate(&self) -> Result<(), NssError> {
        self.encoder.validate()?;
        self.temporal.validate()?;
        self.head.validate()?;
        self.propagation.validate()?;
        if self.temporal.input_dim != self.encoder.latent_dim {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "temporal.input_dim ({}) must equal encoder.latent_dim ({})",
                    self.temporal.input_dim, self.encoder.latent_dim
                ),
            });
        }
        if self.head.state_dim != self.temporal.state_dim {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "head.state_dim ({}) must equal temporal.state_dim ({})",
                    self.head.state_dim, self.temporal.state_dim
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

    /// Canonical bytes — sub-configs concatenated.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);
        bytes.extend_from_slice(&self.encoder.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.temporal.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.head.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.propagation.canonical_bytes());
        bytes.push(b'#');
        bytes.extend_from_slice(&self.calibration_gain.to_bits().to_le_bytes());
        bytes
    }
}

/// One NSS prediction: failure probabilities + causal attribution + the
/// transition record that audit/replay validators consume.
#[derive(Clone, Debug, PartialEq)]
pub struct NssPrediction {
    /// Content-addressed run identifier. Two predictions with identical
    /// `(seed, config, input)` produce identical `run_id`s.
    pub run_id: NssRunId,
    /// Hash of the input state's canonical bytes (one component of
    /// `run_id`).
    pub input_hash: InputHash,
    /// Failure head output.
    pub failure: FailurePrediction,
    /// Causal attribution over `PressureKind`s.
    pub attribution: CausalAttribution,
    /// One-step transition record (audit trace).
    pub transition: TransitionRecord,
}

/// The Phase 1 NSS predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralSystemsSimulator {
    cfg: NssConfig,
    seed: NssSeed,
    encoder: SystemEncoder,
    temporal: TemporalStateEngine,
    head: FailurePredictionHead,
    attribution: CausalAttributionHead,
    /// `true` after `fit` has been called at least once.
    fitted: bool,
    /// Hidden state across `predict_next` calls if the caller streams
    /// (Phase 3+ will expose `predict_stream`). Phase 1: always zeros.
    h0: Vec<f64>,
}

impl NeuralSystemsSimulator {
    /// Build from `(cfg, seed)`. Validates the config.
    pub fn from_seed(cfg: NssConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let encoder = SystemEncoder::from_seed(cfg.encoder, seed)?;
        let temporal = TemporalStateEngine::from_seed(cfg.temporal, seed)?;
        let head = FailurePredictionHead::from_seed(cfg.head, seed)?;
        let attribution = CausalAttributionHead::new(cfg.head)?;
        let h0 = temporal.zero_state();
        Ok(Self {
            cfg,
            seed,
            encoder,
            temporal,
            head,
            attribution,
            fitted: false,
            h0,
        })
    }

    /// Borrow the config.
    pub fn config(&self) -> &NssConfig {
        &self.cfg
    }

    /// Borrow the seed.
    pub fn seed(&self) -> NssSeed {
        self.seed
    }

    /// `true` once `fit` has run successfully.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Phase 1 calibration: walk the trajectory and adjust per-feature
    /// collapse-logit weights by the (P(feature | next=Collapse) -
    /// P(feature | next=Nominal)) contrast. Idempotent if you call it
    /// twice with the same trajectory.
    pub fn fit(&mut self, traj: &SystemTrajectory) -> Result<(), NssError> {
        if traj.len() < 2 {
            return Err(NssError::InvalidTrajectory {
                detail: "fit requires a trajectory of length >= 2".into(),
            });
        }

        // Reset weights to a fresh deterministic init so fit is
        // idempotent: running fit twice with the same traj yields
        // identical weights.
        self.head = FailurePredictionHead::from_seed(self.cfg.head, self.seed)?;
        self.h0 = self.temporal.zero_state();

        // Tally per-feature means under (next=Collapse) vs (next≠Collapse).
        let n_pressure_features = PressureKind::all().len(); // 9
        let mut sum_collapse = vec![0.0_f64; n_pressure_features];
        let mut sum_other = vec![0.0_f64; n_pressure_features];
        let mut n_collapse: u64 = 0;
        let mut n_other: u64 = 0;

        for window in traj.as_slice().windows(2) {
            let now = &window[0];
            let next = &window[1];
            let saturations: Vec<f64> = PressureKind::all()
                .iter()
                .map(|k| {
                    now.state
                        .pressures
                        .get(*k)
                        .map(|p| p.saturation())
                        .unwrap_or(0.0)
                })
                .collect();
            if next.failure.kind == FailureKind::Collapse {
                for (i, v) in saturations.iter().enumerate() {
                    sum_collapse[i] += v;
                }
                n_collapse += 1;
            } else {
                for (i, v) in saturations.iter().enumerate() {
                    sum_other[i] += v;
                }
                n_other += 1;
            }
        }

        if n_collapse == 0 && n_other == 0 {
            // No useful transitions at all; leave weights at random
            // init.
            self.fitted = true;
            return Ok(());
        }

        // When one bucket is empty, fall back to a centred prior of
        // 0.5 (saturations are bounded to [0, 1], so 0.5 is the
        // uninformed midpoint). This makes the fit semantics
        // well-defined on pathological trajectories — e.g. one whose
        // every transition has `next=Collapse` (instant overload).
        let mean_collapse: Vec<f64> = if n_collapse == 0 {
            vec![0.5; n_pressure_features]
        } else {
            sum_collapse.iter().map(|s| s / n_collapse as f64).collect()
        };
        let mean_other: Vec<f64> = if n_other == 0 {
            vec![0.5; n_pressure_features]
        } else {
            sum_other.iter().map(|s| s / n_other as f64).collect()
        };
        let state_dim = self.cfg.head.state_dim;

        // Mutating the head's collapse weights in-place. Because
        // `FailurePredictionHead` doesn't expose a mutable accessor,
        // we rebuild it from `(cfg, seed)` (which we just did above)
        // and add the calibration boost via a controlled API.
        let mut w_new = self.head.collapse_weights().to_vec();
        // Pressure features start at index `state_dim` in the head
        // input.
        for i in 0..n_pressure_features {
            let boost = self.cfg.calibration_gain * (mean_collapse[i] - mean_other[i]);
            w_new[state_dim + i] += boost;
        }
        // Set via a small re-build helper.
        self.head = FailurePredictionHead::with_collapse_weights(self.cfg.head, self.seed, w_new)?;
        self.fitted = true;
        Ok(())
    }

    /// One-step prediction. The state is encoded, the temporal engine
    /// is stepped from `h0`, and the heads fire.
    pub fn predict_next(&self, state: &SystemState) -> Result<NssPrediction, NssError> {
        state.validate()?;
        let z = self.encoder.forward(state);
        // Phase 1: single step from the zero state. Streaming
        // prediction comes in Phase 3.
        let h = self.temporal.step(&self.h0, &z)?;
        let x = SystemEncoder::features_of(state);
        let failure = self.head.predict(&h, &x)?;
        let attribution = self.attribution.attribute(&self.head, &x);

        let input_hash = InputHash::of_bytes(&state.canonical_bytes());
        let run_id = NssRunId::build(
            self.seed,
            NSS_MODEL_VERSION,
            &self.cfg.canonical_bytes(),
            input_hash,
        );

        let transition = TransitionRecord {
            run_id,
            input_hash,
            input_tick: state.tick,
            collapse_probability: failure.collapse_probability,
            degraded_probability: failure.degraded_probability,
            confidence: failure.confidence,
            dominant_source: attribution.dominant_source.kind,
            dominant_magnitude: attribution.dominant_source.magnitude,
            model_version: NSS_MODEL_VERSION.to_string(),
        };

        Ok(NssPrediction {
            run_id,
            input_hash,
            failure,
            attribution,
            transition,
        })
    }
}

// Small extension on FailurePredictionHead to swap weights without
// exposing a generic mutable accessor.
impl FailurePredictionHead {
    /// Build a head whose collapse weights are pre-set (used by
    /// `NeuralSystemsSimulator::fit` to apply a calibration boost
    /// without exposing arbitrary write access). The degraded head is
    /// still seeded normally.
    pub fn with_collapse_weights(
        cfg: HeadConfig,
        seed: NssSeed,
        w_collapse: Vec<f64>,
    ) -> Result<Self, NssError> {
        cfg.validate()?;
        let n = cfg.input_size();
        if w_collapse.len() != n {
            return Err(NssError::InvalidConfig {
                detail: format!(
                    "with_collapse_weights: expected {} weights, got {}",
                    n,
                    w_collapse.len()
                ),
            });
        }
        let mut base = FailurePredictionHead::from_seed(cfg, seed)?;
        // Replace just the collapse weights, leave bias and degraded
        // weights alone.
        base.set_collapse_weights(w_collapse);
        Ok(base)
    }

    /// Internal setter used by `with_collapse_weights`. Not exposed in
    /// the public API to keep the surface narrow.
    pub(crate) fn set_collapse_weights(&mut self, w: Vec<f64>) {
        // Public visibility lives in this same crate; the helper
        // function above gatekeeps usage so the head can't be silently
        // mutated from outside.
        let cfg_n = self.config().input_size();
        debug_assert_eq!(w.len(), cfg_n);
        // SAFETY: we own the only handle to self; no aliasing.
        // We re-derive a fresh head from the seed and then overwrite
        // the collapse weight slice. Bias/degraded stay at their
        // seeded init.
        self.replace_collapse_inplace(w);
    }

    /// In-place collapse-weight replacement. Tiny private helper kept
    /// here so the struct's fields stay private to this crate.
    fn replace_collapse_inplace(&mut self, w: Vec<f64>) {
        // This depends on the head struct's private field. We use
        // `self.collapse_weights()` as a check and mutate via a
        // module-internal interface defined in heads.rs.
        crate::heads::__set_collapse_weights(self, w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::{QueueConfig, QueueSimulator};

    #[test]
    fn config_validates_dimension_agreement() {
        let mut bad = NssConfig::default();
        bad.encoder.latent_dim = 8; // temporal.input_dim defaults to 16
        assert!(bad.validate().is_err());
        let mut bad = NssConfig::default();
        bad.head.state_dim = 8;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn determinism_two_instances_match() {
        let a = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let b = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let s = SystemState::initial();
        let pa = a.predict_next(&s).unwrap();
        let pb = b.predict_next(&s).unwrap();
        assert_eq!(pa.run_id, pb.run_id);
        assert_eq!(pa.failure, pb.failure);
        assert_eq!(pa.attribution, pb.attribution);
    }

    #[test]
    fn predict_runs_on_default_state() {
        let nss = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let s = SystemState::initial();
        let p = nss.predict_next(&s).unwrap();
        assert!(p.failure.collapse_probability >= 0.0 && p.failure.collapse_probability <= 1.0);
    }

    #[test]
    fn fit_is_idempotent_on_same_trajectory() {
        let mut sim = QueueSimulator::new(QueueConfig::default(), NssSeed(42)).unwrap();
        let traj = sim.run(64).unwrap();
        let mut a = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let mut b = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        a.fit(&traj).unwrap();
        a.fit(&traj).unwrap();
        b.fit(&traj).unwrap();
        let s = traj.last_state().unwrap();
        let pa = a.predict_next(s).unwrap();
        let pb = b.predict_next(s).unwrap();
        assert_eq!(pa.failure, pb.failure);
    }

    #[test]
    fn fit_changes_head_weights_when_label_signal_exists() {
        // Build a trajectory with a mix of Collapse and non-Collapse
        // ticks (the overload config ramps from Nominal → Degraded →
        // Collapse, so there *is* meaningful contrast across the
        // ramp-up window).
        let cfg = QueueConfig {
            workers: 1,
            queue_capacity: 4,
            arrival_rate: 8.0,
            service_min: 1.0,
            service_max: 1.0,
            degraded_knee: 0.25,
            collapse_window: 2,
            retry_amplifier: 1.0,
            propagation: PropagationConfig::default(),
        };
        let mut sim = QueueSimulator::new(cfg, NssSeed(42)).unwrap();
        let traj = sim.run(64).unwrap();
        // Sanity-check the trajectory carries both labels.
        let n_collapse = traj
            .iter()
            .filter(|ev| ev.failure.kind == crate::FailureKind::Collapse)
            .count();
        let n_other = traj.len() - n_collapse;
        assert!(
            n_collapse > 0 && n_other > 0,
            "trajectory must carry both labels"
        );

        let mut nss = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let before_weights = nss.head.collapse_weights().to_vec();
        nss.fit(&traj).unwrap();
        let after_weights = nss.head.collapse_weights().to_vec();
        // At least one pressure-feature collapse weight must have moved.
        // The bias and the latent-state weights are untouched, so the
        // diff lives exclusively in indices [state_dim .. state_dim+9].
        let mut any_changed = false;
        for i in 0..9 {
            let idx = nss.cfg.head.state_dim + i;
            if (before_weights[idx] - after_weights[idx]).abs() > 1e-15 {
                any_changed = true;
                break;
            }
        }
        assert!(
            any_changed,
            "fit did not adjust any pressure-feature collapse weight"
        );
    }

    #[test]
    fn run_id_stable_across_predict_calls() {
        let nss = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let s = SystemState::initial();
        let a = nss.predict_next(&s).unwrap();
        let b = nss.predict_next(&s).unwrap();
        assert_eq!(a.run_id, b.run_id);
    }

    #[test]
    fn run_id_changes_when_input_changes() {
        let nss = NeuralSystemsSimulator::from_seed(NssConfig::default(), NssSeed(42)).unwrap();
        let s = SystemState::initial();
        let mut s2 = SystemState::initial();
        s2.pressures.set(
            PressureKind::Queue,
            crate::pressure::Pressure::new(0.5, 1.0, 0.1).unwrap(),
        );
        let a = nss.predict_next(&s).unwrap();
        let b = nss.predict_next(&s2).unwrap();
        assert_ne!(a.run_id, b.run_id);
    }
}
