//! Replay engine + audit trace types.
//!
//! Every NSS prediction emits a [`TransitionRecord`] (one-step audit
//! receipt) and an [`NssRunId`] computed over `(seed, model_version,
//! config_bytes, input_hash)`. The [`ReplayValidator`] takes a
//! `TransitionRecord` and the originating `(NssConfig, NssSeed,
//! SystemState)` triple, re-runs the prediction, and verifies the
//! recomputed `run_id` matches.
//!
//! This is infrastructure-grade auditability: anyone can verify that a
//! published prediction was actually produced by the claimed inputs.

use crate::error::NssError;
use crate::nss::{NeuralSystemsSimulator, NssConfig};
use crate::pressure::PressureKind;
use crate::seed::{InputHash, NssRunId, NssSeed};
use crate::system::{SystemState, SystemTrajectory};

/// One-step audit-trace record. Carried inside every
/// [`crate::NssPrediction`].
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionRecord {
    /// Content-addressed run id.
    pub run_id: NssRunId,
    /// Input-state hash.
    pub input_hash: InputHash,
    /// Tick of the input state (audit log convenience).
    pub input_tick: u64,
    /// Predicted collapse probability.
    pub collapse_probability: f64,
    /// Predicted degraded probability.
    pub degraded_probability: f64,
    /// Calibrated confidence.
    pub confidence: f64,
    /// Most-impactful pressure kind for the collapse prediction.
    pub dominant_source: PressureKind,
    /// Signed contribution magnitude (informational; sign matches the
    /// collapse-logit contribution, so a negative value means the
    /// "dominant" source is actually pushing collapse probability
    /// *down* — interpret accordingly).
    pub dominant_magnitude: f64,
    /// Model version stamped at prediction time.
    pub model_version: String,
}

impl TransitionRecord {
    /// Canonical bytes — used for downstream lineage hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(self.run_id.0 .0.to_le_bytes().as_ref());
        bytes.extend_from_slice(self.input_hash.0 .0.to_le_bytes().as_ref());
        bytes.extend_from_slice(&self.input_tick.to_le_bytes());
        bytes.extend_from_slice(&self.collapse_probability.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.degraded_probability.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.confidence.to_bits().to_le_bytes());
        bytes.extend_from_slice(self.dominant_source.label().as_bytes());
        bytes.push(b'|');
        bytes.extend_from_slice(&self.dominant_magnitude.to_bits().to_le_bytes());
        bytes.push(b'|');
        bytes.extend_from_slice(self.model_version.as_bytes());
        bytes
    }
}

/// Full audit-trace record — Phase 1 ships the prediction + one upstream
/// transition; future phases will chain N upstream transitions into a
/// `LineageChain`.
///
/// If the originating predictor was `fit`'d before predicting, the
/// trace MUST carry the training trajectory — otherwise the
/// [`ReplayValidator`] cannot reproduce the fitted weights and the
/// replay verification will fail. For un-fitted predictions, leave
/// `training_trajectory` as `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionTrace {
    /// The transition record (also lives inside the prediction).
    pub transition: TransitionRecord,
    /// Snapshot of the originating system state — included so an
    /// auditor can re-run the prediction without a separate input
    /// channel.
    pub input_state: SystemState,
    /// Snapshot of the originating config — included so an auditor can
    /// rebuild the predictor without out-of-band knowledge.
    pub input_config: NssConfig,
    /// Originating seed.
    pub input_seed: NssSeed,
    /// Training trajectory used to `fit` the predictor before the
    /// prediction was made. `None` means the prediction was made
    /// without fitting. If `Some`, the validator will call `fit` on
    /// the rebuilt NSS with this trajectory before re-predicting.
    pub training_trajectory: Option<SystemTrajectory>,
}

impl PredictionTrace {
    /// Canonical bytes for the full trace. Two traces produced from the
    /// same `(state, config, seed, training_trajectory)` produce
    /// identical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.transition.canonical_bytes();
        bytes.push(b'\n');
        bytes.extend_from_slice(&self.input_state.canonical_bytes());
        bytes.push(b'\n');
        bytes.extend_from_slice(&self.input_config.canonical_bytes());
        bytes.push(b'\n');
        bytes.extend_from_slice(&self.input_seed.0.to_le_bytes());
        bytes.push(b'\n');
        match &self.training_trajectory {
            Some(t) => {
                bytes.extend_from_slice(b"fit:");
                bytes.extend_from_slice(&t.canonical_bytes());
            }
            None => bytes.extend_from_slice(b"unfit"),
        }
        bytes
    }
}

/// Replay validator — recomputes the prediction from the trace inputs
/// and verifies the produced run_id matches.
#[derive(Clone, Debug, Default)]
pub struct ReplayValidator;

impl ReplayValidator {
    /// Build (zero-state — the validator has no internal config).
    pub fn new() -> Self {
        Self
    }

    /// Verify that re-running NSS on `(trace.input_config,
    /// trace.input_seed, trace.input_state)` — with the optional
    /// `trace.training_trajectory` applied via `fit` — reproduces
    /// `trace.transition.run_id` and identical head outputs.
    ///
    /// Returns `Ok(())` on success and
    /// [`NssError::ReplayMismatch`] on disagreement.
    pub fn verify(&self, trace: &PredictionTrace) -> Result<(), NssError> {
        let mut nss = NeuralSystemsSimulator::from_seed(trace.input_config, trace.input_seed)?;
        if let Some(t) = trace.training_trajectory.as_ref() {
            nss.fit(t)?;
        }
        let p = nss.predict_next(&trace.input_state)?;
        if p.run_id != trace.transition.run_id {
            return Err(NssError::ReplayMismatch {
                expected: trace.transition.run_id.to_string(),
                actual: p.run_id.to_string(),
            });
        }
        // Probabilities must be bit-identical — Kahan determinism gives
        // us this for free.
        if p.failure.collapse_probability.to_bits()
            != trace.transition.collapse_probability.to_bits()
            || p.failure.degraded_probability.to_bits()
                != trace.transition.degraded_probability.to_bits()
        {
            return Err(NssError::ReplayMismatch {
                expected: format!(
                    "collapse={} degraded={}",
                    trace.transition.collapse_probability, trace.transition.degraded_probability
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

    #[test]
    fn trace_canonical_bytes_stable() {
        let cfg = NssConfig::default();
        let seed = NssSeed(42);
        let nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        let s = SystemState::initial();
        let p = nss.predict_next(&s).unwrap();
        let trace_a = PredictionTrace {
            transition: p.transition.clone(),
            input_state: s.clone(),
            input_config: cfg,
            input_seed: seed,
            training_trajectory: None,
        };
        let trace_b = PredictionTrace {
            transition: p.transition,
            input_state: s,
            input_config: cfg,
            input_seed: seed,
            training_trajectory: None,
        };
        assert_eq!(trace_a.canonical_bytes(), trace_b.canonical_bytes());
    }

    #[test]
    fn replay_validator_accepts_unmodified_trace() {
        let cfg = NssConfig::default();
        let seed = NssSeed(42);
        let nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        let s = SystemState::initial();
        let p = nss.predict_next(&s).unwrap();
        let trace = PredictionTrace {
            transition: p.transition,
            input_state: s,
            input_config: cfg,
            input_seed: seed,
            training_trajectory: None,
        };
        ReplayValidator::new().verify(&trace).unwrap();
    }

    #[test]
    fn replay_validator_rejects_tampered_seed() {
        let cfg = NssConfig::default();
        let seed = NssSeed(42);
        let nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        let s = SystemState::initial();
        let p = nss.predict_next(&s).unwrap();
        let bad = PredictionTrace {
            transition: p.transition,
            input_state: s,
            input_config: cfg,
            input_seed: NssSeed(99), // tampered
            training_trajectory: None,
        };
        let r = ReplayValidator::new().verify(&bad);
        assert!(matches!(r, Err(NssError::ReplayMismatch { .. })));
    }

    #[test]
    fn replay_validator_rejects_tampered_probability() {
        let cfg = NssConfig::default();
        let seed = NssSeed(42);
        let nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        let s = SystemState::initial();
        let p = nss.predict_next(&s).unwrap();
        let mut tr = p.transition;
        tr.collapse_probability += 1e-9;
        let bad = PredictionTrace {
            transition: tr,
            input_state: s,
            input_config: cfg,
            input_seed: seed,
            training_trajectory: None,
        };
        let r = ReplayValidator::new().verify(&bad);
        assert!(matches!(r, Err(NssError::ReplayMismatch { .. })));
    }

    #[test]
    fn replay_after_fit_is_stable() {
        let cfg = NssConfig::default();
        let seed = NssSeed(42);
        let mut sim = crate::simulator::QueueSimulator::new(
            crate::simulator::QueueConfig::default(),
            seed,
        )
        .unwrap();
        let traj = sim.run(64).unwrap();
        // Build, fit, predict.
        let mut nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        nss.fit(&traj).unwrap();
        let s = traj.last_state().unwrap().clone();
        let p = nss.predict_next(&s).unwrap();

        // Rebuild and verify — must reproduce.
        // (Because fit is deterministic + idempotent on the same traj.)
        let mut nss2 = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        nss2.fit(&traj).unwrap();
        let p2 = nss2.predict_next(&s).unwrap();
        assert_eq!(p.run_id, p2.run_id);
        assert_eq!(
            p.failure.collapse_probability.to_bits(),
            p2.failure.collapse_probability.to_bits()
        );

        // Replay verifier with training_trajectory: Some(traj) must
        // also accept the trace — this is the new path that closes
        // the post-fit replay gap.
        let trace = PredictionTrace {
            transition: p.transition,
            input_state: s,
            input_config: cfg,
            input_seed: seed,
            training_trajectory: Some(traj),
        };
        ReplayValidator::new().verify(&trace).unwrap();
    }

    #[test]
    fn replay_after_fit_rejects_missing_training_trajectory() {
        // Symmetric negative case: if a trace says "this prediction was
        // produced after fit" but doesn't carry the training trajectory,
        // the validator must reject it. We model this by emitting a
        // post-fit prediction and then dropping training_trajectory.
        let cfg = NssConfig::default();
        let seed = NssSeed(42);
        let mut sim = crate::simulator::QueueSimulator::new(
            // Use an overload config so fit *will* change weights.
            crate::simulator::QueueConfig {
                workers: 1,
                queue_capacity: 4,
                arrival_rate: 6.0,
                service_min: 1.0,
                service_max: 1.0,
                degraded_knee: 0.5,
                collapse_window: 2,
                retry_amplifier: 1.0,
                propagation: Default::default(),
            },
            seed,
        )
        .unwrap();
        let traj = sim.run(32).unwrap();
        let mut nss = NeuralSystemsSimulator::from_seed(cfg, seed).unwrap();
        nss.fit(&traj).unwrap();
        let s = traj.last_state().unwrap().clone();
        let p = nss.predict_next(&s).unwrap();

        let trace = PredictionTrace {
            transition: p.transition,
            input_state: s,
            input_config: cfg,
            input_seed: seed,
            training_trajectory: None, // intentionally missing
        };
        // Should reject — the run_id was computed against fitted
        // weights, and the validator (without the trajectory) reads
        // un-fitted weights.
        let r = ReplayValidator::new().verify(&trace);
        assert!(matches!(r, Err(NssError::ReplayMismatch { .. })));
    }
}
