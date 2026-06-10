//! Prediction heads: failure-prediction and causal attribution.
//!
//! Both heads consume the temporal-state hidden vector `h_t` and the raw
//! pressure-saturations of the current state. Phase 1 wires:
//!
//! - [`FailurePredictionHead`] — two affine + sigmoid logits for
//!   "collapse next tick" and "degraded next tick", from `[h_t ⊕ pressure_features]`.
//! - [`CausalAttributionHead`] — decomposes the collapse logit into
//!   per-feature contributions (`weight * feature`), then aggregates by
//!   [`PressureKind`]. The result is the **exact** attribution of the
//!   prediction, not a post-hoc approximation.
//!
//! Phase 3 will add separate heads for throughput-degradation, latency
//! spike, and partial outage, plus a scheduler-advisory head.

use crate::encoder::EncoderConfig;
use crate::error::NssError;
use crate::failure::FailurePrediction;
use crate::pressure::PressureKind;
use crate::seed::NssSeed;
use crate::system::SystemState;
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Knobs for the prediction heads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadConfig {
    /// Hidden state size — must match
    /// [`crate::TemporalStateConfig::state_dim`].
    pub state_dim: usize,
    /// Standard-deviation scale for the head-weight init.
    pub init_scale: f64,
}

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            state_dim: 16,
            init_scale: 0.1,
        }
    }
}

impl HeadConfig {
    /// Total head input size = state_dim + INPUT_FEATURES (so the head
    /// sees both the temporal latent and the raw pressure features).
    pub fn input_size(&self) -> usize {
        self.state_dim + EncoderConfig::INPUT_FEATURES
    }

    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.state_dim == 0 {
            return Err(NssError::InvalidConfig {
                detail: "state_dim must be >= 1".into(),
            });
        }
        if !self.init_scale.is_finite() || self.init_scale <= 0.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("init_scale must be > 0 and finite, got {}", self.init_scale),
            });
        }
        Ok(())
    }

    /// Canonical bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&(self.state_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&self.init_scale.to_bits().to_le_bytes());
        bytes
    }
}

/// Failure-prediction head: two parallel logits (collapse, degraded).
#[derive(Clone, Debug, PartialEq)]
pub struct FailurePredictionHead {
    cfg: HeadConfig,
    /// Collapse-logit weights: length = input_size.
    w_collapse: Vec<f64>,
    /// Collapse-logit bias.
    b_collapse: f64,
    /// Degraded-logit weights: length = input_size.
    w_degraded: Vec<f64>,
    /// Degraded-logit bias.
    b_degraded: f64,
}

impl FailurePredictionHead {
    /// Build from a seed.
    pub fn from_seed(cfg: HeadConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let n = cfg.input_size();
        let mut rng_c = seed.substream("head.collapse.W");
        let mut rng_d = seed.substream("head.degraded.W");
        let mut rng_bc = seed.substream("head.collapse.b");
        let mut rng_bd = seed.substream("head.degraded.b");
        Ok(Self {
            cfg,
            w_collapse: init_vector(&mut rng_c, n, cfg.init_scale),
            b_collapse: cfg.init_scale * rng_bc.next_normal_f64(),
            w_degraded: init_vector(&mut rng_d, n, cfg.init_scale),
            b_degraded: cfg.init_scale * rng_bd.next_normal_f64(),
        })
    }

    /// Predict failure probabilities from latent `h` and raw feature
    /// vector `x`. Returns probabilities in `[0, 1]` (sigmoid).
    pub fn predict(
        &self,
        h: &[f64],
        x: &[f64; EncoderConfig::INPUT_FEATURES],
    ) -> Result<FailurePrediction, NssError> {
        if h.len() != self.cfg.state_dim {
            return Err(NssError::InvalidState {
                detail: format!("h.len() {} != state_dim {}", h.len(), self.cfg.state_dim),
            });
        }
        let z = concat_h_x(h, x);
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

    /// Borrow collapse weights (for the attribution head).
    pub fn collapse_weights(&self) -> &[f64] {
        &self.w_collapse
    }

    /// Borrow collapse bias.
    pub fn collapse_bias(&self) -> f64 {
        self.b_collapse
    }

    /// Borrow config.
    pub fn config(&self) -> &HeadConfig {
        &self.cfg
    }
}

/// Per-pressure-kind contribution to the failure logit. Returned by the
/// attribution head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureContribution {
    /// Which kind.
    pub kind: PressureKind,
    /// Signed contribution to the collapse logit.
    pub magnitude: f64,
}

/// Attribution head — decomposes the failure-head logit into per-kind
/// contributions. Cheap and exact: no post-hoc explainer, no Shapley
/// approximation, just `weight * feature` grouped by feature category.
///
/// In Phase 3 this will also weigh latent contributions via a learnable
/// gating layer.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalAttributionHead {
    cfg: HeadConfig,
}

/// Output of the attribution head.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalAttribution {
    /// One entry per [`PressureKind`], sorted by descending |magnitude|.
    pub contributions: Vec<PressureContribution>,
    /// Best-effort "dominant source": the kind with the largest
    /// positive contribution to the collapse logit. If all
    /// contributions are negative or zero, this defaults to the first
    /// listed entry (largest absolute magnitude) with `.magnitude = 0`.
    pub dominant_source: PressureContribution,
}

impl CausalAttributionHead {
    /// Build (config-only; this head has no learnable parameters in
    /// Phase 1 because it reads the failure-head's weights).
    pub fn new(cfg: HeadConfig) -> Result<Self, NssError> {
        cfg.validate()?;
        Ok(Self { cfg })
    }

    /// Compute attributions from the failure head's collapse weights
    /// and the current pressure features.
    pub fn attribute(
        &self,
        failure_head: &FailurePredictionHead,
        x: &[f64; EncoderConfig::INPUT_FEATURES],
    ) -> CausalAttribution {
        // The head input is [h | x]. The pressure features live at
        // x[0..9] in canonical PressureKind order. Latent contributions
        // (the first state_dim weights) are *not* attributed in Phase
        // 1 — they're folded into a single "latent" residual that we
        // intentionally don't surface as a PressureKind.
        let w = failure_head.collapse_weights();
        let state_dim = failure_head.config().state_dim;
        let kinds = PressureKind::all();
        let mut contribs: Vec<PressureContribution> = kinds
            .iter()
            .enumerate()
            .map(|(i, k)| PressureContribution {
                kind: *k,
                magnitude: w[state_dim + i] * x[i],
            })
            .collect();
        // Largest |magnitude| first; tie-break by canonical PressureKind
        // ordering so two ties don't flip across runs.
        contribs.sort_by(|a, b| {
            b.magnitude
                .abs()
                .partial_cmp(&a.magnitude.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.kind.cmp(&b.kind))
        });
        // Dominant source: largest *positive* contribution (drives
        // collapse up). Fall back to the largest-absolute entry if no
        // positive contribution exists.
        let dominant = contribs
            .iter()
            .find(|c| c.magnitude > 0.0)
            .copied()
            .unwrap_or(contribs[0]);
        CausalAttribution {
            contributions: contribs,
            dominant_source: dominant,
        }
    }

    /// Borrow config.
    pub fn config(&self) -> &HeadConfig {
        &self.cfg
    }
}

fn concat_h_x(h: &[f64], x: &[f64; EncoderConfig::INPUT_FEATURES]) -> Vec<f64> {
    let mut z = Vec::with_capacity(h.len() + EncoderConfig::INPUT_FEATURES);
    z.extend_from_slice(h);
    z.extend_from_slice(x);
    z
}

fn dot_kahan(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = KahanAccumulatorF64::new();
    for (ai, bi) in a.iter().zip(b.iter()) {
        acc.add(ai * bi);
    }
    acc.finalize()
}

fn sigmoid(x: f64) -> f64 {
    // Numerically-safe sigmoid: branch on sign so we never compute
    // exp of a large positive number.
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn init_vector(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for slot in v.iter_mut() {
        *slot = scale * rng.next_normal_f64();
    }
    v
}

/// Module-internal helper: swap the collapse-weight vector on a built
/// head. Used by `NeuralSystemsSimulator::fit` to apply a calibration
/// boost in a controlled way (the head's private fields aren't
/// accessible from `nss.rs`).
pub(crate) fn __set_collapse_weights(head: &mut FailurePredictionHead, w: Vec<f64>) {
    debug_assert_eq!(w.len(), head.cfg.input_size());
    head.w_collapse = w;
}

/// Convenience: build a [`SystemState`]'s features array (re-exported
/// from the encoder for callers that don't want to depend on the
/// encoder module directly).
pub fn features_of(state: &SystemState) -> [f64; EncoderConfig::INPUT_FEATURES] {
    crate::encoder::SystemEncoder::features_of(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{EncoderConfig, SystemEncoder};
    use crate::temporal::{TemporalStateConfig, TemporalStateEngine};

    fn fully_wired_head() -> (FailurePredictionHead, CausalAttributionHead) {
        let cfg = HeadConfig::default();
        let h = FailurePredictionHead::from_seed(cfg, NssSeed(42)).unwrap();
        let a = CausalAttributionHead::new(cfg).unwrap();
        (h, a)
    }

    #[test]
    fn predict_returns_bounded_probabilities() {
        let (head, _) = fully_wired_head();
        let h = vec![0.0; head.config().state_dim];
        let x = [0.0; EncoderConfig::INPUT_FEATURES];
        let p = head.predict(&h, &x).unwrap();
        assert!(p.collapse_probability >= 0.0 && p.collapse_probability <= 1.0);
        assert!(p.degraded_probability >= 0.0 && p.degraded_probability <= 1.0);
        assert!(p.confidence >= 0.5 && p.confidence <= 1.0);
    }

    #[test]
    fn determinism_predict_same_seed_same_output() {
        let cfg = HeadConfig::default();
        let a = FailurePredictionHead::from_seed(cfg, NssSeed(42)).unwrap();
        let b = FailurePredictionHead::from_seed(cfg, NssSeed(42)).unwrap();
        let h = vec![0.2; cfg.state_dim];
        let mut x = [0.0; EncoderConfig::INPUT_FEATURES];
        for i in 0..x.len() {
            x[i] = (i as f64) * 0.05;
        }
        let pa = a.predict(&h, &x).unwrap();
        let pb = b.predict(&h, &x).unwrap();
        assert_eq!(pa, pb);
    }

    #[test]
    fn attribution_orders_by_absolute_magnitude() {
        let (head, attr) = fully_wired_head();
        let mut x = [0.0; EncoderConfig::INPUT_FEATURES];
        // Strongly load queue (index 4 in PressureKind::all()).
        let queue_idx = PressureKind::all()
            .iter()
            .position(|k| *k == PressureKind::Queue)
            .unwrap();
        x[queue_idx] = 1.0;
        let out = attr.attribute(&head, &x);
        let largest = &out.contributions[0];
        assert_eq!(largest.kind, PressureKind::Queue);
    }

    #[test]
    fn end_to_end_pipeline_smoke() {
        // Encoder → temporal → head → attribution. Verifies wire-up.
        let enc = SystemEncoder::from_seed(EncoderConfig::default(), NssSeed(42)).unwrap();
        let tem =
            TemporalStateEngine::from_seed(TemporalStateConfig::default(), NssSeed(42)).unwrap();
        let (head, attr) = fully_wired_head();
        let state = SystemState::initial();
        let z = enc.forward(&state);
        let mut h = tem.zero_state();
        for _ in 0..8 {
            h = tem.step(&h, &z).unwrap();
        }
        let x = SystemEncoder::features_of(&state);
        let pred = head.predict(&h, &x).unwrap();
        let _att = attr.attribute(&head, &x);
        assert!(pred.collapse_probability.is_finite());
    }
}
