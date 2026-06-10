//! System Encoder: maps a [`SystemState`] into a fixed-length latent
//! vector that the [`crate::TemporalStateEngine`] can consume.
//!
//! Phase 1 is intentionally a *single* affine map + tanh activation. The
//! input feature vector is a deterministic, hand-built mapping over the
//! [`PressureKind`] saturations + a handful of scalar features (in-flight
//! count, completed delta, throughput proxy, mean service time). The
//! linear layer is initialised from `NssSeed` sub-streams; same seed ⇒
//! same weights.
//!
//! Complexity in NSS comes from the *graph-aware propagation*, not from
//! deep stacks. Phase 3+ may grow this; Phase 1 keeps it transparent.

use crate::error::NssError;
use crate::pressure::PressureKind;
use crate::seed::NssSeed;
use crate::system::SystemState;
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Knobs for the [`SystemEncoder`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncoderConfig {
    /// Latent dimension. Must be ≥ 1. Default 16.
    pub latent_dim: usize,
    /// Standard-deviation scale for the random init (Box–Muller).
    /// Default 0.1.
    pub init_scale: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            init_scale: 0.1,
        }
    }
}

impl EncoderConfig {
    /// Number of input features. Fixed at compile-time: one per
    /// [`PressureKind`] saturation + 4 scalar features.
    pub const INPUT_FEATURES: usize = 9 + 4;

    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.latent_dim == 0 {
            return Err(NssError::InvalidConfig {
                detail: "latent_dim must be >= 1".into(),
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
        bytes.extend_from_slice(&(self.latent_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&self.init_scale.to_bits().to_le_bytes());
        bytes
    }
}

/// One affine layer + tanh. `W` is `[latent_dim, INPUT_FEATURES]`
/// row-major; `b` is `[latent_dim]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemEncoder {
    cfg: EncoderConfig,
    w: Vec<f64>,
    b: Vec<f64>,
}

impl SystemEncoder {
    /// Build from a seed. Two encoders from the same `(cfg, seed)` are
    /// bit-identical.
    pub fn from_seed(cfg: EncoderConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let mut rng_w = seed.substream("encoder.W");
        let mut rng_b = seed.substream("encoder.b");
        let w = init_matrix(
            &mut rng_w,
            cfg.latent_dim,
            EncoderConfig::INPUT_FEATURES,
            cfg.init_scale,
        );
        let b = init_vector(&mut rng_b, cfg.latent_dim, cfg.init_scale);
        Ok(Self { cfg, w, b })
    }

    /// Latent dimensionality.
    pub fn latent_dim(&self) -> usize {
        self.cfg.latent_dim
    }

    /// Borrow the config.
    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// Build the deterministic input feature vector from a
    /// [`SystemState`]. Length = [`EncoderConfig::INPUT_FEATURES`].
    /// Feature ordering is fixed:
    /// `[saturations…, in_flight_log1p, completed_log1p_per_tick,
    ///   rejected_fraction, mean_service_time]`.
    pub fn features_of(state: &SystemState) -> [f64; EncoderConfig::INPUT_FEATURES] {
        let mut out = [0.0; EncoderConfig::INPUT_FEATURES];
        // Saturations in canonical PressureKind order.
        for (i, k) in PressureKind::all().iter().enumerate() {
            out[i] = state
                .pressures
                .get(*k)
                .map(|p| p.saturation())
                .unwrap_or(0.0);
        }
        // Scalar features. log1p keeps them bounded for growing
        // counters.
        let n = 9; // start of scalar features
        out[n] = (state.in_flight as f64).ln_1p();
        // Use saturating_add so fuzz inputs with `tick = u64::MAX`
        // can't overflow the denominator computation. Same for the
        // `completed + rejected` total below.
        let denom = state.tick.saturating_add(1) as f64;
        out[n + 1] = (state.completed as f64 / denom).ln_1p();
        let total = state.completed.saturating_add(state.rejected).max(1) as f64;
        out[n + 2] = state.rejected as f64 / total;
        out[n + 3] = state.mean_service_time;
        out
    }

    /// Forward pass. Returns a `latent_dim`-length vector with `tanh`
    /// activation. Kahan-summed for determinism.
    pub fn forward(&self, state: &SystemState) -> Vec<f64> {
        let x = Self::features_of(state);
        let mut out = vec![0.0; self.cfg.latent_dim];
        for i in 0..self.cfg.latent_dim {
            let mut acc = KahanAccumulatorF64::new();
            acc.add(self.b[i]);
            let row_off = i * EncoderConfig::INPUT_FEATURES;
            for j in 0..EncoderConfig::INPUT_FEATURES {
                acc.add(self.w[row_off + j] * x[j]);
            }
            out[i] = acc.finalize().tanh();
        }
        out
    }

    /// Borrow weights (read-only, for tests and audit).
    pub fn weights(&self) -> &[f64] {
        &self.w
    }

    /// Borrow biases (read-only, for tests and audit).
    pub fn biases(&self) -> &[f64] {
        &self.b
    }
}

fn init_matrix(rng: &mut Rng, rows: usize, cols: usize, scale: f64) -> Vec<f64> {
    let mut v = vec![0.0; rows * cols];
    for slot in v.iter_mut() {
        *slot = scale * rng.next_normal_f64();
    }
    v
}

fn init_vector(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    let mut v = vec![0.0; n];
    for slot in v.iter_mut() {
        *slot = scale * rng.next_normal_f64();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_features_constant_matches_layout() {
        // 9 PressureKinds + 4 scalars.
        assert_eq!(EncoderConfig::INPUT_FEATURES, 13);
    }

    #[test]
    fn determinism_same_seed_same_weights() {
        let a = SystemEncoder::from_seed(EncoderConfig::default(), NssSeed(42)).unwrap();
        let b = SystemEncoder::from_seed(EncoderConfig::default(), NssSeed(42)).unwrap();
        assert_eq!(a.weights(), b.weights());
        assert_eq!(a.biases(), b.biases());
    }

    #[test]
    fn determinism_different_seed_different_weights() {
        let a = SystemEncoder::from_seed(EncoderConfig::default(), NssSeed(42)).unwrap();
        let b = SystemEncoder::from_seed(EncoderConfig::default(), NssSeed(43)).unwrap();
        assert_ne!(a.weights(), b.weights());
    }

    #[test]
    fn forward_bounded_by_tanh() {
        let enc = SystemEncoder::from_seed(EncoderConfig::default(), NssSeed(42)).unwrap();
        let s = SystemState::initial();
        let z = enc.forward(&s);
        assert_eq!(z.len(), 16);
        for v in z {
            assert!(v.is_finite());
            assert!(v.abs() <= 1.0);
        }
    }

    #[test]
    fn features_in_canonical_pressure_order() {
        let s = SystemState::initial();
        let f = SystemEncoder::features_of(&s);
        // All default saturations are 0.0; scalar features are 0 too
        // because in_flight=0, completed=0, rejected=0 and mean_service_time=1.
        for i in 0..9 {
            assert_eq!(f[i], 0.0);
        }
        assert_eq!(f[9], 0.0);
        assert_eq!(f[10], 0.0);
        assert_eq!(f[11], 0.0);
        assert_eq!(f[12], 1.0);
    }
}
