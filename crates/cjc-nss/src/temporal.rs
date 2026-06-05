//! Temporal State Engine — the recurrent core that propagates latent
//! state across ticks.
//!
//! Phase 1 ships a simple SSM-style recurrence:
//!
//! ```text
//! h_{t+1} = tanh(A · h_t + B · z_t)
//! ```
//!
//! where `z_t` is the encoder output at tick `t`. Stability is
//! **structural**: the transition matrix `A` is constructed so that its
//! spectral norm satisfies `||A||_2 ≤ alpha < 1` by construction — the
//! same trick used by [`cjc_cronos_gan::StateSpaceModel`]. With `alpha =
//! 0.95` (default) the engine forgets perturbations exponentially fast
//! and the rollout cannot blow up, regardless of training state.
//!
//! Phase 3 adds:
//! - Liquid-time-constant gating (per-dim adaptive forgetting).
//! - Multi-timescale memory (short/medium/long/structural buffers).
//! - State-space layer with diagonal-plus-low-rank parameterisation.

use crate::error::NssError;
use crate::seed::NssSeed;
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Knobs for the temporal-state engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalStateConfig {
    /// Hidden state dimensionality. Must be ≥ 1. Default 16.
    pub state_dim: usize,
    /// Encoder output dimensionality (must equal
    /// [`crate::EncoderConfig::latent_dim`]). Must be ≥ 1.
    pub input_dim: usize,
    /// Spectral-norm bound on the transition matrix. Must be in
    /// `(0, 1)`. Default 0.95.
    pub alpha: f64,
    /// Standard-deviation scale for the input matrix init. Default 0.1.
    pub init_scale: f64,
}

impl Default for TemporalStateConfig {
    fn default() -> Self {
        Self {
            state_dim: 16,
            input_dim: 16,
            alpha: 0.95,
            init_scale: 0.1,
        }
    }
}

impl TemporalStateConfig {
    /// Validate.
    pub fn validate(&self) -> Result<(), NssError> {
        if self.state_dim == 0 {
            return Err(NssError::InvalidConfig {
                detail: "state_dim must be >= 1".into(),
            });
        }
        if self.input_dim == 0 {
            return Err(NssError::InvalidConfig {
                detail: "input_dim must be >= 1".into(),
            });
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(NssError::InvalidConfig {
                detail: format!("alpha must satisfy 0 < alpha < 1, got {}", self.alpha),
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
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&(self.state_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&self.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.init_scale.to_bits().to_le_bytes());
        bytes
    }
}

/// The temporal-state engine.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalStateEngine {
    cfg: TemporalStateConfig,
    /// Transition matrix `A`, `[state_dim × state_dim]`, row-major.
    /// Constructed so `||A||_2 ≤ alpha` by row-L²-normalisation +
    /// scale by `alpha / sqrt(state_dim)`.
    a: Vec<f64>,
    /// Input matrix `B`, `[state_dim × input_dim]`, row-major.
    b: Vec<f64>,
}

impl TemporalStateEngine {
    /// Build from a seed.
    pub fn from_seed(cfg: TemporalStateConfig, seed: NssSeed) -> Result<Self, NssError> {
        cfg.validate()?;
        let mut rng_a = seed.substream("temporal.A");
        let mut rng_b = seed.substream("temporal.B");
        let a = init_transition_matrix(&mut rng_a, cfg.state_dim, cfg.alpha);
        let b = init_matrix(&mut rng_b, cfg.state_dim, cfg.input_dim, cfg.init_scale);
        Ok(Self { cfg, a, b })
    }

    /// Borrow the config.
    pub fn config(&self) -> &TemporalStateConfig {
        &self.cfg
    }

    /// Borrow the transition matrix (audit + tests).
    pub fn transition_matrix(&self) -> &[f64] {
        &self.a
    }

    /// One step: `h_{t+1} = tanh(A·h_t + B·z_t)`. Kahan-summed.
    pub fn step(&self, h_prev: &[f64], z: &[f64]) -> Result<Vec<f64>, NssError> {
        if h_prev.len() != self.cfg.state_dim {
            return Err(NssError::InvalidState {
                detail: format!(
                    "h_prev length {} != state_dim {}",
                    h_prev.len(),
                    self.cfg.state_dim
                ),
            });
        }
        if z.len() != self.cfg.input_dim {
            return Err(NssError::InvalidState {
                detail: format!("z length {} != input_dim {}", z.len(), self.cfg.input_dim),
            });
        }
        let mut h_next = vec![0.0; self.cfg.state_dim];
        for i in 0..self.cfg.state_dim {
            let mut acc = KahanAccumulatorF64::new();
            let a_off = i * self.cfg.state_dim;
            for j in 0..self.cfg.state_dim {
                acc.add(self.a[a_off + j] * h_prev[j]);
            }
            let b_off = i * self.cfg.input_dim;
            for j in 0..self.cfg.input_dim {
                acc.add(self.b[b_off + j] * z[j]);
            }
            h_next[i] = acc.finalize().tanh();
        }
        Ok(h_next)
    }

    /// Zero initial state.
    pub fn zero_state(&self) -> Vec<f64> {
        vec![0.0; self.cfg.state_dim]
    }
}

/// Initialise an `n × n` transition matrix with Frobenius norm exactly
/// `alpha`, so `||A||_2 ≤ alpha < 1` by construction. We draw an N(0, 1)
/// matrix, then rescale so the Frobenius norm matches `alpha`. Since
/// `||A||_2 ≤ ||A||_F` for any matrix, this is a structural-stability
/// guarantee, not an empirical hope.
fn init_transition_matrix(rng: &mut Rng, n: usize, alpha: f64) -> Vec<f64> {
    let mut v = vec![0.0; n * n];
    for slot in v.iter_mut() {
        *slot = rng.next_normal_f64();
    }
    // Frobenius norm, Kahan-summed for determinism.
    let mut acc = KahanAccumulatorF64::new();
    for x in &v {
        acc.add(x * x);
    }
    let frob = acc.finalize().sqrt();
    if frob == 0.0 {
        // Effectively impossible from a continuous N(0,1) draw but
        // guard for the corner anyway: leave the matrix zero, which
        // is trivially stable.
        return v;
    }
    let scale = alpha / frob;
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

fn init_matrix(rng: &mut Rng, rows: usize, cols: usize, scale: f64) -> Vec<f64> {
    let mut v = vec![0.0; rows * cols];
    for slot in v.iter_mut() {
        *slot = scale * rng.next_normal_f64();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frobenius_norm_equals_alpha() {
        let cfg = TemporalStateConfig::default();
        let eng = TemporalStateEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let mut acc = KahanAccumulatorF64::new();
        for x in eng.transition_matrix() {
            acc.add(x * x);
        }
        let frob = acc.finalize().sqrt();
        assert!((frob - cfg.alpha).abs() < 1e-12);
    }

    #[test]
    fn determinism_same_seed_same_state_after_n_steps() {
        let cfg = TemporalStateConfig::default();
        let a = TemporalStateEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let b = TemporalStateEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let z = vec![0.1; cfg.input_dim];
        let mut ha = a.zero_state();
        let mut hb = b.zero_state();
        for _ in 0..32 {
            ha = a.step(&ha, &z).unwrap();
            hb = b.step(&hb, &z).unwrap();
        }
        assert_eq!(ha, hb);
    }

    #[test]
    fn stability_under_constant_input() {
        let cfg = TemporalStateConfig::default();
        let eng = TemporalStateEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let z = vec![1.0; cfg.input_dim];
        let mut h = eng.zero_state();
        for _ in 0..256 {
            h = eng.step(&h, &z).unwrap();
            for v in &h {
                assert!(v.is_finite() && v.abs() <= 1.0);
            }
        }
    }

    #[test]
    fn rejects_wrong_shapes() {
        let cfg = TemporalStateConfig::default();
        let eng = TemporalStateEngine::from_seed(cfg, NssSeed(42)).unwrap();
        let bad_h = vec![0.0; cfg.state_dim + 1];
        let z = vec![0.0; cfg.input_dim];
        assert!(eng.step(&bad_h, &z).is_err());
        let h = eng.zero_state();
        let bad_z = vec![0.0; cfg.input_dim + 1];
        assert!(eng.step(&h, &bad_z).is_err());
    }
}
