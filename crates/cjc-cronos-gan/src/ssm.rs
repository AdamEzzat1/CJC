//! State Space Model — the **stable long-range latent dynamics adversary** of
//! the Temporal GAN.
//!
//! Phase 1 ships a deterministic, linear, noise-free SSM:
//!
//! ```text
//! x_{t+1} = A · x_t + B · u_t
//! y_t     = C · x_t + D · u_t
//! ```
//!
//! Stability is **structural, not empirical**. The transition matrix `A`
//! is constructed so its spectral norm `||A||_2 ≤ α < 1` by construction:
//!
//!   `A = (α / √state_dim) · R`
//!
//! where each row of `R` is drawn from a standard normal, then normalised
//! to unit L² length. The spectral norm of a unit-row matrix is at most
//! `√state_dim`, so the `α / √state_dim` scale collapses that bound to
//! `α`. With `α = 0.95` (the default) the state's transient decay is
//! exponential at rate `α^t`, and no rollout can blow up — the
//! "no NaN or infinite states under normal inputs" test is therefore a
//! property of the *construction* rather than a hope.
//!
//! Determinism contract:
//! - All parameter initialisation uses `cjc_repro::Rng` sub-streams from
//!   [`CronosSeed::substream`], one per matrix (`"ssm.A"`, `"ssm.B"`,
//!   `"ssm.C"`, `"ssm.D"`). Same seed ⇒ same parameters bit-for-bit.
//! - All matrix-vector products and per-row L² norm reductions use
//!   `cjc_repro::KahanAccumulatorF64`.
//! - Box-Muller standard-normal draws consume exactly two uniforms per
//!   draw, matching the cjc-tempest sampler convention so cross-stream
//!   determinism debugging is uniform across the workspace.

use crate::error::CronosGanError;
use crate::seed::CronosSeed;
use crate::temporal_state::TemporalState;
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Configuration for a [`StateSpaceModel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateSpaceConfig {
    pub state_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Spectral-norm bound on the transition matrix. Must satisfy
    /// `0 < alpha < 1`. Default: 0.95.
    pub alpha: f64,
    /// Standard-deviation scale for the random-normal init of `B`, `C`,
    /// `D` matrices. Default: 0.1.
    pub init_scale: f64,
}

impl StateSpaceConfig {
    pub fn new(state_dim: usize, input_dim: usize, output_dim: usize) -> Self {
        Self {
            state_dim,
            input_dim,
            output_dim,
            alpha: 0.95,
            init_scale: 0.1,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_init_scale(mut self, init_scale: f64) -> Self {
        self.init_scale = init_scale;
        self
    }

    fn validate(&self) -> Result<(), CronosGanError> {
        if self.state_dim == 0 {
            return Err(CronosGanError::InvalidConfig {
                detail: "StateSpaceConfig.state_dim must be >= 1".to_string(),
            });
        }
        if self.input_dim == 0 || self.output_dim == 0 {
            return Err(CronosGanError::InvalidConfig {
                detail: "StateSpaceConfig.input_dim and output_dim must be >= 1".to_string(),
            });
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(CronosGanError::InvalidConfig {
                detail: format!(
                    "StateSpaceConfig.alpha must satisfy 0 < alpha < 1, got {}",
                    self.alpha
                ),
            });
        }
        if !self.init_scale.is_finite() || self.init_scale <= 0.0 {
            return Err(CronosGanError::InvalidConfig {
                detail: format!(
                    "StateSpaceConfig.init_scale must be > 0 and finite, got {}",
                    self.init_scale
                ),
            });
        }
        Ok(())
    }

    /// Canonical byte representation used by run-ID hashing. Includes
    /// every field; if any field changes the hash changes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&(self.state_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.output_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&self.alpha.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.init_scale.to_bits().to_le_bytes());
        bytes
    }
}

/// Parameters of a [`StateSpaceModel`]: row-major `Vec<f64>` matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSpaceParams {
    /// Transition matrix `A`, shape `[state_dim, state_dim]`.
    pub a: Vec<f64>,
    /// Input matrix `B`, shape `[state_dim, input_dim]`.
    pub b: Vec<f64>,
    /// Output matrix `C`, shape `[output_dim, state_dim]`.
    pub c: Vec<f64>,
    /// Direct-feedthrough matrix `D`, shape `[output_dim, input_dim]`.
    pub d: Vec<f64>,
}

/// Hidden state of an SSM: a single `state_dim`-long `f64` vector.
#[derive(Clone, Debug, PartialEq)]
pub struct StateSpaceState {
    pub x: Vec<f64>,
}

impl StateSpaceState {
    /// Construct a zero state of the requested dimension.
    pub fn zeros(state_dim: usize) -> Self {
        Self { x: vec![0.0; state_dim] }
    }
}

impl TemporalState for StateSpaceState {
    fn dim(&self) -> usize {
        self.x.len()
    }
    fn data(&self) -> &[f64] {
        &self.x
    }
}

/// Result of one forward step of an SSM.
#[derive(Clone, Debug)]
pub struct StateSpaceStepResult {
    pub prev_state: StateSpaceState,
    pub new_state: StateSpaceState,
    pub output: Vec<f64>,
}

/// Result of rolling an SSM forward across a full sequence.
#[derive(Clone, Debug)]
pub struct StateSpaceRolloutResult {
    pub states: Vec<StateSpaceState>,
    pub outputs: Vec<Vec<f64>>,
}

impl StateSpaceRolloutResult {
    pub fn n_steps(&self) -> usize {
        self.outputs.len()
    }

    pub fn final_state(&self) -> &StateSpaceState {
        self.states.last().expect("rollout always contains initial state")
    }
}

/// A deterministic linear state-space model.
#[derive(Clone, Debug)]
pub struct StateSpaceModel {
    config: StateSpaceConfig,
    params: StateSpaceParams,
}

impl StateSpaceModel {
    /// Initialise parameters deterministically from a seed.
    pub fn from_seed(config: StateSpaceConfig, seed: CronosSeed) -> Result<Self, CronosGanError> {
        config.validate()?;

        let mut rng_a = seed.substream("ssm.A");
        let mut rng_b = seed.substream("ssm.B");
        let mut rng_c = seed.substream("ssm.C");
        // D is initialised to zeros (no direct feedthrough) — common
        // choice that makes the SSM purely state-driven and matches the
        // determinism property "no hidden random calls" by literally
        // having no random draws here.
        let _ = seed.substream("ssm.D"); // reserved for future use

        // Build A as (alpha / sqrt(state_dim)) * (rows of R normalised to
        // unit L2 norm). Spectral norm <= alpha by construction.
        let a = build_stable_transition(config.state_dim, config.alpha, &mut rng_a);

        // B, C are i.i.d. normal scaled by `init_scale`.
        let b = sample_normal(config.state_dim * config.input_dim, config.init_scale, &mut rng_b);
        let c = sample_normal(config.output_dim * config.state_dim, config.init_scale, &mut rng_c);

        // D = 0 (no direct feedthrough).
        let d = vec![0.0; config.output_dim * config.input_dim];

        Ok(Self {
            config,
            params: StateSpaceParams { a, b, c, d },
        })
    }

    pub fn config(&self) -> &StateSpaceConfig {
        &self.config
    }

    pub fn params(&self) -> &StateSpaceParams {
        &self.params
    }

    /// Frobenius norm of the transition matrix `A`. Used by stability
    /// assertions in tests and audit logs.
    pub fn transition_frobenius_norm(&self) -> f64 {
        let mut acc = KahanAccumulatorF64::new();
        for &v in &self.params.a {
            acc.add(v * v);
        }
        acc.finalize().sqrt()
    }

    /// One forward step. `u` must be `input_dim`-long; `state` must be
    /// `state_dim`-long.
    pub fn step(
        &self,
        state: &StateSpaceState,
        u: &[f64],
    ) -> Result<StateSpaceStepResult, CronosGanError> {
        if state.x.len() != self.config.state_dim {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "StateSpaceModel::step state.dim={} but config.state_dim={}",
                    state.x.len(),
                    self.config.state_dim
                ),
            });
        }
        if u.len() != self.config.input_dim {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "StateSpaceModel::step u.len={} but config.input_dim={}",
                    u.len(),
                    self.config.input_dim
                ),
            });
        }
        for (i, &v) in u.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosGanError::NonFiniteInput {
                    detail: format!("StateSpaceModel::step u[{}] is non-finite", i),
                });
            }
        }

        // x_new = A x + B u
        let mut x_new = matvec_kahan(&self.params.a, &state.x, self.config.state_dim, self.config.state_dim);
        let bu = matvec_kahan(&self.params.b, u, self.config.state_dim, self.config.input_dim);
        for d in 0..self.config.state_dim {
            x_new[d] += bu[d];
        }

        // y = C x + D u
        let mut y = matvec_kahan(&self.params.c, &state.x, self.config.output_dim, self.config.state_dim);
        let du = matvec_kahan(&self.params.d, u, self.config.output_dim, self.config.input_dim);
        for d in 0..self.config.output_dim {
            y[d] += du[d];
        }

        Ok(StateSpaceStepResult {
            prev_state: state.clone(),
            new_state: StateSpaceState { x: x_new },
            output: y,
        })
    }

    /// Roll the model forward across an input sequence of length
    /// `n_steps`. `inputs` must be row-major with shape `[n_steps,
    /// input_dim]` — i.e. `inputs.len() == n_steps * input_dim`.
    ///
    /// Starts from `initial_state` (use [`StateSpaceState::zeros`] for
    /// the canonical zero start). The returned rollout has
    /// `states.len() == n_steps + 1` and `outputs.len() == n_steps`.
    pub fn rollout(
        &self,
        initial_state: &StateSpaceState,
        inputs: &[f64],
    ) -> Result<StateSpaceRolloutResult, CronosGanError> {
        let input_dim = self.config.input_dim;
        if inputs.len() % input_dim != 0 {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "StateSpaceModel::rollout inputs.len()={} not divisible by input_dim={}",
                    inputs.len(),
                    input_dim
                ),
            });
        }
        let n_steps = inputs.len() / input_dim;

        let mut states = Vec::with_capacity(n_steps + 1);
        let mut outputs = Vec::with_capacity(n_steps);
        states.push(initial_state.clone());
        let mut cur = initial_state.clone();
        for t in 0..n_steps {
            let u = &inputs[t * input_dim..(t + 1) * input_dim];
            let step = self.step(&cur, u)?;
            cur = step.new_state.clone();
            states.push(step.new_state);
            outputs.push(step.output);
        }
        Ok(StateSpaceRolloutResult { states, outputs })
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Build a stable transition matrix `A` of shape `[state_dim, state_dim]`
/// with `||A||_2 ≤ alpha`.
///
/// Method: draw each row from a standard normal, normalise the row to
/// unit L² length, scale the whole matrix by `alpha / √state_dim`. See
/// the module-level doc-comment for the spectral-norm proof.
fn build_stable_transition(state_dim: usize, alpha: f64, rng: &mut Rng) -> Vec<f64> {
    let mut a = vec![0.0_f64; state_dim * state_dim];
    let scale = alpha / (state_dim as f64).sqrt();
    for row in 0..state_dim {
        // Fill the row with N(0,1) draws.
        for col in 0..state_dim {
            a[row * state_dim + col] = standard_normal(rng);
        }
        // Row L² norm (Kahan-summed).
        let mut sq_acc = KahanAccumulatorF64::new();
        for col in 0..state_dim {
            let v = a[row * state_dim + col];
            sq_acc.add(v * v);
        }
        let norm = sq_acc.finalize().sqrt();
        let row_scale = if norm > 0.0 { scale / norm } else { scale };
        for col in 0..state_dim {
            a[row * state_dim + col] *= row_scale;
        }
    }
    a
}

/// Draw `n` i.i.d. `N(0, scale²)` values via Box-Muller.
fn sample_normal(n: usize, scale: f64, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| scale * standard_normal(rng)).collect()
}

/// Box-Muller standard-normal draw consuming two uniforms. Mirrors the
/// cjc-tempest helper exactly so cross-crate seed semantics are uniform.
fn standard_normal(rng: &mut Rng) -> f64 {
    let mut u1 = rng.next_f64();
    while u1 == 0.0 {
        u1 = rng.next_f64();
    }
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Matrix-vector product with Kahan-compensated dot products.
///
/// `m` is `[rows, cols]` row-major. `v` is `cols`-long. Returns a
/// `rows`-long `Vec<f64>`.
fn matvec_kahan(m: &[f64], v: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(m.len(), rows * cols);
    debug_assert_eq!(v.len(), cols);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut acc = KahanAccumulatorF64::new();
        for c in 0..cols {
            acc.add(m[r * cols + c] * v[c]);
        }
        out.push(acc.finalize());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_small() -> StateSpaceConfig {
        StateSpaceConfig::new(8, 4, 2)
    }

    #[test]
    fn invalid_config_state_dim_zero() {
        let cfg = StateSpaceConfig::new(0, 4, 2);
        let err = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }

    #[test]
    fn invalid_config_alpha_out_of_range() {
        let cfg = StateSpaceConfig::new(8, 4, 2).with_alpha(1.5);
        let err = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }

    #[test]
    fn from_seed_is_deterministic_across_runs() {
        let cfg = cfg_small();
        let m1 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
        let m2 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
        assert_eq!(m1.params(), m2.params(), "same seed must give identical params");
    }

    #[test]
    fn from_seed_diverges_on_different_seeds() {
        let cfg = cfg_small();
        let m1 = StateSpaceModel::from_seed(cfg, CronosSeed(1)).unwrap();
        let m2 = StateSpaceModel::from_seed(cfg, CronosSeed(2)).unwrap();
        assert_ne!(
            m1.params(),
            m2.params(),
            "different seeds must give different params (collision probability is 2^-64)"
        );
    }

    #[test]
    fn transition_matrix_is_structurally_stable() {
        // Construction: rows of R are unit L² norm, A = (alpha/sqrt(d))·R.
        // => each row of A has L² norm alpha/sqrt(d)
        // => ||A||_F = sqrt(d · (alpha/sqrt(d))²) = alpha *exactly*.
        // Since ||A||_2 ≤ ||A||_F for any matrix, this also proves
        // ||A||_2 ≤ alpha < 1 — the actual stability invariant.
        let cfg = StateSpaceConfig::new(8, 4, 2).with_alpha(0.9);
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(7)).unwrap();
        let fnorm = m.transition_frobenius_norm();
        assert!(
            (fnorm - cfg.alpha).abs() < 1e-12,
            "Frobenius norm {} differs from alpha {} by more than 1e-12",
            fnorm, cfg.alpha,
        );
    }

    #[test]
    fn step_output_and_state_have_correct_dims() {
        let cfg = cfg_small();
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
        let s = StateSpaceState::zeros(cfg.state_dim);
        let u = vec![0.1, 0.2, 0.3, 0.4];
        let res = m.step(&s, &u).unwrap();
        assert_eq!(res.new_state.dim(), cfg.state_dim);
        assert_eq!(res.output.len(), cfg.output_dim);
    }

    #[test]
    fn rollout_length_matches_input() {
        let cfg = cfg_small();
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
        let n_steps = 20;
        // input sequence of zeros works fine (B*0 + A*x progression only)
        let inputs = vec![0.05_f64; n_steps * cfg.input_dim];
        let s0 = StateSpaceState::zeros(cfg.state_dim);
        let r = m.rollout(&s0, &inputs).unwrap();
        assert_eq!(r.outputs.len(), n_steps);
        assert_eq!(r.states.len(), n_steps + 1);
    }

    #[test]
    fn rollout_states_stay_finite_under_bounded_inputs() {
        // With ||A||_2 <= alpha < 1 and bounded inputs, the state stays
        // bounded — verify empirically for a longish rollout.
        let cfg = StateSpaceConfig::new(8, 4, 2).with_alpha(0.95);
        let m = StateSpaceModel::from_seed(cfg, CronosSeed(11)).unwrap();
        let n_steps = 200;
        let inputs: Vec<f64> = (0..n_steps * cfg.input_dim)
            .map(|i| (i as f64 * 0.01).sin())
            .collect();
        let s0 = StateSpaceState::zeros(cfg.state_dim);
        let r = m.rollout(&s0, &inputs).unwrap();
        for (t, s) in r.states.iter().enumerate() {
            for (d, &v) in s.x.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "state[t={},d={}] = {} is non-finite",
                    t, d, v
                );
            }
        }
        for (t, y) in r.outputs.iter().enumerate() {
            for (d, &v) in y.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "output[t={},d={}] = {} is non-finite",
                    t, d, v
                );
            }
        }
    }

    #[test]
    fn rollout_byte_identical_across_runs() {
        // The headline reproducibility test.
        let cfg = cfg_small();
        let m1 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
        let m2 = StateSpaceModel::from_seed(cfg, CronosSeed(42)).unwrap();
        let inputs: Vec<f64> = (0..10 * cfg.input_dim).map(|i| (i as f64 * 0.1).cos()).collect();
        let s0 = StateSpaceState::zeros(cfg.state_dim);
        let r1 = m1.rollout(&s0, &inputs).unwrap();
        let r2 = m2.rollout(&s0, &inputs).unwrap();
        assert_eq!(r1.states, r2.states);
        for (a, b) in r1.outputs.iter().zip(r2.outputs.iter()) {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "rollout outputs must be byte-identical across runs"
            );
        }
    }
}
