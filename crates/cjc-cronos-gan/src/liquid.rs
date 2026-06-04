//! Liquid Neural Network — the **adaptive nonlinear local dynamics
//! adversary** of the Temporal GAN.
//!
//! Discrete liquid time-constant network (after Hasani et al. 2020,
//! simplified for determinism + auditability). The state update is:
//!
//! ```text
//! pre   = W_h · h_t + W_x · u_t + b
//! act   = tanh(pre)
//! s     = sigmoid(W_tau_u · u_t + W_tau_h · h_t + b_tau)        ∈ (0,1)
//! tau   = tau_min + (tau_max − tau_min) · s                      ∈ (tau_min, tau_max)
//! h_{t+1} = h_t + (dt / tau) ⊙ (-h_t + act)
//! y_t   = W_out · h_t + b_out
//! ```
//!
//! Where `⊙` is elementwise multiply. The range-scaled-sigmoid formulation
//! is mathematically equivalent to the softplus-then-clip variant from
//! Phase 0 — `tau` is bounded by construction — but is **smoothly
//! differentiable everywhere**, which is what the Phase 2 autodiff
//! adapter requires. The two `W_tau_u` / `W_tau_h` matrices are also
//! cleaner than the concatenated `W_tau` because cjc-ad's GradGraph has
//! no `concat` op; splitting the input-and-state contribution into two
//! separate matmuls lets the gradient flow through both without requiring
//! a graph extension.
//!
//! This is the deliberate counterpart to the SSM: where the SSM's
//! transition matrix is time-invariant and linear, the Liquid net's
//! effective time-constant depends on the current input *and* the
//! current state. When the input is stationary the gate stays slow
//! (long τ, memory-like); when the input is volatile the gate becomes
//! fast (short τ, reactive). The disagreement between SSM and Liquid
//! on a given timestep is then a meaningful "is this a stable regime
//! or a local spike" signal — exactly what the brief asks the
//! `TemporalDisagreement` score to surface in a later phase.
//!
//! Determinism contract:
//! - All parameter init uses [`CronosSeed`] sub-streams, one per matrix
//!   (`"liquid.W_h"`, `"liquid.W_x"`, `"liquid.bias"`, `"liquid.W_tau"`,
//!   `"liquid.bias_tau"`, `"liquid.W_out"`, `"liquid.bias_out"`).
//! - All matrix-vector products and softplus reductions use
//!   `cjc_repro::KahanAccumulatorF64`.
//! - `tau` is *clipped* into `[tau_min, tau_max]` so it never produces
//!   NaN or unbounded updates even under adversarial inputs — the
//!   "gates/time constants remain bounded" test is a property of the
//!   construction, not a hope.
//! - `softplus(x)` is implemented as `if x > 20 then x else ln(1 + e^x)`
//!   so it cannot overflow for any finite `f64` input.

use crate::error::CronosGanError;
use crate::seed::CronosSeed;
use crate::temporal_state::TemporalState;
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Configuration for a [`LiquidNetwork`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiquidConfig {
    pub state_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Discrete integration step. Default: 0.1.
    pub dt: f64,
    /// Lower clip bound on per-dim time constants. Must satisfy
    /// `0 < tau_min < tau_max`. Default: 0.1.
    pub tau_min: f64,
    /// Upper clip bound. Default: 8.0.
    pub tau_max: f64,
    /// Standard-deviation scale for random-normal initialisation of all
    /// weight matrices. Default: 0.1.
    pub init_scale: f64,
}

impl LiquidConfig {
    pub fn new(state_dim: usize, input_dim: usize, output_dim: usize) -> Self {
        Self {
            state_dim,
            input_dim,
            output_dim,
            dt: 0.1,
            tau_min: 0.1,
            tau_max: 8.0,
            init_scale: 0.1,
        }
    }

    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_tau_bounds(mut self, tau_min: f64, tau_max: f64) -> Self {
        self.tau_min = tau_min;
        self.tau_max = tau_max;
        self
    }

    pub fn with_init_scale(mut self, init_scale: f64) -> Self {
        self.init_scale = init_scale;
        self
    }

    fn validate(&self) -> Result<(), CronosGanError> {
        if self.state_dim == 0 || self.input_dim == 0 || self.output_dim == 0 {
            return Err(CronosGanError::InvalidConfig {
                detail: "LiquidConfig.state_dim/input_dim/output_dim must all be >= 1".to_string(),
            });
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(CronosGanError::InvalidConfig {
                detail: format!("LiquidConfig.dt must be > 0 and finite, got {}", self.dt),
            });
        }
        if !self.tau_min.is_finite() || !self.tau_max.is_finite() {
            return Err(CronosGanError::InvalidConfig {
                detail: "LiquidConfig.tau_min/tau_max must be finite".to_string(),
            });
        }
        if !(self.tau_min > 0.0 && self.tau_min < self.tau_max) {
            return Err(CronosGanError::InvalidConfig {
                detail: format!(
                    "LiquidConfig requires 0 < tau_min < tau_max; got tau_min={}, tau_max={}",
                    self.tau_min, self.tau_max
                ),
            });
        }
        if !self.init_scale.is_finite() || self.init_scale <= 0.0 {
            return Err(CronosGanError::InvalidConfig {
                detail: format!(
                    "LiquidConfig.init_scale must be > 0 and finite, got {}",
                    self.init_scale
                ),
            });
        }
        Ok(())
    }

    /// Canonical byte representation used by run-ID hashing.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(56);
        bytes.extend_from_slice(&(self.state_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.input_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.output_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&self.dt.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.tau_min.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.tau_max.to_bits().to_le_bytes());
        bytes.extend_from_slice(&self.init_scale.to_bits().to_le_bytes());
        bytes
    }
}

/// Parameters of a [`LiquidNetwork`].
#[derive(Clone, Debug, PartialEq)]
pub struct LiquidParams {
    /// Recurrent weights, shape `[state_dim, state_dim]`.
    pub w_h: Vec<f64>,
    /// Input weights, shape `[state_dim, input_dim]`.
    pub w_x: Vec<f64>,
    /// Pre-activation bias, shape `[state_dim]`.
    pub bias: Vec<f64>,
    /// Time-constant gating weights for the input `u`, shape
    /// `[state_dim, input_dim]`.
    pub w_tau_u: Vec<f64>,
    /// Time-constant gating weights for the state `h`, shape
    /// `[state_dim, state_dim]`.
    pub w_tau_h: Vec<f64>,
    /// Time-constant gating bias, shape `[state_dim]`.
    pub bias_tau: Vec<f64>,
    /// Output projection, shape `[output_dim, state_dim]`.
    pub w_out: Vec<f64>,
    /// Output bias, shape `[output_dim]`.
    pub bias_out: Vec<f64>,
}

/// Hidden state of a Liquid network.
#[derive(Clone, Debug, PartialEq)]
pub struct LiquidState {
    pub h: Vec<f64>,
}

impl LiquidState {
    pub fn zeros(state_dim: usize) -> Self {
        Self { h: vec![0.0; state_dim] }
    }
}

impl TemporalState for LiquidState {
    fn dim(&self) -> usize {
        self.h.len()
    }
    fn data(&self) -> &[f64] {
        &self.h
    }
}

/// Per-step effective time constant `tau ∈ [tau_min, tau_max]^state_dim`.
///
/// Exposed (not opaque) because one of the brief's stated goals is
/// inspectable hidden state — anomaly / regime-shift scoring will read
/// the per-step tau trajectory to detect when the network shifts
/// between "slow memory mode" and "fast reactive mode".
#[derive(Clone, Debug, PartialEq)]
pub struct LiquidTimeConstant {
    pub tau: Vec<f64>,
}

/// Per-step gating signal `dt / tau` — handy because it's what actually
/// scales the state update.
#[derive(Clone, Debug, PartialEq)]
pub struct LiquidGate {
    pub gate: Vec<f64>,
}

/// Result of one forward step of a Liquid network.
#[derive(Clone, Debug)]
pub struct LiquidStepResult {
    pub prev_state: LiquidState,
    pub new_state: LiquidState,
    pub output: Vec<f64>,
    pub time_constant: LiquidTimeConstant,
    pub gate: LiquidGate,
}

/// Result of rolling a Liquid network forward across a sequence.
#[derive(Clone, Debug)]
pub struct LiquidRolloutResult {
    pub states: Vec<LiquidState>,
    pub outputs: Vec<Vec<f64>>,
    pub time_constants: Vec<LiquidTimeConstant>,
    pub gates: Vec<LiquidGate>,
}

impl LiquidRolloutResult {
    pub fn n_steps(&self) -> usize {
        self.outputs.len()
    }

    pub fn final_state(&self) -> &LiquidState {
        self.states
            .last()
            .expect("rollout always contains initial state")
    }
}

/// A deterministic liquid time-constant network.
#[derive(Clone, Debug)]
pub struct LiquidNetwork {
    config: LiquidConfig,
    params: LiquidParams,
}

impl LiquidNetwork {
    /// Initialise parameters deterministically from a seed.
    pub fn from_seed(config: LiquidConfig, seed: CronosSeed) -> Result<Self, CronosGanError> {
        config.validate()?;

        let mut rng_wh = seed.substream("liquid.W_h");
        let mut rng_wx = seed.substream("liquid.W_x");
        let mut rng_bias = seed.substream("liquid.bias");
        let mut rng_wtau_u = seed.substream("liquid.W_tau_u");
        let mut rng_wtau_h = seed.substream("liquid.W_tau_h");
        let mut rng_btau = seed.substream("liquid.bias_tau");
        let mut rng_wout = seed.substream("liquid.W_out");
        let mut rng_bout = seed.substream("liquid.bias_out");

        let w_h = sample_normal(config.state_dim * config.state_dim, config.init_scale, &mut rng_wh);
        let w_x = sample_normal(config.state_dim * config.input_dim, config.init_scale, &mut rng_wx);
        let bias = sample_normal(config.state_dim, config.init_scale, &mut rng_bias);
        let w_tau_u = sample_normal(
            config.state_dim * config.input_dim,
            config.init_scale,
            &mut rng_wtau_u,
        );
        let w_tau_h = sample_normal(
            config.state_dim * config.state_dim,
            config.init_scale,
            &mut rng_wtau_h,
        );
        let bias_tau = sample_normal(config.state_dim, config.init_scale, &mut rng_btau);
        let w_out = sample_normal(config.output_dim * config.state_dim, config.init_scale, &mut rng_wout);
        let bias_out = sample_normal(config.output_dim, config.init_scale, &mut rng_bout);

        Ok(Self {
            config,
            params: LiquidParams {
                w_h,
                w_x,
                bias,
                w_tau_u,
                w_tau_h,
                bias_tau,
                w_out,
                bias_out,
            },
        })
    }

    pub fn config(&self) -> &LiquidConfig {
        &self.config
    }

    pub fn params(&self) -> &LiquidParams {
        &self.params
    }

    /// One forward step.
    pub fn step(
        &self,
        state: &LiquidState,
        u: &[f64],
    ) -> Result<LiquidStepResult, CronosGanError> {
        let cfg = &self.config;
        if state.h.len() != cfg.state_dim {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "LiquidNetwork::step state.dim={} but config.state_dim={}",
                    state.h.len(),
                    cfg.state_dim
                ),
            });
        }
        if u.len() != cfg.input_dim {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "LiquidNetwork::step u.len={} but config.input_dim={}",
                    u.len(),
                    cfg.input_dim
                ),
            });
        }
        for (i, &v) in u.iter().enumerate() {
            if !v.is_finite() {
                return Err(CronosGanError::NonFiniteInput {
                    detail: format!("LiquidNetwork::step u[{}] is non-finite", i),
                });
            }
        }

        // pre = W_h h + W_x u + bias
        let mut pre = matvec_kahan(&self.params.w_h, &state.h, cfg.state_dim, cfg.state_dim);
        let wx = matvec_kahan(&self.params.w_x, u, cfg.state_dim, cfg.input_dim);
        for d in 0..cfg.state_dim {
            pre[d] += wx[d] + self.params.bias[d];
        }

        // act = tanh(pre)
        let act: Vec<f64> = pre.iter().map(|v| v.tanh()).collect();

        // tau = tau_min + (tau_max − tau_min) · sigmoid(W_tau_u u + W_tau_h h + b_tau)
        // — bounded by construction, smoothly differentiable everywhere.
        let mut tau_pre = matvec_kahan(&self.params.w_tau_u, u, cfg.state_dim, cfg.input_dim);
        let tau_h = matvec_kahan(&self.params.w_tau_h, &state.h, cfg.state_dim, cfg.state_dim);
        for d in 0..cfg.state_dim {
            tau_pre[d] += tau_h[d] + self.params.bias_tau[d];
        }
        let tau_range = cfg.tau_max - cfg.tau_min;
        let tau: Vec<f64> = tau_pre
            .iter()
            .map(|v| cfg.tau_min + tau_range * sigmoid(*v))
            .collect();
        let gate: Vec<f64> = tau.iter().map(|t| cfg.dt / *t).collect();

        // h_new = h + gate ⊙ (-h + act)
        let mut h_new = Vec::with_capacity(cfg.state_dim);
        for d in 0..cfg.state_dim {
            let upd = gate[d] * (-state.h[d] + act[d]);
            h_new.push(state.h[d] + upd);
        }

        // y = W_out h + bias_out  (read off the *previous* state h to make
        // the output depend on the state the model conditions on, matching
        // the SSM's y_t = C x_t + D u_t convention)
        let mut y = matvec_kahan(&self.params.w_out, &state.h, cfg.output_dim, cfg.state_dim);
        for d in 0..cfg.output_dim {
            y[d] += self.params.bias_out[d];
        }

        Ok(LiquidStepResult {
            prev_state: state.clone(),
            new_state: LiquidState { h: h_new },
            output: y,
            time_constant: LiquidTimeConstant { tau },
            gate: LiquidGate { gate },
        })
    }

    /// Roll the network forward across an input sequence. Same shape
    /// conventions as [`crate::ssm::StateSpaceModel::rollout`]:
    /// `inputs` is row-major `[n_steps, input_dim]`.
    pub fn rollout(
        &self,
        initial_state: &LiquidState,
        inputs: &[f64],
    ) -> Result<LiquidRolloutResult, CronosGanError> {
        let input_dim = self.config.input_dim;
        if inputs.len() % input_dim != 0 {
            return Err(CronosGanError::DimensionMismatch {
                detail: format!(
                    "LiquidNetwork::rollout inputs.len()={} not divisible by input_dim={}",
                    inputs.len(),
                    input_dim
                ),
            });
        }
        let n_steps = inputs.len() / input_dim;
        let mut states = Vec::with_capacity(n_steps + 1);
        let mut outputs = Vec::with_capacity(n_steps);
        let mut taus = Vec::with_capacity(n_steps);
        let mut gates = Vec::with_capacity(n_steps);
        states.push(initial_state.clone());
        let mut cur = initial_state.clone();
        for t in 0..n_steps {
            let u = &inputs[t * input_dim..(t + 1) * input_dim];
            let step = self.step(&cur, u)?;
            cur = step.new_state.clone();
            states.push(step.new_state);
            outputs.push(step.output);
            taus.push(step.time_constant);
            gates.push(step.gate);
        }
        Ok(LiquidRolloutResult {
            states,
            outputs,
            time_constants: taus,
            gates,
        })
    }
}

/// Internal setter used by the Phase 2 [`crate::training::Trainable`]
/// adapter. NOT public — same rationale as `crate::ssm::set_params_internal`.
pub(crate) fn set_params_internal(model: &mut LiquidNetwork, new_params: LiquidParams) {
    model.params = new_params;
}

// ── Internal helpers ─────────────────────────────────────────────────────

/// Overflow-safe sigmoid. `1 / (1 + e^{-x})` directly overflows for
/// `x < -709` and the equivalent `e^x / (1+e^x)` overflows for `x > 709`.
/// Branching at ±20 keeps every finite f64 input producing a finite
/// output in (0, 1), which is what the `tau ∈ (tau_min, tau_max)`
/// invariant depends on.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Draw `n` i.i.d. `N(0, scale²)` values via Box-Muller. Mirrors
/// [`crate::ssm`] exactly so cross-network determinism is uniform.
fn sample_normal(n: usize, scale: f64, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| scale * standard_normal(rng)).collect()
}

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

    fn cfg_small() -> LiquidConfig {
        LiquidConfig::new(8, 4, 2)
    }

    #[test]
    fn invalid_config_zero_dim() {
        let cfg = LiquidConfig::new(0, 4, 2);
        let err = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }

    #[test]
    fn invalid_config_tau_min_ge_tau_max() {
        let cfg = LiquidConfig::new(8, 4, 2).with_tau_bounds(2.0, 1.0);
        let err = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap_err();
        assert!(matches!(err, CronosGanError::InvalidConfig { .. }));
    }

    #[test]
    fn from_seed_is_deterministic_across_runs() {
        let cfg = cfg_small();
        let n1 = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
        let n2 = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
        assert_eq!(n1.params(), n2.params());
    }

    #[test]
    fn from_seed_diverges_on_different_seeds() {
        let cfg = cfg_small();
        let n1 = LiquidNetwork::from_seed(cfg, CronosSeed(1)).unwrap();
        let n2 = LiquidNetwork::from_seed(cfg, CronosSeed(2)).unwrap();
        assert_ne!(n1.params(), n2.params());
    }

    #[test]
    fn step_dims_match_config_and_tau_is_bounded() {
        let cfg = cfg_small();
        let net = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
        let s = LiquidState::zeros(cfg.state_dim);
        let u = vec![0.1, 0.2, 0.3, 0.4];
        let res = net.step(&s, &u).unwrap();
        assert_eq!(res.new_state.dim(), cfg.state_dim);
        assert_eq!(res.output.len(), cfg.output_dim);
        assert_eq!(res.time_constant.tau.len(), cfg.state_dim);
        assert_eq!(res.gate.gate.len(), cfg.state_dim);
        for &t in &res.time_constant.tau {
            assert!(
                t >= cfg.tau_min && t <= cfg.tau_max,
                "tau {} outside [{}, {}]",
                t,
                cfg.tau_min,
                cfg.tau_max
            );
        }
    }

    #[test]
    fn tau_stays_bounded_under_extreme_inputs() {
        // Adversarial inputs: huge magnitudes. softplus + clip must
        // keep tau in [tau_min, tau_max] regardless.
        let cfg = cfg_small().with_tau_bounds(0.05, 5.0);
        let net = LiquidNetwork::from_seed(cfg, CronosSeed(13)).unwrap();
        let s = LiquidState::zeros(cfg.state_dim);
        // Mix of very large positive and very negative magnitudes.
        let u = vec![1e6, -1e6, 1e3, -1e3];
        let res = net.step(&s, &u).unwrap();
        for &t in &res.time_constant.tau {
            assert!(
                t >= cfg.tau_min && t <= cfg.tau_max,
                "tau {} outside [{}, {}] under extreme inputs",
                t,
                cfg.tau_min,
                cfg.tau_max
            );
            assert!(t.is_finite(), "tau is non-finite: {}", t);
        }
        for &v in &res.new_state.h {
            assert!(v.is_finite(), "state went non-finite under extreme input");
        }
    }

    #[test]
    fn rollout_length_matches_input() {
        let cfg = cfg_small();
        let net = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
        let n_steps = 25;
        let inputs = vec![0.05_f64; n_steps * cfg.input_dim];
        let s0 = LiquidState::zeros(cfg.state_dim);
        let r = net.rollout(&s0, &inputs).unwrap();
        assert_eq!(r.outputs.len(), n_steps);
        assert_eq!(r.states.len(), n_steps + 1);
        assert_eq!(r.time_constants.len(), n_steps);
        assert_eq!(r.gates.len(), n_steps);
    }

    #[test]
    fn rollout_byte_identical_across_runs() {
        let cfg = cfg_small();
        let n1 = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
        let n2 = LiquidNetwork::from_seed(cfg, CronosSeed(42)).unwrap();
        let inputs: Vec<f64> = (0..15 * cfg.input_dim).map(|i| (i as f64 * 0.07).sin()).collect();
        let s0 = LiquidState::zeros(cfg.state_dim);
        let r1 = n1.rollout(&s0, &inputs).unwrap();
        let r2 = n2.rollout(&s0, &inputs).unwrap();
        assert_eq!(r1.states, r2.states);
        for (a, b) in r1.outputs.iter().zip(r2.outputs.iter()) {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            );
        }
        for (a, b) in r1.time_constants.iter().zip(r2.time_constants.iter()) {
            assert_eq!(a, b);
        }
    }
}
