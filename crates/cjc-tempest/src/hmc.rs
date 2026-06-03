//! Hamiltonian Monte Carlo with leapfrog integration and identity mass matrix.
//!
//! This is the **Session 2** sampler. NUTS (Session 3) and Vehtari R-hat / ESS
//! diagnostics (Session 4) follow. v0.1 ships HMC with caller-supplied
//! gradient — the convenience constructor that auto-derives the gradient via
//! `cjc_ad::GradGraph` is deferred until cjc-ad's A1 determinism audit has
//! merged to master (see Phase 2 handoff §1.1).
//!
//! ## Algorithm
//!
//! For each chain c and each iteration:
//!
//! 1. Draw momentum `p ~ N(0, I)` — one Box-Muller pair per dimension.
//! 2. Compute initial Hamiltonian `H = -log_p(q) + 0.5 · |p|²` (potential +
//!    kinetic), with the kinetic sum Kahan-compensated.
//! 3. Run `L` leapfrog steps:
//!    - `p ← p + (ε/2) · ∇log_p(q)` (half momentum kick)
//!    - `q ← q + ε · p` (position drift)
//!    - `p ← p + (ε/2) · ∇log_p(q_new)` (half momentum kick)
//! 4. Compute final Hamiltonian `H_new`.
//! 5. If `|H_new - H_initial| > divergence_threshold` ⇒ divergent transition,
//!    reject (and consume the acceptance uniform anyway — determinism).
//! 6. Otherwise Metropolis-accept on `exp(-(H_new - H_initial))`.
//!
//! Warmup iterations use the same kernel — no mass-matrix adaptation in v0.1.
//! Step-size adaptation (dual averaging) ships with NUTS (Session 3).
//!
//! ## Determinism contract
//!
//! - **All randomness** routes through `cjc_repro::Rng` (SplitMix64) seeded
//!   from `base_seed` via a per-chain stretch (the same `PER_CHAIN_MIX`
//!   constant as Metropolis — identical seeding behaviour across samplers).
//! - **Kinetic energy** sum is Kahan-compensated. Hamiltonian energy is
//!   reconstructed every step rather than being incrementally updated, so
//!   round-off doesn't compound over the trajectory.
//! - **Acceptance coin flip ALWAYS consumes one uniform** regardless of
//!   accept / reject / divergent. Identical invariant to
//!   [`crate::metropolis::mh_accept_reject`]; without it, divergent
//!   transitions would shift the RNG stream relative to non-divergent runs
//!   and break content-hash reproducibility.
//! - **Same `(log_posterior, log_posterior_grad, initial_state, ε, L,
//!   n_warmup, n_iter, n_chains, base_seed)` ⇒ byte-identical
//!   `PosteriorSamples.content_hash`.**

use crate::error::TempestError;
use crate::posterior::{ConvergenceDiagnostics, PosteriorSamples};
use crate::sampler::AcceptResult;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Stable string label for content-hashing.
pub const SAMPLER_LABEL: &str = "hamiltonian_monte_carlo";

/// Default energy-error threshold for divergence detection.
/// Stan and PyMC use 1000 by default; we match.
pub const DEFAULT_DIVERGENCE_THRESHOLD: f64 = 1000.0;

/// Per-chain seed-stretch constant — IDENTICAL to
/// [`crate::metropolis`]'s constant, so seed-sharing semantics are
/// consistent across samplers.
const PER_CHAIN_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Hamiltonian Monte Carlo sampler.
///
/// Holds the caller-fixed leapfrog hyper-parameters. **Stateless across
/// chains** — every chain derives a fresh RNG from `base_seed`.
#[derive(Clone, Debug)]
pub struct HamiltonianMonteCarlo {
    epsilon: f64,
    trajectory_length: usize,
    divergence_threshold: f64,
}

impl HamiltonianMonteCarlo {
    /// Construct with explicit leapfrog `epsilon` and `trajectory_length`
    /// (number of leapfrog steps `L` per HMC iteration). v0.1 has no
    /// step-size adaptation — both knobs are fixed.
    pub fn new(epsilon: f64, trajectory_length: usize) -> Self {
        Self {
            epsilon,
            trajectory_length,
            divergence_threshold: DEFAULT_DIVERGENCE_THRESHOLD,
        }
    }

    /// Override the divergence threshold. Energy errors exceeding this value
    /// during a trajectory mark that iteration as a divergent transition
    /// (incrementing [`ConvergenceDiagnostics::divergences`]) and force a
    /// rejection.
    pub fn with_divergence_threshold(mut self, t: f64) -> Self {
        self.divergence_threshold = t;
        self
    }

    /// Run `n_chains` independent chains, each with `n_warmup + n_iter`
    /// HMC iterations.
    ///
    /// # Arguments
    ///
    /// - `log_posterior` — log-density closure `Fn(&[f64]) -> f64`.
    /// - `log_posterior_grad` — gradient closure `Fn(&[f64]) -> Vec<f64>`
    ///   returning `∇ log π(θ|y)`. Caller's responsibility to provide an
    ///   analytic gradient or an autodiff-derived one. A future convenience
    ///   constructor will wire this through `cjc_ad::GradGraph` directly.
    /// - `initial_state` — starting point for every chain. Must be all
    ///   finite.
    /// - `n_chains` — number of independent chains. Must be `≥ 1`.
    /// - `n_warmup` — warmup iterations per chain. Must be `≥ 10`.
    /// - `n_iter` — post-warmup samples per chain. Must be `≥ 1`.
    /// - `base_seed` — base SplitMix64 seed.
    pub fn run<F, G>(
        &self,
        log_posterior: F,
        log_posterior_grad: G,
        initial_state: &[f64],
        n_chains: usize,
        n_warmup: usize,
        n_iter: usize,
        base_seed: u64,
    ) -> Result<PosteriorSamples, TempestError>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        // ── Config validation ────────────────────────────────────────────
        if n_chains == 0 {
            return Err(TempestError::Unsupported {
                detail: "n_chains must be >= 1".to_string(),
            });
        }
        if n_iter == 0 {
            return Err(TempestError::Unsupported {
                detail: "n_iter must be >= 1".to_string(),
            });
        }
        if n_warmup < 10 {
            return Err(TempestError::Unsupported {
                detail: format!("n_warmup must be >= 10 (got {})", n_warmup),
            });
        }
        if self.trajectory_length == 0 {
            return Err(TempestError::Unsupported {
                detail: "trajectory_length must be >= 1".to_string(),
            });
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(TempestError::Unsupported {
                detail: format!(
                    "epsilon must be > 0 and finite, got {}",
                    self.epsilon
                ),
            });
        }
        if initial_state.is_empty() {
            return Err(TempestError::InvalidInitialState {
                detail: "initial_state is empty".to_string(),
            });
        }
        for (i, v) in initial_state.iter().enumerate() {
            if !v.is_finite() {
                return Err(TempestError::InvalidInitialState {
                    detail: format!("initial_state[{}] = {} is non-finite", i, v),
                });
            }
        }
        let n_dim = initial_state.len();

        // Probe log-posterior and gradient at initial state.
        let initial_lp = log_posterior(initial_state);
        if initial_lp.is_nan() {
            return Err(TempestError::InvalidLogPosterior {
                detail: "log_posterior returned NaN at initial_state".to_string(),
            });
        }
        let initial_grad = log_posterior_grad(initial_state);
        if initial_grad.len() != n_dim {
            return Err(TempestError::InvalidLogPosterior {
                detail: format!(
                    "log_posterior_grad returned {} dims, expected {}",
                    initial_grad.len(),
                    n_dim
                ),
            });
        }

        // ── Sampling loop ────────────────────────────────────────────────
        let mut chains: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_chains);
        let mut total_divergences: u64 = 0;

        for chain_idx in 0..n_chains {
            let chain_seed = base_seed
                .wrapping_add((chain_idx as u64).wrapping_mul(PER_CHAIN_MIX));
            let mut rng = Rng::seeded(chain_seed);

            let mut state: Vec<f64> = initial_state.to_vec();
            let mut state_grad: Vec<f64> = initial_grad.clone();
            let mut state_lp = initial_lp;
            let mut samples = Vec::with_capacity(n_iter);

            for iter in 0..(n_warmup + n_iter) {
                // Draw momentum p ~ N(0, I), one Box-Muller pair per dim.
                let momentum: Vec<f64> = (0..n_dim).map(|_| standard_normal(&mut rng)).collect();

                // Initial Hamiltonian.
                let h_initial = hamiltonian(state_lp, &momentum);
                if !h_initial.is_finite() {
                    // Non-finite initial energy is a model issue, not a
                    // sampler one. We still consume the acceptance uniform
                    // to preserve the RNG stream invariant.
                    let _ = rng.next_f64();
                    if iter >= n_warmup {
                        samples.push(state.clone());
                    }
                    total_divergences += 1;
                    continue;
                }

                // Run L leapfrog steps. We mutate copies and only commit if
                // accepted.
                let (proposed_state, proposed_grad, proposed_p, leapfrog_ok) = leapfrog(
                    &state,
                    &state_grad,
                    &momentum,
                    self.epsilon,
                    self.trajectory_length,
                    &log_posterior_grad,
                );

                let outcome = if !leapfrog_ok {
                    // Leapfrog produced non-finite values somewhere in the
                    // trajectory. Always consume the uniform; record
                    // divergence.
                    let _ = rng.next_f64();
                    AcceptResult::Divergent
                } else {
                    let proposed_lp = log_posterior(&proposed_state);
                    let h_new = hamiltonian(proposed_lp, &proposed_p);
                    let energy_delta = h_new - h_initial;
                    hmc_accept_reject(
                        proposed_lp,
                        energy_delta,
                        self.divergence_threshold,
                        &mut rng,
                    )
                };

                match outcome {
                    AcceptResult::Accept { new_log_posterior } => {
                        state = proposed_state;
                        state_grad = proposed_grad;
                        state_lp = new_log_posterior;
                    }
                    AcceptResult::Reject => {
                        // Leave state unchanged.
                    }
                    AcceptResult::Divergent => {
                        total_divergences += 1;
                        // Leave state unchanged.
                    }
                }

                if iter >= n_warmup {
                    samples.push(state.clone());
                }
            }
            chains.push(samples);
        }

        // Build diagnostics. v0.1 ships divergence counter only; R-hat / ESS
        // populate in Session 4 once Vehtari diagnostics module lands.
        let diagnostics = ConvergenceDiagnostics {
            r_hat: vec![f64::NAN; n_dim],
            ess_bulk: vec![f64::NAN; n_dim],
            ess_tail: vec![f64::NAN; n_dim],
            divergences: total_divergences,
            n_max_treedepth: 0,
        };

        let content_hash = compute_content_hash(
            SAMPLER_LABEL,
            n_chains,
            n_iter,
            n_dim,
            base_seed,
            self.epsilon,
            self.trajectory_length,
            &chains,
        );

        Ok(PosteriorSamples {
            chains,
            n_chains,
            n_samples_per_chain: n_iter,
            n_dim,
            diagnostics,
            content_hash,
        })
    }
}

/// Run `L` leapfrog steps starting from `(q, p)` with current gradient
/// `grad_q`. Returns the final state, final gradient, final momentum, and
/// a flag indicating whether all intermediate states were finite.
///
/// The gradient closure is called exactly `L` times — once per full step
/// (the half-step kicks reuse the gradient cached from the previous step).
fn leapfrog<G>(
    q: &[f64],
    grad_q: &[f64],
    p: &[f64],
    epsilon: f64,
    l: usize,
    log_posterior_grad: &G,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool)
where
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n_dim = q.len();
    let mut q = q.to_vec();
    let mut p = p.to_vec();
    let mut grad_q = grad_q.to_vec();
    let half_eps = 0.5 * epsilon;

    for _ in 0..l {
        // Half momentum kick.
        for d in 0..n_dim {
            p[d] += half_eps * grad_q[d];
        }
        // Full position drift.
        for d in 0..n_dim {
            q[d] += epsilon * p[d];
        }
        if q.iter().any(|v| !v.is_finite()) {
            return (q, grad_q, p, false);
        }
        // Gradient at the new position.
        grad_q = log_posterior_grad(&q);
        if grad_q.len() != n_dim || grad_q.iter().any(|v| !v.is_finite()) {
            return (q, grad_q, p, false);
        }
        // Half momentum kick.
        for d in 0..n_dim {
            p[d] += half_eps * grad_q[d];
        }
        if p.iter().any(|v| !v.is_finite()) {
            return (q, grad_q, p, false);
        }
    }
    (q, grad_q, p, true)
}

/// Hamiltonian `H = -log_p + 0.5 · Σ p_d²` with Kahan-compensated kinetic
/// sum. Returning a non-finite `H` is propagated up as a divergence.
fn hamiltonian(log_p: f64, momentum: &[f64]) -> f64 {
    let mut kinetic_acc = KahanAccumulatorF64::new();
    for &m in momentum {
        kinetic_acc.add(0.5 * m * m);
    }
    -log_p + kinetic_acc.finalize()
}

/// HMC accept/reject decision. ALWAYS consumes one uniform regardless of
/// outcome to preserve the RNG-stream invariant across divergent and
/// non-divergent iterations.
///
/// Returns [`AcceptResult::Divergent`] when `|energy_delta| >
/// divergence_threshold` OR when `proposed_lp` is non-finite (treated as
/// model breakdown).
pub(crate) fn hmc_accept_reject(
    proposed_lp: f64,
    energy_delta: f64,
    divergence_threshold: f64,
    rng: &mut Rng,
) -> AcceptResult {
    // ALWAYS consume the uniform, then decide.
    let u = rng.next_f64();
    if !proposed_lp.is_finite() || !energy_delta.is_finite() {
        return AcceptResult::Divergent;
    }
    if energy_delta.abs() > divergence_threshold {
        return AcceptResult::Divergent;
    }
    // Metropolis criterion on exp(-energy_delta).
    if energy_delta <= 0.0 || u.ln() < -energy_delta {
        AcceptResult::Accept { new_log_posterior: proposed_lp }
    } else {
        AcceptResult::Reject
    }
}

/// Box-Muller standard-normal draw consuming two uniforms.
///
/// Mirrors [`crate::metropolis`]'s helper exactly so cross-sampler
/// determinism is preserved (same seed + same dim ⇒ same momentum draws).
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

/// Compute the content-addressed hash of an HMC posterior. The hash is
/// salted with `epsilon` and `trajectory_length` so two posteriors with
/// identical chains but different leapfrog hyper-parameters get distinct
/// IDs.
#[allow(clippy::too_many_arguments)]
fn compute_content_hash(
    sampler_label: &str,
    n_chains: usize,
    n_iter: usize,
    n_dim: usize,
    base_seed: u64,
    epsilon: f64,
    trajectory_length: usize,
    chains: &[Vec<Vec<f64>>],
) -> FingerprintId {
    let mut parts: Vec<FingerprintId> = Vec::with_capacity(7 + n_chains);
    parts.push(fingerprint_str(IdDomain::CausalClaim, sampler_label));
    parts.push(fingerprint(IdDomain::CausalClaim, &(n_chains as u64).to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &(n_iter as u64).to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &(n_dim as u64).to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &base_seed.to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &epsilon.to_bits().to_le_bytes()));
    parts.push(fingerprint(
        IdDomain::CausalClaim,
        &(trajectory_length as u64).to_le_bytes(),
    ));
    for chain in chains {
        let mut chain_bytes: Vec<u8> = Vec::with_capacity(chain.len() * n_dim * 8);
        for sample in chain {
            for v in sample {
                chain_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        parts.push(fingerprint(IdDomain::CausalClaim, &chain_bytes));
    }
    fingerprint_compose(IdDomain::CausalClaim, "hmc_posterior", &parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// 1D unit Gaussian: log π(x) = -0.5 x² - 0.5 log(2π).
    fn log_p_1d(x: &[f64]) -> f64 {
        -0.5 * x[0] * x[0] - 0.5 * std::f64::consts::TAU.ln()
    }
    fn grad_log_p_1d(x: &[f64]) -> Vec<f64> {
        vec![-x[0]]
    }

    /// 2D independent unit Gaussian.
    fn log_p_2d(x: &[f64]) -> f64 {
        -0.5 * (x[0] * x[0] + x[1] * x[1]) - std::f64::consts::TAU.ln()
    }
    fn grad_log_p_2d(x: &[f64]) -> Vec<f64> {
        vec![-x[0], -x[1]]
    }

    // ── § Validation ─────────────────────────────────────────────────────

    #[test]
    fn empty_initial_state_returns_invalid() {
        let hmc = HamiltonianMonteCarlo::new(0.1, 20);
        let err = hmc
            .run(log_p_1d, grad_log_p_1d, &[], 1, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::InvalidInitialState { .. }));
    }

    #[test]
    fn n_chains_zero_returns_unsupported() {
        let hmc = HamiltonianMonteCarlo::new(0.1, 20);
        let err = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 0, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::Unsupported { .. }));
    }

    #[test]
    fn trajectory_length_zero_returns_unsupported() {
        let hmc = HamiltonianMonteCarlo::new(0.1, 0);
        let err = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::Unsupported { .. }));
    }

    #[test]
    fn invalid_epsilon_returns_unsupported() {
        let hmc = HamiltonianMonteCarlo::new(-0.1, 20);
        let err = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::Unsupported { .. }));
    }

    #[test]
    fn grad_wrong_dim_returns_invalid_log_posterior() {
        let hmc = HamiltonianMonteCarlo::new(0.1, 20);
        let err = hmc
            .run(
                log_p_2d,
                |_: &[f64]| vec![0.0], // returns 1 dim for a 2D problem
                &[0.0, 0.0],
                1,
                100,
                100,
                42,
            )
            .unwrap_err();
        assert!(matches!(err, TempestError::InvalidLogPosterior { .. }));
    }

    // ── § Determinism (the headline contract) ────────────────────────────

    /// The HEADLINE byte-identity test. Same model + seed + config ⇒
    /// byte-identical chain bit-patterns AND byte-identical content hash.
    #[test]
    fn same_seed_byte_identical_posterior() {
        let hmc = HamiltonianMonteCarlo::new(0.15, 20);
        let r1 = hmc
            .run(log_p_2d, grad_log_p_2d, &[0.0, 0.0], 2, 50, 100, 42)
            .unwrap();
        let r2 = hmc
            .run(log_p_2d, grad_log_p_2d, &[0.0, 0.0], 2, 50, 100, 42)
            .unwrap();
        assert_eq!(r1.content_hash, r2.content_hash);
        for c in 0..2 {
            for s in 0..100 {
                for d in 0..2 {
                    assert_eq!(
                        r1.chains[c][s][d].to_bits(),
                        r2.chains[c][s][d].to_bits(),
                        "chain {} sample {} dim {} differs",
                        c, s, d,
                    );
                }
            }
        }
    }

    #[test]
    fn different_seed_produces_different_content_hash() {
        let hmc = HamiltonianMonteCarlo::new(0.15, 20);
        let r1 = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 50, 100, 1)
            .unwrap();
        let r2 = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 50, 100, 2)
            .unwrap();
        assert_ne!(r1.content_hash, r2.content_hash);
    }

    #[test]
    fn different_epsilon_produces_different_content_hash() {
        // Hyper-parameter salt in content_hash: ε difference ⇒ distinct ID,
        // even though chains may happen to look similar.
        let r1 = HamiltonianMonteCarlo::new(0.15, 20)
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 50, 100, 42)
            .unwrap();
        let r2 = HamiltonianMonteCarlo::new(0.16, 20)
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 50, 100, 42)
            .unwrap();
        assert_ne!(r1.content_hash, r2.content_hash);
    }

    #[test]
    fn hmc_accept_reject_consumes_one_uniform_regardless() {
        // Critical determinism invariant — divergent, reject, and accept must
        // all consume exactly one uniform. After each call, the next u64 from
        // each rng must match — meaning all three advanced the same amount.
        let mut rng_accept = Rng::seeded(0);
        let mut rng_reject = Rng::seeded(0);
        let mut rng_divergent = Rng::seeded(0);
        let _ = hmc_accept_reject(1.0, -100.0, 1000.0, &mut rng_accept); // accept
        let _ = hmc_accept_reject(1.0, 100.0, 1000.0, &mut rng_reject); // reject (high ΔH within threshold)
        let _ = hmc_accept_reject(1.0, 5000.0, 1000.0, &mut rng_divergent); // divergent
        let next_accept = rng_accept.next_u64();
        let next_reject = rng_reject.next_u64();
        let next_divergent = rng_divergent.next_u64();
        assert_eq!(next_accept, next_reject, "reject path consumed different uniforms than accept");
        assert_eq!(next_accept, next_divergent, "divergent path consumed different uniforms than accept");
    }

    // ── § Sampler correctness ────────────────────────────────────────────

    /// 1D unit Gaussian recovery: with reasonable ε, L, and 2000 samples,
    /// the sample mean and variance match the truth within generous bounds.
    #[test]
    fn samples_approximate_unit_gaussian_mean_and_variance() {
        let hmc = HamiltonianMonteCarlo::new(0.2, 20);
        let r = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 200, 2000, 42)
            .unwrap();
        let mut mean_acc = KahanAccumulatorF64::new();
        for s in &r.chains[0] {
            mean_acc.add(s[0]);
        }
        let m = mean_acc.finalize() / 2000.0;
        assert!(m.abs() < 0.2, "sample mean was {}", m);
        let mut var_acc = KahanAccumulatorF64::new();
        for s in &r.chains[0] {
            let d = s[0] - m;
            var_acc.add(d * d);
        }
        let v = var_acc.finalize() / 1999.0;
        assert!(
            (v - 1.0).abs() < 0.3,
            "sample variance was {} (expected ≈ 1.0)",
            v,
        );
    }

    /// Degenerate `ε` on a steep posterior should produce divergences. We
    /// use a tightly-curved Gaussian (precision = 100) so even a tiny step
    /// produces energy-error explosions.
    #[test]
    fn divergence_detection_fires_on_degenerate_epsilon() {
        // Massive ε on a steep posterior: 100 × 100 = 10⁴ contribution to
        // Hamiltonian after a single step beyond x = ±1.
        fn log_p_steep(x: &[f64]) -> f64 {
            -50.0 * x[0] * x[0]
        }
        fn grad_log_p_steep(x: &[f64]) -> Vec<f64> {
            vec![-100.0 * x[0]]
        }
        let hmc = HamiltonianMonteCarlo::new(0.5, 50)
            .with_divergence_threshold(10.0);
        let r = hmc
            .run(log_p_steep, grad_log_p_steep, &[1.0], 1, 10, 100, 42)
            .unwrap();
        assert!(
            r.diagnostics.divergences > 0,
            "expected some divergences, got {}",
            r.diagnostics.divergences
        );
    }

    #[test]
    fn chain_shape_matches_config() {
        let hmc = HamiltonianMonteCarlo::new(0.1, 10);
        let r = hmc
            .run(log_p_2d, grad_log_p_2d, &[0.0, 0.0], 3, 50, 200, 99)
            .unwrap();
        assert_eq!(r.chains.len(), 3);
        for chain in &r.chains {
            assert_eq!(chain.len(), 200);
            for sample in chain {
                assert_eq!(sample.len(), 2);
            }
        }
    }

    #[test]
    fn samples_have_finite_values() {
        let hmc = HamiltonianMonteCarlo::new(0.1, 20);
        let r = hmc
            .run(log_p_1d, grad_log_p_1d, &[0.0], 1, 100, 500, 7)
            .unwrap();
        for sample in &r.chains[0] {
            for &v in sample {
                assert!(v.is_finite(), "sample contains non-finite value: {}", v);
            }
        }
    }

    /// Gradient-closure call count: `L × (n_warmup + n_iter)` per chain
    /// (plus one initial probe). Verifying this catches accidental extra
    /// gradient evaluations that would shift the cost model HMC users
    /// reason about.
    #[test]
    fn gradient_closure_called_expected_count() {
        let calls = Cell::new(0usize);
        let l = 15usize;
        let n_warmup = 20usize;
        let n_iter = 30usize;
        let n_chains = 2usize;
        let grad = |x: &[f64]| {
            calls.set(calls.get() + 1);
            vec![-x[0]]
        };
        let hmc = HamiltonianMonteCarlo::new(0.1, l);
        let _ = hmc
            .run(log_p_1d, grad, &[0.0], n_chains, n_warmup, n_iter, 42)
            .unwrap();
        // One initial probe + L per iteration per chain.
        let expected = 1 + n_chains * l * (n_warmup + n_iter);
        assert_eq!(
            calls.get(),
            expected,
            "gradient was called {} times, expected {}",
            calls.get(),
            expected,
        );
    }
}
