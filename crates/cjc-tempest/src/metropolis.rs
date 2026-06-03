//! Metropolis-Hastings sampler with symmetric Gaussian proposal + Welford-
//! adaptive diagonal covariance during warmup.
//!
//! This is the **determinism warmup** sampler for the cjc-tempest v0.1
//! surface. HMC and NUTS follow in later sessions; MH alone exercises the
//! seed-flow audit (initial-state RNG, proposal RNG, acceptance coin) on a
//! simple-enough sampler that any byte-identity failure must be in the
//! framework rather than the sampler-specific math.
//!
//! ## Algorithm
//!
//! For each chain c:
//!
//! 1. Seed a per-chain RNG via SplitMix64 stretch of `base_seed`.
//! 2. Initialise per-chain state from `initial_state.clone()`.
//! 3. **Warmup phase** (`n_warmup` iterations):
//!    - Symmetric Gaussian proposal `x_new = x + N(0, σ_init² · I)`.
//!    - Welford accumulator tracks running mean + variance of every
//!      **accepted** state.
//!    - Half-way through warmup, switch from `σ_init` to the Welford-
//!      estimated standard deviation (per dimension), scaled by the
//!      Roberts-Gelman-Gilks `2.38 / sqrt(d)` rule.
//! 4. **Sampling phase** (`n_iter` iterations):
//!    - Same proposal kernel, but covariance is now frozen at the warmup
//!      estimate.
//!    - Every iteration's state is recorded as a sample.
//!
//! ## Determinism contract
//!
//! - **All randomness** routes through `cjc_repro::Rng` (SplitMix64) seeded
//!   from `base_seed` via a per-chain stretch.
//! - **Adaptive covariance** is built with [`KahanAccumulatorF64`].
//! - **`Accept`/`Reject` coin flip**: a single `next_f64()` draw compared
//!   against `exp(log_p_new - log_p_current).min(1.0)`. Deterministic.
//! - **Same `(log_posterior, initial_state, n_warmup, n_iter, n_chains,
//!   base_seed)` ⇒ byte-identical `PosteriorSamples.content_hash`.**

use crate::error::TempestError;
use crate::posterior::{ConvergenceDiagnostics, PosteriorSamples};
use crate::sampler::AcceptResult;
use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use cjc_repro::{KahanAccumulatorF64, Rng};

/// Stable string label for content-hashing.
pub const SAMPLER_LABEL: &str = "metropolis_hastings";

/// Roberts-Gelman-Gilks (1997) optimal-scaling constant `2.38²`.
const RGG_SCALING_SQ: f64 = 2.38 * 2.38;

/// Default initial proposal standard deviation per dimension.
pub const DEFAULT_INIT_SIGMA: f64 = 0.5;

/// Metropolis-Hastings sampler.
///
/// Holds the per-chain proposal-kernel state. **Stateless across chains** —
/// each call to [`run`] creates fresh per-chain RNGs and Welford
/// accumulators from the supplied `base_seed`.
#[derive(Clone, Debug)]
pub struct MetropolisHastings {
    init_sigma: f64,
}

impl Default for MetropolisHastings {
    fn default() -> Self {
        Self::new()
    }
}

impl MetropolisHastings {
    /// Construct with [`DEFAULT_INIT_SIGMA`] initial proposal SD.
    pub fn new() -> Self {
        Self { init_sigma: DEFAULT_INIT_SIGMA }
    }

    /// Override the initial proposal standard deviation (used until the
    /// warmup adaptation kicks in halfway through warmup).
    pub fn with_init_sigma(mut self, s: f64) -> Self {
        self.init_sigma = s;
        self
    }

    /// Run `n_chains` independent chains, each with `n_warmup + n_iter`
    /// total iterations. Returns `n_chains × n_iter × n_dim` samples plus
    /// content-addressed [`PosteriorSamples::content_hash`].
    ///
    /// # Arguments
    ///
    /// - `log_posterior` — log-density closure `Fn(&[f64]) -> f64`. May
    ///   return `-inf` for hard constraints; if it returns NaN at the
    ///   initial state, [`TempestError::InvalidLogPosterior`] is returned
    ///   immediately.
    /// - `initial_state` — starting point for every chain. Must be all
    ///   finite.
    /// - `n_chains` — number of independent chains. Must be `≥ 1`.
    /// - `n_warmup` — warmup iterations per chain. Must be `≥ 10`.
    /// - `n_iter` — post-warmup samples per chain. Must be `≥ 1`.
    /// - `base_seed` — base SplitMix64 seed. Per-chain seeds derive via
    ///   `Rng::seeded(base_seed.wrapping_mul(chain_seed_stride) ^ chain_idx)`.
    pub fn run<F>(
        &self,
        log_posterior: F,
        initial_state: &[f64],
        n_chains: usize,
        n_warmup: usize,
        n_iter: usize,
        base_seed: u64,
    ) -> Result<PosteriorSamples, TempestError>
    where
        F: Fn(&[f64]) -> f64,
    {
        // Config validation.
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
        if !self.init_sigma.is_finite() || self.init_sigma <= 0.0 {
            return Err(TempestError::Unsupported {
                detail: format!("init_sigma must be > 0 and finite, got {}", self.init_sigma),
            });
        }
        // Probe the log-posterior at the initial state.
        let initial_lp = log_posterior(initial_state);
        if initial_lp.is_nan() {
            return Err(TempestError::InvalidLogPosterior {
                detail: "log_posterior returned NaN at initial_state".to_string(),
            });
        }

        let n_dim = initial_state.len();
        let mut chains: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_chains);
        // Per-chain stride: derive a distinct seed per chain via SplitMix64-
        // flavored mixing of `(base_seed, chain_idx)`. We use a wrapping
        // multiplication by a large odd constant to ensure adjacent chain
        // indices give very different starting points.
        const PER_CHAIN_MIX: u64 = 0x9E37_79B9_7F4A_7C15; // SplitMix64 step constant
        for chain_idx in 0..n_chains {
            let chain_seed = base_seed
                .wrapping_add((chain_idx as u64).wrapping_mul(PER_CHAIN_MIX));
            let mut rng = Rng::seeded(chain_seed);
            let mut current_state: Vec<f64> = initial_state.to_vec();
            let mut current_lp = initial_lp;
            // Per-dimension Welford for the warmup adaptation.
            let mut welford_means: Vec<KahanAccumulatorF64> =
                (0..n_dim).map(|_| KahanAccumulatorF64::new()).collect();
            let mut welford_m2: Vec<KahanAccumulatorF64> =
                (0..n_dim).map(|_| KahanAccumulatorF64::new()).collect();
            let mut welford_n: u64 = 0;
            // Proposal SDs per dimension (initialised to init_sigma).
            let mut proposal_sigma = vec![self.init_sigma; n_dim];
            let adaptation_kicks_in_at = n_warmup / 2;
            let mut samples = Vec::with_capacity(n_iter);
            for iter in 0..(n_warmup + n_iter) {
                // Proposal kernel: x_new = x + N(0, σ_dim² · I). For each
                // dimension, draw an independent standard normal via Box-
                // Muller (two uniforms per draw).
                let mut proposal: Vec<f64> = Vec::with_capacity(n_dim);
                for d in 0..n_dim {
                    let z = standard_normal(&mut rng);
                    proposal.push(current_state[d] + proposal_sigma[d] * z);
                }
                // Evaluate log-posterior at proposal.
                let proposal_lp = log_posterior(&proposal);
                let accept_result = mh_accept_reject(current_lp, proposal_lp, &mut rng);
                if let AcceptResult::Accept { new_log_posterior } = accept_result {
                    current_state = proposal;
                    current_lp = new_log_posterior;
                }
                // Welford update during warmup, applied to the *current*
                // state (which may be the new proposal if accepted, or the
                // old state if rejected).
                if iter < n_warmup {
                    welford_n += 1;
                    let n_f = welford_n as f64;
                    for d in 0..n_dim {
                        let prev_mean = welford_means[d].finalize();
                        let delta = current_state[d] - prev_mean;
                        let new_mean = prev_mean + delta / n_f;
                        // Re-set the mean accumulator with the new mean.
                        // Kahan-accurate single-set: reset + add.
                        welford_means[d] = KahanAccumulatorF64::new();
                        welford_means[d].add(new_mean);
                        let delta2 = current_state[d] - new_mean;
                        welford_m2[d].add(delta * delta2);
                    }
                    // Apply adaptation at the halfway point of warmup.
                    if iter + 1 == adaptation_kicks_in_at && welford_n >= 2 {
                        let variance_denom = (welford_n - 1) as f64;
                        for d in 0..n_dim {
                            let var = welford_m2[d].finalize() / variance_denom;
                            let sd = if var > 0.0 { var.sqrt() } else { self.init_sigma };
                            // Roberts-Gelman-Gilks scaling: σ_proposal² =
                            // (2.38² / d) · σ_estimated².
                            let scale_sq = RGG_SCALING_SQ / n_dim as f64;
                            proposal_sigma[d] = (scale_sq * sd * sd).sqrt().max(f64::EPSILON);
                        }
                    }
                } else {
                    samples.push(current_state.clone());
                }
            }
            chains.push(samples);
        }

        // Build diagnostics. v0.1 ships only `divergences = 0` (MH has no
        // divergences) + R-hat = NaN per parameter (single-chain R-hat is
        // 1.0 by definition; multi-chain R-hat ships in a later session).
        let diagnostics = ConvergenceDiagnostics {
            r_hat: vec![f64::NAN; n_dim],
            ess_bulk: vec![f64::NAN; n_dim],
            ess_tail: vec![f64::NAN; n_dim],
            divergences: 0,
            n_max_treedepth: 0,
        };

        let content_hash =
            compute_content_hash(SAMPLER_LABEL, n_chains, n_iter, n_dim, base_seed, &chains);

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

/// Box-Muller standard-normal draw consuming two uniforms.
fn standard_normal(rng: &mut Rng) -> f64 {
    // Avoid `ln(0)` by guarding the first uniform.
    let mut u1 = rng.next_f64();
    while u1 == 0.0 {
        u1 = rng.next_f64();
    }
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}

/// Metropolis-Hastings accept/reject decision.
///
/// Symmetric proposal ⇒ acceptance ratio is `exp(lp_new - lp_current)`.
/// If `lp_new` is NaN (the closure returned NaN at the proposal), reject
/// without consuming the uniform — but we still consume one to keep the
/// RNG stream byte-identical regardless of how many proposals turn out
/// to be NaN. **This is critical for determinism**: any per-proposal
/// branch that consumes a different number of RNG draws would break
/// content-hash reproducibility across runs that happen to land NaN
/// proposals in different places.
pub(crate) fn mh_accept_reject(
    lp_current: f64,
    lp_new: f64,
    rng: &mut Rng,
) -> AcceptResult {
    // ALWAYS consume the uniform, then decide.
    let u = rng.next_f64();
    if !lp_new.is_finite() {
        return AcceptResult::Reject;
    }
    let log_alpha = lp_new - lp_current;
    if log_alpha >= 0.0 || u.ln() < log_alpha {
        AcceptResult::Accept { new_log_posterior: lp_new }
    } else {
        AcceptResult::Reject
    }
}

/// Compute the content-addressed hash of a posterior chain set.
fn compute_content_hash(
    sampler_label: &str,
    n_chains: usize,
    n_iter: usize,
    n_dim: usize,
    base_seed: u64,
    chains: &[Vec<Vec<f64>>],
) -> FingerprintId {
    let mut parts: Vec<FingerprintId> = Vec::with_capacity(5 + n_chains);
    parts.push(fingerprint_str(IdDomain::CausalClaim, sampler_label));
    parts.push(fingerprint(IdDomain::CausalClaim, &(n_chains as u64).to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &(n_iter as u64).to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &(n_dim as u64).to_le_bytes()));
    parts.push(fingerprint(IdDomain::CausalClaim, &base_seed.to_le_bytes()));
    for chain in chains {
        let mut chain_bytes: Vec<u8> = Vec::with_capacity(chain.len() * n_dim * 8);
        for sample in chain {
            for v in sample {
                chain_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        parts.push(fingerprint(IdDomain::CausalClaim, &chain_bytes));
    }
    fingerprint_compose(IdDomain::CausalClaim, "metropolis_posterior", &parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit-variance Gaussian centered at 0 in 1D.
    fn log_p_unit_gaussian_1d(x: &[f64]) -> f64 {
        -0.5 * x[0] * x[0] - 0.5 * std::f64::consts::TAU.ln()
    }

    /// 2D unit-variance Gaussian.
    fn log_p_unit_gaussian_2d(x: &[f64]) -> f64 {
        -0.5 * (x[0] * x[0] + x[1] * x[1])
            - std::f64::consts::TAU.ln()
    }

    #[test]
    fn empty_initial_state_returns_invalid() {
        let mh = MetropolisHastings::new();
        let err = mh
            .run(log_p_unit_gaussian_1d, &[], 1, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::InvalidInitialState { .. }));
    }

    #[test]
    fn nan_initial_state_returns_invalid() {
        let mh = MetropolisHastings::new();
        let err = mh
            .run(log_p_unit_gaussian_1d, &[f64::NAN], 1, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::InvalidInitialState { .. }));
    }

    #[test]
    fn nan_log_posterior_at_initial_state_returns_invalid_log_posterior() {
        let mh = MetropolisHastings::new();
        let err = mh
            .run(|_: &[f64]| f64::NAN, &[0.0], 1, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::InvalidLogPosterior { .. }));
    }

    #[test]
    fn n_chains_zero_returns_unsupported() {
        let mh = MetropolisHastings::new();
        let err = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 0, 100, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::Unsupported { .. }));
    }

    #[test]
    fn n_iter_zero_returns_unsupported() {
        let mh = MetropolisHastings::new();
        let err = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 100, 0, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::Unsupported { .. }));
    }

    #[test]
    fn warmup_too_small_returns_unsupported() {
        let mh = MetropolisHastings::new();
        let err = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 5, 100, 42)
            .unwrap_err();
        assert!(matches!(err, TempestError::Unsupported { .. }));
    }

    /// The HEADLINE determinism test (cannot be deferred).
    ///
    /// Same model + same seed + same configuration ⇒ byte-identical
    /// posterior content hash AND byte-identical sample bit patterns.
    #[test]
    fn same_seed_byte_identical_posterior() {
        let mh = MetropolisHastings::new();
        let initial = vec![0.0, 0.0];
        let r1 = mh
            .run(log_p_unit_gaussian_2d, &initial, 2, 50, 100, 42)
            .unwrap();
        let r2 = mh
            .run(log_p_unit_gaussian_2d, &initial, 2, 50, 100, 42)
            .unwrap();
        assert_eq!(r1.content_hash, r2.content_hash);
        assert_eq!(r1.n_chains, r2.n_chains);
        assert_eq!(r1.n_samples_per_chain, r2.n_samples_per_chain);
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
        let mh = MetropolisHastings::new();
        let r1 = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 50, 100, 1)
            .unwrap();
        let r2 = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 50, 100, 2)
            .unwrap();
        assert_ne!(r1.content_hash, r2.content_hash);
    }

    #[test]
    fn chain_shape_matches_config() {
        let mh = MetropolisHastings::new();
        let r = mh
            .run(log_p_unit_gaussian_2d, &[0.0, 0.0], 3, 50, 200, 99)
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
        let mh = MetropolisHastings::new();
        let r = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 100, 500, 7)
            .unwrap();
        for sample in &r.chains[0] {
            for &v in sample {
                assert!(v.is_finite(), "sample contains non-finite value: {}", v);
            }
        }
    }

    #[test]
    fn samples_approximate_unit_gaussian_mean_and_variance() {
        // Generous tolerances — MH on a 1D Gaussian with adaptive proposal
        // should converge to mean ~ 0 and variance ~ 1 with 2000 samples.
        let mh = MetropolisHastings::new();
        let r = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 200, 2000, 42)
            .unwrap();
        let mut mean_acc = KahanAccumulatorF64::new();
        for s in &r.chains[0] {
            mean_acc.add(s[0]);
        }
        let m = mean_acc.finalize() / 2000.0;
        assert!(m.abs() < 0.4, "sample mean was {}", m);
        let mut var_acc = KahanAccumulatorF64::new();
        for s in &r.chains[0] {
            let d = s[0] - m;
            var_acc.add(d * d);
        }
        let v = var_acc.finalize() / 1999.0;
        assert!(
            (v - 1.0).abs() < 0.6,
            "sample variance was {} (expected ≈ 1.0)",
            v,
        );
    }

    #[test]
    fn diagonal_proposal_adapts_during_warmup() {
        // Run with a very large init_sigma; if adaptation works, the
        // post-warmup proposal SD should shrink toward the unit-gaussian
        // estimate ≈ 1.0. We can't observe σ directly, but we can observe
        // that the sample autocorrelation is finite (not stuck because of
        // 99% rejection).
        let mh = MetropolisHastings::new().with_init_sigma(20.0);
        let r = mh
            .run(log_p_unit_gaussian_1d, &[0.0], 1, 500, 1000, 11)
            .unwrap();
        // Count distinct values in the chain — if rejection rate were near
        // 100% we'd see one value repeated. Anything > 50 distinct values
        // means the proposal SD adapted to a reasonable scale.
        let mut distinct: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for s in &r.chains[0] {
            distinct.insert(s[0].to_bits());
        }
        assert!(distinct.len() > 50, "only {} distinct values", distinct.len());
    }

    #[test]
    fn mh_accept_reject_consumes_one_uniform_regardless_of_decision() {
        // Critical for determinism: both Accept and Reject must consume
        // the same number of RNG draws.
        let mut rng1 = Rng::seeded(0);
        let mut rng2 = Rng::seeded(0);
        // Forced accept: log_alpha = 100 - 0 ≫ 0.
        let _ = mh_accept_reject(0.0, 100.0, &mut rng1);
        // Forced reject: log_alpha = 0 - 100 ≪ 0; only accepts if u.ln() < -100.
        let _ = mh_accept_reject(100.0, 0.0, &mut rng2);
        // After one call each, both RNGs must be at the same state — meaning
        // the next call returns the same value.
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
}
