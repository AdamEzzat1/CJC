//! Posterior-output types: [`PosteriorSamples`] + [`ConvergenceDiagnostics`].
//!
//! The whole reason this crate exists lives in
//! [`PosteriorSamples::content_hash`]: a SplitMix64-derived 64-bit
//! fingerprint over the canonical byte representation of the entire chain
//! tensor + diagnostics. Two runs with the same model + same data + same
//! seed produce the same `content_hash`. Published Bayesian analyses cite
//! this hash; anyone re-running the analysis gets the same hash.

use cjc_locke::id::FingerprintId;

/// MCMC posterior output.
///
/// `chains[c][s][d]` is the value of parameter `d` in sample `s` of chain
/// `c`. The shape is `(n_chains, n_samples_per_chain, n_dim)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorSamples {
    /// Per-chain sample matrix. Outermost is chain index.
    pub chains: Vec<Vec<Vec<f64>>>,
    /// Number of independent chains run.
    pub n_chains: usize,
    /// Number of post-warmup samples per chain.
    pub n_samples_per_chain: usize,
    /// Parameter dimensionality.
    pub n_dim: usize,
    /// Convergence diagnostics aggregated across chains.
    pub diagnostics: ConvergenceDiagnostics,
    /// Content-addressed fingerprint of `(sampler_label, model_hash, seed,
    /// n_chains, n_samples_per_chain, every sample bit-pattern, diagnostics)`.
    ///
    /// This is the headline determinism artifact — two runs with bit-identical
    /// inputs produce the same `content_hash`.
    pub content_hash: FingerprintId,
}

/// Vehtari et al. 2021 split-rank-normalised convergence diagnostics.
///
/// All vectors have length [`PosteriorSamples::n_dim`].
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ConvergenceDiagnostics {
    /// Split-rank-normalised R-hat per parameter. Values close to 1.0
    /// indicate the chains have mixed; values above 1.01 trigger
    /// [`super::TempestError::ConvergenceFailure`].
    pub r_hat: Vec<f64>,

    /// Effective sample size (bulk) per parameter. Vehtari et al. 2021
    /// formulation. Values below 400 are flagged as Locke `E9302`.
    pub ess_bulk: Vec<f64>,

    /// Effective sample size (tail) per parameter. Tracks the precision of
    /// the 5%/95% quantile estimates.
    pub ess_tail: Vec<f64>,

    /// Count of divergent transitions across all chains. HMC/NUTS only;
    /// 0 for Metropolis. Flagged as Locke `E9300` when > 0.
    pub divergences: u64,

    /// Count of NUTS samples that hit the maximum tree-depth limit.
    /// Flagged as Locke `E9303` when > 0.
    pub n_max_treedepth: u64,
}
