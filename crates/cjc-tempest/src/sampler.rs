//! [`Sampler`] trait — the low-level interface every cjc-tempest sampler
//! implements.
//!
//! Concrete implementations (Metropolis-Hastings, HMC, NUTS) live in their
//! own modules that subsequent implementation sessions add.
//!
//! The trait is intentionally low-level — it represents *one step* of a
//! Markov chain. Higher-level `sample(n_chains, n_iter, ...)` drivers live
//! outside this trait so callers can compose them with adaptation phases,
//! cross-chain mixing, and so on.

use cjc_repro::Rng;

/// Result of a single sampler step.
///
/// `Accept` carries the new log-posterior so the driver doesn't have to
/// recompute it on the next step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AcceptResult {
    /// Proposal accepted. `new_log_posterior` is the value at the new state.
    Accept { new_log_posterior: f64 },
    /// Proposal rejected; the state was left unchanged.
    Reject,
    /// Numerical breakdown (e.g., HMC energy error exceeded threshold).
    /// The driver should record this as a divergence and continue.
    Divergent,
}

/// One step of a Markov chain.
///
/// `state` is the current parameter vector (mutated in place if the
/// proposal is accepted). `rng` is the deterministic
/// [`cjc_repro::Rng`](Rng) the sampler MUST use for every random draw —
/// no `rand::thread_rng()`, no system entropy, no platform-dependent
/// thread-local state.
pub trait Sampler {
    /// Stable string label for this sampler kind. Used in
    /// content-addressed hashing of the posterior so two runs with the
    /// same data but different samplers produce different IDs.
    fn label(&self) -> &'static str;

    /// Advance the chain by one step. Mutates `state` in place when the
    /// proposal is accepted; leaves it unchanged when rejected.
    fn step(&mut self, state: &mut [f64], rng: &mut Rng) -> AcceptResult;
}
