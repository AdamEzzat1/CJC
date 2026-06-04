//! Deterministic seeding contract for cjc-cronos-gan.
//!
//! Every random draw in the crate routes through a [`CronosSeed`]
//! sub-stream so two runs with the same `CronosSeed` produce byte-identical
//! parameters, batches, and losses. The sub-stream pattern matches the
//! cjc-tempest seed-flow precedent: instead of one global RNG, every
//! domain (parameter init, batch ordering, gate noise, audit IDs) gets its
//! own SplitMix64 sub-stream derived from the master seed.
//!
//! The `CronosRunId` is the content-addressed fingerprint of the seed +
//! configuration; published experiments cite the run ID and anyone
//! re-running gets the same ID.

use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use cjc_repro::Rng;

/// SplitMix64 mixing step constant — same one used by Metropolis,
/// HMC, and the canonical SplitMix64 update. Keeps cross-sampler /
/// cross-stream seed semantics consistent across the workspace.
const SUBSTREAM_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Master deterministic seed for a Cronos experiment.
///
/// Sub-streams are derived by mixing the master seed with a salt string
/// hashed via [`fingerprint_str`]. Same `(CronosSeed, salt)` ⇒ same
/// [`Rng`]. The salt names the *domain* of randomness (e.g. `"ssm_init"`,
/// `"liquid_init"`, `"batch_shuffle"`) so two domains never share a
/// stream and therefore can't accidentally couple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CronosSeed(pub u64);

impl CronosSeed {
    /// Derive a deterministic [`Rng`] sub-stream for the given salt.
    ///
    /// The salt's bytes are SplitMix64-hashed and mixed into the master
    /// seed via wrapping multiply by the SplitMix step constant. Two
    /// distinct salts always produce distinct streams (with overwhelming
    /// probability for the 64-bit fingerprint).
    pub fn substream(&self, salt: &str) -> Rng {
        let salt_hash = fingerprint_str(IdDomain::CausalClaim, salt).0;
        let mixed = self.0.wrapping_add(salt_hash.wrapping_mul(SUBSTREAM_MIX));
        Rng::seeded(mixed)
    }
}

/// Content-addressed run identifier for a Cronos experiment.
///
/// Computed over `(seed, sampler_label, primary_config_bytes)`. Two runs
/// of the same `(seed, config)` produce the same `CronosRunId`; any
/// perturbation of the config — including a hyperparameter as small as
/// `tau_min` — produces a different ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CronosRunId(pub FingerprintId);

impl CronosRunId {
    /// Build a run ID from the seed, a stable sampler label, and the
    /// caller's canonicalised config bytes.
    pub fn build(seed: CronosSeed, sampler_label: &str, config_bytes: &[u8]) -> Self {
        let parts = [
            fingerprint(IdDomain::CausalClaim, &seed.0.to_le_bytes()),
            fingerprint_str(IdDomain::CausalClaim, sampler_label),
            fingerprint(IdDomain::CausalClaim, config_bytes),
        ];
        CronosRunId(fingerprint_compose(
            IdDomain::CausalClaim,
            "cronos_run_id",
            &parts,
        ))
    }
}

impl std::fmt::Display for CronosRunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_substream() {
        let seed = CronosSeed(42);
        let mut rng_a = seed.substream("ssm_init");
        let mut rng_b = seed.substream("ssm_init");
        for _ in 0..16 {
            assert_eq!(rng_a.next_u64(), rng_b.next_u64());
        }
    }

    #[test]
    fn different_salts_diverge() {
        let seed = CronosSeed(42);
        let mut rng_a = seed.substream("ssm_init");
        let mut rng_b = seed.substream("liquid_init");
        // First draws must differ (probability of collision is 2^-64).
        assert_ne!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn different_seeds_diverge() {
        let mut rng_a = CronosSeed(1).substream("ssm_init");
        let mut rng_b = CronosSeed(2).substream("ssm_init");
        assert_ne!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn run_id_stable_per_seed_and_config() {
        let id_a = CronosRunId::build(CronosSeed(42), "ssm_vs_liquid", b"state_dim=8");
        let id_b = CronosRunId::build(CronosSeed(42), "ssm_vs_liquid", b"state_dim=8");
        assert_eq!(id_a, id_b);
    }

    #[test]
    fn run_id_diverges_on_seed_change() {
        let id_a = CronosRunId::build(CronosSeed(42), "ssm_vs_liquid", b"state_dim=8");
        let id_b = CronosRunId::build(CronosSeed(43), "ssm_vs_liquid", b"state_dim=8");
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn run_id_diverges_on_config_change() {
        let id_a = CronosRunId::build(CronosSeed(42), "ssm_vs_liquid", b"state_dim=8");
        let id_b = CronosRunId::build(CronosSeed(42), "ssm_vs_liquid", b"state_dim=16");
        assert_ne!(id_a, id_b);
    }
}
