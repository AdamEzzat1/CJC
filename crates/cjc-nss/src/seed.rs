//! Deterministic seeding contract for the NSS crate.
//!
//! Every random draw in NSS routes through an [`NssSeed`] sub-stream so two
//! runs with the same `NssSeed` produce byte-identical parameters,
//! trajectories, predictions, and traces. Mirrors the `cjc-cronos-gan`
//! `CronosSeed` precedent exactly: same SplitMix mixing constant, same
//! salt-hash domain, same sub-stream layout. The cross-crate uniformity
//! makes debugging across the workspace trivial — a seed that diverges
//! between NSS and Cronos can only do so for a *structural* reason.

use cjc_locke::id::{fingerprint, fingerprint_compose, fingerprint_str, FingerprintId, IdDomain};
use cjc_repro::Rng;

/// SplitMix64 mixing-step constant. Same constant used by `cjc-cronos-gan`
/// and the canonical SplitMix64 update so cross-stream seed semantics stay
/// uniform across the workspace.
const SUBSTREAM_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Master deterministic seed for an NSS run.
///
/// Sub-streams are derived by mixing the master seed with a salt string
/// hashed via [`fingerprint_str`]. Same `(NssSeed, salt)` ⇒ same [`Rng`].
/// The salt names the *domain* of randomness (e.g. `"encoder_init"`,
/// `"queue_arrivals"`, `"propagation_jitter"`) so two domains never share
/// a stream and therefore can't accidentally couple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NssSeed(pub u64);

impl NssSeed {
    /// Derive a deterministic [`Rng`] sub-stream for the given salt.
    pub fn substream(&self, salt: &str) -> Rng {
        let salt_hash = fingerprint_str(IdDomain::CausalClaim, salt).0;
        let mixed = self.0.wrapping_add(salt_hash.wrapping_mul(SUBSTREAM_MIX));
        Rng::seeded(mixed)
    }
}

/// Content-addressed hash of an input trajectory or system state. Used by
/// [`PredictionTrace`](crate::PredictionTrace) so an audit verifier can
/// reject a trace whose claimed-input bytes don't match the actual input.
///
/// The hash is intentionally not opaque — it's a `FingerprintId` so it
/// composes with `NssRunId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputHash(pub FingerprintId);

impl InputHash {
    /// Hash a byte slice into an `InputHash`. Determinism: same bytes ⇒
    /// same hash on every platform.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(fingerprint(IdDomain::CausalClaim, bytes))
    }
}

impl std::fmt::Display for InputHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Content-addressed run identifier for an NSS prediction.
///
/// Computed over `(seed, model_version, config_bytes, input_hash)`. Two
/// runs with identical inputs produce the same `NssRunId`; any
/// perturbation of any input — config field, seed, model version, or one
/// byte of the input trajectory — produces a different ID.
///
/// Published audit traces cite the run ID and anyone re-running with the
/// same `(seed, config, model_version, input)` gets the same ID. The
/// `ReplayValidator` rejects a trace whose recomputed ID disagrees.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NssRunId(pub FingerprintId);

impl NssRunId {
    /// Build a run ID from the seed, model version, canonical config
    /// bytes, and input hash.
    pub fn build(
        seed: NssSeed,
        model_version: &str,
        config_bytes: &[u8],
        input_hash: InputHash,
    ) -> Self {
        let parts = [
            fingerprint(IdDomain::CausalClaim, &seed.0.to_le_bytes()),
            fingerprint_str(IdDomain::CausalClaim, model_version),
            fingerprint(IdDomain::CausalClaim, config_bytes),
            input_hash.0,
        ];
        NssRunId(fingerprint_compose(
            IdDomain::CausalClaim,
            "nss_run_id",
            &parts,
        ))
    }
}

impl std::fmt::Display for NssRunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_substream() {
        let seed = NssSeed(42);
        let mut a = seed.substream("encoder_init");
        let mut b = seed.substream("encoder_init");
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_salts_diverge() {
        let seed = NssSeed(42);
        let mut a = seed.substream("encoder_init");
        let mut b = seed.substream("queue_arrivals");
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn different_seeds_diverge_on_same_salt() {
        let mut a = NssSeed(1).substream("encoder_init");
        let mut b = NssSeed(2).substream("encoder_init");
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn run_id_stable_per_inputs() {
        let id_a = NssRunId::build(
            NssSeed(42),
            "nss-0.1.0",
            b"state_dim=8",
            InputHash::of_bytes(b"trace-A"),
        );
        let id_b = NssRunId::build(
            NssSeed(42),
            "nss-0.1.0",
            b"state_dim=8",
            InputHash::of_bytes(b"trace-A"),
        );
        assert_eq!(id_a, id_b);
    }

    #[test]
    fn run_id_diverges_on_any_input_change() {
        let base = NssRunId::build(
            NssSeed(42),
            "nss-0.1.0",
            b"state_dim=8",
            InputHash::of_bytes(b"trace-A"),
        );
        let by_seed = NssRunId::build(
            NssSeed(43),
            "nss-0.1.0",
            b"state_dim=8",
            InputHash::of_bytes(b"trace-A"),
        );
        let by_version = NssRunId::build(
            NssSeed(42),
            "nss-0.1.1",
            b"state_dim=8",
            InputHash::of_bytes(b"trace-A"),
        );
        let by_config = NssRunId::build(
            NssSeed(42),
            "nss-0.1.0",
            b"state_dim=16",
            InputHash::of_bytes(b"trace-A"),
        );
        let by_input = NssRunId::build(
            NssSeed(42),
            "nss-0.1.0",
            b"state_dim=8",
            InputHash::of_bytes(b"trace-B"),
        );
        assert_ne!(base, by_seed);
        assert_ne!(base, by_version);
        assert_ne!(base, by_config);
        assert_ne!(base, by_input);
    }
}
