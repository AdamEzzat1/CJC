//! Deterministic 64-bit content fingerprints.
//!
//! Every Locke entity that needs a stable identity (findings, impressions,
//! ideas, lineage nodes) is keyed by a `FingerprintId` derived from a
//! canonical byte representation of its content. Two runs that produce the
//! same logical content produce byte-identical IDs.
//!
//! We deliberately avoid pulling in an external hashing crate. The mixer
//! is a `SplitMix64` permutation seeded from a domain salt and chained
//! over input bytes — the same primitive `cjc-repro` uses for RNG, so the
//! determinism story is consistent across the workspace.

use std::fmt;

/// 64-bit content fingerprint. Equal for equal canonical inputs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FingerprintId(pub u64);

impl fmt::Display for FingerprintId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Domain salt prevents collisions between different ID kinds — a finding
/// with the same byte stream as an idea should not share an ID.
#[derive(Clone, Copy, Debug)]
pub enum IdDomain {
    Finding,
    Impression,
    Idea,
    LineageNode,
    LineageEdge,
    AuditEvent,
    CausalClaim,
}

impl IdDomain {
    const fn salt(self) -> u64 {
        // Distinct 64-bit domain salts. Hex literals only — every constant
        // differs in many bits to avoid weak domain separation.
        match self {
            IdDomain::Finding => 0xF1ED_BEEF_0001_0001,
            IdDomain::Impression => 0x1A1A_C0DE_0002_0002,
            IdDomain::Idea => 0xDEAD_BEEF_0003_0003,
            IdDomain::LineageNode => 0xC0DE_C0DE_0004_0004,
            IdDomain::LineageEdge => 0xEDED_EDED_0005_0005,
            IdDomain::AuditEvent => 0xA0A0_A0A0_0006_0006,
            IdDomain::CausalClaim => 0xCACA_CACA_0007_0007,
        }
    }
}

/// One step of the SplitMix64 permutation — bijective on `u64`.
#[inline]
fn splitmix64_step(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a fingerprint from a domain salt and a canonical byte stream.
///
/// Determinism: identical (`domain`, `bytes`) inputs always produce the
/// same `FingerprintId` regardless of platform, run, or process state.
pub fn fingerprint(domain: IdDomain, bytes: &[u8]) -> FingerprintId {
    let mut state = domain.salt();
    state = splitmix64_step(state ^ bytes.len() as u64);
    // Fold bytes 8 at a time, little-endian, then mix the trailing remainder.
    let chunks = bytes.chunks_exact(8);
    let remainder = chunks.remainder();
    for ch in chunks {
        let mut acc = 0u64;
        for (i, b) in ch.iter().enumerate() {
            acc |= (*b as u64) << (i * 8);
        }
        state = splitmix64_step(state ^ acc);
    }
    if !remainder.is_empty() {
        let mut tail = 0u64;
        for (i, b) in remainder.iter().enumerate() {
            tail |= (*b as u64) << (i * 8);
        }
        state = splitmix64_step(state ^ tail);
    }
    FingerprintId(state)
}

/// Convenience for fingerprinting a UTF-8 string in the given domain.
pub fn fingerprint_str(domain: IdDomain, s: &str) -> FingerprintId {
    fingerprint(domain, s.as_bytes())
}

/// Combine an arbitrary number of sub-IDs (and an optional discriminator
/// label) into a parent fingerprint. Used when a finding's identity is a
/// function of several inputs.
pub fn fingerprint_compose(domain: IdDomain, label: &str, parts: &[FingerprintId]) -> FingerprintId {
    let mut buf: Vec<u8> = Vec::with_capacity(label.len() + parts.len() * 8 + 1);
    buf.extend_from_slice(label.as_bytes());
    buf.push(0);
    for p in parts {
        buf.extend_from_slice(&p.0.to_le_bytes());
    }
    fingerprint(domain, &buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_deterministic() {
        let a = fingerprint(IdDomain::Finding, b"missingness:age:42");
        let b = fingerprint(IdDomain::Finding, b"missingness:age:42");
        assert_eq!(a, b);
    }

    #[test]
    fn domain_separates_namespaces() {
        let a = fingerprint(IdDomain::Finding, b"x");
        let b = fingerprint(IdDomain::Idea, b"x");
        assert_ne!(a, b, "same bytes in different domains must hash differently");
    }

    #[test]
    fn empty_input_is_well_defined() {
        let a = fingerprint(IdDomain::Finding, b"");
        let b = fingerprint(IdDomain::Finding, b"");
        assert_eq!(a, b);
    }

    #[test]
    fn small_change_changes_fingerprint() {
        let a = fingerprint(IdDomain::Finding, b"col=age");
        let b = fingerprint(IdDomain::Finding, b"col=Age");
        assert_ne!(a, b, "case change must alter fingerprint");
    }

    #[test]
    fn compose_is_stable_in_order() {
        let p1 = fingerprint(IdDomain::Idea, b"p1");
        let p2 = fingerprint(IdDomain::Idea, b"p2");
        let a = fingerprint_compose(IdDomain::Idea, "join", &[p1, p2]);
        let b = fingerprint_compose(IdDomain::Idea, "join", &[p1, p2]);
        assert_eq!(a, b);
        let swapped = fingerprint_compose(IdDomain::Idea, "join", &[p2, p1]);
        assert_ne!(a, swapped, "parent order is part of identity");
    }

    #[test]
    fn display_is_lowercase_hex_16_chars() {
        let fp = fingerprint(IdDomain::Finding, b"x");
        let s = fp.to_string();
        assert_eq!(s.len(), 16);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
