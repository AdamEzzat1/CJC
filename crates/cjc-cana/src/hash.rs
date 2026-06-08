//! Deterministic content-addressing for CANA reports.
//!
//! We use **FNV-1a** for all CANA hashes. Rationale:
//!
//! - **Cross-platform stable.** Only `u64::wrapping_mul` + `^` — bit-identical
//!   on x86_64, ARM, RISC-V, regardless of endianness because we feed bytes
//!   one at a time.
//! - **Cheap.** ~1 cycle/byte; the entire featurizer's hashing budget is
//!   measured in microseconds even on million-block programs.
//! - **No dependencies.** A 12-line implementation; we don't take on
//!   `sha2`/`blake3` for Phase 1 audit-trail needs.
//! - **Not cryptographic.** CANA reports are inspected, not attacked. If a
//!   future audit-grade story requires collision-resistance, swap in BLAKE3
//!   behind the [`CanaHasher`] facade without touching call sites.
//!
//! ## Why not `std::collections::hash_map::DefaultHasher`?
//!
//! `DefaultHasher` is SipHash *seeded with a random per-process key*. Same
//! program would produce different `feature_hash` values across runs. That
//! would silently break the determinism contract.

// ---------------------------------------------------------------------------
// FNV-1a 64-bit constants
// ---------------------------------------------------------------------------

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

// ---------------------------------------------------------------------------
// CanaHasher — stable streaming hasher
// ---------------------------------------------------------------------------

/// Streaming FNV-1a 64-bit hasher.
///
/// Feed bytes via [`write`](Self::write) (or the convenience methods); finalize
/// with [`finish`](Self::finish). The state is `Clone + Copy + Eq` so partial
/// hashes can be snapshotted and compared (useful for finding the first
/// divergent feature when two reports differ).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanaHasher {
    state: u64,
}

impl CanaHasher {
    /// Construct a fresh hasher initialized to the FNV-1a offset basis.
    pub fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }

    /// Feed a byte slice into the hash.
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    /// Feed a `u8`.
    pub fn write_u8(&mut self, v: u8) {
        self.write(&[v]);
    }

    /// Feed a `u32` in little-endian byte order.
    pub fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }

    /// Feed a `u64` in little-endian byte order.
    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }

    /// Feed a `usize` (as `u64`) in little-endian byte order. Cross-platform
    /// safe because we widen to `u64` before serializing.
    pub fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }

    /// Feed a length-prefixed string. The length prefix prevents
    /// concatenation collisions: `"ab" + "c"` and `"a" + "bc"` hash distinctly.
    pub fn write_str(&mut self, s: &str) {
        self.write_usize(s.len());
        self.write(s.as_bytes());
    }

    /// Feed a discriminator tag — used at the start of each enum branch's
    /// `feed_into` to make `Variant::A(x)` and `Variant::B(x)` distinct
    /// even when they carry the same payload bytes.
    pub fn write_tag(&mut self, tag: u8) {
        self.write_u8(tag);
    }

    /// Finalize and return the 64-bit hash.
    pub fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for CanaHasher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// One-shot helper
// ---------------------------------------------------------------------------

/// Convenience: hash a byte slice in one call.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = CanaHasher::new();
    h.write(bytes);
    h.finish()
}

// ---------------------------------------------------------------------------
// Newtype hashes — distinct types prevent mixing semantically different IDs
// ---------------------------------------------------------------------------

/// Content-addressed fingerprint of a [`cjc_mir::MirProgram`].
///
/// Computed over: each function's name + param signatures + body shape, in
/// `MirFnId` order. Two structurally identical programs produce the same
/// `ProgramHash`; renaming a function changes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramHash(pub u64);

/// Content-addressed fingerprint of a derived [`crate::CanaFeatures`].
///
/// Computed over the per-function feature struct, in function-name order.
/// Two MIR programs that happen to extract identical features (e.g., after
/// dead-code elimination) produce the same `FeatureHash` even if their
/// `ProgramHash` differs — that's the *point* of CANA: hash equivalence
/// classes by what the optimizer can see, not by source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FeatureHash(pub u64);

/// Content-addressed fingerprint of a single function's CFG shape.
///
/// Useful for Phase-2 cost-model caches: `(CfgHash, pass_id) -> CostEstimate`
/// is the natural key. Phase 1 emits these for inspection only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CfgHash(pub u64);

impl ProgramHash {
    /// Render as a lowercase 16-char hex string for JSON / logs.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

impl FeatureHash {
    /// Render as a lowercase 16-char hex string for JSON / logs.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

impl CfgHash {
    /// Render as a lowercase 16-char hex string for JSON / logs.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_empty_matches_offset_basis() {
        // The "hash" of an empty input is the offset basis itself.
        let h = CanaHasher::new();
        assert_eq!(h.finish(), FNV_OFFSET_BASIS);
    }

    #[test]
    fn fnv1a_known_vector() {
        // Spot-check against a published FNV-1a 64-bit test vector.
        // hash("a") = 0xaf63dc4c8601ec8c
        assert_eq!(hash_bytes(b"a"), 0xaf63dc4c8601ec8c);
        // hash("foobar") = 0x85944171f73967e8
        assert_eq!(hash_bytes(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn length_prefix_prevents_concat_collision() {
        // Without a length prefix, write_str("ab") then write_str("c") would
        // hash the same as write_str("a") then write_str("bc"). With it, they
        // must differ.
        let mut h1 = CanaHasher::new();
        h1.write_str("ab");
        h1.write_str("c");

        let mut h2 = CanaHasher::new();
        h2.write_str("a");
        h2.write_str("bc");

        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn tag_makes_same_payload_distinguishable() {
        let mut h1 = CanaHasher::new();
        h1.write_tag(0);
        h1.write_u64(42);

        let mut h2 = CanaHasher::new();
        h2.write_tag(1);
        h2.write_u64(42);

        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn hash_is_deterministic_across_repeated_calls() {
        // The whole point: hashing the same bytes a billion times always
        // gives the same answer. (We only do 1000 here for test speed.)
        let first = hash_bytes(b"determinism");
        for _ in 0..1000 {
            assert_eq!(hash_bytes(b"determinism"), first);
        }
    }

    #[test]
    fn newtype_hex_padding_is_16_chars() {
        assert_eq!(ProgramHash(0).to_hex(), "0000000000000000");
        assert_eq!(ProgramHash(u64::MAX).to_hex(), "ffffffffffffffff");
        assert_eq!(FeatureHash(1).to_hex(), "0000000000000001");
        assert_eq!(CfgHash(0xdeadbeef).to_hex(), "00000000deadbeef");
    }

    #[test]
    fn newtype_ordering_works() {
        // Used for BTreeMap keys downstream.
        assert!(ProgramHash(1) < ProgramHash(2));
        assert!(FeatureHash(100) > FeatureHash(50));
    }
}
