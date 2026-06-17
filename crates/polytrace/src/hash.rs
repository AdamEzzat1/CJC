//! Deterministic content-addressing for Polytrace traces and reports.
//!
//! We use **FNV-1a 64-bit** — the same choice and rationale as `cjc-cana`
//! (`crates/cjc-cana/src/hash.rs`):
//!
//! - **Cross-platform stable.** Only `u64::wrapping_mul` + `^`, fed bytes one
//!   at a time, so the result is bit-identical on x86_64 / ARM / RISC-V
//!   regardless of endianness.
//! - **No dependencies.** A ~15-line implementation; Polytrace does not take on
//!   `sha2`/`blake3` for an inspected-not-attacked audit trail.
//! - **NOT `DefaultHasher`.** `std`'s `DefaultHasher` is SipHash seeded with a
//!   random per-process key — it would silently break the determinism contract
//!   (same trace → different hash across runs).

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Streaming FNV-1a 64-bit hasher. `Copy` so partial hashes can be snapshotted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolytraceHasher {
    state: u64,
}

impl Default for PolytraceHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl PolytraceHasher {
    /// Fresh hasher initialized to the FNV-1a offset basis.
    pub fn new() -> Self {
        Self {
            state: FNV_OFFSET_BASIS,
        }
    }

    /// Feed a byte slice.
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    /// Feed a single byte.
    pub fn write_u8(&mut self, v: u8) {
        self.write(&[v]);
    }

    /// Feed a `u32` in fixed little-endian order (endianness-independent because
    /// we choose the byte order explicitly).
    pub fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }

    /// Feed a `u64` in fixed little-endian order.
    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }

    /// Feed an `i64` (two's-complement bit pattern, fixed little-endian).
    pub fn write_i64(&mut self, v: i64) {
        self.write(&v.to_le_bytes());
    }

    /// Feed a length-prefixed string so `"ab" + "c"` never collides with
    /// `"a" + "bc"`.
    pub fn write_str(&mut self, s: &str) {
        self.write_u64(s.len() as u64);
        self.write(s.as_bytes());
    }

    /// Finalize and return the 64-bit digest.
    pub fn finish(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_input_same_hash() {
        let mut a = PolytraceHasher::new();
        let mut b = PolytraceHasher::new();
        a.write_str("process_batch");
        a.write_u64(42);
        b.write_str("process_batch");
        b.write_u64(42);
        assert_eq!(a.finish(), b.finish());
    }

    #[test]
    fn length_prefix_avoids_concat_collision() {
        let mut a = PolytraceHasher::new();
        let mut b = PolytraceHasher::new();
        a.write_str("ab");
        a.write_str("c");
        b.write_str("a");
        b.write_str("bc");
        assert_ne!(a.finish(), b.finish());
    }

    #[test]
    fn order_sensitive() {
        let mut a = PolytraceHasher::new();
        let mut b = PolytraceHasher::new();
        a.write_u64(1);
        a.write_u64(2);
        b.write_u64(2);
        b.write_u64(1);
        assert_ne!(a.finish(), b.finish());
    }
}
