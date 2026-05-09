//! Hand-rolled SHA-256 implementation following FIPS 180-4.
//!
//! Zero external dependencies. Produces a 256-bit (32-byte) digest.

/// SHA-256 round constants: first 32 bits of the fractional parts of the
/// cube roots of the first 64 primes (2..311).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 initial hash values: first 32 bits of the fractional parts of
/// the square roots of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
const H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Right-rotate a 32-bit word by `n` bits.
#[inline(always)]
fn rotr(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

/// SHA-256 Ch(x,y,z) = (x AND y) XOR (NOT x AND z)
#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ ((!x) & z)
}

/// SHA-256 Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z)
#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

/// SHA-256 big sigma 0: ROTR^2(x) XOR ROTR^13(x) XOR ROTR^22(x)
#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}

/// SHA-256 big sigma 1: ROTR^6(x) XOR ROTR^11(x) XOR ROTR^25(x)
#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}

/// SHA-256 small sigma 0: ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)
#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}

/// SHA-256 small sigma 1: ROTR^17(x) XOR ROTR^19(x) XOR SHR^10(x)
#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}

/// Pad the message according to FIPS 180-4 section 5.1.1.
///
/// Append bit '1', then k zero bits so that (len + 1 + k) mod 512 == 448,
/// then append the 64-bit big-endian original length in bits.
fn pad_message(data: &[u8]) -> Vec<u8> {
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);

    // Append 0x80 (the '1' bit followed by seven '0' bits)
    padded.push(0x80);

    // Pad with zeros until length mod 64 == 56
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }

    // Append the original message length in bits as a 64-bit big-endian integer
    padded.extend_from_slice(&bit_len.to_be_bytes());

    padded
}

/// Process a single 512-bit (64-byte) block and update the hash state.
fn process_block(state: &mut [u32; 8], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);

    // 1. Prepare the message schedule W[0..63]
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        w[i] = small_sigma1(w[i - 2])
            .wrapping_add(w[i - 7])
            .wrapping_add(small_sigma0(w[i - 15]))
            .wrapping_add(w[i - 16]);
    }

    // 2. Initialize working variables
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    // 3. Compression loop
    for i in 0..64 {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    // 4. Update state
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// Compute the SHA-256 hash of the input data (FIPS 180-4).
///
/// This is a zero-dependency, hand-rolled implementation used throughout
/// `cjc-snap` for content-addressable hashing. The output is deterministic
/// and platform-independent.
///
/// # Arguments
///
/// * `data` - Arbitrary byte slice to hash.
///
/// # Returns
///
/// A 32-byte array containing the 256-bit digest.
///
/// # Examples
///
/// ```
/// let digest = cjc_snap::hash::sha256(b"abc");
/// assert_eq!(
///     cjc_snap::hash::hex_string(&digest),
///     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
/// );
/// ```
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let padded = pad_message(data);
    let mut state = H_INIT;

    // Process each 512-bit block
    for block in padded.chunks_exact(64) {
        process_block(&mut state, block);
    }

    // Produce the final 256-bit digest
    let mut digest = [0u8; 32];
    for (i, &word) in state.iter().enumerate() {
        let bytes = word.to_be_bytes();
        digest[i * 4] = bytes[0];
        digest[i * 4 + 1] = bytes[1];
        digest[i * 4 + 2] = bytes[2];
        digest[i * 4 + 3] = bytes[3];
    }
    digest
}

/// Streaming SHA-256 hasher.
///
/// Phase 0.7 (C) — adds an incremental hashing API alongside the existing
/// one-shot [`sha256`] function. Unlike `sha256(data)`, which requires the
/// caller to materialize the full input into a `Vec<u8>` before hashing,
/// the streaming API processes data in chunks via [`Sha256::update`] and
/// produces the digest with [`Sha256::finalize`]. This eliminates the
/// per-event `Vec::with_capacity(32 + payload.len())` concat that
/// `AuditEvent::compute_new_hash` previously paid on every chain step.
///
/// The output is byte-identical to one-shot `sha256` for every input — a
/// streaming `update(prev) + update(payload) + finalize()` produces the
/// same 32-byte digest as `sha256(prev || payload)`. Verified by the
/// in-crate `streaming_matches_one_shot_*` tests.
///
/// # Example
///
/// ```
/// use cjc_snap::hash::{Sha256, sha256};
/// let one_shot = sha256(b"hello world");
/// let streamed = {
///     let mut h = Sha256::new();
///     h.update(b"hello ");
///     h.update(b"world");
///     h.finalize()
/// };
/// assert_eq!(one_shot, streamed);
/// ```
#[derive(Debug, Clone)]
pub struct Sha256 {
    /// Internal SHA-256 state — 8 × u32 = 256 bits.
    state: [u32; 8],
    /// Partial-block buffer; holds 0..=63 bytes between `update` calls.
    buffer: [u8; 64],
    /// Number of bytes currently in `buffer` (0..=63 invariant).
    buffer_len: usize,
    /// Total bytes consumed across all `update` calls. Used to write the
    /// 64-bit length suffix during `finalize`.
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Construct a fresh streaming hasher initialized to the FIPS 180-4
    /// initial hash values.
    pub fn new() -> Self {
        Self {
            state: H_INIT,
            buffer: [0u8; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    /// Feed `data` into the hasher. Can be called any number of times; the
    /// final digest depends on the concatenation of all `data` slices in
    /// the order they were supplied.
    pub fn update(&mut self, data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        let mut input = data;
        // 1. Fill any pending partial block from previous calls.
        if self.buffer_len > 0 {
            let needed = 64 - self.buffer_len;
            if input.len() < needed {
                // Not enough to complete the partial block — buffer it.
                self.buffer[self.buffer_len..self.buffer_len + input.len()]
                    .copy_from_slice(input);
                self.buffer_len += input.len();
                return;
            }
            self.buffer[self.buffer_len..64].copy_from_slice(&input[..needed]);
            process_block(&mut self.state, &self.buffer);
            self.buffer_len = 0;
            input = &input[needed..];
        }
        // 2. Process full 64-byte blocks directly from the input slice
        //    without copying through `self.buffer`.
        while input.len() >= 64 {
            process_block(&mut self.state, &input[..64]);
            input = &input[64..];
        }
        // 3. Buffer the trailing bytes for the next call (or finalize).
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len();
        }
    }

    /// Consume the hasher and produce the 32-byte digest. Applies FIPS
    /// 180-4 padding (0x80 + zeros + 64-bit BE length) to the trailing
    /// partial block.
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        // Append the trailing '1' bit (byte 0x80).
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            // Not enough room for the 8-byte length suffix in this block.
            // Zero-pad the rest, process it, then start a fresh block of
            // zeros that will hold the length suffix.
            for i in self.buffer_len..64 {
                self.buffer[i] = 0;
            }
            process_block(&mut self.state, &self.buffer);
            self.buffer = [0u8; 64];
            self.buffer_len = 0;
        }
        // Zero-pad up to byte 56.
        for i in self.buffer_len..56 {
            self.buffer[i] = 0;
        }
        // Write the 64-bit big-endian total bit length at bytes 56..64.
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        process_block(&mut self.state, &self.buffer);

        // Convert state to 32-byte digest (4 BE bytes per u32 word).
        let mut digest = [0u8; 32];
        for (i, &word) in self.state.iter().enumerate() {
            let bytes = word.to_be_bytes();
            digest[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        digest
    }
}

/// Parse a 64-character hex string into a 32-byte hash array.
///
/// # Arguments
///
/// * `hex` - A string of exactly 64 lowercase or uppercase hex characters.
///
/// # Errors
///
/// Returns an error message if `hex` is not exactly 64 characters or
/// contains non-hex characters.
///
/// # Examples
///
/// ```
/// let hash = cjc_snap::hash::hex_to_hash(
///     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
/// ).unwrap();
/// assert_eq!(hash, cjc_snap::hash::sha256(b""));
/// ```
pub fn hex_to_hash(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars, got {}", hex.len()));
    }
    let mut hash = [0u8; 32];
    for i in 0..32 {
        hash[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| format!("invalid hex at position {}", i * 2))?;
    }
    Ok(hash)
}

/// Convert a 32-byte hash to a 64-character lowercase hex string.
///
/// Useful for logging, display, and test assertions.
///
/// # Arguments
///
/// * `hash` - A 32-byte SHA-256 digest.
///
/// # Returns
///
/// A 64-character `String` of lowercase hexadecimal digits.
pub fn hex_string(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for &byte in hash {
        s.push_str(&format!("{:02x}", byte));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        assert_eq!(
            hex_string(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let digest = sha256(b"abc");
        assert_eq!(
            hex_string(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn test_sha256_448_bit_message() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        // = 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
        let digest = sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex_string(&digest),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn test_sha256_deterministic() {
        let d1 = sha256(b"hello world");
        let d2 = sha256(b"hello world");
        assert_eq!(d1, d2, "SHA-256 must be deterministic");
    }

    #[test]
    fn test_sha256_different_inputs() {
        let d1 = sha256(b"hello");
        let d2 = sha256(b"Hello");
        assert_ne!(d1, d2, "Different inputs must produce different hashes");
    }

    #[test]
    fn test_sha256_single_char() {
        // SHA-256("a") = ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb
        let digest = sha256(b"a");
        assert_eq!(
            hex_string(&digest),
            "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
        );
    }

    #[test]
    fn test_hex_string_format() {
        let hash = [0u8; 32];
        let s = hex_string(&hash);
        assert_eq!(s.len(), 64);
        assert_eq!(s, "0000000000000000000000000000000000000000000000000000000000000000");
    }

    // ── Phase 0.7 (C) — streaming SHA-256 parity tests ────────────────

    /// Helper: compute via streaming with a given chunking sequence.
    fn streamed(chunks: &[&[u8]]) -> [u8; 32] {
        let mut h = Sha256::new();
        for c in chunks {
            h.update(c);
        }
        h.finalize()
    }

    #[test]
    fn streaming_matches_one_shot_empty() {
        // SHA-256("") via streaming with no updates must equal
        // sha256(""). This is the degenerate "all padding" case where
        // the bit-length suffix lives in the same final block as the
        // trailing 0x80 (since byte_len < 56).
        assert_eq!(streamed(&[]), sha256(b""));
        assert_eq!(streamed(&[b""]), sha256(b""));
    }

    #[test]
    fn streaming_matches_one_shot_short() {
        assert_eq!(streamed(&[b"abc"]), sha256(b"abc"));
        assert_eq!(streamed(&[b"a"]), sha256(b"a"));
        assert_eq!(streamed(&[b"hello world"]), sha256(b"hello world"));
    }

    #[test]
    fn streaming_matches_one_shot_chunked() {
        // The whole point of streaming: identical output regardless of
        // how the input is split across update calls.
        let full = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let one_shot = sha256(full);
        assert_eq!(streamed(&[full]), one_shot);
        assert_eq!(streamed(&[&full[..3], &full[3..]]), one_shot);
        assert_eq!(streamed(&[&full[..1], &full[1..2], &full[2..]]), one_shot);
        assert_eq!(
            streamed(&[&full[..16], &full[16..32], &full[32..48], &full[48..]]),
            one_shot
        );
    }

    #[test]
    fn streaming_matches_one_shot_block_boundary() {
        // Inputs at exactly 55 / 56 / 57 / 63 / 64 / 65 / 119 / 128 bytes
        // exercise every padding case (single-block-suffix vs
        // overflow-into-extra-block).
        for &n in &[55usize, 56, 57, 63, 64, 65, 119, 128, 191, 192, 256, 1000] {
            let input: Vec<u8> = (0..n).map(|i| (i & 0xff) as u8).collect();
            let one_shot = sha256(&input);
            // Single-update.
            assert_eq!(streamed(&[&input]), one_shot, "len={n} single-update");
            // Byte-by-byte.
            let chunks: Vec<&[u8]> = (0..n).map(|i| &input[i..i + 1]).collect();
            assert_eq!(streamed(&chunks), one_shot, "len={n} byte-by-byte");
            // Half-and-half.
            assert_eq!(
                streamed(&[&input[..n / 2], &input[n / 2..]]),
                one_shot,
                "len={n} half-and-half"
            );
        }
    }

    #[test]
    fn streaming_matches_concat_one_shot_for_audit_pattern() {
        // The audit chain hashing pattern: SHA-256(prev_hash || payload).
        // Streaming form: update(prev_hash); update(payload); finalize().
        // Must produce the same bytes as the one-shot
        // sha256(&prev_hash[..] ++ payload), since that is what
        // `AuditEvent::compute_new_hash` emits today.
        let prev_hash = [0xa5u8; 32];
        for payload_len in [0usize, 1, 16, 32, 56, 63, 64, 65, 96, 128, 257] {
            let payload: Vec<u8> = (0..payload_len).map(|i| (i & 0xff) as u8).collect();
            // One-shot reference.
            let mut concat = Vec::with_capacity(32 + payload.len());
            concat.extend_from_slice(&prev_hash);
            concat.extend_from_slice(&payload);
            let one_shot = sha256(&concat);
            // Streaming.
            let streamed = {
                let mut h = Sha256::new();
                h.update(&prev_hash);
                h.update(&payload);
                h.finalize()
            };
            assert_eq!(streamed, one_shot, "audit pattern, payload_len={payload_len}");
        }
    }

    #[test]
    fn streaming_default_equals_new() {
        let mut a = Sha256::new();
        let mut b = Sha256::default();
        a.update(b"abc");
        b.update(b"abc");
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn streaming_clone_independent() {
        // Cloning the hasher mid-stream must produce two independent
        // states; finalizing one doesn't affect the other. (Useful for
        // future commit-and-continue patterns; verified here.)
        let mut a = Sha256::new();
        a.update(b"prefix");
        let b = a.clone();
        a.update(b"-suffix-A");
        let mut c = b;
        c.update(b"-suffix-B");
        assert_eq!(a.finalize(), sha256(b"prefix-suffix-A"));
        assert_eq!(c.finalize(), sha256(b"prefix-suffix-B"));
    }
}
