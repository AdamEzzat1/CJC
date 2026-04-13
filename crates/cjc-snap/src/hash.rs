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
}
