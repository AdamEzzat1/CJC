//! `snap_decode` must treat every count and length in its input as untrusted.
//!
//! The decoder used to call `Vec::with_capacity(count)` (and once
//! `vec![0.0; numel]`) with counts read straight from the payload. Five bytes
//! were enough to ask for gigabytes: CI's `snap_decode_no_panic` proptest hit
//! `[TAG_CATEGORICAL, u32 n_levels]` and aborted the whole test binary with
//! "memory allocation of 56857839384 bytes failed" (Windows,
//! STATUS_STACK_BUFFER_OVERRUN). An allocation failure is an abort, not a
//! panic, so `catch_unwind` cannot contain it.
//!
//! Every input below is far too short to hold what it claims, so the only
//! correct answer is an error. None of these tests use `catch_unwind`: a
//! panic or an abort is a failure.

use cjc_snap::{snap_decode, snap_decode_v2, SnapError};

// Tag bytes (crates/cjc-snap/src/encode.rs).
const TAG_STRING: u8 = 0x04;
const TAG_ARRAY: u8 = 0x05;
const TAG_TUPLE: u8 = 0x06;
const TAG_STRUCT: u8 = 0x07;
const TAG_TENSOR: u8 = 0x08;
const TAG_ENUM: u8 = 0x09;
const TAG_BYTES: u8 = 0x0A;
const TAG_MAP: u8 = 0x11;
const TAG_TYPED_TENSOR: u8 = 0x12;
const TAG_CHUNKED_TENSOR: u8 = 0x13;
const TAG_SPARSE_CSR: u8 = 0x14;
const TAG_CATEGORICAL: u8 = 0x15;
const TAG_SCHEMA: u8 = 0x16;
const TAG_DATAFRAME: u8 = 0x17;

fn u64le(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

fn payload(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

fn assert_err(bytes: &[u8], what: &str) {
    match snap_decode(bytes) {
        Err(_) => {}
        Ok(v) => panic!("{what}: expected an error, got {v:?}"),
    }
}

const HUGE: u64 = 1 << 40; // ~10^12 elements: terabytes if trusted
const MAX: u64 = u64::MAX;

#[test]
fn categorical_level_count_from_ci_crash() {
    // The exact shape of the CI failure: tag + a u32 level count, nothing else.
    for n in [0x8D35_A5A1u32, u32::MAX] {
        assert_err(&payload(&[&[TAG_CATEGORICAL], &n.to_le_bytes()]), "categorical n_levels");
    }
    // Zero levels, then a huge row count.
    assert_err(
        &payload(&[&[TAG_CATEGORICAL], &0u32.to_le_bytes(), &u64le(HUGE)]),
        "categorical n_rows",
    );
}

#[test]
fn container_counts() {
    for tag in [TAG_ARRAY, TAG_TUPLE, TAG_MAP] {
        for n in [HUGE, MAX] {
            assert_err(&payload(&[&[tag], &u64le(n)]), "container count");
        }
    }
    // Struct / enum: name strings first, then the count.
    let empty_str = u64le(0);
    for n in [HUGE, MAX] {
        assert_err(&payload(&[&[TAG_STRUCT], &empty_str, &u64le(n)]), "struct count");
        assert_err(
            &payload(&[&[TAG_ENUM], &empty_str, &empty_str, &u64le(n)]),
            "enum count",
        );
    }
    assert_err(&payload(&[&[TAG_SCHEMA], &u32::MAX.to_le_bytes()]), "schema n_fields");
}

#[test]
fn byte_lengths_do_not_overflow() {
    // `pos + len` used to overflow (a panic in debug builds) for len near u64::MAX.
    for tag in [TAG_STRING, TAG_BYTES] {
        for n in [HUGE, MAX, MAX - 1] {
            assert_err(&payload(&[&[tag], &u64le(n)]), "string/bytes length");
        }
    }
}

#[test]
fn tensor_shapes() {
    // ndim huge.
    assert_err(&payload(&[&[TAG_TENSOR], &u64le(MAX)]), "tensor ndim");
    // One huge dimension, no data.
    assert_err(&payload(&[&[TAG_TENSOR], &u64le(1), &u64le(HUGE)]), "tensor numel");
    // Shape whose product overflows usize.
    assert_err(
        &payload(&[&[TAG_TENSOR], &u64le(2), &u64le(MAX), &u64le(3)]),
        "tensor numel overflow",
    );
}

#[test]
fn typed_and_chunked_tensor_shapes() {
    for dtype in 0u8..=9 {
        // Huge shape, zero raw bytes. dtype >= 5 used to zero-fill `numel`
        // elements regardless of how many bytes the payload carried.
        assert_err(
            &payload(&[&[TAG_TYPED_TENSOR, dtype], &u64le(1), &u64le(HUGE), &u64le(0)]),
            "typed tensor numel",
        );
        // Product overflow (used to panic in `Iterator::product`).
        assert_err(
            &payload(&[&[TAG_TYPED_TENSOR, dtype], &u64le(2), &u64le(MAX), &u64le(4), &u64le(0)]),
            "typed tensor numel overflow",
        );
        // Chunked: shape, chunk_size, zero chunks.
        assert_err(
            &payload(&[&[TAG_CHUNKED_TENSOR, dtype], &u64le(1), &u64le(HUGE), &u64le(64), &u64le(0)]),
            "chunked tensor numel",
        );
        assert_err(
            &payload(&[&[TAG_CHUNKED_TENSOR, dtype], &u64le(2), &u64le(MAX), &u64le(2), &u64le(64), &u64le(0)]),
            "chunked tensor numel overflow",
        );
    }
    assert_err(&payload(&[&[TAG_TYPED_TENSOR, 0], &u64le(MAX)]), "typed tensor ndim");
}

#[test]
fn sparse_csr_counts() {
    // nrows = u64::MAX used to overflow `nrows + 1`.
    for nrows in [HUGE, MAX] {
        assert_err(
            &payload(&[&[TAG_SPARSE_CSR, 0], &u64le(nrows), &u64le(1), &u64le(0)]),
            "sparse nrows",
        );
    }
    // Valid row_ptr for 0 rows, then a huge nnz.
    assert_err(
        &payload(&[&[TAG_SPARSE_CSR, 0], &u64le(0), &u64le(0), &u64le(HUGE), &u64le(0)]),
        "sparse nnz",
    );
}

#[test]
fn dataframe_counts() {
    // Huge column count.
    assert_err(
        &payload(&[&[TAG_DATAFRAME], &u32::MAX.to_le_bytes(), &u64le(0)]),
        "dataframe n_cols",
    );
    // One column of each type with a huge row count and no data.
    for col_type in 0u8..=5 {
        let mut p = payload(&[&[TAG_DATAFRAME], &1u32.to_le_bytes(), &u64le(HUGE), &u64le(0), &[col_type]]);
        if col_type == 4 {
            p.extend_from_slice(&0u32.to_le_bytes()); // zero levels
        }
        assert_err(&p, "dataframe n_rows");
    }
    // Categorical column with a huge level count.
    assert_err(
        &payload(&[&[TAG_DATAFRAME], &1u32.to_le_bytes(), &u64le(0), &u64le(0), &[4], &u32::MAX.to_le_bytes()]),
        "dataframe categorical n_levels",
    );
}

#[test]
fn v2_header_path_is_bounded_too() {
    let mut p = cjc_snap::SNAP_MAGIC.to_vec();
    p.extend_from_slice(&[cjc_snap::SNAP_VERSION, 0, TAG_CATEGORICAL]);
    p.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(snap_decode_v2(&p).is_err());
}

/// Deterministic fuzz: every tag followed by random bytes, much longer inputs
/// than the proptest uses. Decoding may succeed or fail but must return.
#[test]
fn random_payloads_after_every_tag_return() {
    let mut state: u64 = 0x5EED_CAFE_F00D_0001;
    let mut next = || {
        // SplitMix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut decoded = 0usize;
    for tag in 0u8..=0x19 {
        for _ in 0..2000 {
            let len = (next() % 96) as usize;
            let mut p = Vec::with_capacity(len + 1);
            p.push(tag);
            while p.len() < len + 1 {
                p.extend_from_slice(&next().to_le_bytes());
            }
            p.truncate(len + 1);
            if snap_decode(&p).is_ok() {
                decoded += 1;
            }
        }
    }
    // Sanity: the fuzz is not trivially rejecting everything.
    assert!(decoded > 0);
}

/// A well-formed payload that declares exactly as many elements as it holds
/// still decodes: the caps only bite when the input is too short.
#[test]
fn exact_fit_payloads_still_decode() {
    // Array of 3 ints: each element is 9 bytes (tag + i64).
    let mut p = payload(&[&[TAG_ARRAY], &u64le(3)]);
    for v in [1i64, 2, 3] {
        p.push(0x01);
        p.extend_from_slice(&v.to_le_bytes());
    }
    assert!(matches!(snap_decode(&p), Ok(cjc_runtime::Value::Array(a)) if a.len() == 3));

    // Array of 4 voids: each element is a single tag byte.
    let p = payload(&[&[TAG_ARRAY], &u64le(4), &[0x00, 0x00, 0x00, 0x00]]);
    assert!(matches!(snap_decode(&p), Ok(cjc_runtime::Value::Array(a)) if a.len() == 4));

    // One element short is an error, not a smaller array.
    let p = payload(&[&[TAG_ARRAY], &u64le(4), &[0x00, 0x00, 0x00]]);
    assert!(matches!(snap_decode(&p), Err(SnapError::UnexpectedEof)));
}
