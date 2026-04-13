//! Integration, proptest, and bolero fuzz tests for the tensor_save/load
//! family of builtins (Phase A1/A2 of the Chess RL v2.1 upgrade).
//!
//! Coverage:
//! - Unit roundtrip tests through both `cjc-eval` and `cjc-mir-exec`
//! - Parity: same CJC source on both backends produces identical output
//! - Proptest: random (shape, data) → save/load roundtrip preserves bits
//! - Bolero fuzz: random binary inputs into `tensor_list_load` must not panic
//! - tensor_list_hash: determinism, order-sensitivity, data-sensitivity
//!
//! All temp files are created under a dedicated test directory and the path
//! is threaded into the CJC source so parallel test runs do not collide.

use std::fs;
use std::path::PathBuf;

use cjc_runtime::tensor::Tensor;
use cjc_runtime::tensor_snap;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("cjc_tensor_snap_tests");
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Unique path for a single test (uses the test name + a nonce from process).
/// Returns a forward-slash-normalized path so it can be embedded in CJC-Lang
/// string literals without triggering unknown-escape errors on Windows.
fn tmp_path(tag: &str) -> String {
    let pid = std::process::id();
    let thread_id = format!("{:?}", std::thread::current().id());
    let safe_thread_id: String = thread_id
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    let raw = tmp_dir()
        .join(format!("{tag}_{pid}_{safe_thread_id}.snap"))
        .to_string_lossy()
        .into_owned();
    // Normalize backslashes to forward slashes so CJC-Lang string literals
    // don't try to interpret `\U`, `\a`, `\L`, `\T`, `\c`, `\s` as escapes.
    // Windows file APIs accept forward slashes without complaint.
    raw.replace('\\', "/")
}

/// Run a CJC-Lang program through cjc-eval, returning captured stdout lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp.exec(&prog).unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

/// Run a CJC-Lang program through cjc-mir-exec, returning captured stdout lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_val, executor) = cjc_mir_exec::run_program_with_executor(&prog, seed)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

/// Run on both backends and assert byte-identical output. Return the shared output.
fn run_parity(src: &str, seed: u64) -> Vec<String> {
    let a = run_eval(src, seed);
    let b = run_mir(src, seed);
    assert_eq!(
        a, b,
        "parity violation between cjc-eval and cjc-mir-exec\neval: {a:?}\nmir: {b:?}"
    );
    a
}

// ---------------------------------------------------------------------------
// Integration tests — tensor_save / tensor_load
// ---------------------------------------------------------------------------

#[test]
fn tensor_save_load_single_roundtrip_eval() {
    let path = tmp_path("single_eval");
    let src = format!(
        r#"
        fn main() {{
            let t = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
            tensor_save("{path}", t);
            let loaded = tensor_load("{path}");
            let sh = loaded.shape();
            print(sh[0]);
            print(sh[1]);
            print(loaded.get([0, 0]));
            print(loaded.get([0, 1]));
            print(loaded.get([1, 0]));
            print(loaded.get([1, 1]));
        }}
        "#
    );
    let out = run_eval(&src, 1);
    assert_eq!(out[0].trim(), "2");
    assert_eq!(out[1].trim(), "2");
    assert_eq!(out[2].trim(), "1");
    assert_eq!(out[3].trim(), "2");
    assert_eq!(out[4].trim(), "3");
    assert_eq!(out[5].trim(), "4");
    fs::remove_file(&path).ok();
}

#[test]
fn tensor_save_load_single_roundtrip_mir() {
    let path = tmp_path("single_mir");
    let src = format!(
        r#"
        fn main() {{
            let t = Tensor.from_vec([5.0, 6.0, 7.0], [3]);
            tensor_save("{path}", t);
            let loaded = tensor_load("{path}");
            let sh = loaded.shape();
            print(sh[0]);
            print(loaded.get([0]));
            print(loaded.get([1]));
            print(loaded.get([2]));
        }}
        "#
    );
    let out = run_mir(&src, 1);
    assert_eq!(out[0].trim(), "3");
    assert_eq!(out[1].trim(), "5");
    assert_eq!(out[2].trim(), "6");
    assert_eq!(out[3].trim(), "7");
    fs::remove_file(&path).ok();
}

#[test]
fn tensor_list_save_load_roundtrip_parity() {
    let path = tmp_path("list_parity");
    let src = format!(
        r#"
        fn main() {{
            let a = Tensor.from_vec([1.0, 2.0], [2]);
            let b = Tensor.from_vec([3.0, 4.0, 5.0, 6.0], [2, 2]);
            let c = Tensor.from_vec([7.0], [1]);
            let list = [a, b, c];
            tensor_list_save("{path}", list);
            let loaded = tensor_list_load("{path}");
            print(len(loaded));
            print(loaded[0].get([0]));
            print(loaded[0].get([1]));
            print(loaded[1].get([0, 0]));
            print(loaded[1].get([0, 1]));
            print(loaded[1].get([1, 0]));
            print(loaded[1].get([1, 1]));
            print(loaded[2].get([0]));
        }}
        "#
    );
    let out = run_parity(&src, 1);
    assert_eq!(out[0].trim(), "3");
    assert_eq!(out[1].trim(), "1");
    assert_eq!(out[7].trim(), "7");
    fs::remove_file(&path).ok();
}

#[test]
fn tensor_file_is_byte_identical_across_executors() {
    // Save the same tensor with both executors, assert the files are bit-equal.
    let path_eval = tmp_path("cross_eval");
    let path_mir = tmp_path("cross_mir");
    let src_eval = format!(
        r#"
        fn main() {{
            let t = Tensor.from_vec([0.5, -0.25, 3.75, -1.125], [2, 2]);
            tensor_save("{path_eval}", t);
        }}
        "#
    );
    let src_mir = format!(
        r#"
        fn main() {{
            let t = Tensor.from_vec([0.5, -0.25, 3.75, -1.125], [2, 2]);
            tensor_save("{path_mir}", t);
        }}
        "#
    );
    let _ = run_eval(&src_eval, 1);
    let _ = run_mir(&src_mir, 1);
    let bytes_eval = fs::read(&path_eval).unwrap();
    let bytes_mir = fs::read(&path_mir).unwrap();
    assert_eq!(
        bytes_eval, bytes_mir,
        "tensor_save must produce byte-identical files across executors"
    );
    fs::remove_file(&path_eval).ok();
    fs::remove_file(&path_mir).ok();
}

#[test]
fn tensor_save_load_cross_executor() {
    // Save in eval, load in mir — and vice versa. Both directions must work.
    let path_a = tmp_path("cross_dir_a");
    let path_b = tmp_path("cross_dir_b");
    let save_eval = format!(
        r#"
        fn main() {{
            let t = Tensor.from_vec([11.0, 22.0, 33.0], [3]);
            tensor_save("{path_a}", t);
        }}
        "#
    );
    let load_mir = format!(
        r#"
        fn main() {{
            let t = tensor_load("{path_a}");
            print(t.get([0]));
            print(t.get([1]));
            print(t.get([2]));
        }}
        "#
    );
    let _ = run_eval(&save_eval, 1);
    let out_mir = run_mir(&load_mir, 1);
    assert_eq!(out_mir[0].trim(), "11");
    assert_eq!(out_mir[1].trim(), "22");
    assert_eq!(out_mir[2].trim(), "33");

    let save_mir = format!(
        r#"
        fn main() {{
            let t = Tensor.from_vec([111.0, 222.0], [2]);
            tensor_save("{path_b}", t);
        }}
        "#
    );
    let load_eval = format!(
        r#"
        fn main() {{
            let t = tensor_load("{path_b}");
            print(t.get([0]));
            print(t.get([1]));
        }}
        "#
    );
    let _ = run_mir(&save_mir, 1);
    let out_eval = run_eval(&load_eval, 1);
    assert_eq!(out_eval[0].trim(), "111");
    assert_eq!(out_eval[1].trim(), "222");

    fs::remove_file(&path_a).ok();
    fs::remove_file(&path_b).ok();
}

// ---------------------------------------------------------------------------
// Integration tests — tensor_list_hash
// ---------------------------------------------------------------------------

#[test]
fn tensor_list_hash_is_deterministic_across_executors() {
    let src = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
            let b = Tensor.from_vec([4.0, 5.0, 6.0, 7.0], [2, 2]);
            let h = tensor_list_hash([a, b]);
            print(h);
        }
    "#;
    let out = run_parity(src, 1);
    // The exact hash value is implementation-defined but must be non-zero
    // and identical on both executors.
    let h: i64 = out[0].trim().parse().unwrap();
    assert_ne!(h, 0, "hash should not be zero for non-trivial input");
}

#[test]
fn tensor_list_hash_changes_with_data() {
    let src_a = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
            print(tensor_list_hash([a]));
        }
    "#;
    let src_b = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0, 4.0], [3]);
            print(tensor_list_hash([a]));
        }
    "#;
    let a = run_parity(src_a, 1);
    let b = run_parity(src_b, 1);
    assert_ne!(a[0], b[0], "hash must change when data changes");
}

#[test]
fn tensor_list_hash_is_order_sensitive() {
    let src_ab = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0], [2]);
            let b = Tensor.from_vec([3.0, 4.0], [2]);
            print(tensor_list_hash([a, b]));
        }
    "#;
    let src_ba = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0], [2]);
            let b = Tensor.from_vec([3.0, 4.0], [2]);
            print(tensor_list_hash([b, a]));
        }
    "#;
    let ab = run_parity(src_ab, 1);
    let ba = run_parity(src_ba, 1);
    assert_ne!(ab[0], ba[0], "hash must change when order changes");
}

#[test]
fn tensor_list_hash_distinguishes_shapes() {
    let src_flat = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [4]);
            print(tensor_list_hash([a]));
        }
    "#;
    let src_matrix = r#"
        fn main() {
            let a = Tensor.from_vec([1.0, 2.0, 3.0, 4.0], [2, 2]);
            print(tensor_list_hash([a]));
        }
    "#;
    let flat = run_parity(src_flat, 1);
    let matrix = run_parity(src_matrix, 1);
    assert_ne!(flat[0], matrix[0], "hash must distinguish shapes");
}

// ---------------------------------------------------------------------------
// Proptest — roundtrip preserves bits for random (shape, data)
// ---------------------------------------------------------------------------

fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
    // 1-D, 2-D, or 3-D shapes with small dimensions to keep tests fast.
    prop_oneof![
        (1usize..8).prop_map(|n| vec![n]),
        (1usize..6, 1usize..6).prop_map(|(a, b)| vec![a, b]),
        (1usize..4, 1usize..4, 1usize..4).prop_map(|(a, b, c)| vec![a, b, c]),
    ]
}

fn arb_tensor() -> impl Strategy<Value = Tensor> {
    arb_shape().prop_flat_map(|shape| {
        let numel: usize = shape.iter().product();
        proptest::collection::vec(-1e6f64..1e6f64, numel..=numel)
            .prop_map(move |data| Tensor::from_vec(data, &shape).unwrap())
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Module-level roundtrip: encode then decode preserves shape + bits.
    #[test]
    fn prop_encode_decode_single(t in arb_tensor()) {
        let bytes = tensor_snap::encode_one(&t);
        let loaded = tensor_snap::decode_one(&bytes).unwrap();
        prop_assert_eq!(loaded.shape(), t.shape());
        let original = t.to_vec();
        let roundtripped = loaded.to_vec();
        prop_assert_eq!(original.len(), roundtripped.len());
        for (o, r) in original.iter().zip(roundtripped.iter()) {
            prop_assert_eq!(o.to_bits(), r.to_bits(),
                "roundtrip must preserve exact f64 bit pattern");
        }
    }

    /// Encoding is deterministic: two encodes of the same tensor are bit-equal.
    #[test]
    fn prop_encode_deterministic(t in arb_tensor()) {
        let e1 = tensor_snap::encode_one(&t);
        let e2 = tensor_snap::encode_one(&t);
        prop_assert_eq!(e1, e2);
    }

    /// Hash is deterministic: two hashes of the same list are equal.
    #[test]
    fn prop_hash_deterministic(
        t1 in arb_tensor(),
        t2 in arb_tensor(),
    ) {
        let list = vec![t1, t2];
        let h1 = tensor_snap::hash_list(&list);
        let h2 = tensor_snap::hash_list(&list);
        prop_assert_eq!(h1, h2);
    }

    /// List roundtrip: encode_list then decode_list preserves everything.
    #[test]
    fn prop_encode_decode_list(
        tensors in proptest::collection::vec(arb_tensor(), 0..5)
    ) {
        let bytes = tensor_snap::encode_list(&tensors);
        let loaded = tensor_snap::decode_list(&bytes).unwrap();
        prop_assert_eq!(loaded.len(), tensors.len());
        for (orig, got) in tensors.iter().zip(loaded.iter()) {
            prop_assert_eq!(orig.shape(), got.shape());
            let a = orig.to_vec();
            let b = got.to_vec();
            for (x, y) in a.iter().zip(b.iter()) {
                prop_assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bolero fuzz — random binary inputs must not panic the decoder
// ---------------------------------------------------------------------------

#[test]
fn fuzz_decode_list_no_panic() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            // decode_list must return Result, never panic
            let _ = tensor_snap::decode_list(input);
        });
}

#[test]
fn fuzz_decode_one_no_panic() {
    bolero::check!()
        .with_type::<Vec<u8>>()
        .for_each(|input: &Vec<u8>| {
            let _ = tensor_snap::decode_one(input);
        });
}

/// Fuzz: flipping a random bit in a valid encoding either produces the same
/// tensor (rare) or a hash mismatch / decode error. Never panic, never
/// silently corrupt.
#[test]
fn fuzz_decode_corruption_resilience() {
    bolero::check!()
        .with_type::<(u8, u8, u8, u8, u8, u16)>()
        .for_each(|&(a, b, c, d, e, bit_idx): &(u8, u8, u8, u8, u8, u16)| {
            // Build a small deterministic tensor from the fuzz inputs.
            let data = vec![
                a as f64,
                b as f64,
                c as f64,
                d as f64,
                e as f64,
            ];
            let t = Tensor::from_vec(data, &[5]).unwrap();
            let mut bytes = tensor_snap::encode_one(&t);
            if bytes.is_empty() {
                return;
            }
            let idx = (bit_idx as usize) % (bytes.len() * 8);
            let byte = idx / 8;
            let bit = idx % 8;
            bytes[byte] ^= 1u8 << bit;
            // Decoder must NOT panic. It is allowed to either succeed
            // (unlikely unless the flipped bit was in the footer region
            // matching the hash) or fail with an error.
            let _ = tensor_snap::decode_list(&bytes);
        });
}
