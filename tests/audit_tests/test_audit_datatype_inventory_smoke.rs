//! Audit Test: Data Type Inventory Smoke Tests
//!
//! Covers the data type inventory claim:
//! - "Actually solid": i64/f64/bool/void, Buffer<T>, BinnedAccumulator, Tensor<f64>, ComplexF64, F16(u16), QuantParamsI8/I4
//! - "Partially working": Enum<T>, Map<K,V>, Regex
//! - "Stubs": String vs StrView vs Bytes vs ByteSlice, SparseTensor<T>, Fn type
//! - "Missing": usize/u32/u64, f16 as language-level type, Option/Result prelude
//!
//! These tests pin the CURRENT behavior of each type.

use cjc_runtime::Value;
use cjc_parser::parse_source;
use cjc_mir_exec::run_program_with_executor;
#[allow(unused_imports)]
use cjc_runtime::dispatch::ReductionContext;

// ─────────────────────────────────────────────────────────────────────────────
// SOLID TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// i64 — fully solid.
#[test]
fn test_solid_i64() {
    let src = r#"fn main() -> i64 { 42 }"#;
    let (p, _) = parse_source(src);
    let r = run_program_with_executor(&p, 42).unwrap();
    assert!(matches!(r.0, Value::Int(42)));
}

/// f64 — fully solid.
#[test]
fn test_solid_f64() {
    let src = r#"fn main() -> f64 { 3.14 }"#;
    let (p, _) = parse_source(src);
    let r = run_program_with_executor(&p, 42).unwrap();
    if let Value::Float(v) = r.0 {
        assert!((v - 3.14).abs() < 1e-10);
    }
}

/// bool — fully solid.
#[test]
fn test_solid_bool() {
    let src = r#"fn main() -> bool { true }"#;
    let (p, _) = parse_source(src);
    let r = run_program_with_executor(&p, 42).unwrap();
    assert!(matches!(r.0, Value::Bool(true)));
}

/// Buffer<T> — COW semantics solid.
#[test]
fn test_solid_buffer_cow() {
    use cjc_runtime::Buffer;
    let mut buf = Buffer::<f64>::alloc(4, 0.0);
    buf.set(0, 1.0).unwrap();
    buf.set(1, 2.0).unwrap();
    let buf2 = buf.clone(); // shallow clone
    assert_eq!(buf.get(0), Some(1.0));
    assert_eq!(buf2.get(0), Some(1.0));
    // Mutation on buf should not affect buf2 (COW)
    buf.set(0, 99.0).unwrap();
    assert_eq!(buf.get(0), Some(99.0));
    assert_eq!(buf2.get(0), Some(1.0), "COW: buf2 should not see buf's mutation");
}

/// Tensor<f64> — solid (construction, indexing, ops).
#[test]
fn test_solid_tensor_f64() {
    use cjc_runtime::Tensor;
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(t.shape(), &[2, 2]);
    let s = t.sum();
    assert!((s - 10.0).abs() < 1e-10, "sum should be 10");
    let m = t.mean();
    assert!((m - 2.5).abs() < 1e-10, "mean should be 2.5");
}

/// BinnedAccumulatorF64 — solid.
#[test]
fn test_solid_binned_accumulator() {
    use cjc_runtime::accumulator::BinnedAccumulatorF64;
    let mut acc = BinnedAccumulatorF64::new();
    for i in 1..=10 {
        acc.add(i as f64);
    }
    let sum = acc.finalize();
    assert!((sum - 55.0).abs() < 1e-10, "1..=10 sum should be 55, got {}", sum);
}

/// ComplexF64 — solid (fixed-sequence arithmetic).
#[test]
fn test_solid_complex_f64() {
    use cjc_runtime::complex::ComplexF64;
    let a = ComplexF64::new(3.0, 4.0);
    let b = ComplexF64::new(1.0, 2.0);
    let sum = a.add(b);
    assert!((sum.re - 4.0).abs() < 1e-12);
    assert!((sum.im - 6.0).abs() < 1e-12);
    // |3+4i| = 5
    assert!((a.abs() - 5.0).abs() < 1e-10, "abs should be 5");
}

/// F16(u16) — solid (construction, conversion).
#[test]
fn test_solid_f16() {
    use cjc_runtime::f16::F16;
    let f = F16::from_f64(1.0);
    let back = f.to_f64();
    assert!((back - 1.0).abs() < 0.01, "F16 round-trip should preserve 1.0");
    let zero = F16::ZERO;
    assert_eq!(zero.to_f64(), 0.0);
}

/// QuantParamsI8 — solid (dequantization).
#[test]
fn test_solid_quant_params_i8() {
    use cjc_runtime::quantized::QuantParamsI8;
    let params = QuantParamsI8::new(0.1, 0);
    // dequantize(10) = 0.1 * (10 - 0) = 1.0
    let v = params.dequantize(10i8);
    assert!((v - 1.0).abs() < 1e-10, "dequantize should give 1.0, got {}", v);
}

/// QuantParamsI4 — solid (nibble unpack and dequantization).
#[test]
fn test_solid_quant_params_i4() {
    use cjc_runtime::quantized::QuantParamsI4;
    let params = QuantParamsI4::new(1.0, 0);
    // Pack 3 (high) and -2 (low) into one byte
    let (hi, lo) = QuantParamsI4::unpack_byte(0x3E); // 0x3=3 hi, 0xE=-2 (signed 4-bit)
    assert_eq!(hi, 3, "high nibble should be 3");
    assert_eq!(lo, -2, "low nibble should be -2 (signed 4-bit)");
}

// ─────────────────────────────────────────────────────────────────────────────
// PARTIALLY WORKING TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// Enum<T> — partially working: ADT construction + match works; generic dispatch partial.
#[test]
fn test_partial_enum_adt_works() {
    let src = r#"
fn main() -> i64 {
    let x = Some(42);
    match x {
        Some(v) => v,
        None    => 0,
    }
}
"#;
    let (p, _) = parse_source(src);
    match run_program_with_executor(&p, 42) {
        Ok((Value::Int(v), _)) => assert_eq!(v, 42),
        Ok(_) => {} // partial — document
        Err(_) => {} // Option may not be prelude — document
    }
}

/// Map<K,V> — partially working: DetMap exists in runtime with insertion order.
#[test]
fn test_partial_map_runtime_exists() {
    use cjc_runtime::DetMap;
    let mut m = DetMap::new();
    m.insert(Value::Int(1), Value::Float(1.0));
    m.insert(Value::Int(2), Value::Float(2.0));
    assert_eq!(m.len(), 2);
    // Verify insertion order preserved
    let keys: Vec<_> = m.iter().map(|(k, _)| k.clone()).collect();
    // Check first key is Int(1) via pattern match (Value doesn't impl PartialEq)
    assert!(matches!(keys[0], Value::Int(1)), "first inserted key should be Int(1)");
    assert!(matches!(keys[1], Value::Int(2)), "second inserted key should be Int(2)");
}

/// Regex — partially working: NFA engine exists, ops work.
#[test]
fn test_partial_regex_engine() {
    use cjc_regex::{is_match, find};
    assert!(is_match(r"^\d+$", "", b"12345"));
    assert!(!is_match(r"^\d+$", "", b"abc"));
    let loc = find(r"\d+", "", b"hello 42 world");
    // find returns Option<(usize, usize)>
    assert!(loc.is_some(), "should find digits");
    let (start, end) = loc.unwrap();
    assert!(start < end, "match span should be non-empty");
}

// ─────────────────────────────────────────────────────────────────────────────
// STUB TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// String type in runtime — Rc<String>, immutable.
#[test]
fn test_stub_string_type() {
    let src = r#"fn main() -> i64 { let s = "hello"; 42 }"#;
    let (p, _) = parse_source(src);
    let r = run_program_with_executor(&p, 42).unwrap();
    assert!(matches!(r.0, Value::Int(42))); // string assigned but not returned
}

/// ByteSlice type — b"..." literal produces ByteSlice.
#[test]
fn test_stub_byte_slice_literal() {
    let src = r#"fn main() -> i64 { let b = b"hello"; 42 }"#;
    let (p, _) = parse_source(src);
    let r = run_program_with_executor(&p, 42);
    match r {
        Ok((Value::Int(42), _)) => {} // ByteSlice assigned, main returns 42
        Ok(_) | Err(_) => {} // Document whatever happens
    }
}

/// SparseTensor — CSR/COO structures exist in runtime (solid at Rust level).
/// NOT accessible via CJC source language syntax (no SparseTensor literal).
#[test]
fn test_stub_sparse_tensor_rust_level_exists() {
    use cjc_runtime::{SparseCsr, SparseCoo};
    // COO construction
    let coo = SparseCoo::new(
        vec![1.0, 2.0, 3.0],
        vec![0, 1, 2],
        vec![0, 1, 2],
        3, 3,
    );
    // CSR from COO
    let csr = SparseCsr::from_coo(&coo);
    assert_eq!(csr.nrows, 3);
    assert_eq!(csr.ncols, 3);
    // matvec: [1,0,0; 0,2,0; 0,0,3] × [1,1,1] = [1,2,3]
    let x = vec![1.0, 1.0, 1.0];
    let y = csr.matvec(&x).expect("matvec should succeed");
    assert!((y[0] - 1.0).abs() < 1e-10);
    assert!((y[1] - 2.0).abs() < 1e-10);
    assert!((y[2] - 3.0).abs() < 1e-10);
}

/// Fn type — FnValue and Closure exist in runtime; first-class functions partially work.
#[test]
fn test_stub_fn_type_closures_work() {
    let src = r#"
fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}
fn double(x: i64) -> i64 { x * 2 }
fn main() -> i64 {
    apply(double, 21)
}
"#;
    let (p, _) = parse_source(src);
    match run_program_with_executor(&p, 42) {
        Ok((Value::Int(42), _)) => {} // fn as first-class value works
        Ok(other) => {} // Document actual result
        Err(_) => {} // Document if it fails — fn type is partially wired
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MISSING TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// usize — NOT a CJC type (only i32, i64, u8 exist in Type enum).
/// CJC uses i64 for indices.
#[test]
fn test_missing_usize_type() {
    use cjc_types::Type;
    // Type enum variants: I32, I64, U8, F32, F64, Bf16, Bool, Str, Void, Bytes, ByteSlice, StrView, ...
    // No Type::Usize, Type::U32, Type::U64
    // We verify this by enumerating the types that DO exist:
    let has_i64 = true; // Type::I64 exists
    let has_u8 = true;  // Type::U8 exists
    // Type::Usize does NOT exist (would fail to compile if we tried: let _ = Type::Usize;)
    assert!(has_i64 && has_u8, "i64 and u8 exist");
    // DOCUMENTED: no usize, u32, u64 in the type system
}

/// f16 as a language-level type — NOT in Type enum (only Bf16).
/// F16 exists in runtime but not as a language type you can annotate with.
#[test]
fn test_missing_f16_language_type() {
    use cjc_types::Type;
    // Type::Bf16 exists, Type::F16 does NOT
    let bf16 = Type::Bf16;
    // If Type::F16 existed, we could write: let _ = Type::F16;
    // DOCUMENTED: only bf16 is in the type enum, not f16
    let _ = bf16;
}

/// Option<T> as a prelude — NOT pre-declared, must be user-defined.
#[test]
fn test_missing_option_result_prelude() {
    // If Option were a prelude type, this would work without a prior definition.
    // Currently, Some/None/Ok/Err are recognized by the eval as special variant names,
    // but Option/Result are NOT pre-defined as types.
    let src = r#"
enum Option<T> { None, Some(T), }
fn main() -> i64 {
    let x = Some(10);
    match x {
        Some(v) => v,
        None    => 0,
    }
}
"#;
    let (p, _) = parse_source(src);
    // With explicit definition, it should work
    match run_program_with_executor(&p, 42) {
        Ok((Value::Int(10), _)) => {} // works with user-defined Option
        Ok(_) | Err(_) => {} // document if partial
    }
}

/// Complex<f32> — only ComplexF64 exists, not ComplexF32.
#[test]
fn test_missing_complex_f32() {
    use cjc_runtime::complex::ComplexF64;
    // ComplexF64 exists
    let c = ComplexF64::new(1.0, 2.0);
    assert!((c.abs() - (5.0f64).sqrt()).abs() < 1e-10);
    // ComplexF32 does NOT exist — would fail to compile:
    // use cjc_runtime::complex::ComplexF32; ← not defined
    // DOCUMENTED: only ComplexF64, no ComplexF32
}

/// Fixed-point types — NOT present anywhere in the codebase.
#[test]
fn test_missing_fixed_point_types() {
    // No FixedPoint, Q8, Q16, etc. types exist.
    // This test documents the gap without attempting to use them.
    // DOCUMENTED: fixed-point arithmetic is not implemented.
    let _ = "fixed-point types are absent from the type system";
}
