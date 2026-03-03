//! LH-11: CJC Snap — Content-Addressable Serialization
//!
//! Tests for the `snap` and `restore` functions, SHA-256 hashing,
//! deterministic round-trip encoding, and hash mismatch detection.

use std::collections::HashMap;
use std::rc::Rc;
use cjc_runtime::Value;
use cjc_snap::{snap, restore, snap_encode, snap_decode, sha256, SnapError};

// ---------------------------------------------------------------------------
// 1. SHA-256 correctness
// ---------------------------------------------------------------------------

#[test]
fn test_sha256_empty() {
    let hash = sha256(b"");
    let hex = cjc_snap::hash::hex_string(&hash);
    assert_eq!(
        hex,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_sha256_abc() {
    let hash = sha256(b"abc");
    let hex = cjc_snap::hash::hex_string(&hash);
    assert_eq!(
        hex,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn test_sha256_deterministic() {
    let h1 = sha256(b"cjc snap determinism");
    let h2 = sha256(b"cjc snap determinism");
    assert_eq!(h1, h2);
}

// ---------------------------------------------------------------------------
// 2. Round-trip: snap(v) → restore → v
// ---------------------------------------------------------------------------

#[test]
fn test_roundtrip_int() {
    let v = Value::Int(42);
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "42");
}

#[test]
fn test_roundtrip_float() {
    let v = Value::Float(3.14);
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "3.14");
}

#[test]
fn test_roundtrip_bool() {
    let v = Value::Bool(true);
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "true");
}

#[test]
fn test_roundtrip_string() {
    let v = Value::String(Rc::new("hello, CJC!".to_string()));
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "hello, CJC!");
}

#[test]
fn test_roundtrip_void() {
    let v = Value::Void;
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "void");
}

#[test]
fn test_roundtrip_array() {
    let v = Value::Array(Rc::new(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "[1, 2, 3]");
}

#[test]
fn test_roundtrip_tuple() {
    let v = Value::Tuple(Rc::new(vec![Value::Int(1), Value::Float(2.5)]));
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    assert_eq!(format!("{restored}"), "(1, 2.5)");
}

#[test]
fn test_roundtrip_struct() {
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::Float(1.0));
    fields.insert("y".to_string(), Value::Float(2.0));
    let v = Value::Struct {
        name: "Point".to_string(),
        fields,
    };
    let blob = snap(&v);
    let restored = restore(&blob).unwrap();
    // Struct restored with correct fields
    let s = format!("{restored}");
    assert!(s.contains("Point"));
    assert!(s.contains("x:"));
    assert!(s.contains("y:"));
}

// ---------------------------------------------------------------------------
// 3. Content-addressable: same value → same hash
// ---------------------------------------------------------------------------

#[test]
fn test_content_addressable() {
    let v1 = Value::Int(12345);
    let v2 = Value::Int(12345);
    let blob1 = snap(&v1);
    let blob2 = snap(&v2);
    assert_eq!(blob1.content_hash, blob2.content_hash);
    assert_eq!(blob1.data, blob2.data);
}

#[test]
fn test_different_values_different_hash() {
    let blob1 = snap(&Value::Int(1));
    let blob2 = snap(&Value::Int(2));
    assert_ne!(blob1.content_hash, blob2.content_hash);
}

// ---------------------------------------------------------------------------
// 4. Struct field order determinism
// ---------------------------------------------------------------------------

#[test]
fn test_struct_deterministic_encoding() {
    // Create structs with fields inserted in different order — encoding must match.
    let mut f1 = HashMap::new();
    f1.insert("b".to_string(), Value::Int(2));
    f1.insert("a".to_string(), Value::Int(1));
    let v1 = Value::Struct { name: "S".to_string(), fields: f1 };

    let mut f2 = HashMap::new();
    f2.insert("a".to_string(), Value::Int(1));
    f2.insert("b".to_string(), Value::Int(2));
    let v2 = Value::Struct { name: "S".to_string(), fields: f2 };

    let enc1 = snap_encode(&v1);
    let enc2 = snap_encode(&v2);
    assert_eq!(enc1, enc2, "struct encoding must be deterministic regardless of field insertion order");
}

// ---------------------------------------------------------------------------
// 5. Hash mismatch detection
// ---------------------------------------------------------------------------

#[test]
fn test_hash_mismatch_rejected() {
    let blob = snap(&Value::Int(42));
    // Tamper with the hash
    let mut tampered = blob;
    tampered.content_hash[0] ^= 0xFF;
    let result = restore(&tampered);
    assert!(result.is_err());
    match result.unwrap_err() {
        SnapError::HashMismatch { .. } => {},
        other => panic!("expected HashMismatch, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 6. NaN canonicalization
// ---------------------------------------------------------------------------

#[test]
fn test_nan_canonicalization() {
    let v1 = Value::Float(f64::NAN);
    let v2 = Value::Float(f64::NAN);
    let blob1 = snap(&v1);
    let blob2 = snap(&v2);
    assert_eq!(blob1.content_hash, blob2.content_hash, "all NaN values should produce the same encoding");
}

// ---------------------------------------------------------------------------
// 7. Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_decode_empty() {
    let result = snap_decode(&[]);
    assert!(result.is_err());
}

#[test]
fn test_decode_invalid_tag() {
    let result = snap_decode(&[0xFF]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// 8. Parity — encode in eval, decode produces same value
// ---------------------------------------------------------------------------

#[test]
fn test_snap_from_eval_program() {
    // Create a CJC program that produces a struct, then snap it.
    let src = r#"
record Point {
    x: f64,
    y: f64
}
fn main() -> Any {
    Point { x: 3.0, y: 4.0 }
}
"#;
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors());
    let mut interp = cjc_eval::Interpreter::new(42);
    let val = interp.exec(&program).unwrap();

    // Snap and restore
    let blob = snap(&val);
    let restored = restore(&blob).unwrap();
    let orig_str = format!("{val}");
    let rest_str = format!("{restored}");
    // Both should represent the same Point
    assert!(orig_str.contains("Point"), "original: {orig_str}");
    assert!(rest_str.contains("Point"), "restored: {rest_str}");
}

// ===========================================================================
// CJC Snap Builtin Integration Tests (new)
// ===========================================================================

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    for d in &diags.diagnostics {
        eprintln!("  parse diag: {d:?}");
    }
    assert!(!diags.has_errors(), "unexpected parse errors");
    program
}

fn eval_str(src: &str) -> String {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let val = interp.exec(&program).expect("eval failed");
    format!("{val}")
}

fn mir_str(src: &str) -> String {
    let program = parse(src);
    let (val, _exec) =
        cjc_mir_exec::run_program_with_executor(&program, 42).expect("MIR exec failed");
    format!("{val}")
}

fn eval_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("eval failed");
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let program = parse(src);
    let (_val, exec) =
        cjc_mir_exec::run_program_with_executor(&program, 42).expect("MIR exec failed");
    exec.output.clone()
}

// ---------------------------------------------------------------------------
// 9. snap() builtin from CJC programs
// ---------------------------------------------------------------------------

#[test]
fn test_snap_builtin_returns_struct() {
    let result = eval_str(r#"
fn main() -> Any {
    let blob: Any = snap(42);
    blob.size
}
"#);
    // Size should be 9 (1 tag byte + 8 bytes for i64)
    assert_eq!(result, "9");
}

#[test]
fn test_snap_builtin_hash_is_64_chars() {
    let result = eval_str(r#"
fn main() -> Any {
    let blob: Any = snap(42);
    blob.hash
}
"#);
    assert_eq!(result.len(), 64, "hash should be 64-char hex string, got: {result}");
}

#[test]
fn test_snap_builtin_mir() {
    let result = mir_str(r#"
fn main() -> Any {
    let blob: Any = snap(42);
    blob.size
}
"#);
    assert_eq!(result, "9");
}

// ---------------------------------------------------------------------------
// 10. restore() builtin round-trip from CJC
// ---------------------------------------------------------------------------

#[test]
fn test_restore_roundtrip_int() {
    let result = eval_str(r#"
fn main() -> i64 {
    let blob: Any = snap(42);
    let val: i64 = restore(blob);
    val
}
"#);
    assert_eq!(result, "42");
}

#[test]
fn test_restore_roundtrip_string() {
    let result = eval_str(r#"
fn main() -> Any {
    let blob: Any = snap("hello CJC");
    restore(blob)
}
"#);
    assert_eq!(result, "hello CJC");
}

#[test]
fn test_restore_roundtrip_mir() {
    let result = mir_str(r#"
fn main() -> i64 {
    let blob: Any = snap(99);
    let val: i64 = restore(blob);
    val
}
"#);
    assert_eq!(result, "99");
}

// ---------------------------------------------------------------------------
// 11. snap_hash()
// ---------------------------------------------------------------------------

#[test]
fn test_snap_hash_deterministic() {
    let src = r#"
fn main() -> Any {
    let h1: Any = snap_hash(42);
    let h2: Any = snap_hash(42);
    if h1 == h2 {
        return 1;
    }
    0
}
"#;
    assert_eq!(eval_str(src), "1");
}

#[test]
fn test_snap_hash_different_values() {
    let src = r#"
fn main() -> Any {
    let h1: Any = snap_hash(1);
    let h2: Any = snap_hash(2);
    if h1 == h2 {
        return 0;
    }
    1
}
"#;
    assert_eq!(eval_str(src), "1");
}

// ---------------------------------------------------------------------------
// 12. snap_save / snap_load (file persistence)
// ---------------------------------------------------------------------------

#[test]
fn test_snap_save_load_int() {
    let result = eval_str(r#"
fn main() -> i64 {
    snap_save(42, "__test_eval_int.snap");
    let loaded: i64 = snap_load("__test_eval_int.snap");
    loaded
}
"#);
    assert_eq!(result, "42");
    let _ = std::fs::remove_file("__test_eval_int.snap");
}

#[test]
fn test_snap_save_load_struct() {
    let result = eval_str(r#"
struct Point {
    x: f64,
    y: f64
}
fn main() -> f64 {
    let p: Any = Point { x: 3.0, y: 4.0 };
    snap_save(p, "__test_eval_struct.snap");
    let loaded: Any = snap_load("__test_eval_struct.snap");
    loaded.x + loaded.y
}
"#);
    assert_eq!(result, "7");
    let _ = std::fs::remove_file("__test_eval_struct.snap");
}

#[test]
fn test_snap_save_load_mir() {
    let result = mir_str(r#"
fn main() -> i64 {
    snap_save(99, "__test_mir_int.snap");
    let loaded: i64 = snap_load("__test_mir_int.snap");
    loaded
}
"#);
    assert_eq!(result, "99");
    let _ = std::fs::remove_file("__test_mir_int.snap");
}

#[test]
fn test_snap_load_missing_file() {
    let program = parse(r#"
fn main() -> Any {
    snap_load("__nonexistent_12345.snap")
}
"#);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err(), "snap_load on missing file should fail");
}

// ---------------------------------------------------------------------------
// 13. snap_to_json()
// ---------------------------------------------------------------------------

#[test]
fn test_snap_to_json_int() {
    let result = eval_str(r#"
fn main() -> Any {
    snap_to_json(42)
}
"#);
    assert_eq!(result, "42");
}

#[test]
fn test_snap_to_json_string() {
    let result = eval_str(r#"
fn main() -> Any {
    snap_to_json("hello")
}
"#);
    assert_eq!(result, "\"hello\"");
}

#[test]
fn test_snap_to_json_array() {
    let result = eval_str(r#"
fn main() -> Any {
    snap_to_json([1, 2, 3])
}
"#);
    assert_eq!(result, "[1,2,3]");
}

#[test]
fn test_snap_to_json_struct() {
    let result = eval_str(r#"
struct Pt { x: f64, y: f64 }
fn main() -> Any {
    snap_to_json(Pt { x: 1.0, y: 2.0 })
}
"#);
    assert!(result.contains("\"__type\":\"Struct\""));
    assert!(result.contains("\"name\":\"Pt\""));
    assert!(result.contains("\"x\":"));
    assert!(result.contains("\"y\":"));
}

#[test]
fn test_snap_to_json_mir() {
    let result = mir_str(r#"
fn main() -> Any {
    snap_to_json(42)
}
"#);
    assert_eq!(result, "42");
}

// ---------------------------------------------------------------------------
// 14. memo_call() memoization
// ---------------------------------------------------------------------------

#[test]
fn test_memo_call_returns_correct_result() {
    let result = eval_str(r#"
fn square(x: i64) -> i64 {
    x * x
}
fn main() -> i64 {
    memo_call("square", 7)
}
"#);
    assert_eq!(result, "49");
}

#[test]
fn test_memo_call_caches_result() {
    let output = eval_output(r#"
fn expensive(x: i64) -> i64 {
    print("called");
    x * x
}
fn main() -> i64 {
    let a: i64 = memo_call("expensive", 5);
    let b: i64 = memo_call("expensive", 5);
    a + b
}
"#);
    // "called" should appear exactly once (second call is cached)
    let call_count = output.iter().filter(|l| l.as_str() == "called").count();
    assert_eq!(call_count, 1, "expected 1 call, got {call_count}: {output:?}");
}

#[test]
fn test_memo_call_different_args_not_cached() {
    let output = eval_output(r#"
fn compute(x: i64) -> i64 {
    print("called");
    x + 1
}
fn main() -> i64 {
    let a: i64 = memo_call("compute", 1);
    let b: i64 = memo_call("compute", 2);
    a + b
}
"#);
    let call_count = output.iter().filter(|l| l.as_str() == "called").count();
    assert_eq!(call_count, 2, "different args should not share cache");
}

#[test]
fn test_memo_call_mir() {
    let output = mir_output(r#"
fn expensive(x: i64) -> i64 {
    print("called");
    x * x
}
fn main() -> i64 {
    let a: i64 = memo_call("expensive", 5);
    let b: i64 = memo_call("expensive", 5);
    a + b
}
"#);
    let call_count = output.iter().filter(|l| l.as_str() == "called").count();
    assert_eq!(call_count, 1, "MIR memo_call should cache: {output:?}");
}

// ---------------------------------------------------------------------------
// 15. Parity — eval and MIR produce same results
// ---------------------------------------------------------------------------

#[test]
fn test_parity_snap_hash() {
    let src = r#"
fn main() -> Any {
    snap_hash(42)
}
"#;
    assert_eq!(eval_str(src), mir_str(src), "snap_hash parity");
}

#[test]
fn test_parity_snap_to_json() {
    let src = r#"
fn main() -> Any {
    snap_to_json([1, 2, 3])
}
"#;
    assert_eq!(eval_str(src), mir_str(src), "snap_to_json parity");
}

#[test]
fn test_parity_memo_call() {
    let src = r#"
fn double(x: i64) -> i64 {
    x * 2
}
fn main() -> i64 {
    memo_call("double", 21)
}
"#;
    assert_eq!(eval_str(src), mir_str(src), "memo_call parity");
}

// ---------------------------------------------------------------------------
// 16. Tensor snap
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_snap_roundtrip() {
    // Rust-level tensor round-trip through snap
    use cjc_runtime::Tensor;
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let blob = snap(&Value::Tensor(t));
    let restored = restore(&blob).unwrap();
    match restored {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        _ => panic!("expected Tensor"),
    }
}

#[test]
fn test_tensor_content_addressable() {
    use cjc_runtime::Tensor;
    let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let t2 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let blob1 = snap(&Value::Tensor(t1));
    let blob2 = snap(&Value::Tensor(t2));
    assert_eq!(blob1.content_hash, blob2.content_hash, "same tensor data should hash equal");
}

#[test]
fn test_tensor_json() {
    use cjc_runtime::Tensor;
    let t = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let json = cjc_snap::snap_to_json(&Value::Tensor(t)).unwrap();
    assert!(json.contains("\"__type\":\"Tensor\""));
    assert!(json.contains("\"shape\":[2]"));
    assert!(json.contains("\"data\":[1.0,2.0]"));
}

// ---------------------------------------------------------------------------
// 17. Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_snap_load_bad_path_error() {
    let program = parse(r#"
fn main() -> Any {
    snap_load("__definitely_missing_file.snap")
}
"#);
    let mut interp = cjc_eval::Interpreter::new(42);
    let result = interp.exec(&program);
    assert!(result.is_err());
}

#[test]
fn test_restore_bad_blob() {
    // Construct a SnapBlob with wrong hash from CJC
    // We can't easily tamper from CJC, so test at Rust level
    let blob = snap(&Value::Int(42));
    let mut tampered = blob.clone();
    tampered.data = snap_encode(&Value::Int(999)); // data doesn't match hash
    let result = restore(&tampered);
    assert!(result.is_err());
}

#[test]
fn test_is_snappable() {
    assert!(cjc_snap::is_snappable(&Value::Int(42)));
    assert!(cjc_snap::is_snappable(&Value::String(Rc::new("hello".into()))));
    assert!(cjc_snap::is_snappable(&Value::Array(Rc::new(vec![Value::Int(1)]))));
    // Void is snappable
    assert!(cjc_snap::is_snappable(&Value::Void));
}
