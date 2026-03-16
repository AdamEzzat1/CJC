//! Serialization roundtrip and byte-level invariant tests.

use cjc_runtime::Value;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Bf16;
use std::rc::Rc;
use std::collections::BTreeMap;

fn roundtrip(v: &Value) -> Value {
    let bytes = cjc_snap::snap_encode(v);
    cjc_snap::snap_decode(&bytes).expect("decode failed")
}

#[test]
fn roundtrip_void() {
    assert!(matches!(roundtrip(&Value::Void), Value::Void));
}

#[test]
fn roundtrip_int_boundaries() {
    for &n in &[0i64, 1, -1, i64::MAX, i64::MIN, 42] {
        match roundtrip(&Value::Int(n)) {
            Value::Int(m) => assert_eq!(n, m),
            _ => panic!("expected Int"),
        }
    }
}

#[test]
fn roundtrip_float_special() {
    for &f in &[0.0f64, -0.0, 1.0, f64::INFINITY, f64::NEG_INFINITY, f64::MAX, 5e-324] {
        match roundtrip(&Value::Float(f)) {
            Value::Float(g) => assert_eq!(f.to_bits(), g.to_bits(), "failed for {f}"),
            _ => panic!("expected Float"),
        }
    }
}

#[test]
fn roundtrip_nan_canonical() {
    let b1 = cjc_snap::snap_encode(&Value::Float(f64::NAN));
    let b2 = cjc_snap::snap_encode(&Value::Float(f64::from_bits(0xFFF8_0000_0000_0000)));
    assert_eq!(b1, b2, "NaN should be canonicalized");
}

#[test]
fn roundtrip_bool() {
    for &b in &[true, false] {
        match roundtrip(&Value::Bool(b)) { Value::Bool(g) => assert_eq!(b, g), _ => panic!() }
    }
}

#[test]
fn roundtrip_string() {
    for s in ["", "hello", "unicode: 日本語"] {
        let v = Value::String(Rc::new(s.to_string()));
        match roundtrip(&v) {
            Value::String(g) => assert_eq!(s, g.as_str()),
            _ => panic!("expected String"),
        }
    }
}

#[test]
fn roundtrip_bytes() {
    let data = vec![0u8, 1, 255, 128];
    let v = Value::Bytes(Rc::new(std::cell::RefCell::new(data.clone())));
    match roundtrip(&v) {
        Value::Bytes(g) => assert_eq!(data, *g.borrow()),
        _ => panic!("expected Bytes"),
    }
}

#[test]
fn roundtrip_u8() {
    for &b in &[0u8, 128, 255] {
        match roundtrip(&Value::U8(b)) { Value::U8(g) => assert_eq!(b, g), _ => panic!() }
    }
}

#[test]
fn roundtrip_bf16() {
    let bf = Bf16::from_f32(3.14);
    match roundtrip(&Value::Bf16(bf)) { Value::Bf16(g) => assert_eq!(bf.0, g.0), _ => panic!() }
}

#[test]
fn roundtrip_array() {
    let arr = Value::Array(Rc::new(vec![Value::Int(1), Value::Float(2.5), Value::Bool(true)]));
    match roundtrip(&arr) { Value::Array(a) => assert_eq!(a.len(), 3), _ => panic!() }
}

#[test]
fn roundtrip_tuple() {
    let t = Value::Tuple(Rc::new(vec![Value::Int(42), Value::String(Rc::new("hi".into()))]));
    match roundtrip(&t) { Value::Tuple(a) => assert_eq!(a.len(), 2), _ => panic!() }
}

#[test]
fn roundtrip_struct() {
    let mut fields = BTreeMap::new();
    fields.insert("x".to_string(), Value::Float(1.5));
    fields.insert("y".to_string(), Value::Float(2.5));
    let s = Value::Struct { name: "Point".into(), fields };
    match roundtrip(&s) {
        Value::Struct { name, fields } => { assert_eq!(name, "Point"); assert_eq!(fields.len(), 2); }
        _ => panic!(),
    }
}

#[test]
fn roundtrip_enum() {
    let v = Value::Enum { enum_name: "Option".into(), variant: "Some".into(), fields: vec![Value::Int(42)] };
    match roundtrip(&v) {
        Value::Enum { enum_name, variant, fields } => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "Some");
            assert_eq!(fields.len(), 1);
        }
        _ => panic!(),
    }
}

#[test]
fn roundtrip_tensor() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    match roundtrip(&Value::Tensor(t)) {
        Value::Tensor(t2) => { assert_eq!(t2.shape(), &[2, 3]); assert_eq!(t2.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); }
        _ => panic!(),
    }
}

#[test]
fn snap_encoding_deterministic() {
    let v = Value::Array(Rc::new(vec![Value::Float(1.0), Value::String(Rc::new("test".into()))]));
    assert_eq!(cjc_snap::snap_encode(&v), cjc_snap::snap_encode(&v));
}

#[test]
fn snap_struct_fields_sorted() {
    let mut f1 = BTreeMap::new();
    f1.insert("z".into(), Value::Int(1));
    f1.insert("a".into(), Value::Int(2));
    let s1 = Value::Struct { name: "T".into(), fields: f1 };
    let mut f2 = BTreeMap::new();
    f2.insert("a".into(), Value::Int(2));
    f2.insert("z".into(), Value::Int(1));
    let s2 = Value::Struct { name: "T".into(), fields: f2 };
    assert_eq!(cjc_snap::snap_encode(&s1), cjc_snap::snap_encode(&s2));
}

#[test]
fn content_hash_deterministic() {
    let v = Value::Array(Rc::new(vec![Value::Float(3.14), Value::Int(42)]));
    let b1 = cjc_snap::snap(&v);
    let b2 = cjc_snap::snap(&v);
    assert_eq!(b1.content_hash, b2.content_hash);
}

#[test]
fn content_hash_differs() {
    let b1 = cjc_snap::snap(&Value::Int(1));
    let b2 = cjc_snap::snap(&Value::Int(2));
    assert_ne!(b1.content_hash, b2.content_hash);
}
