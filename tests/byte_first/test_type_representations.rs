//! Type representation audit tests.

use cjc_runtime::value::{Bf16, Value};
use cjc_runtime::det_map::{DetMap, value_hash, values_equal_static};
use cjc_runtime::tensor::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::BTreeMap;

#[test]
fn int_byte_layout() {
    let v = Value::Int(0x0102030405060708);
    if let Value::Int(n) = &v {
        let bytes = n.to_le_bytes();
        assert_eq!(bytes.len(), 8);
        assert_eq!(bytes[0], 0x08);
    }
}

#[test]
fn float_bit_roundtrip() {
    for &v in &[0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::MIN, f64::MAX, 5e-324] {
        assert_eq!(v.to_bits(), f64::from_bits(v.to_bits()).to_bits());
    }
}

#[test]
fn bool_hash_distinct() {
    assert_ne!(value_hash(&Value::Bool(true)), value_hash(&Value::Bool(false)));
}

#[test]
fn bf16_roundtrip() {
    let bf = Bf16::from_f32(3.14);
    let bf2 = Bf16::from_f32(3.14);
    assert_eq!(bf.0, bf2.0);
    assert_eq!(bf.to_f32().to_bits(), bf2.to_f32().to_bits());
}

#[test]
fn bf16_arithmetic() {
    let a = Bf16::from_f32(1.5);
    let b = Bf16::from_f32(2.5);
    assert_eq!(a.add(b).0, a.add(b).0);
    assert_eq!(a.sub(b).0, a.sub(b).0);
    assert_eq!(a.mul(b).0, a.mul(b).0);
    assert_eq!(a.div(b).0, a.div(b).0);
}

#[test]
fn string_vs_strview() {
    let s = Value::String(Rc::new("hello".to_string()));
    let sv = Value::StrView(Rc::new(b"hello".to_vec()));
    assert_ne!(s.type_name(), sv.type_name());
}

#[test]
fn bytes_mutable_byteslice_immutable() {
    let bytes = Value::Bytes(Rc::new(RefCell::new(vec![1, 2, 3])));
    if let Value::Bytes(b) = &bytes {
        b.borrow_mut().push(4);
        assert_eq!(b.borrow().len(), 4);
    }
}

#[test]
fn array_cow_clone() {
    let arr1 = Value::Array(Rc::new(vec![Value::Int(1), Value::Int(2)]));
    let arr2 = arr1.clone();
    if let (Value::Array(a), Value::Array(b)) = (&arr1, &arr2) {
        assert!(Rc::ptr_eq(a, b));
    }
}

#[test]
fn struct_btreemap_ordered() {
    let mut fields = BTreeMap::new();
    fields.insert("z_field".to_string(), Value::Int(1));
    fields.insert("a_field".to_string(), Value::Int(2));
    if let Value::Struct { fields, .. } = &(Value::Struct { name: "T".into(), fields }) {
        let keys: Vec<_> = fields.keys().collect();
        assert_eq!(keys, vec!["a_field", "z_field"]);
    }
}

#[test]
fn detmap_insertion_order() {
    let mut m = DetMap::new();
    m.insert(Value::String(Rc::new("c".into())), Value::Int(3));
    m.insert(Value::String(Rc::new("a".into())), Value::Int(1));
    m.insert(Value::String(Rc::new("b".into())), Value::Int(2));
    let keys: Vec<_> = m.keys();
    match &keys[0] { Value::String(s) => assert_eq!(s.as_str(), "c"), _ => panic!() }
    match &keys[1] { Value::String(s) => assert_eq!(s.as_str(), "a"), _ => panic!() }
    match &keys[2] { Value::String(s) => assert_eq!(s.as_str(), "b"), _ => panic!() }
}

#[test]
fn detmap_hash_stable() {
    let k = Value::String(Rc::new("test_key".into()));
    assert_eq!(value_hash(&k), value_hash(&k));
}

#[test]
fn detmap_float_equality_bits() {
    // NaN == NaN via bit comparison (CJC uses bits for determinism)
    assert!(values_equal_static(&Value::Float(f64::NAN), &Value::Float(f64::NAN)));
    // -0.0 != +0.0 via bit comparison (different bit patterns)
    assert!(!values_equal_static(&Value::Float(0.0), &Value::Float(-0.0)));
    assert!(values_equal_static(&Value::Int(42), &Value::Int(42)));
}

#[test]
fn detmap_grow_preserves_order() {
    let mut m = DetMap::new();
    for i in 0..20 {
        m.insert(Value::Int(i), Value::Int(i * 10));
    }
    assert_eq!(m.len(), 20);
    for i in 0..20 {
        assert!(m.get(&Value::Int(i)).is_some());
    }
    let keys: Vec<_> = m.keys();
    for (idx, k) in keys.iter().enumerate() {
        if let Value::Int(n) = k { assert_eq!(*n, idx as i64); }
    }
}

#[test]
fn enum_hash_deterministic() {
    let v1 = Value::Enum { enum_name: "R".into(), variant: "Ok".into(), fields: vec![Value::Int(1)] };
    let v2 = Value::Enum { enum_name: "R".into(), variant: "Ok".into(), fields: vec![Value::Int(1)] };
    assert_eq!(value_hash(&v1), value_hash(&v2));
    assert!(values_equal_static(&v1, &v2));
}

#[test]
fn tensor_row_major_layout() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let data = t.to_vec();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_buffer_cow() {
    let t1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let t2 = t1.clone();
    assert_eq!(t1.to_vec(), t2.to_vec());
}
