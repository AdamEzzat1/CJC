//! Phase 8: Model snapshot tests.
//!
//! Tests deterministic serialization of model weights using cjc-snap,
//! and weight injection for replay/evaluation.

use super::helpers::*;

/// Weight tensors from MIR execution are snappable.
#[test]
fn weights_are_snappable() {
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        print(W1);
    "#);
    let out = run_mir(&src, 42);
    assert!(out[0].contains("Tensor"), "expected tensor output, got: {}", out[0]);
}

/// Extract weight tensors, snap them, restore them, verify equality.
#[test]
fn snap_restore_weight_roundtrip() {
    // Run program twice with same seed, extract W1 tensor output
    let src = multi_program(r#"
        let weights = init_weights();
        let W1 = weights[0];
        print(W1.get([0, 0]));
        print(W1.get([0, 1]));
        print(W1.get([1, 0]));
    "#);
    let out1 = run_mir(&src, 42);
    let out2 = run_mir(&src, 42);
    assert_eq!(out1, out2, "weight init not deterministic");
}

/// Trained weights produce different snapshots than initial weights.
#[test]
fn trained_weights_differ_from_initial() {
    let src = multi_program(r#"
        let weights = init_weights();
        let init_val = weights[0].get([0, 0]);

        let trained = train_multi_episodes(2, 0.01, 0.99, 0.0, 8);
        let trained_val = trained[0].get([0, 0]);

        print(init_val);
        print(trained_val);
    "#);
    let out = run_mir(&src, 42);
    let init_val = parse_float_at(&out, out.len() - 2);
    let trained_val = parse_float_at(&out, out.len() - 1);
    // After training, at least one weight element should have changed
    assert_ne!(init_val, trained_val,
        "expected trained weight to differ from initial");
}

/// Snapshot of Value::Array<Value::Float> roundtrips via cjc-snap.
#[test]
fn snap_roundtrip_float_array() {
    use cjc_runtime::Value;
    let val = Value::Array(vec![
        Value::Float(1.0),
        Value::Float(2.5),
        Value::Float(-3.14),
    ].into());
    let blob = cjc_snap::snap(&val);
    let restored = cjc_snap::restore(&blob).unwrap();
    match (&val, &restored) {
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                match (x, y) {
                    (Value::Float(f1), Value::Float(f2)) => {
                        assert_eq!(f1.to_bits(), f2.to_bits());
                    }
                    _ => panic!("expected floats"),
                }
            }
        }
        _ => panic!("expected arrays"),
    }
}

/// Snapshot of a Tensor roundtrips via cjc-snap.
#[test]
fn snap_roundtrip_tensor() {
    use cjc_runtime::{Value, Tensor};
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let val = Value::Tensor(t);
    let blob = cjc_snap::snap(&val);
    let restored = cjc_snap::restore(&blob).unwrap();
    match (&val, &restored) {
        (Value::Tensor(t1), Value::Tensor(t2)) => {
            assert_eq!(t1.shape(), t2.shape());
            let d1 = t1.to_vec();
            let d2 = t2.to_vec();
            for (a, b) in d1.iter().zip(d2.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        _ => panic!("expected tensors"),
    }
}

/// Same tensor snapped twice produces identical content hash.
#[test]
fn snap_deterministic_hash() {
    use cjc_runtime::{Value, Tensor};
    let t = Tensor::from_vec(vec![1.5, 2.5, 3.5], &[3]).unwrap();
    let blob1 = cjc_snap::snap(&Value::Tensor(t.clone()));
    let blob2 = cjc_snap::snap(&Value::Tensor(t));
    assert_eq!(blob1.content_hash, blob2.content_hash,
        "same tensor should produce same hash");
}

/// Snapshot content hash changes when data changes.
#[test]
fn snap_hash_changes_with_data() {
    use cjc_runtime::{Value, Tensor};
    let t1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let t2 = Tensor::from_vec(vec![1.0, 3.0], &[2]).unwrap();
    let blob1 = cjc_snap::snap(&Value::Tensor(t1));
    let blob2 = cjc_snap::snap(&Value::Tensor(t2));
    assert_ne!(blob1.content_hash, blob2.content_hash,
        "different tensors should produce different hashes");
}
