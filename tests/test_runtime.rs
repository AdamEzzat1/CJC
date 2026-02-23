// CJC Test Suite — cjc-runtime (17 tests)
// Source: crates/cjc-runtime/src/lib.rs
// These tests are extracted from the inline #[cfg(test)] modules for regression tracking.

use std::rc::Rc;
use cjc_runtime::*;
use cjc_repro::Rng;

// -- Buffer tests -------------------------------------------------------

#[test]
fn test_buffer_alloc_get_set() {
    let mut buf = Buffer::alloc(5, 0.0f64);
    assert_eq!(buf.len(), 5);
    assert_eq!(buf.get(0), Some(0.0));
    assert_eq!(buf.get(4), Some(0.0));
    assert_eq!(buf.get(5), None);

    buf.set(2, 42.0).unwrap();
    assert_eq!(buf.get(2), Some(42.0));

    assert!(buf.set(10, 1.0).is_err());
}

#[test]
fn test_buffer_from_vec() {
    let buf = Buffer::from_vec(vec![1, 2, 3, 4, 5]);
    assert_eq!(buf.len(), 5);
    assert_eq!(buf.get(0), Some(1));
    assert_eq!(buf.get(4), Some(5));
    assert_eq!(buf.as_slice(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_buffer_cow_behavior() {
    let buf_a = Buffer::from_vec(vec![10, 20, 30]);
    let mut buf_b = buf_a.clone();

    // Both share the same backing storage.
    assert_eq!(buf_a.refcount(), 2);
    assert_eq!(buf_b.refcount(), 2);

    // Mutating buf_b triggers COW — buf_a is unaffected.
    buf_b.set(0, 99).unwrap();

    assert_eq!(buf_a.refcount(), 1);
    assert_eq!(buf_b.refcount(), 1);
    assert_eq!(buf_a.get(0), Some(10)); // original unchanged
    assert_eq!(buf_b.get(0), Some(99)); // copy modified
}

#[test]
fn test_buffer_clone_buffer_forces_deep_copy() {
    let buf_a = Buffer::from_vec(vec![1, 2, 3]);
    let buf_b = buf_a.clone_buffer();

    // clone_buffer always creates an independent copy.
    assert_eq!(buf_a.refcount(), 1);
    assert_eq!(buf_b.refcount(), 1);
    assert_eq!(buf_a.as_slice(), buf_b.as_slice());
}

// -- Tensor tests -------------------------------------------------------

#[test]
fn test_tensor_creation_and_indexing() {
    let t = Tensor::zeros(&[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.len(), 6);
    assert_eq!(t.get(&[0, 0]).unwrap(), 0.0);
    assert_eq!(t.get(&[1, 2]).unwrap(), 0.0);

    // Out-of-bounds dimension.
    assert!(t.get(&[2, 0]).is_err());
    // Wrong number of indices.
    assert!(t.get(&[0]).is_err());
}

#[test]
fn test_tensor_from_vec_and_set() {
    let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(t.get(&[0, 2]).unwrap(), 3.0);
    assert_eq!(t.get(&[1, 0]).unwrap(), 4.0);
    assert_eq!(t.get(&[1, 2]).unwrap(), 6.0);

    t.set(&[1, 1], 99.0).unwrap();
    assert_eq!(t.get(&[1, 1]).unwrap(), 99.0);
}

#[test]
fn test_tensor_elementwise_ops() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let sum = a.add(&b).unwrap();
    assert_eq!(sum.to_vec(), vec![6.0, 8.0, 10.0, 12.0]);

    let diff = a.sub(&b).unwrap();
    assert_eq!(diff.to_vec(), vec![-4.0, -4.0, -4.0, -4.0]);

    let prod = a.mul_elem(&b).unwrap();
    assert_eq!(prod.to_vec(), vec![5.0, 12.0, 21.0, 32.0]);

    let quot = b.div_elem(&a).unwrap();
    assert_eq!(quot.to_vec(), vec![5.0, 3.0, 7.0 / 3.0, 2.0]);
}

#[test]
fn test_tensor_matmul_correctness() {
    // [[1, 2], [3, 4]] x [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]).unwrap(), 19.0);
    assert_eq!(c.get(&[0, 1]).unwrap(), 22.0);
    assert_eq!(c.get(&[1, 0]).unwrap(), 43.0);
    assert_eq!(c.get(&[1, 1]).unwrap(), 50.0);
}

#[test]
fn test_tensor_matmul_nonsquare() {
    // (2x3) x (3x2) => (2x2)
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    assert_eq!(c.get(&[0, 0]).unwrap(), 58.0);
    assert_eq!(c.get(&[0, 1]).unwrap(), 64.0);
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(c.get(&[1, 0]).unwrap(), 139.0);
    assert_eq!(c.get(&[1, 1]).unwrap(), 154.0);
}

#[test]
fn test_tensor_reshape_shares_buffer() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let r = t.reshape(&[3, 2]).unwrap();

    assert_eq!(r.shape(), &[3, 2]);
    // Data order is preserved.
    assert_eq!(r.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(r.get(&[2, 1]).unwrap(), 6.0);

    // Underlying buffers are shared.
    assert_eq!(t.buffer.refcount(), 2);

    // Incompatible reshape fails.
    assert!(t.reshape(&[4, 2]).is_err());
}

#[test]
fn test_tensor_sum_and_mean() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    assert!((t.sum() - 10.0).abs() < 1e-12);
    assert!((t.mean() - 2.5).abs() < 1e-12);
}

// -- GC tests -----------------------------------------------------------

#[test]
fn test_gc_alloc_and_read() {
    let mut heap = GcHeap::new(100);
    let r1 = heap.alloc(42i64);
    let r2 = heap.alloc("hello".to_string());

    assert_eq!(heap.live_count(), 2);
    assert_eq!(*heap.get::<i64>(r1).unwrap(), 42);
    assert_eq!(heap.get::<String>(r2).unwrap().as_str(), "hello");

    // Wrong type returns None.
    assert!(heap.get::<f64>(r1).is_none());
}

#[test]
fn test_gc_collect_frees_unreachable() {
    let mut heap = GcHeap::new(100);
    let r1 = heap.alloc(1i64);
    let r2 = heap.alloc(2i64);
    let _r3 = heap.alloc(3i64);

    assert_eq!(heap.live_count(), 3);

    // Only r1 and r2 are roots — r3 should be collected.
    heap.collect(&[r1, r2]);

    assert_eq!(heap.live_count(), 2);
    assert_eq!(*heap.get::<i64>(r1).unwrap(), 1);
    assert_eq!(*heap.get::<i64>(r2).unwrap(), 2);
}

#[test]
fn test_gc_slot_reuse() {
    let mut heap = GcHeap::new(100);
    let _r1 = heap.alloc(1i64);
    let _r2 = heap.alloc(2i64);
    let _r3 = heap.alloc(3i64);

    // Collect with no roots — frees everything.
    heap.collect(&[]);
    assert_eq!(heap.live_count(), 0);
    assert_eq!(heap.free_list.len(), 3);

    // New allocation reuses a freed slot.
    let r4 = heap.alloc(99i64);
    assert!(r4.index < 3); // reused one of the first 3 slots
    assert_eq!(*heap.get::<i64>(r4).unwrap(), 99);
}

// -- Stable summation test ----------------------------------------------

#[test]
fn test_stable_summation_via_tensor() {
    // Summing many small values where naive floating-point addition would
    // accumulate error. Kahan summation keeps the result accurate.
    let n = 100_000;
    let data: Vec<f64> = (0..n).map(|_| 0.00001).collect();
    let t = Tensor::from_vec(data, &[n]).unwrap();
    let result = t.sum();
    let expected = 0.00001 * n as f64;
    assert!(
        (result - expected).abs() < 1e-10,
        "Kahan sum drift: expected {expected}, got {result}"
    );
}

// -- Tensor randn determinism test --------------------------------------

#[test]
fn test_tensor_randn_deterministic() {
    let mut rng1 = Rng::seeded(42);
    let mut rng2 = Rng::seeded(42);

    let t1 = Tensor::randn(&[3, 4], &mut rng1);
    let t2 = Tensor::randn(&[3, 4], &mut rng2);

    assert_eq!(t1.to_vec(), t2.to_vec());
}

// -- Value display test -------------------------------------------------

#[test]
fn test_value_display() {
    assert_eq!(format!("{}", Value::Int(42)), "42");
    assert_eq!(format!("{}", Value::Bool(true)), "true");
    assert_eq!(format!("{}", Value::Void), "void");
    assert_eq!(format!("{}", Value::String(Rc::new("hi".into()))), "hi");
}
