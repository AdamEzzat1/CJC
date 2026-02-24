//! CJC Runtime System
//!
//! Provides the core runtime infrastructure for the CJC programming language:
//! - `Buffer<T>`: Deterministic memory allocation with COW (Copy-On-Write) semantics
//! - `Tensor`: N-dimensional tensor with element-wise ops, matmul, and stable reductions
//! - `GcHeap` / `GcRef`: Simple mark-sweep garbage collector for Layer 3 objects
//! - `Value`: Tagged union for the CJC interpreter
//! - `accumulator`: BinnedAccumulator for order-invariant deterministic summation
//! - `dispatch`: Hybrid summation strategy dispatch (Kahan vs Binned)

// --- Existing standalone modules ---
pub mod accumulator;
pub mod complex;
pub mod dispatch;
pub mod f16;
pub mod quantized;

// --- Shared builtin dispatch (used by both cjc-eval and cjc-mir-exec) ---
pub mod builtins;

// --- Newly extracted modules (from the former monolithic lib.rs) ---
pub mod buffer;
pub mod tensor;
pub mod scratchpad;
pub mod aligned_pool;
mod kernel_bridge;
pub use kernel_bridge::kernel;
pub mod paged_kv;
pub mod gc;
pub mod sparse;
pub mod det_map;
pub mod linalg;
pub mod value;
pub mod error;

// --- Re-exports for backward compatibility ---
// All downstream crates that were doing `use cjc_runtime::Tensor` etc. continue to work.
pub use buffer::Buffer;
pub use tensor::Tensor;
pub use scratchpad::Scratchpad;
pub use aligned_pool::{AlignedPool, AlignedByteSlice};
pub use paged_kv::{KvBlock, PagedKvCache};
pub use gc::{GcRef, GcObject, GcHeap};
pub use sparse::{SparseCsr, SparseCoo};
pub use det_map::{DetMap, murmurhash3, murmurhash3_finalize, value_hash, values_equal_static};
pub use value::{Value, Bf16, FnValue};
pub use error::RuntimeError;

// ---------------------------------------------------------------------------
// Tests — remain here so they can use `super::*` to access all re-exports.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
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

        assert_eq!(buf_a.refcount(), 2);
        assert_eq!(buf_b.refcount(), 2);

        buf_b.set(0, 99).unwrap();

        assert_eq!(buf_a.refcount(), 1);
        assert_eq!(buf_b.refcount(), 1);
        assert_eq!(buf_a.get(0), Some(10));
        assert_eq!(buf_b.get(0), Some(99));
    }

    #[test]
    fn test_buffer_clone_buffer_forces_deep_copy() {
        let buf_a = Buffer::from_vec(vec![1, 2, 3]);
        let buf_b = buf_a.clone_buffer();

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

        assert!(t.get(&[2, 0]).is_err());
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
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.get(&[0, 0]).unwrap(), 58.0);
        assert_eq!(c.get(&[0, 1]).unwrap(), 64.0);
        assert_eq!(c.get(&[1, 0]).unwrap(), 139.0);
        assert_eq!(c.get(&[1, 1]).unwrap(), 154.0);
    }

    #[test]
    fn test_tensor_reshape_shares_buffer() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();

        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(r.get(&[2, 1]).unwrap(), 6.0);

        assert_eq!(t.buffer.refcount(), 2);

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

        assert!(heap.get::<f64>(r1).is_none());
    }

    #[test]
    fn test_gc_collect_frees_unreachable() {
        let mut heap = GcHeap::new(100);
        let r1 = heap.alloc(1i64);
        let r2 = heap.alloc(2i64);
        let _r3 = heap.alloc(3i64);

        assert_eq!(heap.live_count(), 3);

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

        heap.collect(&[]);
        assert_eq!(heap.live_count(), 0);
        assert_eq!(heap.free_list.len(), 3);

        let r4 = heap.alloc(99i64);
        assert!(r4.index < 3);
        assert_eq!(*heap.get::<i64>(r4).unwrap(), 99);
    }

    // -- Stable summation test ----------------------------------------------

    #[test]
    fn test_stable_summation_via_tensor() {
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

    #[test]
    fn test_cow_string_clone_shares() {
        let s = Value::String(Rc::new("hello".into()));
        let s2 = s.clone();
        if let (Value::String(a), Value::String(b)) = (&s, &s2) {
            assert!(Rc::ptr_eq(a, b));
        } else {
            panic!("expected String values");
        }
    }

    #[test]
    fn test_cow_string_display() {
        let s = Value::String(Rc::new("world".into()));
        assert_eq!(format!("{}", s), "world");
    }

    // -- ByteSlice / StrView / U8 tests ------------------------------------

    #[test]
    fn test_byteslice_value_display_utf8() {
        let bs = Value::ByteSlice(Rc::new(b"hello".to_vec()));
        assert_eq!(format!("{}", bs), r#"b"hello""#);
    }

    #[test]
    fn test_byteslice_value_display_hex() {
        let bs = Value::ByteSlice(Rc::new(vec![0xff, 0x00, 0x41]));
        assert_eq!(format!("{}", bs), r#"b"\xff\x00A""#);
    }

    #[test]
    fn test_strview_value_display() {
        let sv = Value::StrView(Rc::new(b"world".to_vec()));
        assert_eq!(format!("{}", sv), "world");
    }

    #[test]
    fn test_u8_value_display() {
        assert_eq!(format!("{}", Value::U8(65)), "65");
    }

    #[test]
    fn test_byteslice_hash_deterministic() {
        let a = Value::ByteSlice(Rc::new(b"hello".to_vec()));
        let b = Value::ByteSlice(Rc::new(b"hello".to_vec()));
        assert_eq!(value_hash(&a), value_hash(&b));
    }

    #[test]
    fn test_byteslice_hash_different_content() {
        let a = Value::ByteSlice(Rc::new(b"hello".to_vec()));
        let b = Value::ByteSlice(Rc::new(b"world".to_vec()));
        assert_ne!(value_hash(&a), value_hash(&b));
    }

    #[test]
    fn test_byteslice_equality() {
        let a = Value::ByteSlice(Rc::new(b"abc".to_vec()));
        let b = Value::ByteSlice(Rc::new(b"abc".to_vec()));
        let c = Value::ByteSlice(Rc::new(b"def".to_vec()));
        assert!(values_equal_static(&a, &b));
        assert!(!values_equal_static(&a, &c));
    }

    #[test]
    fn test_strview_equality() {
        let a = Value::StrView(Rc::new(b"test".to_vec()));
        let b = Value::StrView(Rc::new(b"test".to_vec()));
        assert!(values_equal_static(&a, &b));
    }

    #[test]
    fn test_u8_hash_and_equality() {
        let a = Value::U8(42);
        let b = Value::U8(42);
        let c = Value::U8(99);
        assert_eq!(value_hash(&a), value_hash(&b));
        assert_ne!(value_hash(&a), value_hash(&c));
        assert!(values_equal_static(&a, &b));
        assert!(!values_equal_static(&a, &c));
    }

    #[test]
    fn test_byteslice_clone_shares_rc() {
        let bs = Value::ByteSlice(Rc::new(b"data".to_vec()));
        let bs2 = bs.clone();
        if let (Value::ByteSlice(a), Value::ByteSlice(b)) = (&bs, &bs2) {
            assert!(Rc::ptr_eq(a, b));
        } else {
            panic!("expected ByteSlice values");
        }
    }

    #[test]
    fn test_byteslice_in_detmap() {
        let mut map = DetMap::new();
        let key = Value::ByteSlice(Rc::new(b"token".to_vec()));
        map.insert(key.clone(), Value::Int(1));

        let lookup = Value::ByteSlice(Rc::new(b"token".to_vec()));
        assert!(map.contains_key(&lookup));
        match map.get(&lookup) {
            Some(Value::Int(1)) => {},
            _ => panic!("expected Int(1)"),
        }
    }

    #[test]
    fn test_murmurhash3_byteslice_stability() {
        let h1 = murmurhash3(b"hello");
        let h2 = murmurhash3(b"hello");
        assert_eq!(h1, h2);

        let h3 = murmurhash3(b"");
        let h4 = murmurhash3(b"");
        assert_eq!(h3, h4);

        assert_ne!(murmurhash3(b"hello"), murmurhash3(b"world"));
    }
}
