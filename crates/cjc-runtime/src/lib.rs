//! CJC Runtime System
//!
//! This crate provides the core runtime infrastructure for the CJC deterministic
//! numerical programming language. It is the largest crate in the workspace and
//! underpins both the AST tree-walk interpreter (`cjc-eval`) and the MIR register
//! machine executor (`cjc-mir-exec`).
//!
//! # Core Abstractions
//!
//! - [`Buffer<T>`] -- Deterministic memory allocation with COW (copy-on-write)
//!   semantics. Cloning is O(1); mutation triggers a deep copy only when shared.
//! - [`Tensor`] -- N-dimensional tensor backed by `Buffer<f64>`. Supports
//!   element-wise arithmetic (SIMD-accelerated), matrix multiplication (tiled +
//!   parallel), and numerically-stable reductions via [`BinnedAccumulatorF64`].
//! - [`Value`] -- The universal tagged-union value type that flows through both
//!   interpreters. Covers scalars, strings, tensors, closures, structs, enums,
//!   and opaque type-erased objects (AD graphs, tidy views, quantum states).
//! - [`RuntimeError`] -- Error type for all fallible runtime operations.
//!
//! # Determinism Guarantees
//!
//! - All floating-point reductions use Kahan or [`BinnedAccumulatorF64`] summation.
//! - Ordered containers only ([`BTreeMap`]/[`BTreeSet`]) -- no `HashMap`/`HashSet`.
//! - [`DetMap`] provides a deterministic hash map with [`murmurhash3`] hashing.
//! - SIMD kernels avoid hardware FMA for bit-identical cross-platform results.
//! - RNG is SplitMix64 with explicit seed threading (`cjc-repro`).
//!
//! # Memory Model
//!
//! - **NoGC tier:** [`Buffer<T>`], [`Tensor`], [`AlignedByteSlice`] -- zero GC
//!   overhead, COW semantics.
//! - **GC tier:** [`GcHeap`] / [`GcRef`] -- RC-backed object slab for class
//!   instances.
//! - **Arena tier:** [`FrameArena`] / [`ArenaStore`] -- bump allocation per
//!   function frame for non-escaping temporaries.
//!
//! # Module Organization
//!
//! | Layer | Modules |
//! |-------|---------|
//! | Core types | [`value`], [`error`], [`buffer`], [`tensor`], [`tensor_dtype`] |
//! | Builtins | [`builtins`] -- shared stateless dispatch for both executors |
//! | Accumulation | [`accumulator`], [`dispatch`] |
//! | Linear algebra | [`linalg`], [`sparse`], [`sparse_solvers`], [`sparse_eigen`] |
//! | Statistics | [`stats`], [`distributions`], [`hypothesis`] |
//! | Data | [`json`], [`datetime`], [`window`], [`timeseries`] |
//! | ML / NN | [`ml`], [`fft`], [`clustering`], [`optimize`], [`interpolate`] |
//! | Memory | [`gc`], [`object_slab`], [`frame_arena`], [`binned_alloc`], [`aligned_pool`] |
//! | SIMD / Perf | [`tensor_simd`], [`tensor_tiled`], [`tensor_pool`] |
//!
//! [`BinnedAccumulatorF64`]: accumulator::BinnedAccumulatorF64
//! [`BTreeMap`]: std::collections::BTreeMap
//! [`BTreeSet`]: std::collections::BTreeSet

// --- Core standalone modules ---

/// BinnedAccumulator for order-invariant deterministic floating-point summation.
pub mod accumulator;
/// Complex number arithmetic with deterministic fixed-sequence operations.
pub mod complex;
/// Hybrid summation strategy dispatch (Kahan vs Binned) based on execution context.
pub mod dispatch;
/// IEEE 754 half-precision (f16) floating-point type.
pub mod f16;
/// Quantized tensor storage (4-bit, 8-bit) for memory-efficient inference.
pub mod quantized;

// --- Shared builtin dispatch (used by both cjc-eval and cjc-mir-exec) ---

/// Stateless builtin function dispatch shared by both interpreters.
///
/// Every builtin registered here is callable from CJC source code. Functions
/// that require interpreter state (print, GC, clock, RNG) stay in the
/// individual executors.
pub mod builtins;

// --- Core data structures ---

/// COW (copy-on-write) buffer -- the memory primitive under [`Tensor`].
pub mod buffer;
/// N-dimensional tensor with element-wise, reduction, linalg, and NN operations.
pub mod tensor;
/// Deterministic binary serialization for tensors and tensor lists.
pub mod tensor_snap;
/// Pre-allocated KV-cache scratchpad for zero-allocation transformer inference.
pub mod scratchpad;
/// 16-byte-aligned memory pool for SIMD-friendly byte buffers.
pub mod aligned_pool;
/// Internal bridge to compiled SIMD/tiled kernel functions.
mod kernel_bridge;
pub use kernel_bridge::kernel;
/// Block-paged KV-cache (vLLM-style) for efficient autoregressive decoding.
pub mod paged_kv;
/// Size-class binned allocator for deterministic memory management.
pub mod binned_alloc;
/// Bump-arena per function frame for non-escaping temporaries.
pub mod frame_arena;
/// RC-backed object slab for class instances (replaces mark-sweep GC).
pub mod object_slab;
/// GC heap abstraction wrapping the RC-backed object slab.
pub mod gc;
/// Sparse matrix types: CSR and COO representations.
pub mod sparse;
/// Direct sparse solvers (LU factorization, triangular solve).
pub mod sparse_solvers;
/// L2-cache-friendly tiled matrix multiplication engine.
pub mod tensor_tiled;
/// AVX2 SIMD kernels for element-wise and unary tensor operations.
pub mod tensor_simd;
/// Tensor memory pool for reducing allocation pressure in hot loops.
pub mod tensor_pool;
/// Deterministic hash map using MurmurHash3 -- iteration order is fixed.
pub mod det_map;
/// Dense linear algebra: determinant, solve, eigenvalues, SVD, QR, LU.
pub mod linalg;
/// The universal [`Value`] tagged union and supporting types ([`Bf16`], [`FnValue`]).
pub mod value;
/// [`RuntimeError`] enum for all fallible runtime operations.
pub mod error;
/// Library registry for module-system symbol lookup.
pub mod lib_registry;
/// JSON parse/stringify builtins for CJC values.
pub mod json;
/// Pure-arithmetic datetime manipulation (epoch-based, no system clock).
pub mod datetime;
/// Rolling window aggregations (sum, mean, min, max).
pub mod window;
/// Descriptive and inferential statistics functions.
pub mod stats;
/// Probability distribution functions (CDF, PDF, PPF) for Normal, t, chi2, F.
pub mod distributions;
/// Hypothesis testing: t-test, chi-squared test, paired t-test.
pub mod hypothesis;
/// Machine learning loss functions and optimizer state types.
pub mod ml;
/// Fast Fourier Transform (radix-2 Cooley-Tukey).
pub mod fft;
/// Time-series stationarity tests (ADF).
pub mod stationarity;
/// ODE solver primitives (Euler, RK4 step functions).
pub mod ode;
/// Sparse eigenvalue solvers (Lanczos, Arnoldi).
pub mod sparse_eigen;
/// Interpolation primitives (linear, cubic spline).
pub mod interpolate;
/// Numerical optimization (gradient descent, L-BFGS, Nelder-Mead).
pub mod optimize;
/// Clustering algorithms (k-means, DBSCAN).
pub mod clustering;
/// Typed tensor storage: [`DType`] enum and byte-first [`TypedStorage`].
pub mod tensor_dtype;
/// Time-series analysis utilities (autocorrelation, differencing).
pub mod timeseries;
/// Numerical integration (trapezoidal, Simpson's rule).
pub mod integrate;
/// Numerical differentiation (finite differences).
pub mod differentiate;
/// Deterministic profile counters (Tier 2 of Chess RL v2.3). Write-only
/// timing sink that does not perturb program state, RNG, or weight hashes.
pub mod profile;

// --- Re-exports for backward compatibility ---
// All downstream crates that were doing `use cjc_runtime::Tensor` etc. continue to work.

/// Re-export: COW buffer with deterministic copy-on-write semantics.
pub use buffer::Buffer;
/// Re-export: N-dimensional tensor -- the primary numerical type.
pub use tensor::Tensor;
/// Re-export: Pre-allocated KV-cache scratchpad.
pub use scratchpad::Scratchpad;
/// Re-export: 16-byte-aligned memory pool and byte slice.
pub use aligned_pool::{AlignedPool, AlignedByteSlice};
/// Re-export: Block-paged KV-cache types.
pub use paged_kv::{KvBlock, PagedKvCache};
/// Re-export: GC heap and reference types.
pub use gc::{GcRef, GcHeap};
/// Re-export: Size-class binned allocator.
pub use binned_alloc::BinnedAllocator;
/// Re-export: Bump-arena and arena store types.
pub use frame_arena::{FrameArena, ArenaStore};
/// Re-export: Object slab and slab reference types.
pub use object_slab::{ObjectSlab, SlabRef};
/// Re-export: Sparse matrix types (CSR and COO).
pub use sparse::{SparseCsr, SparseCoo};
/// Re-export: Tiled matrix multiplication engine.
pub use tensor_tiled::TiledMatmul;
/// Re-export: Deterministic map, MurmurHash3, and value hashing utilities.
pub use det_map::{DetMap, murmurhash3, murmurhash3_finalize, value_hash, values_equal_static};
/// Re-export: Universal value type and supporting types.
pub use value::{Value, Bf16, FnValue};
/// Re-export: Runtime error type.
pub use error::RuntimeError;
/// Re-export: Typed tensor storage types.
pub use tensor_dtype::{DType, TypedStorage};

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
    fn test_gc_collect_is_noop_rc_backed() {
        // In the RC-backed ObjectSlab, collect() is a no-op.
        // All objects survive regardless of roots provided.
        let mut heap = GcHeap::new(100);
        let r1 = heap.alloc(1i64);
        let r2 = heap.alloc(2i64);
        let r3 = heap.alloc(3i64);

        assert_eq!(heap.live_count(), 3);

        // collect with partial roots — all objects survive (RC, not GC)
        heap.collect(&[r1, r2]);

        assert_eq!(heap.live_count(), 3, "RC keeps all objects alive");
        assert_eq!(*heap.get::<i64>(r1).unwrap(), 1);
        assert_eq!(*heap.get::<i64>(r2).unwrap(), 2);
        assert_eq!(*heap.get::<i64>(r3).unwrap(), 3);
    }

    #[test]
    fn test_gc_explicit_free_and_slot_reuse() {
        // Slot reuse now requires explicit free() — no automatic GC collection.
        let mut heap = GcHeap::new(100);
        let r1 = heap.alloc(1i64);
        let r2 = heap.alloc(2i64);
        let r3 = heap.alloc(3i64);

        // Explicitly free all three slots
        heap.free(r1);
        heap.free(r2);
        heap.free(r3);
        assert_eq!(heap.live_count(), 0);
        assert_eq!(heap.free_list().len(), 3);

        // New alloc reuses freed slot (LIFO)
        let r4 = heap.alloc(99i64);
        assert!(r4.index < 3, "should reuse a freed slot");
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
