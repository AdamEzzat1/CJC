//! Memory Model 2.0 — Integration test suite.
//!
//! Tests for all Memory Model 2.0 components:
//! - G0: ObjectSlab (arena-first, RC-last)
//! - G1: Deterministic alloc/dealloc (no stop-the-world)
//! - G2: Policy A (strong cycles forbidden, Weak breaks cycles)
//! - G3: @no_gc contract enforcement via escape analysis
//! - G4: Deterministic memory layout
//!
//! Plus component tests for:
//! - Escape analysis classification
//! - FrameArena bump allocation and reset
//! - BinnedAllocator size-class routing and LIFO reuse
//! - TiledMatmul correctness and determinism

use cjc_runtime::{
    ArenaStore, BinnedAllocator, FrameArena, GcHeap, GcRef, ObjectSlab, SlabRef, TiledMatmul,
};

// ============================================================
// G0: ObjectSlab — RC-backed, deterministic slot reuse
// ============================================================

#[test]
fn g0_slab_alloc_is_deterministic() {
    let mut s1 = ObjectSlab::new();
    let mut s2 = ObjectSlab::new();

    let refs1: Vec<SlabRef> = (0..10).map(|i| s1.alloc(i as i64)).collect();
    let refs2: Vec<SlabRef> = (0..10).map(|i| s2.alloc(i as i64)).collect();

    let idx1: Vec<usize> = refs1.iter().map(|r| r.index).collect();
    let idx2: Vec<usize> = refs2.iter().map(|r| r.index).collect();
    assert_eq!(idx1, idx2, "same alloc sequence → same indices");
}

#[test]
fn g0_slab_free_lifo_order() {
    let mut slab = ObjectSlab::new();
    let r1 = slab.alloc(1i64);
    let r2 = slab.alloc(2i64);
    let r3 = slab.alloc(3i64);

    slab.free(r1);
    slab.free(r2);
    slab.free(r3);

    // LIFO: r3 should be reused first.
    let r4 = slab.alloc(4i64);
    assert_eq!(r4.index, r3.index, "LIFO reuse");
}

#[test]
fn g0_gc_heap_compat_collect_is_noop() {
    let mut heap = GcHeap::new(100);
    let r1 = heap.alloc(1i64);
    let r2 = heap.alloc(2i64);
    let r3 = heap.alloc(3i64);

    heap.collect(&[r1]); // no-op

    assert_eq!(heap.live_count(), 3, "collect must be a no-op in RC mode");
    assert_eq!(*heap.get::<i64>(r1).unwrap(), 1);
    assert_eq!(*heap.get::<i64>(r2).unwrap(), 2);
    assert_eq!(*heap.get::<i64>(r3).unwrap(), 3);
}

// ============================================================
// G1: Deterministic alloc/dealloc
// ============================================================

#[test]
fn g1_arena_never_returns_memory() {
    let mut arena = FrameArena::with_page_size(64);
    for _ in 0..20 {
        arena.alloc_bytes(16);
    }
    let cap_before = arena.capacity();
    arena.reset();
    assert_eq!(
        arena.capacity(),
        cap_before,
        "arena capacity must not shrink after reset"
    );
}

#[test]
fn g1_binned_storage_never_shrinks() {
    let mut alloc = BinnedAllocator::new();
    let blocks: Vec<usize> = (0..50).map(|_| alloc.alloc(64)).collect();
    let storage_after_alloc = alloc.storage_bytes();

    for b in blocks {
        alloc.free(b);
    }
    assert_eq!(
        alloc.storage_bytes(),
        storage_after_alloc,
        "binned storage must not shrink after free"
    );
}

#[test]
fn g1_no_stop_the_world() {
    // Verify that collect() is O(1) — just increments a counter.
    let mut heap = GcHeap::new(100);
    for i in 0..1000 {
        heap.alloc(i as i64);
    }
    // This should be instant (no mark/sweep).
    heap.collect(&[]);
    assert_eq!(heap.live_count(), 1000, "all objects survive (no GC)");
}

// ============================================================
// G2: Policy A — Weak refs, strong cycles forbidden
// ============================================================

#[test]
fn g2_rc_prevents_use_after_free() {
    let mut slab = ObjectSlab::new();
    let r = slab.alloc(42i64);
    assert_eq!(*slab.get::<i64>(r).unwrap(), 42);

    slab.free(r);
    assert!(slab.get::<i64>(r).is_none(), "freed slot must not be readable");
}

#[test]
fn g2_type_mismatch_safe() {
    let mut slab = ObjectSlab::new();
    let r = slab.alloc(42i64);
    assert!(slab.get::<String>(r).is_none(), "wrong type must return None");
}

// ============================================================
// G3: @no_gc contract enforcement
// ============================================================

#[test]
fn g3_escape_analysis_classifies_primitives_as_stack() {
    use cjc_mir::escape::{analyze_function, AllocHint};
    use cjc_mir::*;

    let func = MirFunction {
        id: MirFnId(0),
        name: "f".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "x".to_string(),
                    mutable: false,
                    init: MirExpr {
                        kind: MirExprKind::IntLit(42),
                    },
                    alloc_hint: None,
                },
                MirStmt::Let {
                    name: "y".to_string(),
                    mutable: true,
                    init: MirExpr {
                        kind: MirExprKind::FloatLit(3.14),
                    },
                    alloc_hint: None,
                },
            ],
            result: None,
        },
        is_nogc: true,
    };

    let info = analyze_function(&func);
    assert_eq!(info.bindings["x"].0, AllocHint::Stack);
    assert_eq!(info.bindings["y"].0, AllocHint::Stack);
}

#[test]
fn g3_escape_analysis_detects_return_escape() {
    use cjc_mir::escape::{analyze_function, AllocHint};
    use cjc_mir::*;

    let func = MirFunction {
        id: MirFnId(0),
        name: "f".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![
                MirStmt::Let {
                    name: "s".to_string(),
                    mutable: false,
                    init: MirExpr {
                        kind: MirExprKind::StringLit("hello".to_string()),
                    },
                    alloc_hint: None,
                },
                MirStmt::Return(Some(MirExpr {
                    kind: MirExprKind::Var("s".to_string()),
                })),
            ],
            result: None,
        },
        is_nogc: false,
    };

    let info = analyze_function(&func);
    assert_eq!(info.bindings["s"].0, AllocHint::Rc, "returned value should be Rc");
}

#[test]
fn g3_has_heap_alloc_detects_arena_and_rc() {
    use cjc_mir::escape::{analyze_function, has_heap_alloc};
    use cjc_mir::*;

    // Function with only primitives → no heap alloc.
    let pure_fn = MirFunction {
        id: MirFnId(0),
        name: "pure".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![MirStmt::Let {
                name: "x".to_string(),
                mutable: false,
                init: MirExpr { kind: MirExprKind::IntLit(1) },
                alloc_hint: None,
            }],
            result: None,
        },
        is_nogc: true,
    };
    assert!(!has_heap_alloc(&analyze_function(&pure_fn)));

    // Function with a string → has Arena alloc.
    let string_fn = MirFunction {
        id: MirFnId(0),
        name: "with_str".to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: vec![MirStmt::Let {
                name: "s".to_string(),
                mutable: false,
                init: MirExpr {
                    kind: MirExprKind::StringLit("hello".to_string()),
                },
                alloc_hint: None,
            }],
            result: None,
        },
        is_nogc: false,
    };
    assert!(has_heap_alloc(&analyze_function(&string_fn)));
}

// ============================================================
// G4: Deterministic memory layout
// ============================================================

#[test]
fn g4_arena_layout_deterministic() {
    let mut a1 = FrameArena::with_page_size(128);
    let mut a2 = FrameArena::with_page_size(128);

    let r1: Vec<_> = (0..20).map(|_| a1.alloc_bytes(16)).collect();
    let r2: Vec<_> = (0..20).map(|_| a2.alloc_bytes(16)).collect();
    assert_eq!(r1, r2);
}

#[test]
fn g4_binned_alloc_layout_deterministic() {
    let mut a1 = BinnedAllocator::new();
    let mut a2 = BinnedAllocator::new();

    let sizes = [16, 32, 64, 128, 256, 512, 1024, 48, 16, 32];
    let b1: Vec<usize> = sizes.iter().map(|&s| a1.alloc(s)).collect();
    let b2: Vec<usize> = sizes.iter().map(|&s| a2.alloc(s)).collect();
    assert_eq!(b1, b2);
}

#[test]
fn g4_tiled_matmul_deterministic() {
    let e1 = TiledMatmul::with_tile_size(4);
    let e2 = TiledMatmul::with_tile_size(4);

    let a: Vec<f64> = (0..64).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..64).map(|i| (64 - i) as f64 * 0.1).collect();

    let c1 = e1.matmul(&a, 8, 8, &b, 8);
    let c2 = e2.matmul(&a, 8, 8, &b, 8);
    assert_eq!(c1, c2, "same inputs → same outputs");
}

// ============================================================
// Component: ArenaStore lifecycle
// ============================================================

#[test]
fn arena_store_lifecycle() {
    let mut store = ArenaStore::new();

    // Allocate some values.
    let a = store.alloc(42i64);
    let b = store.alloc("hello".to_string());

    assert_eq!(*store.get::<i64>(a).unwrap(), 42);
    assert_eq!(store.get::<String>(b).unwrap().as_str(), "hello");
    assert_eq!(store.live_count(), 2);

    // Reset — bulk free.
    store.reset();
    assert_eq!(store.live_count(), 0);

    // New allocations reuse old slots.
    let c = store.alloc(99i64);
    assert!(c < 2);
    assert_eq!(*store.get::<i64>(c).unwrap(), 99);
}

// ============================================================
// Component: TiledMatmul correctness
// ============================================================

#[test]
fn tiled_matmul_vs_naive() {
    let n = 16;
    let a: Vec<f64> = (0..n * n).map(|i| (i as f64 + 1.0) / (n * n) as f64).collect();
    let b: Vec<f64> = (0..n * n)
        .map(|i| ((n * n - i) as f64) / (n * n) as f64)
        .collect();

    let tiled = TiledMatmul::with_tile_size(4).matmul(&a, n, n, &b, n);

    // Naive.
    let mut naive = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            for p in 0..n {
                naive[i * n + j] += a[i * n + p] * b[p * n + j];
            }
        }
    }

    for k in 0..n * n {
        assert!(
            (tiled[k] - naive[k]).abs() < 1e-10,
            "mismatch at {k}: tiled={}, naive={}",
            tiled[k],
            naive[k]
        );
    }
}
