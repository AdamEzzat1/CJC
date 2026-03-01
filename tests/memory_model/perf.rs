//! Memory Model 2.0 — Performance harness.
//!
//! Run with: `cargo test --test test_memory_model -- --ignored --nocapture`

use std::time::Instant;

use cjc_runtime::{
    ArenaStore, BinnedAllocator, FrameArena, ObjectSlab, TiledMatmul,
};

struct PerfResult {
    name: &'static str,
    ops: usize,
    elapsed_us: u128,
}

impl PerfResult {
    fn ops_per_sec(&self) -> f64 {
        if self.elapsed_us == 0 {
            return f64::INFINITY;
        }
        self.ops as f64 / (self.elapsed_us as f64 / 1_000_000.0)
    }
}

fn bench<F: FnOnce() -> usize>(name: &'static str, f: F) -> PerfResult {
    let start = Instant::now();
    let ops = f();
    let elapsed_us = start.elapsed().as_micros();
    PerfResult { name, ops, elapsed_us }
}

#[test]
#[ignore]
fn memory_model_perf_harness() {
    let mut results = Vec::new();

    // 1. ObjectSlab alloc/free throughput.
    results.push(bench("slab_alloc_free_10k", || {
        let mut slab = ObjectSlab::new();
        let n = 10_000;
        let refs: Vec<_> = (0..n).map(|i| slab.alloc(i as i64)).collect();
        for r in refs {
            slab.free(r);
        }
        n * 2 // alloc + free
    }));

    // 2. FrameArena alloc + reset throughput.
    results.push(bench("arena_alloc_reset_10k", || {
        let mut arena = FrameArena::new();
        let n = 10_000;
        for _ in 0..n {
            arena.alloc_bytes(64);
        }
        arena.reset();
        n + 1 // alloc + reset
    }));

    // 3. ArenaStore alloc/read/reset cycle.
    results.push(bench("arena_store_cycle_10k", || {
        let mut store = ArenaStore::new();
        let n = 10_000;
        let indices: Vec<_> = (0..n).map(|i| store.alloc(i as i64)).collect();
        let mut sum = 0i64;
        for &idx in &indices {
            sum += store.get::<i64>(idx).unwrap();
        }
        assert!(sum > 0);
        store.reset();
        n * 2 + 1
    }));

    // 4. BinnedAllocator alloc/free mixed sizes.
    results.push(bench("binned_mixed_alloc_free_10k", || {
        let mut alloc = BinnedAllocator::new();
        let sizes = [16, 32, 64, 128, 256, 512, 1024, 48, 16, 32];
        let n = 10_000;
        let blocks: Vec<_> = (0..n).map(|i| alloc.alloc(sizes[i % sizes.len()])).collect();
        for b in blocks {
            alloc.free(b);
        }
        n * 2
    }));

    // 5. TiledMatmul 128x128.
    results.push(bench("tiled_matmul_128x128", || {
        let engine = TiledMatmul::new();
        let n = 128;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.001).collect();
        let _c = engine.matmul(&a, n, n, &b, n);
        n * n * n * 2 // 2n^3 FLOPs for matmul
    }));

    // 6. TiledMatmul 256x256.
    results.push(bench("tiled_matmul_256x256", || {
        let engine = TiledMatmul::new();
        let n = 256;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.0001).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (n * n - i) as f64 * 0.0001).collect();
        let _c = engine.matmul(&a, n, n, &b, n);
        n * n * n * 2
    }));

    // Print markdown table.
    println!();
    println!("| Benchmark | Ops | Time (us) | Ops/sec |");
    println!("|-----------|-----|-----------|---------|");
    for r in &results {
        println!(
            "| {} | {} | {} | {:.0} |",
            r.name, r.ops, r.elapsed_us, r.ops_per_sec()
        );
    }
    println!();
}
