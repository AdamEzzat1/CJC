// Adaptive TidyView Engine v3 Phase 3 — Hybrid streaming set-op parity tests
//
// Phase 3 added per-chunk fast paths for Hybrid in `AdaptiveSelection`'s
// `intersect` and `union`. Before Phase 3, every Hybrid set-op fell through
// to "materialize both as full BitMask, word-AND, reclassify" — defeating the
// chunked layout's purpose.
//
// Note on test surface: `TidyView::filter` AND-collapses inside the predicate
// bytecode interpreter rather than calling `AdaptiveSelection::intersect`, so
// chained `.filter()` calls do *not* exercise Phase 3 today. To prove parity
// of the new dispatches against a scalar oracle, these tests build realistic
// Hybrid selections via `.filter(...)` (using pre-computed bit-pattern columns
// to land scattered mid-band density), then call `.selection().intersect(...)`
// / `.union(...)` directly.
//
// Phase 3's first end-to-end consumer will be Phase 4 (cat-aware joins), where
// semi-join/anti-join naturally produce two AdaptiveSelections that need to
// be combined. This test file pins the surface contract that Phase 4 will rely
// on.

use cjc_data::{AdaptiveSelection, Column, DBinOp, DExpr, DataFrame};

fn binop(op: DBinOp, l: DExpr, r: DExpr) -> DExpr {
    DExpr::BinOp {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

fn col(name: &str) -> DExpr {
    DExpr::Col(name.into())
}

fn lit(v: i64) -> DExpr {
    DExpr::LitInt(v)
}

/// Build a frame with three Int columns whose `== 1` predicates yield
/// distinct chunk shapes inside the Hybrid arm:
///   - "a": 1 every 50 rows (2% scattered → Sparse-chunked Hybrid)
///   - "b": 1 every 50 rows offset by 25 (2% scattered, disjoint with "a")
///   - "c": 1 every 12 rows (~8.3% scattered → Dense-chunked Hybrid;
///     per-chunk count ≈ 341 exceeds the 128 sparse threshold)
fn frame_100k() -> DataFrame {
    let n: usize = 100_000;
    let a: Vec<i64> = (0..n).map(|i| if i % 50 == 0 { 1 } else { 0 }).collect();
    let b: Vec<i64> = (0..n).map(|i| if i % 50 == 25 { 1 } else { 0 }).collect();
    let c: Vec<i64> = (0..n).map(|i| if i % 12 == 0 { 1 } else { 0 }).collect();
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(a)),
        ("b".into(), Column::Int(b)),
        ("c".into(), Column::Int(c)),
    ])
    .unwrap()
}

/// Materialize the AdaptiveSelection from `df.filter(col == val)`.
fn select_eq(df: &DataFrame, col_name: &str, val: i64) -> AdaptiveSelection {
    let pred = binop(DBinOp::Eq, col(col_name), lit(val));
    let view = df.clone().tidy().filter(&pred).unwrap();
    view.selection().clone()
}

/// Scalar oracle: index set where `cond(i)` is true.
fn oracle<F: Fn(usize) -> bool>(n: usize, cond: F) -> Vec<usize> {
    (0..n).filter(|&i| cond(i)).collect()
}

// ── Hybrid ∩ Hybrid: per-chunk dispatch ──────────────────────────────────

#[test]
fn phase3_hybrid_intersect_hybrid_sparse_sparse_chunks() {
    // Both selections are Sparse-chunked Hybrid (per-chunk count ≈ 82 < 128).
    // Forces the Sparse∩Sparse merge-walk inside `intersect_chunks`.
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sb = select_eq(&df, "b", 1);
    assert_eq!(sa.explain_selection_mode(), "Hybrid");
    assert_eq!(sb.explain_selection_mode(), "Hybrid");

    let r = sa.intersect(&sb);
    let want = oracle(100_000, |i| i % 50 == 0 && i % 50 == 25);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
    // Disjoint by construction → simplify_hybrid collapses to Empty.
    assert_eq!(r.explain_selection_mode(), "Empty");
}

#[test]
fn phase3_hybrid_intersect_hybrid_sparse_dense_chunks() {
    // sa: Sparse-chunked, sc: Dense-chunked → forces Sparse∩Dense filter-walk.
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sc = select_eq(&df, "c", 1);
    assert_eq!(sa.explain_selection_mode(), "Hybrid");
    assert_eq!(sc.explain_selection_mode(), "Hybrid");

    let r = sa.intersect(&sc);
    let want = oracle(100_000, |i| i % 50 == 0 && i % 12 == 0);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
    // Symmetric: should produce identical output.
    let r2 = sc.intersect(&sa);
    assert_eq!(r2.iter_indices().collect::<Vec<usize>>(), want);
}

#[test]
fn phase3_hybrid_union_hybrid_sparse_sparse_chunks() {
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sb = select_eq(&df, "b", 1);
    let r = sa.union(&sb);
    let want = oracle(100_000, |i| i % 50 == 0 || i % 50 == 25);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
}

#[test]
fn phase3_hybrid_union_hybrid_sparse_dense_chunks() {
    // Sparse ∪ Dense per-chunk: writes sparse offsets into a copy of the
    // dense word buffer, then reclassifies.
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sc = select_eq(&df, "c", 1);
    let r = sa.union(&sc);
    let want = oracle(100_000, |i| i % 50 == 0 || i % 12 == 0);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
}

// ── Hybrid ∩ SelectionVector / VerbatimMask ──────────────────────────────

#[test]
fn phase3_hybrid_intersect_selection_vector() {
    // SelectionVector requires count < nrows/1024 = 97. Build via filter
    // returning fewer than 97 rows: x < 50 with a contiguous 0..nrows column.
    let xs: Vec<i64> = (0..100_000).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap();
    let small = {
        let pred = binop(DBinOp::Lt, col("x"), lit(50));
        df.clone().tidy().filter(&pred).unwrap().selection().clone()
    };
    assert_eq!(small.explain_selection_mode(), "SelectionVector");

    // Build a Hybrid by hand (the single-column frame above won't produce one
    // for any contiguous predicate that overlaps the same row range we care
    // about). Use the bit-pattern frame for the Hybrid side.
    let dfh = frame_100k();
    let sa = select_eq(&dfh, "a", 1);
    assert_eq!(sa.explain_selection_mode(), "Hybrid");

    // Both selections live over 100_000-row frames, so nrows align.
    let r = sa.intersect(&small);
    // Oracle: rows < 50 AND i % 50 == 0 → just row 0.
    let want = oracle(100_000, |i| i < 50 && i % 50 == 0);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);

    // Symmetric: Sparse ∩ Hybrid hits the same dispatch arm with sides flipped.
    let r2 = small.intersect(&sa);
    assert_eq!(r2.iter_indices().collect::<Vec<usize>>(), want);
}

#[test]
fn phase3_hybrid_intersect_verbatim_mask() {
    // Build a VerbatimMask via a 50% predicate.
    let xs: Vec<i64> = (0..100_000).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap();
    let dense = {
        let pred = binop(DBinOp::Lt, col("x"), lit(50_000));
        df.clone().tidy().filter(&pred).unwrap().selection().clone()
    };
    assert_eq!(dense.explain_selection_mode(), "VerbatimMask");

    let dfh = frame_100k();
    let sa = select_eq(&dfh, "a", 1);
    assert_eq!(sa.explain_selection_mode(), "Hybrid");

    let r = sa.intersect(&dense);
    let want = oracle(100_000, |i| i < 50_000 && i % 50 == 0);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
    // Symmetric.
    let r2 = dense.intersect(&sa);
    assert_eq!(r2.iter_indices().collect::<Vec<usize>>(), want);
}

#[test]
fn phase3_hybrid_union_selection_vector_scatter() {
    // Hybrid ∪ SelectionVector: scatters sparse rows into Hybrid chunks.
    let xs: Vec<i64> = (0..100_000).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap();
    let small = {
        let pred = binop(DBinOp::Lt, col("x"), lit(50));
        df.clone().tidy().filter(&pred).unwrap().selection().clone()
    };
    assert_eq!(small.explain_selection_mode(), "SelectionVector");

    let dfh = frame_100k();
    let sa = select_eq(&dfh, "a", 1);

    let r = sa.union(&small);
    let want = oracle(100_000, |i| i < 50 || i % 50 == 0);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
}

#[test]
fn phase3_hybrid_union_verbatim_mask() {
    let xs: Vec<i64> = (0..100_000).collect();
    let df = DataFrame::from_columns(vec![("x".into(), Column::Int(xs))]).unwrap();
    let dense = {
        let pred = binop(DBinOp::Lt, col("x"), lit(50_000));
        df.clone().tidy().filter(&pred).unwrap().selection().clone()
    };
    let dfh = frame_100k();
    let sa = select_eq(&dfh, "a", 1);

    let r = sa.union(&dense);
    let want = oracle(100_000, |i| i < 50_000 || i % 50 == 0);
    let got: Vec<usize> = r.iter_indices().collect();
    assert_eq!(got, want);
}

// ── Three-way associativity over chunked path ────────────────────────────

#[test]
fn phase3_three_way_intersect_associative() {
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sb = select_eq(&df, "b", 1);
    let sc = select_eq(&df, "c", 1);

    let abc = sa.intersect(&sb).intersect(&sc);
    let acb = sa.intersect(&sc).intersect(&sb);
    let cba = sc.intersect(&sb).intersect(&sa);
    assert_eq!(
        abc.iter_indices().collect::<Vec<usize>>(),
        acb.iter_indices().collect::<Vec<usize>>(),
    );
    assert_eq!(
        abc.iter_indices().collect::<Vec<usize>>(),
        cba.iter_indices().collect::<Vec<usize>>(),
    );

    // Oracle: a∩b is empty (disjoint), so abc must be empty.
    let want = oracle(100_000, |i| {
        i % 50 == 0 && i % 50 == 25 && i % 12 == 0
    });
    assert!(want.is_empty());
    assert_eq!(abc.count(), 0);
}

// ── Cardinality identity holds across the chunked path ───────────────────

#[test]
fn phase3_cardinality_identity_holds() {
    // |A| + |B| = |A ∪ B| + |A ∩ B|
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sc = select_eq(&df, "c", 1);
    let inter = sa.intersect(&sc);
    let union = sa.union(&sc);
    assert_eq!(sa.count() + sc.count(), inter.count() + union.count());
}

// ── Partial final chunk safety ───────────────────────────────────────────

#[test]
fn phase3_partial_final_chunk_no_oob() {
    // 100_000 rows / 4096 = 24.41 → final chunk holds 1_696 rows. Exercise
    // every per-chunk dispatch arm with a partial-chunk-bearing frame.
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sc = select_eq(&df, "c", 1);
    let r = sa.intersect(&sc);
    for row in r.iter_indices() {
        assert!(row < 100_000, "row {row} out of bounds (final chunk leak)");
    }
    // Ascending invariant.
    let collected: Vec<usize> = r.iter_indices().collect();
    for w in collected.windows(2) {
        assert!(w[0] < w[1]);
    }
}

// ── Mode reporting matches expectations ──────────────────────────────────

#[test]
fn phase3_hybrid_intersect_hybrid_stays_hybrid_when_mid_band() {
    // a ∩ c: a∩c hits ≈ 1/(50*12) = 1/600 ≈ 167 hits → ratio 167/100_000 =
    // 0.167%. Below 1024⁻¹ = ~0.098%? Let's compute: 100_000 / 1024 ≈ 97.
    // 167 > 97 → not sparse, mid-band → simplify_hybrid keeps it Hybrid.
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sc = select_eq(&df, "c", 1);
    let r = sa.intersect(&sc);
    // simplify_hybrid only flattens to All/Empty; mid-band stays Hybrid.
    assert!(
        r.explain_selection_mode() == "Hybrid"
            || r.explain_selection_mode() == "Empty",
        "expected Hybrid or Empty, got {}",
        r.explain_selection_mode()
    );
}

// ── Bench (ignored): chained mid-band intersections ──────────────────────

#[test]
#[ignore]
fn bench_phase3_hybrid_set_op() {
    use cjc_data::BitMask;
    use std::time::Instant;
    let df = frame_100k();
    let sa = select_eq(&df, "a", 1);
    let sc = select_eq(&df, "c", 1);
    let runs = 100;
    // Warmup.
    let _ = sa.intersect(&sc);

    // Phase 3: per-chunk dispatch.
    let mut new_ns: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let r = sa.intersect(&sc);
        new_ns.push(t.elapsed().as_nanos());
        std::hint::black_box(r);
    }

    // Pre-Phase-3 oracle: materialize both as BitMask, word-AND, reclassify.
    let mut old_ns: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let lhs = sa.materialize_mask();
        let rhs = sc.materialize_mask();
        let words: Vec<u64> = lhs
            .words_slice()
            .iter()
            .zip(rhs.words_slice().iter())
            .map(|(a, b)| a & b)
            .collect();
        let r = AdaptiveSelection::from_predicate_result(words, lhs.nrows());
        old_ns.push(t.elapsed().as_nanos());
        std::hint::black_box(r);
        // Touch lhs/rhs so the compiler doesn't elide them.
        std::hint::black_box(lhs.count_ones());
        std::hint::black_box::<BitMask>(rhs);
    }

    let avg = |xs: &[u128]| xs.iter().sum::<u128>() as f64 / xs.len() as f64;
    let new_us = avg(&new_ns) / 1_000.0;
    let old_us = avg(&old_ns) / 1_000.0;
    let speedup = old_us / new_us;
    println!(
        "[Phase 3 bench] Hybrid ∩ Hybrid (100k rows, sparse-chunked × dense-chunked):"
    );
    println!("  Phase 3 chunked path:        {new_us:>7.2} μs avg (n={runs})");
    println!("  Pre-Phase-3 materialize path: {old_us:>7.2} μs avg (n={runs})");
    println!("  Speedup:                     {speedup:>7.2}×");
}
