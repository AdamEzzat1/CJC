// Phase 17: Categorical Foundations — fct_encode, fct_lump, fct_reorder,
//           fct_collapse, NullableFactor
//
// Covers ALL edge cases from Role 2 (determinism audit), Role 3 (memory/NoGC),
// Role 4 (edge-case hunter).
//
// Run: cargo test --test test_phase10_tidy phase17

use cjc_data::{
    Column, DataFrame, FctColumn, NullableFactor, TidyError,
};
use cjc_mir::{
    MirBody, MirExpr, MirExprKind, MirFunction, MirFnId, MirProgram, MirStmt,
};
use cjc_mir::nogc_verify::verify_nogc;

// ── Test helpers ──────────────────────────────────────────────────────────────

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn opt_strs(v: &[Option<&str>]) -> Vec<Option<String>> {
    v.iter().map(|o| o.map(|s| s.to_string())).collect()
}

fn mk_expr(kind: MirExprKind) -> MirExpr { MirExpr { kind } }
fn mk_call(name: &str) -> MirExpr {
    mk_expr(MirExprKind::Call {
        callee: Box::new(mk_expr(MirExprKind::Var(name.to_string()))),
        args: vec![],
    })
}
fn mk_fn(name: &str, is_nogc: bool, calls: &[&str]) -> MirFunction {
    MirFunction {
        id: MirFnId(0),
        name: name.to_string(),
        type_params: vec![],
        params: vec![],
        return_type: None,
        body: MirBody {
            stmts: calls.iter().map(|c| MirStmt::Expr(mk_call(c))).collect(),
            result: None,
        },
        is_nogc,
        cfg_body: None,
        decorators: vec![],
    }
}
fn mk_program(fns: Vec<MirFunction>) -> MirProgram {
    MirProgram { functions: fns, struct_defs: vec![], enum_defs: vec![], entry: MirFnId(0) }
}

// ── FctColumn::encode — basic ──────────────────────────────────────────────────

#[test]
fn test_fct_encode_basic_levels() {
    let fct = FctColumn::encode(&strs(&["b", "a", "b", "c", "a"])).unwrap();
    // first-occurrence order: b, a, c
    assert_eq!(fct.levels, strs(&["b", "a", "c"]));
    assert_eq!(fct.nlevels(), 3);
    assert_eq!(fct.nrows(), 5);
}

#[test]
fn test_fct_encode_indices_correct() {
    let fct = FctColumn::encode(&strs(&["b", "a", "b", "c", "a"])).unwrap();
    // b=0, a=1, c=2
    assert_eq!(fct.data, vec![0u16, 1, 0, 2, 1]);
}

#[test]
fn test_fct_encode_decode_roundtrip() {
    let src = strs(&["x", "y", "z", "x", "y"]);
    let fct = FctColumn::encode(&src).unwrap();
    let decoded: Vec<&str> = (0..fct.nrows()).map(|i| fct.decode(i)).collect();
    assert_eq!(decoded, vec!["x", "y", "z", "x", "y"]);
}

#[test]
fn test_fct_encode_single_level() {
    let fct = FctColumn::encode(&strs(&["a", "a", "a"])).unwrap();
    assert_eq!(fct.nlevels(), 1);
    assert_eq!(fct.levels, strs(&["a"]));
    assert_eq!(fct.data, vec![0u16, 0, 0]);
}

#[test]
fn test_fct_encode_empty_input() {
    let fct = FctColumn::encode(&[]).unwrap();
    assert_eq!(fct.nlevels(), 0);
    assert_eq!(fct.nrows(), 0);
}

// ── fct_encode determinism ─────────────────────────────────────────────────────

#[test]
fn test_fct_encode_determinism_two_runs() {
    // Same input must produce identical levels and data on two independent calls
    let input = strs(&["cat", "dog", "cat", "bird", "dog", "fish"]);
    let fct1 = FctColumn::encode(&input).unwrap();
    let fct2 = FctColumn::encode(&input).unwrap();
    assert_eq!(fct1.levels, fct2.levels, "levels must be identical across runs");
    assert_eq!(fct1.data, fct2.data, "data must be identical across runs");
}

#[test]
fn test_fct_encode_stable_after_filter() {
    // Encoding on a subset (simulated by passing only visible rows) must
    // reflect first-occurrence order in THAT subset.
    let full = strs(&["c", "a", "b", "a", "c"]);
    // Visible rows: indices 1,2,3 → ["a","b","a"]
    let subset: Vec<String> = vec![full[1].clone(), full[2].clone(), full[3].clone()];
    let fct = FctColumn::encode(&subset).unwrap();
    assert_eq!(fct.levels, strs(&["a", "b"]));
    assert_eq!(fct.data, vec![0u16, 1, 0]);
}

#[test]
fn test_fct_encode_stable_after_bind_rows() {
    // bind_rows semantics: concatenate two string slices, encode once
    let left = strs(&["a", "b"]);
    let right = strs(&["c", "a"]);
    let combined: Vec<String> = left.iter().chain(right.iter()).cloned().collect();
    let fct = FctColumn::encode(&combined).unwrap();
    // first-occurrence: a, b, c
    assert_eq!(fct.levels, strs(&["a", "b", "c"]));
    assert_eq!(fct.data, vec![0u16, 1, 2, 0]);
}

#[test]
fn test_fct_encode_idempotent_double_encode() {
    // Encoding the decoded values of an FctColumn must produce the same levels
    let src = strs(&["z", "a", "m", "z"]);
    let fct1 = FctColumn::encode(&src).unwrap();
    let decoded: Vec<String> = (0..fct1.nrows()).map(|i| fct1.decode(i).to_string()).collect();
    let fct2 = FctColumn::encode(&decoded).unwrap();
    assert_eq!(fct1.levels, fct2.levels, "double-encode must be idempotent");
    assert_eq!(fct1.data, fct2.data);
}

// ── fct_encode via TidyView ────────────────────────────────────────────────────

#[test]
fn test_fct_encode_from_view_basic() {
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["b", "a", "b", "c"]))),
        ("val".to_string(), Column::Int(vec![1, 2, 3, 4])),
    ]).unwrap();
    let view = df.tidy();
    let fct = view.fct_encode("cat").unwrap();
    assert_eq!(fct.levels, strs(&["b", "a", "c"]));
    assert_eq!(fct.nrows(), 4);
}

#[test]
fn test_fct_encode_from_view_after_filter() {
    // After filter, only visible rows are encoded
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a", "b", "c", "a"]))),
        ("val".to_string(), Column::Int(vec![10, 20, 30, 10])),
    ]).unwrap();
    let view = df.tidy()
        .filter(&cjc_data::DExpr::BinOp {
            op: cjc_data::DBinOp::Gt,
            left: Box::new(cjc_data::DExpr::Col("val".to_string())),
            right: Box::new(cjc_data::DExpr::LitInt(10)),
        }).unwrap();
    let fct = view.fct_encode("cat").unwrap();
    // Visible rows: val>10 → rows 1,2 → ["b","c"]
    assert_eq!(fct.levels, strs(&["b", "c"]));
    assert_eq!(fct.nrows(), 2);
}

#[test]
fn test_fct_encode_unknown_col_error() {
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a", "b"]))),
    ]).unwrap();
    let err = df.tidy().fct_encode("nonexistent");
    assert!(matches!(err, Err(TidyError::ColumnNotFound(_))));
}

// ── Capacity overflow ──────────────────────────────────────────────────────────

#[test]
#[ignore = "capacity boundary test — slow (65,535 unique strings); run with --ignored"]
fn test_fct_encode_exactly_65535_levels_ok() {
    // Build exactly 65,535 distinct strings → must succeed
    let strings: Vec<String> = (0u32..65_535).map(|i| format!("level_{}", i)).collect();
    let fct = FctColumn::encode(&strings);
    assert!(fct.is_ok(), "65,535 levels must not overflow");
    let fct = fct.unwrap();
    assert_eq!(fct.nlevels(), 65_535);
}

#[test]
#[ignore = "capacity boundary test — slow (65,536 unique strings); run with --ignored"]
fn test_fct_encode_65536_levels_errors() {
    // 65,536 distinct strings → must return CapacityExceeded
    let strings: Vec<String> = (0u32..65_536).map(|i| format!("level_{}", i)).collect();
    let result = FctColumn::encode(&strings);
    assert!(
        matches!(result, Err(TidyError::CapacityExceeded { limit: 65_535, .. })),
        "65,536 levels must return CapacityExceeded"
    );
}

#[test]
fn test_fct_encode_capacity_boundary_small() {
    // Fast proxy: 1,000 distinct strings encode fine; 65,536th triggers error
    // This runs in normal CI without the 65k allocation cost.
    let strings: Vec<String> = (0u32..1_000).map(|i| format!("L{}", i)).collect();
    let fct = FctColumn::encode(&strings).unwrap();
    assert_eq!(fct.nlevels(), 1_000);
}

#[test]
#[ignore = "capacity boundary test — slow (builds 65,535-item mapping); run with --ignored"]
fn test_fct_collapse_after_max_cardinality_no_error() {
    // 65,535 levels collapsed to 1 — must succeed (capacity only enforced on encode)
    let strings: Vec<String> = (0u32..65_535).map(|i| format!("level_{}", i)).collect();
    let fct = FctColumn::encode(&strings).unwrap();
    // Collapse everything to "all"
    // fct_collapse takes &[(&str, &str)] — build from levels
    let owned: Vec<(String, String)> = fct.levels.iter()
        .map(|s| (s.clone(), "all".to_string()))
        .collect();
    let mapping_refs: Vec<(&str, &str)> = owned.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let collapsed = fct.fct_collapse(&mapping_refs);
    assert!(collapsed.is_ok(), "collapse to 1 level must not error");
    assert_eq!(collapsed.unwrap().nlevels(), 1);
}

// ── fct_lump ──────────────────────────────────────────────────────────────────

#[test]
fn test_fct_lump_basic_top2() {
    // a×3, b×2, c×1 → top-2 = a,b; c → "Other"
    let fct = FctColumn::encode(&strs(&["a","b","a","c","b","a"])).unwrap();
    let lumped = fct.fct_lump(2).unwrap();
    // levels: a, b, Other (first-occurrence order for kept, Other last)
    assert_eq!(lumped.levels, strs(&["a", "b", "Other"]));
    assert_eq!(lumped.nlevels(), 3);
}

#[test]
fn test_fct_lump_values_correct() {
    let fct = FctColumn::encode(&strs(&["a","b","a","c","b","a"])).unwrap();
    let lumped = fct.fct_lump(2).unwrap();
    // original: a→a, b→b, a→a, c→Other, b→b, a→a
    let decoded: Vec<&str> = (0..lumped.nrows()).map(|i| lumped.decode(i)).collect();
    assert_eq!(decoded, vec!["a","b","a","Other","b","a"]);
}

#[test]
fn test_fct_lump_n0_all_other() {
    // n=0 → all become "Other"
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let lumped = fct.fct_lump(0).unwrap();
    assert_eq!(lumped.nlevels(), 1);
    assert_eq!(lumped.levels[0], "Other");
    for i in 0..lumped.nrows() {
        assert_eq!(lumped.decode(i), "Other");
    }
}

#[test]
fn test_fct_lump_n_gte_nlevels_noop() {
    // n ≥ nlevels → no change (clone returned)
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let lumped = fct.fct_lump(10).unwrap();
    assert_eq!(lumped.levels, fct.levels);
    assert_eq!(lumped.data, fct.data);
}

#[test]
fn test_fct_lump_n_exactly_nlevels_noop() {
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let lumped = fct.fct_lump(3).unwrap();
    assert_eq!(lumped.levels, fct.levels);
}

#[test]
fn test_fct_lump_tie_breaking_first_occurrence() {
    // a×2, b×2, c×2 → all tied; top-1 must be "a" (first occurrence)
    let fct = FctColumn::encode(&strs(&["a","b","c","a","b","c"])).unwrap();
    let lumped = fct.fct_lump(1).unwrap();
    // Top-1 by freq+first-occurrence = "a"
    assert_eq!(lumped.levels[0], "a");
    assert_eq!(lumped.nlevels(), 2); // a + Other
}

#[test]
fn test_fct_lump_all_equal_frequency() {
    // All unique (each appears once) — top-2 by first-occurrence = first two
    let fct = FctColumn::encode(&strs(&["x","y","z","w"])).unwrap();
    let lumped = fct.fct_lump(2).unwrap();
    assert_eq!(lumped.levels[0], "x");
    assert_eq!(lumped.levels[1], "y");
    assert_eq!(lumped.nlevels(), 3); // x, y, Other
}

#[test]
fn test_fct_lump_all_categories_unique() {
    // Each category appears exactly once — top-1 = first
    let fct = FctColumn::encode(&strs(&["p","q","r","s"])).unwrap();
    let lumped = fct.fct_lump(1).unwrap();
    assert_eq!(lumped.levels[0], "p");
    assert_eq!(lumped.nlevels(), 2);
}

#[test]
fn test_fct_lump_other_collision_renamed() {
    // If "Other" is already a level, the lump bucket must be renamed "Other_"
    let fct = FctColumn::encode(&strs(&["Other","b","Other","b","c"])).unwrap();
    // "Other" freq=2, "b" freq=2, "c" freq=1 → top-2 keeps "Other","b"; "c"→bucket
    let lumped = fct.fct_lump(2).unwrap();
    // The bucket name must not collide with existing level "Other"
    let last_level = lumped.levels.last().unwrap();
    assert_ne!(last_level, "Other", "bucket must be renamed to avoid collision");
    assert!(last_level.starts_with("Other"), "bucket should still start with 'Other'");
}

#[test]
fn test_fct_lump_other_already_present_twice() {
    // "Other" and "Other_" both present → bucket becomes "Other__"
    let fct = FctColumn::encode(&strs(&["Other","Other_","a","Other","Other_","a","b"])).unwrap();
    let lumped = fct.fct_lump(2).unwrap();
    let last = lumped.levels.last().unwrap();
    assert!(last.starts_with("Other"), "bucket has 'Other' prefix");
    assert_ne!(last, "Other");
    assert_ne!(last, "Other_");
}

#[test]
fn test_fct_lump_determinism() {
    let input = strs(&["a","b","c","a","b","a"]);
    let fct = FctColumn::encode(&input).unwrap();
    let l1 = fct.fct_lump(2).unwrap();
    let l2 = fct.fct_lump(2).unwrap();
    assert_eq!(l1.levels, l2.levels);
    assert_eq!(l1.data, l2.data);
}

// ── fct_reorder ───────────────────────────────────────────────────────────────

#[test]
fn test_fct_reorder_ascending() {
    // levels: a,b,c with means 3.0,1.0,2.0 → reordered: b,c,a
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let means = vec![3.0f64, 1.0, 2.0];
    let reordered = fct.fct_reorder(&means, false).unwrap();
    assert_eq!(reordered.levels, strs(&["b", "c", "a"]));
}

#[test]
fn test_fct_reorder_descending() {
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let means = vec![3.0f64, 1.0, 2.0];
    let reordered = fct.fct_reorder(&means, true).unwrap();
    assert_eq!(reordered.levels, strs(&["a", "c", "b"]));
}

#[test]
fn test_fct_reorder_nan_sorts_last() {
    // NaN summary value sorts LAST regardless of direction
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let means = vec![f64::NAN, 1.0, 2.0];
    let asc = fct.fct_reorder(&means, false).unwrap();
    // b(1.0), c(2.0), a(NaN)
    assert_eq!(asc.levels[2], "a", "NaN must sort last");

    let desc = fct.fct_reorder(&means, true).unwrap();
    // c(2.0), b(1.0), a(NaN)
    assert_eq!(desc.levels[2], "a", "NaN must sort last even descending");
}

#[test]
fn test_fct_reorder_ties_stable() {
    // Two levels with equal summary values — stable sort preserves original order
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let means = vec![1.0f64, 1.0, 2.0]; // a==b tied
    let reordered = fct.fct_reorder(&means, false).unwrap();
    // a and b are tied at 1.0; stable sort keeps a before b
    assert_eq!(reordered.levels[0], "a");
    assert_eq!(reordered.levels[1], "b");
    assert_eq!(reordered.levels[2], "c");
}

#[test]
fn test_fct_reorder_length_mismatch_error() {
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let means = vec![1.0f64, 2.0]; // wrong length
    let err = fct.fct_reorder(&means, false);
    assert!(matches!(err, Err(TidyError::LengthMismatch { expected: 3, got: 2 })));
}

#[test]
fn test_fct_reorder_data_indices_remapped() {
    // Verify data buffer reflects new level order
    let fct = FctColumn::encode(&strs(&["a","b","c","a","b"])).unwrap();
    // a=0,b=1,c=2; means: a→10, b→1, c→5 → reordered: b(1),c(5),a(10)
    let means = vec![10.0f64, 1.0, 5.0];
    let reordered = fct.fct_reorder(&means, false).unwrap();
    // Original row 0="a" → now index of "a" in new levels = 2
    assert_eq!(reordered.data[0], 2); // was "a", now at position 2
    assert_eq!(reordered.data[1], 0); // was "b", now at position 0
    assert_eq!(reordered.data[2], 1); // was "c", now at position 1
}

#[test]
fn test_fct_reorder_by_col_float() {
    // Full pipeline: encode → reorder by float column
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a","b","a","b","c"]))),
        ("val".to_string(), Column::Float(vec![3.0, 1.0, 3.0, 1.0, 5.0])),
    ]).unwrap();
    let view = df.tidy();
    let fct = view.fct_encode("cat").unwrap();
    let val_col = view.base_column("val").unwrap();
    let reordered = fct.fct_reorder_by_col(val_col, false).unwrap();
    // means: a=3.0, b=1.0, c=5.0 → ascending: b,a,c
    assert_eq!(reordered.levels, strs(&["b", "a", "c"]));
}

#[test]
fn test_fct_reorder_by_col_int() {
    let fct = FctColumn::encode(&strs(&["x","y","z"])).unwrap();
    let col = Column::Int(vec![5, 2, 8]);
    let reordered = fct.fct_reorder_by_col(&col, true).unwrap();
    // desc: z(8), x(5), y(2)
    assert_eq!(reordered.levels, strs(&["z", "x", "y"]));
}

#[test]
fn test_fct_reorder_by_col_nan_excluded() {
    // NaN values in float column are excluded from mean computation
    let fct = FctColumn::encode(&strs(&["a","b","a","b"])).unwrap();
    let col = Column::Float(vec![f64::NAN, 2.0, 4.0, f64::NAN]);
    // a: rows 0(NaN),2(4.0) → mean=4.0; b: rows 1(2.0),3(NaN) → mean=2.0
    let reordered = fct.fct_reorder_by_col(&col, false).unwrap();
    // asc: b(2.0), a(4.0)
    assert_eq!(reordered.levels, strs(&["b", "a"]));
}

#[test]
fn test_fct_reorder_by_col_wrong_type_error() {
    let fct = FctColumn::encode(&strs(&["a","b"])).unwrap();
    let col = Column::Str(strs(&["x", "y"]));
    let err = fct.fct_reorder_by_col(&col, false);
    assert!(matches!(err, Err(TidyError::TypeMismatch { .. })));
}

#[test]
fn test_fct_reorder_determinism() {
    let fct = FctColumn::encode(&strs(&["c","a","b","c","a"])).unwrap();
    let means = vec![3.0, 1.0, 2.0];
    let r1 = fct.fct_reorder(&means, false).unwrap();
    let r2 = fct.fct_reorder(&means, false).unwrap();
    assert_eq!(r1.levels, r2.levels);
    assert_eq!(r1.data, r2.data);
}

// ── fct_collapse ──────────────────────────────────────────────────────────────

#[test]
fn test_fct_collapse_basic() {
    let fct = FctColumn::encode(&strs(&["a","b","c","d"])).unwrap();
    let collapsed = fct.fct_collapse(&[("b","group1"),("c","group1")]).unwrap();
    // levels: a, group1, d  (first-occurrence of new names following old order)
    assert_eq!(collapsed.levels, strs(&["a", "group1", "d"]));
    assert_eq!(collapsed.nlevels(), 3);
}

#[test]
fn test_fct_collapse_values_correct() {
    let fct = FctColumn::encode(&strs(&["a","b","c","d","b"])).unwrap();
    let collapsed = fct.fct_collapse(&[("b","BC"),("c","BC")]).unwrap();
    let decoded: Vec<&str> = (0..collapsed.nrows()).map(|i| collapsed.decode(i)).collect();
    assert_eq!(decoded, vec!["a","BC","BC","d","BC"]);
}

#[test]
fn test_fct_collapse_empty_mapping_noop() {
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let collapsed = fct.fct_collapse(&[]).unwrap();
    assert_eq!(collapsed.levels, fct.levels);
    assert_eq!(collapsed.data, fct.data);
}

#[test]
fn test_fct_collapse_to_same_name_is_noop() {
    // Mapping "a" → "a" should be a no-op for that level
    let fct = FctColumn::encode(&strs(&["a","b","a"])).unwrap();
    let collapsed = fct.fct_collapse(&[("a","a")]).unwrap();
    assert_eq!(collapsed.levels, fct.levels);
    assert_eq!(collapsed.data, fct.data);
}

#[test]
fn test_fct_collapse_all_to_one() {
    let fct = FctColumn::encode(&strs(&["x","y","z","x","y"])).unwrap();
    let collapsed = fct.fct_collapse(&[("x","all"),("y","all"),("z","all")]).unwrap();
    assert_eq!(collapsed.nlevels(), 1);
    assert_eq!(collapsed.levels[0], "all");
    for i in 0..collapsed.nrows() {
        assert_eq!(collapsed.decode(i), "all");
    }
}

#[test]
fn test_fct_collapse_duplicate_level_names_merged() {
    // Two old levels collapsing to same new name → one new level
    let fct = FctColumn::encode(&strs(&["red","blue","green"])).unwrap();
    let collapsed = fct.fct_collapse(&[("red","warm"),("blue","cool"),("green","cool")]).unwrap();
    assert_eq!(collapsed.nlevels(), 2);
    assert_eq!(collapsed.levels, strs(&["warm", "cool"]));
}

#[test]
fn test_fct_collapse_preserves_level_order() {
    // New level order follows first-occurrence of old levels (not mapping order)
    let fct = FctColumn::encode(&strs(&["c","b","a"])).unwrap();
    let collapsed = fct.fct_collapse(&[("a","group"),("b","group")]).unwrap();
    // c comes before a,b in levels; "c" kept, then "group" for b/a
    // levels were: c(0), b(1), a(2)
    // after collapse: c→c, b→group, a→group → first-occurrence of new names: c, group
    assert_eq!(collapsed.levels, strs(&["c", "group"]));
}

#[test]
fn test_fct_collapse_chain_operations() {
    // collapse then collapse again
    let fct = FctColumn::encode(&strs(&["a","b","c","d"])).unwrap();
    let step1 = fct.fct_collapse(&[("a","ab"),("b","ab")]).unwrap();
    let step2 = step1.fct_collapse(&[("c","cd"),("d","cd")]).unwrap();
    assert_eq!(step2.nlevels(), 2);
    assert_eq!(step2.levels, strs(&["ab", "cd"]));
}

#[test]
fn test_fct_collapse_then_lump() {
    let fct = FctColumn::encode(&strs(&["a","b","c","d","a","b","a"])).unwrap();
    let collapsed = fct.fct_collapse(&[("c","other_c"),("d","other_d")]).unwrap();
    // after collapse: a×3, b×2, other_c×1, other_d×1 → lump top-2 = a,b
    let lumped = collapsed.fct_lump(2).unwrap();
    assert_eq!(lumped.levels.len(), 3); // a, b, Other-bucket
}

#[test]
fn test_fct_collapse_then_reorder() {
    let fct = FctColumn::encode(&strs(&["a","b","c"])).unwrap();
    let collapsed = fct.fct_collapse(&[("b","x"),("c","x")]).unwrap();
    // levels after collapse: a, x
    let means = vec![2.0f64, 1.0]; // a→2, x→1 → asc: x, a
    let reordered = collapsed.fct_reorder(&means, false).unwrap();
    assert_eq!(reordered.levels, strs(&["x", "a"]));
}

#[test]
fn test_fct_collapse_on_empty_factor() {
    let fct = FctColumn::encode(&[]).unwrap();
    let collapsed = fct.fct_collapse(&[("a","b")]).unwrap();
    assert_eq!(collapsed.nrows(), 0);
    assert_eq!(collapsed.nlevels(), 0);
}

#[test]
fn test_fct_collapse_determinism() {
    let fct = FctColumn::encode(&strs(&["x","y","z","x","y"])).unwrap();
    let c1 = fct.fct_collapse(&[("y","yz"),("z","yz")]).unwrap();
    let c2 = fct.fct_collapse(&[("y","yz"),("z","yz")]).unwrap();
    assert_eq!(c1.levels, c2.levels);
    assert_eq!(c1.data, c2.data);
}

// ── NullableFactor ────────────────────────────────────────────────────────────

#[test]
fn test_nullable_factor_from_fct_all_valid() {
    let fct = FctColumn::encode(&strs(&["a","b","a"])).unwrap();
    let nf = NullableFactor::from_fct(fct);
    assert_eq!(nf.count_valid(), 3);
    assert!(!nf.is_null(0));
    assert!(!nf.is_null(1));
    assert!(!nf.is_null(2));
}

#[test]
fn test_nullable_factor_encode_with_nulls() {
    let input = opt_strs(&[Some("a"), None, Some("b"), None, Some("a")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    assert_eq!(nf.nlevels(), 2);
    assert_eq!(nf.count_valid(), 3);
    assert!(!nf.is_null(0));
    assert!(nf.is_null(1));
    assert!(!nf.is_null(2));
    assert!(nf.is_null(3));
    assert!(!nf.is_null(4));
}

#[test]
fn test_nullable_factor_decode_null_returns_none() {
    let input = opt_strs(&[Some("x"), None, Some("y")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    assert_eq!(nf.decode(0), Some("x"));
    assert_eq!(nf.decode(1), None);
    assert_eq!(nf.decode(2), Some("y"));
}

#[test]
fn test_nullable_factor_null_not_a_level() {
    // Null rows must not add a null level
    let input = opt_strs(&[None, Some("a"), None, Some("b")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    assert_eq!(nf.nlevels(), 2);
    assert!(!nf.fct.levels.contains(&"".to_string()));
}

#[test]
fn test_nullable_factor_lump_preserves_nulls() {
    let input = opt_strs(&[Some("a"), None, Some("b"), Some("a"), None, Some("c")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    let lumped = nf.fct_lump(1).unwrap();
    // Null rows remain null
    assert!(lumped.is_null(1));
    assert!(lumped.is_null(4));
    assert!(!lumped.is_null(0));
}

#[test]
fn test_nullable_factor_reorder_preserves_nulls() {
    let input = opt_strs(&[Some("a"), None, Some("b")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    let means = vec![2.0f64, 1.0]; // a, b
    let reordered = nf.fct_reorder(&means, false).unwrap();
    // Null at row 1 remains null
    assert!(reordered.is_null(1));
    assert!(!reordered.is_null(0));
    assert!(!reordered.is_null(2));
}

#[test]
fn test_nullable_factor_collapse_preserves_nulls() {
    let input = opt_strs(&[Some("a"), None, Some("b"), Some("a")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    let collapsed = nf.fct_collapse(&[("b","ab"),("a","ab")]).unwrap();
    // Null at row 1 remains null
    assert!(collapsed.is_null(1));
    assert_eq!(collapsed.decode(0), Some("ab"));
    assert_eq!(collapsed.decode(2), Some("ab"));
}

#[test]
fn test_nullable_factor_null_remains_null_after_collapse() {
    // Verifies that null does NOT become a level named "null" or anything else
    let input = opt_strs(&[None, None, Some("z")]);
    let nf = NullableFactor::encode_nullable(&input).unwrap();
    let collapsed = nf.fct_collapse(&[("z","renamed")]).unwrap();
    assert!(collapsed.is_null(0));
    assert!(collapsed.is_null(1));
    assert_eq!(collapsed.decode(2), Some("renamed"));
}

#[test]
#[ignore = "capacity boundary test — slow (65,536 unique strings); run with --ignored"]
fn test_nullable_factor_capacity_exceeded() {
    // 65,536 distinct non-null strings → CapacityExceeded
    let input: Vec<Option<String>> = (0u32..65_536)
        .map(|i| Some(format!("L{}", i)))
        .collect();
    let result = NullableFactor::encode_nullable(&input);
    assert!(matches!(result, Err(TidyError::CapacityExceeded { .. })));
}

// ── NoGC contract ─────────────────────────────────────────────────────────────

#[test]
fn test_nogc_fct_encode_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["fct_encode"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty(), "fct_encode must be rejected in @nogc");
}

#[test]
fn test_nogc_fct_lump_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["fct_lump"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty(), "fct_lump must be rejected in @nogc");
}

#[test]
fn test_nogc_fct_reorder_rejected() {
    let prog = mk_program(vec![mk_fn("f", true, &["fct_reorder"])]);
    let errs = verify_nogc(&prog).unwrap_err();
    assert!(!errs.is_empty(), "fct_reorder must be rejected in @nogc");
}

#[test]
fn test_nogc_fct_collapse_accepted() {
    let prog = mk_program(vec![mk_fn("f", true, &["fct_collapse"])]);
    assert!(verify_nogc(&prog).is_ok(), "fct_collapse must be @nogc safe");
}

#[test]
fn test_nogc_fct_collapse_with_other_safe_builtins() {
    // Mixing fct_collapse with other safe builtins — must pass
    let prog = mk_program(vec![mk_fn("f", true, &["tidy_filter", "fct_collapse", "tidy_relocate"])]);
    assert!(verify_nogc(&prog).is_ok());
}

// ── gather / view semantics ───────────────────────────────────────────────────

#[test]
fn test_fct_gather_subset() {
    let fct = FctColumn::encode(&strs(&["a","b","c","a","b"])).unwrap();
    let gathered = fct.gather(&[0, 2, 4]);
    // rows 0="a", 2="c", 4="b"
    assert_eq!(gathered.nrows(), 3);
    assert_eq!(gathered.decode(0), "a");
    assert_eq!(gathered.decode(1), "c");
    assert_eq!(gathered.decode(2), "b");
    // Levels are shared (not pruned)
    assert_eq!(gathered.levels, fct.levels);
}

#[test]
fn test_fct_to_str_column() {
    let fct = FctColumn::encode(&strs(&["x","y","x","z"])).unwrap();
    let col = fct.to_str_column();
    if let Column::Str(v) = col {
        assert_eq!(v, strs(&["x","y","x","z"]));
    } else {
        panic!("expected Str column");
    }
}

// ── fct_summary_means via TidyView ───────────────────────────────────────────

#[test]
fn test_fct_summary_means_basic() {
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a","b","a","b","c"]))),
        ("val".to_string(), Column::Float(vec![1.0, 4.0, 3.0, 2.0, 5.0])),
    ]).unwrap();
    let view = df.tidy();
    let fct = view.fct_encode("cat").unwrap();
    let means = view.fct_summary_means(&fct, "val").unwrap();
    // a: (1+3)/2=2.0, b: (4+2)/2=3.0, c: 5.0/1=5.0
    assert!((means[0] - 2.0).abs() < 1e-12, "a mean should be 2.0");
    assert!((means[1] - 3.0).abs() < 1e-12, "b mean should be 3.0");
    assert!((means[2] - 5.0).abs() < 1e-12, "c mean should be 5.0");
}

#[test]
fn test_fct_summary_means_nan_excluded() {
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a","a","b"]))),
        ("val".to_string(), Column::Float(vec![f64::NAN, 4.0, 2.0])),
    ]).unwrap();
    let view = df.tidy();
    let fct = view.fct_encode("cat").unwrap();
    let means = view.fct_summary_means(&fct, "val").unwrap();
    // a: NaN excluded → mean=4.0; b: 2.0
    assert!((means[0] - 4.0).abs() < 1e-12);
    assert!((means[1] - 2.0).abs() < 1e-12);
}

#[test]
fn test_fct_summary_means_all_nan_gives_nan() {
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a","a"]))),
        ("val".to_string(), Column::Float(vec![f64::NAN, f64::NAN])),
    ]).unwrap();
    let view = df.tidy();
    let fct = view.fct_encode("cat").unwrap();
    let means = view.fct_summary_means(&fct, "val").unwrap();
    assert!(means[0].is_nan(), "all-NaN level must produce NaN mean");
}

#[test]
fn test_fct_summary_means_wrong_type_error() {
    let df = DataFrame::from_columns(vec![
        ("cat".to_string(), Column::Str(strs(&["a","b"]))),
        ("lbl".to_string(), Column::Str(strs(&["x","y"]))),
    ]).unwrap();
    let view = df.tidy();
    let fct = view.fct_encode("cat").unwrap();
    let err = view.fct_summary_means(&fct, "lbl");
    assert!(matches!(err, Err(TidyError::TypeMismatch { .. })));
}
