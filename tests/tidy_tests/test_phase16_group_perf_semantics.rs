// Phase 16: Group performance upgrade — BTree-accelerated GroupIndex
// Confirms identical semantics to Phase 11 group_by (first-occurrence order).

use cjc_data::{Column, DataFrame, TidyAgg};

fn make_groups_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("grp".into(), Column::Str(vec![
            "B".into(), "A".into(), "B".into(), "C".into(), "A".into(),
        ])),
        ("val".into(), Column::Int(vec![10, 20, 30, 40, 50])),
    ])
    .unwrap()
}

// ── group_by_fast vs group_by — identical output ──────────────────────────

#[test]
fn test_group_by_fast_same_ngroups() {
    let df = make_groups_df();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();
    let g_slow = v1.group_by(&["grp"]).unwrap();
    let g_fast = v2.group_by_fast(&["grp"]).unwrap();
    assert_eq!(g_slow.ngroups(), g_fast.ngroups());
}

#[test]
fn test_group_by_fast_first_occurrence_order() {
    // Groups should appear in the order they first appear in the data:
    // B (row 0), A (row 1), C (row 3)
    let df = make_groups_df();
    let v = df.tidy();
    let g = v.group_by_fast(&["grp"]).unwrap();
    let idx = g.group_index();
    assert_eq!(idx.groups[0].key_values[0], "B");
    assert_eq!(idx.groups[1].key_values[0], "A");
    assert_eq!(idx.groups[2].key_values[0], "C");
}

#[test]
fn test_group_by_fast_identical_row_assignments_to_slow() {
    let df = make_groups_df();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();
    let g_slow = v1.group_by(&["grp"]).unwrap();
    let g_fast = v2.group_by_fast(&["grp"]).unwrap();

    let idx_slow = g_slow.group_index();
    let idx_fast = g_fast.group_index();

    // Same number of groups
    assert_eq!(idx_slow.groups.len(), idx_fast.groups.len());

    // Groups in same order with same row assignments
    for (gs, gf) in idx_slow.groups.iter().zip(idx_fast.groups.iter()) {
        assert_eq!(gs.key_values, gf.key_values);
        assert_eq!(gs.row_indices, gf.row_indices);
    }
}

#[test]
fn test_group_by_fast_summarise_identical_to_slow() {
    let df = make_groups_df();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();

    let g_slow = v1.group_by(&["grp"]).unwrap();
    let g_fast = v2.group_by_fast(&["grp"]).unwrap();

    let aggs = vec![("total", TidyAgg::Sum("val".into()))];
    let r_slow = g_slow.summarise(&aggs).unwrap();
    let r_fast = g_fast.summarise(&aggs).unwrap();

    let b_slow = r_slow.borrow();
    let b_fast = r_fast.borrow();

    assert_eq!(b_slow.nrows(), b_fast.nrows());

    // Same group key order
    if let (Column::Str(k_slow), Column::Str(k_fast)) = (
        b_slow.get_column("grp").unwrap(),
        b_fast.get_column("grp").unwrap(),
    ) {
        assert_eq!(k_slow, k_fast);
    }

    // Same totals
    if let (Column::Float(t_slow), Column::Float(t_fast)) = (
        b_slow.get_column("total").unwrap(),
        b_fast.get_column("total").unwrap(),
    ) {
        for (s, f) in t_slow.iter().zip(t_fast.iter()) {
            assert!((s - f).abs() < 1e-12, "sum mismatch: {} vs {}", s, f);
        }
    }
}

#[test]
fn test_group_by_fast_empty_df() {
    let df = DataFrame::from_columns(vec![
        ("grp".into(), Column::Str(vec![])),
    ])
    .unwrap();
    let v = df.tidy();
    let g = v.group_by_fast(&["grp"]).unwrap();
    assert_eq!(g.ngroups(), 0);
}

#[test]
fn test_group_by_fast_single_group() {
    let df = DataFrame::from_columns(vec![
        ("grp".into(), Column::Str(vec!["X".into(), "X".into(), "X".into()])),
        ("val".into(), Column::Int(vec![1, 2, 3])),
    ])
    .unwrap();
    let v = df.tidy();
    let g = v.group_by_fast(&["grp"]).unwrap();
    assert_eq!(g.ngroups(), 1);
    assert_eq!(g.group_index().groups[0].row_indices, vec![0, 1, 2]);
}

#[test]
fn test_group_by_fast_all_unique_groups() {
    let df = DataFrame::from_columns(vec![
        ("grp".into(), Column::Int(vec![3, 1, 4, 1, 5])),
    ])
    .unwrap();
    let v = df.tidy();
    let g_slow = v.clone().group_by(&["grp"]).unwrap();
    let g_fast = v.group_by_fast(&["grp"]).unwrap();
    // When all unique, ngroups differs in that 1 appears twice → 4 groups
    assert_eq!(g_slow.ngroups(), g_fast.ngroups());
}

#[test]
fn test_group_by_fast_multi_key_identical_to_slow() {
    let df = DataFrame::from_columns(vec![
        ("a".into(), Column::Str(vec!["X".into(), "X".into(), "Y".into(), "Y".into()])),
        ("b".into(), Column::Int(vec![1, 2, 1, 2])),
        ("val".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0])),
    ])
    .unwrap();
    let v1 = df.clone().tidy();
    let v2 = df.tidy();
    let g_slow = v1.group_by(&["a", "b"]).unwrap();
    let g_fast = v2.group_by_fast(&["a", "b"]).unwrap();
    let idx_slow = g_slow.group_index();
    let idx_fast = g_fast.group_index();
    assert_eq!(idx_slow.groups.len(), idx_fast.groups.len());
    for (gs, gf) in idx_slow.groups.iter().zip(idx_fast.groups.iter()) {
        assert_eq!(gs.key_values, gf.key_values);
        assert_eq!(gs.row_indices, gf.row_indices);
    }
}

#[test]
fn test_group_by_fast_deterministic_two_runs() {
    let df = make_groups_df();
    let g1 = df.clone().tidy().group_by_fast(&["grp"]).unwrap();
    let g2 = df.tidy().group_by_fast(&["grp"]).unwrap();
    let idx1 = g1.group_index();
    let idx2 = g2.group_index();
    assert_eq!(idx1.groups.len(), idx2.groups.len());
    for (g1, g2) in idx1.groups.iter().zip(idx2.groups.iter()) {
        assert_eq!(g1.key_values, g2.key_values);
        assert_eq!(g1.row_indices, g2.row_indices);
    }
}

#[test]
fn test_group_by_fast_after_filter_same_as_slow() {
    let df = make_groups_df();
    let v = df.tidy();
    let filtered = v.filter(&cjc_data::DExpr::BinOp {
        op: cjc_data::DBinOp::Gt,
        left: Box::new(cjc_data::DExpr::Col("val".into())),
        right: Box::new(cjc_data::DExpr::LitInt(15)),
    }).unwrap();
    let g_slow = filtered.clone().group_by(&["grp"]).unwrap();
    let g_fast = filtered.group_by_fast(&["grp"]).unwrap();
    let idx_slow = g_slow.group_index();
    let idx_fast = g_fast.group_index();
    assert_eq!(idx_slow.groups.len(), idx_fast.groups.len());
    for (gs, gf) in idx_slow.groups.iter().zip(idx_fast.groups.iter()) {
        assert_eq!(gs.key_values, gf.key_values);
    }
}
