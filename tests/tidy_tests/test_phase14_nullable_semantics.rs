// Phase 14: Nullable column semantics tests

use cjc_data::{BitMask, Column, NullableColumn, NullCol};

// ── NullableColumn basic ───────────────────────────────────────────────────

#[test]
fn test_nullable_column_from_values_all_valid() {
    let nc: NullableColumn<i64> = NullableColumn::from_values(vec![1, 2, 3]);
    assert!(!nc.is_null(0));
    assert!(!nc.is_null(1));
    assert!(!nc.is_null(2));
    assert_eq!(nc.count_valid(), 3);
}

#[test]
fn test_nullable_column_explicit_validity() {
    let nc: NullableColumn<i64> = NullableColumn::new(
        vec![1, 0, 3],
        BitMask::from_bools(&[true, false, true]),
    );
    assert!(!nc.is_null(0));
    assert!(nc.is_null(1));
    assert!(!nc.is_null(2));
    assert_eq!(nc.count_valid(), 2);
}

#[test]
fn test_nullable_column_get() {
    let nc: NullableColumn<i64> = NullableColumn::new(
        vec![42, 0],
        BitMask::from_bools(&[true, false]),
    );
    assert_eq!(nc.get(0), Some(&42));
    assert_eq!(nc.get(1), None);
}

#[test]
fn test_nullable_column_gather() {
    let nc: NullableColumn<i64> = NullableColumn::new(
        vec![10, 20, 30],
        BitMask::from_bools(&[true, false, true]),
    );
    let gathered = nc.gather(&[0, 2]);
    assert!(!gathered.is_null(0));
    assert!(!gathered.is_null(1));
    assert_eq!(gathered.values, vec![10, 30]);
}

#[test]
fn test_nullable_column_gather_with_null() {
    let nc: NullableColumn<i64> = NullableColumn::new(
        vec![10, 20, 30],
        BitMask::from_bools(&[true, false, true]),
    );
    // Gather index 1 (which is null)
    let gathered = nc.gather(&[0, 1]);
    assert!(!gathered.is_null(0));
    assert!(gathered.is_null(1));
}

// ── NullCol enum ──────────────────────────────────────────────────────────

#[test]
fn test_nullcol_from_column_fully_valid() {
    let col = Column::Int(vec![1, 2, 3]);
    let nc = NullCol::from_column(&col);
    for i in 0..3 {
        assert!(!nc.is_null(i));
    }
}

#[test]
fn test_nullcol_to_column_strict_all_valid() {
    let col = Column::Float(vec![1.0, 2.0]);
    let nc = NullCol::from_column(&col);
    let result = nc.to_column_strict().unwrap();
    assert!(matches!(result, Column::Float(_)));
}

#[test]
fn test_nullcol_to_column_strict_with_null_errors() {
    let nc = NullCol::Int(NullableColumn::new(
        vec![1, 0],
        BitMask::from_bools(&[true, false]),
    ));
    assert!(nc.to_column_strict().is_err());
}

#[test]
fn test_nullcol_to_column_filled_float_nan() {
    let nc = NullCol::Float(NullableColumn::new(
        vec![1.0, 0.0],
        BitMask::from_bools(&[true, false]),
    ));
    let col = nc.to_column_filled();
    if let Column::Float(v) = col {
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());
    } else {
        panic!("expected Float");
    }
}

#[test]
fn test_nullcol_type_name() {
    assert_eq!(NullCol::from_column(&Column::Int(vec![])).type_name(), "Int");
    assert_eq!(NullCol::from_column(&Column::Float(vec![])).type_name(), "Float");
    assert_eq!(NullCol::from_column(&Column::Str(vec![])).type_name(), "Str");
    assert_eq!(NullCol::from_column(&Column::Bool(vec![])).type_name(), "Bool");
}

#[test]
fn test_nullcol_null_display() {
    let nc = NullCol::Int(NullableColumn::new(
        vec![42, 0],
        BitMask::from_bools(&[true, false]),
    ));
    assert_eq!(nc.get_display(0), "42");
    assert_eq!(nc.get_display(1), "null");
}

#[test]
fn test_nullcol_null_of_type_all_null() {
    let nc = NullCol::null_of_type("Int", 5);
    for i in 0..5 {
        assert!(nc.is_null(i), "expected null at {}", i);
    }
}

// ── Null fill in pivot_wider (null semantics) ─────────────────────────────

#[test]
fn test_nullcol_validity_bitmap_tail_safe() {
    // Validity bitmap for 65 elements — two words; tail bits must be clean
    let vals: Vec<i64> = (0..65).collect();
    let mut bools = vec![true; 65];
    bools[64] = false; // last element null
    let nc: NullableColumn<i64> = NullableColumn::new(vals, BitMask::from_bools(&bools));
    assert!(!nc.is_null(63));
    assert!(nc.is_null(64));
    assert_eq!(nc.count_valid(), 64);
}

// ── NullableFrame conversions ─────────────────────────────────────────────

#[test]
fn test_nullable_frame_to_dataframe_filled() {
    use cjc_data::NullableFrame;
    let mut frame = NullableFrame::new();
    let nc = NullCol::Int(NullableColumn::new(
        vec![10, 0, 30],
        BitMask::from_bools(&[true, false, true]),
    ));
    frame.columns.push(("x".into(), nc));
    let df = frame.to_dataframe_filled();
    // null position gets 0 fill for Int
    if let Column::Int(v) = df.get_column("x").unwrap() {
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 0); // null fill = 0
        assert_eq!(v[2], 30);
    }
}

#[test]
fn test_nullable_frame_column_names() {
    use cjc_data::NullableFrame;
    let mut frame = NullableFrame::new();
    frame.columns.push(("a".into(), NullCol::from_column(&Column::Int(vec![1]))));
    frame.columns.push(("b".into(), NullCol::from_column(&Column::Float(vec![1.0]))));
    assert_eq!(frame.column_names(), vec!["a", "b"]);
}
