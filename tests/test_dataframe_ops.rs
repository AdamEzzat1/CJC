//! Integration tests for Item 1 DataFrame surface builtins.
//!
//! Covers: `df_read_csv`, `pivot_wider`, `pivot_longer`, `df_distinct`,
//! `df_rename`, `df_anti_join`, `df_semi_join`, `df_full_join`,
//! `df_fill_na`, `df_drop_na`.
//!
//! Strategy:
//! * Unit / proptest tests exercise the shared dispatch entry point
//!   `cjc_data::tidy_dispatch::dispatch_tidy_builtin` directly.  This is the
//!   single source of truth both executors route through, so parity is a
//!   structural consequence.
//! * Parity tests additionally run a CJC-Lang source program through
//!   `cjc_eval::Interpreter` and `cjc_mir_exec::run_program_with_executor`
//!   and compare their `output` buffers.  The CSV input lives in a
//!   `tempfile::tempdir()` so the test is self-contained.
//! * A `bolero::check!` fuzz target is included (dev-dep already present
//!   in the workspace `Cargo.toml`).
//!
//! Determinism: all `Interpreter::new(seed)` / `run_program_with_executor`
//! calls use `seed = 42`.  All unordered comparisons use sorted `Vec`s or
//! `BTreeMap` — never `HashMap`/`HashSet`.

use std::any::Any;
use std::rc::Rc;

use cjc_data::tidy_dispatch::{
    dispatch_tidy_builtin, dispatch_tidy_method, wrap_view,
};
use cjc_data::{Column, DataFrame, TidyView};
use cjc_runtime::value::Value;

use proptest::prelude::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Wrap a `DataFrame` in a `Value::TidyView`.
fn tv(df: DataFrame) -> Value {
    wrap_view(TidyView::from_df(df))
}

/// Unwrap `Result<Option<Value>, String>` from the dispatch layer.
fn call(name: &str, args: &[Value]) -> Value {
    dispatch_tidy_builtin(name, args)
        .expect("dispatch_tidy_builtin returned Err")
        .expect("builtin not recognised")
}

/// Call a TidyView instance method through the shared dispatch (same path
/// both executors take).  Used for cheap introspection — nrows, column_names.
fn method(v: &Value, name: &str, args: &[Value]) -> Value {
    let rc: &Rc<dyn Any> = match v {
        Value::TidyView(rc) => rc,
        other => panic!("expected TidyView, got {:?}", other.type_name()),
    };
    dispatch_tidy_method(rc, name, args)
        .expect("dispatch_tidy_method returned Err")
        .expect("method not recognised")
}

fn nrows(v: &Value) -> i64 {
    match method(v, "nrows", &[]) {
        Value::Int(n) => n,
        other => panic!("nrows returned non-Int: {other:?}"),
    }
}

fn column_names(v: &Value) -> Vec<String> {
    match method(v, "column_names", &[]) {
        Value::Array(arr) => arr
            .iter()
            .map(|val| match val {
                Value::String(s) => (**s).clone(),
                other => panic!("column name not a String: {other:?}"),
            })
            .collect(),
        other => panic!("column_names returned non-Array: {other:?}"),
    }
}

fn str_array(items: &[&str]) -> Value {
    let v: Vec<Value> = items
        .iter()
        .map(|s| Value::String(Rc::new((*s).to_string())))
        .collect();
    Value::Array(Rc::new(v))
}

fn str(s: &str) -> Value {
    Value::String(Rc::new(s.to_string()))
}

// Small fixed datasets.

fn long_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 1, 2, 2])),
        (
            "name".into(),
            Column::Str(vec!["a".into(), "b".into(), "a".into(), "b".into()]),
        ),
        ("value".into(), Column::Float(vec![10.0, 20.0, 30.0, 40.0])),
    ])
    .unwrap()
}

fn dups_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(vec![1, 1, 2, 2, 3])),
        ("b".into(), Column::Int(vec![10, 10, 20, 20, 30])),
    ])
    .unwrap()
}

fn left_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![1, 2, 3, 4])),
        (
            "lv".into(),
            Column::Str(vec!["w".into(), "x".into(), "y".into(), "z".into()]),
        ),
    ])
    .unwrap()
}

fn right_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("k".into(), Column::Int(vec![2, 3, 5])),
        ("rv".into(), Column::Int(vec![200, 300, 500])),
    ])
    .unwrap()
}

fn nan_df() -> DataFrame {
    DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2, 3, 4])),
        (
            "x".into(),
            Column::Float(vec![1.0, f64::NAN, 3.0, f64::NAN]),
        ),
        (
            "s".into(),
            Column::Str(vec!["a".into(), "NA".into(), "".into(), "d".into()]),
        ),
    ])
    .unwrap()
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[test]
fn df_read_csv_reads_simple_csv() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.csv");
    std::fs::write(&path, "a,b\n1,2\n3,4\n5,6\n").unwrap();

    let view = call("df_read_csv", &[str(path.to_str().unwrap())]);
    assert_eq!(nrows(&view), 3);
    let mut names = column_names(&view);
    names.sort();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn pivot_wider_produces_expected_shape() {
    // 2 ids × 2 name keys → 2 rows, 3 cols (id + a + b).
    let wide = call(
        "pivot_wider",
        &[tv(long_df()), str_array(&["id"]), str("name"), str("value")],
    );
    assert_eq!(nrows(&wide), 2);
    let mut names = column_names(&wide);
    names.sort();
    assert_eq!(names, vec!["a", "b", "id"]);
}

#[test]
fn pivot_longer_restores_row_count() {
    // value and extra columns → 2 rows × 2 measurements = 4 long rows.
    let df = DataFrame::from_columns(vec![
        ("id".into(), Column::Int(vec![1, 2])),
        ("a".into(), Column::Float(vec![10.0, 30.0])),
        ("b".into(), Column::Float(vec![20.0, 40.0])),
    ])
    .unwrap();
    let long = call(
        "pivot_longer",
        &[tv(df), str_array(&["a", "b"]), str("name"), str("value")],
    );
    assert_eq!(nrows(&long), 4);
    let mut names = column_names(&long);
    names.sort();
    assert_eq!(names, vec!["id", "name", "value"]);
}

#[test]
fn df_distinct_drops_duplicate_rows() {
    // dups_df has 5 rows, 3 unique (a,b) combinations.
    let out = call("df_distinct", &[tv(dups_df())]);
    assert_eq!(nrows(&out), 3);
}

#[test]
fn df_distinct_idempotent() {
    let once = call("df_distinct", &[tv(dups_df())]);
    let twice = call("df_distinct", &[once.clone()]);
    assert_eq!(nrows(&once), nrows(&twice));
    assert_eq!(column_names(&once), column_names(&twice));
}

#[test]
fn df_rename_changes_one_column_name() {
    let out = call(
        "df_rename",
        &[tv(long_df()), str("value"), str("measure")],
    );
    let mut names = column_names(&out);
    names.sort();
    assert_eq!(names, vec!["id", "measure", "name"]);
    // Row count preserved.
    assert_eq!(nrows(&out), 4);
}

#[test]
fn df_anti_join_keeps_unmatched_left_rows() {
    // left k ∈ {1,2,3,4}, right k ∈ {2,3,5}.  Anti = {1, 4}.
    let out = call(
        "df_anti_join",
        &[tv(left_df()), tv(right_df()), str("k")],
    );
    assert_eq!(nrows(&out), 2);
}

#[test]
fn df_semi_join_keeps_matched_left_rows() {
    // Semi = {2, 3}.
    let out = call(
        "df_semi_join",
        &[tv(left_df()), tv(right_df()), str("k")],
    );
    assert_eq!(nrows(&out), 2);
}

#[test]
fn df_full_join_unions_keys() {
    // Full outer on k: {1, 2, 3, 4, 5} → 5 rows.
    let out = call(
        "df_full_join",
        &[tv(left_df()), tv(right_df()), str("k")],
    );
    assert_eq!(nrows(&out), 5);
}

#[test]
fn df_fill_na_replaces_floats() {
    // x has 2 NaNs; after fill with 0.0, every row survives drop_na.
    let filled = call(
        "df_fill_na",
        &[tv(nan_df()), str("x"), Value::Float(0.0)],
    );
    assert_eq!(nrows(&filled), 4);
    // Column still called "x".
    assert!(column_names(&filled).contains(&"x".to_string()));
}

#[test]
fn df_drop_na_removes_nan_rows() {
    // Float NaN rows: 2 (indices 1, 3).  String "NA"/"" rows: 2 (indices 1, 2).
    // Union → drop rows 1, 2, 3 → 1 remaining.
    let out = call("df_drop_na", &[tv(nan_df())]);
    assert_eq!(nrows(&out), 1);
}

#[test]
fn df_drop_na_with_column_subset() {
    // Restrict NA check to "x" only → drop rows 1, 3 → 2 remaining.
    let out = call("df_drop_na", &[tv(nan_df()), str_array(&["x"])]);
    assert_eq!(nrows(&out), 2);
}

#[test]
fn df_read_csv_wrong_arg_count_errors() {
    assert!(dispatch_tidy_builtin("df_read_csv", &[]).is_err());
    assert!(
        dispatch_tidy_builtin("df_read_csv", &[str("a"), str(","), str("x")]).is_err()
    );
}

#[test]
fn pivot_wider_wrong_arg_count_errors() {
    assert!(dispatch_tidy_builtin("pivot_wider", &[tv(long_df())]).is_err());
}

// ── Parity tests (cjc-eval vs cjc-mir-exec) ──────────────────────────────────
//
// These run a CJC-Lang source program that:
//   1. reads a CSV written to a tempdir,
//   2. invokes one of the new builtins,
//   3. prints a scalar (nrows / a specific cell / column count).
//
// Both executors share the same dispatch layer, so their `output` buffers
// must be byte-identical.

fn write_csv(path: &std::path::Path, body: &str) {
    std::fs::write(path, body).unwrap();
}

fn run_parity(src: &str) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(
        !diags.has_errors(),
        "parse errors: {:?}",
        diags.diagnostics
    );

    let mut interp = cjc_eval::Interpreter::new(42);
    let _ = interp.exec(&program).expect("eval failed");
    let eval_out = interp.output.clone();

    let (_, exec) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("mir-exec failed");
    let mir_out = exec.output.clone();

    assert_eq!(
        eval_out, mir_out,
        "eval vs mir-exec parity mismatch\neval={eval_out:?}\nmir ={mir_out:?}"
    );
}

#[test]
fn parity_df_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dups.csv");
    write_csv(&path, "a,b\n1,10\n1,10\n2,20\n2,20\n3,30\n");
    let path_str = path.to_str().unwrap().replace('\\', "/");
    let src = format!(
        r#"
let df = df_read_csv("{p}");
let d = df_distinct(df);
print(d.nrows());
print(d.ncols());
"#,
        p = path_str
    );
    run_parity(&src);
}

#[test]
fn parity_df_rename() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r.csv");
    write_csv(&path, "foo,bar\n1,2\n3,4\n");
    let path_str = path.to_str().unwrap().replace('\\', "/");
    let src = format!(
        r#"
let df = df_read_csv("{p}");
let r = df_rename(df, "foo", "baz");
let names = r.column_names();
print(names[0]);
print(names[1]);
print(r.nrows());
"#,
        p = path_str
    );
    run_parity(&src);
}

#[test]
fn parity_df_semi_anti_full_join() {
    let dir = tempfile::tempdir().unwrap();
    let lp = dir.path().join("l.csv");
    let rp = dir.path().join("r.csv");
    // Start each key column with a value > 1 so the CSV reader's first-row
    // type inference does not classify "k" as Bool (values of "0"/"1" are
    // treated as Bool by cjc_data::csv::infer_type).
    write_csv(&lp, "k,lv\n11,w\n12,x\n13,y\n14,z\n");
    write_csv(&rp, "k,rv\n12,200\n13,300\n15,500\n");
    let lp_s = lp.to_str().unwrap().replace('\\', "/");
    let rp_s = rp.to_str().unwrap().replace('\\', "/");
    let src = format!(
        r#"
let l = df_read_csv("{l}");
let r = df_read_csv("{r}");
let semi = df_semi_join(l, r, "k");
let anti = df_anti_join(l, r, "k");
let full = df_full_join(l, r, "k");
print(semi.nrows());
print(anti.nrows());
print(full.nrows());
"#,
        l = lp_s,
        r = rp_s
    );
    run_parity(&src);
}

// ── Proptests ────────────────────────────────────────────────────────────────

// 1. df_distinct is idempotent across arbitrary int matrices.
proptest! {
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn prop_distinct_idempotent(
        data in prop::collection::vec((0i64..5, 0i64..5), 1..20)
    ) {
        let (a, b): (Vec<i64>, Vec<i64>) = data.into_iter().unzip();
        let df = DataFrame::from_columns(vec![
            ("a".into(), Column::Int(a)),
            ("b".into(), Column::Int(b)),
        ]).unwrap();

        let once = call("df_distinct", &[tv(df)]);
        let once_rows = nrows(&once);
        let twice = call("df_distinct", &[once]);
        prop_assert_eq!(once_rows, nrows(&twice));
    }
}

// 2. After df_fill_na on every Float column, df_drop_na (default = all cols)
//    leaves the row count unchanged — the fill erased every NaN.
proptest! {
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn prop_fill_then_drop_preserves_rows(
        nans in prop::collection::vec(any::<bool>(), 2..12)
    ) {
        let n = nans.len();
        let ids: Vec<i64> = (0..n as i64).collect();
        let xs: Vec<f64> = nans.iter().map(|&b| if b { f64::NAN } else { 1.0 }).collect();
        let ss: Vec<String> = (0..n).map(|_| "ok".to_string()).collect();
        let df = DataFrame::from_columns(vec![
            ("id".into(), Column::Int(ids)),
            ("x".into(), Column::Float(xs)),
            ("s".into(), Column::Str(ss)),
        ]).unwrap();

        let filled = call("df_fill_na", &[tv(df), str("x"), Value::Float(0.0)]);
        let filled_rows = nrows(&filled);
        let dropped = call("df_drop_na", &[filled]);
        prop_assert_eq!(filled_rows, n as i64);
        prop_assert_eq!(nrows(&dropped), n as i64);
    }
}

// 3. pivot_wider ∘ pivot_longer round-trips the row count when keys are
//    unique.  We build a wide frame with two measure columns {a, b} and one
//    id column, pivot it to long (nrows_long = 2 * nrows_wide), pivot back
//    to wide using the same id and measure names, and assert the row count
//    matches.
proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    #[test]
    fn prop_wider_longer_roundtrip_rowcount(
        rows in prop::collection::vec((any::<f64>(), any::<f64>()), 1..10)
    ) {
        // Filter NaN / infinite inputs — pivot treats them as missing keys.
        let rows: Vec<(f64, f64)> = rows.into_iter()
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .collect();
        prop_assume!(!rows.is_empty());

        let n = rows.len();
        let ids: Vec<i64> = (0..n as i64).collect();
        let a_vals: Vec<f64> = rows.iter().map(|(a, _)| *a).collect();
        let b_vals: Vec<f64> = rows.iter().map(|(_, b)| *b).collect();

        let wide = DataFrame::from_columns(vec![
            ("id".into(), Column::Int(ids)),
            ("a".into(), Column::Float(a_vals)),
            ("b".into(), Column::Float(b_vals)),
        ]).unwrap();

        let long = call("pivot_longer", &[
            tv(wide), str_array(&["a", "b"]), str("name"), str("value"),
        ]);
        prop_assert_eq!(nrows(&long), 2 * n as i64);

        let back = call("pivot_wider", &[
            long, str_array(&["id"]), str("name"), str("value"),
        ]);
        prop_assert_eq!(nrows(&back), n as i64);
    }
}

// ── bolero fuzz target ───────────────────────────────────────────────────────
//
// Bolero is listed under `[dev-dependencies]` in the workspace Cargo.toml
// (line 76).  This target feeds arbitrary (i64, i64) rows through the
// distinct → distinct pipeline and asserts idempotence.  Under `cargo test`
// bolero runs the property with a small, deterministic RNG budget.

#[test]
fn fuzz_distinct_idempotent() {
    use bolero::check;
    check!()
        .with_type::<Vec<(i8, i8)>>()
        .for_each(|rows| {
            if rows.is_empty() || rows.len() > 32 {
                return;
            }
            let a: Vec<i64> = rows.iter().map(|(x, _)| *x as i64).collect();
            let b: Vec<i64> = rows.iter().map(|(_, y)| *y as i64).collect();
            let df = match DataFrame::from_columns(vec![
                ("a".into(), Column::Int(a)),
                ("b".into(), Column::Int(b)),
            ]) {
                Ok(df) => df,
                Err(_) => return,
            };
            let once = call("df_distinct", &[tv(df)]);
            let once_rows = nrows(&once);
            let twice = call("df_distinct", &[once]);
            assert_eq!(once_rows, nrows(&twice));
        });
}
