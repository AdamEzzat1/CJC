// Phase 0 — Tidyverse integration parity (v3 audit)
//
// In CJC-Lang there is no separate "tidyverse" layer above TidyView; the
// language-level surface (`view.filter(...)`, `view.group_by(...)` from
// `.cjcl` source) is dispatched through `cjc_data::tidy_dispatch::
// dispatch_tidy_method` which calls the same `TidyView` Rust API used by
// internal callers. This file pins that 1:1 forwarding contract: for every
// representative pipeline, the language dispatch must produce the same
// AdaptiveSelection and the same materialised data as the direct Rust API.
//
// The dispatch path is bit-equivalent by construction (it literally
// forwards). These tests guard against future regressions where someone
// might add an extra `materialize()` or transform in the dispatch layer
// that would silently break adaptive selection for language users.
//
// Why these tests sit here (not in a separate file): they exercise the
// SAME runtime as the v2.0/v2.1/v2.2 parity tests and need the same
// regression coverage when AdaptiveSelection or predicate_bytecode change.

use cjc_data::tidy_dispatch::{
    dispatch_grouped_method, dispatch_tidy_method, value_to_dexpr, wrap_view,
};
use cjc_data::{ArrangeKey, Column, DBinOp, DExpr, DataFrame, TidyAgg, TidyView};
use cjc_runtime::value::Value;
use std::any::Any;
use std::collections::BTreeMap;
use std::rc::Rc;

// ── helpers ─────────────────────────────────────────────────────────────────

fn col_int(name: &str, xs: Vec<i64>) -> DataFrame {
    DataFrame::from_columns(vec![(name.into(), Column::Int(xs))]).unwrap()
}

fn df_two_int(a: Vec<i64>, b: Vec<i64>) -> DataFrame {
    DataFrame::from_columns(vec![
        ("a".into(), Column::Int(a)),
        ("b".into(), Column::Int(b)),
    ])
    .unwrap()
}

fn pred_lt(col: &str, v: i64) -> DExpr {
    DExpr::BinOp {
        op: DBinOp::Lt,
        left: Box::new(DExpr::Col(col.into())),
        right: Box::new(DExpr::LitInt(v)),
    }
}

fn dexpr_value(e: &DExpr) -> Value {
    // Build a Value::Struct that round-trips through `value_to_dexpr`.
    let mut fields: BTreeMap<String, Value> = BTreeMap::new();
    match e {
        DExpr::Col(name) => {
            fields.insert("kind".into(), Value::String(Rc::new("col".into())));
            fields.insert("value".into(), Value::String(Rc::new(name.clone())));
        }
        DExpr::LitInt(i) => {
            fields.insert("kind".into(), Value::String(Rc::new("lit_int".into())));
            fields.insert("value".into(), Value::Int(*i));
        }
        DExpr::BinOp { op, left, right } => {
            fields.insert("kind".into(), Value::String(Rc::new("binop".into())));
            let op_str = match op {
                DBinOp::Lt => "<",
                DBinOp::Gt => ">",
                DBinOp::Le => "<=",
                DBinOp::Ge => ">=",
                DBinOp::Eq => "==",
                DBinOp::Ne => "!=",
                DBinOp::And => "and",
                DBinOp::Or => "or",
                DBinOp::Add => "+",
                DBinOp::Sub => "-",
                DBinOp::Mul => "*",
                DBinOp::Div => "/",
            };
            fields.insert("op".into(), Value::String(Rc::new(op_str.into())));
            fields.insert("left".into(), dexpr_value(left));
            fields.insert("right".into(), dexpr_value(right));
        }
        DExpr::LitFloat(f) => {
            fields.insert("kind".into(), Value::String(Rc::new("lit_float".into())));
            fields.insert("value".into(), Value::Float(*f));
        }
        DExpr::LitBool(b) => {
            fields.insert("kind".into(), Value::String(Rc::new("lit_bool".into())));
            fields.insert("value".into(), Value::Bool(*b));
        }
        _ => panic!("dexpr_value: unsupported DExpr variant for parity test"),
    }
    let _ = value_to_dexpr; // silence unused-import lint when feature trims
    Value::Struct {
        name: "DExpr".to_string(),
        fields,
    }
}

fn str_array(items: &[&str]) -> Value {
    let v: Vec<Value> = items
        .iter()
        .map(|s| Value::String(Rc::new((*s).to_string())))
        .collect();
    Value::Array(Rc::new(v))
}

fn extract_view(v: Value) -> Rc<TidyView> {
    match v {
        Value::TidyView(rc) => rc.downcast::<TidyView>().expect("downcast TidyView"),
        other => panic!("expected Value::TidyView, got {:?}", other.type_name()),
    }
}

fn extract_grouped(v: Value) -> Value {
    // Grouped values pass straight back to dispatch_grouped_method, no need
    // to downcast in tests — keep as-is.
    v
}

fn dispatch_view_chained(start: TidyView, calls: &[(&str, Vec<Value>)]) -> Rc<TidyView> {
    let mut current_inner: Rc<dyn Any> = Rc::new(start) as Rc<dyn Any>;
    for (method, args) in calls {
        let out = dispatch_tidy_method(&current_inner, method, args)
            .expect("dispatch_tidy_method ok")
            .expect("method recognised");
        current_inner = match out {
            Value::TidyView(rc) => rc,
            _ => unreachable!("non-view output mid-chain"),
        };
    }
    current_inner.downcast::<TidyView>().expect("final downcast")
}

// Compare two views by visible row indices and per-cell values.
fn assert_views_equal(label: &str, a: &TidyView, b: &TidyView) {
    let a_idx: Vec<usize> = a.selection().iter_indices().collect();
    let b_idx: Vec<usize> = b.selection().iter_indices().collect();
    assert_eq!(
        a_idx, b_idx,
        "{label}: visible row indices diverge\n direct={a_idx:?}\n dispatch={b_idx:?}"
    );

    let a_cols = a.column_names();
    let b_cols = b.column_names();
    assert_eq!(a_cols, b_cols, "{label}: column names diverge");

    // Materialise both and compare column by column. Materialise only depends
    // on the mask + base, so equal masks + equal bases give equal output.
    let a_df = a.materialize().expect("materialize direct");
    let b_df = b.materialize().expect("materialize dispatch");
    assert_eq!(a_df.nrows(), b_df.nrows(), "{label}: nrows diverge");
    for ((an, ac), (bn, bc)) in a_df.columns.iter().zip(b_df.columns.iter()) {
        assert_eq!(an, bn, "{label}: column ordering diverges");
        match (ac, bc) {
            (Column::Int(x), Column::Int(y)) => assert_eq!(x, y, "{label}: int col {an}"),
            (Column::Float(x), Column::Float(y)) => assert_eq!(
                x.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                y.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "{label}: float col {an} (bit comparison)"
            ),
            _ => panic!("{label}: unsupported column type for parity test ({an})"),
        }
    }
}

// ── 1. filter chain parity (sparse + dense) ─────────────────────────────────

#[test]
fn dispatch_filter_chain_matches_direct_api() {
    // 1M rows so the second filter exercises the v2.2 sparse-gather path
    // (parent retains 100 rows = 0.01%, well below the 25% threshold).
    let xs: Vec<i64> = (0..1_000_000).collect();
    let p1 = pred_lt("x", 100);
    let p2 = pred_lt("x", 50);

    let direct = col_int("x", xs.clone()).tidy().filter(&p1).unwrap().filter(&p2).unwrap();

    let dispatched = dispatch_view_chained(
        col_int("x", xs).tidy(),
        &[
            ("filter", vec![dexpr_value(&p1)]),
            ("filter", vec![dexpr_value(&p2)]),
        ],
    );

    assert_views_equal("filter_chain_sparse", &direct, &dispatched);
    // Also assert the adaptive selection mode matches — proves the dispatch
    // layer did NOT silently materialise the mask between calls.
    assert_eq!(
        direct.explain_selection_mode(),
        dispatched.explain_selection_mode(),
        "selection mode diverges between direct and dispatch paths"
    );
}

// ── 2. select preserves mask through dispatch ───────────────────────────────

#[test]
fn dispatch_filter_then_select_matches_direct_api() {
    let df = df_two_int((0..100i64).collect(), (100..200i64).collect());
    let p = pred_lt("a", 30);

    let direct = df
        .clone()
        .tidy()
        .filter(&p)
        .unwrap()
        .select(&["b"])
        .unwrap();

    let dispatched = dispatch_view_chained(
        df.tidy(),
        &[
            ("filter", vec![dexpr_value(&p)]),
            ("select", vec![str_array(&["b"])]),
        ],
    );

    assert_views_equal("filter_then_select", &direct, &dispatched);
    // After select(b) the dispatched path must still own the same selection
    // bits as the direct path (mask preservation through select).
    assert_eq!(
        direct.selection().count(),
        dispatched.selection().count(),
        "selection cardinality diverges through select"
    );
}

// ── 3. distinct via dispatch matches direct ─────────────────────────────────

#[test]
fn dispatch_distinct_matches_direct_api() {
    let xs: Vec<i64> = (0..50).map(|i| i % 10).collect(); // 10 distinct values, 5x each
    let direct = col_int("k", xs.clone()).tidy().distinct(&["k"]).unwrap();
    let dispatched = dispatch_view_chained(
        col_int("k", xs).tidy(),
        &[("distinct", vec![str_array(&["k"])])],
    );
    assert_views_equal("distinct", &direct, &dispatched);
}

// ── 4. arrange via dispatch matches direct (re-materialisation parity) ──────

#[test]
fn dispatch_arrange_matches_direct_api() {
    let xs: Vec<i64> = vec![5, 1, 4, 1, 3, 2, 5, 0];
    let direct = col_int("v", xs.clone())
        .tidy()
        .arrange(&[ArrangeKey::asc("v")])
        .unwrap();

    let dispatched = dispatch_view_chained(
        col_int("v", xs).tidy(),
        &[("arrange", vec![str_array(&["v"])])],
    );

    assert_views_equal("arrange_asc", &direct, &dispatched);
}

// ── 5. group_by + summarise parity through grouped dispatch ─────────────────

#[test]
fn dispatch_group_by_summarise_matches_direct_api() {
    // Direct path: build TidyView, group_by, summarise.
    let df = df_two_int(
        vec![1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
        vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    );

    let direct_summary = df
        .clone()
        .tidy()
        .group_by(&["a"])
        .unwrap()
        .summarise(&[("sum_b", TidyAgg::Sum("b".into()))])
        .unwrap();

    // Dispatch path: filter (no-op), group_by via dispatch, then summarise via
    // dispatch_grouped_method.
    let view = df.tidy();
    let val = wrap_view(view);
    let inner = match val {
        Value::TidyView(rc) => rc,
        _ => unreachable!(),
    };
    let grouped_val = dispatch_tidy_method(&inner, "group_by", &[str_array(&["a"])])
        .unwrap()
        .unwrap();
    let grouped_inner = match extract_grouped(grouped_val) {
        Value::GroupedTidyView(rc) => rc,
        _ => panic!("group_by did not return a GroupedTidyView"),
    };
    let mut agg_struct_fields: BTreeMap<String, Value> = BTreeMap::new();
    agg_struct_fields.insert("kind".into(), Value::String(Rc::new("sum".into())));
    agg_struct_fields.insert("col".into(), Value::String(Rc::new("b".into())));
    let agg_value = Value::Struct {
        name: "TidyAgg".into(),
        fields: agg_struct_fields,
    };
    let summary_val = dispatch_grouped_method(
        &grouped_inner,
        "summarise",
        &[Value::String(Rc::new("sum_b".into())), agg_value],
    )
    .unwrap()
    .unwrap();
    let dispatched_summary_view = extract_view(summary_val);

    // summarise() returns a TidyFrame internally and the direct path returns
    // a TidyFrame as well — compare via .view().
    assert_views_equal(
        "group_by_summarise",
        &direct_summary.view(),
        &dispatched_summary_view,
    );
}

// ── 6. sparse filter → group_by preserves AdaptiveSelection through dispatch

#[test]
fn dispatch_sparse_filter_then_group_by_uses_adaptive_selection() {
    // 100k rows; filter leaves 50 → very sparse parent → group_by must see
    // the adaptive iter_indices() path, not a materialised vec.
    let xs: Vec<i64> = (0..100_000).collect();
    let ks: Vec<i64> = (0..100_000).map(|i| i % 5).collect();
    let df = DataFrame::from_columns(vec![
        ("x".into(), Column::Int(xs)),
        ("k".into(), Column::Int(ks)),
    ])
    .unwrap();

    let p = pred_lt("x", 50);

    let direct = df
        .clone()
        .tidy()
        .filter(&p)
        .unwrap()
        .group_by(&["k"])
        .unwrap()
        .summarise(&[("n", TidyAgg::Count)])
        .unwrap();

    // Dispatch path: filter via dispatch_tidy_method, group_by via dispatch,
    // summarise via dispatch_grouped_method.
    let inner: Rc<dyn Any> = Rc::new(df.tidy()) as Rc<dyn Any>;
    let after_filter = dispatch_tidy_method(&inner, "filter", &[dexpr_value(&p)])
        .unwrap()
        .unwrap();
    let filter_inner = match after_filter {
        Value::TidyView(rc) => rc,
        _ => unreachable!(),
    };
    // Sanity: the post-filter view picked the SelectionVector mode (sparse).
    // This is the smoking gun for "dispatch did not silently materialise."
    let post_filter_view = filter_inner
        .clone()
        .downcast::<TidyView>()
        .expect("downcast post-filter");
    assert_eq!(
        post_filter_view.explain_selection_mode(),
        "SelectionVector",
        "dispatch filter must preserve sparse adaptive arm"
    );

    let grouped = dispatch_tidy_method(&filter_inner, "group_by", &[str_array(&["k"])])
        .unwrap()
        .unwrap();
    let grouped_inner = match grouped {
        Value::GroupedTidyView(rc) => rc,
        _ => unreachable!(),
    };
    let mut count_fields: BTreeMap<String, Value> = BTreeMap::new();
    count_fields.insert("kind".into(), Value::String(Rc::new("count".into())));
    let count_agg = Value::Struct {
        name: "TidyAgg".into(),
        fields: count_fields,
    };
    let summary_val = dispatch_grouped_method(
        &grouped_inner,
        "summarise",
        &[Value::String(Rc::new("n".into())), count_agg],
    )
    .unwrap()
    .unwrap();
    let dispatched = extract_view(summary_val);

    assert_views_equal(
        "sparse_filter_group_by_summarise",
        &direct.view(),
        &dispatched,
    );
}
