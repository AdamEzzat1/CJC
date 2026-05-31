//! Language-level Locke primitives (`locke_*` builtins).
//!
//! Two surfaces:
//!
//! 1. **v0.2 numeric primitives** — pure-function builtins over `[Float]`
//!    lists: `locke_missing_count`, `locke_missing_rate`, `locke_ks_d`,
//!    `locke_psi`, `locke_sample_score`, `locke_belief_overall`.
//!
//! 2. **v0.3 table-handle registry** — a thread-local DataFrame store
//!    so CJC-Lang source can actually call `locke_validate(h)` on a
//!    DataFrame built from `.cjcl` code. Matches the precedent set by
//!    `cjc-ad`'s GradGraph and `cjc-abng`'s belief radix graph: handle
//!    is an `Int`, state lives in `thread_local!` storage, no
//!    `Value::DataFrame` variant required.
//!
//! Routing follows the existing satellite-dispatch pattern; both
//! `cjc-eval` and `cjc-mir-exec` call `dispatch_locke(name, args)`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use cjc_data::{Column, DataFrame};
use cjc_runtime::value::Value;

use crate::api::{validate, ValidateOptions};
use crate::belief::{sample_score_from_n, BeliefScore};
use crate::drift::{compare as compare_drift, DriftConfig, InductionRiskReport};
use crate::report::{FindingSeverity, LockeReport};
use crate::stats::ks_d_statistic;
use crate::validation::ValidationConfig;

// ─── v0.3 table-handle registry ────────────────────────────────────────────

thread_local! {
    /// In-construction tables, addressed by integer handle. Tables move
    /// into `READY_TABLES` once frozen via `locke_table_freeze`.
    static MUT_TABLES: RefCell<BTreeMap<i64, Vec<(String, Column)>>> =
        RefCell::new(BTreeMap::new());
    /// Frozen tables ready to be validated.
    static READY_TABLES: RefCell<BTreeMap<i64, DataFrame>> = RefCell::new(BTreeMap::new());
    /// Stored validation reports, addressed by handle.
    static REPORTS: RefCell<BTreeMap<i64, LockeReport>> = RefCell::new(BTreeMap::new());
    /// Stored drift reports.
    static DRIFT_REPORTS: RefCell<BTreeMap<i64, InductionRiskReport>> = RefCell::new(BTreeMap::new());
    /// Monotonic counter for handle allocation. Per-thread so two
    /// independent threads don't fight, and per-thread determinism is
    /// preserved (each thread's pipeline is its own deterministic run).
    static NEXT_HANDLE: RefCell<i64> = RefCell::new(1);
}

fn alloc_handle() -> i64 {
    NEXT_HANDLE.with(|n| {
        let mut n = n.borrow_mut();
        let h = *n;
        *n += 1;
        h
    })
}

/// Reset all Locke registry state. Primarily for tests; not a
/// language-level builtin (callable only from Rust).
pub fn reset_registry() {
    MUT_TABLES.with(|t| t.borrow_mut().clear());
    READY_TABLES.with(|t| t.borrow_mut().clear());
    REPORTS.with(|r| r.borrow_mut().clear());
    DRIFT_REPORTS.with(|r| r.borrow_mut().clear());
    NEXT_HANDLE.with(|n| *n.borrow_mut() = 1);
}

/// Route a builtin call by name. See module docs for the full surface.
pub fn dispatch_locke(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    match name {
        // v0.2 numeric primitives
        "locke_missing_count" => Ok(Some(builtin_missing_count(args)?)),
        "locke_missing_rate" => Ok(Some(builtin_missing_rate(args)?)),
        "locke_ks_d" => Ok(Some(builtin_ks_d(args)?)),
        "locke_psi" => Ok(Some(builtin_psi(args)?)),
        "locke_sample_score" => Ok(Some(builtin_sample_score(args)?)),
        "locke_belief_overall" => Ok(Some(builtin_belief_overall(args)?)),

        // v0.3 table-handle registry
        "locke_table_new" => Ok(Some(builtin_table_new(args)?)),
        "locke_table_add_float_col" => Ok(Some(builtin_table_add_float_col(args)?)),
        "locke_table_add_int_col" => Ok(Some(builtin_table_add_int_col(args)?)),
        "locke_table_add_str_col" => Ok(Some(builtin_table_add_str_col(args)?)),
        "locke_table_add_bool_col" => Ok(Some(builtin_table_add_bool_col(args)?)),
        "locke_table_freeze" => Ok(Some(builtin_table_freeze(args)?)),
        "locke_table_nrows" => Ok(Some(builtin_table_nrows(args)?)),
        "locke_table_ncols" => Ok(Some(builtin_table_ncols(args)?)),
        "locke_validate" => Ok(Some(builtin_validate(args)?)),
        "locke_drift" => Ok(Some(builtin_drift(args)?)),
        "locke_report_worst_severity" => Ok(Some(builtin_report_worst_severity(args)?)),
        "locke_report_n_findings" => Ok(Some(builtin_report_n_findings(args)?)),
        "locke_report_count_by_severity" => Ok(Some(builtin_report_count_by_severity(args)?)),
        "locke_report_overall_score" => Ok(Some(builtin_report_overall_score(args)?)),
        "locke_drift_n_findings" => Ok(Some(builtin_drift_n_findings(args)?)),
        "locke_drift_worst_severity" => Ok(Some(builtin_drift_worst_severity(args)?)),

        _ => Ok(None),
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn as_float_list(v: &Value, arg_name: &str) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, x) in arr.iter().enumerate() {
                match x {
                    Value::Float(f) => out.push(*f),
                    Value::Int(n) => out.push(*n as f64),
                    other => {
                        return Err(format!(
                            "{}: element {} is not a number: {:?}",
                            arg_name, i, other
                        ));
                    }
                }
            }
            Ok(out)
        }
        other => Err(format!(
            "{}: expected an array of numbers, got {:?}",
            arg_name, other
        )),
    }
}

fn as_int(v: &Value, arg_name: &str) -> Result<i64, String> {
    match v {
        Value::Int(n) => Ok(*n),
        Value::Float(f) => Ok(*f as i64),
        other => Err(format!(
            "{}: expected Int, got {:?}",
            arg_name, other
        )),
    }
}

fn as_float(v: &Value, arg_name: &str) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        other => Err(format!(
            "{}: expected Float, got {:?}",
            arg_name, other
        )),
    }
}

fn require_arity(name: &str, args: &[Value], expected: usize) -> Result<(), String> {
    if args.len() != expected {
        Err(format!(
            "{}: expected {} arguments, got {}",
            name,
            expected,
            args.len()
        ))
    } else {
        Ok(())
    }
}

// ─── Builtins ───────────────────────────────────────────────────────────────

fn builtin_missing_count(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_missing_count", args, 1)?;
    let xs = as_float_list(&args[0], "values")?;
    let n = xs.iter().filter(|x| x.is_nan()).count() as i64;
    Ok(Value::Int(n))
}

fn builtin_missing_rate(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_missing_rate", args, 1)?;
    let xs = as_float_list(&args[0], "values")?;
    let n = xs.len();
    if n == 0 {
        return Ok(Value::Float(0.0));
    }
    let miss = xs.iter().filter(|x| x.is_nan()).count();
    Ok(Value::Float(miss as f64 / n as f64))
}

fn builtin_ks_d(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_ks_d", args, 2)?;
    let xs = as_float_list(&args[0], "train")?;
    let ys = as_float_list(&args[1], "test")?;
    match ks_d_statistic(&xs, &ys) {
        Some(d) => Ok(Value::Float(d)),
        None => Err("locke_ks_d: both samples must have ≥ 2 non-NaN values".into()),
    }
}

fn builtin_psi(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_psi", args, 2)?;
    let p = as_float_list(&args[0], "p")?;
    let q = as_float_list(&args[1], "q")?;
    if p.len() != q.len() {
        return Err(format!(
            "locke_psi: p and q must have equal length, got {} and {}",
            p.len(),
            q.len()
        ));
    }
    if p.is_empty() {
        return Err("locke_psi: distributions must be non-empty".into());
    }
    // Reuse the same calculation that drift.rs does internally.
    let eps = 1e-6;
    let mut acc = cjc_repro::KahanAccumulatorF64::new();
    for i in 0..p.len() {
        let pi = p[i].max(eps);
        let qi = q[i].max(eps);
        acc.add((qi - pi) * (qi / pi).ln());
    }
    Ok(Value::Float(acc.finalize()))
}

fn builtin_sample_score(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_sample_score", args, 1)?;
    let n = as_int(&args[0], "n")?;
    if n < 0 {
        return Err("locke_sample_score: n must be non-negative".into());
    }
    Ok(Value::Float(sample_score_from_n(n as u64)))
}

fn builtin_belief_overall(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_belief_overall", args, 8)?;
    let mut subs = [0.0f64; 8];
    for (i, a) in args.iter().enumerate() {
        subs[i] = as_float(a, &format!("dim_{}", i + 1))?;
    }
    let score = BeliefScore::from_dimensions(
        subs[0], subs[1], subs[2], subs[3], subs[4], subs[5], subs[6], subs[7],
    );
    Ok(Value::Float(score.overall))
}

// ─── v0.3 table-handle builtins ────────────────────────────────────────────

fn as_str_list(v: &Value, arg_name: &str) -> Result<Vec<String>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, x) in arr.iter().enumerate() {
                match x {
                    Value::String(s) => out.push(s.as_ref().clone()),
                    other => {
                        return Err(format!(
                            "{}: element {} is not a string: {:?}",
                            arg_name, i, other
                        ));
                    }
                }
            }
            Ok(out)
        }
        other => Err(format!("{}: expected an array of strings, got {:?}", arg_name, other)),
    }
}

fn as_bool_list(v: &Value, arg_name: &str) -> Result<Vec<bool>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, x) in arr.iter().enumerate() {
                match x {
                    Value::Bool(b) => out.push(*b),
                    other => {
                        return Err(format!(
                            "{}: element {} is not a bool: {:?}",
                            arg_name, i, other
                        ));
                    }
                }
            }
            Ok(out)
        }
        other => Err(format!("{}: expected an array of bools, got {:?}", arg_name, other)),
    }
}

fn as_int_list(v: &Value, arg_name: &str) -> Result<Vec<i64>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, x) in arr.iter().enumerate() {
                match x {
                    Value::Int(n) => out.push(*n),
                    other => {
                        return Err(format!(
                            "{}: element {} is not an int: {:?}",
                            arg_name, i, other
                        ));
                    }
                }
            }
            Ok(out)
        }
        other => Err(format!("{}: expected an array of ints, got {:?}", arg_name, other)),
    }
}

fn as_str(v: &Value, arg_name: &str) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.as_ref().clone()),
        other => Err(format!("{}: expected String, got {:?}", arg_name, other)),
    }
}

fn builtin_table_new(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_table_new", args, 0)?;
    let h = alloc_handle();
    MUT_TABLES.with(|t| t.borrow_mut().insert(h, Vec::new()));
    Ok(Value::Int(h))
}

fn add_col(args: &[Value], col_name: &str, builder: impl FnOnce(&Value) -> Result<Column, String>) -> Result<Value, String> {
    require_arity(col_name, args, 3)?;
    let h = as_int(&args[0], "handle")?;
    let name = as_str(&args[1], "name")?;
    let col = builder(&args[2])?;
    MUT_TABLES.with(|t| -> Result<(), String> {
        let mut t = t.borrow_mut();
        let cols = t
            .get_mut(&h)
            .ok_or_else(|| format!("{}: no in-construction table for handle {}", col_name, h))?;
        if let Some(prev) = cols.first() {
            if prev.1.len() != col.len() {
                return Err(format!(
                    "{}: column `{}` has length {} but table already has columns of length {}",
                    col_name,
                    name,
                    col.len(),
                    prev.1.len()
                ));
            }
        }
        cols.push((name, col));
        Ok(())
    })?;
    Ok(Value::Int(0))
}

fn builtin_table_add_float_col(args: &[Value]) -> Result<Value, String> {
    add_col(args, "locke_table_add_float_col", |v| {
        let xs = as_float_list(v, "values")?;
        Ok(Column::Float(xs))
    })
}

fn builtin_table_add_int_col(args: &[Value]) -> Result<Value, String> {
    add_col(args, "locke_table_add_int_col", |v| {
        let xs = as_int_list(v, "values")?;
        Ok(Column::Int(xs))
    })
}

fn builtin_table_add_str_col(args: &[Value]) -> Result<Value, String> {
    add_col(args, "locke_table_add_str_col", |v| {
        let xs = as_str_list(v, "values")?;
        Ok(Column::Str(xs))
    })
}

fn builtin_table_add_bool_col(args: &[Value]) -> Result<Value, String> {
    add_col(args, "locke_table_add_bool_col", |v| {
        let xs = as_bool_list(v, "values")?;
        Ok(Column::Bool(xs))
    })
}

fn builtin_table_freeze(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_table_freeze", args, 1)?;
    let h = as_int(&args[0], "handle")?;
    let cols = MUT_TABLES.with(|t| t.borrow_mut().remove(&h)).ok_or_else(|| {
        format!("locke_table_freeze: no in-construction table for handle {}", h)
    })?;
    let df = DataFrame::from_columns(cols).map_err(|e| format!("locke_table_freeze: {:?}", e))?;
    READY_TABLES.with(|t| t.borrow_mut().insert(h, df));
    Ok(Value::Int(h))
}

fn lookup_ready_or_freeze(h: i64) -> Result<DataFrame, String> {
    // If the table is still under construction, freeze it transparently.
    let already_frozen = READY_TABLES.with(|t| t.borrow().contains_key(&h));
    if !already_frozen {
        if MUT_TABLES.with(|t| t.borrow().contains_key(&h)) {
            let _ = builtin_table_freeze(&[Value::Int(h)])?;
        }
    }
    READY_TABLES
        .with(|t| t.borrow().get(&h).cloned())
        .ok_or_else(|| format!("no table for handle {}", h))
}

fn builtin_table_nrows(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_table_nrows", args, 1)?;
    let h = as_int(&args[0], "handle")?;
    let df = lookup_ready_or_freeze(h)?;
    Ok(Value::Int(df.nrows() as i64))
}

fn builtin_table_ncols(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_table_ncols", args, 1)?;
    let h = as_int(&args[0], "handle")?;
    let df = lookup_ready_or_freeze(h)?;
    Ok(Value::Int(df.ncols() as i64))
}

fn builtin_validate(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_validate", args, 1)?;
    let h = as_int(&args[0], "handle")?;
    let df = lookup_ready_or_freeze(h)?;
    let opts = ValidateOptions {
        dataset_label: format!("table#{}", h),
        config: ValidationConfig::default(),
        ..Default::default()
    };
    let report = validate(&df, &opts);
    let report_h = alloc_handle();
    REPORTS.with(|r| r.borrow_mut().insert(report_h, report));
    Ok(Value::Int(report_h))
}

fn builtin_drift(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_drift", args, 2)?;
    let train_h = as_int(&args[0], "train_handle")?;
    let test_h = as_int(&args[1], "test_handle")?;
    let train = lookup_ready_or_freeze(train_h)?;
    let test = lookup_ready_or_freeze(test_h)?;
    let report = compare_drift(&train, &test, &DriftConfig::default());
    let report_h = alloc_handle();
    DRIFT_REPORTS.with(|r| r.borrow_mut().insert(report_h, report));
    Ok(Value::Int(report_h))
}

fn severity_ordinal(s: FindingSeverity) -> i64 {
    match s {
        FindingSeverity::Info => 0,
        FindingSeverity::Notice => 1,
        FindingSeverity::Warning => 2,
        FindingSeverity::Error => 3,
    }
}

fn builtin_report_worst_severity(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_report_worst_severity", args, 1)?;
    let h = as_int(&args[0], "report_handle")?;
    let sev = REPORTS
        .with(|r| r.borrow().get(&h).map(|rep| rep.worst_severity()))
        .ok_or_else(|| format!("no validation report for handle {}", h))?;
    Ok(Value::Int(severity_ordinal(sev)))
}

fn builtin_report_n_findings(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_report_n_findings", args, 1)?;
    let h = as_int(&args[0], "report_handle")?;
    let n = REPORTS
        .with(|r| r.borrow().get(&h).map(|rep| rep.findings.len()))
        .ok_or_else(|| format!("no validation report for handle {}", h))?;
    Ok(Value::Int(n as i64))
}

fn builtin_report_count_by_severity(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_report_count_by_severity", args, 2)?;
    let h = as_int(&args[0], "report_handle")?;
    let sev_ord = as_int(&args[1], "severity_ordinal")?;
    let count = REPORTS
        .with(|r| {
            r.borrow().get(&h).map(|rep| {
                rep.findings
                    .iter()
                    .filter(|f| severity_ordinal(f.severity) == sev_ord)
                    .count()
            })
        })
        .ok_or_else(|| format!("no validation report for handle {}", h))?;
    Ok(Value::Int(count as i64))
}

fn builtin_report_overall_score(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_report_overall_score", args, 1)?;
    let h = as_int(&args[0], "report_handle")?;
    let report = REPORTS
        .with(|r| r.borrow().get(&h).cloned())
        .ok_or_else(|| format!("no validation report for handle {}", h))?;
    let belief = crate::api::belief_report_from_locke(&report);
    Ok(Value::Float(belief.score.overall))
}

fn builtin_drift_n_findings(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_drift_n_findings", args, 1)?;
    let h = as_int(&args[0], "drift_handle")?;
    let n = DRIFT_REPORTS
        .with(|r| r.borrow().get(&h).map(|rep| rep.findings.len()))
        .ok_or_else(|| format!("no drift report for handle {}", h))?;
    Ok(Value::Int(n as i64))
}

fn builtin_drift_worst_severity(args: &[Value]) -> Result<Value, String> {
    require_arity("locke_drift_worst_severity", args, 1)?;
    let h = as_int(&args[0], "drift_handle")?;
    let sev = DRIFT_REPORTS
        .with(|r| r.borrow().get(&h).map(|rep| rep.worst_severity()))
        .ok_or_else(|| format!("no drift report for handle {}", h))?;
    Ok(Value::Int(severity_ordinal(sev)))
}

// Silence unused-import warning when none of the helpers below are used.
#[allow(dead_code)]
fn _link() -> Rc<()> {
    Rc::new(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_runtime::value::Value;
    use std::rc::Rc;

    fn arr(xs: &[f64]) -> Value {
        Value::Array(Rc::new(xs.iter().map(|x| Value::Float(*x)).collect()))
    }

    #[test]
    fn dispatch_returns_none_for_unknown() {
        let r = dispatch_locke("not_a_locke_builtin", &[]).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn missing_count_counts_nans() {
        let v = arr(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let r = dispatch_locke("locke_missing_count", &[v]).unwrap().unwrap();
        match r {
            Value::Int(n) => assert_eq!(n, 2),
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    fn ks_d_for_identical_lists_is_zero() {
        let v = arr(&(0..100).map(|i| i as f64).collect::<Vec<_>>());
        let r = dispatch_locke("locke_ks_d", &[v.clone(), v]).unwrap().unwrap();
        match r {
            Value::Float(d) => assert!(d.abs() < 1e-12),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn ks_d_disjoint_is_one() {
        let a = arr(&(0..50).map(|i| i as f64).collect::<Vec<_>>());
        let b = arr(&(1000..1050).map(|i| i as f64).collect::<Vec<_>>());
        let r = dispatch_locke("locke_ks_d", &[a, b]).unwrap().unwrap();
        match r {
            Value::Float(d) => assert!((d - 1.0).abs() < 1e-12),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn ks_d_short_samples_error() {
        let a = arr(&[1.0]);
        let b = arr(&[1.0, 2.0, 3.0]);
        let r = dispatch_locke("locke_ks_d", &[a, b]);
        assert!(r.is_err());
    }

    #[test]
    fn psi_identical_distributions_is_zero() {
        let p = arr(&[0.25, 0.25, 0.25, 0.25]);
        let q = arr(&[0.25, 0.25, 0.25, 0.25]);
        let r = dispatch_locke("locke_psi", &[p, q]).unwrap().unwrap();
        match r {
            Value::Float(d) => assert!(d.abs() < 1e-9),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn psi_mismatched_length_is_error() {
        let p = arr(&[0.5, 0.5]);
        let q = arr(&[0.25, 0.25, 0.5]);
        assert!(dispatch_locke("locke_psi", &[p, q]).is_err());
    }

    #[test]
    fn sample_score_at_30_is_about_half() {
        let r = dispatch_locke("locke_sample_score", &[Value::Int(30)])
            .unwrap()
            .unwrap();
        match r {
            Value::Float(d) => assert!((d - 0.5).abs() < 0.005),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn belief_overall_average_of_eights() {
        let args: Vec<Value> = vec![
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(1.0),
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(0.0),
        ];
        let r = dispatch_locke("locke_belief_overall", &args).unwrap().unwrap();
        match r {
            Value::Float(d) => assert!((d - 0.5).abs() < 1e-12),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn missing_rate_nan_only() {
        let v = arr(&[f64::NAN, f64::NAN, f64::NAN]);
        let r = dispatch_locke("locke_missing_rate", &[v]).unwrap().unwrap();
        match r {
            Value::Float(d) => assert!((d - 1.0).abs() < 1e-12),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn missing_rate_empty_is_zero() {
        let v = Value::Array(Rc::new(vec![]));
        let r = dispatch_locke("locke_missing_rate", &[v]).unwrap().unwrap();
        match r {
            Value::Float(d) => assert!((d - 0.0).abs() < 1e-12),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    // ─── v0.3 table-handle registry tests ─────────────────────────────

    fn int_arr(xs: &[i64]) -> Value {
        Value::Array(Rc::new(xs.iter().map(|x| Value::Int(*x)).collect()))
    }
    fn str_arr(xs: &[&str]) -> Value {
        Value::Array(Rc::new(
            xs.iter()
                .map(|s| Value::String(Rc::new((*s).into())))
                .collect(),
        ))
    }
    fn unwrap_int(v: Value) -> i64 {
        match v {
            Value::Int(n) => n,
            other => panic!("expected Int, got {:?}", other),
        }
    }
    fn unwrap_float(v: Value) -> f64 {
        match v {
            Value::Float(f) => f,
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn table_new_then_validate_returns_report_handle() {
        super::reset_registry();
        let h = unwrap_int(
            dispatch_locke("locke_table_new", &[]).unwrap().unwrap(),
        );
        dispatch_locke(
            "locke_table_add_int_col",
            &[Value::Int(h), Value::String(Rc::new("age".into())), int_arr(&[20, 30, 40])],
        )
        .unwrap();
        let nrows = unwrap_int(
            dispatch_locke("locke_table_nrows", &[Value::Int(h)])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(nrows, 3);
        let report_h = unwrap_int(
            dispatch_locke("locke_validate", &[Value::Int(h)])
                .unwrap()
                .unwrap(),
        );
        let n = unwrap_int(
            dispatch_locke("locke_report_n_findings", &[Value::Int(report_h)])
                .unwrap()
                .unwrap(),
        );
        assert!(n >= 1, "should at least emit E9002 limitation note for Int column");
    }

    #[test]
    fn validate_with_nan_float_emits_e9001_via_handle_api() {
        super::reset_registry();
        let h = unwrap_int(dispatch_locke("locke_table_new", &[]).unwrap().unwrap());
        dispatch_locke(
            "locke_table_add_float_col",
            &[
                Value::Int(h),
                Value::String(Rc::new("x".into())),
                arr(&[1.0, f64::NAN, 3.0, f64::NAN]),
            ],
        )
        .unwrap();
        let r = unwrap_int(
            dispatch_locke("locke_validate", &[Value::Int(h)])
                .unwrap()
                .unwrap(),
        );
        let worst = unwrap_int(
            dispatch_locke("locke_report_worst_severity", &[Value::Int(r)])
                .unwrap()
                .unwrap(),
        );
        // 2/4 = 50% NaN → Error severity.
        assert_eq!(worst, 3);
    }

    #[test]
    fn drift_through_handle_api_emits_findings() {
        super::reset_registry();
        let train_h = unwrap_int(dispatch_locke("locke_table_new", &[]).unwrap().unwrap());
        dispatch_locke(
            "locke_table_add_float_col",
            &[
                Value::Int(train_h),
                Value::String(Rc::new("x".into())),
                arr(&(0..50).map(|i| i as f64).collect::<Vec<_>>()),
            ],
        )
        .unwrap();
        let test_h = unwrap_int(dispatch_locke("locke_table_new", &[]).unwrap().unwrap());
        dispatch_locke(
            "locke_table_add_float_col",
            &[
                Value::Int(test_h),
                Value::String(Rc::new("x".into())),
                arr(&(1000..1050).map(|i| i as f64).collect::<Vec<_>>()),
            ],
        )
        .unwrap();
        let dh = unwrap_int(
            dispatch_locke("locke_drift", &[Value::Int(train_h), Value::Int(test_h)])
                .unwrap()
                .unwrap(),
        );
        let n = unwrap_int(
            dispatch_locke("locke_drift_n_findings", &[Value::Int(dh)])
                .unwrap()
                .unwrap(),
        );
        assert!(n >= 1, "drift should emit at least one finding (E9039)");
        let worst = unwrap_int(
            dispatch_locke("locke_drift_worst_severity", &[Value::Int(dh)])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(worst, 3, "disjoint-support KS D = 1.0 → Error");
    }

    #[test]
    fn overall_score_is_in_unit_interval_via_handle() {
        super::reset_registry();
        let h = unwrap_int(dispatch_locke("locke_table_new", &[]).unwrap().unwrap());
        dispatch_locke(
            "locke_table_add_float_col",
            &[
                Value::Int(h),
                Value::String(Rc::new("x".into())),
                arr(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            ],
        )
        .unwrap();
        let r = unwrap_int(
            dispatch_locke("locke_validate", &[Value::Int(h)])
                .unwrap()
                .unwrap(),
        );
        let overall = unwrap_float(
            dispatch_locke("locke_report_overall_score", &[Value::Int(r)])
                .unwrap()
                .unwrap(),
        );
        assert!(overall >= 0.0 && overall <= 1.0);
    }

    #[test]
    fn table_with_mismatched_column_lengths_is_rejected() {
        super::reset_registry();
        let h = unwrap_int(dispatch_locke("locke_table_new", &[]).unwrap().unwrap());
        dispatch_locke(
            "locke_table_add_int_col",
            &[Value::Int(h), Value::String(Rc::new("a".into())), int_arr(&[1, 2, 3])],
        )
        .unwrap();
        let res = dispatch_locke(
            "locke_table_add_str_col",
            &[
                Value::Int(h),
                Value::String(Rc::new("b".into())),
                str_arr(&["x", "y"]),
            ],
        );
        assert!(res.is_err(), "mismatched lengths must error");
    }

    #[test]
    fn validate_handle_for_nonexistent_table_errors() {
        super::reset_registry();
        let res = dispatch_locke("locke_validate", &[Value::Int(9999)]);
        assert!(res.is_err());
    }
}
