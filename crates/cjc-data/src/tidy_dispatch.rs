//! Shared tidy dispatch: maps CJC language method calls on TidyView /
//! GroupedTidyView values to the concrete cjc_data API.
//!
//! Both `cjc-eval` and `cjc-mir-exec` call into `dispatch_tidy_method` and
//! `dispatch_grouped_method` so that every tidy operation has a single source
//! of truth.  The executors only need to pattern-match `Value::TidyView` or
//! `Value::GroupedTidyView` and delegate here.
//!
//! # Error handling
//! All errors are returned as `Err(String)`.  The caller wraps the string
//! into its own error type (EvalError / MirExecError).

use std::rc::Rc;
use std::any::Any;

use cjc_runtime::value::Value;

use crate::{
    ArrangeKey, Column, DExpr, DBinOp, DataFrame, GroupedTidyView,
    TidyAgg, TidyView,
};

// ============================================================================
//  Public entry points
// ============================================================================

/// Dispatch a method call on a `Value::TidyView`.
///
/// Returns `Ok(Some(value))` if the method is known, `Ok(None)` if not
/// recognised (allows the caller to fall through to other dispatch paths).
pub fn dispatch_tidy_method(
    inner: &Rc<dyn Any>,
    method: &str,
    args: &[Value],
) -> Result<Option<Value>, String> {
    let view = downcast_view(inner)?;
    match method {
        // -- shape ----------------------------------------------------------
        "nrows" => Ok(Some(Value::Int(view.nrows() as i64))),
        "ncols" => Ok(Some(Value::Int(view.ncols() as i64))),
        "column_names" => {
            let names: Vec<Value> = view
                .column_names()
                .into_iter()
                .map(|s| Value::String(Rc::new(s.to_string())))
                .collect();
            Ok(Some(Value::Array(Rc::new(names))))
        }

        // -- filter ---------------------------------------------------------
        "filter" => {
            if args.len() != 1 {
                return Err("TidyView.filter requires 1 argument: predicate DExpr".into());
            }
            let predicate = value_to_dexpr(&args[0])?;
            let new_view = view.filter(&predicate).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }

        // -- select ---------------------------------------------------------
        "select" => {
            if args.len() != 1 {
                return Err("TidyView.select requires 1 argument: column names array".into());
            }
            let cols = value_to_str_vec(&args[0])?;
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            let new_view = view.select(&col_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }

        // -- mutate ---------------------------------------------------------
        "mutate" => {
            // mutate(name, expr) or mutate([(name, expr), ...])
            // We support: mutate("col_name", dexpr_value)
            if args.len() != 2 {
                return Err("TidyView.mutate requires 2 arguments: column_name and expression".into());
            }
            let col_name = value_to_string(&args[0])?;
            let expr = value_to_dexpr(&args[1])?;
            let frame = view.mutate(&[(&col_name, expr)]).map_err(|e| format!("{e}"))?;
            // mutate returns TidyFrame; convert to TidyView for pipeline continuity
            Ok(Some(wrap_view(frame.view())))
        }

        // -- group_by -------------------------------------------------------
        "group_by" => {
            if args.len() != 1 {
                return Err("TidyView.group_by requires 1 argument: key columns array".into());
            }
            let keys = value_to_str_vec(&args[0])?;
            let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
            let grouped = view.group_by(&key_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_grouped(grouped)))
        }

        // -- arrange --------------------------------------------------------
        "arrange" => {
            if args.len() != 1 {
                return Err("TidyView.arrange requires 1 argument: sort keys array".into());
            }
            let keys = value_to_arrange_keys(&args[0])?;
            let new_view = view.arrange(&keys).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }

        // -- distinct -------------------------------------------------------
        "distinct" => {
            let cols = if args.is_empty() {
                view.column_names().iter().map(|s| s.to_string()).collect::<Vec<_>>()
            } else {
                value_to_str_vec(&args[0])?
            };
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            let new_view = view.distinct(&col_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }

        // -- slice family ---------------------------------------------------
        "slice" => {
            if args.len() != 2 {
                return Err("TidyView.slice requires 2 arguments: start, end".into());
            }
            let start = value_to_usize(&args[0])?;
            let end = value_to_usize(&args[1])?;
            Ok(Some(wrap_view(view.slice(start, end))))
        }
        "slice_head" => {
            if args.len() != 1 {
                return Err("TidyView.slice_head requires 1 argument: n".into());
            }
            let n = value_to_usize(&args[0])?;
            Ok(Some(wrap_view(view.slice_head(n))))
        }
        "slice_tail" => {
            if args.len() != 1 {
                return Err("TidyView.slice_tail requires 1 argument: n".into());
            }
            let n = value_to_usize(&args[0])?;
            Ok(Some(wrap_view(view.slice_tail(n))))
        }
        "slice_sample" => {
            if args.len() != 2 {
                return Err("TidyView.slice_sample requires 2 arguments: n, seed".into());
            }
            let n = value_to_usize(&args[0])?;
            let seed = match &args[1] {
                Value::Int(i) => *i as u64,
                _ => return Err("slice_sample seed must be Int".into()),
            };
            Ok(Some(wrap_view(view.slice_sample(n, seed))))
        }

        // -- joins ----------------------------------------------------------
        "inner_join" | "left_join" | "semi_join" | "anti_join" => {
            dispatch_join(view, args, method)
        }

        // -- reshape --------------------------------------------------------
        "pivot_longer" => {
            if args.len() < 2 || args.len() > 3 {
                return Err(
                    "TidyView.pivot_longer requires 2-3 args: cols, names_to, [values_to]".into(),
                );
            }
            let cols = value_to_str_vec(&args[0])?;
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            let names_to = value_to_string(&args[1])?;
            let values_to = if args.len() == 3 {
                value_to_string(&args[2])?
            } else {
                "value".to_string()
            };
            let frame = view
                .pivot_longer(&col_refs, &names_to, &values_to)
                .map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(frame.view())))
        }
        "pivot_wider" => {
            if args.len() != 3 {
                return Err(
                    "TidyView.pivot_wider requires 3 args: id_cols, names_from, values_from"
                        .into(),
                );
            }
            let id_cols = value_to_str_vec(&args[0])?;
            let id_refs: Vec<&str> = id_cols.iter().map(|s| s.as_str()).collect();
            let names_from = value_to_string(&args[1])?;
            let values_from = value_to_string(&args[2])?;
            let nullable_frame = view
                .pivot_wider(&id_refs, &names_from, &values_from)
                .map_err(|e| format!("{e}"))?;
            // NullableFrame → fill nulls with defaults → TidyView
            Ok(Some(wrap_view(nullable_frame.to_tidy_view_filled())))
        }

        // -- rename / relocate / drop_cols / bind ----------------------------
        "rename" => {
            if args.len() != 1 {
                return Err("TidyView.rename requires 1 argument: array of [old, new] pairs".into());
            }
            let pairs = value_to_rename_pairs(&args[0])?;
            let pair_refs: Vec<(&str, &str)> =
                pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let new_view = view.rename(&pair_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }
        "drop_cols" => {
            if args.len() != 1 {
                return Err("TidyView.drop_cols requires 1 argument: column names array".into());
            }
            let cols = value_to_str_vec(&args[0])?;
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            let new_view = view.drop_cols(&col_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }
        "bind_rows" => {
            if args.len() != 1 {
                return Err("TidyView.bind_rows requires 1 argument: other TidyView".into());
            }
            let other_rc = match &args[0] {
                Value::TidyView(rc) => rc,
                _ => return Err("bind_rows argument must be a TidyView".into()),
            };
            let other = downcast_view(other_rc)?;
            let frame = view.bind_rows(other).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(frame.view())))
        }
        "bind_cols" => {
            if args.len() != 1 {
                return Err("TidyView.bind_cols requires 1 argument: other TidyView".into());
            }
            let other_rc = match &args[0] {
                Value::TidyView(rc) => rc,
                _ => return Err("bind_cols argument must be a TidyView".into()),
            };
            let other = downcast_view(other_rc)?;
            let frame = view.bind_cols(other).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(frame.view())))
        }

        // -- column extraction / tensor -------------------------------------
        "column" => {
            if args.len() != 1 {
                return Err("TidyView.column requires 1 argument: column_name".into());
            }
            let name = value_to_string(&args[0])?;
            let df = view.materialize().map_err(|e| format!("{e}"))?;
            let col = df
                .get_column(&name)
                .ok_or_else(|| format!("column '{}' not found", name))?;
            Ok(Some(column_to_value(col)))
        }
        "to_tensor" => {
            if args.len() != 1 {
                return Err("TidyView.to_tensor requires 1 argument: column_names array".into());
            }
            let cols = value_to_str_vec(&args[0])?;
            let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            let t = view.to_tensor(&col_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(Value::Tensor(t)))
        }

        // -- materialize to DataFrame Struct --------------------------------
        "collect" => {
            let df = view.materialize().map_err(|e| format!("{e}"))?;
            Ok(Some(dataframe_to_value(df)))
        }

        // -- print (for debugging) ------------------------------------------
        "print" => {
            let df = view.materialize().map_err(|e| format!("{e}"))?;
            let s = format_dataframe(&df);
            // Returning the formatted string; the caller is responsible for
            // printing and capturing in output buffer.
            Ok(Some(Value::String(Rc::new(s))))
        }

        _ => Ok(None), // unknown method — caller falls through
    }
}

/// Dispatch a method call on a `Value::GroupedTidyView`.
pub fn dispatch_grouped_method(
    inner: &Rc<dyn Any>,
    method: &str,
    args: &[Value],
) -> Result<Option<Value>, String> {
    let grouped = downcast_grouped(inner)?;
    match method {
        "ngroups" => Ok(Some(Value::Int(grouped.ngroups() as i64))),

        "summarise" | "summarize" => {
            if args.len() % 2 != 0 || args.is_empty() {
                return Err(
                    "summarise requires pairs of (name, agg) arguments".into(),
                );
            }
            let mut assignments: Vec<(String, TidyAgg)> = Vec::new();
            let mut i = 0;
            while i < args.len() {
                let name = value_to_string(&args[i])?;
                let agg = value_to_tidy_agg(&args[i + 1])?;
                assignments.push((name, agg));
                i += 2;
            }
            let asg_refs: Vec<(&str, TidyAgg)> = assignments
                .iter()
                .map(|(n, a)| (n.as_str(), a.clone()))
                .collect();
            let frame = grouped.summarise(&asg_refs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(frame.view())))
        }

        "ungroup" => {
            let view = grouped.clone().ungroup();
            Ok(Some(wrap_view(view)))
        }

        _ => Ok(None),
    }
}

// ============================================================================
//  Helpers — Value ↔ cjc_data conversions
// ============================================================================

fn downcast_view(inner: &Rc<dyn Any>) -> Result<&TidyView, String> {
    inner
        .downcast_ref::<TidyView>()
        .ok_or_else(|| "internal error: TidyView downcast failed".to_string())
}

fn downcast_grouped(inner: &Rc<dyn Any>) -> Result<&GroupedTidyView, String> {
    inner
        .downcast_ref::<GroupedTidyView>()
        .ok_or_else(|| "internal error: GroupedTidyView downcast failed".to_string())
}

/// Wrap a `TidyView` into `Value::TidyView`.
pub fn wrap_view(view: TidyView) -> Value {
    Value::TidyView(Rc::new(view) as Rc<dyn Any>)
}

/// Wrap a `GroupedTidyView` into `Value::GroupedTidyView`.
pub fn wrap_grouped(grouped: GroupedTidyView) -> Value {
    Value::GroupedTidyView(Rc::new(grouped) as Rc<dyn Any>)
}

/// Convert `Value::String` → `String`.
fn value_to_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.as_ref().clone()),
        _ => Err(format!("expected String, got {}", v.type_name())),
    }
}

/// Convert `Value::Int` → `usize`.
fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        Value::Int(i) => Err(format!("expected non-negative Int, got {i}")),
        _ => Err(format!("expected Int, got {}", v.type_name())),
    }
}

/// Convert `Value::Array([String, ...])` → `Vec<String>`.
fn value_to_str_vec(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.as_ref().clone()),
                _ => Err(format!("expected String in array, got {}", v.type_name())),
            })
            .collect(),
        _ => Err(format!("expected Array, got {}", v.type_name())),
    }
}

/// Parse a `Value::Struct { name: "DExpr", ... }` into a `DExpr`.
///
/// The CJC language constructs DExpr values via helper builtins:
///   col("name")        → Struct { name: "DExpr", kind: "col", value: "name" }
///   binop(">", l, r)   → Struct { name: "DExpr", kind: "binop", op: ">", left: l, right: r }
///   lit_int(42)         → Struct { name: "DExpr", kind: "lit_int", value: 42 }
///   etc.
///
/// For ergonomic use, we also accept raw literals directly:
///   Value::Int(42)      → DExpr::LitInt(42)
///   Value::Float(3.14)  → DExpr::LitFloat(3.14)
///   Value::Bool(true)   → DExpr::LitBool(true)
///   Value::String("x")  → DExpr::Col("x")   -- shorthand for col("x")
pub fn value_to_dexpr(v: &Value) -> Result<DExpr, String> {
    match v {
        // Literal shorthand
        Value::Int(i) => Ok(DExpr::LitInt(*i)),
        Value::Float(f) => Ok(DExpr::LitFloat(*f)),
        Value::Bool(b) => Ok(DExpr::LitBool(*b)),
        Value::String(s) => Ok(DExpr::Col(s.as_ref().clone())),
        // Struct-encoded DExpr
        Value::Struct { name, fields } if name == "DExpr" => {
            let kind = fields
                .get("kind")
                .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().as_str()) } else { None })
                .ok_or("DExpr struct missing 'kind' string field")?;
            match kind {
                "col" => {
                    let col_name = fields
                        .get("value")
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().clone()) } else { None })
                        .ok_or("DExpr col missing 'value' string field")?;
                    Ok(DExpr::Col(col_name))
                }
                "lit_int" => {
                    let val = fields
                        .get("value")
                        .and_then(|v| if let Value::Int(i) = v { Some(*i) } else { None })
                        .ok_or("DExpr lit_int missing 'value' int field")?;
                    Ok(DExpr::LitInt(val))
                }
                "lit_float" => {
                    let val = fields
                        .get("value")
                        .and_then(|v| if let Value::Float(f) = v { Some(*f) } else { None })
                        .ok_or("DExpr lit_float missing 'value' float field")?;
                    Ok(DExpr::LitFloat(val))
                }
                "lit_bool" => {
                    let val = fields
                        .get("value")
                        .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                        .ok_or("DExpr lit_bool missing 'value' bool field")?;
                    Ok(DExpr::LitBool(val))
                }
                "lit_str" => {
                    let val = fields
                        .get("value")
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().clone()) } else { None })
                        .ok_or("DExpr lit_str missing 'value' string field")?;
                    Ok(DExpr::LitStr(val))
                }
                "binop" => {
                    let op_str = fields
                        .get("op")
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().as_str()) } else { None })
                        .ok_or("DExpr binop missing 'op' field")?;
                    let op = parse_binop(op_str)?;
                    let left = fields.get("left").ok_or("DExpr binop missing 'left'")?;
                    let right = fields.get("right").ok_or("DExpr binop missing 'right'")?;
                    Ok(DExpr::BinOp {
                        op,
                        left: Box::new(value_to_dexpr(left)?),
                        right: Box::new(value_to_dexpr(right)?),
                    })
                }
                "count" => Ok(DExpr::Count),
                other => Err(format!("unknown DExpr kind: {other}")),
            }
        }
        _ => Err(format!(
            "cannot convert {} to DExpr (expected DExpr struct, Int, Float, Bool, or String)",
            v.type_name()
        )),
    }
}

fn parse_binop(s: &str) -> Result<DBinOp, String> {
    match s {
        "+" | "add" => Ok(DBinOp::Add),
        "-" | "sub" => Ok(DBinOp::Sub),
        "*" | "mul" => Ok(DBinOp::Mul),
        "/" | "div" => Ok(DBinOp::Div),
        ">" | "gt" => Ok(DBinOp::Gt),
        "<" | "lt" => Ok(DBinOp::Lt),
        ">=" | "ge" => Ok(DBinOp::Ge),
        "<=" | "le" => Ok(DBinOp::Le),
        "==" | "eq" => Ok(DBinOp::Eq),
        "!=" | "ne" => Ok(DBinOp::Ne),
        "&&" | "and" => Ok(DBinOp::And),
        "||" | "or" => Ok(DBinOp::Or),
        other => Err(format!("unknown binop: {other}")),
    }
}

/// Parse a `Value::Struct` representing a TidyAgg, e.g.:
///   Struct { name: "TidyAgg", kind: "sum", col: "salary" }
///   Struct { name: "TidyAgg", kind: "count" }
fn value_to_tidy_agg(v: &Value) -> Result<TidyAgg, String> {
    match v {
        Value::Struct { name, fields } if name == "TidyAgg" => {
            let kind = fields
                .get("kind")
                .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().as_str()) } else { None })
                .ok_or("TidyAgg struct missing 'kind' string")?;
            match kind {
                "count" => Ok(TidyAgg::Count),
                "sum" | "mean" | "min" | "max" | "first" | "last"
                | "median" | "sd" | "var" | "n_distinct" | "iqr" => {
                    let col = fields
                        .get("col")
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().clone()) } else { None })
                        .ok_or_else(|| format!("TidyAgg {kind} missing 'col' string"))?;
                    match kind {
                        "sum" => Ok(TidyAgg::Sum(col)),
                        "mean" => Ok(TidyAgg::Mean(col)),
                        "min" => Ok(TidyAgg::Min(col)),
                        "max" => Ok(TidyAgg::Max(col)),
                        "first" => Ok(TidyAgg::First(col)),
                        "last" => Ok(TidyAgg::Last(col)),
                        "median" => Ok(TidyAgg::Median(col)),
                        "sd" => Ok(TidyAgg::Sd(col)),
                        "var" => Ok(TidyAgg::Var(col)),
                        "n_distinct" => Ok(TidyAgg::NDistinct(col)),
                        "iqr" => Ok(TidyAgg::Iqr(col)),
                        _ => unreachable!(),
                    }
                }
                "quantile" => {
                    let col = fields
                        .get("col")
                        .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().clone()) } else { None })
                        .ok_or("TidyAgg quantile missing 'col' string")?;
                    let p = fields
                        .get("p")
                        .and_then(|v| match v {
                            Value::Float(f) => Some(*f),
                            Value::Int(i) => Some(*i as f64),
                            _ => None,
                        })
                        .ok_or("TidyAgg quantile missing 'p' float")?;
                    Ok(TidyAgg::Quantile(col, p))
                }
                other => Err(format!("unknown TidyAgg kind: {other}")),
            }
        }
        _ => Err(format!("expected TidyAgg struct, got {}", v.type_name())),
    }
}

/// Parse ArrangeKey array. Each element can be:
///   - String "col_name"       → ascending
///   - Struct { name: "ArrangeKey", col: "name", desc: bool }
fn value_to_arrange_keys(v: &Value) -> Result<Vec<ArrangeKey>, String> {
    match v {
        Value::Array(arr) => {
            let mut keys = Vec::with_capacity(arr.len());
            for item in arr.iter() {
                match item {
                    Value::String(s) => keys.push(ArrangeKey::asc(s)),
                    Value::Struct { name, fields } if name == "ArrangeKey" => {
                        let col = fields
                            .get("col")
                            .and_then(|v| if let Value::String(s) = v { Some(s.as_ref().as_str()) } else { None })
                            .ok_or("ArrangeKey missing 'col'")?;
                        let desc = fields
                            .get("desc")
                            .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                            .unwrap_or(false);
                        keys.push(if desc { ArrangeKey::desc(col) } else { ArrangeKey::asc(col) });
                    }
                    _ => return Err(format!("arrange key must be String or ArrangeKey struct, got {}", item.type_name())),
                }
            }
            Ok(keys)
        }
        _ => Err(format!("arrange requires Array of keys, got {}", v.type_name())),
    }
}

/// Parse rename pairs from `[["old","new"], ["old2","new2"]]`.
fn value_to_rename_pairs(v: &Value) -> Result<Vec<(String, String)>, String> {
    match v {
        Value::Array(arr) => {
            let mut pairs = Vec::with_capacity(arr.len());
            for item in arr.iter() {
                match item {
                    Value::Array(pair) if pair.len() == 2 => {
                        let old = value_to_string(&pair[0])?;
                        let new = value_to_string(&pair[1])?;
                        pairs.push((old, new));
                    }
                    _ => return Err("rename pairs must be arrays of [old, new] strings".into()),
                }
            }
            Ok(pairs)
        }
        _ => Err(format!("rename requires Array of pairs, got {}", v.type_name())),
    }
}

// ============================================================================
//  Join dispatcher
// ============================================================================

/// Dispatch inner_join / left_join / semi_join / anti_join.
///
/// The CJC API is: `view.inner_join(other, left_on, right_on)`.
/// The Rust API is: `view.inner_join(&other, &[(&left_on, &right_on)])`.
fn dispatch_join(
    view: &TidyView,
    args: &[Value],
    kind: &str,
) -> Result<Option<Value>, String> {
    if args.len() != 3 {
        return Err(format!(
            "TidyView.{kind} requires 3 args: other_view, left_on, right_on"
        ));
    }
    let other_rc = match &args[0] {
        Value::TidyView(rc) => rc,
        _ => return Err(format!("{kind}: first arg must be a TidyView")),
    };
    let other = downcast_view(other_rc)?;
    let left_on = value_to_string(&args[1])?;
    let right_on = value_to_string(&args[2])?;
    let on_pairs: Vec<(&str, &str)> = vec![(&left_on, &right_on)];

    match kind {
        "inner_join" => {
            let frame = view.inner_join(other, &on_pairs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(frame.view())))
        }
        "left_join" => {
            let frame = view.left_join(other, &on_pairs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(frame.view())))
        }
        "semi_join" => {
            let new_view = view.semi_join(other, &on_pairs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }
        "anti_join" => {
            let new_view = view.anti_join(other, &on_pairs).map_err(|e| format!("{e}"))?;
            Ok(Some(wrap_view(new_view)))
        }
        _ => Ok(None),
    }
}

// ============================================================================
//  Column → Value conversion
// ============================================================================

/// Convert a `Column` to a `Value::Array`.
fn column_to_value(col: &Column) -> Value {
    let vals: Vec<Value> = match col {
        Column::Int(v) => v.iter().map(|i| Value::Int(*i)).collect(),
        Column::Float(v) => v.iter().map(|f| Value::Float(*f)).collect(),
        Column::Str(v) => v
            .iter()
            .map(|s| Value::String(Rc::new(s.clone())))
            .collect(),
        Column::Bool(v) => v.iter().map(|b| Value::Bool(*b)).collect(),
        Column::Categorical { levels, codes } => codes
            .iter()
            .map(|&c| Value::String(Rc::new(levels[c as usize].clone())))
            .collect(),
        Column::DateTime(v) => v.iter().map(|i| Value::Int(*i)).collect(),
    };
    Value::Array(Rc::new(vals))
}

// ============================================================================
//  DataFrame → Value (for .collect())
// ============================================================================

/// Convert a `DataFrame` to the legacy `Value::Struct { name: "DataFrame" }`
/// representation used by existing CJC code.
pub fn dataframe_to_value(df: DataFrame) -> Value {
    let mut fields = std::collections::BTreeMap::new();
    let mut col_names: Vec<Value> = Vec::new();
    let nrows = df.nrows();
    for (name, col) in &df.columns {
        col_names.push(Value::String(Rc::new(name.clone())));
        fields.insert(name.clone(), column_to_value(col));
    }
    fields.insert(
        "__columns".to_string(),
        Value::Array(Rc::new(col_names)),
    );
    fields.insert("__nrows".to_string(), Value::Int(nrows as i64));
    Value::Struct {
        name: "DataFrame".to_string(),
        fields,
    }
}

/// Produce a human-readable table-formatted string from a DataFrame.
fn format_dataframe(df: &DataFrame) -> String {
    let ncols = df.ncols();
    let nrows = df.nrows();
    if ncols == 0 {
        return "DataFrame(0x0)".to_string();
    }

    // Column names
    let names: Vec<&str> = df.columns.iter().map(|(n, _)| n.as_str()).collect();

    // Compute widths
    let mut widths: Vec<usize> = names.iter().map(|n| n.len()).collect();
    let display_rows = nrows.min(20); // cap at 20 rows for display
    let mut cells: Vec<Vec<String>> = Vec::with_capacity(display_rows);
    for r in 0..display_rows {
        let mut row: Vec<String> = Vec::with_capacity(ncols);
        for (ci, (_, col)) in df.columns.iter().enumerate() {
            let s = col.get_display(r);
            if s.len() > widths[ci] {
                widths[ci] = s.len();
            }
            row.push(s);
        }
        cells.push(row);
    }

    let mut out = String::new();
    // Header
    for (ci, name) in names.iter().enumerate() {
        if ci > 0 { out.push_str("  "); }
        out.push_str(&format!("{:>width$}", name, width = widths[ci]));
    }
    out.push('\n');
    // Rows
    for row in &cells {
        for (ci, cell) in row.iter().enumerate() {
            if ci > 0 { out.push_str("  "); }
            out.push_str(&format!("{:>width$}", cell, width = widths[ci]));
        }
        out.push('\n');
    }
    if nrows > display_rows {
        out.push_str(&format!("... ({} more rows)\n", nrows - display_rows));
    }
    out
}

// ============================================================================
//  DExpr builder builtins (col, binop, agg, etc.)
// ============================================================================

/// Build a `Value::Struct { name: "DExpr", kind: "col", ... }` from a column name.
pub fn build_col_expr(name: &str) -> Value {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("kind".to_string(), Value::String(Rc::new("col".to_string())));
    fields.insert("value".to_string(), Value::String(Rc::new(name.to_string())));
    Value::Struct { name: "DExpr".to_string(), fields }
}

/// Build a DExpr binary operation.
pub fn build_binop_expr(op: &str, left: Value, right: Value) -> Value {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("kind".to_string(), Value::String(Rc::new("binop".to_string())));
    fields.insert("op".to_string(), Value::String(Rc::new(op.to_string())));
    fields.insert("left".to_string(), left);
    fields.insert("right".to_string(), right);
    Value::Struct { name: "DExpr".to_string(), fields }
}

/// Build a TidyAgg struct value.
pub fn build_tidy_agg(kind: &str, col: Option<&str>) -> Value {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("kind".to_string(), Value::String(Rc::new(kind.to_string())));
    if let Some(c) = col {
        fields.insert("col".to_string(), Value::String(Rc::new(c.to_string())));
    }
    Value::Struct { name: "TidyAgg".to_string(), fields }
}

/// Build an ArrangeKey struct value.
pub fn build_arrange_key(col: &str, descending: bool) -> Value {
    let mut fields = std::collections::BTreeMap::new();
    fields.insert("col".to_string(), Value::String(Rc::new(col.to_string())));
    fields.insert("desc".to_string(), Value::Bool(descending));
    Value::Struct { name: "ArrangeKey".to_string(), fields }
}

/// Dispatch builder builtins like `col()`, `desc()`, `asc()`, `sum()`, `mean()`, etc.
/// Returns `Ok(Some(value))` if recognised, `Ok(None)` otherwise.
pub fn dispatch_tidy_builtin(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    match name {
        // DExpr builders
        "col" => {
            if args.len() != 1 {
                return Err("col() requires 1 argument: column name".into());
            }
            let name = value_to_string(&args[0])?;
            Ok(Some(build_col_expr(&name)))
        }
        "desc" => {
            if args.len() != 1 {
                return Err("desc() requires 1 argument: column name".into());
            }
            let name = value_to_string(&args[0])?;
            Ok(Some(build_arrange_key(&name, true)))
        }
        "asc" => {
            if args.len() != 1 {
                return Err("asc() requires 1 argument: column name".into());
            }
            let name = value_to_string(&args[0])?;
            Ok(Some(build_arrange_key(&name, false)))
        }
        // DExpr binary op builder
        "dexpr_binop" => {
            if args.len() != 3 {
                return Err("dexpr_binop() requires 3 args: op, left, right".into());
            }
            let op = value_to_string(&args[0])?;
            Ok(Some(build_binop_expr(&op, args[1].clone(), args[2].clone())))
        }

        // TidyAgg builders
        "tidy_count" => Ok(Some(build_tidy_agg("count", None))),
        "tidy_sum" => {
            if args.len() != 1 { return Err("tidy_sum() requires 1 argument: column name".into()); }
            let col = value_to_string(&args[0])?;
            Ok(Some(build_tidy_agg("sum", Some(&col))))
        }
        "tidy_mean" => {
            if args.len() != 1 { return Err("tidy_mean() requires 1 argument: column name".into()); }
            let col = value_to_string(&args[0])?;
            Ok(Some(build_tidy_agg("mean", Some(&col))))
        }
        "tidy_min" => {
            if args.len() != 1 { return Err("tidy_min() requires 1 argument: column name".into()); }
            let col = value_to_string(&args[0])?;
            Ok(Some(build_tidy_agg("min", Some(&col))))
        }
        "tidy_max" => {
            if args.len() != 1 { return Err("tidy_max() requires 1 argument: column name".into()); }
            let col = value_to_string(&args[0])?;
            Ok(Some(build_tidy_agg("max", Some(&col))))
        }
        "tidy_first" => {
            if args.len() != 1 { return Err("tidy_first() requires 1 argument: column name".into()); }
            let col = value_to_string(&args[0])?;
            Ok(Some(build_tidy_agg("first", Some(&col))))
        }
        "tidy_last" => {
            if args.len() != 1 { return Err("tidy_last() requires 1 argument: column name".into()); }
            let col = value_to_string(&args[0])?;
            Ok(Some(build_tidy_agg("last", Some(&col))))
        }

        // =====================================================================
        //  stringr builtins — byte-first string view approach
        //
        //  CJC strings are UTF-8 byte sequences. These functions operate on the
        //  byte representation via cjc-regex's Thompson NFA. Where possible,
        //  results are slices (zero-copy views) of the input. Allocation happens
        //  only when replacement or splitting creates new buffers.
        //
        //  Key design point: patterns are compiled fresh per call. For hot-loop
        //  use, prefer the compiled Regex value type (regex literal `/pattern/`).
        // =====================================================================

        "str_detect" => {
            // str_detect(haystack, pattern) → bool
            if args.len() != 2 { return Err("str_detect requires 2 args: string, pattern".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            let matched = cjc_regex::is_match(&pat, "", hay.as_bytes());
            Ok(Some(Value::Bool(matched)))
        }
        "str_extract" => {
            // str_extract(haystack, pattern) → string (first match) or ""
            if args.len() != 2 { return Err("str_extract requires 2 args: string, pattern".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            match cjc_regex::find(&pat, "", hay.as_bytes()) {
                Some((start, end)) => {
                    let slice = &hay.as_bytes()[start..end];
                    let s = String::from_utf8_lossy(slice).to_string();
                    Ok(Some(Value::String(Rc::new(s))))
                }
                None => Ok(Some(Value::String(Rc::new(String::new())))),
            }
        }
        "str_extract_all" => {
            // str_extract_all(haystack, pattern) → [string]
            if args.len() != 2 { return Err("str_extract_all requires 2 args: string, pattern".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            let matches = cjc_regex::find_all(&pat, "", hay.as_bytes());
            let vals: Vec<Value> = matches
                .iter()
                .map(|&(start, end)| {
                    let slice = &hay.as_bytes()[start..end];
                    Value::String(Rc::new(String::from_utf8_lossy(slice).to_string()))
                })
                .collect();
            Ok(Some(Value::Array(Rc::new(vals))))
        }
        "str_replace" => {
            // str_replace(haystack, pattern, replacement) → string (first match replaced)
            if args.len() != 3 { return Err("str_replace requires 3 args: string, pattern, replacement".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            let rep = value_to_string(&args[2])?;
            match cjc_regex::find(&pat, "", hay.as_bytes()) {
                Some((start, end)) => {
                    let mut result = String::with_capacity(hay.len());
                    result.push_str(&hay[..start]);
                    result.push_str(&rep);
                    result.push_str(&hay[end..]);
                    Ok(Some(Value::String(Rc::new(result))))
                }
                None => Ok(Some(Value::String(Rc::new(hay)))),
            }
        }
        "str_replace_all" => {
            // str_replace_all(haystack, pattern, replacement) → string (all matches replaced)
            if args.len() != 3 { return Err("str_replace_all requires 3 args: string, pattern, replacement".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            let rep = value_to_string(&args[2])?;
            let matches = cjc_regex::find_all(&pat, "", hay.as_bytes());
            if matches.is_empty() {
                return Ok(Some(Value::String(Rc::new(hay))));
            }
            let mut result = String::with_capacity(hay.len());
            let mut last_end = 0;
            for &(start, end) in &matches {
                result.push_str(&hay[last_end..start]);
                result.push_str(&rep);
                last_end = end;
            }
            result.push_str(&hay[last_end..]);
            Ok(Some(Value::String(Rc::new(result))))
        }
        "str_split" => {
            // str_split(haystack, pattern) → [string]
            if args.len() != 2 { return Err("str_split requires 2 args: string, pattern".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            let spans = cjc_regex::split(&pat, "", hay.as_bytes());
            let vals: Vec<Value> = spans
                .iter()
                .map(|&(start, end)| {
                    Value::String(Rc::new(
                        String::from_utf8_lossy(&hay.as_bytes()[start..end]).to_string(),
                    ))
                })
                .collect();
            Ok(Some(Value::Array(Rc::new(vals))))
        }
        "str_count" => {
            // str_count(haystack, pattern) → int (number of matches)
            if args.len() != 2 { return Err("str_count requires 2 args: string, pattern".into()); }
            let hay = value_to_string(&args[0])?;
            let pat = value_to_string(&args[1])?;
            let count = cjc_regex::find_all(&pat, "", hay.as_bytes()).len();
            Ok(Some(Value::Int(count as i64)))
        }
        "str_trim" => {
            // str_trim(string) → string with leading/trailing whitespace removed
            if args.len() != 1 { return Err("str_trim requires 1 arg: string".into()); }
            let s = value_to_string(&args[0])?;
            Ok(Some(Value::String(Rc::new(s.trim().to_string()))))
        }
        "str_to_upper" => {
            if args.len() != 1 { return Err("str_to_upper requires 1 arg: string".into()); }
            let s = value_to_string(&args[0])?;
            Ok(Some(Value::String(Rc::new(s.to_uppercase()))))
        }
        "str_to_lower" => {
            if args.len() != 1 { return Err("str_to_lower requires 1 arg: string".into()); }
            let s = value_to_string(&args[0])?;
            Ok(Some(Value::String(Rc::new(s.to_lowercase()))))
        }
        "str_starts" => {
            if args.len() != 2 { return Err("str_starts requires 2 args: string, prefix".into()); }
            let s = value_to_string(&args[0])?;
            let prefix = value_to_string(&args[1])?;
            Ok(Some(Value::Bool(s.starts_with(&prefix))))
        }
        "str_ends" => {
            if args.len() != 2 { return Err("str_ends requires 2 args: string, suffix".into()); }
            let s = value_to_string(&args[0])?;
            let suffix = value_to_string(&args[1])?;
            Ok(Some(Value::Bool(s.ends_with(&suffix))))
        }
        "str_sub" => {
            // str_sub(string, start, end) → substring (byte-indexed, clamped)
            if args.len() != 3 { return Err("str_sub requires 3 args: string, start, end".into()); }
            let s = value_to_string(&args[0])?;
            let start = value_to_usize(&args[1])?.min(s.len());
            let end = value_to_usize(&args[2])?.min(s.len());
            if start > end {
                Ok(Some(Value::String(Rc::new(String::new()))))
            } else {
                // Clamp to char boundaries for safety
                let actual_start = clamp_to_char_boundary(&s, start);
                let actual_end = clamp_to_char_boundary(&s, end);
                Ok(Some(Value::String(Rc::new(s[actual_start..actual_end].to_string()))))
            }
        }
        "str_len" => {
            // str_len(string) → int (byte length, consistent with byte-first view)
            if args.len() != 1 { return Err("str_len requires 1 arg: string".into()); }
            let s = value_to_string(&args[0])?;
            Ok(Some(Value::Int(s.len() as i64)))
        }

        // =====================================================================
        //  Stats builtins (operate on Array of numbers)
        // =====================================================================

        "median" => {
            if args.len() != 1 { return Err("median requires 1 arg: numeric array".into()); }
            let nums = value_to_f64_vec(&args[0])?;
            if nums.is_empty() {
                return Ok(Some(Value::Float(f64::NAN)));
            }
            let mut sorted = nums;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            let med = if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            };
            Ok(Some(Value::Float(med)))
        }
        "sd" => {
            // Population standard deviation
            if args.len() != 1 { return Err("sd requires 1 arg: numeric array".into()); }
            let nums = value_to_f64_vec(&args[0])?;
            if nums.len() < 2 {
                return Ok(Some(Value::Float(f64::NAN)));
            }
            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
            let var = nums.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
                / (nums.len() - 1) as f64;
            Ok(Some(Value::Float(var.sqrt())))
        }
        "variance" => {
            // Sample variance (N-1 denominator)
            if args.len() != 1 { return Err("variance requires 1 arg: numeric array".into()); }
            let nums = value_to_f64_vec(&args[0])?;
            if nums.len() < 2 {
                return Ok(Some(Value::Float(f64::NAN)));
            }
            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
            let var = nums.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
                / (nums.len() - 1) as f64;
            Ok(Some(Value::Float(var)))
        }
        "n_distinct" => {
            // Count distinct values in an array
            if args.len() != 1 { return Err("n_distinct requires 1 arg: array".into()); }
            match &args[0] {
                Value::Array(arr) => {
                    let mut seen = std::collections::HashSet::new();
                    for v in arr.iter() {
                        seen.insert(format!("{v}"));
                    }
                    Ok(Some(Value::Int(seen.len() as i64)))
                }
                _ => Err(format!("n_distinct expects Array, got {}", args[0].type_name())),
            }
        }

        _ => Ok(None),
    }
}

/// Clamp a byte index to the nearest char boundary (round down).
fn clamp_to_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut i = idx;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Convert a Value::Array of numbers to Vec<f64>.
fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => {
            arr.iter()
                .map(|v| match v {
                    Value::Float(f) => Ok(*f),
                    Value::Int(i) => Ok(*i as f64),
                    _ => Err(format!("expected numeric value in array, got {}", v.type_name())),
                })
                .collect()
        }
        _ => Err(format!("expected Array, got {}", v.type_name())),
    }
}
