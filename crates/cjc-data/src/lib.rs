//! CJC Data DSL â€” Typed expression trees, logical plans, plan optimizer, and
//! column-buffer kernel execution.
//!
//! This implements the tidyverse-inspired data pipeline:
//! ```text
//! df |> filter(col("age") > 18) |> group_by("dept") |> summarize(avg_salary = mean(col("salary")))
//! ```

use cjc_repro::kahan_sum_f64;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

mod csv;
pub use csv::{CsvConfig, CsvReader, StreamingCsvProcessor};

pub mod tidy_dispatch;

// â”€â”€ Column Storage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A single column in a DataFrame.
#[derive(Debug, Clone)]
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
    Bool(Vec<bool>),
    /// Categorical column: sorted unique level names + per-row index into levels.
    Categorical {
        levels: Vec<String>,
        codes: Vec<u32>,
    },
    /// DateTime column: epoch milliseconds.
    DateTime(Vec<i64>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Categorical { codes, .. } => codes.len(),
            Column::DateTime(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Column::Int(_) => "Int",
            Column::Float(_) => "Float",
            Column::Str(_) => "Str",
            Column::Bool(_) => "Bool",
            Column::Categorical { .. } => "Categorical",
            Column::DateTime(_) => "DateTime",
        }
    }

    /// Get a display-friendly value at index.
    pub fn get_display(&self, idx: usize) -> String {
        match self {
            Column::Int(v) => format!("{}", v[idx]),
            Column::Float(v) => format!("{}", v[idx]),
            Column::Str(v) => v[idx].clone(),
            Column::Bool(v) => format!("{}", v[idx]),
            Column::Categorical { levels, codes } => levels[codes[idx] as usize].clone(),
            Column::DateTime(v) => format!("{}ms", v[idx]),
        }
    }
}

// â”€â”€ DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A columnar DataFrame.
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<(String, Column)>,
}

impl DataFrame {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    pub fn from_columns(columns: Vec<(String, Column)>) -> Result<Self, DataError> {
        if columns.is_empty() {
            return Ok(Self { columns });
        }
        let len = columns[0].1.len();
        for (name, col) in &columns {
            if col.len() != len {
                return Err(DataError::ColumnLengthMismatch {
                    expected: len,
                    got: col.len(),
                    column: name.clone(),
                });
            }
        }
        Ok(Self { columns })
    }

    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }

    /// Convert selected columns to a flat Vec<f64> (for tensor bridge).
    pub fn to_tensor_data(&self, col_names: &[&str]) -> Result<(Vec<f64>, Vec<usize>), DataError> {
        let nrows = self.nrows();
        let ncols = col_names.len();
        let mut data = Vec::with_capacity(nrows * ncols);

        for row in 0..nrows {
            for &col_name in col_names {
                let col = self
                    .get_column(col_name)
                    .ok_or_else(|| DataError::ColumnNotFound(col_name.to_string()))?;
                let val = match col {
                    Column::Float(v) => v[row],
                    Column::Int(v) => v[row] as f64,
                    _ => {
                        return Err(DataError::InvalidOperation(format!(
                            "column `{}` is not numeric",
                            col_name
                        )))
                    }
                };
                data.push(val);
            }
        }

        Ok((data, vec![nrows, ncols]))
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.columns.is_empty() {
            return write!(f, "(empty DataFrame)");
        }

        // Header
        let names: Vec<&str> = self.columns.iter().map(|(n, _)| n.as_str()).collect();
        let mut col_widths: Vec<usize> = names.iter().map(|n| n.len()).collect();

        // Compute column widths
        let nrows = self.nrows();
        for (col_idx, (_, col)) in self.columns.iter().enumerate() {
            for row in 0..nrows {
                let s = col.get_display(row);
                col_widths[col_idx] = col_widths[col_idx].max(s.len());
            }
        }

        // Print header
        for (i, name) in names.iter().enumerate() {
            if i > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{:>width$}", name, width = col_widths[i])?;
        }
        writeln!(f)?;

        // Separator
        for (i, &w) in col_widths.iter().enumerate() {
            if i > 0 {
                write!(f, "-+-")?;
            }
            write!(f, "{}", "-".repeat(w))?;
        }
        writeln!(f)?;

        // Rows
        for row in 0..nrows {
            for (col_idx, (_, col)) in self.columns.iter().enumerate() {
                if col_idx > 0 {
                    write!(f, " | ")?;
                }
                let s = col.get_display(row);
                write!(f, "{:>width$}", s, width = col_widths[col_idx])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

// â”€â”€ Data DSL Expression Trees â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// An expression in the Data DSL.
#[derive(Debug, Clone)]
pub enum DExpr {
    /// Column reference: col("name")
    Col(String),
    /// Literal integer
    LitInt(i64),
    /// Literal float
    LitFloat(f64),
    /// Literal bool
    LitBool(bool),
    /// Literal string
    LitStr(String),
    /// Binary operation
    BinOp {
        op: DBinOp,
        left: Box<DExpr>,
        right: Box<DExpr>,
    },
    /// Aggregation function
    Agg(AggFunc, Box<DExpr>),
    /// Count (no argument)
    Count,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

impl fmt::Display for DExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DExpr::Col(name) => write!(f, "col(\"{}\")", name),
            DExpr::LitInt(v) => write!(f, "{}", v),
            DExpr::LitFloat(v) => write!(f, "{}", v),
            DExpr::LitBool(b) => write!(f, "{}", b),
            DExpr::LitStr(s) => write!(f, "\"{}\"", s),
            DExpr::BinOp { op, left, right } => {
                let op_str = match op {
                    DBinOp::Add => "+",
                    DBinOp::Sub => "-",
                    DBinOp::Mul => "*",
                    DBinOp::Div => "/",
                    DBinOp::Gt => ">",
                    DBinOp::Lt => "<",
                    DBinOp::Ge => ">=",
                    DBinOp::Le => "<=",
                    DBinOp::Eq => "==",
                    DBinOp::Ne => "!=",
                    DBinOp::And => "&&",
                    DBinOp::Or => "||",
                };
                write!(f, "({} {} {})", left, op_str, right)
            }
            DExpr::Agg(func, expr) => {
                let name = match func {
                    AggFunc::Sum => "sum",
                    AggFunc::Mean => "mean",
                    AggFunc::Min => "min",
                    AggFunc::Max => "max",
                    AggFunc::Count => "count",
                };
                write!(f, "{}({})", name, expr)
            }
            DExpr::Count => write!(f, "count()"),
        }
    }
}

// â”€â”€ Logical Plan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A logical query plan node.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a source DataFrame.
    Scan {
        source: DataFrame,
    },
    /// Filter rows by predicate.
    Filter {
        input: Box<LogicalPlan>,
        predicate: DExpr,
    },
    /// Group by one or more columns.
    GroupBy {
        input: Box<LogicalPlan>,
        keys: Vec<String>,
    },
    /// Aggregate with named expressions.
    Aggregate {
        input: Box<LogicalPlan>,
        keys: Vec<String>,
        aggs: Vec<(String, DExpr)>,
    },
    /// Select/project specific columns.
    Project {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    /// Inner join: rows matching on both sides.
    InnerJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_on: String,
        right_on: String,
    },
    /// Left join: all left rows, matching right rows or null.
    LeftJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_on: String,
        right_on: String,
    },
    /// Cross join: cartesian product.
    CrossJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },
}

impl LogicalPlan {
    /// Collect the column names referenced by this plan (for pruning).
    pub fn referenced_columns(&self) -> Vec<String> {
        let mut cols = Vec::new();
        self.collect_columns(&mut cols);
        cols.sort();
        cols.dedup();
        cols
    }

    fn collect_columns(&self, cols: &mut Vec<String>) {
        match self {
            LogicalPlan::Scan { .. } => {}
            LogicalPlan::Filter { input, predicate } => {
                input.collect_columns(cols);
                collect_expr_columns(predicate, cols);
            }
            LogicalPlan::GroupBy { input, keys } => {
                input.collect_columns(cols);
                cols.extend(keys.clone());
            }
            LogicalPlan::Aggregate {
                input, keys, aggs, ..
            } => {
                input.collect_columns(cols);
                cols.extend(keys.clone());
                for (_, expr) in aggs {
                    collect_expr_columns(expr, cols);
                }
            }
            LogicalPlan::Project { input, columns } => {
                input.collect_columns(cols);
                cols.extend(columns.clone());
            }
            LogicalPlan::InnerJoin {
                left,
                right,
                left_on,
                right_on,
            }
            | LogicalPlan::LeftJoin {
                left,
                right,
                left_on,
                right_on,
            } => {
                left.collect_columns(cols);
                right.collect_columns(cols);
                cols.push(left_on.clone());
                cols.push(right_on.clone());
            }
            LogicalPlan::CrossJoin { left, right } => {
                left.collect_columns(cols);
                right.collect_columns(cols);
            }
        }
    }
}

fn collect_expr_columns(expr: &DExpr, cols: &mut Vec<String>) {
    match expr {
        DExpr::Col(name) => cols.push(name.clone()),
        DExpr::BinOp { left, right, .. } => {
            collect_expr_columns(left, cols);
            collect_expr_columns(right, cols);
        }
        DExpr::Agg(_, inner) => collect_expr_columns(inner, cols),
        _ => {}
    }
}

// â”€â”€ Plan Optimizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Optimize a logical plan.
pub fn optimize(plan: LogicalPlan) -> LogicalPlan {
    let plan = push_down_predicates(plan);
    let plan = prune_columns(plan);
    plan
}

/// Predicate pushdown: move Filter below GroupBy/Aggregate when possible.
fn push_down_predicates(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Filter {
            input,
            predicate,
        } => {
            let optimized_input = push_down_predicates(*input);
            match optimized_input {
                // Push filter below GroupBy if predicate only references keys
                LogicalPlan::GroupBy {
                    input: inner,
                    keys,
                } => {
                    let pred_cols = {
                        let mut c = Vec::new();
                        collect_expr_columns(&predicate, &mut c);
                        c
                    };
                    let can_push = pred_cols.iter().all(|c| !keys.contains(c))
                        || pred_cols.iter().all(|c| {
                            // Check if column exists in the input (not an aggregation)
                            !keys.contains(c) || keys.contains(c)
                        });
                    // Conservative: only push if predicate refs columns available before groupby
                    if can_push && pred_cols.iter().all(|c| !keys.contains(c)) {
                        LogicalPlan::GroupBy {
                            input: Box::new(LogicalPlan::Filter {
                                input: inner,
                                predicate,
                            }),
                            keys,
                        }
                    } else {
                        LogicalPlan::Filter {
                            input: Box::new(LogicalPlan::GroupBy {
                                input: inner,
                                keys,
                            }),
                            predicate,
                        }
                    }
                }
                other => LogicalPlan::Filter {
                    input: Box::new(other),
                    predicate,
                },
            }
        }
        LogicalPlan::GroupBy { input, keys } => LogicalPlan::GroupBy {
            input: Box::new(push_down_predicates(*input)),
            keys,
        },
        LogicalPlan::Aggregate {
            input,
            keys,
            aggs,
        } => LogicalPlan::Aggregate {
            input: Box::new(push_down_predicates(*input)),
            keys,
            aggs,
        },
        LogicalPlan::Project { input, columns } => LogicalPlan::Project {
            input: Box::new(push_down_predicates(*input)),
            columns,
        },
        LogicalPlan::InnerJoin {
            left,
            right,
            left_on,
            right_on,
        } => LogicalPlan::InnerJoin {
            left: Box::new(push_down_predicates(*left)),
            right: Box::new(push_down_predicates(*right)),
            left_on,
            right_on,
        },
        LogicalPlan::LeftJoin {
            left,
            right,
            left_on,
            right_on,
        } => LogicalPlan::LeftJoin {
            left: Box::new(push_down_predicates(*left)),
            right: Box::new(push_down_predicates(*right)),
            left_on,
            right_on,
        },
        LogicalPlan::CrossJoin { left, right } => LogicalPlan::CrossJoin {
            left: Box::new(push_down_predicates(*left)),
            right: Box::new(push_down_predicates(*right)),
        },
        other => other,
    }
}

/// Column pruning: add Project nodes to avoid materializing unused columns.
fn prune_columns(plan: LogicalPlan) -> LogicalPlan {
    // For v1, this is a no-op structural pass. Full implementation tracks
    // which columns are actually needed downstream and inserts Project nodes.
    plan
}

// â”€â”€ Plan Executor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Execute a logical plan against in-memory data.
pub fn execute(plan: &LogicalPlan) -> Result<DataFrame, DataError> {
    match plan {
        LogicalPlan::Scan { source } => Ok(source.clone()),

        LogicalPlan::Filter { input, predicate } => {
            let df = execute(input)?;
            execute_filter(&df, predicate)
        }

        LogicalPlan::GroupBy { input, keys: _ } => {
            // GroupBy alone just passes through; it's Aggregate that does the work
            let df = execute(input)?;
            // Return the data with a hint that it's grouped
            Ok(df)
        }

        LogicalPlan::Aggregate { input, keys, aggs } => {
            let df = execute(input)?;
            execute_aggregate(&df, keys, aggs)
        }

        LogicalPlan::Project { input, columns } => {
            let df = execute(input)?;
            let projected = df
                .columns
                .into_iter()
                .filter(|(name, _)| columns.contains(name))
                .collect();
            Ok(DataFrame { columns: projected })
        }

        LogicalPlan::InnerJoin {
            left,
            right,
            left_on,
            right_on,
        } => {
            let left_df = execute(left)?;
            let right_df = execute(right)?;
            execute_inner_join(&left_df, &right_df, left_on, right_on)
        }

        LogicalPlan::LeftJoin {
            left,
            right,
            left_on,
            right_on,
        } => {
            let left_df = execute(left)?;
            let right_df = execute(right)?;
            execute_left_join(&left_df, &right_df, left_on, right_on)
        }

        LogicalPlan::CrossJoin { left, right } => {
            let left_df = execute(left)?;
            let right_df = execute(right)?;
            execute_cross_join(&left_df, &right_df)
        }
    }
}

fn execute_filter(df: &DataFrame, predicate: &DExpr) -> Result<DataFrame, DataError> {
    let nrows = df.nrows();
    let mut mask = vec![false; nrows];

    for row in 0..nrows {
        let val = eval_expr_row(df, predicate, row)?;
        mask[row] = match val {
            ExprValue::Bool(b) => b,
            _ => return Err(DataError::InvalidOperation("filter predicate must be boolean".into())),
        };
    }

    let mut new_columns = Vec::new();
    for (name, col) in &df.columns {
        let filtered = filter_column(col, &mask);
        new_columns.push((name.clone(), filtered));
    }

    Ok(DataFrame {
        columns: new_columns,
    })
}

fn filter_column(col: &Column, mask: &[bool]) -> Column {
    match col {
        Column::Int(v) => Column::Int(
            v.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| *v)
                .collect(),
        ),
        Column::Float(v) => Column::Float(
            v.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| *v)
                .collect(),
        ),
        Column::Str(v) => Column::Str(
            v.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| v.clone())
                .collect(),
        ),
        Column::Bool(v) => Column::Bool(
            v.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| *v)
                .collect(),
        ),
        Column::Categorical { levels, codes } => Column::Categorical {
            levels: levels.clone(),
            codes: codes
                .iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| *v)
                .collect(),
        },
        Column::DateTime(v) => Column::DateTime(
            v.iter()
                .zip(mask)
                .filter(|(_, &m)| m)
                .map(|(v, _)| *v)
                .collect(),
        ),
    }
}

fn execute_aggregate(
    df: &DataFrame,
    keys: &[String],
    aggs: &[(String, DExpr)],
) -> Result<DataFrame, DataError> {
    // Build groups
    let nrows = df.nrows();
    let mut groups: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();

    for row in 0..nrows {
        let key: Vec<String> = keys
            .iter()
            .map(|k| {
                df.get_column(k)
                    .map(|col| col.get_display(row))
                    .ok_or_else(|| DataError::ColumnNotFound(k.to_string()))
            })
            .collect::<Result<Vec<String>, DataError>>()?;
        groups.entry(key).or_default().push(row);
    }

    // Sort groups for deterministic output
    let mut sorted_groups: Vec<(Vec<String>, Vec<usize>)> = groups.into_iter().collect();
    sorted_groups.sort_by(|a, b| a.0.cmp(&b.0));

    // Build result columns
    let mut result_columns: Vec<(String, Column)> = Vec::new();

    // Key columns
    for (key_idx, key_name) in keys.iter().enumerate() {
        let values: Vec<String> = sorted_groups
            .iter()
            .map(|(key, _)| key[key_idx].clone())
            .collect();
        // Determine type from source
        let source_col = df.get_column(key_name).ok_or_else(|| {
            DataError::ColumnNotFound(key_name.clone())
        })?;
        match source_col {
            Column::Int(_) => {
                let int_vals: Vec<i64> = values.iter().map(|s| s.parse().unwrap_or(0)).collect();
                result_columns.push((key_name.clone(), Column::Int(int_vals)));
            }
            Column::Str(_) => {
                result_columns.push((key_name.clone(), Column::Str(values)));
            }
            _ => {
                result_columns.push((key_name.clone(), Column::Str(values)));
            }
        }
    }

    // Aggregation columns
    for (agg_name, agg_expr) in aggs {
        let mut values = Vec::new();
        for (_, row_indices) in &sorted_groups {
            let val = eval_agg_expr(df, agg_expr, row_indices)?;
            values.push(val);
        }
        result_columns.push((agg_name.clone(), Column::Float(values)));
    }

    Ok(DataFrame {
        columns: result_columns,
    })
}

// â”€â”€ Expression Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[derive(Debug, Clone)]
enum ExprValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

fn eval_expr_row(df: &DataFrame, expr: &DExpr, row: usize) -> Result<ExprValue, DataError> {
    match expr {
        DExpr::Col(name) => {
            let col = df
                .get_column(name)
                .ok_or_else(|| DataError::ColumnNotFound(name.clone()))?;
            match col {
                Column::Int(v) => Ok(ExprValue::Int(v[row])),
                Column::Float(v) => Ok(ExprValue::Float(v[row])),
                Column::Str(v) => Ok(ExprValue::Str(v[row].clone())),
                Column::Bool(v) => Ok(ExprValue::Bool(v[row])),
                Column::Categorical { levels, codes } => {
                    Ok(ExprValue::Str(levels[codes[row] as usize].clone()))
                }
                Column::DateTime(v) => Ok(ExprValue::Int(v[row])),
            }
        }
        DExpr::LitInt(v) => Ok(ExprValue::Int(*v)),
        DExpr::LitFloat(v) => Ok(ExprValue::Float(*v)),
        DExpr::LitBool(b) => Ok(ExprValue::Bool(*b)),
        DExpr::LitStr(s) => Ok(ExprValue::Str(s.clone())),
        DExpr::BinOp { op, left, right } => {
            let lv = eval_expr_row(df, left, row)?;
            let rv = eval_expr_row(df, right, row)?;
            eval_binop(*op, lv, rv)
        }
        DExpr::Agg(_, _) | DExpr::Count => Err(DataError::InvalidOperation(
            "aggregation not allowed in row context".into(),
        )),
    }
}

fn eval_binop(op: DBinOp, left: ExprValue, right: ExprValue) -> Result<ExprValue, DataError> {
    match (left, right) {
        (ExprValue::Int(a), ExprValue::Int(b)) => match op {
            DBinOp::Add => Ok(ExprValue::Int(a + b)),
            DBinOp::Sub => Ok(ExprValue::Int(a - b)),
            DBinOp::Mul => Ok(ExprValue::Int(a * b)),
            DBinOp::Div => Ok(ExprValue::Int(a / b)),
            DBinOp::Gt => Ok(ExprValue::Bool(a > b)),
            DBinOp::Lt => Ok(ExprValue::Bool(a < b)),
            DBinOp::Ge => Ok(ExprValue::Bool(a >= b)),
            DBinOp::Le => Ok(ExprValue::Bool(a <= b)),
            DBinOp::Eq => Ok(ExprValue::Bool(a == b)),
            DBinOp::Ne => Ok(ExprValue::Bool(a != b)),
            _ => Err(DataError::InvalidOperation(format!(
                "unsupported op {:?} on Int",
                op
            ))),
        },
        (ExprValue::Float(a), ExprValue::Float(b)) => match op {
            DBinOp::Add => Ok(ExprValue::Float(a + b)),
            DBinOp::Sub => Ok(ExprValue::Float(a - b)),
            DBinOp::Mul => Ok(ExprValue::Float(a * b)),
            DBinOp::Div => Ok(ExprValue::Float(a / b)),
            DBinOp::Gt => Ok(ExprValue::Bool(a > b)),
            DBinOp::Lt => Ok(ExprValue::Bool(a < b)),
            DBinOp::Ge => Ok(ExprValue::Bool(a >= b)),
            DBinOp::Le => Ok(ExprValue::Bool(a <= b)),
            DBinOp::Eq => Ok(ExprValue::Bool(a == b)),
            DBinOp::Ne => Ok(ExprValue::Bool(a != b)),
            _ => Err(DataError::InvalidOperation(format!(
                "unsupported op {:?} on Float",
                op
            ))),
        },
        // Int promoted to Float
        (ExprValue::Int(a), ExprValue::Float(b)) => {
            eval_binop(op, ExprValue::Float(a as f64), ExprValue::Float(b))
        }
        (ExprValue::Float(a), ExprValue::Int(b)) => {
            eval_binop(op, ExprValue::Float(a), ExprValue::Float(b as f64))
        }
        (ExprValue::Bool(a), ExprValue::Bool(b)) => match op {
            DBinOp::And => Ok(ExprValue::Bool(a && b)),
            DBinOp::Or => Ok(ExprValue::Bool(a || b)),
            DBinOp::Eq => Ok(ExprValue::Bool(a == b)),
            DBinOp::Ne => Ok(ExprValue::Bool(a != b)),
            _ => Err(DataError::InvalidOperation(format!(
                "unsupported op {:?} on Bool",
                op
            ))),
        },
        (ExprValue::Str(a), ExprValue::Str(b)) => match op {
            DBinOp::Eq => Ok(ExprValue::Bool(a == b)),
            DBinOp::Ne => Ok(ExprValue::Bool(a != b)),
            _ => Err(DataError::InvalidOperation(format!(
                "unsupported op {:?} on String",
                op
            ))),
        },
        _ => Err(DataError::InvalidOperation(
            "type mismatch in binary operation".into(),
        )),
    }
}

fn eval_agg_expr(
    df: &DataFrame,
    expr: &DExpr,
    rows: &[usize],
) -> Result<f64, DataError> {
    match expr {
        DExpr::Agg(func, inner) => {
            let values = extract_float_values(df, inner, rows)?;
            match func {
                AggFunc::Sum => Ok(kahan_sum_f64(&values)),
                AggFunc::Mean => {
                    if values.is_empty() {
                        Ok(0.0)
                    } else {
                        Ok(kahan_sum_f64(&values) / values.len() as f64)
                    }
                }
                AggFunc::Min => Ok(values
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min)),
                AggFunc::Max => Ok(values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)),
                AggFunc::Count => Ok(values.len() as f64),
            }
        }
        DExpr::Count => Ok(rows.len() as f64),
        _ => Err(DataError::InvalidOperation(
            "expected aggregation expression".into(),
        )),
    }
}

fn extract_float_values(
    df: &DataFrame,
    expr: &DExpr,
    rows: &[usize],
) -> Result<Vec<f64>, DataError> {
    match expr {
        DExpr::Col(name) => {
            let col = df
                .get_column(name)
                .ok_or_else(|| DataError::ColumnNotFound(name.clone()))?;
            let vals: Vec<f64> = match col {
                Column::Float(v) => rows.iter().map(|&r| v[r]).collect(),
                Column::Int(v) => rows.iter().map(|&r| v[r] as f64).collect(),
                _ => {
                    return Err(DataError::InvalidOperation(format!(
                        "cannot aggregate non-numeric column `{}`",
                        name
                    )))
                }
            };
            Ok(vals)
        }
        _ => Err(DataError::InvalidOperation(
            "expected column reference in aggregation".into(),
        )),
    }
}

// â”€â”€ Pipeline Builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Fluent builder for data pipelines.
pub struct Pipeline {
    plan: LogicalPlan,
}

impl Pipeline {
    pub fn scan(df: DataFrame) -> Self {
        Self {
            plan: LogicalPlan::Scan { source: df },
        }
    }

    pub fn filter(self, predicate: DExpr) -> Self {
        Self {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate,
            },
        }
    }

    pub fn group_by(self, keys: Vec<String>) -> Self {
        Self {
            plan: LogicalPlan::GroupBy {
                input: Box::new(self.plan),
                keys,
            },
        }
    }

    pub fn summarize(self, keys: Vec<String>, aggs: Vec<(String, DExpr)>) -> Self {
        Self {
            plan: LogicalPlan::Aggregate {
                input: Box::new(self.plan),
                keys,
                aggs,
            },
        }
    }

    pub fn select(self, columns: Vec<String>) -> Self {
        Self {
            plan: LogicalPlan::Project {
                input: Box::new(self.plan),
                columns,
            },
        }
    }

    pub fn inner_join(self, right: DataFrame, left_on: &str, right_on: &str) -> Self {
        Self {
            plan: LogicalPlan::InnerJoin {
                left: Box::new(self.plan),
                right: Box::new(LogicalPlan::Scan { source: right }),
                left_on: left_on.to_string(),
                right_on: right_on.to_string(),
            },
        }
    }

    pub fn left_join(self, right: DataFrame, left_on: &str, right_on: &str) -> Self {
        Self {
            plan: LogicalPlan::LeftJoin {
                left: Box::new(self.plan),
                right: Box::new(LogicalPlan::Scan { source: right }),
                left_on: left_on.to_string(),
                right_on: right_on.to_string(),
            },
        }
    }

    pub fn cross_join(self, right: DataFrame) -> Self {
        Self {
            plan: LogicalPlan::CrossJoin {
                left: Box::new(self.plan),
                right: Box::new(LogicalPlan::Scan { source: right }),
            },
        }
    }

    /// Optimize and execute the pipeline.
    pub fn collect(self) -> Result<DataFrame, DataError> {
        let optimized = optimize(self.plan);
        execute(&optimized)
    }

    /// Get the logical plan (for inspection/testing).
    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }
}

// â”€â”€ Errors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[derive(Debug, Clone)]
pub enum DataError {
    ColumnNotFound(String),
    ColumnLengthMismatch {
        expected: usize,
        got: usize,
        column: String,
    },
    InvalidOperation(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::ColumnNotFound(name) => write!(f, "column `{}` not found", name),
            DataError::ColumnLengthMismatch {
                expected,
                got,
                column,
            } => write!(
                f,
                "column `{}` has {} rows, expected {}",
                column, got, expected
            ),
            DataError::InvalidOperation(msg) => write!(f, "invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for DataError {}

// â”€â”€ Join Execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Get a column value as a string for join key comparison.
fn column_value_str(col: &Column, row: usize) -> String {
    match col {
        Column::Int(v) => v[row].to_string(),
        Column::Float(v) => v[row].to_string(),
        Column::Str(v) => v[row].clone(),
        Column::Bool(v) => v[row].to_string(),
        Column::Categorical { levels, codes } => levels[codes[row] as usize].clone(),
        Column::DateTime(v) => v[row].to_string(),
    }
}

fn execute_inner_join(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &str,
    right_on: &str,
) -> Result<DataFrame, DataError> {
    let left_col = left.get_column(left_on)
        .ok_or_else(|| DataError::InvalidOperation(format!("join key `{}` not found in left", left_on)))?;
    let right_col = right.get_column(right_on)
        .ok_or_else(|| DataError::InvalidOperation(format!("join key `{}` not found in right", right_on)))?;

    // Build hash index on right table
    let right_nrows = right.nrows();
    let mut index: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..right_nrows {
        let key = column_value_str(right_col, i);
        index.entry(key).or_default().push(i);
    }

    let left_nrows = left.nrows();
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for i in 0..left_nrows {
        let key = column_value_str(left_col, i);
        if let Some(matches) = index.get(&key) {
            for &j in matches {
                left_indices.push(i);
                right_indices.push(j);
            }
        }
    }

    build_join_result(left, right, &left_indices, &right_indices, right_on)
}

fn execute_left_join(
    left: &DataFrame,
    right: &DataFrame,
    left_on: &str,
    right_on: &str,
) -> Result<DataFrame, DataError> {
    let left_col = left.get_column(left_on)
        .ok_or_else(|| DataError::InvalidOperation(format!("join key `{}` not found in left", left_on)))?;
    let right_col = right.get_column(right_on)
        .ok_or_else(|| DataError::InvalidOperation(format!("join key `{}` not found in right", right_on)))?;

    let right_nrows = right.nrows();
    let mut index: std::collections::BTreeMap<String, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..right_nrows {
        let key = column_value_str(right_col, i);
        index.entry(key).or_default().push(i);
    }

    let left_nrows = left.nrows();
    let mut left_indices = Vec::new();
    let mut right_indices: Vec<Option<usize>> = Vec::new();

    for i in 0..left_nrows {
        let key = column_value_str(left_col, i);
        if let Some(matches) = index.get(&key) {
            for &j in matches {
                left_indices.push(i);
                right_indices.push(Some(j));
            }
        } else {
            left_indices.push(i);
            right_indices.push(None);
        }
    }

    build_left_join_result(left, right, &left_indices, &right_indices, right_on)
}

fn execute_cross_join(left: &DataFrame, right: &DataFrame) -> Result<DataFrame, DataError> {
    let left_nrows = left.nrows();
    let right_nrows = right.nrows();
    let mut left_indices = Vec::with_capacity(left_nrows * right_nrows);
    let mut right_indices = Vec::with_capacity(left_nrows * right_nrows);

    for i in 0..left_nrows {
        for j in 0..right_nrows {
            left_indices.push(i);
            right_indices.push(j);
        }
    }

    build_join_result(left, right, &left_indices, &right_indices, "")
}

fn build_join_result(
    left: &DataFrame,
    right: &DataFrame,
    left_indices: &[usize],
    right_indices: &[usize],
    right_on: &str,
) -> Result<DataFrame, DataError> {
    let mut columns = Vec::new();

    // Add all left columns
    for (name, col) in &left.columns {
        columns.push((name.clone(), gather_column(col, left_indices)));
    }

    // Add right columns (skip the join key to avoid duplication)
    for (name, col) in &right.columns {
        if name == right_on {
            continue;
        }
        let out_name = if left.get_column(name).is_some() {
            format!("{}_right", name)
        } else {
            name.clone()
        };
        columns.push((out_name, gather_column(col, right_indices)));
    }

    Ok(DataFrame { columns })
}

fn build_left_join_result(
    left: &DataFrame,
    right: &DataFrame,
    left_indices: &[usize],
    right_indices: &[Option<usize>],
    right_on: &str,
) -> Result<DataFrame, DataError> {
    let mut columns = Vec::new();

    for (name, col) in &left.columns {
        columns.push((name.clone(), gather_column(col, left_indices)));
    }

    for (name, col) in &right.columns {
        if name == right_on {
            continue;
        }
        let out_name = if left.get_column(name).is_some() {
            format!("{}_right", name)
        } else {
            name.clone()
        };
        columns.push((out_name, gather_column_nullable(col, right_indices)));
    }

    Ok(DataFrame { columns })
}

fn gather_column(col: &Column, indices: &[usize]) -> Column {
    match col {
        Column::Int(v) => Column::Int(indices.iter().map(|&i| v[i]).collect()),
        Column::Float(v) => Column::Float(indices.iter().map(|&i| v[i]).collect()),
        Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
        Column::Bool(v) => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
        Column::Categorical { levels, codes } => Column::Categorical {
            levels: levels.clone(),
            codes: indices.iter().map(|&i| codes[i]).collect(),
        },
        Column::DateTime(v) => Column::DateTime(indices.iter().map(|&i| v[i]).collect()),
    }
}

fn gather_column_nullable(col: &Column, indices: &[Option<usize>]) -> Column {
    match col {
        Column::Int(v) => Column::Int(indices.iter().map(|opt| opt.map_or(0, |i| v[i])).collect()),
        Column::Float(v) => Column::Float(indices.iter().map(|opt| opt.map_or(f64::NAN, |i| v[i])).collect()),
        Column::Str(v) => Column::Str(indices.iter().map(|opt| opt.map_or_else(String::new, |i| v[i].clone())).collect()),
        Column::Bool(v) => Column::Bool(indices.iter().map(|opt| opt.map_or(false, |i| v[i])).collect()),
        Column::Categorical { levels, codes } => Column::Categorical {
            levels: levels.clone(),
            codes: indices.iter().map(|opt| opt.map_or(0, |i| codes[i])).collect(),
        },
        Column::DateTime(v) => Column::DateTime(indices.iter().map(|opt| opt.map_or(0, |i| v[i])).collect()),
    }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::from_columns(vec![
            (
                "name".into(),
                Column::Str(vec![
                    "Alice".into(),
                    "Bob".into(),
                    "Carol".into(),
                    "Dave".into(),
                    "Eve".into(),
                    "Frank".into(),
                ]),
            ),
            (
                "dept".into(),
                Column::Str(vec![
                    "eng".into(),
                    "eng".into(),
                    "sales".into(),
                    "eng".into(),
                    "sales".into(),
                    "eng".into(),
                ]),
            ),
            (
                "salary".into(),
                Column::Float(vec![95000.0, 102000.0, 78000.0, 110000.0, 82000.0, 98000.0]),
            ),
            (
                "tenure".into(),
                Column::Int(vec![3, 7, 2, 10, 1, 5]),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn test_dataframe_creation() {
        let df = sample_df();
        assert_eq!(df.nrows(), 6);
        assert_eq!(df.ncols(), 4);
        assert_eq!(
            df.column_names(),
            vec!["name", "dept", "salary", "tenure"]
        );
    }

    #[test]
    fn test_filter() {
        let df = sample_df();

        // Filter tenure > 2
        let result = Pipeline::scan(df)
            .filter(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("tenure".into())),
                right: Box::new(DExpr::LitInt(2)),
            })
            .collect()
            .unwrap();

        assert_eq!(result.nrows(), 4); // Alice(3), Bob(7), Dave(10), Frank(5)
    }

    #[test]
    fn test_group_by_summarize() {
        let df = sample_df();

        let result = Pipeline::scan(df)
            .summarize(
                vec!["dept".into()],
                vec![
                    (
                        "avg_salary".into(),
                        DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("salary".into()))),
                    ),
                    ("headcount".into(), DExpr::Count),
                ],
            )
            .collect()
            .unwrap();

        assert_eq!(result.nrows(), 2); // eng, sales

        // Find eng row
        let dept_col = result.get_column("dept").unwrap();
        let avg_col = result.get_column("avg_salary").unwrap();
        let count_col = result.get_column("headcount").unwrap();

        if let (Column::Str(depts), Column::Float(avgs), Column::Float(counts)) =
            (dept_col, avg_col, count_col)
        {
            let eng_idx = depts.iter().position(|d| d == "eng").unwrap();
            let sales_idx = depts.iter().position(|d| d == "sales").unwrap();

            // eng: (95000 + 102000 + 110000 + 98000) / 4 = 101250
            assert!((avgs[eng_idx] - 101250.0).abs() < 0.01);
            assert!((counts[eng_idx] - 4.0).abs() < 0.01);

            // sales: (78000 + 82000) / 2 = 80000
            assert!((avgs[sales_idx] - 80000.0).abs() < 0.01);
            assert!((counts[sales_idx] - 2.0).abs() < 0.01);
        } else {
            panic!("unexpected column types");
        }
    }

    #[test]
    fn test_filter_then_aggregate() {
        let df = sample_df();

        // Filter tenure > 2, then aggregate by dept
        let result = Pipeline::scan(df)
            .filter(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("tenure".into())),
                right: Box::new(DExpr::LitInt(2)),
            })
            .summarize(
                vec!["dept".into()],
                vec![
                    (
                        "avg_salary".into(),
                        DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("salary".into()))),
                    ),
                    (
                        "max_tenure".into(),
                        DExpr::Agg(AggFunc::Max, Box::new(DExpr::Col("tenure".into()))),
                    ),
                    ("headcount".into(), DExpr::Count),
                ],
            )
            .collect()
            .unwrap();

        // After filter: Alice(3,eng), Bob(7,eng), Dave(10,eng), Frank(5,eng)
        // Only eng remains
        assert_eq!(result.nrows(), 1);

        if let Column::Float(avgs) = result.get_column("avg_salary").unwrap() {
            // (95000 + 102000 + 110000 + 98000) / 4 = 101250
            assert!((avgs[0] - 101250.0).abs() < 0.01);
        }
        if let Column::Float(maxes) = result.get_column("max_tenure").unwrap() {
            assert!((maxes[0] - 10.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_to_tensor_data() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Float(vec![1.0, 2.0, 3.0])),
            ("y".into(), Column::Float(vec![4.0, 5.0, 6.0])),
        ])
        .unwrap();

        let (data, shape) = df.to_tensor_data(&["x", "y"]).unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_display() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Int(vec![1, 2, 3])),
            ("y".into(), Column::Float(vec![4.5, 5.5, 6.5])),
        ])
        .unwrap();

        let output = format!("{}", df);
        assert!(output.contains("x"));
        assert!(output.contains("y"));
        assert!(output.contains("4.5"));
    }

    #[test]
    fn test_column_not_found() {
        let df = sample_df();
        let result = Pipeline::scan(df)
            .filter(DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("nonexistent".into())),
                right: Box::new(DExpr::LitInt(0)),
            })
            .collect();

        assert!(result.is_err());
    }

    #[test]
    fn test_aggregation_functions() {
        let df = DataFrame::from_columns(vec![
            ("group".into(), Column::Str(vec!["a".into(), "a".into(), "a".into()])),
            ("val".into(), Column::Float(vec![10.0, 20.0, 30.0])),
        ])
        .unwrap();

        let result = Pipeline::scan(df)
            .summarize(
                vec!["group".into()],
                vec![
                    ("total".into(), DExpr::Agg(AggFunc::Sum, Box::new(DExpr::Col("val".into())))),
                    ("avg".into(), DExpr::Agg(AggFunc::Mean, Box::new(DExpr::Col("val".into())))),
                    ("lo".into(), DExpr::Agg(AggFunc::Min, Box::new(DExpr::Col("val".into())))),
                    ("hi".into(), DExpr::Agg(AggFunc::Max, Box::new(DExpr::Col("val".into())))),
                    ("n".into(), DExpr::Count),
                ],
            )
            .collect()
            .unwrap();

        if let Column::Float(totals) = result.get_column("total").unwrap() {
            assert!((totals[0] - 60.0).abs() < 0.01);
        }
        if let Column::Float(avgs) = result.get_column("avg").unwrap() {
            assert!((avgs[0] - 20.0).abs() < 0.01);
        }
        if let Column::Float(mins) = result.get_column("lo").unwrap() {
            assert!((mins[0] - 10.0).abs() < 0.01);
        }
        if let Column::Float(maxs) = result.get_column("hi").unwrap() {
            assert!((maxs[0] - 30.0).abs() < 0.01);
        }
        if let Column::Float(counts) = result.get_column("n").unwrap() {
            assert!((counts[0] - 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_empty_dataframe() {
        let df = DataFrame::new();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 0);
    }

    #[test]
    fn test_expr_display() {
        let expr = DExpr::BinOp {
            op: DBinOp::Gt,
            left: Box::new(DExpr::Col("age".into())),
            right: Box::new(DExpr::LitInt(18)),
        };
        assert_eq!(format!("{}", expr), "(col(\"age\") > 18)");
    }

    // ── Categorical Column and Encoding Tests ──────────────────────────────

    #[test]
    fn test_categorical_column_basics() {
        let col = Column::Categorical {
            levels: vec!["bird".into(), "cat".into(), "dog".into()],
            codes: vec![1, 2, 1, 0],
        };
        assert_eq!(col.len(), 4);
        assert_eq!(col.type_name(), "Categorical");
        assert_eq!(col.get_display(0), "cat");
        assert_eq!(col.get_display(1), "dog");
        assert_eq!(col.get_display(2), "cat");
        assert_eq!(col.get_display(3), "bird");
    }

    #[test]
    fn test_datetime_column_basics() {
        let col = Column::DateTime(vec![1000, 2000, 3000]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.type_name(), "DateTime");
        assert_eq!(col.get_display(0), "1000ms");
        assert_eq!(col.get_display(1), "2000ms");
    }

    #[test]
    fn test_label_encode() {
        let data: Vec<String> = vec!["cat".into(), "dog".into(), "cat".into(), "bird".into()];
        let (levels, codes) = label_encode(&data);
        assert_eq!(levels, vec!["bird", "cat", "dog"]);
        assert_eq!(codes, vec![1, 2, 1, 0]);
    }

    #[test]
    fn test_label_encode_empty() {
        let data: Vec<String> = vec![];
        let (levels, codes) = label_encode(&data);
        assert!(levels.is_empty());
        assert!(codes.is_empty());
    }

    #[test]
    fn test_label_encode_single_level() {
        let data: Vec<String> = vec!["x".into(), "x".into(), "x".into()];
        let (levels, codes) = label_encode(&data);
        assert_eq!(levels, vec!["x"]);
        assert_eq!(codes, vec![0, 0, 0]);
    }

    #[test]
    fn test_label_encode_deterministic() {
        // Run twice, must produce identical results (determinism)
        let data: Vec<String> = vec!["z".into(), "a".into(), "m".into(), "a".into(), "z".into()];
        let (levels1, codes1) = label_encode(&data);
        let (levels2, codes2) = label_encode(&data);
        assert_eq!(levels1, levels2);
        assert_eq!(codes1, codes2);
        // Sorted order
        assert_eq!(levels1, vec!["a", "m", "z"]);
    }

    #[test]
    fn test_ordinal_encode() {
        let data: Vec<String> = vec!["low".into(), "high".into(), "medium".into(), "low".into()];
        let order: Vec<String> = vec!["low".into(), "medium".into(), "high".into()];
        let (levels, codes) = ordinal_encode(&data, &order).unwrap();
        assert_eq!(levels, vec!["low", "medium", "high"]);
        assert_eq!(codes, vec![0, 2, 1, 0]);
    }

    #[test]
    fn test_ordinal_encode_missing_value() {
        let data: Vec<String> = vec!["low".into(), "unknown".into()];
        let order: Vec<String> = vec!["low".into(), "medium".into(), "high".into()];
        let result = ordinal_encode(&data, &order);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown"));
    }

    #[test]
    fn test_one_hot_encode() {
        let levels = vec!["bird".to_string(), "cat".to_string(), "dog".to_string()];
        let codes = vec![1u32, 2, 1, 0];
        let (names, cols) = one_hot_encode(&levels, &codes);
        assert_eq!(names, vec!["bird", "cat", "dog"]);
        assert_eq!(cols.len(), 3);
        // bird column: [false, false, false, true]
        assert_eq!(cols[0], vec![false, false, false, true]);
        // cat column: [true, false, true, false]
        assert_eq!(cols[1], vec![true, false, true, false]);
        // dog column: [false, true, false, false]
        assert_eq!(cols[2], vec![false, true, false, false]);

        // Each row has exactly one true
        for row in 0..4 {
            let count: usize = cols.iter().map(|c| if c[row] { 1 } else { 0 }).sum();
            assert_eq!(count, 1, "row {} should have exactly one true", row);
        }
    }

    #[test]
    fn test_one_hot_encode_empty() {
        let levels = vec!["a".to_string(), "b".to_string()];
        let codes: Vec<u32> = vec![];
        let (names, cols) = one_hot_encode(&levels, &codes);
        assert_eq!(names.len(), 2);
        assert!(cols[0].is_empty());
        assert!(cols[1].is_empty());
    }

    #[test]
    fn test_categorical_column_in_dataframe() {
        let data: Vec<String> = vec!["cat".into(), "dog".into(), "cat".into()];
        let (levels, codes) = label_encode(&data);
        let df = DataFrame::from_columns(vec![
            ("animal".into(), Column::Categorical { levels, codes }),
            ("score".into(), Column::Float(vec![1.0, 2.0, 3.0])),
        ])
        .unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.get_column("animal").unwrap().type_name(), "Categorical");
    }

    #[test]
    fn test_datetime_column_in_dataframe() {
        let df = DataFrame::from_columns(vec![
            ("ts".into(), Column::DateTime(vec![1000, 2000, 3000])),
            ("val".into(), Column::Float(vec![1.0, 2.0, 3.0])),
        ])
        .unwrap();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.get_column("ts").unwrap().type_name(), "DateTime");
    }

    #[test]
    fn test_label_encode_to_column_roundtrip() {
        let data: Vec<String> = vec!["cat".into(), "dog".into(), "cat".into(), "bird".into()];
        let (levels, codes) = label_encode(&data);
        let col = Column::Categorical { levels: levels.clone(), codes: codes.clone() };
        // Verify roundtrip: display values match originals
        for (i, original) in data.iter().enumerate() {
            assert_eq!(col.get_display(i), *original);
        }
    }
}

// â”€â”€ Phase 8: CSV Ingestion & Tensor Bridge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// â”€â”€ DataFrame â†” Tensor bridge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl DataFrame {
    /// Convert selected numeric columns to a `cjc_runtime::Tensor` with shape
    /// `[nrows, len(col_names)]` (row-major).
    ///
    /// All selected columns must be `Float` or `Int`; `Str` and `Bool` columns
    /// will return `DataError::InvalidOperation`.
    pub fn to_tensor(
        &self,
        col_names: &[&str],
    ) -> Result<cjc_runtime::Tensor, DataError> {
        let (data, shape) = self.to_tensor_data(col_names)?;
        cjc_runtime::Tensor::from_vec(data, &shape)
            .map_err(|e| DataError::InvalidOperation(format!("tensor conversion: {}", e)))
    }

    /// Append a single row of string values (parsed to the column type).
    ///
    /// `values` must match `self.ncols()` in length.
    /// Each string is parsed according to the existing column type:
    /// - `Float`: parsed as f64, falls back to 0.0 on parse error
    /// - `Int`:   parsed as i64, falls back to 0 on parse error
    /// - `Str`:   stored as-is
    /// - `Bool`:  `"true"` / `"1"` â†’ true, anything else â†’ false
    pub fn push_row(&mut self, values: &[&str]) -> Result<(), DataError> {
        if values.len() != self.ncols() {
            return Err(DataError::ColumnLengthMismatch {
                expected: self.ncols(),
                got: values.len(),
                column: "row".to_string(),
            });
        }
        for (i, (_, col)) in self.columns.iter_mut().enumerate() {
            let s = values[i];
            match col {
                Column::Float(v) => v.push(s.trim().parse::<f64>().unwrap_or(0.0)),
                Column::Int(v)   => v.push(s.trim().parse::<i64>().unwrap_or(0)),
                Column::Str(v)   => v.push(s.to_string()),
                Column::Bool(v)  => v.push(matches!(s.trim(), "true" | "1")),
                Column::Categorical { .. } => {
                    // Categorical columns are not populated via push_row
                }
                Column::DateTime(v) => v.push(s.trim().parse::<i64>().unwrap_or(0)),
            }
        }
        Ok(())
    }
}

// -- Phase 10: Tidy Primitives ------------------------------------------------
//
// Design goals:
//   - filter()  -> zero-allocation view (bitmask), O(N) time, O(N/8) extra mem
//   - select()  -> zero-allocation view (projection map), O(ncols) time/mem
//   - mutate()  -> new column buffers only, copy-on-write on alias, @nogc-safe
//   - All operations are bit-deterministic with stable iteration/column order.

// â”€â”€ BitMask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A compact, word-aligned bitmask over `nrows` rows.
///
/// Words are `u64`, stored LSB-first within each word. Bit `i` is set in
/// `words[i / 64]` at position `i % 64`. Tail bits (above `nrows`) are
/// guaranteed to be zero so that iteration never yields phantom rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitMask {
    words: Vec<u64>,
    nrows: usize,
}

impl BitMask {
    /// Construct a mask covering all `nrows` rows (all-true).
    pub fn all_true(nrows: usize) -> Self {
        let nwords = nwords_for(nrows);
        let mut words = vec![u64::MAX; nwords];
        // Zero tail bits for determinism
        if nrows % 64 != 0 && nwords > 0 {
            let tail = nrows % 64;
            words[nwords - 1] = (1u64 << tail) - 1;
        }
        BitMask { words, nrows }
    }

    /// Construct a mask where no rows are set (all-false).
    pub fn all_false(nrows: usize) -> Self {
        let nwords = nwords_for(nrows);
        BitMask {
            words: vec![0u64; nwords],
            nrows,
        }
    }

    /// Construct from a `Vec<bool>`, one entry per row.
    pub fn from_bools(bools: &[bool]) -> Self {
        let nrows = bools.len();
        let nwords = nwords_for(nrows);
        let mut words = vec![0u64; nwords];
        for (i, &b) in bools.iter().enumerate() {
            if b {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        BitMask { words, nrows }
    }

    /// Get bit at row `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.nrows);
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Number of set bits (masked-in rows).
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Merge two masks with AND semantics (chain of filter().filter()).
    ///
    /// Panics if `nrows` differs â€” this is a programming error (same base df).
    pub fn and(&self, other: &BitMask) -> BitMask {
        assert_eq!(
            self.nrows, other.nrows,
            "BitMask::and: nrows mismatch ({} vs {})",
            self.nrows, other.nrows
        );
        let words = self
            .words
            .iter()
            .zip(other.words.iter())
            .map(|(a, b)| a & b)
            .collect();
        BitMask {
            words,
            nrows: self.nrows,
        }
    }

    /// Iterate over set row indices in ascending order (deterministic).
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.nrows).filter(move |&i| self.get(i))
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of backing u64 words (= ceil(nrows / 64)).
    pub fn nwords(&self) -> usize {
        self.words.len()
    }
}

#[inline]
fn nwords_for(nrows: usize) -> usize {
    (nrows + 63) / 64
}

// â”€â”€ ProjectionMap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A stable ordered list of column indices into the base DataFrame.
///
/// Selecting 0 columns yields an empty projection (valid empty view).
/// Duplicate names are rejected at construction time â€” callers must deduplicate.
/// Column ordering is exactly the order supplied by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionMap {
    /// Column indices into the base DataFrame's `columns` vec.
    indices: Vec<usize>,
}

impl ProjectionMap {
    /// Identity projection (all columns, in original order).
    pub fn identity(ncols: usize) -> Self {
        ProjectionMap {
            indices: (0..ncols).collect(),
        }
    }

    /// Construct from explicit column indices.
    pub fn from_indices(indices: Vec<usize>) -> Self {
        ProjectionMap { indices }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

// â”€â”€ TidyView â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A lazy, zero-allocation view over a base `DataFrame`.
///
/// Holds:
///   â€¢ `base`   â€” shared reference to the underlying columnar data
///   â€¢ `mask`   â€” bitmask of which rows are visible
///   â€¢ `proj`   â€” ordered list of visible column indices
///
/// No column buffers are copied until `materialize()` / `to_tensor()` is called.
#[derive(Debug, Clone)]
pub struct TidyView {
    base: Rc<DataFrame>,
    mask: BitMask,
    proj: ProjectionMap,
}

impl TidyView {
    // â”€â”€ constructors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Wrap a `DataFrame` as a full view (all rows, all columns).
    pub fn from_df(df: DataFrame) -> Self {
        let nrows = df.nrows();
        let ncols = df.ncols();
        TidyView {
            base: Rc::new(df),
            mask: BitMask::all_true(nrows),
            proj: ProjectionMap::identity(ncols),
        }
    }

    /// Wrap a shared `Rc<DataFrame>` as a full view.
    pub fn from_rc(df: Rc<DataFrame>) -> Self {
        let nrows = df.nrows();
        let ncols = df.ncols();
        TidyView {
            base: df,
            mask: BitMask::all_true(nrows),
            proj: ProjectionMap::identity(ncols),
        }
    }

    // â”€â”€ shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Number of visible rows (set bits in mask).
    pub fn nrows(&self) -> usize {
        self.mask.count_ones()
    }

    /// Number of visible columns (length of projection).
    pub fn ncols(&self) -> usize {
        self.proj.len()
    }

    /// Names of projected columns in stable projection order.
    pub fn column_names(&self) -> Vec<&str> {
        self.proj
            .indices()
            .iter()
            .map(|&ci| self.base.columns[ci].0.as_str())
            .collect()
    }

    // â”€â”€ filter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Filter rows by a `DExpr` predicate.
    ///
    /// Returns a new `TidyView` with a tighter bitmask (AND with existing mask).
    /// Does NOT copy any column buffers.
    ///
    /// Edge cases:
    ///   â€¢ 0-row base â†’ empty mask returned, no panic.
    ///   â€¢ Non-bool predicate â†’ `TidyError::PredicateNotBool`.
    ///   â€¢ Float NaN comparisons â†’ deterministic: `NaN != NaN` (IEEE 754).
    ///   â€¢ Chained filters compose masks with AND without materializing.
    pub fn filter(&self, predicate: &DExpr) -> Result<TidyView, TidyError> {
        // Validate predicate type references known projected columns
        validate_expr_columns_proj(predicate, &self.base, &self.proj)?;

        let nrows_base = self.base.nrows();
        let mut new_words = self.mask.words.clone();

        // Evaluate predicate over every currently-masked-in row.
        // Rows masked out remain 0 (no change needed, AND semantics).
        for row in self.mask.iter_set() {
            let b = eval_expr_row_proj(&self.base, predicate, row, &self.proj)?;
            let pass = match b {
                ExprValue::Bool(v) => v,
                _ => {
                    return Err(TidyError::PredicateNotBool {
                        got: b.type_name().to_string(),
                    })
                }
            };
            if !pass {
                new_words[row / 64] &= !(1u64 << (row % 64));
            }
        }

        Ok(TidyView {
            base: Rc::clone(&self.base),
            mask: BitMask {
                words: new_words,
                nrows: nrows_base,
            },
            proj: self.proj.clone(),
        })
    }

    // â”€â”€ select â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Project to a subset of named columns (in the given order).
    ///
    /// Returns a new `TidyView` with an updated `ProjectionMap`.
    /// No column buffers are copied.
    ///
    /// Edge cases:
    ///   â€¢ 0 columns selected â†’ valid empty-column view (no error).
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ Duplicate column name in `cols` â†’ `TidyError::DuplicateColumn`.
    ///   â€¢ Column ordering is exactly as supplied.
    pub fn select(&self, cols: &[&str]) -> Result<TidyView, TidyError> {
        // Check for duplicates in the requested list
        {
            let mut seen = std::collections::BTreeSet::new();
            for &name in cols {
                if !seen.insert(name) {
                    return Err(TidyError::DuplicateColumn(name.to_string()));
                }
            }
        }

        // Resolve each name to an index in `self.base`
        let mut new_indices = Vec::with_capacity(cols.len());
        for &name in cols {
            let idx = self
                .base
                .columns
                .iter()
                .position(|(n, _)| n == name)
                .ok_or_else(|| TidyError::ColumnNotFound(name.to_string()))?;
            new_indices.push(idx);
        }

        Ok(TidyView {
            base: Rc::clone(&self.base),
            mask: self.mask.clone(),
            proj: ProjectionMap::from_indices(new_indices),
        })
    }

    // â”€â”€ mutate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Apply column-wise assignments and return a materialized `TidyFrame`.
    ///
    /// `assignments` is an ordered list of `(col_name, expr)` pairs evaluated
    /// left-to-right. Each assignment sees the *snapshot* of columns at entry
    /// to the mutate call (snapshot semantics â€” new columns created in earlier
    /// assignments are NOT visible to later assignments within the same call).
    ///
    /// Semantics decisions:
    ///   â€¢ Existing column â†’ overwritten (copy-on-write safe).
    ///   â€¢ New column â†’ appended after existing projected columns.
    ///   â€¢ Scalar broadcasting â†’ a scalar expr is broadcast to all visible rows.
    ///   â€¢ Mask-awareness: only masked-in rows are computed; masked-out rows in
    ///     the materialized output retain the base value (or zero for new cols).
    ///   â€¢ Type promotion: Int + Float â†’ Float; Int overflow â†’ wrapping.
    ///   â€¢ Multiple assignments with the same target name in one call â†’ error.
    ///   â€¢ Mutate on masked view produces a *materialized* `TidyFrame` where
    ///     only visible rows are present (mask applied during materialization).
    pub fn mutate(&self, assignments: &[(&str, DExpr)]) -> Result<TidyFrame, TidyError> {
        // Check for duplicate targets within this call
        {
            let mut seen = std::collections::BTreeSet::new();
            for &(name, _) in assignments {
                if !seen.insert(name) {
                    return Err(TidyError::DuplicateColumn(name.to_string()));
                }
            }
        }

        // Materialize the view into a fresh DataFrame (mask applied, cols projected)
        let mut df = self.materialize()?;

        // Snapshot: take column names present before any assignment
        let snapshot_names: Vec<String> = df.columns.iter().map(|(n, _)| n.clone()).collect();

        for &(col_name, ref expr) in assignments {
            // Validate that all column refs in expr exist in the snapshot
            validate_expr_columns_snapshot(expr, &snapshot_names)?;

            let nrows = df.nrows();
            // Evaluate expr for each row to build new column buffer
            let new_col = eval_expr_column(&df, expr, nrows)?;

            // Find or append column
            if let Some(pos) = df.columns.iter().position(|(n, _)| n == col_name) {
                df.columns[pos].1 = new_col;
            } else {
                df.columns.push((col_name.to_string(), new_col));
            }
        }

        Ok(TidyFrame {
            inner: Rc::new(RefCell::new(df)),
        })
    }

    // â”€â”€ materialize â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Materialize the view into a new `DataFrame` (applies mask + projection).
    ///
    /// Triggers exactly one allocation per visible column buffer.
    /// Rows are emitted in ascending index order (stable/deterministic).
    ///
    /// Edge cases:
    ///   â€¢ Empty rows â†’ 0-row DataFrame.
    ///   â€¢ Empty cols â†’ 0-column DataFrame.
    ///   â€¢ Row-major iteration is stable.
    pub fn materialize(&self) -> Result<DataFrame, TidyError> {
        let row_indices: Vec<usize> = self.mask.iter_set().collect();

        let mut columns = Vec::with_capacity(self.proj.len());
        for &ci in self.proj.indices() {
            let (name, col) = &self.base.columns[ci];
            let new_col = gather_column(col, &row_indices);
            columns.push((name.clone(), new_col));
        }

        DataFrame::from_columns(columns)
            .map_err(|e| TidyError::Internal(e.to_string()))
    }

    /// Convert visible numeric columns to a tensor (row-major).
    ///
    /// Only `Float` and `Int` columns are supported.
    pub fn to_tensor(&self, col_names: &[&str]) -> Result<cjc_runtime::Tensor, TidyError> {
        let df = self.materialize()?;
        df.to_tensor(col_names)
            .map_err(|e| TidyError::Internal(e.to_string()))
    }

    /// Access the underlying mask (for testing/inspection).
    pub fn mask(&self) -> &BitMask {
        &self.mask
    }

    /// Access the underlying projection (for testing/inspection).
    pub fn proj(&self) -> &ProjectionMap {
        &self.proj
    }

    /// Access a column from the underlying base DataFrame by name.
    ///
    /// Returns the raw `Column` (full length, unmasked) â€” callers must apply
    /// the mask themselves if needed.  Used by `fct_summary_means` and similar.
    pub fn base_column(&self, name: &str) -> Option<&Column> {
        self.base.columns.iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }
}

// â”€â”€ TidyFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A materialized, mutable DataFrame with copy-on-write alias safety.
///
/// Wraps `Rc<RefCell<DataFrame>>`. Cloning a `TidyFrame` shares the buffer;
/// writing through `mutate()` triggers a deep copy if the refcount > 1,
/// ensuring other views are not corrupted.
#[derive(Debug, Clone)]
pub struct TidyFrame {
    inner: Rc<RefCell<DataFrame>>,
}

impl TidyFrame {
    /// Wrap an existing `DataFrame`.
    pub fn from_df(df: DataFrame) -> Self {
        TidyFrame {
            inner: Rc::new(RefCell::new(df)),
        }
    }

    /// Get a shared view of the inner DataFrame.
    pub fn borrow(&self) -> std::cell::Ref<'_, DataFrame> {
        self.inner.borrow()
    }

    /// Apply further tidy operations on this frame.
    pub fn view(&self) -> TidyView {
        let df = self.inner.borrow().clone();
        TidyView::from_df(df)
    }

    /// Alias-safe mutate: if this `TidyFrame` is shared, clones first.
    pub fn mutate(&mut self, assignments: &[(&str, DExpr)]) -> Result<(), TidyError> {
        // Copy-on-write: if refcount > 1, deep-clone the inner DataFrame
        if Rc::strong_count(&self.inner) > 1 {
            let cloned = self.inner.borrow().clone();
            self.inner = Rc::new(RefCell::new(cloned));
        }

        // Check for duplicate targets
        {
            let mut seen = std::collections::BTreeSet::new();
            for &(name, _) in assignments {
                if !seen.insert(name) {
                    return Err(TidyError::DuplicateColumn(name.to_string()));
                }
            }
        }

        let mut df = self.inner.borrow_mut();

        // Snapshot column names before mutation
        let snapshot_names: Vec<String> = df.columns.iter().map(|(n, _)| n.clone()).collect();

        for &(col_name, ref expr) in assignments {
            validate_expr_columns_snapshot(expr, &snapshot_names)?;

            let nrows = df.nrows();
            let new_col = eval_expr_column(&df, expr, nrows)?;

            if let Some(pos) = df.columns.iter().position(|(n, _)| n == col_name) {
                df.columns[pos].1 = new_col;
            } else {
                df.columns.push((col_name.to_string(), new_col));
            }
        }

        Ok(())
    }
}

// â”€â”€ TidyError â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Errors produced by Phase 10 tidy operations.
#[derive(Debug, Clone, PartialEq)]
pub enum TidyError {
    /// A column referenced in the expression was not found.
    ColumnNotFound(String),
    /// A duplicate column name was supplied in select/mutate.
    DuplicateColumn(String),
    /// The filter predicate evaluated to a non-boolean value.
    PredicateNotBool { got: String },
    /// A mutate expression produced a type mismatch.
    TypeMismatch { expected: String, got: String },
    /// Scalar broadcast to a vector of non-matching length.
    LengthMismatch { expected: usize, got: usize },
    /// An internal/unexpected error (wraps DataError strings).
    Internal(String),
    /// `first()` or `last()` called on an empty group.
    EmptyGroup,
    /// Phase 17: too many distinct levels for a u16 FctColumn.
    CapacityExceeded { limit: usize, got: usize },
}

impl fmt::Display for TidyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TidyError::ColumnNotFound(n) => write!(f, "column `{}` not found", n),
            TidyError::DuplicateColumn(n) => write!(f, "duplicate column `{}`", n),
            TidyError::PredicateNotBool { got } => {
                write!(f, "filter predicate must be Bool, got {}", got)
            }
            TidyError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            TidyError::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "length mismatch: expected {} rows, got {}",
                    expected, got
                )
            }
            TidyError::Internal(msg) => write!(f, "internal error: {}", msg),
            TidyError::EmptyGroup => write!(f, "aggregation on empty group"),
            TidyError::CapacityExceeded { limit, got } => {
                write!(f, "factor capacity exceeded: limit {} distinct levels, got {}", limit, got)
            }
        }
    }
}

impl std::error::Error for TidyError {}

// â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Note: tidy uses the existing `gather_column(col, indices)` defined earlier in
// this file (line ~1160). No duplicate needed.

/// Evaluate a `DExpr` for all `nrows` rows, returning a typed `Column`.
///
/// Int + Float â†’ Float promotion.
/// Int overflow â†’ wrapping (i64 wrapping_add/mul etc.).
/// Scalar expression â†’ broadcast to all rows.
fn eval_expr_column(df: &DataFrame, expr: &DExpr, nrows: usize) -> Result<Column, TidyError> {
    if nrows == 0 {
        // Infer column type from a dry-run on nothing; default to Float for empty
        return Ok(Column::Float(vec![]));
    }

    // Evaluate row 0 to determine result type
    let sample = eval_dexpr_row(df, expr, 0)?;
    match sample {
        ExprValue::Int(_) => {
            let vals: Result<Vec<i64>, TidyError> = (0..nrows)
                .map(|r| {
                    eval_dexpr_row(df, expr, r).and_then(|v| match v {
                        ExprValue::Int(i) => Ok(i),
                        ExprValue::Float(f) => Ok(f as i64),
                        other => Err(TidyError::TypeMismatch {
                            expected: "Int".into(),
                            got: other.type_name().into(),
                        }),
                    })
                })
                .collect();
            Ok(Column::Int(vals?))
        }
        ExprValue::Float(_) => {
            let vals: Result<Vec<f64>, TidyError> = (0..nrows)
                .map(|r| {
                    eval_dexpr_row(df, expr, r).and_then(|v| match v {
                        ExprValue::Float(f) => Ok(f),
                        ExprValue::Int(i) => Ok(i as f64),
                        other => Err(TidyError::TypeMismatch {
                            expected: "Float".into(),
                            got: other.type_name().into(),
                        }),
                    })
                })
                .collect();
            Ok(Column::Float(vals?))
        }
        ExprValue::Bool(_) => {
            let vals: Result<Vec<bool>, TidyError> = (0..nrows)
                .map(|r| {
                    eval_dexpr_row(df, expr, r).and_then(|v| match v {
                        ExprValue::Bool(b) => Ok(b),
                        other => Err(TidyError::TypeMismatch {
                            expected: "Bool".into(),
                            got: other.type_name().into(),
                        }),
                    })
                })
                .collect();
            Ok(Column::Bool(vals?))
        }
        ExprValue::Str(_) => {
            let vals: Result<Vec<String>, TidyError> = (0..nrows)
                .map(|r| {
                    eval_dexpr_row(df, expr, r).and_then(|v| match v {
                        ExprValue::Str(s) => Ok(s),
                        other => Err(TidyError::TypeMismatch {
                            expected: "Str".into(),
                            got: other.type_name().into(),
                        }),
                    })
                })
                .collect();
            Ok(Column::Str(vals?))
        }
    }
}

/// Evaluate a `DExpr` at a single row (returns `ExprValue`).
fn eval_dexpr_row(df: &DataFrame, expr: &DExpr, row: usize) -> Result<ExprValue, TidyError> {
    eval_expr_row(df, expr, row).map_err(|e| TidyError::Internal(e.to_string()))
}

/// Evaluate a `DExpr` at a single row using the projection-aware base.
fn eval_expr_row_proj(
    base: &DataFrame,
    expr: &DExpr,
    row: usize,
    _proj: &ProjectionMap,
) -> Result<ExprValue, TidyError> {
    // We always evaluate against the full base (all columns accessible).
    // Projection only restricts what filter/select *expose*, not what predicates
    // can reference in the base frame.
    eval_expr_row(base, expr, row).map_err(|e| TidyError::Internal(e.to_string()))
}

/// Validate that all column references in `expr` exist in `base` (via projection
/// columns â€” filter predicates may reference any base column visible in proj).
///
/// For simplicity: filter predicates may reference ANY column in `base` because
/// the view is a window over the same base DataFrame. This is analogous to SQL
/// WHERE clauses that can reference any column, not just SELECT-listed ones.
fn validate_expr_columns_proj(
    expr: &DExpr,
    base: &DataFrame,
    _proj: &ProjectionMap,
) -> Result<(), TidyError> {
    let mut refs = Vec::new();
    collect_expr_columns(expr, &mut refs);
    for col_name in refs {
        if base.get_column(&col_name).is_none() {
            return Err(TidyError::ColumnNotFound(col_name));
        }
    }
    Ok(())
}

/// Validate that all column references in `expr` exist in `snapshot_names`.
fn validate_expr_columns_snapshot(
    expr: &DExpr,
    snapshot_names: &[String],
) -> Result<(), TidyError> {
    let mut refs = Vec::new();
    collect_expr_columns(expr, &mut refs);
    for col_name in refs {
        if !snapshot_names.iter().any(|n| n == &col_name) {
            return Err(TidyError::ColumnNotFound(col_name));
        }
    }
    Ok(())
}

impl ExprValue {
    fn type_name(&self) -> &'static str {
        match self {
            ExprValue::Int(_) => "Int",
            ExprValue::Float(_) => "Float",
            ExprValue::Str(_) => "Str",
            ExprValue::Bool(_) => "Bool",
        }
    }
}

// â”€â”€ DataFrame::tidy() convenience entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl DataFrame {
    /// Wrap this DataFrame as a `TidyView` for Phase 10 tidy operations.
    ///
    /// Consumes `self` (zero-copy â€” moves into an `Rc`).
    pub fn tidy(self) -> TidyView {
        TidyView::from_df(self)
    }

    /// Wrap this DataFrame as a `TidyFrame` for mutable tidy operations.
    pub fn tidy_mut(self) -> TidyFrame {
        TidyFrame::from_df(self)
    }
}

// â”€â”€ NoGC annotation gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// The @nogc verifier in cjc-mir/src/nogc_verify.rs tracks "safe builtins" that
// are known not to trigger GC. Phase 10 tidy operations on the Rust side are
// outside the CJC language runtime (they are library calls), so @nogc
// annotation at the CJC language level means: the CJC function body does not
// call gc_alloc. The Rust implementation of filter/select produces a TidyView
// that holds Rc references â€” no GC heap involvement.
//
// For the NoGC verifier to accept tidy calls inside @nogc CJC functions, the
// builtins "tidy_filter", "tidy_select", "tidy_materialize" must be added to
// the safe-builtins list in cjc-mir/src/nogc_verify.rs. See that file for
// the `is_safe_builtin` function.
//
// Allocation budget per operation (no GC heap, only Rust stack/heap via alloc):
//   filter  : O(N/64) u64 words for new mask   (â‰ˆ 8 bytes / 64 rows)
//   select  : O(K) usize indices (K = ncols selected)
//   mutate  : O(N) per new column buffer (allowed â€” one allocation per column)
//   materialize: O(N * K) total for visible data

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Phase 11â€“12: Grouping, Summarise, Arrange, Slice, Distinct, Joins
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//
// Spec-Lock Table
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// | Decision point            | Choice                                           |
// |---------------------------|--------------------------------------------------|
// | Group ordering            | First-occurrence order of keys in visible rows   |
// | Null/missing in keys      | Not applicable (CJC Column has no null type);    |
// |                           | NaN in Float keys: each NaN is its own group key |
// |                           | (NaN != NaN â†’ separate groups per NaN position)  |
// | summarise output order    | Stable: same order as group creation             |
// | Empty group agg behavior  | countâ†’0, sumâ†’0.0, meanâ†’NaN, minâ†’NaN, maxâ†’NaN,   |
// |                           | first/last â†’ TidyError::EmptyGroup               |
// | arrange tie-breaking      | Stable sort (Rust's slice::sort_by is stable);   |
// |                           | equal-key rows preserve original row order       |
// | NaN ordering in arrange   | NaN sorts LAST (greater than any finite value)   |
// | null ordering             | N/A â€” no null type in CJC                       |
// | slice_sample seed         | Deterministic LCG with caller-supplied u64 seed  |
// | slice_sample n > nrows    | Clamp to nrows (no error)                        |
// | distinct ordering         | First-occurrence order of distinct key combos    |
// | Join left order           | Preserved â€” output rows follow left row order    |
// | Join right match order    | Stable: sorted by right-side row index ascending  |
// | Null matching in joins    | N/A â€” no null type in CJC                       |
// | Join duplicate keys       | All matches included, deterministic order        |
// | many-many explosion order | Left outer loop, right inner loop (stable)       |
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// â”€â”€ RowIndexMap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A permutation / selection vector over base-frame row indices.
///
/// Used by `arrange` (sort) and `slice` to represent a reordering of the
/// rows visible through a `TidyView`'s bitmask, without copying column data.
/// The indices stored here are indices into the BASE `DataFrame`, not relative
/// to the mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowIndexMap {
    /// Ordered list of base-frame row indices that are visible after this op.
    /// Length == number of visible rows.
    pub(crate) indices: Vec<usize>,
}

impl RowIndexMap {
    pub fn new(indices: Vec<usize>) -> Self {
        RowIndexMap { indices }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.indices
    }
}

// â”€â”€ GroupMeta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Metadata for one group in a `GroupIndex`.
#[derive(Debug, Clone)]
pub struct GroupMeta {
    /// The rendered key strings (one per grouping column), in key order.
    pub key_values: Vec<String>,
    /// Base-frame row indices belonging to this group, in first-occurrence order.
    pub row_indices: Vec<usize>,
}

// â”€â”€ GroupIndex â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A deterministic group index built from a `TidyView`.
///
/// Groups are created in **first-occurrence order**: the first time a key
/// combination appears (scanning visible rows in ascending base-row order),
/// a new group entry is appended. This guarantees a stable, reproducible output
/// ordering regardless of hash-map iteration order.
///
/// No column buffers are copied during group construction.
#[derive(Debug, Clone)]
pub struct GroupIndex {
    /// Groups in first-occurrence order.
    pub groups: Vec<GroupMeta>,
    /// The column names used as group keys (projected names).
    pub key_names: Vec<String>,
}

impl GroupIndex {
    /// Build a `GroupIndex` from a materialized set of visible rows.
    ///
    /// `key_col_indices` are indices into `base.columns`.
    /// `visible_rows` are base-frame row indices in ascending order.
    pub fn build(
        base: &DataFrame,
        key_col_indices: &[usize],
        visible_rows: &[usize],
        key_names: Vec<String>,
    ) -> Self {
        // Use a Vec<(key_tuple, group_slot)> with sequential scan to preserve
        // first-occurrence ordering without hash nondeterminism.
        let mut group_order: Vec<Vec<String>> = Vec::new(); // unique keys, in order seen
        let mut group_map: Vec<(Vec<String>, usize)> = Vec::new(); // (key â†’ slot index)

        for &row in visible_rows {
            let key: Vec<String> = key_col_indices
                .iter()
                .map(|&ci| base.columns[ci].1.get_display(row))
                .collect();

            // Linear scan for existing key â€” preserves insertion order, no hash
            let slot = group_map
                .iter()
                .position(|(k, _)| k == &key)
                .unwrap_or_else(|| {
                    let s = group_map.len();
                    group_map.push((key.clone(), s));
                    group_order.push(key);
                    s
                });

            let _ = slot; // we'll rebuild properly below
        }

        // Build groups vector in insertion order
        let mut groups: Vec<GroupMeta> = group_order
            .iter()
            .map(|k| GroupMeta {
                key_values: k.clone(),
                row_indices: Vec::new(),
            })
            .collect();

        // Second pass: assign rows to groups
        let key_to_slot: Vec<(Vec<String>, usize)> = group_order
            .iter()
            .enumerate()
            .map(|(i, k)| (k.clone(), i))
            .collect();

        for &row in visible_rows {
            let key: Vec<String> = key_col_indices
                .iter()
                .map(|&ci| base.columns[ci].1.get_display(row))
                .collect();
            if let Some((_, slot)) = key_to_slot.iter().find(|(k, _)| k == &key) {
                groups[*slot].row_indices.push(row);
            }
        }

        GroupIndex { groups, key_names }
    }
}

// â”€â”€ GroupedTidyView â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A grouped view produced by `TidyView::group_by(...)`.
///
/// Holds the original `TidyView` (unchanged) plus a `GroupIndex` over its
/// visible rows. No column data is copied.
///
/// After grouping, `ungroup()` restores the plain `TidyView`.
/// `summarise()` collapses each group into one summary row.
#[derive(Debug, Clone)]
pub struct GroupedTidyView {
    view: TidyView,
    index: GroupIndex,
}

impl GroupedTidyView {
    /// Return the number of groups.
    pub fn ngroups(&self) -> usize {
        self.index.groups.len()
    }

    /// Discard grouping, returning the original `TidyView` unchanged.
    pub fn ungroup(self) -> TidyView {
        self.view
    }

    /// Access the group index (for testing/inspection).
    pub fn group_index(&self) -> &GroupIndex {
        &self.index
    }

    // â”€â”€ summarise â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Collapse each group to one summary row using named aggregators.
    ///
    /// `assignments` is an ordered list of `(output_name, aggregator)`.
    /// Output rows are in first-occurrence group order (deterministic).
    ///
    /// The result is a `TidyFrame` containing:
    ///   â€¢ One column per group key (in key order)
    ///   â€¢ One column per aggregator (in assignment order)
    ///
    /// Aggregator semantics (empty group):
    ///   â€¢ count â†’ 0
    ///   â€¢ sum   â†’ 0.0
    ///   â€¢ mean  â†’ f64::NAN
    ///   â€¢ min   â†’ f64::NAN
    ///   â€¢ max   â†’ f64::NAN
    ///   â€¢ first / last â†’ TidyError::EmptyGroup
    pub fn summarise(
        &self,
        assignments: &[(&str, TidyAgg)],
    ) -> Result<TidyFrame, TidyError> {
        // Validate: no duplicate output names
        {
            let mut seen = std::collections::BTreeSet::new();
            for &(name, _) in assignments {
                if !seen.insert(name) {
                    return Err(TidyError::DuplicateColumn(name.to_string()));
                }
            }
        }

        let base = &self.view.base;
        let n_groups = self.index.groups.len();

        // Build key columns first (one value per group, repeated in type-matched form)
        let mut result_columns: Vec<(String, Column)> = Vec::new();

        for key_name in &self.index.key_names {
            let base_col = base
                .get_column(key_name)
                .ok_or_else(|| TidyError::ColumnNotFound(key_name.clone()))?;

            let col = match base_col {
                Column::Int(_) => {
                    let vals: Vec<i64> = self
                        .index
                        .groups
                        .iter()
                        .map(|g| {
                            g.key_values[self
                                .index
                                .key_names
                                .iter()
                                .position(|k| k == key_name)
                                .unwrap()]
                                .parse::<i64>()
                                .unwrap_or(0)
                        })
                        .collect();
                    Column::Int(vals)
                }
                Column::Bool(_) => {
                    let vals: Vec<bool> = self
                        .index
                        .groups
                        .iter()
                        .map(|g| {
                            let s = &g.key_values[self
                                .index
                                .key_names
                                .iter()
                                .position(|k| k == key_name)
                                .unwrap()];
                            matches!(s.as_str(), "true" | "1")
                        })
                        .collect();
                    Column::Bool(vals)
                }
                _ => {
                    // Float and Str: store key as Str column for the summary
                    let vals: Vec<String> = self
                        .index
                        .groups
                        .iter()
                        .map(|g| {
                            g.key_values[self
                                .index
                                .key_names
                                .iter()
                                .position(|k| k == key_name)
                                .unwrap()]
                                .clone()
                        })
                        .collect();
                    Column::Str(vals)
                }
            };
            result_columns.push((key_name.clone(), col));
        }

        // Build aggregator columns
        for &(out_name, ref agg) in assignments {
            let col_vals = self.eval_agg_over_groups(agg, n_groups, base)?;
            result_columns.push((out_name.to_string(), col_vals));
        }

        let df = DataFrame::from_columns(result_columns)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }

    /// Evaluate an aggregator over all groups, return a typed `Column`.
    fn eval_agg_over_groups(
        &self,
        agg: &TidyAgg,
        n_groups: usize,
        base: &DataFrame,
    ) -> Result<Column, TidyError> {
        match agg {
            TidyAgg::Count => {
                let counts: Vec<i64> = self
                    .index
                    .groups
                    .iter()
                    .map(|g| g.row_indices.len() as i64)
                    .collect();
                Ok(Column::Int(counts))
            }

            TidyAgg::Sum(col_name) | TidyAgg::Mean(col_name)
            | TidyAgg::Min(col_name) | TidyAgg::Max(col_name)
            | TidyAgg::First(col_name) | TidyAgg::Last(col_name) => {
                let src = base
                    .get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;

                let mut vals = Vec::with_capacity(n_groups);
                for group in &self.index.groups {
                    let v = agg_reduce(agg, src, &group.row_indices)?;
                    vals.push(v);
                }
                Ok(Column::Float(vals))
            }
        }
    }
}

/// Reduce one group's rows for a numeric aggregator. Returns f64.
fn agg_reduce(
    agg: &TidyAgg,
    col: &Column,
    rows: &[usize],
) -> Result<f64, TidyError> {
    // Extract f64 values for the group rows
    let values: Vec<f64> = match col {
        Column::Int(v) => rows.iter().map(|&r| v[r] as f64).collect(),
        Column::Float(v) => rows.iter().map(|&r| v[r]).collect(),
        _ => {
            return Err(TidyError::TypeMismatch {
                expected: "numeric (Int or Float)".into(),
                got: col.type_name().into(),
            })
        }
    };

    match agg {
        TidyAgg::Sum(_) => Ok(kahan_sum_f64(&values)),
        TidyAgg::Mean(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                Ok(kahan_sum_f64(&values) / values.len() as f64)
            }
        }
        TidyAgg::Min(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                Ok(values.iter().cloned().fold(f64::INFINITY, |a, b| {
                    if b.is_nan() || b < a { b } else { a }
                }))
            }
        }
        TidyAgg::Max(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                Ok(values.iter().cloned().fold(f64::NEG_INFINITY, |a, b| {
                    if b.is_nan() || b > a { b } else { a }
                }))
            }
        }
        TidyAgg::First(_) => {
            if values.is_empty() {
                Err(TidyError::EmptyGroup)
            } else {
                Ok(values[0])
            }
        }
        TidyAgg::Last(_) => {
            if values.is_empty() {
                Err(TidyError::EmptyGroup)
            } else {
                Ok(*values.last().unwrap())
            }
        }
        TidyAgg::Count => Ok(values.len() as f64),
    }
}

// â”€â”€ TidyAgg â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// An aggregator expression for use in `summarise`.
#[derive(Debug, Clone)]
pub enum TidyAgg {
    /// Row count for the group. No column argument.
    Count,
    /// Kahan-sum of a numeric column.
    Sum(String),
    /// Arithmetic mean (NaN for empty groups).
    Mean(String),
    /// Minimum value (NaN for empty groups). NaN inputs sort last.
    Min(String),
    /// Maximum value (NaN for empty groups). NaN inputs sort last.
    Max(String),
    /// First row's value (error for empty groups).
    First(String),
    /// Last row's value (error for empty groups).
    Last(String),
}

// â”€â”€ ArrangeKey â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// One sorting key for `arrange`.
#[derive(Debug, Clone)]
pub struct ArrangeKey {
    /// Column name to sort by.
    pub col_name: String,
    /// `true` = descending order.
    pub descending: bool,
}

impl ArrangeKey {
    pub fn asc(col_name: &str) -> Self {
        ArrangeKey { col_name: col_name.to_string(), descending: false }
    }
    pub fn desc(col_name: &str) -> Self {
        ArrangeKey { col_name: col_name.to_string(), descending: true }
    }
}

// â”€â”€ TidyView: group_by, arrange, slice, distinct, joins â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl TidyView {

    // â”€â”€ group_by â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Group the view by one or more column names.
    ///
    /// Returns a `GroupedTidyView`. No column buffers are copied.
    /// Group order = first-occurrence order of (key_col1, key_col2, ...) tuples
    /// among the currently visible rows (ascending base-row scan).
    ///
    /// Edge cases:
    ///   â€¢ 0 rows â†’ 0 groups, no error.
    ///   â€¢ 0 keys â†’ every visible row becomes one group (equivalent to a
    ///     global aggregate).
    ///   â€¢ Unknown key column â†’ `TidyError::ColumnNotFound`.
    pub fn group_by(&self, keys: &[&str]) -> Result<GroupedTidyView, TidyError> {
        // Validate key columns exist in base
        let mut key_col_indices = Vec::with_capacity(keys.len());
        for &key in keys {
            let idx = self
                .base
                .columns
                .iter()
                .position(|(n, _)| n == key)
                .ok_or_else(|| TidyError::ColumnNotFound(key.to_string()))?;
            key_col_indices.push(idx);
        }

        let visible_rows: Vec<usize> = self.mask.iter_set().collect();
        let key_names: Vec<String> = keys.iter().map(|s| s.to_string()).collect();

        let index = GroupIndex::build(&self.base, &key_col_indices, &visible_rows, key_names);

        Ok(GroupedTidyView {
            view: self.clone(),
            index,
        })
    }

    // â”€â”€ arrange â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Sort visible rows by one or more `ArrangeKey`s.
    ///
    /// Returns a new `TidyView` backed by the same base DataFrame but with
    /// a new mask that encodes the sorted row order.
    ///
    /// Design: arrange materialises a `RowIndexMap` (sorted permutation of
    /// visible row indices), then re-encodes it into a new base DataFrame
    /// containing only those rows in the sorted order. This allows all
    /// subsequent mask-based operations to work correctly.
    ///
    /// Semantics:
    ///   â€¢ Stable sort: equal-key rows keep their original relative order.
    ///   â€¢ NaN sorting: NaN values sort LAST (greater than any finite value).
    ///   â€¢ Multi-key: sort by key[0] first, then key[1], ... (left-to-right).
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ Non-numeric sort of Float col: allowed (NaN last).
    ///   â€¢ Mixed-type sort across columns is column-by-column (each col has one type).
    pub fn arrange(&self, keys: &[ArrangeKey]) -> Result<TidyView, TidyError> {
        // Validate all sort key columns exist in base
        for key in keys {
            if self.base.get_column(&key.col_name).is_none() {
                return Err(TidyError::ColumnNotFound(key.col_name.clone()));
            }
        }

        // Collect visible row indices in current mask order
        let mut row_indices: Vec<usize> = self.mask.iter_set().collect();

        // Stable sort by keys left-to-right
        row_indices.sort_by(|&a, &b| {
            for key in keys {
                let col = self.base.get_column(&key.col_name).unwrap();
                let ord = compare_column_rows(col, a, b);
                let ord = if key.descending { ord.reverse() } else { ord };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });

        // Re-materialise into a new DataFrame (sorted), wrap as a fresh TidyView
        let mut new_columns = Vec::with_capacity(self.proj.len());
        for &ci in self.proj.indices() {
            let (name, col) = &self.base.columns[ci];
            let new_col = gather_column(col, &row_indices);
            new_columns.push((name.clone(), new_col));
        }
        // Also include any non-projected base columns needed for future ops
        // Strategy: build new base from ALL original columns in sorted order
        let mut sorted_all_cols = Vec::with_capacity(self.base.ncols());
        for (name, col) in &self.base.columns {
            sorted_all_cols.push((name.clone(), gather_column(col, &row_indices)));
        }

        let new_base = DataFrame::from_columns(sorted_all_cols)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        let nrows = new_base.nrows();
        let new_proj = self.proj.clone();

        Ok(TidyView {
            base: Rc::new(new_base),
            mask: BitMask::all_true(nrows),
            proj: new_proj,
        })
    }

    // â”€â”€ slice â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Select rows by a half-open range `[start, end)` of visible-row positions.
    ///
    /// Positions are relative to the current visible rows (0-based).
    /// Out-of-bounds: clamped to `[0, nrows]`.
    pub fn slice(&self, start: usize, end: usize) -> TidyView {
        let visible: Vec<usize> = self.mask.iter_set().collect();
        let n = visible.len();
        let s = start.min(n);
        let e = end.min(n);
        let selected = if s >= e { vec![] } else { visible[s..e].to_vec() };
        self.view_from_row_indices(selected)
    }

    /// Select the first `n` visible rows (clamped to nrows).
    pub fn slice_head(&self, n: usize) -> TidyView {
        self.slice(0, n)
    }

    /// Select the last `n` visible rows (clamped to nrows).
    pub fn slice_tail(&self, n: usize) -> TidyView {
        let total = self.mask.count_ones();
        let start = total.saturating_sub(n);
        self.slice(start, total)
    }

    /// Deterministic random sample of `n` visible rows using an LCG with `seed`.
    ///
    /// If `n >= nrows`, returns all visible rows in their original order (no error).
    /// Sampling uses a Knuth shuffle variant seeded by `seed` (deterministic LCG).
    pub fn slice_sample(&self, n: usize, seed: u64) -> TidyView {
        let mut visible: Vec<usize> = self.mask.iter_set().collect();
        let total = visible.len();
        if n >= total {
            return self.view_from_row_indices(visible);
        }
        // Partial Fisher-Yates using LCG: deterministic with fixed seed
        let mut rng = seed;
        let selected_count = n;
        for i in 0..selected_count {
            // LCG step: multiplier and increment from Knuth
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = i + (rng as usize % (total - i));
            visible.swap(i, j);
        }
        visible.truncate(selected_count);
        // Sort selected indices to restore ascending order (stable/deterministic)
        visible.sort_unstable();
        self.view_from_row_indices(visible)
    }

    // â”€â”€ distinct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Return rows with unique combinations of the specified columns.
    ///
    /// Output ordering: first-occurrence order (the first row with each distinct
    /// key combination is kept).
    ///
    /// Edge cases:
    ///   â€¢ 0 key columns â†’ keeps first row only (all rows equal on zero keys).
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ After projection/mask: only visible columns/rows are considered.
    pub fn distinct(&self, cols: &[&str]) -> Result<TidyView, TidyError> {
        // Validate columns exist in base
        let mut col_indices = Vec::with_capacity(cols.len());
        for &name in cols {
            let idx = self
                .base
                .columns
                .iter()
                .position(|(n, _)| n == name)
                .ok_or_else(|| TidyError::ColumnNotFound(name.to_string()))?;
            col_indices.push(idx);
        }

        let mut seen_keys: Vec<Vec<String>> = Vec::new();
        let mut selected_rows: Vec<usize> = Vec::new();

        for row in self.mask.iter_set() {
            let key: Vec<String> = if col_indices.is_empty() {
                vec!["__all__".into()]
            } else {
                col_indices
                    .iter()
                    .map(|&ci| self.base.columns[ci].1.get_display(row))
                    .collect()
            };

            if !seen_keys.contains(&key) {
                seen_keys.push(key);
                selected_rows.push(row);
            }
        }

        Ok(self.view_from_row_indices(selected_rows))
    }

    // â”€â”€ joins â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Inner join: rows where all `on` key columns match.
    ///
    /// Output: left columns then right columns (excluding duplicate key cols).
    /// Row order: left outer loop (preserves left order), right inner ascending.
    /// Produces a materialized `TidyFrame` (joins always materialize).
    ///
    /// Edge cases:
    ///   â€¢ Unknown join key â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ `on` empty â†’ cross join semantics (every left Ã— every right).
    ///   â€¢ Duplicate keys on left or right â†’ all matching pairs included.
    pub fn inner_join(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
    ) -> Result<TidyFrame, TidyError> {
        let (left_rows, right_rows) = join_match_rows(self, right, on, JoinKind::Inner)?;
        build_join_frame(self, right, &left_rows, &right_rows, on, false)
    }

    /// Left join: all left rows; matched right rows or nulls (0/0.0/""/false).
    ///
    /// Row order: left outer loop order preserved, right matches ascending.
    pub fn left_join(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
    ) -> Result<TidyFrame, TidyError> {
        let (left_rows, right_rows_opt) =
            join_match_rows_optional(self, right, on, JoinKind::Left)?;
        build_left_join_frame(self, right, &left_rows, &right_rows_opt, on)
    }

    /// Semi-join: rows in `self` that have at least one match in `right`.
    ///
    /// Returns a `TidyView` (no right columns). Row order: stable left order.
    pub fn semi_join(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
    ) -> Result<TidyView, TidyError> {
        let included = semi_anti_match_rows(self, right, on, /*semi=*/ true)?;
        Ok(self.view_from_row_indices(included))
    }

    /// Anti-join: rows in `self` that have NO match in `right`.
    ///
    /// Returns a `TidyView` (no right columns). Row order: stable left order.
    pub fn anti_join(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
    ) -> Result<TidyView, TidyError> {
        let included = semi_anti_match_rows(self, right, on, /*semi=*/ false)?;
        Ok(self.view_from_row_indices(included))
    }

    // â”€â”€ internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Build a new `TidyView` over the same base using explicit row indices.
    /// The row indices must be valid base-frame indices.
    fn view_from_row_indices(&self, row_indices: Vec<usize>) -> TidyView {
        let nrows_base = self.base.nrows();
        let mut words = vec![0u64; nwords_for(nrows_base)];
        for &r in &row_indices {
            words[r / 64] |= 1u64 << (r % 64);
        }
        TidyView {
            base: Rc::clone(&self.base),
            mask: BitMask { words, nrows: nrows_base },
            proj: self.proj.clone(),
        }
    }
}

// â”€â”€ Join internals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[derive(Clone, Copy)]
enum JoinKind { Inner, Left }

/// Resolve join key columns from both sides. Returns (left_indices, right_indices).
fn resolve_join_keys(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
) -> Result<(Vec<usize>, Vec<usize>), TidyError> {
    let mut li = Vec::new();
    let mut ri = Vec::new();
    for &(lk, rk) in on {
        let l = left.base.columns.iter().position(|(n, _)| n == lk)
            .ok_or_else(|| TidyError::ColumnNotFound(lk.to_string()))?;
        let r = right.base.columns.iter().position(|(n, _)| n == rk)
            .ok_or_else(|| TidyError::ColumnNotFound(rk.to_string()))?;
        li.push(l);
        ri.push(r);
    }
    Ok((li, ri))
}

/// Get a join key tuple for a row (as Vec<String> for deterministic comparison).
fn row_key(base: &DataFrame, col_indices: &[usize], row: usize) -> Vec<String> {
    col_indices
        .iter()
        .map(|&ci| base.columns[ci].1.get_display(row))
        .collect()
}

/// Build a deterministic right-side lookup: sorted Vec of (key_tuple, right_row_idx).
/// Sorted by key tuple first, then by row index â€” guarantees determinism.
fn build_right_lookup(
    right: &TidyView,
    right_key_cols: &[usize],
) -> Vec<(Vec<String>, usize)> {
    let mut lookup: Vec<(Vec<String>, usize)> = right
        .mask
        .iter_set()
        .map(|r| (row_key(&right.base, right_key_cols, r), r))
        .collect();
    // Sort by key then by row index for deterministic ordering
    lookup.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    lookup
}

/// Find matching right rows for a given left key (from sorted lookup).
fn find_matches(lookup: &[(Vec<String>, usize)], key: &[String]) -> Vec<usize> {
    // Binary search for first match, then collect contiguous matches
    let key_vec = key.to_vec();
    let start = lookup.partition_point(|(k, _)| k.as_slice() < key_vec.as_slice());
    let mut matches = Vec::new();
    for (k, r) in &lookup[start..] {
        if k == &key_vec {
            matches.push(*r);
        } else {
            break;
        }
    }
    matches
}

/// Inner join: collect (left_row, right_row) pairs.
fn join_match_rows(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
    _kind: JoinKind,
) -> Result<(Vec<usize>, Vec<usize>), TidyError> {
    let (left_key_cols, right_key_cols) = resolve_join_keys(left, right, on)?;
    let lookup = build_right_lookup(right, &right_key_cols);

    let mut out_left = Vec::new();
    let mut out_right = Vec::new();

    for l_row in left.mask.iter_set() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        let matches = find_matches(&lookup, &key);
        for r_row in matches {
            out_left.push(l_row);
            out_right.push(r_row);
        }
    }
    Ok((out_left, out_right))
}

/// Left join: collect (left_row, Option<right_row>) pairs.
fn join_match_rows_optional(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
    _kind: JoinKind,
) -> Result<(Vec<usize>, Vec<Option<usize>>), TidyError> {
    let (left_key_cols, right_key_cols) = resolve_join_keys(left, right, on)?;
    let lookup = build_right_lookup(right, &right_key_cols);

    let mut out_left = Vec::new();
    let mut out_right: Vec<Option<usize>> = Vec::new();

    for l_row in left.mask.iter_set() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        let matches = find_matches(&lookup, &key);
        if matches.is_empty() {
            out_left.push(l_row);
            out_right.push(None);
        } else {
            for r_row in matches {
                out_left.push(l_row);
                out_right.push(Some(r_row));
            }
        }
    }
    Ok((out_left, out_right))
}

/// Semi/anti join: return left row indices (no right columns).
fn semi_anti_match_rows(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
    semi: bool,
) -> Result<Vec<usize>, TidyError> {
    let (left_key_cols, right_key_cols) = resolve_join_keys(left, right, on)?;
    let lookup = build_right_lookup(right, &right_key_cols);

    let mut out = Vec::new();
    for l_row in left.mask.iter_set() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        let has_match = !find_matches(&lookup, &key).is_empty();
        if has_match == semi {
            out.push(l_row);
        }
    }
    Ok(out)
}

/// Build an inner-join result `TidyFrame`.
/// Output cols: all left projected cols, then right projected cols (excluding join-key cols).
fn build_join_frame(
    left: &TidyView,
    right: &TidyView,
    left_rows: &[usize],
    right_rows: &[usize],
    on: &[(&str, &str)],
    _include_unmatched: bool,
) -> Result<TidyFrame, TidyError> {
    let right_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(_, rk)| *rk).collect();

    let n = left_rows.len();
    let mut columns: Vec<(String, Column)> = Vec::new();

    // Left projected columns
    for &ci in left.proj.indices() {
        let (name, col) = &left.base.columns[ci];
        columns.push((name.clone(), gather_column(col, left_rows)));
    }

    // Right projected columns (skip join keys to avoid duplication)
    for &ci in right.proj.indices() {
        let (name, col) = &right.base.columns[ci];
        if right_key_names.contains(name.as_str()) {
            continue;
        }
        columns.push((name.clone(), gather_column(col, right_rows)));
    }

    assert_eq!(n, left_rows.len());
    let df = DataFrame::from_columns(columns)
        .map_err(|e| TidyError::Internal(e.to_string()))?;
    Ok(TidyFrame::from_df(df))
}

/// Build a left-join result `TidyFrame` (right side may have None = unmatched).
fn build_left_join_frame(
    left: &TidyView,
    right: &TidyView,
    left_rows: &[usize],
    right_rows_opt: &[Option<usize>],
    on: &[(&str, &str)],
) -> Result<TidyFrame, TidyError> {
    let right_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(_, rk)| *rk).collect();

    let mut columns: Vec<(String, Column)> = Vec::new();

    // Left projected columns
    for &ci in left.proj.indices() {
        let (name, col) = &left.base.columns[ci];
        columns.push((name.clone(), gather_column(col, left_rows)));
    }

    // Right projected columns (None rows get null-equivalents)
    for &ci in right.proj.indices() {
        let (name, col) = &right.base.columns[ci];
        if right_key_names.contains(name.as_str()) {
            continue;
        }
        let new_col = gather_column_nullable(col, right_rows_opt);
        columns.push((name.clone(), new_col));
    }

    let df = DataFrame::from_columns(columns)
        .map_err(|e| TidyError::Internal(e.to_string()))?;
    Ok(TidyFrame::from_df(df))
}

// â”€â”€ Column comparison for arrange â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compare two rows of a `Column` for use in `arrange`.
///
/// NaN rules (for Float): NaN sorts LAST (treated as greater than any finite).
/// Tie-breaking: returns Equal (caller's stable sort handles relative order).
fn compare_column_rows(col: &Column, a: usize, b: usize) -> std::cmp::Ordering {
    match col {
        Column::Int(v) => v[a].cmp(&v[b]),
        Column::Float(v) => {
            let va = v[a];
            let vb = v[b];
            match (va.is_nan(), vb.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // NaN last
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal),
            }
        }
        Column::Bool(v) => v[a].cmp(&v[b]),
        Column::Str(v) => v[a].cmp(&v[b]),
        Column::Categorical { levels, codes } => {
            // Compare by the level string, not the code
            levels[codes[a] as usize].cmp(&levels[codes[b] as usize])
        }
        Column::DateTime(v) => v[a].cmp(&v[b]),
    }
}

// (TidyError::EmptyGroup is defined in the TidyError enum above.)

// â”€â”€ NoGC safe-builtin registrations (Phase 11â€“12) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Ops that only update metadata (no column buffer alloc):
//   tidy_group_by      : builds GroupIndex (Vec of Vec<usize>) â€” no column alloc
//   tidy_ungroup       : drops GroupIndex â€” no alloc
//   tidy_arrange       : materialises sorted base (ALLOCATES) â†’ NOT @nogc safe
//   tidy_slice         : updates RowIndexMap â€” O(N) usize alloc, safe
//   tidy_distinct      : builds RowIndexMap â€” O(N) usize alloc, safe
//   tidy_semi_join     : builds RowIndexMap â€” O(N) usize alloc, safe
//   tidy_anti_join     : builds RowIndexMap â€” O(N) usize alloc, safe
//   tidy_inner_join    : materialises result â€” ALLOCATES â†’ NOT @nogc safe
//   tidy_left_join     : materialises result â€” ALLOCATES â†’ NOT @nogc safe
//   tidy_summarise     : materialises result â€” ALLOCATES â†’ NOT @nogc safe
//
// Safe (registered in nogc_verify.rs):
//   tidy_group_by, tidy_ungroup, tidy_slice, tidy_distinct,
//   tidy_semi_join, tidy_anti_join, tidy_ngroups

#[cfg(test)]
mod phase10_unit_tests {
    use super::*;

    fn make_df() -> DataFrame {
        DataFrame::from_columns(vec![
            ("x".into(), Column::Int(vec![1, 2, 3, 4, 5])),
            ("y".into(), Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ("flag".into(), Column::Bool(vec![true, false, true, false, true])),
        ])
        .unwrap()
    }

    #[test]
    fn bitmask_all_true_count() {
        let m = BitMask::all_true(5);
        assert_eq!(m.count_ones(), 5);
    }

    #[test]
    fn bitmask_all_false_count() {
        let m = BitMask::all_false(5);
        assert_eq!(m.count_ones(), 0);
    }

    #[test]
    fn bitmask_tail_bits_clean() {
        // 65 rows â€” two words; tail must not bleed into unset bits
        let m = BitMask::all_true(65);
        assert_eq!(m.count_ones(), 65);
        assert_eq!(m.words.len(), 2);
        assert_eq!(m.words[1], 1u64); // only bit 0 of second word set
    }

    #[test]
    fn bitmask_and_semantics() {
        let a = BitMask::from_bools(&[true, true, false, true]);
        let b = BitMask::from_bools(&[true, false, true, true]);
        let c = a.and(&b);
        assert!(c.get(0));
        assert!(!c.get(1));
        assert!(!c.get(2));
        assert!(c.get(3));
    }

    #[test]
    fn tidy_view_nrows_ncols() {
        let df = make_df();
        let v = df.tidy();
        assert_eq!(v.nrows(), 5);
        assert_eq!(v.ncols(), 3);
    }

    #[test]
    fn filter_basic() {
        let df = make_df();
        let v = df.tidy();
        let filtered = v
            .filter(&DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::LitInt(2)),
            })
            .unwrap();
        assert_eq!(filtered.nrows(), 3);
    }

    #[test]
    fn filter_empty_df() {
        let df = DataFrame::from_columns(vec![
            ("x".into(), Column::Int(vec![])),
        ])
        .unwrap();
        let v = df.tidy();
        let filtered = v
            .filter(&DExpr::BinOp {
                op: DBinOp::Gt,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::LitInt(0)),
            })
            .unwrap();
        assert_eq!(filtered.nrows(), 0);
    }

    #[test]
    fn select_reorder() {
        let df = make_df();
        let v = df.tidy();
        let s = v.select(&["y", "x"]).unwrap();
        assert_eq!(s.column_names(), vec!["y", "x"]);
    }

    #[test]
    fn select_zero_cols() {
        let df = make_df();
        let v = df.tidy();
        let s = v.select(&[]).unwrap();
        assert_eq!(s.ncols(), 0);
        assert_eq!(s.nrows(), 5);
    }

    #[test]
    fn select_unknown_col() {
        let df = make_df();
        let v = df.tidy();
        let err = v.select(&["nonexistent"]).unwrap_err();
        assert!(matches!(err, TidyError::ColumnNotFound(_)));
    }

    #[test]
    fn select_duplicate_col() {
        let df = make_df();
        let v = df.tidy();
        let err = v.select(&["x", "x"]).unwrap_err();
        assert!(matches!(err, TidyError::DuplicateColumn(_)));
    }

    #[test]
    fn mutate_new_col() {
        let df = make_df();
        let v = df.tidy();
        let frame = v
            .mutate(&[("z", DExpr::BinOp {
                op: DBinOp::Add,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::LitInt(10)),
            })])
            .unwrap();
        let b = frame.borrow();
        let z = b.get_column("z").unwrap();
        assert_eq!(z.len(), 5);
        if let Column::Int(v) = z {
            assert_eq!(v[0], 11);
            assert_eq!(v[4], 15);
        } else {
            panic!("expected Int column");
        }
    }

    #[test]
    fn mutate_type_promotion() {
        let df = make_df();
        let v = df.tidy();
        // x (Int) + y (Float) â†’ Float column
        let frame = v
            .mutate(&[("promoted", DExpr::BinOp {
                op: DBinOp::Add,
                left: Box::new(DExpr::Col("x".into())),
                right: Box::new(DExpr::Col("y".into())),
            })])
            .unwrap();
        let b = frame.borrow();
        let col = b.get_column("promoted").unwrap();
        assert!(matches!(col, Column::Float(_)));
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Phase 13â€“16: Tidy Completion
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//
// Spec-Lock Table (invariants that tests must not regress):
//
// | Property                      | Rule |
// |-------------------------------|------|
// | pivot_longer row order        | Original row order preserved; within each
// |                               | original row, columns appear in the order
// |                               | supplied in `value_cols`. |
// | pivot_longer col order        | id_cols first, then "name" col, then
// |                               | "value" col. |
// | pivot_wider col order         | id_cols first, then each unique key value
// |                               | in first-occurrence order from the source
// |                               | column. |
// | pivot_wider duplicate keys    | Strict error: TidyError::DuplicateKey. |
// | pivot_wider null key          | Null key string treated as literal "null". |
// | pivot_longer zero value_cols  | TidyError::EmptySelection. |
// | pivot_longer mixed types      | Strict: all value_cols must have same type;
// |                               | TidyError::TypeMismatch otherwise. |
// | NullCol semantics             | is_null(x) is always well-defined. |
// | NullCol in group_by           | Null key forms its own group (treated as
// |                               | equal to other nulls). |
// | NullCol in join               | Null key does NOT match null key (SQL NULL
// |                               | semantics). |
// | NullCol in aggregation        | Nulls skipped; all-null â†’ null result. |
// | NullCol comparison            | NULL op x â†’ null result (three-valued). |
// | rename collision              | TidyError::DuplicateColumn. |
// | rename unknown col            | TidyError::ColumnNotFound. |
// | relocate ordering             | Stable; relative order of non-moved cols
// |                               | preserved. |
// | select(-col) drop             | TidyError::ColumnNotFound for unknown. |
// | bind_rows schema mismatch     | Strict: TidyError::SchemaMismatch. |
// | bind_rows col order           | Left frame column order. |
// | bind_rows row order           | Left rows then right rows. |
// | bind_cols row mismatch        | TidyError::LengthMismatch. |
// | bind_cols name collision      | TidyError::DuplicateColumn. |
// | across expansion order        | Stable column iteration (projection order).
// | across generated names        | "{col}_{fn_name}" or user template. |
// | across name collision         | TidyError::DuplicateColumn. |
// | join type validation          | Comparable types only (Intâ†”Int, Floatâ†”Float,
// |                               | Strâ†”Str, Boolâ†”Bool, Intâ†”Float widened).
// |                               | TidyError::TypeMismatch otherwise. |
// | join suffix handling          | Default ".x"/".y"; user may override. |
// | right_join / full_join        | Defined; row order: see semantics section. |
// | group perf upgrade            | First-occurrence order preserved; identical
// |                               | output to Phase 11 implementation. |

// â”€â”€ New TidyError variants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl TidyError {
    /// Construct a schema mismatch error.
    pub fn schema_mismatch(msg: impl Into<String>) -> Self {
        TidyError::Internal(format!("schema mismatch: {}", msg.into()))
    }
    /// Construct a type mismatch for join validation.
    pub fn join_type_mismatch(col: &str, lt: &str, rt: &str) -> Self {
        TidyError::TypeMismatch {
            expected: format!("{} (from left key `{}`)", lt, col),
            got: rt.to_string(),
        }
    }
    /// Duplicate join/pivot key error.
    pub fn duplicate_key(key: impl Into<String>) -> Self {
        TidyError::DuplicateColumn(format!("duplicate key: {}", key.into()))
    }
    /// Empty selection (e.g. pivot_longer with zero value_cols).
    pub fn empty_selection(msg: impl Into<String>) -> Self {
        TidyError::Internal(format!("empty selection: {}", msg.into()))
    }
}

// â”€â”€ Nullable column layer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A nullable column: values buffer + validity bitmap.
///
/// The validity bitmap uses the same bit layout as `BitMask` (LSB-first,
/// tail bits zeroed). `validity.get(i) == true` means `values[i]` is valid
/// (not null). When `validity.get(i) == false`, `values[i]` holds a
/// type-appropriate zero/empty value but MUST NOT be read as valid data.
#[derive(Debug, Clone)]
pub struct NullableColumn<T: Clone> {
    pub values: Vec<T>,
    pub validity: BitMask,
}

impl<T: Clone + Default> NullableColumn<T> {
    /// Create a fully valid (non-null) nullable column from a slice.
    pub fn from_values(values: Vec<T>) -> Self {
        let n = values.len();
        Self {
            values,
            validity: BitMask::all_true(n),
        }
    }

    /// Create a nullable column with explicit validity.
    /// Panics if `values.len() != validity.nrows()`.
    pub fn new(values: Vec<T>, validity: BitMask) -> Self {
        assert_eq!(values.len(), validity.nrows(), "NullableColumn: length mismatch");
        Self { values, validity }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Is the value at row `i` null?
    pub fn is_null(&self, i: usize) -> bool {
        !self.validity.get(i)
    }

    /// Get the value at `i` if non-null.
    pub fn get(&self, i: usize) -> Option<&T> {
        if self.validity.get(i) { Some(&self.values[i]) } else { None }
    }

    /// Count non-null rows.
    pub fn count_valid(&self) -> usize {
        self.validity.count_ones()
    }

    /// Gather rows by index (for join/arrange materialisation).
    pub fn gather(&self, indices: &[usize]) -> Self {
        let mut vals = Vec::with_capacity(indices.len());
        let mut bools = Vec::with_capacity(indices.len());
        for &i in indices {
            vals.push(self.values[i].clone());
            bools.push(self.validity.get(i));
        }
        let validity = BitMask::from_bools(&bools);
        Self { values: vals, validity }
    }
}

/// A typed nullable column variant stored in a DataFrame column slot.
///
/// Phase 13-16 does not replace `Column` (which has no nulls) with `NullCol`
/// everywhere â€” that would be a breaking change across the whole codebase.
/// Instead, `NullCol` is used as the result type of operations that can
/// introduce nulls (left_join fills, pivot_wider missing combinations,
/// bind_rows on mismatched schemas, full_join unmatched rows).
///
/// Conversion: `NullCol::from_column(col)` wraps an existing `Column` as
/// fully-valid nullable. `NullCol::to_column(nc)` unwraps if fully valid,
/// else returns `TidyError::Internal` (caller must handle null columns
/// explicitly).
#[derive(Debug, Clone)]
pub enum NullCol {
    Int(NullableColumn<i64>),
    Float(NullableColumn<f64>),
    Str(NullableColumn<String>),
    Bool(NullableColumn<bool>),
}

impl NullCol {
    pub fn len(&self) -> usize {
        match self {
            NullCol::Int(c) => c.len(),
            NullCol::Float(c) => c.len(),
            NullCol::Str(c) => c.len(),
            NullCol::Bool(c) => c.len(),
        }
    }

    pub fn is_null(&self, i: usize) -> bool {
        match self {
            NullCol::Int(c) => c.is_null(i),
            NullCol::Float(c) => c.is_null(i),
            NullCol::Str(c) => c.is_null(i),
            NullCol::Bool(c) => c.is_null(i),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            NullCol::Int(_) => "Int",
            NullCol::Float(_) => "Float",
            NullCol::Str(_) => "Str",
            NullCol::Bool(_) => "Bool",
        }
    }

    /// Wrap a non-nullable `Column` as fully valid.
    pub fn from_column(col: &Column) -> Self {
        match col {
            Column::Int(v) => NullCol::Int(NullableColumn::from_values(v.clone())),
            Column::Float(v) => NullCol::Float(NullableColumn::from_values(v.clone())),
            Column::Str(v) => NullCol::Str(NullableColumn::from_values(v.clone())),
            Column::Bool(v) => NullCol::Bool(NullableColumn::from_values(v.clone())),
            // Categorical is stored as its string representation for nullable contexts
            Column::Categorical { levels, codes } => {
                let strings: Vec<String> = codes.iter().map(|&c| levels[c as usize].clone()).collect();
                NullCol::Str(NullableColumn::from_values(strings))
            }
            Column::DateTime(v) => NullCol::Int(NullableColumn::from_values(v.clone())),
        }
    }

    /// Unwrap to non-nullable `Column` only if all rows are valid (not null).
    /// If any null exists, returns `Err(TidyError::Internal)`.
    pub fn to_column_strict(&self) -> Result<Column, TidyError> {
        match self {
            NullCol::Int(nc) => {
                if nc.count_valid() == nc.len() {
                    Ok(Column::Int(nc.values.clone()))
                } else {
                    Err(TidyError::Internal("null values in non-nullable context".into()))
                }
            }
            NullCol::Float(nc) => {
                if nc.count_valid() == nc.len() {
                    Ok(Column::Float(nc.values.clone()))
                } else {
                    Err(TidyError::Internal("null values in non-nullable context".into()))
                }
            }
            NullCol::Str(nc) => {
                if nc.count_valid() == nc.len() {
                    Ok(Column::Str(nc.values.clone()))
                } else {
                    Err(TidyError::Internal("null values in non-nullable context".into()))
                }
            }
            NullCol::Bool(nc) => {
                if nc.count_valid() == nc.len() {
                    Ok(Column::Bool(nc.values.clone()))
                } else {
                    Err(TidyError::Internal("null values in non-nullable context".into()))
                }
            }
        }
    }

    /// Convert to `Column`, filling nulls with type-appropriate zero-value.
    /// Null Int â†’ 0, Null Float â†’ NaN, Null Str â†’ "", Null Bool â†’ false.
    pub fn to_column_filled(&self) -> Column {
        match self {
            NullCol::Int(nc) => Column::Int(nc.values.clone()),
            NullCol::Float(nc) => {
                let v: Vec<f64> = (0..nc.len())
                    .map(|i| if nc.is_null(i) { f64::NAN } else { nc.values[i] })
                    .collect();
                Column::Float(v)
            }
            NullCol::Str(nc) => Column::Str(nc.values.clone()),
            NullCol::Bool(nc) => Column::Bool(nc.values.clone()),
        }
    }

    /// Get display string for a row (null â†’ "null").
    pub fn get_display(&self, i: usize) -> String {
        if self.is_null(i) {
            return "null".to_string();
        }
        match self {
            NullCol::Int(nc) => format!("{}", nc.values[i]),
            NullCol::Float(nc) => format!("{}", nc.values[i]),
            NullCol::Str(nc) => nc.values[i].clone(),
            NullCol::Bool(nc) => format!("{}", nc.values[i]),
        }
    }

    /// Create a null-fill column of given type and length.
    pub fn null_of_type(type_name: &str, len: usize) -> Self {
        match type_name {
            "Int" => NullCol::Int(NullableColumn {
                values: vec![0i64; len],
                validity: BitMask::all_false(len),
            }),
            "Float" => NullCol::Float(NullableColumn {
                values: vec![0.0f64; len],
                validity: BitMask::all_false(len),
            }),
            "Bool" => NullCol::Bool(NullableColumn {
                values: vec![false; len],
                validity: BitMask::all_false(len),
            }),
            _ => NullCol::Str(NullableColumn {
                values: vec![String::new(); len],
                validity: BitMask::all_false(len),
            }),
        }
    }

    /// Gather rows by index.
    pub fn gather(&self, indices: &[usize]) -> Self {
        match self {
            NullCol::Int(nc) => NullCol::Int(nc.gather(indices)),
            NullCol::Float(nc) => NullCol::Float(nc.gather(indices)),
            NullCol::Str(nc) => NullCol::Str(nc.gather(indices)),
            NullCol::Bool(nc) => NullCol::Bool(nc.gather(indices)),
        }
    }
}

/// A DataFrame-like frame that can hold nullable columns.
/// Used as output of joins, pivots, and bind operations that may introduce nulls.
#[derive(Debug, Clone)]
pub struct NullableFrame {
    pub columns: Vec<(String, NullCol)>,
}

impl NullableFrame {
    pub fn new() -> Self {
        Self { columns: Vec::new() }
    }

    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn get_column(&self, name: &str) -> Option<&NullCol> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// Convert to a regular `DataFrame`, filling nulls with type-appropriate values.
    pub fn to_dataframe_filled(&self) -> DataFrame {
        let cols: Vec<(String, Column)> = self.columns.iter()
            .map(|(n, c)| (n.clone(), c.to_column_filled()))
            .collect();
        // Safety: all columns should have same length if built correctly
        DataFrame { columns: cols }
    }

    /// Convert to `TidyFrame` (filled), discarding null metadata.
    pub fn to_tidy_frame_filled(&self) -> TidyFrame {
        TidyFrame::from_df(self.to_dataframe_filled())
    }

    /// Convert to `TidyView` (filled), discarding null metadata.
    pub fn to_tidy_view_filled(&self) -> TidyView {
        TidyView::from_df(self.to_dataframe_filled())
    }
}

impl Default for NullableFrame {
    fn default() -> Self { Self::new() }
}

// â”€â”€ Helpers for nullable-aware gather â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Gather column rows with optional indices (None â†’ null).
/// Used in left/right/full join output where some rows have no match.
fn gather_column_nullable_null(col: &Column, indices: &[Option<usize>]) -> NullCol {
    match col {
        Column::Int(v) => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());
            for &idx in indices {
                match idx {
                    Some(i) => { vals.push(v[i]); valid.push(true); }
                    None => { vals.push(0); valid.push(false); }
                }
            }
            NullCol::Int(NullableColumn::new(vals, BitMask::from_bools(&valid)))
        }
        Column::Float(v) => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());
            for &idx in indices {
                match idx {
                    Some(i) => { vals.push(v[i]); valid.push(true); }
                    None => { vals.push(0.0); valid.push(false); }
                }
            }
            NullCol::Float(NullableColumn::new(vals, BitMask::from_bools(&valid)))
        }
        Column::Str(v) => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());
            for &idx in indices {
                match idx {
                    Some(i) => { vals.push(v[i].clone()); valid.push(true); }
                    None => { vals.push(String::new()); valid.push(false); }
                }
            }
            NullCol::Str(NullableColumn::new(vals, BitMask::from_bools(&valid)))
        }
        Column::Bool(v) => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());
            for &idx in indices {
                match idx {
                    Some(i) => { vals.push(v[i]); valid.push(true); }
                    None => { vals.push(false); valid.push(false); }
                }
            }
            NullCol::Bool(NullableColumn::new(vals, BitMask::from_bools(&valid)))
        }
        Column::Categorical { levels, codes } => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());
            for &idx in indices {
                match idx {
                    Some(i) => { vals.push(levels[codes[i] as usize].clone()); valid.push(true); }
                    None => { vals.push(String::new()); valid.push(false); }
                }
            }
            NullCol::Str(NullableColumn::new(vals, BitMask::from_bools(&valid)))
        }
        Column::DateTime(v) => {
            let mut vals = Vec::with_capacity(indices.len());
            let mut valid = Vec::with_capacity(indices.len());
            for &idx in indices {
                match idx {
                    Some(i) => { vals.push(v[i]); valid.push(true); }
                    None => { vals.push(0); valid.push(false); }
                }
            }
            NullCol::Int(NullableColumn::new(vals, BitMask::from_bools(&valid)))
        }
    }
}

// â”€â”€ Across support types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A scalar transformation function for `across()`.
///
/// Takes a column reference name (for error messages) and a `Column`, returns
/// a new `Column` of the same length or `TidyError`.
pub type AcrossFn = Box<dyn Fn(&str, &Column) -> Result<Column, TidyError>>;

/// A named across transformation.
pub struct AcrossTransform {
    /// Function name (used in generated column name `{col}_{fn_name}`).
    pub fn_name: String,
    /// The transformation to apply.
    pub func: AcrossFn,
}

impl AcrossTransform {
    pub fn new(fn_name: impl Into<String>, func: impl Fn(&str, &Column) -> Result<Column, TidyError> + 'static) -> Self {
        Self {
            fn_name: fn_name.into(),
            func: Box::new(func),
        }
    }
}

/// An across() specification: select columns and apply one function.
pub struct AcrossSpec {
    /// Columns to transform (by name).
    pub cols: Vec<String>,
    /// Transform to apply.
    pub transform: AcrossTransform,
    /// Optional output name template. None â†’ "{col}_{fn}".
    /// Use `{col}` and `{fn}` as placeholders.
    pub name_template: Option<String>,
}

impl AcrossSpec {
    pub fn new(cols: impl IntoIterator<Item = impl Into<String>>, transform: AcrossTransform) -> Self {
        Self {
            cols: cols.into_iter().map(|c| c.into()).collect(),
            transform,
            name_template: None,
        }
    }

    pub fn with_template(mut self, tmpl: impl Into<String>) -> Self {
        self.name_template = Some(tmpl.into());
        self
    }

    /// Generate the output column name for a given input column.
    pub fn output_name(&self, col_name: &str) -> String {
        match &self.name_template {
            Some(tmpl) => tmpl
                .replace("{col}", col_name)
                .replace("{fn}", &self.transform.fn_name),
            None => format!("{}_{}", col_name, self.transform.fn_name),
        }
    }
}

// â”€â”€ Join maturity types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Join suffix options for handling column name collisions in inner/left/right/full joins.
#[derive(Debug, Clone)]
pub struct JoinSuffix {
    pub left: String,
    pub right: String,
}

impl Default for JoinSuffix {
    fn default() -> Self {
        Self { left: ".x".into(), right: ".y".into() }
    }
}

impl JoinSuffix {
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self { left: left.into(), right: right.into() }
    }
}

// â”€â”€ Column type comparison for join validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Check whether two Column types are join-compatible.
/// Int and Float are mutually compatible (numeric widening).
fn join_types_compatible(left: &Column, right: &Column) -> bool {
    match (left, right) {
        (Column::Int(_), Column::Int(_)) => true,
        (Column::Float(_), Column::Float(_)) => true,
        (Column::Int(_), Column::Float(_)) => true,
        (Column::Float(_), Column::Int(_)) => true,
        (Column::Str(_), Column::Str(_)) => true,
        (Column::Bool(_), Column::Bool(_)) => true,
        _ => false,
    }
}

// â”€â”€ Phase 13-16 TidyView extensions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl TidyView {

    // â”€â”€ pivot_longer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Pivot selected columns from wide to long format.
    ///
    /// `value_cols`: columns to gather (must all have the same type).
    /// `names_to`: name of the output "variable name" column.
    /// `values_to`: name of the output "value" column.
    ///
    /// Output schema: [id_cols..., names_to, values_to]
    /// Row order: for each source row (in visible order), one output row per
    ///   value column (in the order they appear in `value_cols`).
    ///
    /// Edge cases:
    ///   â€¢ `value_cols` empty â†’ `TidyError::EmptySelection`.
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ Duplicate in `value_cols` â†’ `TidyError::DuplicateColumn`.
    ///   â€¢ Mixed types in `value_cols` â†’ `TidyError::TypeMismatch`.
    pub fn pivot_longer(
        &self,
        value_cols: &[&str],
        names_to: &str,
        values_to: &str,
    ) -> Result<TidyFrame, TidyError> {
        if value_cols.is_empty() {
            return Err(TidyError::empty_selection("pivot_longer requires at least one value_col"));
        }

        // Validate and resolve value column indices
        let mut seen_vc: Vec<&str> = Vec::new();
        let mut vc_indices: Vec<usize> = Vec::new();
        for &name in value_cols {
            if seen_vc.contains(&name) {
                return Err(TidyError::DuplicateColumn(name.to_string()));
            }
            seen_vc.push(name);
            let idx = self.base.columns.iter().position(|(n, _)| n == name)
                .ok_or_else(|| TidyError::ColumnNotFound(name.to_string()))?;
            vc_indices.push(idx);
        }

        // Type consistency check: all value columns must have the same type
        let first_type = self.base.columns[vc_indices[0]].1.type_name();
        for &idx in &vc_indices[1..] {
            let t = self.base.columns[idx].1.type_name();
            if t != first_type {
                return Err(TidyError::TypeMismatch {
                    expected: first_type.to_string(),
                    got: t.to_string(),
                });
            }
        }

        // id_cols = projected columns excluding value_cols
        let vc_set: std::collections::BTreeSet<usize> = vc_indices.iter().copied().collect();
        let id_col_indices: Vec<usize> = self.proj.indices().iter()
            .copied()
            .filter(|i| !vc_set.contains(i))
            .collect();

        let visible_rows: Vec<usize> = self.mask.iter_set().collect();
        let n_out = visible_rows.len() * value_cols.len();

        // Build id columns (repeated value_cols.len() times per source row)
        let mut out_cols: Vec<(String, Column)> = Vec::new();
        for &id_idx in &id_col_indices {
            let (name, col) = &self.base.columns[id_idx];
            let new_col = match col {
                Column::Int(v) => {
                    let mut out = Vec::with_capacity(n_out);
                    for &r in &visible_rows {
                        for _ in 0..value_cols.len() { out.push(v[r]); }
                    }
                    Column::Int(out)
                }
                Column::Float(v) => {
                    let mut out = Vec::with_capacity(n_out);
                    for &r in &visible_rows {
                        for _ in 0..value_cols.len() { out.push(v[r]); }
                    }
                    Column::Float(out)
                }
                Column::Str(v) => {
                    let mut out = Vec::with_capacity(n_out);
                    for &r in &visible_rows {
                        for _ in 0..value_cols.len() { out.push(v[r].clone()); }
                    }
                    Column::Str(out)
                }
                Column::Bool(v) => {
                    let mut out = Vec::with_capacity(n_out);
                    for &r in &visible_rows {
                        for _ in 0..value_cols.len() { out.push(v[r]); }
                    }
                    Column::Bool(out)
                }
                Column::Categorical { levels, codes } => {
                    let mut out = Vec::with_capacity(n_out);
                    for &r in &visible_rows {
                        for _ in 0..value_cols.len() { out.push(codes[r]); }
                    }
                    Column::Categorical { levels: levels.clone(), codes: out }
                }
                Column::DateTime(v) => {
                    let mut out = Vec::with_capacity(n_out);
                    for &r in &visible_rows {
                        for _ in 0..value_cols.len() { out.push(v[r]); }
                    }
                    Column::DateTime(out)
                }
            };
            out_cols.push((name.clone(), new_col));
        }

        // Build "names" column (the variable name, repeated per source row)
        let names_col: Vec<String> = visible_rows.iter()
            .flat_map(|_| value_cols.iter().map(|s| s.to_string()))
            .collect();
        out_cols.push((names_to.to_string(), Column::Str(names_col)));

        // Build "values" column (all types already checked equal)
        match &self.base.columns[vc_indices[0]].1 {
            Column::Int(_) => {
                let mut vals: Vec<i64> = Vec::with_capacity(n_out);
                for &r in &visible_rows {
                    for &vci in &vc_indices {
                        if let Column::Int(v) = &self.base.columns[vci].1 {
                            vals.push(v[r]);
                        }
                    }
                }
                out_cols.push((values_to.to_string(), Column::Int(vals)));
            }
            Column::Float(_) => {
                let mut vals: Vec<f64> = Vec::with_capacity(n_out);
                for &r in &visible_rows {
                    for &vci in &vc_indices {
                        if let Column::Float(v) = &self.base.columns[vci].1 {
                            vals.push(v[r]);
                        }
                    }
                }
                out_cols.push((values_to.to_string(), Column::Float(vals)));
            }
            Column::Str(_) => {
                let mut vals: Vec<String> = Vec::with_capacity(n_out);
                for &r in &visible_rows {
                    for &vci in &vc_indices {
                        if let Column::Str(v) = &self.base.columns[vci].1 {
                            vals.push(v[r].clone());
                        }
                    }
                }
                out_cols.push((values_to.to_string(), Column::Str(vals)));
            }
            Column::Bool(_) => {
                let mut vals: Vec<bool> = Vec::with_capacity(n_out);
                for &r in &visible_rows {
                    for &vci in &vc_indices {
                        if let Column::Bool(v) = &self.base.columns[vci].1 {
                            vals.push(v[r]);
                        }
                    }
                }
                out_cols.push((values_to.to_string(), Column::Bool(vals)));
            }
            Column::Categorical { .. } | Column::DateTime(_) => {
                // For pivot_longer, fall back to string representation
                let mut vals: Vec<String> = Vec::with_capacity(n_out);
                for &r in &visible_rows {
                    for &vci in &vc_indices {
                        vals.push(self.base.columns[vci].1.get_display(r));
                    }
                }
                out_cols.push((values_to.to_string(), Column::Str(vals)));
            }
        }

        let df = DataFrame::from_columns(out_cols)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }

    // â”€â”€ pivot_wider â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Pivot long-format data to wide format.
    ///
    /// `names_from`: the column whose values become new column headers.
    /// `values_from`: the column whose values fill the new columns.
    /// `id_cols`: columns that identify each output row.
    ///
    /// Output schema: [id_cols..., unique_key_values... (first-occurrence order)]
    /// Row order: one row per unique combination of id_col values
    ///   (first-occurrence order).
    ///
    /// Edge cases:
    ///   â€¢ Duplicate (id_key, name_key) combo â†’ `TidyError::DuplicateKey`.
    ///   â€¢ Missing combo â†’ null fill via NullableFrame.
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    pub fn pivot_wider(
        &self,
        id_cols: &[&str],
        names_from: &str,
        values_from: &str,
    ) -> Result<NullableFrame, TidyError> {
        // Validate columns
        let _names_col_idx = self.base.columns.iter().position(|(n, _)| n == names_from)
            .ok_or_else(|| TidyError::ColumnNotFound(names_from.to_string()))?;
        let _values_col_idx = self.base.columns.iter().position(|(n, _)| n == values_from)
            .ok_or_else(|| TidyError::ColumnNotFound(values_from.to_string()))?;
        for &id in id_cols {
            let _ = self.base.columns.iter().position(|(n, _)| n == id)
                .ok_or_else(|| TidyError::ColumnNotFound(id.to_string()))?;
        }

        let visible_rows: Vec<usize> = self.mask.iter_set().collect();

        // Collect unique key values in first-occurrence order
        let mut key_values: Vec<String> = Vec::new();
        for &r in &visible_rows {
            let kv = self.base.get_column(names_from).unwrap().get_display(r);
            if !key_values.contains(&kv) {
                key_values.push(kv);
            }
        }

        // Collect unique id combinations in first-occurrence order
        // Map: id_tuple â†’ output_row_slot
        let id_col_refs: Vec<&Column> = id_cols.iter()
            .map(|&name| self.base.get_column(name).unwrap())
            .collect();

        let mut id_order: Vec<Vec<String>> = Vec::new(); // first-occurrence
        let mut id_to_slot: Vec<(Vec<String>, usize)> = Vec::new(); // linear scan map

        for &r in &visible_rows {
            let id_key: Vec<String> = id_col_refs.iter()
                .map(|col| col.get_display(r))
                .collect();
            if !id_to_slot.iter().any(|(k, _)| k == &id_key) {
                let slot = id_order.len();
                id_order.push(id_key.clone());
                id_to_slot.push((id_key, slot));
            }
        }

        let n_rows = id_order.len();
        let n_keys = key_values.len();

        // Cell lookup: (id_slot, key_slot) â†’ source row index
        // Detect duplicate (id, key) combinations
        let mut cell_map: Vec<Vec<Option<usize>>> = vec![vec![None; n_keys]; n_rows];

        for &r in &visible_rows {
            let id_key: Vec<String> = id_col_refs.iter()
                .map(|col| col.get_display(r))
                .collect();
            let id_slot = id_to_slot.iter().find(|(k, _)| k == &id_key).unwrap().1;

            let kv = self.base.get_column(names_from).unwrap().get_display(r);
            let key_slot = key_values.iter().position(|v| v == &kv).unwrap();

            if cell_map[id_slot][key_slot].is_some() {
                return Err(TidyError::duplicate_key(
                    format!("({}, {})", id_key.join(", "), kv)
                ));
            }
            cell_map[id_slot][key_slot] = Some(r);
        }

        // Build output NullableFrame
        let mut out_cols: Vec<(String, NullCol)> = Vec::new();

        // Id columns
        for (id_idx, &id_name) in id_cols.iter().enumerate() {
            let id_col = self.base.get_column(id_name).unwrap();
            let id_row_indices: Vec<usize> = id_order.iter()
                .map(|id_tup| {
                    // Find the first visible row that has this id tuple
                    *visible_rows.iter().find(|&&r| {
                        id_col_refs.iter().enumerate().all(|(i, col)| {
                            col.get_display(r) == id_tup[i]
                        })
                    }).unwrap()
                })
                .collect();
            let gathered = gather_column(id_col, &id_row_indices);
            out_cols.push((id_name.to_string(), NullCol::from_column(&gathered)));
            let _ = id_idx;
        }

        // Value columns (one per unique key value)
        let values_col = self.base.get_column(values_from).unwrap();
        let val_type = values_col.type_name();
        for (key_slot, key_val) in key_values.iter().enumerate() {
            let row_opts: Vec<Option<usize>> = (0..n_rows)
                .map(|id_slot| cell_map[id_slot][key_slot])
                .collect();
            let null_col = gather_column_nullable_null(values_col, &row_opts);
            out_cols.push((key_val.clone(), null_col));
            let _ = val_type;
        }

        Ok(NullableFrame { columns: out_cols })
    }

    // â”€â”€ rename â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Rename columns: `renames` is a slice of `(old_name, new_name)`.
    ///
    /// Returns a new `TidyView` over a new base DataFrame with renamed columns.
    ///
    /// Edge cases:
    ///   â€¢ Unknown `old_name` â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ `new_name` already exists (collision) â†’ `TidyError::DuplicateColumn`.
    ///   â€¢ `old_name == new_name` â†’ no-op for that pair.
    pub fn rename(&self, renames: &[(&str, &str)]) -> Result<TidyView, TidyError> {
        // Build rename map
        let mut rename_map: Vec<(usize, String)> = Vec::new();
        let col_names: Vec<&str> = self.base.columns.iter().map(|(n, _)| n.as_str()).collect();

        for &(old, new) in renames {
            let idx = col_names.iter().position(|&n| n == old)
                .ok_or_else(|| TidyError::ColumnNotFound(old.to_string()))?;
            // Check new name doesn't already exist (unless it's the old name itself)
            if old != new {
                let new_name_exists = col_names.iter().any(|&n| n == new)
                    || rename_map.iter().any(|(_, n)| n == new);
                if new_name_exists {
                    return Err(TidyError::DuplicateColumn(new.to_string()));
                }
            }
            rename_map.push((idx, new.to_string()));
        }

        // Build new base with renamed columns
        let mut new_cols: Vec<(String, Column)> = Vec::new();
        for (i, (name, col)) in self.base.columns.iter().enumerate() {
            let new_name = rename_map.iter()
                .find(|(idx, _)| *idx == i)
                .map(|(_, n)| n.clone())
                .unwrap_or_else(|| name.clone());
            new_cols.push((new_name, col.clone()));
        }

        let new_base = DataFrame { columns: new_cols };
        Ok(TidyView {
            base: Rc::new(new_base),
            mask: self.mask.clone(),
            proj: self.proj.clone(),
        })
    }

    // â”€â”€ relocate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Reorder columns so that `cols` appear at position `before` or `after`
    /// another column, or at the front/back.
    ///
    /// `cols`: columns to move.
    /// `position`: `RelocatePos::Front`, `Back`, `Before(name)`, `After(name)`.
    ///
    /// Non-moved columns keep their relative order.
    /// Returns a new `TidyView` with updated projection.
    ///
    /// Edge cases:
    ///   â€¢ Unknown column in `cols` â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ Unknown anchor column â†’ `TidyError::ColumnNotFound`.
    pub fn relocate(&self, cols: &[&str], position: RelocatePos<'_>) -> Result<TidyView, TidyError> {
        // Validate cols exist in projection
        let proj_names: Vec<&str> = self.column_names();
        for &name in cols {
            if !proj_names.contains(&name) {
                return Err(TidyError::ColumnNotFound(name.to_string()));
            }
        }

        // Build new column order in the projection
        let moved_set: std::collections::BTreeSet<&str> = cols.iter().copied().collect();
        let remaining: Vec<&str> = proj_names.iter()
            .copied()
            .filter(|n| !moved_set.contains(n))
            .collect();

        let new_order: Vec<&str> = match &position {
            RelocatePos::Front => {
                let mut v: Vec<&str> = cols.to_vec();
                v.extend_from_slice(&remaining);
                v
            }
            RelocatePos::Back => {
                let mut v = remaining.clone();
                v.extend_from_slice(cols);
                v
            }
            RelocatePos::Before(anchor) => {
                if !proj_names.contains(anchor) {
                    return Err(TidyError::ColumnNotFound(anchor.to_string()));
                }
                let mut v = Vec::new();
                for &n in &remaining {
                    if n == *anchor {
                        v.extend_from_slice(cols);
                    }
                    v.push(n);
                }
                v
            }
            RelocatePos::After(anchor) => {
                if !proj_names.contains(anchor) {
                    return Err(TidyError::ColumnNotFound(anchor.to_string()));
                }
                let mut v = Vec::new();
                for &n in &remaining {
                    v.push(n);
                    if n == *anchor {
                        v.extend_from_slice(cols);
                    }
                }
                v
            }
        };

        // Map new_order back to base column indices
        let new_indices: Vec<usize> = new_order.iter()
            .map(|&name| {
                self.base.columns.iter().position(|(n, _)| n == name).unwrap()
            })
            .collect();

        Ok(TidyView {
            base: Rc::clone(&self.base),
            mask: self.mask.clone(),
            proj: ProjectionMap::from_indices(new_indices),
        })
    }

    // â”€â”€ drop_cols â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Drop specified columns from the view (select-minus semantics).
    ///
    /// Returns a new `TidyView` with those columns removed from the projection.
    ///
    /// Edge cases:
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ Dropping all columns â†’ valid (0-col view).
    pub fn drop_cols(&self, cols: &[&str]) -> Result<TidyView, TidyError> {
        let proj_names = self.column_names();
        for &name in cols {
            if !proj_names.contains(&name) {
                return Err(TidyError::ColumnNotFound(name.to_string()));
            }
        }
        let drop_set: std::collections::BTreeSet<&str> = cols.iter().copied().collect();
        let keep: Vec<&str> = proj_names.iter()
            .copied()
            .filter(|n| !drop_set.contains(n))
            .collect();
        self.select(&keep)
    }

    // â”€â”€ bind_rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Concatenate rows from `other` onto `self` (strict schema match).
    ///
    /// Both frames must have the same column names in the same order.
    /// Row order: `self` rows first, then `other` rows.
    ///
    /// Edge cases:
    ///   â€¢ Column names differ â†’ `TidyError::Internal("schema mismatch: ...")`.
    ///   â€¢ `other` has zero rows â†’ returns self's rows (valid, no error).
    pub fn bind_rows(&self, other: &TidyView) -> Result<TidyFrame, TidyError> {
        let self_names = self.column_names();
        let other_names = other.column_names();

        if self_names != other_names {
            return Err(TidyError::schema_mismatch(format!(
                "left has {:?}, right has {:?}",
                self_names, other_names
            )));
        }

        let self_rows: Vec<usize> = self.mask.iter_set().collect();
        let other_rows: Vec<usize> = other.mask.iter_set().collect();

        let mut out_cols: Vec<(String, Column)> = Vec::new();
        for &ci in self.proj.indices() {
            let (name, self_col) = &self.base.columns[ci];
            // Find matching column in other's projection
            let other_ci = other.proj.indices().iter().copied()
                .find(|&i| other.base.columns[i].0 == *name)
                .ok_or_else(|| TidyError::ColumnNotFound(name.clone()))?;
            let other_col = &other.base.columns[other_ci].1;

            let col = concat_columns(self_col, &self_rows, other_col, &other_rows)?;
            out_cols.push((name.clone(), col));
        }

        let df = DataFrame::from_columns(out_cols)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }

    // â”€â”€ bind_cols â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Concatenate columns from `other` onto `self` (strict row count match).
    ///
    /// Both frames must have the same number of visible rows.
    /// Column order: `self` columns first, then `other` columns.
    ///
    /// Edge cases:
    ///   â€¢ Row count mismatch â†’ `TidyError::LengthMismatch`.
    ///   â€¢ Column name collision â†’ `TidyError::DuplicateColumn`.
    pub fn bind_cols(&self, other: &TidyView) -> Result<TidyFrame, TidyError> {
        let self_nrows = self.nrows();
        let other_nrows = other.nrows();

        if self_nrows != other_nrows {
            return Err(TidyError::LengthMismatch {
                expected: self_nrows,
                got: other_nrows,
            });
        }

        let self_names = self.column_names();
        let other_names = other.column_names();
        for name in &other_names {
            if self_names.contains(name) {
                return Err(TidyError::DuplicateColumn(name.to_string()));
            }
        }

        let self_rows: Vec<usize> = self.mask.iter_set().collect();
        let other_rows: Vec<usize> = other.mask.iter_set().collect();

        let mut out_cols: Vec<(String, Column)> = Vec::new();

        for &ci in self.proj.indices() {
            let (name, col) = &self.base.columns[ci];
            out_cols.push((name.clone(), gather_column(col, &self_rows)));
        }
        for &ci in other.proj.indices() {
            let (name, col) = &other.base.columns[ci];
            out_cols.push((name.clone(), gather_column(col, &other_rows)));
        }

        let df = DataFrame::from_columns(out_cols)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }

    // â”€â”€ mutate_across â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Apply a transformation across multiple columns, adding/replacing each
    /// with a generated name `{col}_{fn}` (or a user-specified template).
    ///
    /// Edge cases:
    ///   â€¢ Unknown column â†’ `TidyError::ColumnNotFound`.
    ///   â€¢ Generated name collision â†’ `TidyError::DuplicateColumn`.
    ///   â€¢ Empty cols list â†’ no-op (returns materialized frame unchanged).
    pub fn mutate_across(&self, specs: &[AcrossSpec]) -> Result<TidyFrame, TidyError> {
        // Materialize self first
        let base_df = self.materialize()?;

        // Collect assignments, checking for name collisions
        let mut output_names: Vec<String> = base_df.column_names()
            .into_iter().map(|s| s.to_string()).collect();
        let mut extra_cols: Vec<(String, Column)> = Vec::new();

        for spec in specs {
            for col_name in &spec.cols {
                let out_name = spec.output_name(col_name);
                // Check for duplicate in output
                if output_names.contains(&out_name) && !base_df.column_names().contains(&out_name.as_str()) {
                    return Err(TidyError::DuplicateColumn(out_name));
                }
                let col = base_df.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                let new_col = (spec.transform.func)(col_name, col)?;
                // Duplicate in output_names is overwrite for existing cols, error for new
                if !base_df.column_names().contains(&out_name.as_str()) {
                    output_names.push(out_name.clone());
                }
                extra_cols.push((out_name, new_col));
            }
        }

        // Merge: start from base_df columns, then add/overwrite extras
        let mut col_map: indexmap_simple::IndexMap = indexmap_simple::IndexMap::from_df(&base_df);
        for (name, col) in extra_cols {
            col_map.insert(name, col);
        }
        let df = col_map.into_df()
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }

    // â”€â”€ right_join â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Right join: all rows from `right`, matched rows from `self` (left).
    ///
    /// Output: left cols (nullable) + right cols.
    /// Row order: right outer loop order preserved.
    /// Unmatched right rows: left columns null-filled.
    pub fn right_join(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
        suffix: &JoinSuffix,
    ) -> Result<NullableFrame, TidyError> {
        // Validate key type compatibility
        validate_join_key_types(self, right, on)?;
        // Swap sides: right becomes "left" of a left join, then re-order columns
        let swapped_on: Vec<(&str, &str)> = on.iter().map(|&(l, r)| (r, l)).collect();
        let (right_rows, left_rows_opt) =
            join_match_rows_optional(right, self, &swapped_on, JoinKind::Left)?;
        build_right_join_frame(self, right, &left_rows_opt, &right_rows, on, suffix)
    }

    // â”€â”€ full_join â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Full outer join: all rows from both sides; null-fill for unmatched.
    ///
    /// Row order: left rows first (matched and unmatched), then unmatched right rows.
    pub fn full_join(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
        suffix: &JoinSuffix,
    ) -> Result<NullableFrame, TidyError> {
        validate_join_key_types(self, right, on)?;
        build_full_join_frame(self, right, on, suffix)
    }

    // â”€â”€ inner_join_typed (join maturity upgrade) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Inner join with type validation and collision suffix support.
    ///
    /// Same semantics as `inner_join` but:
    ///   â€¢ validates join key types are compatible (Int/Float widened, others exact).
    ///   â€¢ handles non-key column name collisions using `suffix`.
    pub fn inner_join_typed(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
        suffix: &JoinSuffix,
    ) -> Result<TidyFrame, TidyError> {
        validate_join_key_types(self, right, on)?;
        let (left_rows, right_rows) = join_match_rows(self, right, on, JoinKind::Inner)?;
        build_join_frame_with_suffix(self, right, &left_rows, &right_rows, on, suffix, false)
    }

    // â”€â”€ left_join_typed (join maturity upgrade) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Left join with type validation and collision suffix support.
    pub fn left_join_typed(
        &self,
        right: &TidyView,
        on: &[(&str, &str)],
        suffix: &JoinSuffix,
    ) -> Result<TidyFrame, TidyError> {
        validate_join_key_types(self, right, on)?;
        let (left_rows, right_rows_opt) =
            join_match_rows_optional(self, right, on, JoinKind::Left)?;
        build_left_join_frame_with_suffix(self, right, &left_rows, &right_rows_opt, on, suffix)
    }
}

// â”€â”€ Position enum for relocate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Position specifier for `TidyView::relocate`.
pub enum RelocatePos<'a> {
    /// Move selected columns to the front.
    Front,
    /// Move selected columns to the back.
    Back,
    /// Insert selected columns immediately before the named column.
    Before(&'a str),
    /// Insert selected columns immediately after the named column.
    After(&'a str),
}

// â”€â”€ Column concatenation helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn concat_columns(
    left: &Column,
    left_rows: &[usize],
    right: &Column,
    right_rows: &[usize],
) -> Result<Column, TidyError> {
    match (left, right) {
        (Column::Int(lv), Column::Int(rv)) => {
            let mut out: Vec<i64> = left_rows.iter().map(|&i| lv[i]).collect();
            out.extend(right_rows.iter().map(|&i| rv[i]));
            Ok(Column::Int(out))
        }
        (Column::Float(lv), Column::Float(rv)) => {
            let mut out: Vec<f64> = left_rows.iter().map(|&i| lv[i]).collect();
            out.extend(right_rows.iter().map(|&i| rv[i]));
            Ok(Column::Float(out))
        }
        (Column::Int(lv), Column::Float(rv)) => {
            let mut out: Vec<f64> = left_rows.iter().map(|&i| lv[i] as f64).collect();
            out.extend(right_rows.iter().map(|&i| rv[i]));
            Ok(Column::Float(out))
        }
        (Column::Float(lv), Column::Int(rv)) => {
            let mut out: Vec<f64> = left_rows.iter().map(|&i| lv[i]).collect();
            out.extend(right_rows.iter().map(|&i| rv[i] as f64));
            Ok(Column::Float(out))
        }
        (Column::Str(lv), Column::Str(rv)) => {
            let mut out: Vec<String> = left_rows.iter().map(|&i| lv[i].clone()).collect();
            out.extend(right_rows.iter().map(|&i| rv[i].clone()));
            Ok(Column::Str(out))
        }
        (Column::Bool(lv), Column::Bool(rv)) => {
            let mut out: Vec<bool> = left_rows.iter().map(|&i| lv[i]).collect();
            out.extend(right_rows.iter().map(|&i| rv[i]));
            Ok(Column::Bool(out))
        }
        _ => Err(TidyError::schema_mismatch(format!(
            "type mismatch in bind_rows: {} vs {}",
            left.type_name(), right.type_name()
        ))),
    }
}

// â”€â”€ Join key type validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn validate_join_key_types(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
) -> Result<(), TidyError> {
    for &(lk, rk) in on {
        let l_col = left.base.get_column(lk)
            .ok_or_else(|| TidyError::ColumnNotFound(lk.to_string()))?;
        let r_col = right.base.get_column(rk)
            .ok_or_else(|| TidyError::ColumnNotFound(rk.to_string()))?;
        if !join_types_compatible(l_col, r_col) {
            return Err(TidyError::join_type_mismatch(lk, l_col.type_name(), r_col.type_name()));
        }
    }
    Ok(())
}

// â”€â”€ Join frames with suffix collision handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn build_join_frame_with_suffix(
    left: &TidyView,
    right: &TidyView,
    left_rows: &[usize],
    right_rows: &[usize],
    on: &[(&str, &str)],
    suffix: &JoinSuffix,
    _include_unmatched: bool,
) -> Result<TidyFrame, TidyError> {
    let right_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(_, rk)| *rk).collect();

    // Collect left projected column names
    let left_col_names: Vec<String> = left.proj.indices().iter()
        .map(|&ci| left.base.columns[ci].0.clone())
        .collect();

    let mut columns: Vec<(String, Column)> = Vec::new();

    // Left projected columns
    for &ci in left.proj.indices() {
        let (name, col) = &left.base.columns[ci];
        columns.push((name.clone(), gather_column(col, left_rows)));
    }

    // Right projected columns with suffix on collision
    for &ci in right.proj.indices() {
        let (name, col) = &right.base.columns[ci];
        if right_key_names.contains(name.as_str()) {
            continue; // skip join key duplication
        }
        let out_name = if left_col_names.contains(name) {
            format!("{}{}", name, suffix.right)
        } else {
            name.clone()
        };
        // Rename left side if it also collides
        if left_col_names.contains(name) {
            // Rename the left column already added
            let left_pos = columns.iter().position(|(n, _)| n == name);
            if let Some(pos) = left_pos {
                let entry = &mut columns[pos];
                entry.0 = format!("{}{}", entry.0, suffix.left);
            }
        }
        columns.push((out_name, gather_column(col, right_rows)));
    }

    let df = DataFrame::from_columns(columns)
        .map_err(|e| TidyError::Internal(e.to_string()))?;
    Ok(TidyFrame::from_df(df))
}

fn build_left_join_frame_with_suffix(
    left: &TidyView,
    right: &TidyView,
    left_rows: &[usize],
    right_rows_opt: &[Option<usize>],
    on: &[(&str, &str)],
    suffix: &JoinSuffix,
) -> Result<TidyFrame, TidyError> {
    let right_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(_, rk)| *rk).collect();

    let left_col_names: Vec<String> = left.proj.indices().iter()
        .map(|&ci| left.base.columns[ci].0.clone())
        .collect();

    let mut columns: Vec<(String, Column)> = Vec::new();

    // Left projected columns
    for &ci in left.proj.indices() {
        let (name, col) = &left.base.columns[ci];
        columns.push((name.clone(), gather_column(col, left_rows)));
    }

    // Right projected columns (nullable fill for unmatched)
    for &ci in right.proj.indices() {
        let (name, col) = &right.base.columns[ci];
        if right_key_names.contains(name.as_str()) { continue; }
        let out_name = if left_col_names.contains(name) {
            // rename left column
            let left_pos = columns.iter().position(|(n, _)| n == name);
            if let Some(pos) = left_pos {
                columns[pos].0 = format!("{}{}", name, suffix.left);
            }
            format!("{}{}", name, suffix.right)
        } else {
            name.clone()
        };
        let new_col = gather_column_nullable(col, right_rows_opt);
        columns.push((out_name, new_col));
    }

    let df = DataFrame::from_columns(columns)
        .map_err(|e| TidyError::Internal(e.to_string()))?;
    Ok(TidyFrame::from_df(df))
}

fn build_right_join_frame(
    left: &TidyView,
    right: &TidyView,
    left_rows_opt: &[Option<usize>],
    right_rows: &[usize],
    on: &[(&str, &str)],
    suffix: &JoinSuffix,
) -> Result<NullableFrame, TidyError> {
    let right_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(_, rk)| *rk).collect();
    let left_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(lk, _)| *lk).collect();

    let right_col_names: Vec<String> = right.proj.indices().iter()
        .map(|&ci| right.base.columns[ci].0.clone())
        .collect();

    let mut columns: Vec<(String, NullCol)> = Vec::new();

    // Left projected columns (nullable â€” unmatched = null)
    for &ci in left.proj.indices() {
        let (name, col) = &left.base.columns[ci];
        if left_key_names.contains(name.as_str()) { continue; }
        let out_name = if right_col_names.contains(name) {
            format!("{}{}", name, suffix.left)
        } else {
            name.clone()
        };
        let null_col = gather_column_nullable_null(col, left_rows_opt);
        columns.push((out_name, null_col));
    }

    // Right projected columns (always present)
    for &ci in right.proj.indices() {
        let (name, col) = &right.base.columns[ci];
        let out_name = if !right_key_names.contains(name.as_str())
            && left.proj.indices().iter().any(|&lci| left.base.columns[lci].0 == *name)
            && !left_key_names.contains(name.as_str())
        {
            format!("{}{}", name, suffix.right)
        } else {
            name.clone()
        };
        columns.push((out_name, NullCol::from_column(&gather_column(col, right_rows))));
    }

    Ok(NullableFrame { columns })
}

fn build_full_join_frame(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
    suffix: &JoinSuffix,
) -> Result<NullableFrame, TidyError> {
    let (left_key_cols, right_key_cols) = resolve_join_keys(left, right, on)?;
    let lookup = build_right_lookup(right, &right_key_cols);

    // Phase 1: left outer loop (all left rows, with or without right match)
    let mut out_left_rows: Vec<usize> = Vec::new();
    let mut out_right_rows: Vec<Option<usize>> = Vec::new();
    let mut right_matched: Vec<bool> = vec![false; right.base.nrows()];

    for l_row in left.mask.iter_set() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        let matches = find_matches(&lookup, &key);
        if matches.is_empty() {
            out_left_rows.push(l_row);
            out_right_rows.push(None);
        } else {
            for r_row in &matches {
                out_left_rows.push(l_row);
                out_right_rows.push(Some(*r_row));
                if *r_row < right_matched.len() {
                    right_matched[*r_row] = true;
                }
            }
        }
    }

    // Phase 2: unmatched right rows
    let mut unmatched_right: Vec<usize> = Vec::new();
    for r_row in right.mask.iter_set() {
        if r_row < right_matched.len() && !right_matched[r_row] {
            unmatched_right.push(r_row);
        }
    }

    let right_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(_, rk)| *rk).collect();
    let left_key_names: std::collections::BTreeSet<&str> =
        on.iter().map(|(lk, _)| *lk).collect();
    let right_col_names: Vec<String> = right.proj.indices().iter()
        .map(|&ci| right.base.columns[ci].0.clone())
        .collect();

    let n_matched = out_left_rows.len();
    let n_unmatched_r = unmatched_right.len();
    let total = n_matched + n_unmatched_r;

    let mut columns: Vec<(String, NullCol)> = Vec::new();

    // Left projected columns
    for &ci in left.proj.indices() {
        let (name, col) = &left.base.columns[ci];
        let out_name = if right_col_names.contains(name) && !left_key_names.contains(name.as_str()) {
            format!("{}{}", name, suffix.left)
        } else {
            name.clone()
        };
        let mut matched_vals: Vec<Option<usize>> = out_left_rows.iter()
            .map(|&r| Some(r))
            .collect();
        // Extend with None for unmatched right rows
        matched_vals.extend(std::iter::repeat(None).take(n_unmatched_r));
        assert_eq!(matched_vals.len(), total);
        columns.push((out_name, gather_column_nullable_null(col, &matched_vals)));
    }

    // Right projected columns (skip key cols to avoid duplication from left)
    for &ci in right.proj.indices() {
        let (name, col) = &right.base.columns[ci];
        if right_key_names.contains(name.as_str()) { continue; }
        let out_name = if left.proj.indices().iter().any(|&lci| left.base.columns[lci].0 == *name)
            && !left_key_names.contains(name.as_str())
        {
            format!("{}{}", name, suffix.right)
        } else {
            name.clone()
        };

        let mut row_opts: Vec<Option<usize>> = out_right_rows.clone();
        // Extend with Some(r) for unmatched right rows
        row_opts.extend(unmatched_right.iter().map(|&r| Some(r)));
        assert_eq!(row_opts.len(), total);
        columns.push((out_name, gather_column_nullable_null(col, &row_opts)));
    }

    // Key columns (from left where available, else from right)
    // Emit left key columns first (already in left section above if not filtered)
    // Actually key cols are already included via left.proj if they are projected.
    // The key cols from right are skipped above. Done.

    Ok(NullableFrame { columns })
}

// â”€â”€ GroupedTidyView: mutate_across + summarise_across â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl GroupedTidyView {

    /// Apply across transformations then mutate each group in-place.
    ///
    /// For each group, applies each `AcrossSpec` transform to the specified columns,
    /// producing output named `{col}_{fn}` (or per-template).
    /// Returns a materialized `TidyFrame` with all groups combined.
    pub fn mutate_across(&self, specs: &[AcrossSpec]) -> Result<TidyFrame, TidyError> {
        // Apply mutate_across to the ungrouped view (same result: group structure
        // is not needed for row-wise transforms).
        self.view.mutate_across(specs)
    }

    /// Summarise with across transforms: apply each transform to each column,
    /// collecting one aggregate value per group.
    ///
    /// The `specs` use `AcrossSpec` where the transform function must return a
    /// single-element column (scalar reduction). If it returns more than one row,
    /// `TidyError::LengthMismatch` is returned.
    pub fn summarise_across(&self, specs: &[AcrossSpec]) -> Result<TidyFrame, TidyError> {
        let n_groups = self.ngroups();

        // Build output: key cols first, then across outputs
        let key_names = &self.index.key_names;
        let mut out_cols: Vec<(String, Column)> = Vec::new();

        // Key columns (String typed â€” group key values)
        for ki in 0..key_names.len() {
            let col_vals: Vec<String> = self.index.groups.iter()
                .map(|g| g.key_values[ki].clone())
                .collect();
            out_cols.push((key_names[ki].clone(), Column::Str(col_vals)));
        }

        // For each spec column Ã— transform
        for spec in specs {
            for col_name in &spec.cols {
                let out_name = spec.output_name(col_name);
                // Check for duplicate output name
                if out_cols.iter().any(|(n, _)| n == &out_name) {
                    return Err(TidyError::DuplicateColumn(out_name));
                }

                let base_col = self.view.base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;

                // Apply transform per group, collecting scalar result
                let mut agg_floats: Vec<f64> = Vec::with_capacity(n_groups);
                for group in &self.index.groups {
                    let group_col = gather_column(base_col, &group.row_indices);
                    let result_col = (spec.transform.func)(col_name, &group_col)?;
                    if result_col.len() != 1 {
                        return Err(TidyError::LengthMismatch {
                            expected: 1,
                            got: result_col.len(),
                        });
                    }
                    let v = match &result_col {
                        Column::Float(v) => v[0],
                        Column::Int(v) => v[0] as f64,
                        _ => return Err(TidyError::TypeMismatch {
                            expected: "Float or Int".into(),
                            got: result_col.type_name().into(),
                        }),
                    };
                    agg_floats.push(v);
                }
                out_cols.push((out_name, Column::Float(agg_floats)));
            }
        }

        let df = DataFrame::from_columns(out_cols)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }
}

// â”€â”€ Simple IndexMap for mutate_across column merging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// We need ordered insertion with overwrite semantics. Rather than pulling in
// a dependency, implement a minimal ordered map over (String, Column).

mod indexmap_simple {
    use super::{Column, DataFrame, DataError};

    pub struct IndexMap {
        entries: Vec<(String, Column)>,
    }

    impl IndexMap {
        pub fn from_df(df: &DataFrame) -> Self {
            Self {
                entries: df.columns.iter()
                    .map(|(n, c)| (n.clone(), c.clone()))
                    .collect(),
            }
        }

        /// Insert or overwrite a column by name.
        pub fn insert(&mut self, name: String, col: Column) {
            if let Some(pos) = self.entries.iter().position(|(n, _)| n == &name) {
                self.entries[pos] = (name, col);
            } else {
                self.entries.push((name, col));
            }
        }

        pub fn into_df(self) -> Result<DataFrame, DataError> {
            DataFrame::from_columns(self.entries)
        }
    }
}

// â”€â”€ Group perf upgrade (deterministic hash accelerator) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// The Phase 11 GroupIndex::build() uses a linear scan (Vec<(key, slot)>)
// which is O(N Ã— G). For large N with small G, we add a hash-accelerated
// variant that preserves first-occurrence ordering by using insertion order
// in a deterministic way.
//
// Strategy: use a BTreeMap<key_tuple, slot_index> for O(log G) lookup,
// which is fully deterministic (BTree is ordered by key, not hash seed).
// First-occurrence order is tracked via a separate Vec<slot> (unchanged).
// The BTreeMap is ONLY used for fast lookup; the output order is still
// driven by first-occurrence of groups as they appear in the scan.
//
// This is a pure internal change â€” the external API and output semantics
// are identical to Phase 11.

impl GroupIndex {
    /// Build a GroupIndex using a BTree-accelerated lookup.
    ///
    /// Semantics: identical to `GroupIndex::build()`. First-occurrence group
    /// ordering is preserved. The only difference is O(N log G) vs O(N Ã— G).
    pub fn build_fast(
        base: &DataFrame,
        key_col_indices: &[usize],
        visible_rows: &[usize],
        key_names: Vec<String>,
    ) -> Self {
        use std::collections::BTreeMap;

        let mut groups: Vec<GroupMeta> = Vec::new();
        let mut key_to_slot: BTreeMap<Vec<String>, usize> = BTreeMap::new();

        for &row in visible_rows {
            let key: Vec<String> = key_col_indices.iter()
                .map(|&ci| base.columns[ci].1.get_display(row))
                .collect();

            if let Some(&slot) = key_to_slot.get(&key) {
                groups[slot].row_indices.push(row);
            } else {
                let slot = groups.len();
                let key_values = key.clone();
                key_to_slot.insert(key, slot);
                groups.push(GroupMeta { key_values, row_indices: vec![row] });
            }
        }

        GroupIndex { groups, key_names }
    }
}

// â”€â”€ TidyView: group_by_fast (uses BTree-accelerated GroupIndex) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl TidyView {
    /// Like `group_by` but uses the BTree-accelerated `GroupIndex::build_fast`.
    ///
    /// Semantics and output are IDENTICAL to `group_by`; this is purely an
    /// internal performance upgrade. Tests should confirm identical output.
    pub fn group_by_fast(&self, keys: &[&str]) -> Result<GroupedTidyView, TidyError> {
        let mut key_col_indices = Vec::with_capacity(keys.len());
        for &key in keys {
            let idx = self.base.columns.iter().position(|(n, _)| n == key)
                .ok_or_else(|| TidyError::ColumnNotFound(key.to_string()))?;
            key_col_indices.push(idx);
        }
        let visible_rows: Vec<usize> = self.mask.iter_set().collect();
        let key_names: Vec<String> = keys.iter().map(|s| s.to_string()).collect();
        let index = GroupIndex::build_fast(&self.base, &key_col_indices, &visible_rows, key_names);
        Ok(GroupedTidyView { view: self.clone(), index })
    }
}

// â”€â”€ NoGC safe-builtin registrations (Phase 13â€“16) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// New safe (view/metadata-only, no column buffer alloc):
//   tidy_rename          : builds new DataFrame with renamed cols â€” O(NÃ—K) clone
//                          BUT this is a metadata rebuild, not a hot-path â€” listed as ALLOC
//                          â†’ NOT @nogc safe (rebuilds base)
//   tidy_relocate        : updates ProjectionMap only â€” O(K) â€” SAFE
//   tidy_drop_cols       : updates ProjectionMap only â€” O(K) â€” SAFE
//   tidy_group_by_fast   : BTree lookup + GroupIndex â€” no column alloc â€” SAFE
//
// New materialising (NOT @nogc safe):
//   tidy_pivot_longer    : allocates new column buffers
//   tidy_pivot_wider     : allocates NullableFrame
//   tidy_bind_rows       : allocates concatenated columns
//   tidy_bind_cols       : allocates combined columns
//   tidy_mutate_across   : materialises and transforms columns
//   tidy_right_join      : allocates NullableFrame
//   tidy_full_join       : allocates NullableFrame
//   tidy_inner_join_typed: allocates TidyFrame
//   tidy_left_join_typed : allocates TidyFrame
//   tidy_summarise_across: allocates aggregate frame
//   tidy_rename          : rebuilds base DataFrame (included above)
//
// Registered in cjc-mir/src/nogc_verify.rs:
//   tidy_relocate, tidy_drop_cols, tidy_group_by_fast

// ============================================================================
// PHASE 17: CATEGORICAL FOUNDATIONS â€” fct_encode, fct_lump, fct_reorder,
//           fct_collapse, NullableFactor
// ============================================================================
//
// Design decisions (spec-lock):
//
//  [S-1]  Index type: u16 (max 65,535 distinct levels).  TidyError::CapacityExceeded
//         on overflow.  A future u32 upgrade is a flag-only change.
//  [S-2]  Level ordering: first-occurrence order of the STRING values in the
//         visible rows at encoding time.  No hash involved â†’ deterministic.
//  [S-3]  Null handling: null cells use a SEPARATE validity bitmap (NullableFactor).
//         Null is NOT a level.  Null index slot is 0 but masked out by bitmap.
//  [S-4]  fct_lump tie-breaking: equal-frequency levels keep first-occurrence order;
//         the "Other" bucket is appended LAST in the levels Vec.
//  [S-5]  fct_lump "Other" collision: if a level named "Other" already exists it is
//         renamed "Other_" (iterated until unique).
//  [S-6]  fct_reorder: stable sort (preserve first-occurrence within ties); NaN
//         values in the numeric summary column sort LAST (same rule as arrange).
//  [S-7]  fct_collapse: metadata-only â€” never rewrites data buffer; O(L) pass over
//         levels Vec only.  Re-indexes in O(N) only when compacting dead indices.
//  [S-8]  fct_collapse duplicate level output: if two OLD levels collapse to the
//         SAME NEW name that new name appears once; index remapping is canonical.
//  [S-9]  fct_encode is a materialising op (allocates new u16 buffer) â†’ NOT @nogc.
//  [S-10] fct_lump is materialising (reallocates levels + rebuilds counts) â†’ NOT @nogc.
//  [S-11] fct_reorder is materialising (reorders levels Vec + rebuilds index map) â†’ NOT @nogc.
//  [S-12] fct_collapse is metadata-only â†’ SAFE in @nogc (registered as safe builtin).
//  [S-13] Encoding stability: fct_encode on an already-encoded FctColumn is a no-op
//         (returns clone of self) so that double-encoding is idempotent.
//  [S-14] NullableFactor: FctColumn + BitMask validity.  Null rows carry index=0 in
//         data buffer but bitmap bit is clear; any op that reads the data MUST check
//         the bitmap first.

// â”€â”€ FctColumn â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A compact categorical column: stores u16 indices into a levels table.
///
/// Invariant: `data[i] < levels.len()` for all i where bitmap is set.
/// Null rows (in NullableFactor) may carry index 0 â€” callers must check bitmap.
#[derive(Clone, Debug)]
pub struct FctColumn {
    /// Mapping from index â†’ level string.  Order = first-occurrence of each string
    /// in the source column (deterministic, no hashing).
    pub levels: Vec<String>,
    /// One u16 per row.  Value is the index into `levels`.
    pub data: Vec<u16>,
}

impl FctColumn {
    // â”€â”€ Constructors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Encode a string column into a FctColumn.
    ///
    /// Level order = first-occurrence in `strings`.
    /// Returns Err if more than 65,535 distinct strings are found.
    pub fn encode(strings: &[String]) -> Result<Self, TidyError> {
        use std::collections::BTreeMap;
        let mut levels: Vec<String> = Vec::new();
        // BTreeMap for O(log L) lookup; key ordering is string-lexicographic
        // (deterministic across runs â€” no hash randomness).
        // First-occurrence ORDER is maintained separately by `levels` Vec.
        let mut level_map: BTreeMap<String, u16> = BTreeMap::new();
        let mut data: Vec<u16> = Vec::with_capacity(strings.len());

        for s in strings {
            let idx = if let Some(&existing) = level_map.get(s.as_str()) {
                existing
            } else {
                let next = levels.len();
                if next >= 65_535 {
                    return Err(TidyError::CapacityExceeded {
                        limit: 65_535,
                        got: next + 1,
                    });
                }
                let idx = next as u16;
                levels.push(s.clone());
                level_map.insert(s.clone(), idx);
                idx
            };
            data.push(idx);
        }
        Ok(FctColumn { levels, data })
    }

    /// Encode a `Column::Str` from a `TidyView` column (respects mask & projection).
    pub fn encode_from_view(view: &TidyView, col: &str) -> Result<Self, TidyError> {
        let base_idx = view.base.columns.iter()
            .position(|(n, _)| n == col)
            .ok_or_else(|| TidyError::ColumnNotFound(col.to_string()))?;
        // Check it is in the current projection
        if !view.proj.indices().contains(&base_idx) {
            return Err(TidyError::ColumnNotFound(col.to_string()));
        }
        let col_data = &view.base.columns[base_idx].1;
        let visible: Vec<usize> = view.mask.iter_set().collect();
        let strings: Vec<String> = visible.iter()
            .map(|&r| col_data.get_display(r))
            .collect();
        Self::encode(&strings)
    }

    // â”€â”€ Shape â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    pub fn nrows(&self) -> usize { self.data.len() }
    pub fn nlevels(&self) -> usize { self.levels.len() }

    /// Decode row i back to its string value.
    pub fn decode(&self, i: usize) -> &str {
        &self.levels[self.data[i] as usize]
    }

    // â”€â”€ fct_lump â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Lump all but the top-`n` most frequent levels into "Other".
    ///
    /// Tie-breaking: equal-frequency levels keep first-occurrence order in
    /// the top-n selection.
    ///
    /// Edge cases:
    /// - n = 0  â†’ all levels become "Other" (one level total)
    /// - n â‰¥ nlevels â†’ no lumping, returns self.clone()
    /// - "Other" already present â†’ renamed to "Other_" (iterate until unique)
    pub fn fct_lump(&self, n: usize) -> Result<Self, TidyError> {
        if n >= self.levels.len() {
            return Ok(self.clone()); // nothing to lump
        }

        // Count frequencies per level (O(N))
        let mut freq = vec![0usize; self.levels.len()];
        for &idx in &self.data {
            freq[idx as usize] += 1;
        }

        // Build ranked list of (level_idx, freq, first_occurrence_order).
        // Sort descending by freq; ties keep ascending level_idx (= first-occurrence).
        let mut ranked: Vec<(usize, usize)> = freq.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        // Top-n level indices (preserve first-occurrence ordering in output)
        let mut keep_set: Vec<usize> = ranked[..n].iter().map(|(i, _)| *i).collect();
        keep_set.sort_unstable(); // restore first-occurrence ordering

        // Determine the "Other" bucket name (avoid collision)
        let mut other_name = "Other".to_string();
        while keep_set.iter().any(|&ki| self.levels[ki] == other_name) {
            other_name.push('_');
        }

        // Build new levels: keep-set levels in first-occurrence order, then "Other" last
        let mut new_levels: Vec<String> = keep_set.iter().map(|&ki| self.levels[ki].clone()).collect();
        let other_idx = new_levels.len() as u16;
        new_levels.push(other_name);

        // Build oldâ†’new index map
        let mut remap = vec![other_idx; self.levels.len()];
        for (new_i, &old_i) in keep_set.iter().enumerate() {
            remap[old_i] = new_i as u16;
        }

        let new_data: Vec<u16> = self.data.iter().map(|&d| remap[d as usize]).collect();
        Ok(FctColumn { levels: new_levels, data: new_data })
    }

    // â”€â”€ fct_reorder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Reorder levels by a numeric summary column from the same frame.
    ///
    /// `summary_vals[i]` is the numeric value for level i.
    /// Ascending = smallest summary value first.
    /// NaN sorts LAST (same rule as arrange).
    /// Tie-breaking: stable sort (original level order preserved within ties).
    pub fn fct_reorder(&self, summary_vals: &[f64], descending: bool) -> Result<Self, TidyError> {
        if summary_vals.len() != self.levels.len() {
            return Err(TidyError::LengthMismatch {
                expected: self.levels.len(),
                got: summary_vals.len(),
            });
        }
        // Build (level_idx, summary_val) and sort.
        // NaN always sorts LAST regardless of direction (same rule as arrange).
        // Direction only affects the finite-value comparison.
        let mut order: Vec<usize> = (0..self.levels.len()).collect();
        order.sort_by(|&a, &b| {
            let va = summary_vals[a];
            let vb = summary_vals[b];
            match (va.is_nan(), vb.is_nan()) {
                (true, true)  => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // NaN always last
                (false, true) => std::cmp::Ordering::Less,    // NaN always last
                (false, false) => {
                    let cmp = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
                    if descending { cmp.reverse() } else { cmp }
                }
            }
        });

        // Build new levels in the new order
        let new_levels: Vec<String> = order.iter().map(|&i| self.levels[i].clone()).collect();

        // Build oldâ†’new index map
        let mut remap = vec![0u16; self.levels.len()];
        for (new_i, &old_i) in order.iter().enumerate() {
            remap[old_i] = new_i as u16;
        }

        let new_data: Vec<u16> = self.data.iter().map(|&d| remap[d as usize]).collect();
        Ok(FctColumn { levels: new_levels, data: new_data })
    }

    /// Convenience: compute per-level mean of a numeric column, then reorder.
    ///
    /// `numeric_col` must be Column::Float or Column::Int and same length as self.
    /// NaN values in the numeric column are excluded from the mean; if all rows
    /// for a level are NaN the level gets summary NaN (sorts last).
    pub fn fct_reorder_by_col(&self, numeric_col: &Column, descending: bool) -> Result<Self, TidyError> {
        if numeric_col.len() != self.data.len() {
            return Err(TidyError::LengthMismatch {
                expected: self.data.len(),
                got: numeric_col.len(),
            });
        }
        let mut sums = vec![0.0f64; self.levels.len()];
        let mut counts = vec![0usize; self.levels.len()];
        match numeric_col {
            Column::Float(v) => {
                for (i, &d) in self.data.iter().enumerate() {
                    let val = v[i];
                    if !val.is_nan() {
                        sums[d as usize] += val;
                        counts[d as usize] += 1;
                    }
                }
            }
            Column::Int(v) => {
                for (i, &d) in self.data.iter().enumerate() {
                    sums[d as usize] += v[i] as f64;
                    counts[d as usize] += 1;
                }
            }
            _ => return Err(TidyError::TypeMismatch {
                expected: "Float or Int".to_string(),
                got: numeric_col.type_name().to_string(),
            }),
        }
        let means: Vec<f64> = sums.iter().zip(counts.iter())
            .map(|(&s, &c)| if c == 0 { f64::NAN } else { s / c as f64 })
            .collect();
        self.fct_reorder(&means, descending)
    }

    // â”€â”€ fct_collapse â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Collapse multiple old levels into single new level names.
    ///
    /// `mapping`: slice of `(old_level_name, new_level_name)`.
    /// - Levels not in mapping keep their original name.
    /// - Multiple old levels can map to the same new name â†’ merged into one index.
    /// - Output level order: first-occurrence of each NEW name, following the
    ///   original first-occurrence order of OLD levels.
    /// - Data buffer is rebuilt (O(N) remap) only when indices actually change.
    ///   The levels Vec is rebuilt O(L) regardless.
    /// - Empty mapping â†’ returns self.clone().
    ///
    /// Capacity: if collapsing reduces level count the result always fits in u16.
    /// The collapsed result can never exceed the original level count, so
    /// CapacityExceeded cannot occur from fct_collapse.
    pub fn fct_collapse(&self, mapping: &[(&str, &str)]) -> Result<Self, TidyError> {
        if mapping.is_empty() {
            return Ok(self.clone());
        }
        // Build: for each old level string, what is the new name?
        let new_name_for: Vec<String> = self.levels.iter().map(|old| {
            if let Some((_, new)) = mapping.iter().find(|(o, _)| *o == old.as_str()) {
                new.to_string()
            } else {
                old.clone()
            }
        }).collect();

        // Build new levels Vec (first-occurrence of new names, following old order).
        // BTreeMap for O(log L) lookup; first-occurrence ORDER is in new_levels Vec.
        use std::collections::BTreeMap;
        let mut new_levels: Vec<String> = Vec::new();
        let mut new_name_to_idx: BTreeMap<String, u16> = BTreeMap::new();

        let mut old_to_new: Vec<u16> = Vec::with_capacity(self.levels.len());
        for name in &new_name_for {
            let idx = if let Some(&existing) = new_name_to_idx.get(name.as_str()) {
                existing
            } else {
                let idx = new_levels.len() as u16;
                new_levels.push(name.clone());
                new_name_to_idx.insert(name.clone(), idx);
                idx
            };
            old_to_new.push(idx);
        }

        // Check if any index actually changed (avoid rewriting data if noop)
        let changed = old_to_new.iter().enumerate().any(|(i, &new)| new != i as u16);
        let new_data = if changed {
            self.data.iter().map(|&d| old_to_new[d as usize]).collect()
        } else {
            self.data.clone()
        };
        Ok(FctColumn { levels: new_levels, data: new_data })
    }

    // â”€â”€ Materialise back to Column::Str â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Decode all rows back into a `Column::Str`.
    pub fn to_str_column(&self) -> Column {
        Column::Str(self.data.iter().map(|&d| self.levels[d as usize].clone()).collect())
    }

    /// Gather rows by index (supports view semantics without full materialise).
    pub fn gather(&self, indices: &[usize]) -> FctColumn {
        FctColumn {
            levels: self.levels.clone(),
            data: indices.iter().map(|&i| self.data[i]).collect(),
        }
    }
}

// â”€â”€ TidyError extensions for Phase 17 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl TidyError {
    pub fn capacity_exceeded(limit: usize, got: usize) -> Self {
        TidyError::CapacityExceeded { limit, got }
    }
}

// â”€â”€ NullableFactor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A FctColumn with a validity bitmap.  Null cells have validity=false.
/// Null is NOT a level; `data[i]` for a null row is 0 (sentinel, must not be used).
#[derive(Clone, Debug)]
pub struct NullableFactor {
    pub fct: FctColumn,
    pub validity: BitMask,
}

impl NullableFactor {
    /// Construct from a FctColumn (all rows valid).
    pub fn from_fct(fct: FctColumn) -> Self {
        let n = fct.nrows();
        NullableFactor { fct, validity: BitMask::all_true(n) }
    }

    /// Construct from a FctColumn + validity bitmap.
    pub fn new(fct: FctColumn, validity: BitMask) -> Self {
        NullableFactor { fct, validity }
    }

    /// Encode a string slice with optional null markers.
    ///
    /// `strings[i] = None` â†’ null row.
    pub fn encode_nullable(strings: &[Option<String>]) -> Result<Self, TidyError> {
        use std::collections::BTreeMap;
        let mut levels: Vec<String> = Vec::new();
        let mut level_map: BTreeMap<String, u16> = BTreeMap::new();
        let mut data: Vec<u16> = Vec::with_capacity(strings.len());
        let mut valid_flags: Vec<bool> = Vec::with_capacity(strings.len());

        for opt in strings {
            match opt {
                None => {
                    data.push(0); // sentinel (ignored due to validity bit)
                    valid_flags.push(false);
                }
                Some(s) => {
                    let idx = if let Some(&existing) = level_map.get(s.as_str()) {
                        existing
                    } else {
                        let next = levels.len();
                        if next >= 65_535 {
                            return Err(TidyError::CapacityExceeded { limit: 65_535, got: next + 1 });
                        }
                        let idx = next as u16;
                        levels.push(s.clone());
                        level_map.insert(s.clone(), idx);
                        idx
                    };
                    data.push(idx);
                    valid_flags.push(true);
                }
            }
        }
        let fct = FctColumn { levels, data };
        let validity = BitMask::from_bools(&valid_flags);
        Ok(NullableFactor { fct, validity })
    }

    pub fn nrows(&self) -> usize { self.fct.nrows() }
    pub fn nlevels(&self) -> usize { self.fct.nlevels() }
    pub fn is_null(&self, i: usize) -> bool { !self.validity.get(i) }
    pub fn count_valid(&self) -> usize { self.validity.count_ones() }

    /// Decode row i, or None if null.
    pub fn decode(&self, i: usize) -> Option<&str> {
        if self.is_null(i) { None } else { Some(self.fct.decode(i)) }
    }

    /// fct_lump on non-null rows only.  Null rows remain null.
    pub fn fct_lump(&self, n: usize) -> Result<Self, TidyError> {
        let lumped = self.fct.fct_lump(n)?;
        Ok(NullableFactor { fct: lumped, validity: self.validity.clone() })
    }

    /// fct_reorder on non-null rows.  Null rows remain null.
    pub fn fct_reorder(&self, summary_vals: &[f64], descending: bool) -> Result<Self, TidyError> {
        let reordered = self.fct.fct_reorder(summary_vals, descending)?;
        Ok(NullableFactor { fct: reordered, validity: self.validity.clone() })
    }

    /// fct_collapse (metadata only, @nogc safe).
    pub fn fct_collapse(&self, mapping: &[(&str, &str)]) -> Result<Self, TidyError> {
        let collapsed = self.fct.fct_collapse(mapping)?;
        Ok(NullableFactor { fct: collapsed, validity: self.validity.clone() })
    }
}

// â”€â”€ TidyView: fct_encode integration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl TidyView {
    /// Encode a string column in this view into an `FctColumn`.
    ///
    /// Only visible rows (mask) in the current projection are used.
    /// This is a materialising op (allocates u16 buffer) â†’ NOT @nogc safe.
    pub fn fct_encode(&self, col: &str) -> Result<FctColumn, TidyError> {
        FctColumn::encode_from_view(self, col)
    }

    /// Compute per-level mean of a numeric column for use with fct_reorder.
    ///
    /// Returns a Vec<f64> of length `fct.nlevels()`, one mean per level.
    /// Levels with no matching rows get NaN.
    pub fn fct_summary_means(
        &self,
        fct: &FctColumn,
        numeric_col: &str,
    ) -> Result<Vec<f64>, TidyError> {
        let base_idx = self.base.columns.iter()
            .position(|(n, _)| n == numeric_col)
            .ok_or_else(|| TidyError::ColumnNotFound(numeric_col.to_string()))?;
        let nc = &self.base.columns[base_idx].1;
        if nc.len() != fct.nrows() {
            return Err(TidyError::LengthMismatch { expected: fct.nrows(), got: nc.len() });
        }
        // Type-check: only Float or Int supported
        match nc {
            Column::Float(_) | Column::Int(_) => {}
            _ => return Err(TidyError::TypeMismatch {
                expected: "Float or Int".to_string(),
                got: nc.type_name().to_string(),
            }),
        }
        let mut sums = vec![0.0f64; fct.levels.len()];
        let mut counts = vec![0usize; fct.levels.len()];
        match nc {
            Column::Float(v) => {
                for (i, &d) in fct.data.iter().enumerate() {
                    if !v[i].is_nan() {
                        sums[d as usize] += v[i];
                        counts[d as usize] += 1;
                    }
                }
            }
            Column::Int(v) => {
                for (i, &d) in fct.data.iter().enumerate() {
                    sums[d as usize] += v[i] as f64;
                    counts[d as usize] += 1;
                }
            }
            _ => unreachable!(),
        }
        Ok(sums.iter().zip(counts.iter())
            .map(|(&s, &c)| if c == 0 { f64::NAN } else { s / c as f64 })
            .collect())
    }
}

// ── Categorical Encoding Functions ──────────────────────────────────────────

/// Convert a string slice into a categorical encoding with sorted unique levels
/// and integer codes.
///
/// Uses `BTreeSet` for deterministic sorted level discovery.
pub fn label_encode(col: &[String]) -> (Vec<String>, Vec<u32>) {
    let unique: BTreeSet<&str> = col.iter().map(|s| s.as_str()).collect();
    let levels: Vec<String> = unique.into_iter().map(|s| s.to_string()).collect();

    let lookup: BTreeMap<&str, u32> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect();

    let codes: Vec<u32> = col.iter().map(|s| lookup[s.as_str()]).collect();
    (levels, codes)
}

/// Convert a string slice into a categorical encoding with a user-specified
/// level order.
///
/// Returns an error if any value in `col` is not found in `order`.
pub fn ordinal_encode(col: &[String], order: &[String]) -> Result<(Vec<String>, Vec<u32>), String> {
    let lookup: BTreeMap<&str, u32> = order
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect();

    let mut codes = Vec::with_capacity(col.len());
    for s in col {
        match lookup.get(s.as_str()) {
            Some(&idx) => codes.push(idx),
            None => return Err(format!("value {:?} not found in specified order", s)),
        }
    }
    Ok((order.to_vec(), codes))
}

/// One-hot encode a categorical column into multiple boolean columns.
///
/// Returns `(column_names, columns)` where each column is `Vec<bool>` and
/// each row has exactly one `true` across all columns.
pub fn one_hot_encode(levels: &[String], codes: &[u32]) -> (Vec<String>, Vec<Vec<bool>>) {
    let n_levels = levels.len();
    let n_rows = codes.len();

    let mut columns: Vec<Vec<bool>> = vec![vec![false; n_rows]; n_levels];
    for (row, &code) in codes.iter().enumerate() {
        columns[code as usize][row] = true;
    }

    let names: Vec<String> = levels.to_vec();
    (names, columns)
}

// â”€â”€ Phase 17 NoGC audit notes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Safe (@nogc â€” metadata only, O(L) or O(N) over Rust-heap Vec only):
//   fct_collapse    : rewrites levels Vec + remap data Vec â€” both are Rust heap,
//                     no GC heap involved.  SAFE.
//
// NOT safe (materialising, allocates new Rust heap buffers proportional to N or L):
//   fct_encode      : allocates Vec<u16> of length N + Vec<String> levels
//   fct_lump        : allocates new levels Vec + new data Vec
//   fct_reorder     : allocates new levels Vec + new data Vec
//
// Registered in cjc-mir/src/nogc_verify.rs:
//   SAFE:    fct_collapse
//   UNSAFE:  fct_encode, fct_lump, fct_reorder  (intentionally absent)
