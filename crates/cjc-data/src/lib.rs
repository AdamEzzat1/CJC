//! CJC Data DSL â€" Typed expression trees, logical plans, plan optimizer, and
//! column-buffer kernel execution.
//!
//! This implements the tidyverse-inspired data pipeline:
//! ```text
//! df |> filter(col("age") > 18) |> group_by("dept") |> summarize(avg_salary = mean(col("salary")))
//! ```

use cjc_repro::kahan_sum_f64;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::rc::Rc;

mod csv;
pub use csv::{CsvConfig, CsvReader, StreamingCsvProcessor};

pub mod adaptive_selection;
pub mod agg_kernels;
pub mod byte_dict;
pub mod column_meta;
pub mod dataset_plan;
pub mod detcoll;
pub mod dict_encoding;
pub mod lazy;
pub mod predicate_bytecode;
pub mod tidy_dispatch;

pub use adaptive_selection::{AdaptiveSelection, SelectionIndices};
pub use dataset_plan::{
    BatchIterator, BatchSpec, DatasetError, DatasetPlan, EncodingSpec, MaterializedBatch,
    Split, SplitSpec,
};

// â"€â"€ Column Storage â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// A single column in a DataFrame.
#[derive(Debug, Clone)]
pub enum Column {
    /// 64-bit signed integer column.
    Int(Vec<i64>),
    /// 64-bit floating-point column.
    Float(Vec<f64>),
    /// UTF-8 string column.
    Str(Vec<String>),
    /// Boolean column.
    Bool(Vec<bool>),
    /// Categorical column: sorted unique level names + per-row index into levels.
    Categorical {
        levels: Vec<String>,
        codes: Vec<u32>,
    },
    /// Adaptive-width categorical column wrapping Phase 1's
    /// `byte_dict::CategoricalColumn`. Backed by `AdaptiveCodes`
    /// (U8/U16/U32/U64 auto-promoting at 256 / 65 536 / 2³² thresholds)
    /// and a `ByteDictionary` with optional shared/frozen state.
    ///
    /// Coexists with `Column::Categorical` rather than replacing it —
    /// existing column readers continue to use the simpler
    /// `(Vec<String>, Vec<u32>)` storage; new code that needs adaptive
    /// widths or shared dictionaries opts into this variant via
    /// `Column::categorical_adaptive(...)`.
    CategoricalAdaptive(Box<crate::byte_dict::CategoricalColumn>),
    /// DateTime column: epoch milliseconds.
    DateTime(Vec<i64>),
}

impl Column {
    /// Returns the number of rows in this column.
    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Categorical { codes, .. } => codes.len(),
            Column::CategoricalAdaptive(cc) => cc.len(),
            Column::DateTime(v) => v.len(),
        }
    }

    /// Returns `true` if the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the human-readable type name of this column variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Column::Int(_) => "Int",
            Column::Float(_) => "Float",
            Column::Str(_) => "Str",
            Column::Bool(_) => "Bool",
            Column::Categorical { .. } => "Categorical",
            Column::CategoricalAdaptive(_) => "CategoricalAdaptive",
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
            Column::CategoricalAdaptive(cc) => match cc.get(idx) {
                None => String::new(),
                Some(bytes) => String::from_utf8_lossy(bytes).into_owned(),
            },
            Column::DateTime(v) => format!("{}ms", v[idx]),
        }
    }

    /// Construct a `Column::CategoricalAdaptive` from a `CategoricalColumn`.
    /// New-style categorical column with adaptive code widths.
    pub fn categorical_adaptive(cc: crate::byte_dict::CategoricalColumn) -> Self {
        Column::CategoricalAdaptive(Box::new(cc))
    }

    /// Materialize any `Column::CategoricalAdaptive` into a
    /// `Column::Categorical` for consumption by legacy code paths. For
    /// non-adaptive variants this returns `self.clone()`. For adaptive
    /// variants with non-UTF-8 levels or null values this returns
    /// `Column::Str` (display-equivalent) — preserves the consumer's
    /// per-row read semantics while avoiding silent data loss.
    ///
    /// This is the universal back-compat shim for the 19 column-reader
    /// match sites added before the adaptive variant existed. New code
    /// should switch on the variant directly.
    pub fn to_legacy_categorical(&self) -> Column {
        match self {
            Column::CategoricalAdaptive(cc) => {
                // Try lossless conversion to Column::Categorical.
                if let Some(legacy) = Column::from_categorical_column(cc) {
                    return legacy;
                }
                // Fallback: render as Str (handles nulls and non-UTF-8 by
                // using lossy display).
                let n = cc.len();
                let mut out: Vec<String> = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(match cc.get(i) {
                        None => String::new(),
                        Some(b) => String::from_utf8_lossy(b).into_owned(),
                    });
                }
                Column::Str(out)
            }
            _ => self.clone(),
        }
    }

    // ── v3 Phase 4: CategoricalColumn (byte_dict.rs) interop ────────────
    //
    // Limited-scope wiring of Phase 1's adaptive-width categorical engine
    // into the DataFrame surface. The full replacement of
    // `Column::Categorical` with `byte_dict::CategoricalColumn` is
    // deferred to a future phase (every column reader would migrate). For
    // now we expose lossless conversions in both directions so callers
    // that need adaptive code widths or shared/frozen dictionaries can
    // round-trip through the new type.
    //
    // Round-trip is byte-equal: `from_categorical_column(to_categorical_column(c))`
    // produces the same `levels`/`codes` for any `Column::Categorical c`.

    /// Convert a `Column::Categorical` to a `byte_dict::CategoricalColumn`.
    ///
    /// Uses `CategoryOrdering::Explicit` to pin the level→code mapping
    /// exactly as it stands in the source column, so round-tripping back
    /// via `from_categorical_column` yields byte-identical levels and
    /// codes. Returns `None` for non-categorical variants.
    pub fn to_categorical_column(&self) -> Option<crate::byte_dict::CategoricalColumn> {
        use crate::byte_dict::{ByteDictionary, CategoricalColumn};
        match self {
            Column::Categorical { levels, codes } => {
                let explicit: Vec<Vec<u8>> =
                    levels.iter().map(|s| s.as_bytes().to_vec()).collect();
                // `from_explicit` errors only on duplicate levels; the
                // invariant for `Column::Categorical` is unique levels,
                // so any error here indicates upstream corruption.
                let dict = ByteDictionary::from_explicit(explicit).ok()?;
                let mut col = CategoricalColumn::with_dictionary(dict);
                // Push codes via the public surface. `levels` is the
                // authoritative ordering, so we push by `levels[code]`
                // bytes; `Explicit` ordering will assign back the same
                // code value, keeping the codes byte-equal.
                for &c in codes {
                    let bytes = levels[c as usize].as_bytes();
                    // `intern` on an unfrozen Explicit dictionary returns
                    // the existing code for known values.
                    col.push(bytes).ok()?;
                }
                Some(col)
            }
            _ => None,
        }
    }

    /// Build a `Column::Categorical` from a `byte_dict::CategoricalColumn`.
    ///
    /// Iterates the dictionary in code order to reconstruct `levels`, and
    /// the codes via `AdaptiveCodes::iter()` cast to `u32`. Returns
    /// `None` if any level is not valid UTF-8 (the byte dictionary is
    /// byte-keyed; `Column::Categorical` is `String`-keyed) or if
    /// cardinality exceeds `u32::MAX` (would not fit in `Vec<u32>` codes).
    /// Null-bearing categorical columns also return `None` for now —
    /// `Column::Categorical` does not carry a null bitmap.
    pub fn from_categorical_column(
        cat: &crate::byte_dict::CategoricalColumn,
    ) -> Option<Self> {
        if cat.nulls().is_some() {
            return None;
        }
        let dict = cat.dictionary();
        let mut levels: Vec<String> = Vec::with_capacity(dict.len());
        for (_, bytes) in dict.iter() {
            match std::str::from_utf8(bytes) {
                Ok(s) => levels.push(s.to_string()),
                Err(_) => return None,
            }
        }
        let mut codes: Vec<u32> = Vec::with_capacity(cat.len());
        for c in cat.codes().iter() {
            if c > u32::MAX as u64 {
                return None;
            }
            codes.push(c as u32);
        }
        Some(Column::Categorical { levels, codes })
    }
}

// â"€â"€ DataFrame â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// A columnar DataFrame.
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<(String, Column)>,
}

impl DataFrame {
    /// Create an empty DataFrame with no columns.
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Create a DataFrame from a list of named columns.
    ///
    /// Returns an error if column lengths are not all equal.
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

    /// Returns the number of rows (determined from the first column, or 0 if empty).
    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Returns the column names in order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Look up a column by name, returning a reference if found.
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

// â"€â"€ Data DSL Expression Trees â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    /// Named function call: FnCall("log", vec![Col("x")])
    FnCall(String, Vec<DExpr>),
    /// Cumulative sum (window)
    CumSum(Box<DExpr>),
    /// Cumulative product (window)
    CumProd(Box<DExpr>),
    /// Cumulative max (window)
    CumMax(Box<DExpr>),
    /// Cumulative min (window)
    CumMin(Box<DExpr>),
    /// Lag(expr, k): value at row i-k, or NaN if i < k
    Lag(Box<DExpr>, usize),
    /// Lead(expr, k): value at row i+k, or NaN if i+k >= n
    Lead(Box<DExpr>, usize),
    /// Rank of values (1-based, average ties)
    Rank(Box<DExpr>),
    /// Dense rank (1-based, no gaps)
    DenseRank(Box<DExpr>),
    /// Row number (1-indexed sequential)
    RowNumber,
    /// Rolling sum over a fixed-size window (Kahan-compensated removable accumulation)
    RollingSum(String, usize),
    /// Rolling mean over a fixed-size window
    RollingMean(String, usize),
    /// Rolling minimum over a fixed-size window (monotonic deque, O(n) amortized)
    RollingMin(String, usize),
    /// Rolling maximum over a fixed-size window (monotonic deque, O(n) amortized)
    RollingMax(String, usize),
    /// Rolling variance over a fixed-size window (Welford's online with removal)
    RollingVar(String, usize),
    /// Rolling standard deviation over a fixed-size window
    RollingSd(String, usize),
}

/// Binary operator for Data DSL expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DBinOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Greater than (`>`).
    Gt,
    /// Less than (`<`).
    Lt,
    /// Greater than or equal (`>=`).
    Ge,
    /// Less than or equal (`<=`).
    Le,
    /// Equality (`==`).
    Eq,
    /// Not equal (`!=`).
    Ne,
    /// Logical AND (`&&`).
    And,
    /// Logical OR (`||`).
    Or,
}

/// Aggregation function for use in `summarize` expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    /// Kahan-compensated sum.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Row count.
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
            DExpr::FnCall(name, args) => {
                let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
                write!(f, "{}({})", name, args_str.join(", "))
            }
            DExpr::CumSum(e) => write!(f, "cumsum({})", e),
            DExpr::CumProd(e) => write!(f, "cumprod({})", e),
            DExpr::CumMax(e) => write!(f, "cummax({})", e),
            DExpr::CumMin(e) => write!(f, "cummin({})", e),
            DExpr::Lag(e, k) => write!(f, "lag({}, {})", e, k),
            DExpr::Lead(e, k) => write!(f, "lead({}, {})", e, k),
            DExpr::Rank(e) => write!(f, "rank({})", e),
            DExpr::DenseRank(e) => write!(f, "dense_rank({})", e),
            DExpr::RowNumber => write!(f, "row_number()"),
            DExpr::RollingSum(col, w) => write!(f, "rolling_sum(\"{}\", {})", col, w),
            DExpr::RollingMean(col, w) => write!(f, "rolling_mean(\"{}\", {})", col, w),
            DExpr::RollingMin(col, w) => write!(f, "rolling_min(\"{}\", {})", col, w),
            DExpr::RollingMax(col, w) => write!(f, "rolling_max(\"{}\", {})", col, w),
            DExpr::RollingVar(col, w) => write!(f, "rolling_var(\"{}\", {})", col, w),
            DExpr::RollingSd(col, w) => write!(f, "rolling_sd(\"{}\", {})", col, w),
        }
    }
}

// â"€â"€ Logical Plan â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        DExpr::FnCall(_, args) => {
            for arg in args {
                collect_expr_columns(arg, cols);
            }
        }
        DExpr::CumSum(e) | DExpr::CumProd(e) | DExpr::CumMax(e) | DExpr::CumMin(e)
        | DExpr::Lag(e, _) | DExpr::Lead(e, _) | DExpr::Rank(e) | DExpr::DenseRank(e) => {
            collect_expr_columns(e, cols);
        }
        DExpr::RollingSum(col, _) | DExpr::RollingMean(col, _)
        | DExpr::RollingMin(col, _) | DExpr::RollingMax(col, _)
        | DExpr::RollingVar(col, _) | DExpr::RollingSd(col, _) => {
            cols.push(col.clone());
        }
        _ => {}
    }
}

// â"€â"€ Plan Optimizer â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Plan Executor â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    if matches!(col, Column::CategoricalAdaptive(_)) {
        return filter_column(&col.to_legacy_categorical(), mask);
    }
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
        Column::CategoricalAdaptive(_) => unreachable!("handled by early return"),
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

// â"€â"€ Expression Evaluation â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
                Column::CategoricalAdaptive(cc) => Ok(ExprValue::Str(match cc.get(row) {
                    None => String::new(),
                    Some(b) => String::from_utf8_lossy(b).into_owned(),
                })),
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
        DExpr::FnCall(name, args) => {
            if args.len() != 1 {
                return Err(DataError::InvalidOperation(
                    format!("FnCall '{}' requires exactly 1 argument, got {}", name, args.len()),
                ));
            }
            let val = eval_expr_row(df, &args[0], row)?;
            let x = match val {
                ExprValue::Float(f) => f,
                ExprValue::Int(i) => i as f64,
                _ => return Err(DataError::InvalidOperation(
                    format!("FnCall '{}' requires numeric argument", name),
                )),
            };
            let result = match name.as_str() {
                "log" => x.ln(),
                "exp" => x.exp(),
                "sqrt" => x.sqrt(),
                "abs" => x.abs(),
                "ceil" => x.ceil(),
                "floor" => x.floor(),
                "round" => x.round(),
                "sin" => x.sin(),
                "cos" => x.cos(),
                "tan" => x.tan(),
                other => return Err(DataError::InvalidOperation(
                    format!("unknown DExpr function: {}", other),
                )),
            };
            Ok(ExprValue::Float(result))
        }
        DExpr::CumSum(_) | DExpr::CumProd(_) | DExpr::CumMax(_) | DExpr::CumMin(_)
        | DExpr::Lag(_, _) | DExpr::Lead(_, _) | DExpr::Rank(_) | DExpr::DenseRank(_)
        | DExpr::RowNumber
        | DExpr::RollingSum(..) | DExpr::RollingMean(..) | DExpr::RollingMin(..)
        | DExpr::RollingMax(..) | DExpr::RollingVar(..) | DExpr::RollingSd(..) => {
            Err(DataError::InvalidOperation(
                "window function not allowed in row context; use eval_expr_column".into(),
            ))
        }
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

// â"€â"€ Pipeline Builder â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Fluent builder for data pipelines.
pub struct Pipeline {
    plan: LogicalPlan,
}

impl Pipeline {
    /// Start a pipeline by scanning a source DataFrame.
    pub fn scan(df: DataFrame) -> Self {
        Self {
            plan: LogicalPlan::Scan { source: df },
        }
    }

    /// Add a filter step to the pipeline.
    pub fn filter(self, predicate: DExpr) -> Self {
        Self {
            plan: LogicalPlan::Filter {
                input: Box::new(self.plan),
                predicate,
            },
        }
    }

    /// Add a group-by step to the pipeline.
    pub fn group_by(self, keys: Vec<String>) -> Self {
        Self {
            plan: LogicalPlan::GroupBy {
                input: Box::new(self.plan),
                keys,
            },
        }
    }

    /// Add a summarize (aggregate) step to the pipeline.
    pub fn summarize(self, keys: Vec<String>, aggs: Vec<(String, DExpr)>) -> Self {
        Self {
            plan: LogicalPlan::Aggregate {
                input: Box::new(self.plan),
                keys,
                aggs,
            },
        }
    }

    /// Add a column projection step to the pipeline.
    pub fn select(self, columns: Vec<String>) -> Self {
        Self {
            plan: LogicalPlan::Project {
                input: Box::new(self.plan),
                columns,
            },
        }
    }

    /// Add an inner join step to the pipeline.
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

    /// Add a left join step to the pipeline.
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

    /// Add a cross (cartesian) join step to the pipeline.
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

// â"€â"€ Errors â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Errors from DataFrame operations (plan execution, joins, tensor conversion).
#[derive(Debug, Clone)]
pub enum DataError {
    /// A referenced column name does not exist in the DataFrame.
    ColumnNotFound(String),
    /// A column has a different row count than expected.
    ColumnLengthMismatch {
        /// Expected row count.
        expected: usize,
        /// Actual row count.
        got: usize,
        /// Name of the mismatched column.
        column: String,
    },
    /// A generic invalid-operation error with a descriptive message.
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

// â"€â"€ Join Execution â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Get a column value as a string for join key comparison.
fn column_value_str(col: &Column, row: usize) -> String {
    match col {
        Column::Int(v) => v[row].to_string(),
        Column::Float(v) => v[row].to_string(),
        Column::Str(v) => v[row].clone(),
        Column::Bool(v) => v[row].to_string(),
        Column::Categorical { levels, codes } => levels[codes[row] as usize].clone(),
        Column::CategoricalAdaptive(cc) => match cc.get(row) {
            None => String::new(),
            Some(b) => String::from_utf8_lossy(b).into_owned(),
        },
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
    if matches!(col, Column::CategoricalAdaptive(_)) {
        return gather_column(&col.to_legacy_categorical(), indices);
    }
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
        Column::CategoricalAdaptive(_) => unreachable!("handled by early return"),
    }
}

fn gather_column_nullable(col: &Column, indices: &[Option<usize>]) -> Column {
    if matches!(col, Column::CategoricalAdaptive(_)) {
        return gather_column_nullable(&col.to_legacy_categorical(), indices);
    }
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
        Column::CategoricalAdaptive(_) => unreachable!("handled by early return"),
    }
}

// â"€â"€ Tests â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Phase 8: CSV Ingestion & Tensor Bridge â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

// â"€â"€ DataFrame â†" Tensor bridge â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

impl DataFrame {
    /// Convert selected numeric columns to a `cjc_runtime::Tensor` with shape
    /// `[nrows, len(col_names)]` (row-major).
    ///
    /// All selected columns must be `Float` or `Int`; `Str` and `Bool` columns
    /// will return `DataError::InvalidOperation`.
    /// Convert selected numeric columns into a `Tensor` with shape `[nrows, ncols]`.
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
                Column::CategoricalAdaptive(_) => {
                    // CategoricalAdaptive columns are not populated via push_row
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

// â"€â"€ BitMask â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    /// Panics if `nrows` differs â€" this is a programming error (same base df).
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

    /// Returns the total number of rows this mask covers.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of backing u64 words (= ceil(nrows / 64)).
    pub fn nwords(&self) -> usize {
        self.words.len()
    }

    /// Read-only access to the backing words. Used by `AdaptiveSelection`
    /// to perform AND/OR over raw words without re-iterating bit-by-bit.
    pub fn words_slice(&self) -> &[u64] {
        &self.words
    }

    /// Construct a `BitMask` directly from owned words and a row count.
    ///
    /// The caller must ensure tail bits past `nrows` are zero. Used by
    /// `AdaptiveSelection::intersect`/`union` after the AND/OR step.
    pub fn from_words_for_test(words: Vec<u64>, nrows: usize) -> Self {
        debug_assert_eq!(words.len(), nwords_for(nrows));
        BitMask { words, nrows }
    }
}

#[inline]
pub(crate) fn nwords_for(nrows: usize) -> usize {
    (nrows + 63) / 64
}

// â"€â"€ ProjectionMap â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// A stable ordered list of column indices into the base DataFrame.
///
/// Selecting 0 columns yields an empty projection (valid empty view).
/// Duplicate names are rejected at construction time â€" callers must deduplicate.
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

    /// Returns the number of projected columns.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if no columns are projected.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Returns the underlying column-index slice.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

// â"€â"€ TidyView â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// A lazy, zero-allocation view over a base `DataFrame`.
///
/// Holds:
///   â€¢ `base`   â€" shared reference to the underlying columnar data
///   â€¢ `mask`   â€" bitmask of which rows are visible
///   â€¢ `proj`   â€" ordered list of visible column indices
///
/// No column buffers are copied until `materialize()` / `to_tensor()` is called.
#[derive(Debug, Clone)]
pub struct TidyView {
    base: Rc<DataFrame>,
    mask: AdaptiveSelection,
    proj: ProjectionMap,
}

// ── O3: columnar predicate evaluation ────────────────────────────────────────

/// Try to evaluate a filter predicate column-at-a-time for simple `Col op Literal`
/// patterns. Returns `Some(new_mask)` on success, `None` if the predicate shape
/// is unsupported (caller falls back to row-wise evaluation).
///
/// Supported patterns:
///   - `Col op LitFloat` / `LitFloat op Col` (Float columns)
///   - `Col op LitInt`   / `LitInt op Col`   (Int columns, or Float columns with i64→f64 promotion)
///   - `pred And pred` / `pred Or pred`       (compound, recursively tries both sides)
///
/// The returned mask is the AND of the predicate result with `existing_mask`.
/// NaN comparisons follow IEEE 754 (deterministic).
fn try_eval_predicate_columnar(
    base: &DataFrame,
    predicate: &DExpr,
    existing_mask: &BitMask,
) -> Option<BitMask> {
    match predicate {
        // Compound: And — try columnar on both sides, AND the results
        DExpr::BinOp {
            op: DBinOp::And,
            left,
            right,
        } => {
            let lmask = try_eval_predicate_columnar(base, left, existing_mask)?;
            let rmask = try_eval_predicate_columnar(base, right, &lmask)?;
            Some(rmask)
        }
        // Compound: Or — try columnar on both sides, OR the predicate bits
        // then AND with existing mask
        DExpr::BinOp {
            op: DBinOp::Or,
            left,
            right,
        } => {
            // Evaluate each side against a full mask to get raw predicate results,
            // then OR them, then AND with existing mask.
            let all_mask = BitMask::all_true(existing_mask.nrows);
            let lmask = try_eval_predicate_columnar(base, left, &all_mask)?;
            let rmask = try_eval_predicate_columnar(base, right, &all_mask)?;
            // OR the two predicate masks
            let nrows = existing_mask.nrows;
            let or_words: Vec<u64> = lmask
                .words
                .iter()
                .zip(rmask.words.iter())
                .map(|(a, b)| a | b)
                .collect();
            // AND with existing mask
            let final_words: Vec<u64> = or_words
                .iter()
                .zip(existing_mask.words.iter())
                .map(|(a, b)| a & b)
                .collect();
            Some(BitMask {
                words: final_words,
                nrows,
            })
        }
        // Simple comparison: Col op Literal (or Literal op Col)
        DExpr::BinOp { op, left, right } => {
            // Only handle comparison operators
            if !matches!(
                op,
                DBinOp::Gt | DBinOp::Lt | DBinOp::Ge | DBinOp::Le | DBinOp::Eq | DBinOp::Ne
            ) {
                return None;
            }

            // Extract (column_name, literal_value_as_f64_or_i64, is_reversed)
            // "reversed" means Literal op Col, so we flip the comparison direction
            enum LitVal {
                F(f64),
                I(i64),
            }

            let (col_name, lit, reversed) = match (left.as_ref(), right.as_ref()) {
                (DExpr::Col(name), DExpr::LitFloat(v)) => (name.as_str(), LitVal::F(*v), false),
                (DExpr::LitFloat(v), DExpr::Col(name)) => (name.as_str(), LitVal::F(*v), true),
                (DExpr::Col(name), DExpr::LitInt(v)) => (name.as_str(), LitVal::I(*v), false),
                (DExpr::LitInt(v), DExpr::Col(name)) => (name.as_str(), LitVal::I(*v), true),
                _ => return None,
            };

            let column = base.get_column(col_name)?;

            // Flip operator when literal is on the left: `5 > col` becomes `col < 5`
            let effective_op = if reversed {
                match op {
                    DBinOp::Gt => DBinOp::Lt,
                    DBinOp::Lt => DBinOp::Gt,
                    DBinOp::Ge => DBinOp::Le,
                    DBinOp::Le => DBinOp::Ge,
                    other => *other, // Eq, Ne are symmetric
                }
            } else {
                *op
            };

            let nrows = existing_mask.nrows;
            let nwords = nwords_for(nrows);
            let mut words = vec![0u64; nwords];

            match (column, &lit) {
                // Float column, float literal
                (Column::Float(data), LitVal::F(v)) => {
                    columnar_cmp_f64(data, *v, effective_op, &mut words);
                }
                // Float column, int literal (promote i64 → f64)
                (Column::Float(data), LitVal::I(v)) => {
                    columnar_cmp_f64(data, *v as f64, effective_op, &mut words);
                }
                // Int column, int literal
                (Column::Int(data), LitVal::I(v)) => {
                    columnar_cmp_i64(data, *v, effective_op, &mut words);
                }
                // Int column, float literal (promote each i64 → f64)
                (Column::Int(data), LitVal::F(v)) => {
                    // Compare as f64 to match row-wise semantics
                    let floats: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                    columnar_cmp_f64(&floats, *v, effective_op, &mut words);
                }
                _ => return None,
            }

            // AND with existing mask
            for (w, ew) in words.iter_mut().zip(existing_mask.words.iter()) {
                *w &= *ew;
            }

            Some(BitMask { words, nrows })
        }
        _ => None,
    }
}

/// Columnar comparison of f64 slice against a scalar.
/// Sets bits in `out_words` for rows where the comparison is true.
/// NaN follows IEEE 754: NaN != NaN, NaN < x is false, NaN > x is false, etc.
#[inline]
pub(crate) fn columnar_cmp_f64(data: &[f64], lit: f64, op: DBinOp, out_words: &mut [u64]) {
    for (i, &val) in data.iter().enumerate() {
        let pass = match op {
            DBinOp::Gt => val > lit,
            DBinOp::Lt => val < lit,
            DBinOp::Ge => val >= lit,
            DBinOp::Le => val <= lit,
            DBinOp::Eq => val == lit,
            DBinOp::Ne => val != lit,
            _ => false,
        };
        if pass {
            out_words[i / 64] |= 1u64 << (i % 64);
        }
    }
}

/// Columnar comparison of i64 slice against a scalar.
/// Sets bits in `out_words` for rows where the comparison is true.
#[inline]
pub(crate) fn columnar_cmp_i64(data: &[i64], lit: i64, op: DBinOp, out_words: &mut [u64]) {
    for (i, &val) in data.iter().enumerate() {
        let pass = match op {
            DBinOp::Gt => val > lit,
            DBinOp::Lt => val < lit,
            DBinOp::Ge => val >= lit,
            DBinOp::Le => val <= lit,
            DBinOp::Eq => val == lit,
            DBinOp::Ne => val != lit,
            _ => false,
        };
        if pass {
            out_words[i / 64] |= 1u64 << (i % 64);
        }
    }
}

impl TidyView {
    // ── constructors ────────────────────────────────────────────────────

    /// Wrap a `DataFrame` as a full view (all rows, all columns).
    pub fn from_df(df: DataFrame) -> Self {
        let nrows = df.nrows();
        let ncols = df.ncols();
        TidyView {
            base: Rc::new(df),
            mask: AdaptiveSelection::all(nrows),
            proj: ProjectionMap::identity(ncols),
        }
    }

    /// Wrap a shared `Rc<DataFrame>` as a full view.
    pub fn from_rc(df: Rc<DataFrame>) -> Self {
        let nrows = df.nrows();
        let ncols = df.ncols();
        TidyView {
            base: df,
            mask: AdaptiveSelection::all(nrows),
            proj: ProjectionMap::identity(ncols),
        }
    }

    /// Stable identifier of the current selection's adaptive mode. Useful
    /// for tests, instrumentation, and the user-visible `glimpse` output.
    pub fn explain_selection_mode(&self) -> &'static str {
        self.mask.explain_selection_mode()
    }

    // â"€â"€ shape â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    /// Number of visible rows (set bits in mask).
    pub fn nrows(&self) -> usize {
        self.mask.count()
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

    // â"€â"€ filter â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        // v2.1: lower the predicate to bytecode and interpret. Bit-identical
        // output to the legacy AST-walk path on every shape; falls through
        // (None) on unsupported shapes, exactly like try_eval_predicate_columnar.
        if let Some(bc) = predicate_bytecode::PredicateBytecode::lower(predicate, &self.base) {
            let count = self.mask.count();

            // v2.2: when the existing selection has already narrowed the row
            // set substantially, the sparse-aware path beats the full column
            // scan — random-access gather over `iter_indices()` is bounded
            // by O(count) per leaf instead of O(nrows). The dense path needs
            // a full materialized BitMask; the sparse path doesn't, so we
            // defer that allocation until we know it's needed.
            // v3 Phase 5: when the existing selection is already Hybrid
            // (mid-band density, large nrows), evaluate the predicate to
            // a fresh AdaptiveSelection and route through Phase 3's
            // per-chunk dispatch via `existing.intersect(fresh)`. Avoids
            // the O(nrows/64) full-bitmap allocation that the dense
            // BitMask path pays even when the predicate result itself is
            // chunk-sparse.
            //
            // Hybrid activation needs nrows ≥ 8192 and mid-band density
            // — the AdaptiveSelection classifier handles that
            // automatically, so we only need to gate by "existing is
            // Hybrid" to avoid building a fresh AdaptiveSelection
            // pointlessly.
            if matches!(self.mask, AdaptiveSelection::Hybrid { .. })
                && !predicate_bytecode::should_use_sparse_path(count, nrows_base)
            {
                let fresh = bc.evaluate_to_selection(&self.base, nrows_base);
                let intersected = self.mask.intersect(&fresh);
                return Ok(TidyView {
                    base: Rc::clone(&self.base),
                    mask: intersected,
                    proj: self.proj.clone(),
                });
            }

            let new_mask = if predicate_bytecode::should_use_sparse_path(count, nrows_base) {
                let existing_indices: Vec<usize> = self.mask.iter_indices().collect();
                bc.interpret_sparse(&self.base, &existing_indices, nrows_base)
            } else {
                let current_mask = self.mask.materialize_mask();
                bc.interpret(&self.base, &current_mask)
            };

            let words: Vec<u64> = new_mask.words_slice().to_vec();
            return Ok(TidyView {
                base: Rc::clone(&self.base),
                mask: AdaptiveSelection::from_predicate_result(words, nrows_base),
                proj: self.proj.clone(),
            });
        }

        // Materialize once for the legacy columnar path and the row-wise
        // fallback below. (Bytecode handles every shape these support, so
        // we only reach here on unusual predicate trees.)
        let current_mask = self.mask.materialize_mask();

        // Legacy O3 columnar path retained as a safety net (e.g. for the
        // parity oracle in tests/tidy_tests/test_v2_1_bytecode_parity.rs).
        // In production, bytecode handles every shape this path handles, so
        // execution rarely reaches here.
        if let Some(new_mask) = try_eval_predicate_columnar(&self.base, predicate, &current_mask) {
            let words: Vec<u64> = new_mask.words_slice().to_vec();
            return Ok(TidyView {
                base: Rc::clone(&self.base),
                mask: AdaptiveSelection::from_predicate_result(words, nrows_base),
                proj: self.proj.clone(),
            });
        }

        // Fallback: row-wise evaluation
        let mut new_words: Vec<u64> = current_mask.words_slice().to_vec();

        // Evaluate predicate over every currently-masked-in row.
        // Rows masked out remain 0 (no change needed, AND semantics).
        for row in self.mask.iter_indices() {
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
            mask: AdaptiveSelection::from_predicate_result(new_words, nrows_base),
            proj: self.proj.clone(),
        })
    }

    // â"€â"€ select â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ mutate â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    /// Apply column-wise assignments and return a materialized `TidyFrame`.
    ///
    /// `assignments` is an ordered list of `(col_name, expr)` pairs evaluated
    /// left-to-right. Each assignment sees the *snapshot* of columns at entry
    /// to the mutate call (snapshot semantics â€" new columns created in earlier
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

    // â"€â"€ materialize â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        let row_indices: Vec<usize> = self.mask.iter_indices().collect();

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

    /// Access a materialized `BitMask` view of the current selection (for
    /// testing/inspection). Always returns an owned `BitMask` regardless of
    /// the underlying adaptive arm — this preserves the pre-v2 inspection
    /// surface without coupling test code to the adaptive enum.
    pub fn mask(&self) -> BitMask {
        self.mask.materialize_mask()
    }

    /// Access the underlying adaptive selection (for testing/inspection
    /// of the chosen mode). Use `mask()` if you want a materialized
    /// `BitMask` view.
    pub fn selection(&self) -> &AdaptiveSelection {
        &self.mask
    }

    /// Access the underlying projection (for testing/inspection).
    pub fn proj(&self) -> &ProjectionMap {
        &self.proj
    }

    /// Access a column from the underlying base DataFrame by name.
    ///
    /// Returns the raw `Column` (full length, unmasked) â€" callers must apply
    /// the mask themselves if needed.  Used by `fct_summary_means` and similar.
    pub fn base_column(&self, name: &str) -> Option<&Column> {
        self.base.columns.iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }
}

// â"€â"€ TidyFrame â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ TidyError â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Internal helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

// Note: tidy uses the existing `gather_column(col, indices)` defined earlier in
// this file (line ~1160). No duplicate needed.

/// Evaluate a `DExpr` for all `nrows` rows, returning a typed `Column`.
///
/// Int + Float â†’ Float promotion.
/// Int overflow â†’ wrapping (i64 wrapping_add/mul etc.).
/// Scalar expression â†’ broadcast to all rows.
/// Extract f64 values for all rows from a sub-expression.
fn extract_f64_column(df: &DataFrame, expr: &DExpr, nrows: usize) -> Result<Vec<f64>, TidyError> {
    let col = eval_expr_column(df, expr, nrows)?;
    match col {
        Column::Float(v) => Ok(v),
        Column::Int(v) => Ok(v.into_iter().map(|i| i as f64).collect()),
        _ => Err(TidyError::TypeMismatch {
            expected: "numeric".into(),
            got: "non-numeric".into(),
        }),
    }
}

/// Evaluate window DExpr variants that need full-column context.
/// Returns `Ok(Some(column))` if expr is a window function, `Ok(None)` otherwise.
fn eval_window_column(
    df: &DataFrame,
    expr: &DExpr,
    nrows: usize,
) -> Result<Option<Column>, TidyError> {
    match expr {
        DExpr::RowNumber => {
            let vals: Vec<i64> = (1..=nrows as i64).collect();
            Ok(Some(Column::Int(vals)))
        }
        DExpr::CumSum(inner) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut result = Vec::with_capacity(nrows);
            let mut sum = 0.0_f64;
            let mut comp = 0.0_f64; // Kahan compensation
            for &v in &src {
                let y = v - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                result.push(sum);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::CumProd(inner) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut result = Vec::with_capacity(nrows);
            let mut prod = 1.0_f64;
            for &v in &src {
                prod *= v;
                result.push(prod);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::CumMax(inner) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut result = Vec::with_capacity(nrows);
            let mut max = f64::NEG_INFINITY;
            for &v in &src {
                if v > max { max = v; }
                result.push(max);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::CumMin(inner) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut result = Vec::with_capacity(nrows);
            let mut min = f64::INFINITY;
            for &v in &src {
                if v < min { min = v; }
                result.push(min);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::Lag(inner, k) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut result = Vec::with_capacity(nrows);
            for i in 0..nrows {
                if i < *k {
                    result.push(f64::NAN);
                } else {
                    result.push(src[i - k]);
                }
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::Lead(inner, k) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut result = Vec::with_capacity(nrows);
            for i in 0..nrows {
                if i + k >= nrows {
                    result.push(f64::NAN);
                } else {
                    result.push(src[i + k]);
                }
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::Rank(inner) => {
            let src = extract_f64_column(df, inner, nrows)?;
            // Average rank (1-based): sort indices, assign ranks, average ties
            let mut indexed: Vec<(usize, f64)> = src.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut ranks = vec![0.0_f64; nrows];
            let mut i = 0;
            while i < nrows {
                let mut j = i;
                while j < nrows && indexed[j].1 == indexed[i].1 {
                    j += 1;
                }
                let avg_rank = (i + 1 + j) as f64 / 2.0; // 1-based average
                for idx in i..j {
                    ranks[indexed[idx].0] = avg_rank;
                }
                i = j;
            }
            Ok(Some(Column::Float(ranks)))
        }
        DExpr::DenseRank(inner) => {
            let src = extract_f64_column(df, inner, nrows)?;
            let mut indexed: Vec<(usize, f64)> = src.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut ranks = vec![0_i64; nrows];
            let mut rank = 0_i64;
            let mut prev: Option<f64> = None;
            for &(orig_idx, val) in &indexed {
                if prev.is_none() || prev.unwrap() != val {
                    rank += 1;
                }
                ranks[orig_idx] = rank;
                prev = Some(val);
            }
            Ok(Some(Column::Int(ranks)))
        }
        DExpr::RollingSum(col_name, window) => {
            let vals = rolling_get_floats(df, col_name)?;
            let n = vals.len();
            let w = *window;
            let mut result = Vec::with_capacity(n);
            let mut sum = 0.0_f64;
            let mut comp = 0.0_f64;
            for i in 0..n {
                // Kahan add entering element
                let y = vals[i] - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                // Remove leaving element if window is full
                if i >= w {
                    let y2 = -vals[i - w] - comp;
                    let t2 = sum + y2;
                    comp = (t2 - sum) - y2;
                    sum = t2;
                }
                result.push(sum);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::RollingMean(col_name, window) => {
            let vals = rolling_get_floats(df, col_name)?;
            let n = vals.len();
            let w = *window;
            let mut result = Vec::with_capacity(n);
            let mut sum = 0.0_f64;
            let mut comp = 0.0_f64;
            for i in 0..n {
                let y = vals[i] - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
                if i >= w {
                    let y2 = -vals[i - w] - comp;
                    let t2 = sum + y2;
                    comp = (t2 - sum) - y2;
                    sum = t2;
                }
                let count = if i < w { i + 1 } else { w };
                result.push(sum / count as f64);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::RollingMin(col_name, window) => {
            let vals = rolling_get_floats(df, col_name)?;
            let n = vals.len();
            let w = *window;
            let mut result = Vec::with_capacity(n);
            let mut deque: VecDeque<usize> = VecDeque::new();
            for i in 0..n {
                // Remove elements outside window
                while !deque.is_empty() && *deque.front().unwrap() + w <= i {
                    deque.pop_front();
                }
                // Remove elements >= current (maintain increasing monotonic deque)
                while !deque.is_empty() && vals[*deque.back().unwrap()] >= vals[i] {
                    deque.pop_back();
                }
                deque.push_back(i);
                result.push(vals[*deque.front().unwrap()]);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::RollingMax(col_name, window) => {
            let vals = rolling_get_floats(df, col_name)?;
            let n = vals.len();
            let w = *window;
            let mut result = Vec::with_capacity(n);
            let mut deque: VecDeque<usize> = VecDeque::new();
            for i in 0..n {
                while !deque.is_empty() && *deque.front().unwrap() + w <= i {
                    deque.pop_front();
                }
                // Remove elements <= current (maintain decreasing monotonic deque)
                while !deque.is_empty() && vals[*deque.back().unwrap()] <= vals[i] {
                    deque.pop_back();
                }
                deque.push_back(i);
                result.push(vals[*deque.front().unwrap()]);
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::RollingVar(col_name, window) => {
            let vals = rolling_get_floats(df, col_name)?;
            let n = vals.len();
            let w = *window;
            let mut result = Vec::with_capacity(n);
            // Welford's online algorithm with removal
            let mut count = 0_usize;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            for i in 0..n {
                // Add entering element
                count += 1;
                let delta = vals[i] - mean;
                mean += delta / count as f64;
                let delta2 = vals[i] - mean;
                m2 += delta * delta2;
                // Remove leaving element if window is full
                if i >= w {
                    let old = vals[i - w];
                    count -= 1;
                    if count == 0 {
                        mean = 0.0;
                        m2 = 0.0;
                    } else {
                        let delta_old = old - mean;
                        mean -= delta_old / count as f64;
                        let delta_old2 = old - mean;
                        m2 -= delta_old * delta_old2;
                    }
                }
                if count < 2 {
                    result.push(0.0);
                } else {
                    // Population variance (not sample): m2 / count
                    // Use sample variance (Bessel's correction): m2 / (count - 1)
                    result.push(m2 / (count - 1) as f64);
                }
            }
            Ok(Some(Column::Float(result)))
        }
        DExpr::RollingSd(col_name, window) => {
            let vals = rolling_get_floats(df, col_name)?;
            let n = vals.len();
            let w = *window;
            let mut result = Vec::with_capacity(n);
            let mut count = 0_usize;
            let mut mean = 0.0_f64;
            let mut m2 = 0.0_f64;
            for i in 0..n {
                count += 1;
                let delta = vals[i] - mean;
                mean += delta / count as f64;
                let delta2 = vals[i] - mean;
                m2 += delta * delta2;
                if i >= w {
                    let old = vals[i - w];
                    count -= 1;
                    if count == 0 {
                        mean = 0.0;
                        m2 = 0.0;
                    } else {
                        let delta_old = old - mean;
                        mean -= delta_old / count as f64;
                        let delta_old2 = old - mean;
                        m2 -= delta_old * delta_old2;
                    }
                }
                if count < 2 {
                    result.push(0.0);
                } else {
                    result.push((m2 / (count - 1) as f64).sqrt());
                }
            }
            Ok(Some(Column::Float(result)))
        }
        _ => Ok(None),
    }
}

/// Extract a float column from a DataFrame by name (for rolling window functions).
fn rolling_get_floats(df: &DataFrame, col_name: &str) -> Result<Vec<f64>, TidyError> {
    let col = df
        .get_column(col_name)
        .ok_or_else(|| TidyError::ColumnNotFound(col_name.to_string()))?;
    match col {
        Column::Float(v) => Ok(v.clone()),
        Column::Int(v) => Ok(v.iter().map(|&i| i as f64).collect()),
        _ => Err(TidyError::TypeMismatch {
            expected: "numeric".into(),
            got: "non-numeric".into(),
        }),
    }
}

// -- O7: Vectorized column-level DExpr evaluation --------------------------------

/// Apply a binary operation element-wise on two columns.
/// Mirrors the semantics of `eval_binop` exactly for bit-identical results.
fn vectorized_binop(op: DBinOp, left: &Column, right: &Column) -> Result<Column, TidyError> {
    match (left, right) {
        (Column::Int(a), Column::Int(b)) => {
            let n = a.len();
            match op {
                DBinOp::Add => { let mut r = vec![0i64; n]; for i in 0..n { r[i] = a[i] + b[i]; } Ok(Column::Int(r)) }
                DBinOp::Sub => { let mut r = vec![0i64; n]; for i in 0..n { r[i] = a[i] - b[i]; } Ok(Column::Int(r)) }
                DBinOp::Mul => { let mut r = vec![0i64; n]; for i in 0..n { r[i] = a[i] * b[i]; } Ok(Column::Int(r)) }
                DBinOp::Div => { let mut r = vec![0i64; n]; for i in 0..n { r[i] = a[i] / b[i]; } Ok(Column::Int(r)) }
                DBinOp::Gt => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] > b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Lt => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] < b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Ge => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] >= b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Le => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] <= b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Eq => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] == b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Ne => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] != b[i]; } Ok(Column::Bool(r)) }
                _ => Err(TidyError::Internal(format!("unsupported op {:?} on Int", op))),
            }
        }
        (Column::Float(a), Column::Float(b)) => {
            let n = a.len();
            match op {
                DBinOp::Add => { let mut r = vec![0.0f64; n]; for i in 0..n { r[i] = a[i] + b[i]; } Ok(Column::Float(r)) }
                DBinOp::Sub => { let mut r = vec![0.0f64; n]; for i in 0..n { r[i] = a[i] - b[i]; } Ok(Column::Float(r)) }
                DBinOp::Mul => { let mut r = vec![0.0f64; n]; for i in 0..n { r[i] = a[i] * b[i]; } Ok(Column::Float(r)) }
                DBinOp::Div => { let mut r = vec![0.0f64; n]; for i in 0..n { r[i] = a[i] / b[i]; } Ok(Column::Float(r)) }
                DBinOp::Gt => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] > b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Lt => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] < b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Ge => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] >= b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Le => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] <= b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Eq => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] == b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Ne => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] != b[i]; } Ok(Column::Bool(r)) }
                _ => Err(TidyError::Internal(format!("unsupported op {:?} on Float", op))),
            }
        }
        (Column::Int(a), Column::Float(_b)) => {
            let promoted: Vec<f64> = a.iter().map(|&v| v as f64).collect();
            vectorized_binop(op, &Column::Float(promoted), right)
        }
        (Column::Float(_a), Column::Int(b)) => {
            let promoted: Vec<f64> = b.iter().map(|&v| v as f64).collect();
            vectorized_binop(op, left, &Column::Float(promoted))
        }
        (Column::Bool(a), Column::Bool(b)) => {
            let n = a.len();
            match op {
                DBinOp::And => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] && b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Or  => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] || b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Eq  => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] == b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Ne  => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] != b[i]; } Ok(Column::Bool(r)) }
                _ => Err(TidyError::Internal(format!("unsupported op {:?} on Bool", op))),
            }
        }
        (Column::Str(a), Column::Str(b)) => {
            let n = a.len();
            match op {
                DBinOp::Eq => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] == b[i]; } Ok(Column::Bool(r)) }
                DBinOp::Ne => { let mut r = vec![false; n]; for i in 0..n { r[i] = a[i] != b[i]; } Ok(Column::Bool(r)) }
                _ => Err(TidyError::Internal(format!("unsupported op {:?} on String", op))),
            }
        }
        _ => Err(TidyError::Internal("type mismatch in binary operation".into())),
    }
}

/// Apply a unary math function element-wise to a column.
/// Mirrors the semantics of the FnCall arm in `eval_expr_row` exactly.
fn vectorized_fn_call(name: &str, arg: &Column) -> Result<Column, TidyError> {
    let floats: Vec<f64> = match arg {
        Column::Float(v) => v.clone(),
        Column::Int(v) => v.iter().map(|&i| i as f64).collect(),
        _ => return Err(TidyError::Internal(format!(
            "FnCall '{}' requires numeric argument", name
        ))),
    };
    let f: fn(f64) -> f64 = match name {
        "log"   => f64::ln,
        "exp"   => f64::exp,
        "sqrt"  => f64::sqrt,
        "abs"   => f64::abs,
        "ceil"  => f64::ceil,
        "floor" => f64::floor,
        "round" => f64::round,
        "sin"   => f64::sin,
        "cos"   => f64::cos,
        "tan"   => f64::tan,
        _ => return Err(TidyError::Internal(format!(
            "unknown DExpr function: {}", name
        ))),
    };
    let mut result = vec![0.0f64; floats.len()];
    for i in 0..floats.len() {
        result[i] = f(floats[i]);
    }
    Ok(Column::Float(result))
}

/// O7: Try vectorized column-level evaluation of a DExpr.
/// Returns `None` if the expression is too complex for the fast path,
/// causing the caller to fall back to the row-by-row evaluator.
fn try_eval_expr_column_vectorized(
    df: &DataFrame,
    expr: &DExpr,
    nrows: usize,
) -> Option<Result<Column, TidyError>> {
    match expr {
        DExpr::Col(name) => {
            let col = df.get_column(name)?;
            let result = match col {
                Column::Int(v) => Column::Int(v[..nrows].to_vec()),
                Column::Float(v) => Column::Float(v[..nrows].to_vec()),
                Column::Str(v) => Column::Str(v[..nrows].to_vec()),
                Column::Bool(v) => Column::Bool(v[..nrows].to_vec()),
                Column::Categorical { levels, codes } => {
                    let strs: Vec<String> = codes[..nrows]
                        .iter()
                        .map(|&c| levels[c as usize].clone())
                        .collect();
                    Column::Str(strs)
                }
                Column::CategoricalAdaptive(cc) => {
                    let strs: Vec<String> = (0..nrows)
                        .map(|i| match cc.get(i) {
                            None => String::new(),
                            Some(b) => String::from_utf8_lossy(b).into_owned(),
                        })
                        .collect();
                    Column::Str(strs)
                }
                Column::DateTime(v) => Column::Int(v[..nrows].to_vec()),
            };
            Some(Ok(result))
        }
        DExpr::LitFloat(v) => Some(Ok(Column::Float(vec![*v; nrows]))),
        DExpr::LitInt(v) => Some(Ok(Column::Int(vec![*v; nrows]))),
        DExpr::LitBool(b) => Some(Ok(Column::Bool(vec![*b; nrows]))),
        DExpr::LitStr(s) => Some(Ok(Column::Str(vec![s.clone(); nrows]))),
        DExpr::BinOp { op, left, right } => {
            let left_col = try_eval_expr_column_vectorized(df, left, nrows)?.ok()?;
            let right_col = try_eval_expr_column_vectorized(df, right, nrows)?.ok()?;
            Some(vectorized_binop(*op, &left_col, &right_col))
        }
        DExpr::FnCall(name, args) if args.len() == 1 => {
            let arg_col = try_eval_expr_column_vectorized(df, &args[0], nrows)?.ok()?;
            Some(vectorized_fn_call(name, &arg_col))
        }
        _ => None,
    }
}

fn eval_expr_column(df: &DataFrame, expr: &DExpr, nrows: usize) -> Result<Column, TidyError> {
    if nrows == 0 {
        // Infer column type from a dry-run on nothing; default to Float for empty
        return Ok(Column::Float(vec![]));
    }

    // v3 Phase 6: cat-aware mutate. When the expression is a bare `Col(name)`
    // referring to a Categorical or CategoricalAdaptive column, return that
    // column verbatim. Pre-Phase-6 the row-by-row fallback materialized a
    // `Vec<String>` (one per row) and built a `Column::Str` — this loses
    // the level table and forces a downstream re-categorization to recover
    // it. Pass-through preserves both type and (for Adaptive) the
    // dictionary's frozen / shared / sealed state.
    if let DExpr::Col(name) = expr {
        if let Some(src) = df.get_column(name) {
            match src {
                Column::Categorical { .. } | Column::CategoricalAdaptive(_) => {
                    return Ok(src.clone());
                }
                _ => {}
            }
        }
    }

    // Handle window functions at column level
    if let Some(col) = eval_window_column(df, expr, nrows)? {
        return Ok(col);
    }

    // O7: try vectorized fast path before falling back to row-by-row
    if let Some(result) = try_eval_expr_column_vectorized(df, expr, nrows) {
        return result;
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
/// columns â€" filter predicates may reference any base column visible in proj).
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

// â"€â"€ DataFrame::tidy() convenience entry point â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

impl DataFrame {
    /// Wrap this DataFrame as a `TidyView` for Phase 10 tidy operations.
    ///
    /// Consumes `self` (zero-copy â€" moves into an `Rc`).
    pub fn tidy(self) -> TidyView {
        TidyView::from_df(self)
    }

    /// Wrap this DataFrame as a `TidyFrame` for mutable tidy operations.
    pub fn tidy_mut(self) -> TidyFrame {
        TidyFrame::from_df(self)
    }
}

// â"€â"€ NoGC annotation gate â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
//
// The @nogc verifier in cjc-mir/src/nogc_verify.rs tracks "safe builtins" that
// are known not to trigger GC. Phase 10 tidy operations on the Rust side are
// outside the CJC language runtime (they are library calls), so @nogc
// annotation at the CJC language level means: the CJC function body does not
// call gc_alloc. The Rust implementation of filter/select produces a TidyView
// that holds Rc references â€" no GC heap involvement.
//
// For the NoGC verifier to accept tidy calls inside @nogc CJC functions, the
// builtins "tidy_filter", "tidy_select", "tidy_materialize" must be added to
// the safe-builtins list in cjc-mir/src/nogc_verify.rs. See that file for
// the `is_safe_builtin` function.
//
// Allocation budget per operation (no GC heap, only Rust stack/heap via alloc):
//   filter  : O(N/64) u64 words for new mask   (â‰ˆ 8 bytes / 64 rows)
//   select  : O(K) usize indices (K = ncols selected)
//   mutate  : O(N) per new column buffer (allowed â€" one allocation per column)
//   materialize: O(N * K) total for visible data

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Phase 11â€"12: Grouping, Summarise, Arrange, Slice, Distinct, Joins
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//
// Spec-Lock Table
// â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
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
// | null ordering             | N/A â€" no null type in CJC                       |
// | slice_sample seed         | Deterministic LCG with caller-supplied u64 seed  |
// | slice_sample n > nrows    | Clamp to nrows (no error)                        |
// | distinct ordering         | First-occurrence order of distinct key combos    |
// | Join left order           | Preserved â€" output rows follow left row order    |
// | Join right match order    | Stable: sorted by right-side row index ascending  |
// | Null matching in joins    | N/A â€" no null type in CJC                       |
// | Join duplicate keys       | All matches included, deterministic order        |
// | many-many explosion order | Left outer loop, right inner loop (stable)       |
// â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

// â"€â"€ RowIndexMap â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    /// Create a new row index map from explicit indices.
    pub fn new(indices: Vec<usize>) -> Self {
        RowIndexMap { indices }
    }

    /// Returns the number of visible rows.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if no rows are selected.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Borrow the underlying indices as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.indices
    }
}

// â"€â"€ GroupMeta â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Metadata for one group in a `GroupIndex`.
#[derive(Debug, Clone)]
pub struct GroupMeta {
    /// The rendered key strings (one per grouping column), in key order.
    pub key_values: Vec<String>,
    /// Base-frame row indices belonging to this group, in first-occurrence order.
    pub row_indices: Vec<usize>,
}

// â"€â"€ GroupIndex â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

            // Linear scan for existing key â€" preserves insertion order, no hash
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

// â"€â"€ GroupedTidyView â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// A grouped view produced by `TidyView::group_by(...)`.
///
/// Holds the original `TidyView` (unchanged) plus a `GroupIndex` over its
/// visible rows. No column data is copied.
///
/// After grouping, `ungroup()` restores the plain `TidyView`.
/// `summarise()` collapses each group into one summary row.
#[derive(Debug, Clone)]
/// A TidyView that has been grouped by one or more columns.
///
/// Created by [`TidyView::group_by`]. Holds the original view plus a
/// [`GroupIndex`] that maps rows to groups. Call [`summarise`](Self::summarise)
/// to aggregate or [`ungroup`](Self::ungroup) to return to a flat view.
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

    // â"€â"€ summarise â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        // Build aggregator columns (O5+O9: use fast path)
        for &(out_name, ref agg) in assignments {
            let col_vals = self.eval_agg_over_groups_fast(agg, n_groups, base)?;
            result_columns.push((out_name.to_string(), col_vals));
        }

        let df = DataFrame::from_columns(result_columns)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }

    /// Evaluate an aggregator over all groups, return a typed `Column`.
    #[allow(dead_code)]
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
            | TidyAgg::First(col_name) | TidyAgg::Last(col_name)
            | TidyAgg::Median(col_name) | TidyAgg::Sd(col_name)
            | TidyAgg::Var(col_name) | TidyAgg::Quantile(col_name, _)
            | TidyAgg::NDistinct(col_name) | TidyAgg::Iqr(col_name) => {
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

    /// O5+O9: Fast aggregation using direct index iteration and arena buffer.
    /// Produces bit-identical results to `eval_agg_over_groups`.
    fn eval_agg_over_groups_fast(
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
            TidyAgg::Sum(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_sum(&self.index.groups, src)?))
            }
            TidyAgg::Mean(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_mean(&self.index.groups, src)?))
            }
            TidyAgg::Min(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_min(&self.index.groups, src)?))
            }
            TidyAgg::Max(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_max(&self.index.groups, src)?))
            }
            TidyAgg::First(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_first(&self.index.groups, src)?))
            }
            TidyAgg::Last(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_last(&self.index.groups, src)?))
            }
            TidyAgg::Var(col_name)
            | TidyAgg::Sd(col_name)
            | TidyAgg::Median(col_name)
            | TidyAgg::Quantile(col_name, _)
            | TidyAgg::NDistinct(col_name)
            | TidyAgg::Iqr(col_name) => {
                let src = base.get_column(col_name)
                    .ok_or_else(|| TidyError::ColumnNotFound(col_name.clone()))?;
                Ok(Column::Float(fast_agg_arena(
                    agg, &self.index.groups, src, n_groups,
                )?))
            }
        }
    }
}

// -- O5: Direct index-based aggregation (no per-group Vec) --------------------

enum ColRef<'a> {
    Float(&'a [f64]),
    Int(&'a [i64]),
}

fn col_to_ref(col: &Column) -> Result<ColRef<'_>, TidyError> {
    match col {
        Column::Float(v) => Ok(ColRef::Float(v)),
        Column::Int(v) => Ok(ColRef::Int(v)),
        _ => Err(TidyError::TypeMismatch {
            expected: "numeric (Int or Float)".into(),
            got: col.type_name().into(),
        }),
    }
}

fn fast_agg_sum(groups: &[GroupMeta], col: &Column) -> Result<Vec<f64>, TidyError> {
    use cjc_repro::kahan::KahanAccumulatorF64;
    let cr = col_to_ref(col)?;
    Ok(groups.iter().map(|g| {
        let mut acc = KahanAccumulatorF64::new();
        match cr {
            ColRef::Float(d) => { for &i in &g.row_indices { acc.add(d[i]); } }
            ColRef::Int(d) => { for &i in &g.row_indices { acc.add(d[i] as f64); } }
        }
        acc.finalize()
    }).collect())
}

fn fast_agg_mean(groups: &[GroupMeta], col: &Column) -> Result<Vec<f64>, TidyError> {
    use cjc_repro::kahan::KahanAccumulatorF64;
    let cr = col_to_ref(col)?;
    Ok(groups.iter().map(|g| {
        if g.row_indices.is_empty() { return f64::NAN; }
        let mut acc = KahanAccumulatorF64::new();
        match cr {
            ColRef::Float(d) => { for &i in &g.row_indices { acc.add(d[i]); } }
            ColRef::Int(d) => { for &i in &g.row_indices { acc.add(d[i] as f64); } }
        }
        acc.finalize() / g.row_indices.len() as f64
    }).collect())
}

fn fast_agg_min(groups: &[GroupMeta], col: &Column) -> Result<Vec<f64>, TidyError> {
    let cr = col_to_ref(col)?;
    Ok(groups.iter().map(|g| {
        if g.row_indices.is_empty() { return f64::NAN; }
        match cr {
            ColRef::Float(d) => g.row_indices.iter().fold(f64::INFINITY, |a, &i| {
                let b = d[i]; if b.is_nan() || b < a { b } else { a }
            }),
            ColRef::Int(d) => g.row_indices.iter().fold(f64::INFINITY, |a, &i| {
                let b = d[i] as f64; if b.is_nan() || b < a { b } else { a }
            }),
        }
    }).collect())
}

fn fast_agg_max(groups: &[GroupMeta], col: &Column) -> Result<Vec<f64>, TidyError> {
    let cr = col_to_ref(col)?;
    Ok(groups.iter().map(|g| {
        if g.row_indices.is_empty() { return f64::NAN; }
        match cr {
            ColRef::Float(d) => g.row_indices.iter().fold(f64::NEG_INFINITY, |a, &i| {
                let b = d[i]; if b.is_nan() || b > a { b } else { a }
            }),
            ColRef::Int(d) => g.row_indices.iter().fold(f64::NEG_INFINITY, |a, &i| {
                let b = d[i] as f64; if b.is_nan() || b > a { b } else { a }
            }),
        }
    }).collect())
}

fn fast_agg_first(groups: &[GroupMeta], col: &Column) -> Result<Vec<f64>, TidyError> {
    let cr = col_to_ref(col)?;
    let mut vals = Vec::with_capacity(groups.len());
    for g in groups {
        if g.row_indices.is_empty() { return Err(TidyError::EmptyGroup); }
        match cr {
            ColRef::Float(d) => vals.push(d[g.row_indices[0]]),
            ColRef::Int(d) => vals.push(d[g.row_indices[0]] as f64),
        }
    }
    Ok(vals)
}

fn fast_agg_last(groups: &[GroupMeta], col: &Column) -> Result<Vec<f64>, TidyError> {
    let cr = col_to_ref(col)?;
    let mut vals = Vec::with_capacity(groups.len());
    for g in groups {
        if g.row_indices.is_empty() { return Err(TidyError::EmptyGroup); }
        let last = *g.row_indices.last().unwrap();
        match cr {
            ColRef::Float(d) => vals.push(d[last]),
            ColRef::Int(d) => vals.push(d[last] as f64),
        }
    }
    Ok(vals)
}

// -- O9: Arena-based aggregation for sort-dependent ops -----------------------

fn fast_agg_arena(
    agg: &TidyAgg,
    groups: &[GroupMeta],
    col: &Column,
    n_groups: usize,
) -> Result<Vec<f64>, TidyError> {
    let cr = col_to_ref(col)?;
    let max_size = groups.iter().map(|g| g.row_indices.len()).max().unwrap_or(0);
    let mut arena: Vec<f64> = Vec::with_capacity(max_size);
    let mut results = Vec::with_capacity(n_groups);
    for group in groups {
        arena.clear();
        match cr {
            ColRef::Float(d) => { for &i in &group.row_indices { arena.push(d[i]); } }
            ColRef::Int(d) => { for &i in &group.row_indices { arena.push(d[i] as f64); } }
        }
        let val = agg_reduce_slice(agg, &mut arena)?;
        results.push(val);
    }
    Ok(results)
}

/// Reduce a pre-gathered f64 slice for sort-dependent aggregators (O9).
/// Bit-identical to `agg_reduce`.
fn agg_reduce_slice(agg: &TidyAgg, values: &mut [f64]) -> Result<f64, TidyError> {
    match agg {
        TidyAgg::Var(_) => {
            if values.len() < 2 {
                Ok(f64::NAN)
            } else {
                let n = values.len() as f64;
                let mean = kahan_sum_f64(values) / n;
                let sq_diffs: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
                Ok(kahan_sum_f64(&sq_diffs) / (n - 1.0))
            }
        }
        TidyAgg::Sd(_) => {
            if values.len() < 2 {
                Ok(f64::NAN)
            } else {
                let n = values.len() as f64;
                let mean = kahan_sum_f64(values) / n;
                let sq_diffs: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
                Ok((kahan_sum_f64(&sq_diffs) / (n - 1.0)).sqrt())
            }
        }
        TidyAgg::Median(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = values.len();
                if n % 2 == 1 { Ok(values[n / 2]) }
                else { Ok((values[n / 2 - 1] + values[n / 2]) / 2.0) }
            }
        }
        TidyAgg::Quantile(_, p) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = values.len();
                if n == 1 { return Ok(values[0]); }
                let pos = p * (n as f64 - 1.0);
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                if lo == hi || hi >= n { Ok(values[lo.min(n - 1)]) }
                else { Ok(values[lo] + frac * (values[hi] - values[lo])) }
            }
        }
        TidyAgg::NDistinct(_) => {
            let distinct: std::collections::BTreeSet<u64> = values.iter().map(|v| v.to_bits()).collect();
            Ok(distinct.len() as f64)
        }
        TidyAgg::Iqr(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = values.len();
                if n == 1 { return Ok(0.0); }
                let q1 = {
                    let pos = 0.25 * (n as f64 - 1.0);
                    let lo = pos.floor() as usize;
                    let hi = pos.ceil() as usize;
                    let frac = pos - lo as f64;
                    if lo == hi || hi >= n { values[lo.min(n - 1)] }
                    else { values[lo] + frac * (values[hi] - values[lo]) }
                };
                let q3 = {
                    let pos = 0.75 * (n as f64 - 1.0);
                    let lo = pos.floor() as usize;
                    let hi = pos.ceil() as usize;
                    let frac = pos - lo as f64;
                    if lo == hi || hi >= n { values[lo.min(n - 1)] }
                    else { values[lo] + frac * (values[hi] - values[lo]) }
                };
                Ok(q3 - q1)
            }
        }
        _ => unreachable!("agg_reduce_slice called for non-arena aggregator"),
    }
}

/// Reduce one group's rows for a numeric aggregator. Returns f64.
#[allow(dead_code)]
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
        TidyAgg::Median(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n % 2 == 1 {
                    Ok(sorted[n / 2])
                } else {
                    Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
                }
            }
        }
        TidyAgg::Var(_) => {
            if values.len() < 2 {
                Ok(f64::NAN)
            } else {
                let n = values.len() as f64;
                let mean = kahan_sum_f64(&values) / n;
                let sq_diffs: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
                Ok(kahan_sum_f64(&sq_diffs) / (n - 1.0))
            }
        }
        TidyAgg::Sd(_) => {
            if values.len() < 2 {
                Ok(f64::NAN)
            } else {
                let n = values.len() as f64;
                let mean = kahan_sum_f64(&values) / n;
                let sq_diffs: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
                Ok((kahan_sum_f64(&sq_diffs) / (n - 1.0)).sqrt())
            }
        }
        TidyAgg::Quantile(_, p) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n == 1 {
                    return Ok(sorted[0]);
                }
                let pos = p * (n as f64 - 1.0);
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                if lo == hi || hi >= n {
                    Ok(sorted[lo.min(n - 1)])
                } else {
                    Ok(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
                }
            }
        }
        TidyAgg::NDistinct(_) => {
            use std::collections::BTreeSet;
            let distinct: BTreeSet<u64> = values.iter().map(|v| v.to_bits()).collect();
            Ok(distinct.len() as f64)
        }
        TidyAgg::Iqr(_) => {
            if values.is_empty() {
                Ok(f64::NAN)
            } else {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n == 1 {
                    return Ok(0.0);
                }
                let q1 = {
                    let pos = 0.25 * (n as f64 - 1.0);
                    let lo = pos.floor() as usize;
                    let hi = pos.ceil() as usize;
                    let frac = pos - lo as f64;
                    if lo == hi || hi >= n { sorted[lo.min(n - 1)] }
                    else { sorted[lo] + frac * (sorted[hi] - sorted[lo]) }
                };
                let q3 = {
                    let pos = 0.75 * (n as f64 - 1.0);
                    let lo = pos.floor() as usize;
                    let hi = pos.ceil() as usize;
                    let frac = pos - lo as f64;
                    if lo == hi || hi >= n { sorted[lo.min(n - 1)] }
                    else { sorted[lo] + frac * (sorted[hi] - sorted[lo]) }
                };
                Ok(q3 - q1)
            }
        }
    }
}

// â"€â"€ TidyAgg â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    /// Median of a numeric column.
    Median(String),
    /// Sample standard deviation (Kahan-based).
    Sd(String),
    /// Sample variance (Kahan-based).
    Var(String),
    /// Quantile at probability p ∈ [0, 1], using linear interpolation.
    Quantile(String, f64),
    /// Count of distinct values (uses BTreeSet).
    NDistinct(String),
    /// Interquartile range (Q3 − Q1).
    Iqr(String),
}

// â"€â"€ ArrangeKey â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// One sorting key for `arrange`.
#[derive(Debug, Clone)]
pub struct ArrangeKey {
    /// Column name to sort by.
    pub col_name: String,
    /// `true` = descending order.
    pub descending: bool,
}

impl ArrangeKey {
    /// Create an ascending sort key for the given column.
    pub fn asc(col_name: &str) -> Self {
        ArrangeKey { col_name: col_name.to_string(), descending: false }
    }
    /// Create a descending sort key for the given column.
    pub fn desc(col_name: &str) -> Self {
        ArrangeKey { col_name: col_name.to_string(), descending: true }
    }
}

// â"€â"€ TidyView: group_by, arrange, slice, distinct, joins â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

impl TidyView {

    // â"€â"€ group_by â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        let key_names: Vec<String> = keys.iter().map(|s| s.to_string()).collect();

        // O1 optimization: use BTree-accelerated build_fast for O(N log G) instead of O(N × G).
        // v2.2: pass the selection iterator directly — avoids a Vec<usize> allocation that
        // can run to millions of entries on large frames.
        let index = GroupIndex::build_fast(&self.base, &key_col_indices, self.mask.iter_indices(), key_names);

        Ok(GroupedTidyView {
            view: self.clone(),
            index,
        })
    }

    // â"€â"€ arrange â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        let mut row_indices: Vec<usize> = self.mask.iter_indices().collect();

        // v3 Phase 5: cat-aware arrange. For each key column that is
        // `Column::Categorical` with **lex-sorted levels** (the Phase 17
        // `forcats` invariant), `levels[code].cmp(&levels[other_code])`
        // is byte-equal to `code.cmp(&other_code)` — so we can sort by
        // u32 codes directly, skipping the per-call string lookup and
        // bytewise comparison. Mixed-type or unsorted-levels key cols
        // fall back to the string comparator.
        //
        // Pre-resolve each key once to either ("sorted-cat code slice",
        // descending) or ("legacy compare via column", descending).
        enum ArrangeKeyResolved<'a> {
            CatCodes { codes: &'a [u32], descending: bool },
            Legacy { col: &'a Column, descending: bool },
        }

        fn levels_are_sorted(levels: &[String]) -> bool {
            levels.windows(2).all(|w| w[0] <= w[1])
        }

        let resolved: Vec<ArrangeKeyResolved> = keys
            .iter()
            .map(|key| {
                let col = self.base.get_column(&key.col_name).unwrap();
                match col {
                    Column::Categorical { levels, codes } if levels_are_sorted(levels) => {
                        ArrangeKeyResolved::CatCodes {
                            codes: codes.as_slice(),
                            descending: key.descending,
                        }
                    }
                    _ => ArrangeKeyResolved::Legacy { col, descending: key.descending },
                }
            })
            .collect();

        // Stable sort by keys left-to-right.
        row_indices.sort_by(|&a, &b| {
            for resolved_key in &resolved {
                let ord = match resolved_key {
                    ArrangeKeyResolved::CatCodes { codes, descending } => {
                        let raw = codes[a].cmp(&codes[b]);
                        if *descending { raw.reverse() } else { raw }
                    }
                    ArrangeKeyResolved::Legacy { col, descending } => {
                        let raw = compare_column_rows(col, a, b);
                        if *descending { raw.reverse() } else { raw }
                    }
                };
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
            mask: AdaptiveSelection::all(nrows),
            proj: new_proj,
        })
    }

    // â"€â"€ slice â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    /// Select rows by a half-open range `[start, end)` of visible-row positions.
    ///
    /// Positions are relative to the current visible rows (0-based).
    /// Out-of-bounds: clamped to `[0, nrows]`.
    pub fn slice(&self, start: usize, end: usize) -> TidyView {
        let visible: Vec<usize> = self.mask.iter_indices().collect();
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
        let total = self.mask.count();
        let start = total.saturating_sub(n);
        self.slice(start, total)
    }

    /// Deterministic random sample of `n` visible rows using an LCG with `seed`.
    ///
    /// If `n >= nrows`, returns all visible rows in their original order (no error).
    /// Sampling uses a Knuth shuffle variant seeded by `seed` (deterministic LCG).
    pub fn slice_sample(&self, n: usize, seed: u64) -> TidyView {
        let mut visible: Vec<usize> = self.mask.iter_indices().collect();
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

    // â"€â"€ distinct â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        // Phase 2 cat-aware fast path: when every dedup column is
        // Column::Categorical, dedup on Vec<u32> of codes instead of
        // Vec<String> of display values. Bit-identical output: codes
        // are in 1:1 correspondence with display strings within a single
        // DataFrame.
        if let Some(cat_keys) = collect_categorical_keys(&self.base, &col_indices) {
            let mut seen_codes: BTreeSet<Vec<u32>> = BTreeSet::new();
            let mut selected_rows: Vec<usize> = Vec::new();
            let mut key_buf: Vec<u32> = Vec::with_capacity(cat_keys.codes.len());
            for row in self.mask.iter_indices() {
                key_buf.clear();
                for c in &cat_keys.codes {
                    key_buf.push(c[row]);
                }
                if seen_codes.insert(key_buf.clone()) {
                    selected_rows.push(row);
                }
            }
            return Ok(self.view_from_row_indices(selected_rows));
        }

        // O8 optimization: BTreeSet gives O(N log D) instead of O(N × D) linear scan
        let mut seen_keys: BTreeSet<Vec<String>> = BTreeSet::new();
        let mut selected_rows: Vec<usize> = Vec::new();

        for row in self.mask.iter_indices() {
            let key: Vec<String> = if col_indices.is_empty() {
                vec!["__all__".into()]
            } else {
                col_indices
                    .iter()
                    .map(|&ci| self.base.columns[ci].1.get_display(row))
                    .collect()
            };

            if seen_keys.insert(key) {
                selected_rows.push(row);
            }
        }

        Ok(self.view_from_row_indices(selected_rows))
    }

    // â"€â"€ joins â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ internal helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
            mask: AdaptiveSelection::from_predicate_result(words, nrows_base),
            proj: self.proj.clone(),
        }
    }
}

// â"€â"€ Join internals â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
/// Sorted by key tuple first, then by row index â€" guarantees determinism.
fn build_right_lookup(
    right: &TidyView,
    right_key_cols: &[usize],
) -> Vec<(Vec<String>, usize)> {
    let mut lookup: Vec<(Vec<String>, usize)> = right
        .mask
        .iter_indices()
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

/// O6: BTreeMap-accelerated right-side lookup for joins.
/// Groups right rows by their key tuple, enabling O(log K) lookup per left row
/// (where K = unique keys) instead of O(log N) binary search on a flat sorted list.
/// Right rows within each key are in ascending order (iter_set() guarantee).
fn build_right_lookup_btree(
    right: &TidyView,
    right_key_cols: &[usize],
) -> BTreeMap<Vec<String>, Vec<usize>> {
    let mut lookup: BTreeMap<Vec<String>, Vec<usize>> = BTreeMap::new();
    for r in right.mask.iter_indices() {
        let key = row_key(&right.base, right_key_cols, r);
        lookup.entry(key).or_default().push(r);
    }
    lookup
}

// ── v3 Phase 4: cat-aware join key path ─────────────────────────────────
//
// When every join-key column is `Column::Categorical` on BOTH sides, we
// can probe on `Vec<u32>` codes instead of `Vec<String>` displays. Each
// DataFrame owns its own dictionary, so left-side code 3 ≠ right-side
// code 3 in general — we build a per-key-column remap
//   `right_to_left[ki][right_code] -> Option<u32>`
// that translates every right code to the matching left code (or `None`
// if the level doesn't exist on the left, in which case that right row
// can never join).
//
// Bit-identity: codes ↔ levels are 1:1 within one DataFrame, so the
// BTreeMap slot assignment, output row order, and join multiplicity are
// byte-equal to the string-key path. Pinned by parity tests in
// `tests/tidy_tests/test_v3_phase4_categorical_joins.rs`.

/// Borrowed cat-aware join key metadata. Built only when every key column
/// on BOTH frames is `Column::Categorical`.
pub(crate) struct CategoricalJoinKeys<'a> {
    /// `left_codes[ki]` = code array for left-side key column ki.
    pub(crate) left_codes: Vec<&'a [u32]>,
    /// `right_codes[ki]` = code array for right-side key column ki.
    pub(crate) right_codes: Vec<&'a [u32]>,
    /// `right_to_left[ki][right_code] = Some(left_code)` if the level
    /// exists on the left, `None` otherwise (row cannot join).
    pub(crate) right_to_left: Vec<Vec<Option<u32>>>,
}

/// Returns `Some(CategoricalJoinKeys)` when every column index in both
/// `left_cols` and `right_cols` is `Column::Categorical`. Mixed-type
/// keys, length mismatches, or empty key lists → `None` (caller falls
/// back to the string path).
pub(crate) fn collect_categorical_join_keys<'a>(
    left_base: &'a DataFrame,
    left_cols: &[usize],
    right_base: &'a DataFrame,
    right_cols: &[usize],
) -> Option<CategoricalJoinKeys<'a>> {
    if left_cols.is_empty() || left_cols.len() != right_cols.len() {
        return None;
    }
    let mut left_codes = Vec::with_capacity(left_cols.len());
    let mut right_codes = Vec::with_capacity(left_cols.len());
    let mut right_to_left = Vec::with_capacity(left_cols.len());

    for (li, ri) in left_cols.iter().zip(right_cols.iter()) {
        match (&left_base.columns[*li].1, &right_base.columns[*ri].1) {
            (
                Column::Categorical { levels: ll, codes: lc },
                Column::Categorical { levels: rl, codes: rc },
            ) => {
                // Build deterministic left-level→left-code lookup.
                // BTreeMap not HashMap to keep build order stable across
                // runs (matches the determinism contract).
                let mut left_lookup: BTreeMap<&str, u32> = BTreeMap::new();
                for (i, lv) in ll.iter().enumerate() {
                    left_lookup.insert(lv.as_str(), i as u32);
                }
                // remap[right_code] = Some(left_code) | None
                let remap: Vec<Option<u32>> = rl
                    .iter()
                    .map(|rv| left_lookup.get(rv.as_str()).copied())
                    .collect();
                left_codes.push(lc.as_slice());
                right_codes.push(rc.as_slice());
                right_to_left.push(remap);
            }
            _ => return None,
        }
    }
    Some(CategoricalJoinKeys {
        left_codes,
        right_codes,
        right_to_left,
    })
}

/// Cat-aware right-side BTreeMap lookup. Keyed on `Vec<u32>` left-code
/// tuples (right-side codes are remapped to left-side codes via
/// `right_to_left`). Right rows whose remap returns `None` for any key
/// column are skipped — they cannot join.
fn build_right_lookup_btree_categorical<'a>(
    cat: &CategoricalJoinKeys<'a>,
    right_visible: impl Iterator<Item = usize>,
) -> BTreeMap<Vec<u32>, Vec<usize>> {
    let nkeys = cat.right_codes.len();
    let mut lookup: BTreeMap<Vec<u32>, Vec<usize>> = BTreeMap::new();
    let mut key_buf: Vec<u32> = Vec::with_capacity(nkeys);
    for r in right_visible {
        key_buf.clear();
        let mut all_mappable = true;
        for i in 0..nkeys {
            let rc = cat.right_codes[i][r];
            match cat.right_to_left[i][rc as usize] {
                Some(lc) => key_buf.push(lc),
                None => {
                    all_mappable = false;
                    break;
                }
            }
        }
        if all_mappable {
            lookup.entry(key_buf.clone()).or_default().push(r);
        }
    }
    lookup
}

/// Build the left-side join key (in left-code space) for a row.
#[inline]
fn left_join_key_codes(cat: &CategoricalJoinKeys<'_>, row: usize, buf: &mut Vec<u32>) {
    buf.clear();
    for codes in &cat.left_codes {
        buf.push(codes[row]);
    }
}

/// Inner join: collect (left_row, right_row) pairs.
fn join_match_rows(
    left: &TidyView,
    right: &TidyView,
    on: &[(&str, &str)],
    _kind: JoinKind,
) -> Result<(Vec<usize>, Vec<usize>), TidyError> {
    let (left_key_cols, right_key_cols) = resolve_join_keys(left, right, on)?;

    // v3 Phase 4: cat-aware fast path when every key column is
    // Column::Categorical on both frames. Bit-identical output to the
    // string path; falls back automatically on mixed-type keys.
    if let Some(cat) =
        collect_categorical_join_keys(&left.base, &left_key_cols, &right.base, &right_key_cols)
    {
        let lookup = build_right_lookup_btree_categorical(&cat, right.mask.iter_indices());
        let mut out_left = Vec::new();
        let mut out_right = Vec::new();
        let mut key_buf: Vec<u32> = Vec::with_capacity(cat.left_codes.len());
        for l_row in left.mask.iter_indices() {
            left_join_key_codes(&cat, l_row, &mut key_buf);
            if let Some(matches) = lookup.get(&key_buf) {
                for &r_row in matches {
                    out_left.push(l_row);
                    out_right.push(r_row);
                }
            }
        }
        return Ok((out_left, out_right));
    }

    // O6: use BTreeMap for O(log K) lookup instead of sorted-Vec binary search
    let lookup = build_right_lookup_btree(right, &right_key_cols);

    let mut out_left = Vec::new();
    let mut out_right = Vec::new();

    for l_row in left.mask.iter_indices() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        if let Some(matches) = lookup.get(&key) {
            for &r_row in matches {
                out_left.push(l_row);
                out_right.push(r_row);
            }
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

    // v3 Phase 4: cat-aware fast path. See `join_match_rows`.
    if let Some(cat) =
        collect_categorical_join_keys(&left.base, &left_key_cols, &right.base, &right_key_cols)
    {
        let lookup = build_right_lookup_btree_categorical(&cat, right.mask.iter_indices());
        let mut out_left = Vec::new();
        let mut out_right: Vec<Option<usize>> = Vec::new();
        let mut key_buf: Vec<u32> = Vec::with_capacity(cat.left_codes.len());
        for l_row in left.mask.iter_indices() {
            left_join_key_codes(&cat, l_row, &mut key_buf);
            match lookup.get(&key_buf) {
                Some(matches) if !matches.is_empty() => {
                    for &r_row in matches {
                        out_left.push(l_row);
                        out_right.push(Some(r_row));
                    }
                }
                _ => {
                    out_left.push(l_row);
                    out_right.push(None);
                }
            }
        }
        return Ok((out_left, out_right));
    }

    // O6: use BTreeMap for O(log K) lookup
    let lookup = build_right_lookup_btree(right, &right_key_cols);

    let mut out_left = Vec::new();
    let mut out_right: Vec<Option<usize>> = Vec::new();

    for l_row in left.mask.iter_indices() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        match lookup.get(&key) {
            Some(matches) if !matches.is_empty() => {
                for &r_row in matches {
                    out_left.push(l_row);
                    out_right.push(Some(r_row));
                }
            }
            _ => {
                out_left.push(l_row);
                out_right.push(None);
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

    // v3 Phase 4: cat-aware fast path. See `join_match_rows`.
    if let Some(cat) =
        collect_categorical_join_keys(&left.base, &left_key_cols, &right.base, &right_key_cols)
    {
        let lookup = build_right_lookup_btree_categorical(&cat, right.mask.iter_indices());
        let mut out = Vec::new();
        let mut key_buf: Vec<u32> = Vec::with_capacity(cat.left_codes.len());
        for l_row in left.mask.iter_indices() {
            left_join_key_codes(&cat, l_row, &mut key_buf);
            let has_match = lookup.contains_key(&key_buf);
            if has_match == semi {
                out.push(l_row);
            }
        }
        return Ok(out);
    }

    // O6: use BTreeMap for O(log K) lookup
    let lookup = build_right_lookup_btree(right, &right_key_cols);

    let mut out = Vec::new();
    for l_row in left.mask.iter_indices() {
        let key = row_key(&left.base, &left_key_cols, l_row);
        let has_match = lookup.contains_key(&key);
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

// â"€â"€ Column comparison for arrange â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        Column::CategoricalAdaptive(cc) => {
            // Byte-lex comparison via dictionary lookup. Bytes are the
            // determinism contract anchor for the byte_dict engine.
            cc.get(a).cmp(&cc.get(b))
        }
        Column::DateTime(v) => v[a].cmp(&v[b]),
    }
}

// (TidyError::EmptyGroup is defined in the TidyError enum above.)

// â"€â"€ NoGC safe-builtin registrations (Phase 11â€"12) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
//
// Ops that only update metadata (no column buffer alloc):
//   tidy_group_by      : builds GroupIndex (Vec of Vec<usize>) â€" no column alloc
//   tidy_ungroup       : drops GroupIndex â€" no alloc
//   tidy_arrange       : materialises sorted base (ALLOCATES) â†’ NOT @nogc safe
//   tidy_slice         : updates RowIndexMap â€" O(N) usize alloc, safe
//   tidy_distinct      : builds RowIndexMap â€" O(N) usize alloc, safe
//   tidy_semi_join     : builds RowIndexMap â€" O(N) usize alloc, safe
//   tidy_anti_join     : builds RowIndexMap â€" O(N) usize alloc, safe
//   tidy_inner_join    : materialises result â€" ALLOCATES â†’ NOT @nogc safe
//   tidy_left_join     : materialises result â€" ALLOCATES â†’ NOT @nogc safe
//   tidy_summarise     : materialises result â€" ALLOCATES â†’ NOT @nogc safe
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
        // 65 rows â€" two words; tail must not bleed into unset bits
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
// Phase 13â€"16: Tidy Completion
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
// | join type validation          | Comparable types only (Intâ†"Int, Floatâ†"Float,
// |                               | Strâ†"Str, Boolâ†"Bool, Intâ†"Float widened).
// |                               | TidyError::TypeMismatch otherwise. |
// | join suffix handling          | Default ".x"/".y"; user may override. |
// | right_join / full_join        | Defined; row order: see semantics section. |
// | group perf upgrade            | First-occurrence order preserved; identical
// |                               | output to Phase 11 implementation. |

// â"€â"€ New TidyError variants â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Nullable column layer â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
/// everywhere â€" that would be a breaking change across the whole codebase.
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
    /// Nullable 64-bit signed integer column.
    Int(NullableColumn<i64>),
    /// Nullable 64-bit floating-point column.
    Float(NullableColumn<f64>),
    /// Nullable UTF-8 string column.
    Str(NullableColumn<String>),
    /// Nullable boolean column.
    Bool(NullableColumn<bool>),
}

impl NullCol {
    /// Returns the number of rows (including nulls).
    pub fn len(&self) -> usize {
        match self {
            NullCol::Int(c) => c.len(),
            NullCol::Float(c) => c.len(),
            NullCol::Str(c) => c.len(),
            NullCol::Bool(c) => c.len(),
        }
    }

    /// Returns `true` if row `i` is null.
    pub fn is_null(&self, i: usize) -> bool {
        match self {
            NullCol::Int(c) => c.is_null(i),
            NullCol::Float(c) => c.is_null(i),
            NullCol::Str(c) => c.is_null(i),
            NullCol::Bool(c) => c.is_null(i),
        }
    }

    /// Returns the human-readable type name of this nullable column variant.
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
            Column::CategoricalAdaptive(cc) => {
                let n = cc.len();
                let strings: Vec<String> = (0..n)
                    .map(|i| match cc.get(i) {
                        None => String::new(),
                        Some(b) => String::from_utf8_lossy(b).into_owned(),
                    })
                    .collect();
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
    /// Create an empty NullableFrame with no columns.
    pub fn new() -> Self {
        Self { columns: Vec::new() }
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.columns.first().map(|(_, c)| c.len()).unwrap_or(0)
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Returns column names in order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Look up a nullable column by name.
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

// â"€â"€ Helpers for nullable-aware gather â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Gather column rows with optional indices (None â†’ null).
/// Used in left/right/full join output where some rows have no match.
fn gather_column_nullable_null(col: &Column, indices: &[Option<usize>]) -> NullCol {
    if matches!(col, Column::CategoricalAdaptive(_)) {
        return gather_column_nullable_null(&col.to_legacy_categorical(), indices);
    }
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
        Column::CategoricalAdaptive(_) => unreachable!("handled by early return"),
    }
}

// â"€â"€ Across support types â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    /// Create a new across transformation with the given name and column-mapping function.
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
    /// Create a new across specification targeting the given columns with one transform.
    pub fn new(cols: impl IntoIterator<Item = impl Into<String>>, transform: AcrossTransform) -> Self {
        Self {
            cols: cols.into_iter().map(|c| c.into()).collect(),
            transform,
            name_template: None,
        }
    }

    /// Set a custom output-name template (use `{col}` and `{fn}` as placeholders).
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

// â"€â"€ Join maturity types â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
    /// Create custom suffixes for left and right table columns on name collision.
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> Self {
        Self { left: left.into(), right: right.into() }
    }
}

// â"€â"€ Column type comparison for join validation â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Phase 13-16 TidyView extensions â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

impl TidyView {

    // â"€â"€ pivot_longer â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        let visible_rows: Vec<usize> = self.mask.iter_indices().collect();
        let n_out = visible_rows.len() * value_cols.len();

        // Build id columns (repeated value_cols.len() times per source row)
        let mut out_cols: Vec<(String, Column)> = Vec::new();
        for &id_idx in &id_col_indices {
            let (name, col_orig) = &self.base.columns[id_idx];
            let legacy_owned;
            let col: &Column = if matches!(col_orig, Column::CategoricalAdaptive(_)) {
                legacy_owned = col_orig.to_legacy_categorical();
                &legacy_owned
            } else {
                col_orig
            };
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
                Column::CategoricalAdaptive(_) => unreachable!("converted via legacy_owned"),
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
            Column::Categorical { .. } | Column::CategoricalAdaptive(_) | Column::DateTime(_) => {
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

    // â"€â"€ pivot_wider â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        let visible_rows: Vec<usize> = self.mask.iter_indices().collect();

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

    // â"€â"€ rename â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ relocate â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ drop_cols â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ bind_rows â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        let self_rows: Vec<usize> = self.mask.iter_indices().collect();
        let other_rows: Vec<usize> = other.mask.iter_indices().collect();

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

    // â"€â"€ bind_cols â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        let self_rows: Vec<usize> = self.mask.iter_indices().collect();
        let other_rows: Vec<usize> = other.mask.iter_indices().collect();

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

    // â"€â"€ mutate_across â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ right_join â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ full_join â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ inner_join_typed (join maturity upgrade) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ left_join_typed (join maturity upgrade) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Position enum for relocate â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Column concatenation helper â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Join key type validation â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ Join frames with suffix collision handling â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // Left projected columns (nullable â€" unmatched = null)
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

    for l_row in left.mask.iter_indices() {
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
    for r_row in right.mask.iter_indices() {
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

// â"€â"€ GroupedTidyView: mutate_across + summarise_across â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

        // Key columns (String typed â€" group key values)
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

// â"€â"€ Simple IndexMap for mutate_across column merging â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
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

// â"€â"€ Group perf upgrade (deterministic hash accelerator) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
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
// This is a pure internal change â€" the external API and output semantics
// are identical to Phase 11.

impl GroupIndex {
    /// Build a GroupIndex using a BTree-accelerated lookup.
    ///
    /// Semantics: identical to `GroupIndex::build()`. First-occurrence group
    /// ordering is preserved. The only difference is O(N log G) vs O(N Ã— G).
    ///
    /// **Phase 2 (v3) cat-aware fast path**: when every key column is
    /// `Column::Categorical`, the lookup BTreeMap keys are
    /// `Vec<u32>` of category codes instead of `Vec<String>` of display
    /// values. This eliminates `levels[code].clone()` per row per key
    /// column. The fast path is bit-identical to the string path:
    ///   - Group slots are still assigned in first-occurrence row order.
    ///   - `GroupMeta::key_values` is still `Vec<String>` of display
    ///     values, computed once per group (not once per row).
    ///   - Mixed-type keys (e.g., categorical + int) fall back to the
    ///     string path automatically.
    pub fn build_fast<I: IntoIterator<Item = usize>>(
        base: &DataFrame,
        key_col_indices: &[usize],
        visible_rows: I,
        key_names: Vec<String>,
    ) -> Self {
        use std::collections::BTreeMap;

        // Phase 2 cat-aware fast path: try categorical-only key encoding
        // first. Returns Some(...) when every key col is categorical.
        if let Some(cat_keys) = collect_categorical_keys(base, key_col_indices) {
            return build_groupindex_categorical(cat_keys, visible_rows, key_names);
        }

        let mut groups: Vec<GroupMeta> = Vec::new();
        let mut key_to_slot: BTreeMap<Vec<String>, usize> = BTreeMap::new();

        for row in visible_rows {
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

// Phase 2 cat-aware key encoding.
//
// When every key column is `Column::Categorical`, group_by + distinct
// build their lookup BTrees over `Vec<u32>` of codes instead of
// `Vec<String>` of display values. This eliminates `String::clone()` per
// row per key column on a hot path.
//
// Bit-identical to the string path:
//   - First-occurrence group ordering is preserved (driven by row scan
//     order, independent of key encoding).
//   - `GroupMeta::key_values` / dedup output uses `levels[code]` lookup
//     once per *group* (or once per *unique row*, in distinct), not per
//     row. The display values are byte-for-byte identical.
//
// Mixed-type keys (e.g., one Categorical + one Int) cause
// `collect_categorical_keys` to return `None` so callers fall back.

/// Borrowed view onto the per-key categorical metadata.
pub(crate) struct CategoricalKeys<'a> {
    /// `levels[i]` = level table for key column i.
    pub(crate) levels: Vec<&'a [String]>,
    /// `codes[i]` = code array for key column i.
    pub(crate) codes: Vec<&'a [u32]>,
}

/// Returns `Some(CategoricalKeys)` when every column index in
/// `key_col_indices` is `Column::Categorical`. Returns `None` if any key
/// column is a non-categorical variant (caller must fall back).
pub(crate) fn collect_categorical_keys<'a>(
    base: &'a DataFrame,
    key_col_indices: &[usize],
) -> Option<CategoricalKeys<'a>> {
    if key_col_indices.is_empty() {
        return None;
    }
    let mut levels: Vec<&[String]> = Vec::with_capacity(key_col_indices.len());
    let mut codes: Vec<&[u32]> = Vec::with_capacity(key_col_indices.len());
    for &ci in key_col_indices {
        match &base.columns[ci].1 {
            Column::Categorical { levels: l, codes: c } => {
                levels.push(l.as_slice());
                codes.push(c.as_slice());
            }
            _ => return None,
        }
    }
    Some(CategoricalKeys { levels, codes })
}

/// Cat-aware GroupIndex builder. Bit-identical output to the string-key
/// path: same slot assignment, same `key_values`, same `row_indices`.
fn build_groupindex_categorical<I: IntoIterator<Item = usize>>(
    cat: CategoricalKeys<'_>,
    visible_rows: I,
    key_names: Vec<String>,
) -> GroupIndex {
    use std::collections::BTreeMap;
    let nkeys = cat.codes.len();
    let mut groups: Vec<GroupMeta> = Vec::new();
    let mut key_to_slot: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
    let mut key_buf: Vec<u32> = Vec::with_capacity(nkeys);

    for row in visible_rows {
        key_buf.clear();
        for c in &cat.codes {
            key_buf.push(c[row]);
        }
        if let Some(&slot) = key_to_slot.get(&key_buf) {
            groups[slot].row_indices.push(row);
        } else {
            // Materialise display strings exactly once per group
            let key_values: Vec<String> = (0..nkeys)
                .map(|i| cat.levels[i][key_buf[i] as usize].clone())
                .collect();
            let slot = groups.len();
            key_to_slot.insert(key_buf.clone(), slot);
            groups.push(GroupMeta { key_values, row_indices: vec![row] });
        }
    }

    GroupIndex { groups, key_names }
}

// â"€â"€ TidyView: group_by_fast (uses BTree-accelerated GroupIndex) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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
        let key_names: Vec<String> = keys.iter().map(|s| s.to_string()).collect();
        let index = GroupIndex::build_fast(&self.base, &key_col_indices, self.mask.iter_indices(), key_names);
        Ok(GroupedTidyView { view: self.clone(), index })
    }
}

// ── v3 Phase 6: Streaming aggregations ─────────────────────────────────────
//
// `summarise_streaming` is a sibling of `summarise` that skips the
// `GroupIndex` materialization step entirely. The legacy path builds
// `Vec<usize>` row indices per group (8 bytes × N rows ≈ 800 MB for
// 100M rows), then walks each per-group vector once per aggregation.
// The streaming path walks visible rows ONCE, maintaining a
// `BTreeMap<key, Vec<AccState>>` where each accumulator holds
// running-state O(constant) memory: Kahan sum carry, count, min, max,
// or Welford running mean+M2 for variance/stddev.
//
// Memory: O(K · acc_size) instead of O(N · usize). For 100M rows /
// 1000 groups / 32-byte accumulator, that's ~32 KB vs ~800 MB —
// roughly 25 000× less memory.
//
// Determinism: BTreeMap (not HashMap), Kahan / Welford (not naive
// floating-point sum). Output row order is the BTreeMap iteration
// order — byte-equal to the legacy path's first-occurrence ordering
// when keys are integers/strings/sorted-Categorical (see Phase 6
// integration test `phase6_streaming_summarise_matches_legacy`).
//
// Cat-aware: when every key column is `Column::Categorical`, the key
// tuple is `Vec<u32>` codes (bit-equal to Phase 2's group_by fast
// path). Falls back to `Vec<String>` displays on mixed-type keys.
//
// Aggregation surface: Count / Sum / Mean / Min / Max / Var / Sd.
// Median / Quantile / NDistinct require the full row index list and
// are not streaming — callers fall back to the legacy `summarise`.

/// Streamable aggregation operations. Subset of `TidyAgg` that admits
/// a constant-state per-group accumulator.
#[derive(Debug, Clone)]
pub enum StreamingAgg {
    /// Row count per group.
    Count,
    /// Kahan-sum of a numeric column.
    Sum(String),
    /// Arithmetic mean of a numeric column.
    Mean(String),
    /// Minimum value (NaN-aware: NaN never wins).
    Min(String),
    /// Maximum value (NaN-aware: NaN never wins).
    Max(String),
    /// Sample variance via Welford's algorithm (numerically stable).
    Var(String),
    /// Sample standard deviation (sqrt of `Var`).
    Sd(String),
}

/// Per-group running state. One per `(group, agg)` pair.
#[derive(Debug, Clone)]
enum AccState {
    Count {
        n: u64,
    },
    Sum {
        // Kahan running sum.
        sum: f64,
        c: f64,
    },
    Mean {
        // Kahan-stable accumulator + count.
        sum: f64,
        c: f64,
        n: u64,
    },
    Min {
        cur: f64,
        any: bool,
    },
    Max {
        cur: f64,
        any: bool,
    },
    /// Welford running mean + sum of squared deviations from mean.
    Welford {
        n: u64,
        mean: f64,
        m2: f64,
    },
}

impl AccState {
    fn from_agg(agg: &StreamingAgg) -> Self {
        match agg {
            StreamingAgg::Count => AccState::Count { n: 0 },
            StreamingAgg::Sum(_) => AccState::Sum { sum: 0.0, c: 0.0 },
            StreamingAgg::Mean(_) => AccState::Mean { sum: 0.0, c: 0.0, n: 0 },
            StreamingAgg::Min(_) => AccState::Min {
                cur: f64::INFINITY,
                any: false,
            },
            StreamingAgg::Max(_) => AccState::Max {
                cur: f64::NEG_INFINITY,
                any: false,
            },
            StreamingAgg::Var(_) | StreamingAgg::Sd(_) => AccState::Welford {
                n: 0,
                mean: 0.0,
                m2: 0.0,
            },
        }
    }

    fn update(&mut self, x: f64) {
        match self {
            AccState::Count { n } => *n += 1,
            AccState::Sum { sum, c } => {
                // Kahan summation.
                let y = x - *c;
                let t = *sum + y;
                *c = (t - *sum) - y;
                *sum = t;
            }
            AccState::Mean { sum, c, n } => {
                let y = x - *c;
                let t = *sum + y;
                *c = (t - *sum) - y;
                *sum = t;
                *n += 1;
            }
            AccState::Min { cur, any } => {
                if !x.is_nan() {
                    if !*any || x < *cur {
                        *cur = x;
                        *any = true;
                    }
                }
            }
            AccState::Max { cur, any } => {
                if !x.is_nan() {
                    if !*any || x > *cur {
                        *cur = x;
                        *any = true;
                    }
                }
            }
            AccState::Welford { n, mean, m2 } => {
                // Welford's online variance.
                *n += 1;
                let delta = x - *mean;
                *mean += delta / (*n as f64);
                let delta2 = x - *mean;
                *m2 += delta * delta2;
            }
        }
    }

    fn finalize(&self, agg: &StreamingAgg) -> f64 {
        match (self, agg) {
            (AccState::Count { n }, StreamingAgg::Count) => *n as f64,
            (AccState::Sum { sum, .. }, StreamingAgg::Sum(_)) => *sum,
            (AccState::Mean { sum, n, .. }, StreamingAgg::Mean(_)) => {
                if *n == 0 {
                    f64::NAN
                } else {
                    *sum / (*n as f64)
                }
            }
            (AccState::Min { cur, any }, StreamingAgg::Min(_)) => {
                if *any {
                    *cur
                } else {
                    f64::NAN
                }
            }
            (AccState::Max { cur, any }, StreamingAgg::Max(_)) => {
                if *any {
                    *cur
                } else {
                    f64::NAN
                }
            }
            (AccState::Welford { n, m2, .. }, StreamingAgg::Var(_)) => {
                if *n < 2 {
                    f64::NAN
                } else {
                    *m2 / ((*n - 1) as f64)
                }
            }
            (AccState::Welford { n, m2, .. }, StreamingAgg::Sd(_)) => {
                if *n < 2 {
                    f64::NAN
                } else {
                    (*m2 / ((*n - 1) as f64)).sqrt()
                }
            }
            _ => f64::NAN,
        }
    }
}

/// Pull a row's value from a numeric column as f64 (NaN for non-numeric).
fn row_as_f64(col: &Column, row: usize) -> f64 {
    match col {
        Column::Float(v) => v[row],
        Column::Int(v) => v[row] as f64,
        _ => f64::NAN,
    }
}

impl TidyView {
    /// v3 Phase 6: streaming summarise. Single-pass aggregation that
    /// avoids materializing per-group row index vectors.
    ///
    /// Returns a `TidyFrame` with key columns followed by aggregate
    /// columns (one per assignment). Output row order is BTreeMap
    /// iteration over key tuples — byte-equal to the legacy path when
    /// keys are integers / strings / sorted-Categorical.
    pub fn summarise_streaming(
        &self,
        keys: &[&str],
        aggs: &[(&str, StreamingAgg)],
    ) -> Result<TidyFrame, TidyError> {
        // Validate: no duplicate output names; no key/agg name collision.
        {
            let mut seen = std::collections::BTreeSet::new();
            for &(name, _) in aggs {
                if !seen.insert(name) {
                    return Err(TidyError::DuplicateColumn(name.to_string()));
                }
            }
            for &k in keys {
                if seen.contains(k) {
                    return Err(TidyError::DuplicateColumn(k.to_string()));
                }
            }
        }

        // Resolve key column indices.
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

        // Resolve aggregation source columns once.
        let agg_col_indices: Vec<Option<usize>> = aggs
            .iter()
            .map(|(_, agg)| match agg {
                StreamingAgg::Count => None,
                StreamingAgg::Sum(c)
                | StreamingAgg::Mean(c)
                | StreamingAgg::Min(c)
                | StreamingAgg::Max(c)
                | StreamingAgg::Var(c)
                | StreamingAgg::Sd(c) => self.base.columns.iter().position(|(n, _)| n == c),
            })
            .collect();
        for (i, (_, agg)) in aggs.iter().enumerate() {
            if matches!(agg, StreamingAgg::Count) {
                continue;
            }
            if agg_col_indices[i].is_none() {
                let col_name = match agg {
                    StreamingAgg::Sum(c)
                    | StreamingAgg::Mean(c)
                    | StreamingAgg::Min(c)
                    | StreamingAgg::Max(c)
                    | StreamingAgg::Var(c)
                    | StreamingAgg::Sd(c) => c.clone(),
                    _ => String::new(),
                };
                return Err(TidyError::ColumnNotFound(col_name));
            }
        }

        // Cat-aware fast path: when every key is Categorical, key on Vec<u32>.
        let cat = collect_categorical_keys(&self.base, &key_col_indices);

        // BTreeMap keyed on Vec<u32> (cat) or Vec<String> (legacy).
        // Both produce deterministic iteration.
        use std::collections::BTreeMap;

        let n_aggs = aggs.len();
        let init_accs = || -> Vec<AccState> {
            aggs.iter().map(|(_, a)| AccState::from_state(a)).collect()
        };

        // Helper inside summarise_streaming to keep AccState constructor private.
        // (alias to make readable above)
        fn _unused() {}

        // Streaming pass.
        let (cat_state, str_state) = if let Some(cat) = cat.as_ref() {
            let mut state: BTreeMap<Vec<u32>, Vec<AccState>> = BTreeMap::new();
            let mut key_buf: Vec<u32> = Vec::with_capacity(cat.codes.len());
            for row in self.mask.iter_indices() {
                key_buf.clear();
                for c in &cat.codes {
                    key_buf.push(c[row]);
                }
                let entry = state
                    .entry(key_buf.clone())
                    .or_insert_with(&init_accs);
                for (i, (_, agg)) in aggs.iter().enumerate() {
                    if let Some(col_idx) = agg_col_indices[i] {
                        entry[i].update(row_as_f64(&self.base.columns[col_idx].1, row));
                    } else {
                        entry[i].update(0.0); // Count ignores value.
                    }
                }
            }
            (Some(state), None)
        } else {
            let mut state: BTreeMap<Vec<String>, Vec<AccState>> = BTreeMap::new();
            for row in self.mask.iter_indices() {
                let key: Vec<String> = key_col_indices
                    .iter()
                    .map(|&ci| self.base.columns[ci].1.get_display(row))
                    .collect();
                let entry = state.entry(key).or_insert_with(&init_accs);
                for (i, (_, agg)) in aggs.iter().enumerate() {
                    if let Some(col_idx) = agg_col_indices[i] {
                        entry[i].update(row_as_f64(&self.base.columns[col_idx].1, row));
                    } else {
                        entry[i].update(0.0);
                    }
                }
            }
            (None, Some(state))
        };

        // Materialize output frame.
        let n_groups = cat_state
            .as_ref()
            .map(|s| s.len())
            .unwrap_or_else(|| str_state.as_ref().unwrap().len());

        let mut result_columns: Vec<(String, Column)> = Vec::with_capacity(keys.len() + n_aggs);

        // Key columns. For cat-aware: rebuild via levels[code]. For str: take strings as-is.
        if let Some(state) = &cat_state {
            let cat = cat.as_ref().unwrap();
            for (ki, &key_col_idx) in key_col_indices.iter().enumerate() {
                let mut vals: Vec<String> = Vec::with_capacity(n_groups);
                for key_codes in state.keys() {
                    let code = key_codes[ki] as usize;
                    vals.push(cat.levels[ki][code].clone());
                }
                let key_name = self.base.columns[key_col_idx].0.clone();
                // Materialize as Categorical to preserve type.
                let levels: Vec<String> = cat.levels[ki].to_vec();
                let codes: Vec<u32> = state.keys().map(|k| k[ki]).collect();
                result_columns.push((key_name, Column::Categorical { levels, codes }));
                let _ = vals;
            }
        } else {
            let state = str_state.as_ref().unwrap();
            for (ki, &key_col_idx) in key_col_indices.iter().enumerate() {
                let key_name = self.base.columns[key_col_idx].0.clone();
                let vals: Vec<String> = state.keys().map(|k| k[ki].clone()).collect();
                result_columns.push((key_name, Column::Str(vals)));
            }
        }

        // Aggregate columns.
        for (i, (out_name, agg)) in aggs.iter().enumerate() {
            let vals: Vec<f64> = if let Some(state) = &cat_state {
                state.values().map(|accs| accs[i].finalize(agg)).collect()
            } else {
                str_state
                    .as_ref()
                    .unwrap()
                    .values()
                    .map(|accs| accs[i].finalize(agg))
                    .collect()
            };
            let col = if matches!(agg, StreamingAgg::Count) {
                Column::Int(vals.into_iter().map(|x| x as i64).collect())
            } else {
                Column::Float(vals)
            };
            result_columns.push((out_name.to_string(), col));
        }

        let df = DataFrame::from_columns(result_columns)
            .map_err(|e| TidyError::Internal(e.to_string()))?;
        Ok(TidyFrame::from_df(df))
    }
}

impl AccState {
    fn from_state(agg: &StreamingAgg) -> Self {
        Self::from_agg(agg)
    }
}

// â"€â"€ NoGC safe-builtin registrations (Phase 13â€"16) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
//
// New safe (view/metadata-only, no column buffer alloc):
//   tidy_rename          : builds new DataFrame with renamed cols â€" O(NÃ—K) clone
//                          BUT this is a metadata rebuild, not a hot-path â€" listed as ALLOC
//                          â†’ NOT @nogc safe (rebuilds base)
//   tidy_relocate        : updates ProjectionMap only â€" O(K) â€" SAFE
//   tidy_drop_cols       : updates ProjectionMap only â€" O(K) â€" SAFE
//   tidy_group_by_fast   : BTree lookup + GroupIndex â€" no column alloc â€" SAFE
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
// PHASE 17: CATEGORICAL FOUNDATIONS â€" fct_encode, fct_lump, fct_reorder,
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
//  [S-7]  fct_collapse: metadata-only â€" never rewrites data buffer; O(L) pass over
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

// â"€â"€ FctColumn â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// A compact categorical column: stores u16 indices into a levels table.
///
/// Invariant: `data[i] < levels.len()` for all i where bitmap is set.
/// Null rows (in NullableFactor) may carry index 0 â€" callers must check bitmap.
#[derive(Clone, Debug)]
pub struct FctColumn {
    /// Mapping from index â†’ level string.  Order = first-occurrence of each string
    /// in the source column (deterministic, no hashing).
    pub levels: Vec<String>,
    /// One u16 per row.  Value is the index into `levels`.
    pub data: Vec<u16>,
}

impl FctColumn {
    // â"€â"€ Constructors â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    /// Encode a string column into a FctColumn.
    ///
    /// Level order = first-occurrence in `strings`.
    /// Returns Err if more than 65,535 distinct strings are found.
    pub fn encode(strings: &[String]) -> Result<Self, TidyError> {
        use std::collections::BTreeMap;
        let mut levels: Vec<String> = Vec::new();
        // BTreeMap for O(log L) lookup; key ordering is string-lexicographic
        // (deterministic across runs â€" no hash randomness).
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
        let visible: Vec<usize> = view.mask.iter_indices().collect();
        let strings: Vec<String> = visible.iter()
            .map(|&r| col_data.get_display(r))
            .collect();
        Self::encode(&strings)
    }

    // â"€â"€ Shape â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    /// Returns the number of rows in this factor column.
    pub fn nrows(&self) -> usize { self.data.len() }
    /// Returns the number of distinct levels.
    pub fn nlevels(&self) -> usize { self.levels.len() }

    /// Decode row i back to its string value.
    pub fn decode(&self, i: usize) -> &str {
        &self.levels[self.data[i] as usize]
    }

    // â"€â"€ fct_lump â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ fct_reorder â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ fct_collapse â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    // â"€â"€ Materialise back to Column::Str â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

// â"€â"€ TidyError extensions for Phase 17 â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

impl TidyError {
    /// Convenience constructor for a capacity-exceeded error.
    pub fn capacity_exceeded(limit: usize, got: usize) -> Self {
        TidyError::CapacityExceeded { limit, got }
    }
}

// â"€â"€ NullableFactor â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

    /// Returns the total number of rows (including nulls).
    pub fn nrows(&self) -> usize { self.fct.nrows() }
    /// Returns the number of distinct factor levels.
    pub fn nlevels(&self) -> usize { self.fct.nlevels() }
    /// Returns `true` if row `i` is null.
    pub fn is_null(&self, i: usize) -> bool { !self.validity.get(i) }
    /// Returns the count of non-null rows.
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

// â"€â"€ TidyView: fct_encode integration â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

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

#[cfg(test)]
mod rolling_window_tests {
    use super::*;

    /// Helper: build a simple DataFrame with a single float column.
    fn make_df(col_name: &str, vals: Vec<f64>) -> DataFrame {
        DataFrame {
            columns: vec![(col_name.to_string(), Column::Float(vals))],
        }
    }

    #[test]
    fn rolling_sum_basic() {
        // [1,2,3,4,5] with window=3
        // Expected: [1, 3, 6, 9, 12]
        let df = make_df("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let expr = DExpr::RollingSum("x".into(), 3);
        let col = eval_expr_column(&df, &expr, 5).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 5);
                assert!((v[0] - 1.0).abs() < 1e-12);
                assert!((v[1] - 3.0).abs() < 1e-12);
                assert!((v[2] - 6.0).abs() < 1e-12);
                assert!((v[3] - 9.0).abs() < 1e-12);
                assert!((v[4] - 12.0).abs() < 1e-12);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_mean_basic() {
        // [1,2,3,4,5] with window=3
        // Expected: [1/1, 3/2, 6/3, 9/3, 12/3] = [1, 1.5, 2, 3, 4]
        let df = make_df("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let expr = DExpr::RollingMean("x".into(), 3);
        let col = eval_expr_column(&df, &expr, 5).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 5);
                assert!((v[0] - 1.0).abs() < 1e-12);
                assert!((v[1] - 1.5).abs() < 1e-12);
                assert!((v[2] - 2.0).abs() < 1e-12);
                assert!((v[3] - 3.0).abs() < 1e-12);
                assert!((v[4] - 4.0).abs() < 1e-12);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_min_basic() {
        // [5,3,4,1,2] with window=3
        // Expected: [5, 3, 3, 1, 1]
        let df = make_df("x", vec![5.0, 3.0, 4.0, 1.0, 2.0]);
        let expr = DExpr::RollingMin("x".into(), 3);
        let col = eval_expr_column(&df, &expr, 5).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 5);
                assert!((v[0] - 5.0).abs() < 1e-12);
                assert!((v[1] - 3.0).abs() < 1e-12);
                assert!((v[2] - 3.0).abs() < 1e-12);
                assert!((v[3] - 1.0).abs() < 1e-12);
                assert!((v[4] - 1.0).abs() < 1e-12);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_max_basic() {
        // [1,5,3,2,4] with window=3
        // Expected: [1, 5, 5, 5, 4]
        let df = make_df("x", vec![1.0, 5.0, 3.0, 2.0, 4.0]);
        let expr = DExpr::RollingMax("x".into(), 3);
        let col = eval_expr_column(&df, &expr, 5).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 5);
                assert!((v[0] - 1.0).abs() < 1e-12);
                assert!((v[1] - 5.0).abs() < 1e-12);
                assert!((v[2] - 5.0).abs() < 1e-12);
                assert!((v[3] - 5.0).abs() < 1e-12);
                assert!((v[4] - 4.0).abs() < 1e-12);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_var_basic() {
        // [2,4,6,8] with window=3
        let df = make_df("x", vec![2.0, 4.0, 6.0, 8.0]);
        let expr = DExpr::RollingVar("x".into(), 3);
        let col = eval_expr_column(&df, &expr, 4).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 4);
                // i=0: count=1, var=0
                assert!((v[0] - 0.0).abs() < 1e-12);
                // i=1: count=2, sample var of [2,4] = 2.0
                assert!((v[1] - 2.0).abs() < 1e-10);
                // i=2: count=3, sample var of [2,4,6] = 4.0
                assert!((v[2] - 4.0).abs() < 1e-10);
                // i=3: count=3, sample var of [4,6,8] = 4.0
                assert!((v[3] - 4.0).abs() < 1e-10);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_sd_basic() {
        let df = make_df("x", vec![2.0, 4.0, 6.0, 8.0]);
        let expr = DExpr::RollingSd("x".into(), 3);
        let col = eval_expr_column(&df, &expr, 4).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 4);
                assert!((v[0] - 0.0).abs() < 1e-12);
                assert!((v[1] - 2.0_f64.sqrt()).abs() < 1e-10);
                assert!((v[2] - 2.0).abs() < 1e-10);
                assert!((v[3] - 2.0).abs() < 1e-10);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_window_larger_than_data() {
        let df = make_df("x", vec![1.0, 2.0, 3.0]);
        let expr = DExpr::RollingSum("x".into(), 10);
        let col = eval_expr_column(&df, &expr, 3).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 3);
                assert!((v[0] - 1.0).abs() < 1e-12);
                assert!((v[1] - 3.0).abs() < 1e-12);
                assert!((v[2] - 6.0).abs() < 1e-12);
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_window_of_one() {
        let df = make_df("x", vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        let expr_min = DExpr::RollingMin("x".into(), 1);
        let expr_max = DExpr::RollingMax("x".into(), 1);
        let col_min = eval_expr_column(&df, &expr_min, 5).unwrap();
        let col_max = eval_expr_column(&df, &expr_max, 5).unwrap();
        match (col_min, col_max) {
            (Column::Float(mins), Column::Float(maxs)) => {
                let expected = [3.0, 1.0, 4.0, 1.0, 5.0];
                for i in 0..5 {
                    assert!((mins[i] - expected[i]).abs() < 1e-12, "min[{}]", i);
                    assert!((maxs[i] - expected[i]).abs() < 1e-12, "max[{}]", i);
                }
            }
            _ => panic!("expected Float columns"),
        }
    }

    #[test]
    fn rolling_sum_with_nan() {
        let df = make_df("x", vec![1.0, f64::NAN, 3.0, 4.0]);
        let expr = DExpr::RollingSum("x".into(), 2);
        let col = eval_expr_column(&df, &expr, 4).unwrap();
        match col {
            Column::Float(v) => {
                assert_eq!(v.len(), 4);
                assert!((v[0] - 1.0).abs() < 1e-12);
                assert!(v[1].is_nan());
                assert!(v[2].is_nan());
                assert!(v[3].is_nan()); // NaN poisons Kahan accumulator
            }
            _ => panic!("expected Float column"),
        }
    }

    #[test]
    fn rolling_determinism() {
        let df = make_df("x", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let expr = DExpr::RollingSum("x".into(), 4);
        let mut runs: Vec<Vec<f64>> = Vec::new();
        for _ in 0..3 {
            let col = eval_expr_column(&df, &expr, 10).unwrap();
            match col {
                Column::Float(v) => runs.push(v),
                _ => panic!("expected Float column"),
            }
        }
        assert_eq!(runs[0], runs[1]);
        assert_eq!(runs[1], runs[2]);
    }

    #[test]
    fn rolling_display() {
        let expr = DExpr::RollingSum("val".into(), 5);
        assert_eq!(format!("{}", expr), "rolling_sum(\"val\", 5)");
        let expr2 = DExpr::RollingMean("col".into(), 3);
        assert_eq!(format!("{}", expr2), "rolling_mean(\"col\", 3)");
    }

    #[test]
    fn rolling_collect_columns() {
        let expr = DExpr::RollingSum("revenue".into(), 7);
        let mut cols = Vec::new();
        collect_expr_columns(&expr, &mut cols);
        assert_eq!(cols, vec!["revenue".to_string()]);
    }

    #[test]
    fn rolling_not_allowed_in_row_context() {
        let df = make_df("x", vec![1.0, 2.0, 3.0]);
        let expr = DExpr::RollingSum("x".into(), 2);
        let result = eval_expr_row(&df, &expr, 0);
        assert!(result.is_err());
    }

    // ── v3 Phase 4 unit tests ─────────────────────────────────────────

    fn cat_col(levels: &[&str], codes: &[u32]) -> Column {
        Column::Categorical {
            levels: levels.iter().map(|s| s.to_string()).collect(),
            codes: codes.to_vec(),
        }
    }

    #[test]
    fn phase4_collect_cat_keys_returns_some_when_all_categorical() {
        let left = DataFrame::from_columns(vec![
            ("k".into(), cat_col(&["a", "b", "c"], &[0, 1, 2, 0])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            ("k".into(), cat_col(&["b", "a"], &[0, 1, 1])),
        ])
        .unwrap();
        let cat = collect_categorical_join_keys(&left, &[0], &right, &[0]).unwrap();
        // right_to_left for the right's "k" col: right code 0 = "b" → left code 1; right code 1 = "a" → left code 0.
        assert_eq!(cat.right_to_left[0], vec![Some(1u32), Some(0u32)]);
    }

    #[test]
    fn phase4_collect_cat_keys_returns_none_on_mixed_types() {
        let left = DataFrame::from_columns(vec![
            ("k".into(), cat_col(&["a"], &[0])),
            ("n".into(), Column::Int(vec![1])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            ("k".into(), cat_col(&["a"], &[0])),
            ("n".into(), Column::Int(vec![1])),
        ])
        .unwrap();
        // First col cat-aware OK, second col Int → fallback.
        assert!(collect_categorical_join_keys(&left, &[0, 1], &right, &[0, 1]).is_none());
    }

    #[test]
    fn phase4_collect_cat_keys_unknown_right_level_yields_none_in_remap() {
        let left = DataFrame::from_columns(vec![
            ("k".into(), cat_col(&["a", "b"], &[0, 1])),
        ])
        .unwrap();
        let right = DataFrame::from_columns(vec![
            ("k".into(), cat_col(&["a", "z"], &[0, 1])),
        ])
        .unwrap();
        let cat = collect_categorical_join_keys(&left, &[0], &right, &[0]).unwrap();
        // "a" exists on the left → Some(0). "z" does not → None.
        assert_eq!(cat.right_to_left[0], vec![Some(0u32), None]);
    }

    #[test]
    fn phase4_column_to_categorical_column_roundtrip() {
        let original = cat_col(&["red", "green", "blue"], &[0, 1, 2, 1, 0]);
        let cc = original.to_categorical_column().unwrap();
        let restored = Column::from_categorical_column(&cc).unwrap();
        match (&original, &restored) {
            (
                Column::Categorical { levels: l1, codes: c1 },
                Column::Categorical { levels: l2, codes: c2 },
            ) => {
                assert_eq!(l1, l2);
                assert_eq!(c1, c2);
            }
            _ => panic!("expected Categorical"),
        }
    }

    #[test]
    fn phase4_column_to_categorical_column_none_for_non_categorical() {
        assert!(Column::Int(vec![1, 2, 3]).to_categorical_column().is_none());
        assert!(Column::Str(vec!["a".into()]).to_categorical_column().is_none());
        assert!(Column::Float(vec![1.0]).to_categorical_column().is_none());
    }

    #[test]
    fn phase4_column_from_categorical_column_rejects_nulls() {
        // CategoricalColumn with a null cannot map to Column::Categorical
        // (which has no null bitmap). Verifies the safety check.
        use crate::byte_dict::CategoricalColumn;
        let mut cc = CategoricalColumn::new();
        cc.push(b"a").unwrap();
        cc.push_null();
        cc.push(b"b").unwrap();
        assert!(Column::from_categorical_column(&cc).is_none());
    }

    #[test]
    fn phase4_column_from_categorical_column_rejects_non_utf8() {
        use crate::byte_dict::CategoricalColumn;
        let mut cc = CategoricalColumn::new();
        // 0xFF is invalid as a standalone UTF-8 byte.
        cc.push(&[0xFFu8]).unwrap();
        assert!(Column::from_categorical_column(&cc).is_none());
    }
}

// â"€â"€ Phase 17 NoGC audit notes â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
//
// Safe (@nogc â€" metadata only, O(L) or O(N) over Rust-heap Vec only):
//   fct_collapse    : rewrites levels Vec + remap data Vec â€" both are Rust heap,
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
