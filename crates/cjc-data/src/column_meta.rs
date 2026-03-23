//! Column-level metadata for query optimization.
//!
//! Zone maps (min/max per column) and sorted flags enable
//! the TidyView optimizer to skip unnecessary work.

use crate::Column;
use std::collections::BTreeSet;

/// Comparison operator for filter predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
}

/// Statistical metadata for a single column.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Minimum value (for numeric columns)
    pub min_f64: Option<f64>,
    /// Maximum value (for numeric columns)
    pub max_f64: Option<f64>,
    /// Minimum value (for int columns)
    pub min_i64: Option<i64>,
    /// Maximum value (for int columns)
    pub max_i64: Option<i64>,
    /// Whether the column is sorted ascending
    pub sorted_asc: bool,
    /// Whether the column is sorted descending
    pub sorted_desc: bool,
    /// Number of distinct values (exact for small columns, estimated for large)
    pub n_distinct: Option<usize>,
    /// Number of null/NaN values
    pub n_null: usize,
    /// Total number of rows
    pub nrows: usize,
}

impl ColumnStats {
    /// Compute stats for a column.
    pub fn compute(col: &Column) -> Self {
        match col {
            Column::Float(v) => Self::compute_float(v),
            Column::Int(v) => Self::compute_int(v),
            Column::Str(v) => Self::compute_str(v),
            Column::Bool(v) => Self::compute_bool(v),
            Column::Categorical { codes, levels } => Self::compute_categorical(codes, levels),
            Column::DateTime(v) => Self::compute_datetime(v),
        }
    }

    fn compute_float(v: &[f64]) -> Self {
        if v.is_empty() {
            return Self::empty();
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut n_null = 0usize;
        let mut sorted_asc = true;
        let mut sorted_desc = true;
        let mut prev: Option<f64> = None;

        for &val in v {
            if val.is_nan() {
                n_null += 1;
                // NaN breaks sort order
                if prev.is_some() {
                    sorted_asc = false;
                    sorted_desc = false;
                }
            } else {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
                if let Some(p) = prev {
                    if !p.is_nan() {
                        if val < p {
                            sorted_asc = false;
                        }
                        if val > p {
                            sorted_desc = false;
                        }
                    }
                }
            }
            prev = Some(val);
        }

        ColumnStats {
            min_f64: if min.is_finite() { Some(min) } else { None },
            max_f64: if max.is_finite() { Some(max) } else { None },
            min_i64: None,
            max_i64: None,
            sorted_asc,
            sorted_desc,
            n_distinct: None,
            n_null,
            nrows: v.len(),
        }
    }

    fn compute_int(v: &[i64]) -> Self {
        if v.is_empty() {
            return Self::empty();
        }

        let mut min = i64::MAX;
        let mut max = i64::MIN;
        let mut sorted_asc = true;
        let mut sorted_desc = true;

        for i in 0..v.len() {
            let val = v[i];
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
            if i > 0 {
                if val < v[i - 1] {
                    sorted_asc = false;
                }
                if val > v[i - 1] {
                    sorted_desc = false;
                }
            }
        }

        ColumnStats {
            min_f64: None,
            max_f64: None,
            min_i64: Some(min),
            max_i64: Some(max),
            sorted_asc,
            sorted_desc,
            n_distinct: None,
            n_null: 0,
            nrows: v.len(),
        }
    }

    fn compute_str(v: &[String]) -> Self {
        if v.is_empty() {
            return Self::empty();
        }

        let mut sorted_asc = true;
        let mut sorted_desc = true;
        // Use BTreeSet for deterministic distinct counting
        let mut distinct = BTreeSet::new();

        for i in 0..v.len() {
            distinct.insert(v[i].as_str());
            if i > 0 {
                if v[i] < v[i - 1] {
                    sorted_asc = false;
                }
                if v[i] > v[i - 1] {
                    sorted_desc = false;
                }
            }
        }

        ColumnStats {
            min_f64: None,
            max_f64: None,
            min_i64: None,
            max_i64: None,
            sorted_asc,
            sorted_desc,
            n_distinct: Some(distinct.len()),
            n_null: 0,
            nrows: v.len(),
        }
    }

    fn compute_bool(v: &[bool]) -> Self {
        if v.is_empty() {
            return Self::empty();
        }

        let mut sorted_asc = true;
        let mut sorted_desc = true;
        let mut has_true = false;
        let mut has_false = false;

        for i in 0..v.len() {
            if v[i] {
                has_true = true;
            } else {
                has_false = true;
            }
            if i > 0 {
                // false < true in Rust's Ord
                if v[i] < v[i - 1] {
                    sorted_asc = false;
                }
                if v[i] > v[i - 1] {
                    sorted_desc = false;
                }
            }
        }

        let n_distinct = match (has_true, has_false) {
            (true, true) => 2,
            (true, false) | (false, true) => 1,
            _ => 0,
        };

        ColumnStats {
            min_f64: None,
            max_f64: None,
            min_i64: None,
            max_i64: None,
            sorted_asc,
            sorted_desc,
            n_distinct: Some(n_distinct),
            n_null: 0,
            nrows: v.len(),
        }
    }

    fn compute_categorical(codes: &[u32], levels: &[String]) -> Self {
        if codes.is_empty() {
            return Self::empty();
        }

        let mut min = u32::MAX;
        let mut max = u32::MIN;
        let mut sorted_asc = true;
        let mut sorted_desc = true;
        let mut distinct = BTreeSet::new();

        for i in 0..codes.len() {
            let c = codes[i];
            if c < min {
                min = c;
            }
            if c > max {
                max = c;
            }
            distinct.insert(c);
            if i > 0 {
                if c < codes[i - 1] {
                    sorted_asc = false;
                }
                if c > codes[i - 1] {
                    sorted_desc = false;
                }
            }
        }

        ColumnStats {
            min_f64: None,
            max_f64: None,
            min_i64: Some(min as i64),
            max_i64: Some(max as i64),
            sorted_asc,
            sorted_desc,
            n_distinct: Some(distinct.len().min(levels.len())),
            n_null: 0,
            nrows: codes.len(),
        }
    }

    fn compute_datetime(v: &[i64]) -> Self {
        // DateTime columns store epoch millis as i64; reuse int logic.
        Self::compute_int(v)
    }

    fn empty() -> Self {
        ColumnStats {
            min_f64: None,
            max_f64: None,
            min_i64: None,
            max_i64: None,
            sorted_asc: true,  // vacuously sorted
            sorted_desc: true, // vacuously sorted
            n_distinct: Some(0),
            n_null: 0,
            nrows: 0,
        }
    }

    // ── Zone-map skip predicates ──────────────────────────────────────

    /// Check if a filter `col > threshold` can be ruled out (all values <= threshold).
    pub fn can_skip_gt(&self, threshold: f64) -> bool {
        self.max_f64.map_or(false, |max| max <= threshold)
    }

    /// Check if a filter `col < threshold` can be ruled out (all values >= threshold).
    pub fn can_skip_lt(&self, threshold: f64) -> bool {
        self.min_f64.map_or(false, |min| min >= threshold)
    }

    /// Check if a filter `col >= threshold` can be ruled out (all values < threshold).
    pub fn can_skip_ge(&self, threshold: f64) -> bool {
        self.max_f64.map_or(false, |max| max < threshold)
    }

    /// Check if a filter `col <= threshold` can be ruled out (all values > threshold).
    pub fn can_skip_le(&self, threshold: f64) -> bool {
        self.min_f64.map_or(false, |min| min > threshold)
    }

    /// Check if a filter `col == value` can be ruled out (value outside [min, max]).
    pub fn can_skip_eq_f64(&self, value: f64) -> bool {
        match (self.min_f64, self.max_f64) {
            (Some(min), Some(max)) => value < min || value > max,
            _ => false,
        }
    }

    /// Check if a filter `col > threshold` can be ruled out for int columns.
    pub fn can_skip_gt_i64(&self, threshold: i64) -> bool {
        self.max_i64.map_or(false, |max| max <= threshold)
    }

    /// Check if a filter `col < threshold` can be ruled out for int columns.
    pub fn can_skip_lt_i64(&self, threshold: i64) -> bool {
        self.min_i64.map_or(false, |min| min >= threshold)
    }

    /// Check if a filter `col == value` can be ruled out for int columns.
    pub fn can_skip_eq_i64(&self, value: i64) -> bool {
        match (self.min_i64, self.max_i64) {
            (Some(min), Some(max)) => value < min || value > max,
            _ => false,
        }
    }

    // ── Sorted-column helpers ─────────────────────────────────────────

    /// Check if column is sorted (ascending or descending).
    pub fn is_sorted(&self) -> bool {
        self.sorted_asc || self.sorted_desc
    }

    /// For a sorted ascending float column, find the row range satisfying a
    /// comparison predicate via binary search. Returns `(start, end)` where
    /// matching rows are `start..end`.
    ///
    /// Caller must ensure the slice is actually sorted ascending.
    pub fn binary_search_range_f64(col: &[f64], op: CompOp, threshold: f64) -> (usize, usize) {
        match op {
            CompOp::Gt => {
                let start = col.partition_point(|&v| v <= threshold);
                (start, col.len())
            }
            CompOp::Ge => {
                let start = col.partition_point(|&v| v < threshold);
                (start, col.len())
            }
            CompOp::Lt => {
                let end = col.partition_point(|&v| v < threshold);
                (0, end)
            }
            CompOp::Le => {
                let end = col.partition_point(|&v| v <= threshold);
                (0, end)
            }
            CompOp::Eq => {
                let start = col.partition_point(|&v| v < threshold);
                let end = col.partition_point(|&v| v <= threshold);
                (start, end)
            }
        }
    }

    /// For a sorted ascending int column, find the row range satisfying a
    /// comparison predicate via binary search. Returns `(start, end)`.
    pub fn binary_search_range_i64(col: &[i64], op: CompOp, threshold: i64) -> (usize, usize) {
        match op {
            CompOp::Gt => {
                let start = col.partition_point(|&v| v <= threshold);
                (start, col.len())
            }
            CompOp::Ge => {
                let start = col.partition_point(|&v| v < threshold);
                (start, col.len())
            }
            CompOp::Lt => {
                let end = col.partition_point(|&v| v < threshold);
                (0, end)
            }
            CompOp::Le => {
                let end = col.partition_point(|&v| v <= threshold);
                (0, end)
            }
            CompOp::Eq => {
                let start = col.partition_point(|&v| v < threshold);
                let end = col.partition_point(|&v| v <= threshold);
                (start, end)
            }
        }
    }
}

/// Metadata for an entire DataFrame.
#[derive(Debug, Clone)]
pub struct DataFrameStats {
    /// Per-column stats, in column order.
    pub column_stats: Vec<(String, ColumnStats)>,
}

impl DataFrameStats {
    /// Compute stats for all columns in a DataFrame.
    pub fn compute(df: &crate::DataFrame) -> Self {
        let column_stats = df
            .columns
            .iter()
            .map(|(name, col)| (name.clone(), ColumnStats::compute(col)))
            .collect();
        DataFrameStats { column_stats }
    }

    /// Get stats for a named column.
    pub fn get(&self, name: &str) -> Option<&ColumnStats> {
        self.column_stats
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, DataFrame};

    // 1. Float column min/max computation
    #[test]
    fn test_float_min_max() {
        let col = Column::Float(vec![3.0, 1.0, 4.0, 1.5, 9.2]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_f64, Some(1.0));
        assert_eq!(stats.max_f64, Some(9.2));
        assert_eq!(stats.nrows, 5);
        assert_eq!(stats.n_null, 0);
    }

    // 2. Int column min/max computation
    #[test]
    fn test_int_min_max() {
        let col = Column::Int(vec![10, -3, 42, 0, 7]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_i64, Some(-3));
        assert_eq!(stats.max_i64, Some(42));
        assert_eq!(stats.nrows, 5);
    }

    // 3. Sorted ascending detection
    #[test]
    fn test_sorted_ascending() {
        let col = Column::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = ColumnStats::compute(&col);
        assert!(stats.sorted_asc);
        assert!(!stats.sorted_desc);
        assert!(stats.is_sorted());
    }

    // 4. Sorted descending detection
    #[test]
    fn test_sorted_descending() {
        let col = Column::Int(vec![50, 40, 30, 20, 10]);
        let stats = ColumnStats::compute(&col);
        assert!(!stats.sorted_asc);
        assert!(stats.sorted_desc);
        assert!(stats.is_sorted());
    }

    // 5. Unsorted detection
    #[test]
    fn test_unsorted() {
        let col = Column::Float(vec![3.0, 1.0, 4.0, 1.5, 9.2]);
        let stats = ColumnStats::compute(&col);
        assert!(!stats.sorted_asc);
        assert!(!stats.sorted_desc);
        assert!(!stats.is_sorted());
    }

    // 6a. can_skip_gt
    #[test]
    fn test_can_skip_gt() {
        let col = Column::Float(vec![1.0, 2.0, 3.0]);
        let stats = ColumnStats::compute(&col);
        // max is 3.0, threshold 5.0 => all values <= 5.0, can skip
        assert!(stats.can_skip_gt(5.0));
        // max is 3.0, threshold 3.0 => all values <= 3.0, can skip
        assert!(stats.can_skip_gt(3.0));
        // max is 3.0, threshold 2.0 => some values > 2.0, cannot skip
        assert!(!stats.can_skip_gt(2.0));
    }

    // 6b. can_skip_lt
    #[test]
    fn test_can_skip_lt() {
        let col = Column::Float(vec![5.0, 6.0, 7.0]);
        let stats = ColumnStats::compute(&col);
        // min is 5.0, threshold 3.0 => all values >= 3.0, can skip
        assert!(stats.can_skip_lt(3.0));
        // min is 5.0, threshold 5.0 => all values >= 5.0, can skip
        assert!(stats.can_skip_lt(5.0));
        // min is 5.0, threshold 6.0 => some values < 6.0, cannot skip
        assert!(!stats.can_skip_lt(6.0));
    }

    // 6c. can_skip_ge
    #[test]
    fn test_can_skip_ge() {
        let col = Column::Float(vec![1.0, 2.0, 3.0]);
        let stats = ColumnStats::compute(&col);
        // max is 3.0, threshold 4.0 => all values < 4.0, can skip col >= 4.0
        assert!(stats.can_skip_ge(4.0));
        // max is 3.0, threshold 3.0 => 3.0 >= 3.0 is true, cannot skip
        assert!(!stats.can_skip_ge(3.0));
    }

    // 6d. can_skip_le
    #[test]
    fn test_can_skip_le() {
        let col = Column::Float(vec![5.0, 6.0, 7.0]);
        let stats = ColumnStats::compute(&col);
        // min is 5.0, threshold 4.0 => all values > 4.0, can skip col <= 4.0
        assert!(stats.can_skip_le(4.0));
        // min is 5.0, threshold 5.0 => 5.0 <= 5.0 is true, cannot skip
        assert!(!stats.can_skip_le(5.0));
    }

    // 6e. can_skip_eq
    #[test]
    fn test_can_skip_eq() {
        let col = Column::Float(vec![10.0, 20.0, 30.0]);
        let stats = ColumnStats::compute(&col);
        // value outside range
        assert!(stats.can_skip_eq_f64(5.0));
        assert!(stats.can_skip_eq_f64(35.0));
        // value inside range
        assert!(!stats.can_skip_eq_f64(15.0));
        assert!(!stats.can_skip_eq_f64(10.0));
        assert!(!stats.can_skip_eq_f64(30.0));
    }

    // 7. Binary search range on sorted float column
    #[test]
    fn test_binary_search_range_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // col > 3.0 => rows [3, 4] (values 4.0, 5.0)
        assert_eq!(
            ColumnStats::binary_search_range_f64(&data, CompOp::Gt, 3.0),
            (3, 5)
        );
        // col >= 3.0 => rows [2, 3, 4] (values 3.0, 4.0, 5.0)
        assert_eq!(
            ColumnStats::binary_search_range_f64(&data, CompOp::Ge, 3.0),
            (2, 5)
        );
        // col < 3.0 => rows [0, 1] (values 1.0, 2.0)
        assert_eq!(
            ColumnStats::binary_search_range_f64(&data, CompOp::Lt, 3.0),
            (0, 2)
        );
        // col <= 3.0 => rows [0, 1, 2] (values 1.0, 2.0, 3.0)
        assert_eq!(
            ColumnStats::binary_search_range_f64(&data, CompOp::Le, 3.0),
            (0, 3)
        );
        // col == 3.0 => row [2] (value 3.0)
        assert_eq!(
            ColumnStats::binary_search_range_f64(&data, CompOp::Eq, 3.0),
            (2, 3)
        );
        // col == 2.5 => no match
        assert_eq!(
            ColumnStats::binary_search_range_f64(&data, CompOp::Eq, 2.5),
            (2, 2)
        );
    }

    // 7b. Binary search range on sorted int column
    #[test]
    fn test_binary_search_range_i64() {
        let data = vec![10, 20, 30, 40, 50];

        assert_eq!(
            ColumnStats::binary_search_range_i64(&data, CompOp::Gt, 30),
            (3, 5)
        );
        assert_eq!(
            ColumnStats::binary_search_range_i64(&data, CompOp::Eq, 30),
            (2, 3)
        );
        assert_eq!(
            ColumnStats::binary_search_range_i64(&data, CompOp::Lt, 10),
            (0, 0)
        );
    }

    // 8. NaN handling: NaN does not affect min/max
    #[test]
    fn test_nan_handling() {
        let col = Column::Float(vec![f64::NAN, 2.0, f64::NAN, 5.0, 1.0]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_f64, Some(1.0));
        assert_eq!(stats.max_f64, Some(5.0));
        assert_eq!(stats.n_null, 2);
        assert_eq!(stats.nrows, 5);
    }

    // 8b. All-NaN column
    #[test]
    fn test_all_nan_column() {
        let col = Column::Float(vec![f64::NAN, f64::NAN]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_f64, None);
        assert_eq!(stats.max_f64, None);
        assert_eq!(stats.n_null, 2);
        // can_skip should return false (no stats available)
        assert!(!stats.can_skip_gt(0.0));
    }

    // 9. Empty column stats
    #[test]
    fn test_empty_column() {
        let col = Column::Float(vec![]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_f64, None);
        assert_eq!(stats.max_f64, None);
        assert_eq!(stats.nrows, 0);
        assert!(stats.sorted_asc);
        assert!(stats.sorted_desc);
        assert!(stats.is_sorted());
        assert_eq!(stats.n_distinct, Some(0));
    }

    // 9b. Empty int column
    #[test]
    fn test_empty_int_column() {
        let col = Column::Int(vec![]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_i64, None);
        assert_eq!(stats.max_i64, None);
        assert_eq!(stats.nrows, 0);
        assert!(stats.is_sorted());
    }

    // 10. String column n_distinct
    #[test]
    fn test_str_n_distinct() {
        let col = Column::Str(vec![
            "apple".into(),
            "banana".into(),
            "apple".into(),
            "cherry".into(),
        ]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.n_distinct, Some(3));
        assert_eq!(stats.nrows, 4);
    }

    // 10b. Sorted string column
    #[test]
    fn test_sorted_str() {
        let col = Column::Str(vec!["a".into(), "b".into(), "c".into()]);
        let stats = ColumnStats::compute(&col);
        assert!(stats.sorted_asc);
        assert!(!stats.sorted_desc);
    }

    // 11. DataFrameStats compute + get
    #[test]
    fn test_dataframe_stats() {
        let mut df = DataFrame::new();
        df.columns
            .push(("age".into(), Column::Int(vec![25, 30, 45])));
        df.columns
            .push(("score".into(), Column::Float(vec![88.5, 92.0, 76.3])));

        let stats = DataFrameStats::compute(&df);
        assert_eq!(stats.column_stats.len(), 2);

        let age_stats = stats.get("age").unwrap();
        assert_eq!(age_stats.min_i64, Some(25));
        assert_eq!(age_stats.max_i64, Some(45));

        let score_stats = stats.get("score").unwrap();
        assert_eq!(score_stats.min_f64, Some(76.3));
        assert_eq!(score_stats.max_f64, Some(92.0));

        // Non-existent column
        assert!(stats.get("nonexistent").is_none());
    }

    // 12. Zone map correctly identifies "can skip entire column"
    #[test]
    fn test_zone_map_can_skip() {
        // Column with values [100, 200, 300]
        let col = Column::Float(vec![100.0, 200.0, 300.0]);
        let stats = ColumnStats::compute(&col);

        // Filter: col > 500 => max=300 <= 500, can skip
        assert!(stats.can_skip_gt(500.0));
        // Filter: col < 50 => min=100 >= 50, can skip
        assert!(stats.can_skip_lt(50.0));
        // Filter: col == 400 => 400 > max, can skip
        assert!(stats.can_skip_eq_f64(400.0));
        // Filter: col == 50 => 50 < min, can skip
        assert!(stats.can_skip_eq_f64(50.0));
    }

    // 13. Zone map correctly identifies "cannot skip"
    #[test]
    fn test_zone_map_cannot_skip() {
        let col = Column::Float(vec![100.0, 200.0, 300.0]);
        let stats = ColumnStats::compute(&col);

        // Filter: col > 150 => max=300 > 150, cannot skip
        assert!(!stats.can_skip_gt(150.0));
        // Filter: col < 250 => min=100 < 250, cannot skip
        assert!(!stats.can_skip_lt(250.0));
        // Filter: col == 200 => in [100, 300], cannot skip
        assert!(!stats.can_skip_eq_f64(200.0));
    }

    // 14. Determinism test: 3 runs produce identical stats
    #[test]
    fn test_determinism() {
        let data = vec![3.14, 2.71, 1.41, 1.73, 0.577];
        for _ in 0..3 {
            let col = Column::Float(data.clone());
            let stats = ColumnStats::compute(&col);
            assert_eq!(stats.min_f64, Some(0.577));
            assert_eq!(stats.max_f64, Some(3.14));
            assert!(!stats.sorted_asc);
            assert!(!stats.sorted_desc);
            assert_eq!(stats.n_null, 0);
            assert_eq!(stats.nrows, 5);
        }
    }

    // 14b. Determinism for int + str columns
    #[test]
    fn test_determinism_int_str() {
        for _ in 0..3 {
            let icol = Column::Int(vec![9, 1, 5, 3, 7]);
            let is = ColumnStats::compute(&icol);
            assert_eq!(is.min_i64, Some(1));
            assert_eq!(is.max_i64, Some(9));

            let scol = Column::Str(vec!["x".into(), "a".into(), "m".into()]);
            let ss = ColumnStats::compute(&scol);
            assert_eq!(ss.n_distinct, Some(3));
        }
    }

    // 15. Bool column stats
    #[test]
    fn test_bool_column() {
        let col = Column::Bool(vec![true, false, true, true]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.n_distinct, Some(2));
        assert_eq!(stats.nrows, 4);
        assert!(!stats.sorted_asc);
        assert!(!stats.sorted_desc);
    }

    // 16. Categorical column
    #[test]
    fn test_categorical_column() {
        let col = Column::Categorical {
            levels: vec!["low".into(), "med".into(), "high".into()],
            codes: vec![0, 1, 2, 1, 0],
        };
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_i64, Some(0));
        assert_eq!(stats.max_i64, Some(2));
        assert_eq!(stats.n_distinct, Some(3));
        assert!(!stats.is_sorted());
    }

    // 17. Single-element column
    #[test]
    fn test_single_element() {
        let col = Column::Float(vec![42.0]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_f64, Some(42.0));
        assert_eq!(stats.max_f64, Some(42.0));
        assert!(stats.sorted_asc);
        assert!(stats.sorted_desc);
        assert!(stats.is_sorted());
    }

    // 18. Int zone-map skip predicates
    #[test]
    fn test_int_zone_map_skip() {
        let col = Column::Int(vec![10, 20, 30]);
        let stats = ColumnStats::compute(&col);
        assert!(stats.can_skip_gt_i64(30));
        assert!(stats.can_skip_gt_i64(50));
        assert!(!stats.can_skip_gt_i64(15));
        assert!(stats.can_skip_lt_i64(5));
        assert!(stats.can_skip_lt_i64(10));
        assert!(!stats.can_skip_lt_i64(20));
        assert!(stats.can_skip_eq_i64(5));
        assert!(stats.can_skip_eq_i64(35));
        assert!(!stats.can_skip_eq_i64(20));
    }

    // 19. Constant column (all same value)
    #[test]
    fn test_constant_column() {
        let col = Column::Float(vec![7.0, 7.0, 7.0]);
        let stats = ColumnStats::compute(&col);
        assert_eq!(stats.min_f64, Some(7.0));
        assert_eq!(stats.max_f64, Some(7.0));
        // A constant column is both ascending and descending
        assert!(stats.sorted_asc);
        assert!(stats.sorted_desc);
        assert!(stats.can_skip_eq_f64(8.0));
        assert!(!stats.can_skip_eq_f64(7.0));
    }
}
