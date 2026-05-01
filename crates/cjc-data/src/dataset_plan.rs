//! Phase 1 — Deterministic ML training-data plan.
//!
//! `DatasetPlan` is a small, immutable composition over a [`TidyView`]
//! that adds the four ML-specific concerns missing from the data engine:
//!
//! 1. **Feature/label column selection** with deterministic ordering.
//! 2. **Encoding**: float / int / bool / categorical → `f64` features.
//! 3. **Train/val/test splits**, sequential or hashed-by-row.
//! 4. **Batching** with optional seeded SplitMix64 shuffle, materializing
//!    each batch into a row-major `Tensor`.
//!
//! Phase 1 is Rust-only; not yet exposed to `.cjcl`. That's Phase 3. Phase
//! 6 will wire `plan_hash` into a training manifest — for now the field is
//! reserved (`Option<[u8; 32]>`, always `None`).
//!
//! ## Determinism contract
//!
//! - Row IDs are always ascending `u32` by default; `TidyView` already
//!   guarantees this for the underlying selection.
//! - Shuffles use `cjc_repro::Rng::seeded(seed)` (SplitMix64) with
//!   Fisher-Yates over the split's row vector.
//! - Hashed splits use the fixed `splitmix64` mixer keyed by `row ^ seed`.
//! - Categorical dictionaries are built over **all source rows** (not just
//!   train) so val/test see codes consistent with train, then frozen
//!   before any batch is materialized.
//! - Tensor materialization is row-major; no reductions, no FMA — bit
//!   copies only.
//!
//! ## Reuse map
//!
//! | Need                              | Existing primitive                       |
//! |-----------------------------------|------------------------------------------|
//! | Filter / project upstream         | `TidyView::filter`, `TidyView::select`   |
//! | Row mask                          | `AdaptiveSelection` inside the TidyView  |
//! | Categorical encoding              | `ByteDictionary::intern` + `freeze`      |
//! | Column-name → encoding map        | `detcoll::SortedVecMap`                  |
//! | Seeded RNG                        | `cjc_repro::Rng::seeded` (SplitMix64)    |

use crate::byte_dict::{ByteDictionary, CategoryOrdering};
use crate::detcoll::SortedVecMap;
use crate::{Column, DataFrame, TidyError, TidyView};
use cjc_repro::Rng;
use cjc_runtime::tensor::Tensor;

// ════════════════════════════════════════════════════════════════════════
//  Errors
// ════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub enum DatasetError {
    UnknownColumn(String),
    UnsupportedColumnType {
        column: String,
        type_name: &'static str,
    },
    EncodingMismatch {
        column: String,
        encoding: &'static str,
        column_type: &'static str,
    },
    /// Categorical encoding requested but the column row at this index is
    /// null. Phase 1 does not have a null policy — this is an error.
    NullCategorical {
        column: String,
        row: u32,
    },
    EmptySplit(Split),
    /// Fractions must each be in `[0, 1]` and sum to ≤ 1.
    InvalidFractions {
        train: f64,
        val: f64,
        test: f64,
    },
    BadBatchSize(usize),
    NoFeatures,
    /// Encoding was registered for a column not in `feature_cols` or
    /// `label_col`.
    OrphanEncoding(String),
    Tidy(String),
    Shape(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::UnknownColumn(c) => write!(f, "unknown column `{c}`"),
            DatasetError::UnsupportedColumnType { column, type_name } => write!(
                f,
                "column `{column}` has type `{type_name}` which is not supported"
            ),
            DatasetError::EncodingMismatch {
                column,
                encoding,
                column_type,
            } => write!(
                f,
                "column `{column}` (type `{column_type}`) cannot be encoded as `{encoding}`"
            ),
            DatasetError::NullCategorical { column, row } => {
                write!(f, "null value in categorical column `{column}` at row {row}")
            }
            DatasetError::EmptySplit(s) => write!(f, "split `{s:?}` is empty"),
            DatasetError::InvalidFractions { train, val, test } => write!(
                f,
                "invalid split fractions train={train}, val={val}, test={test} \
                 (each must be in [0,1] and sum ≤ 1)"
            ),
            DatasetError::BadBatchSize(n) => write!(f, "batch_size must be ≥ 1 (got {n})"),
            DatasetError::NoFeatures => write!(f, "no feature columns specified"),
            DatasetError::OrphanEncoding(c) => {
                write!(f, "encoding registered for column `{c}` but it is neither a feature nor the label")
            }
            DatasetError::Tidy(m) => write!(f, "tidy error: {m}"),
            DatasetError::Shape(m) => write!(f, "shape error: {m}"),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<TidyError> for DatasetError {
    fn from(e: TidyError) -> Self {
        DatasetError::Tidy(format!("{e:?}"))
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Specs
// ════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    Train,
    Val,
    Test,
    Full,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SplitSpec {
    /// Single full-dataset partition. `Split::Train`, `Split::Val`, and
    /// `Split::Test` all yield empty row vectors; only `Split::Full`
    /// returns rows.
    Full,
    /// Sequential ranges by ascending row index. Train gets the first
    /// `floor(nrows * train)`, val the next `floor(nrows * val)`, test
    /// the next `floor(nrows * test)`. Trailing rows are excluded.
    Sequential { train: f64, val: f64, test: f64 },
    /// Per-row deterministic hash assignment. Bucket =
    /// `splitmix64(row as u64 ^ seed) >> 32` divided by `2^32`. Same
    /// `seed` ⇒ identical assignment, regardless of `nrows`.
    Hashed {
        seed: u64,
        train: f64,
        val: f64,
        test: f64,
    },
}

impl SplitSpec {
    fn validate(&self) -> Result<(), DatasetError> {
        let (t, v, te) = match self {
            SplitSpec::Full => return Ok(()),
            SplitSpec::Sequential { train, val, test } => (*train, *val, *test),
            SplitSpec::Hashed {
                train, val, test, ..
            } => (*train, *val, *test),
        };
        let valid_each = (0.0..=1.0).contains(&t)
            && (0.0..=1.0).contains(&v)
            && (0.0..=1.0).contains(&te);
        let sum = t + v + te;
        if !valid_each || sum > 1.0 + 1e-9 {
            return Err(DatasetError::InvalidFractions {
                train: t,
                val: v,
                test: te,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchSpec {
    pub batch_size: usize,
    pub drop_last: bool,
    /// `None` ⇒ ascending row order. `Some(seed)` ⇒ SplitMix64
    /// Fisher-Yates permutation of the split's row IDs.
    pub shuffle: Option<u64>,
}

impl Default for BatchSpec {
    fn default() -> Self {
        Self {
            batch_size: 1,
            drop_last: false,
            shuffle: None,
        }
    }
}

impl BatchSpec {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            drop_last: false,
            shuffle: None,
        }
    }
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(seed);
        self
    }
}

/// Per-column encoding directive. Each feature/label column must have
/// one and only one of these. Phase 1 supports four encodings; richer
/// schemes (one-hot, embedding lookup) are deferred to Phase 3.
#[derive(Debug, Clone, PartialEq)]
pub enum EncodingSpec {
    /// `Column::Float` → `f64` pass-through.
    Float,
    /// `Column::Int` → `f64` via `as f64` cast (lossy for |x| ≥ 2^53;
    /// caller's responsibility to know).
    IntAsFloat,
    /// `Column::Bool` → `0.0` / `1.0`.
    BoolAsFloat,
    /// `Column::Str | Categorical | CategoricalAdaptive` → integer code
    /// from a fresh `ByteDictionary` (cast to `f64`). Dictionary built
    /// over **all source rows** before any split is materialized.
    Categorical { ordering: CategoryOrdering },
}

impl EncodingSpec {
    fn name(&self) -> &'static str {
        match self {
            EncodingSpec::Float => "Float",
            EncodingSpec::IntAsFloat => "IntAsFloat",
            EncodingSpec::BoolAsFloat => "BoolAsFloat",
            EncodingSpec::Categorical { .. } => "Categorical",
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
//  DatasetPlan
// ════════════════════════════════════════════════════════════════════════

/// Immutable training-data plan. Cheap to clone (`TidyView` holds
/// `Rc<DataFrame>`).
#[derive(Clone)]
pub struct DatasetPlan {
    source: TidyView,
    feature_cols: Vec<String>,
    label_col: Option<String>,
    encodings: SortedVecMap<String, EncodingSpec>,
    split: SplitSpec,
    batch: BatchSpec,
    /// Phase 6 reserved field. Always `None` today.
    plan_hash: Option<[u8; 32]>,
}

impl DatasetPlan {
    pub fn from_view(source: TidyView) -> Self {
        Self {
            source,
            feature_cols: Vec::new(),
            label_col: None,
            encodings: SortedVecMap::new(),
            split: SplitSpec::Full,
            batch: BatchSpec::default(),
            plan_hash: None,
        }
    }

    pub fn from_dataframe(df: DataFrame) -> Self {
        Self::from_view(df.tidy())
    }

    pub fn with_features(mut self, cols: Vec<String>) -> Self {
        self.feature_cols = cols;
        self
    }

    pub fn with_label(mut self, col: String) -> Self {
        self.label_col = Some(col);
        self
    }

    pub fn with_encoding(mut self, col: String, enc: EncodingSpec) -> Self {
        self.encodings.insert(col, enc);
        self
    }

    pub fn with_split(mut self, split: SplitSpec) -> Self {
        self.split = split;
        self
    }

    pub fn with_batch(mut self, batch: BatchSpec) -> Self {
        self.batch = batch;
        self
    }

    pub fn nrows(&self) -> usize {
        self.source.nrows()
    }
    pub fn n_features(&self) -> usize {
        self.feature_cols.len()
    }
    pub fn feature_cols(&self) -> &[String] {
        &self.feature_cols
    }
    pub fn label_col(&self) -> Option<&str> {
        self.label_col.as_deref()
    }
    pub fn split_spec(&self) -> &SplitSpec {
        &self.split
    }
    pub fn batch_spec(&self) -> &BatchSpec {
        &self.batch
    }
    pub fn plan_hash(&self) -> Option<&[u8; 32]> {
        self.plan_hash.as_ref()
    }

    /// Validate the plan against the source schema. Cheap; no
    /// materialization. Called automatically by `iter_batches` and
    /// `split_rows`; useful in tests / dry-runs.
    pub fn validate(&self) -> Result<(), DatasetError> {
        if self.feature_cols.is_empty() {
            return Err(DatasetError::NoFeatures);
        }
        if self.batch.batch_size == 0 {
            return Err(DatasetError::BadBatchSize(self.batch.batch_size));
        }
        self.split.validate()?;

        let known: std::collections::BTreeSet<&str> =
            self.source.column_names().into_iter().collect();
        for c in &self.feature_cols {
            if !known.contains(c.as_str()) {
                return Err(DatasetError::UnknownColumn(c.clone()));
            }
        }
        if let Some(l) = &self.label_col {
            if !known.contains(l.as_str()) {
                return Err(DatasetError::UnknownColumn(l.clone()));
            }
        }
        for (col, _) in self.encodings.iter() {
            let in_features = self.feature_cols.iter().any(|c| c == col);
            let in_label = self.label_col.as_ref().is_some_and(|l| l == col);
            if !in_features && !in_label {
                return Err(DatasetError::OrphanEncoding(col.clone()));
            }
        }
        Ok(())
    }

    /// Ascending row IDs assigned to `which`. Row IDs are indices into the
    /// **materialized** source (post-filter, post-select), not the raw
    /// underlying DataFrame.
    pub fn split_rows(&self, which: Split) -> Result<Vec<u32>, DatasetError> {
        self.validate()?;
        let n = self.nrows();
        Ok(assign_split(n, &self.split, which))
    }

    /// Iterate batches over `which` split. Each batch is fully resolved
    /// into row-major `Tensor`s. Categorical dictionaries are built over
    /// the entire materialized source (so val/test see codes consistent
    /// with train) and frozen before iteration begins.
    pub fn iter_batches(&self, which: Split) -> Result<BatchIterator, DatasetError> {
        self.validate()?;
        let df = self.source.materialize()?;

        // Build dictionaries for any categorically-encoded column over
        // ALL source rows. Frozen after build.
        let mut dictionaries: SortedVecMap<String, ByteDictionary> = SortedVecMap::new();
        for (col, enc) in self.encodings.iter() {
            if let EncodingSpec::Categorical { ordering } = enc {
                let column = df
                    .get_column(col)
                    .ok_or_else(|| DatasetError::UnknownColumn(col.clone()))?;
                let dict = build_dict(col, column, ordering.clone())?;
                dictionaries.insert(col.clone(), dict);
            }
        }

        // Compute split rows + apply shuffle.
        let mut row_ids = assign_split(df.nrows(), &self.split, which);
        if row_ids.is_empty() && !matches!(which, Split::Full) && self.nrows() == 0 {
            return Err(DatasetError::EmptySplit(which));
        }
        if let Some(seed) = self.batch.shuffle {
            shuffle_in_place(&mut row_ids, seed);
        }

        Ok(BatchIterator {
            df,
            feature_cols: self.feature_cols.clone(),
            label_col: self.label_col.clone(),
            encodings: self.encodings.clone(),
            dictionaries,
            row_ids,
            batch_size: self.batch.batch_size,
            drop_last: self.batch.drop_last,
            cursor: 0,
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Split assignment
// ════════════════════════════════════════════════════════════════════════

#[inline]
fn splitmix64_mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn assign_split(nrows: usize, spec: &SplitSpec, which: Split) -> Vec<u32> {
    match spec {
        SplitSpec::Full => match which {
            Split::Full => (0..nrows as u32).collect(),
            _ => Vec::new(),
        },
        SplitSpec::Sequential { train, val, test } => {
            let n = nrows as f64;
            let train_n = (n * train).floor() as usize;
            let val_n = (n * val).floor() as usize;
            let test_n = (n * test).floor() as usize;
            match which {
                Split::Train => (0..train_n as u32).collect(),
                Split::Val => (train_n as u32..(train_n + val_n) as u32).collect(),
                Split::Test => {
                    let start = (train_n + val_n) as u32;
                    let end = (train_n + val_n + test_n) as u32;
                    (start..end).collect()
                }
                Split::Full => (0..nrows as u32).collect(),
            }
        }
        SplitSpec::Hashed {
            seed,
            train,
            val,
            test,
        } => {
            if matches!(which, Split::Full) {
                return (0..nrows as u32).collect();
            }
            let train_t = *train;
            let val_t = train_t + *val;
            let test_t = val_t + *test;
            let mut out = Vec::new();
            for r in 0..nrows as u32 {
                let h = splitmix64_mix((r as u64) ^ *seed);
                // Bucket in [0, 1): top 32 bits of mixed value.
                let bucket = (h >> 32) as f64 / (u32::MAX as f64 + 1.0);
                let pick = if bucket < train_t {
                    Split::Train
                } else if bucket < val_t {
                    Split::Val
                } else if bucket < test_t {
                    Split::Test
                } else {
                    continue; // excluded
                };
                if pick == which {
                    out.push(r);
                }
            }
            out
        }
    }
}

fn shuffle_in_place(rows: &mut [u32], seed: u64) {
    if rows.len() <= 1 {
        return;
    }
    let mut rng = Rng::seeded(seed);
    // Fisher-Yates: for i from n-1 down to 1, swap rows[i] with rows[j],
    // j = rng.next_u64() % (i+1).
    for i in (1..rows.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        rows.swap(i, j);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  Categorical dictionary builder
// ════════════════════════════════════════════════════════════════════════

fn build_dict(
    col_name: &str,
    column: &Column,
    ordering: CategoryOrdering,
) -> Result<ByteDictionary, DatasetError> {
    let mut dict = ByteDictionary::with_ordering(ordering);
    match column {
        Column::Str(values) => {
            for v in values {
                dict.intern(v.as_bytes())
                    .map_err(|e| DatasetError::Tidy(format!("intern: {e:?}")))?;
            }
        }
        Column::Categorical { levels, codes } => {
            for &c in codes {
                let v = &levels[c as usize];
                dict.intern(v.as_bytes())
                    .map_err(|e| DatasetError::Tidy(format!("intern: {e:?}")))?;
            }
        }
        Column::CategoricalAdaptive(cc) => {
            for i in 0..cc.len() {
                match cc.get(i) {
                    Some(b) => {
                        dict.intern(b)
                            .map_err(|e| DatasetError::Tidy(format!("intern: {e:?}")))?;
                    }
                    None => {
                        return Err(DatasetError::NullCategorical {
                            column: col_name.to_string(),
                            row: i as u32,
                        });
                    }
                }
            }
        }
        other => {
            return Err(DatasetError::EncodingMismatch {
                column: col_name.to_string(),
                encoding: "Categorical",
                column_type: other.type_name(),
            });
        }
    }
    dict.freeze();
    Ok(dict)
}

// ════════════════════════════════════════════════════════════════════════
//  BatchIterator + MaterializedBatch
// ════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MaterializedBatch {
    pub row_ids: Vec<u32>,
    /// Shape `[batch_size, n_features]`, row-major.
    pub features: Tensor,
    /// Shape `[batch_size]` (1-D). `None` if no `label_col` was set.
    pub labels: Option<Tensor>,
}

pub struct BatchIterator {
    df: DataFrame,
    feature_cols: Vec<String>,
    label_col: Option<String>,
    encodings: SortedVecMap<String, EncodingSpec>,
    dictionaries: SortedVecMap<String, ByteDictionary>,
    row_ids: Vec<u32>,
    batch_size: usize,
    drop_last: bool,
    cursor: usize,
}

impl BatchIterator {
    /// Number of rows in this iterator's split (after shuffle, before
    /// any drop_last accounting).
    pub fn split_len(&self) -> usize {
        self.row_ids.len()
    }

    /// Return the (already-shuffled) row IDs this iterator will visit.
    pub fn row_ids(&self) -> &[u32] {
        &self.row_ids
    }

    fn encode_cell(
        &self,
        col_name: &str,
        col: &Column,
        row: u32,
    ) -> Result<f64, DatasetError> {
        let enc = self.encodings.get(&col_name.to_string()).cloned();
        match (col, enc) {
            (Column::Float(v), Some(EncodingSpec::Float)) => Ok(v[row as usize]),
            (Column::Float(v), None) => Ok(v[row as usize]),
            (Column::Int(v), Some(EncodingSpec::IntAsFloat)) => Ok(v[row as usize] as f64),
            (Column::Int(v), None) => Ok(v[row as usize] as f64),
            (Column::Bool(v), Some(EncodingSpec::BoolAsFloat)) => {
                Ok(if v[row as usize] { 1.0 } else { 0.0 })
            }
            (Column::Bool(v), None) => Ok(if v[row as usize] { 1.0 } else { 0.0 }),
            (Column::Str(_), Some(EncodingSpec::Categorical { .. }))
            | (Column::Categorical { .. }, Some(EncodingSpec::Categorical { .. }))
            | (Column::CategoricalAdaptive(_), Some(EncodingSpec::Categorical { .. })) => {
                let dict = self
                    .dictionaries
                    .get(&col_name.to_string())
                    .ok_or_else(|| DatasetError::Tidy(format!(
                        "missing dictionary for column `{col_name}`"
                    )))?;
                let bytes: Vec<u8> = match col {
                    Column::Str(v) => v[row as usize].as_bytes().to_vec(),
                    Column::Categorical { levels, codes } => {
                        levels[codes[row as usize] as usize].as_bytes().to_vec()
                    }
                    Column::CategoricalAdaptive(cc) => match cc.get(row as usize) {
                        Some(b) => b.to_vec(),
                        None => {
                            return Err(DatasetError::NullCategorical {
                                column: col_name.to_string(),
                                row,
                            });
                        }
                    },
                    _ => unreachable!(),
                };
                let code = dict.lookup(&bytes).ok_or_else(|| {
                    DatasetError::Tidy(format!(
                        "value at row {row} of `{col_name}` not in frozen dictionary"
                    ))
                })?;
                Ok(code as f64)
            }
            (other, Some(enc)) => Err(DatasetError::EncodingMismatch {
                column: col_name.to_string(),
                encoding: enc.name(),
                column_type: other.type_name(),
            }),
            (other, None) => Err(DatasetError::UnsupportedColumnType {
                column: col_name.to_string(),
                type_name: other.type_name(),
            }),
        }
    }

    fn materialize_chunk(
        &self,
        chunk_rows: &[u32],
    ) -> Result<MaterializedBatch, DatasetError> {
        let n_features = self.feature_cols.len();
        let bsz = chunk_rows.len();

        // Resolve each feature column once (avoids n_rows × n_features lookups).
        let mut feat_columns: Vec<&Column> = Vec::with_capacity(n_features);
        for c in &self.feature_cols {
            let col = self
                .df
                .get_column(c)
                .ok_or_else(|| DatasetError::UnknownColumn(c.clone()))?;
            feat_columns.push(col);
        }

        let mut feat_data: Vec<f64> = Vec::with_capacity(bsz * n_features);
        for &row in chunk_rows {
            for (ci, c) in self.feature_cols.iter().enumerate() {
                feat_data.push(self.encode_cell(c, feat_columns[ci], row)?);
            }
        }
        let features = Tensor::from_vec(feat_data, &[bsz, n_features])
            .map_err(|e| DatasetError::Shape(format!("features: {e:?}")))?;

        let labels = if let Some(lcol) = &self.label_col {
            let col = self
                .df
                .get_column(lcol)
                .ok_or_else(|| DatasetError::UnknownColumn(lcol.clone()))?;
            let mut data: Vec<f64> = Vec::with_capacity(bsz);
            for &row in chunk_rows {
                data.push(self.encode_cell(lcol, col, row)?);
            }
            Some(
                Tensor::from_vec(data, &[bsz])
                    .map_err(|e| DatasetError::Shape(format!("labels: {e:?}")))?,
            )
        } else {
            None
        };

        Ok(MaterializedBatch {
            row_ids: chunk_rows.to_vec(),
            features,
            labels,
        })
    }
}

impl Iterator for BatchIterator {
    type Item = Result<MaterializedBatch, DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let total = self.row_ids.len();
        if self.cursor >= total {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(total);
        let len = end - self.cursor;
        if len < self.batch_size && self.drop_last {
            self.cursor = total;
            return None;
        }
        let chunk = self.row_ids[self.cursor..end].to_vec();
        self.cursor = end;
        Some(self.materialize_chunk(&chunk))
    }
}
