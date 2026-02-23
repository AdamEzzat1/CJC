# Phase 8: Data Logistics Engine

## Overview

Phase 8 delivers a **CSV → DataFrame → Tensor** pipeline built entirely within
CJC's zero-external-dependency constraint. It extends `cjc-data` with a
byte-slice CSV parser, streaming aggregators, and a tensor bridge, then wires
all of these as first-class builtins into both the AST tree-walk interpreter
(`cjc-eval`) and the MIR register executor (`cjc-mir-exec`).

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| Zero external deps | Pure Rust, no `csv`, `serde`, or `tokio` |
| Byte-slice first | `CsvReader::parse(&[u8])` — no file I/O, no per-field `String` during scan |
| O(ncols) streaming | `StreamingCsvProcessor` visits rows once without materialising `DataFrame` |
| Determinism | Kahan summation in streaming paths; bit-identical across runs |
| Parity gate | Every builtin tested eval == MIR |
| NoGC-safe | All new builtins are `@nogc`-safe (no heap allocation in hot paths) |

---

## New API Surface

### `cjc-data` (Rust crate)

#### `CsvConfig`

```rust
pub struct CsvConfig {
    pub delimiter:       u8,             // default: b','
    pub has_header:      bool,           // default: true
    pub max_rows:        Option<usize>,  // default: None (read all)
    pub trim_whitespace: bool,           // default: true
}
```

#### `CsvReader`

```rust
impl CsvReader {
    pub fn new(config: CsvConfig) -> Self;
    pub fn parse(&self, input: &[u8]) -> Result<DataFrame, DataError>;
}
```

**Type inference** (from first data row):

| Pattern | Inferred Type |
|---------|--------------|
| All digits, optional leading `-` | `Int` → `Column::Int` |
| Digits + one `.`, optional leading `-` | `Float` → `Column::Float` |
| `"true"`, `"false"`, `"1"`, `"0"` | `Bool` → `Column::Bool` |
| Anything else | `Str` → `Column::Str` |

#### `StreamingCsvProcessor`

```rust
impl StreamingCsvProcessor {
    pub fn new(config: CsvConfig) -> Self;

    /// Returns (column_names, sums_per_col, row_count).
    /// Non-numeric fields contribute 0.0. Uses Kahan summation.
    pub fn sum_columns(&self, input: &[u8])
        -> Result<(Vec<String>, Vec<f64>, usize), DataError>;

    /// Returns (column_names, mins_per_col, maxs_per_col, row_count).
    pub fn minmax_columns(&self, input: &[u8])
        -> Result<(Vec<String>, Vec<f64>, Vec<f64>, usize), DataError>;
}
```

Memory usage: **O(ncols)** regardless of row count.

#### `DataFrame` extensions

```rust
impl DataFrame {
    /// Append a row from string values (type-coerced to each column's type).
    pub fn push_row(&mut self, values: &[&str]) -> Result<(), DataError>;

    /// Convert selected columns to a [nrows × ncols] Tensor.
    /// Only Float and Int columns are supported; Str/Bool returns an error.
    pub fn to_tensor(&self, col_names: &[&str])
        -> Result<cjc_runtime::Tensor, DataError>;
}
```

---

### CJC Language Builtins

#### `Csv.parse(bytes)` / `Csv.parse(bytes, max_rows)`

Parse a CSV byte string into a `DataFrame` struct.

```cjc
let csv = "name,age,score\nAlice,30,9.5\nBob,25,8.1";
let df = Csv.parse(csv);
print(df.nrows());       // 2
print(df.ncols());       // 3
print(df.column_names()); // ["name", "age", "score"]
```

With row cap:

```cjc
let df = Csv.parse(csv, 5);  // read at most 5 rows
```

#### `Csv.parse_tsv(bytes)`

Parse tab-separated values:

```cjc
let tsv = "x\ty\n1.0\t2.0\n3.0\t4.0";
let df = Csv.parse_tsv(tsv);
print(df.nrows());  // 2
```

#### `Csv.stream_sum(bytes)` → `CsvStats` struct

Compute per-column sums in a single streaming pass. Returns a struct with
one field per numeric column, plus `__row_count`.

```cjc
let csv = "x,y\n1.0,10.0\n2.0,20.0\n3.0,30.0";
let stats = Csv.stream_sum(csv);
print(stats.x);           // 6.0
print(stats.y);           // 60.0
print(stats.__row_count); // 3
```

#### `Csv.stream_minmax(bytes)` → `CsvMinMax` struct

Compute per-column min/max in a single streaming pass. Returns a struct
with `{colname}_min` and `{colname}_max` fields.

```cjc
let csv = "v\n5.0\n1.0\n9.0\n2.0";
let mm = Csv.stream_minmax(csv);
print(mm.v_min);  // 1.0
print(mm.v_max);  // 9.0
```

---

### `DataFrame` Instance Methods (CJC)

| Method | Returns | Description |
|--------|---------|-------------|
| `df.nrows()` | `Int` | Number of data rows |
| `df.ncols()` | `Int` | Number of columns (excludes meta fields) |
| `df.column_names()` | `Array[String]` | Ordered column names |
| `df.column("name")` | `Array[T]` | Values for named column |
| `df.to_tensor(["x","y"])` | `Tensor[nrows,ncols]` | Numeric columns as tensor |

Example — full pipeline:

```cjc
let csv = "feature1,feature2,label\n1.0,2.0,0.0\n3.0,4.0,1.0\n5.0,6.0,0.0";
let df  = Csv.parse(csv);
let t   = df.to_tensor(["feature1", "feature2"]);  // [3, 2] tensor
print(t.shape());  // [3, 2]
print(t.sum());    // 21.0
```

---

## Internal Representation

`DataFrame` is encoded as `Value::Struct { name: "DataFrame", fields }` to
avoid a circular crate dependency (`cjc-data` → `cjc-runtime` → `cjc-data`
would form a cycle if a `Value::DataFrame` variant were added to
`cjc-runtime`).

The struct fields:

| Field | Type | Content |
|-------|------|---------|
| `__columns` | `Value::Array[String]` | Ordered column names |
| `__nrows` | `Value::Int` | Row count |
| `{colname}` | `Value::Array[T]` | Column data |

---

## Test Coverage

`tests/test_phase8_data_logistics.rs` — **48 tests**, 0 failures.

| Section | Count | What's tested |
|---------|-------|---------------|
| CsvReader basic | 5 | 3-col parse, single col, numeric values, empty, header-only |
| Type inference | 5 | Float, Int, Bool, Str, mixed |
| Delimiter config | 2 | TSV, pipe delimiter |
| Config options | 4 | max_rows, max_rows > data, no header, trailing newline |
| Line endings | 2 | CRLF, whitespace trimming |
| DataFrame (Rust) | 3 | from_columns, length mismatch error, empty |
| push_row | 2 | basic append, wrong arity error |
| to_tensor (Rust) | 3 | 2-col layout, Int coercion, unknown column error |
| StreamingCsvProcessor | 3 | sum_columns, minmax_columns, empty input |
| Csv.parse (eval) | 3 | nrows, ncols, column access |
| Csv.parse (MIR) | 2 | nrows, ncols |
| Csv.parse_tsv parity | 1 | eval == MIR |
| DataFrame methods parity | 5 | nrows, ncols, column_names, column, to_tensor |
| stream_sum parity | 2 | eval correctness + eval==MIR |
| stream_minmax parity | 2 | eval correctness + eval==MIR |
| End-to-end | 2 | parse→tensor→sum, parse with max_rows |
| Determinism | 2 | 3-run CSV parse, 3-run streaming sum |

---

## Performance Notes

- `CsvReader` allocates once per column buffer (pre-sized to `nrows`).
- Field references during the scan phase are borrowed `&str` sub-slices of
  the original `&[u8]` — no per-field `String` allocation during type
  inference.
- `StreamingCsvProcessor` uses Kahan summation for numerical stability.
  Memory footprint: `O(ncols)` accumulators only.
- The `to_tensor()` bridge allocates exactly one `Vec<f64>` of size
  `nrows × ncols`.

---

## Scope Decisions

The following were explicitly out of scope for Phase 8 (zero-dep constraint):

| Feature | Reason excluded |
|---------|----------------|
| Parquet / XLSX | Require external parsing libraries |
| `async`/`await` I/O | Requires new language keywords + executor model |
| `parallel for` | Requires threading (`rayon` or `std::thread`) |
| io_uring / IOCP | Platform-specific OS syscall wrappers |

These remain as potential Phase 9+ items if the dependency policy is relaxed
or a native thread model is added.
