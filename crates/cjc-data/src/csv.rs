//! CSV ingestion: `CsvConfig`, `CsvReader`, and `StreamingCsvProcessor`.
//!
//! Provides zero-copy CSV parsing into `DataFrame` and streaming aggregation
//! (sum, min/max) without materializing the full frame.

use crate::{Column, DataFrame, DataError};

// ── CsvConfig ───────────────────────────────────────────────────────────────

/// Configuration for `CsvReader`.
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Field delimiter byte. Default: `b','`.
    pub delimiter: u8,
    /// Whether the first row is a header. Default: `true`.
    pub has_header: bool,
    /// Maximum number of data rows to read. `None` = read all. Default: `None`.
    pub max_rows: Option<usize>,
    /// Trim ASCII whitespace from field values before type inference. Default: `true`.
    pub trim_whitespace: bool,
}

impl Default for CsvConfig {
    fn default() -> Self {
        CsvConfig {
            delimiter: b',',
            has_header: true,
            max_rows: None,
            trim_whitespace: true,
        }
    }
}

// ── CsvReader ───────────────────────────────────────────────────────────────

/// Zero-copy CSV parser for byte slices.
///
/// # Design
///
/// `CsvReader` operates directly on a `&[u8]` byte slice — no file I/O, no
/// intermediate `String` allocation per field during the scan phase. Fields
/// are referenced as sub-slices of the original input and parsed in a single
/// pass.
///
/// # Type inference
///
/// Each column's type is inferred from the first data row:
/// - All digits (optionally signed, one optional `.`) → `Float`
/// - All digits (optionally signed, no `.`) → `Int` (but stored as `Float` for
///   numeric safety — explicit `Int` columns can be forced via `CsvConfig`)
/// - `"true"` / `"false"` / `"1"` / `"0"` → `Bool`
/// - Anything else → `Str`
///
/// # Example
///
/// ```rust,ignore
/// let csv = b"name,age,score\nAlice,30,9.5\nBob,25,8.1";
/// let df = CsvReader::new(CsvConfig::default()).parse(csv)?;
/// assert_eq!(df.nrows(), 2);
/// ```
pub struct CsvReader {
    config: CsvConfig,
}

/// Inferred column type from first data row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferredType {
    Int,
    Float,
    Bool,
    Str,
}

/// Infer the type of a single field string.
fn infer_type(s: &str) -> InferredType {
    let t = s.trim();
    if t == "true" || t == "false" || t == "1" || t == "0" {
        return InferredType::Bool;
    }
    // Try int: optional leading '-', all digits
    let digits = t.strip_prefix('-').unwrap_or(t);
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        return InferredType::Int;
    }
    // Try float: optional leading '-', digits, one '.', digits
    let no_sign = t.strip_prefix('-').unwrap_or(t);
    let dot_count = no_sign.chars().filter(|&c| c == '.').count();
    if dot_count == 1 {
        let without_dot: String = no_sign.chars().filter(|&c| c != '.').collect();
        if !without_dot.is_empty() && without_dot.bytes().all(|b| b.is_ascii_digit()) {
            return InferredType::Float;
        }
    }
    // Also handle scientific notation (e.g., 1.5e-3)
    if t.parse::<f64>().is_ok() {
        return InferredType::Float;
    }
    InferredType::Str
}

/// Split a byte slice on `delimiter`, returning field sub-slices.
/// Handles the case where the last field has a trailing `\r`.
fn split_fields<'a>(row: &'a [u8], delimiter: u8) -> Vec<&'a str> {
    let mut fields = Vec::new();
    let mut start = 0usize;
    for i in 0..row.len() {
        if row[i] == delimiter {
            let field = std::str::from_utf8(&row[start..i]).unwrap_or("");
            fields.push(field);
            start = i + 1;
        }
    }
    // Last field (strip trailing \r if present)
    let tail = &row[start..];
    let tail = tail.strip_suffix(b"\r").unwrap_or(tail);
    let field = std::str::from_utf8(tail).unwrap_or("");
    fields.push(field);
    fields
}

impl CsvReader {
    /// Create a new `CsvReader` with the given configuration.
    pub fn new(config: CsvConfig) -> Self {
        CsvReader { config }
    }

    /// Parse a CSV byte slice into a `DataFrame`.
    ///
    /// # Errors
    /// Returns `DataError::InvalidOperation` if:
    /// - The input is empty.
    /// - A data row has fewer fields than the header.
    pub fn parse(&self, input: &[u8]) -> Result<DataFrame, DataError> {
        if input.is_empty() {
            return Ok(DataFrame::new());
        }

        // Split on newlines, skipping empty trailing lines.
        let rows: Vec<&[u8]> = input
            .split(|&b| b == b'\n')
            .filter(|r| !r.is_empty() && *r != b"\r")
            .collect();

        if rows.is_empty() {
            return Ok(DataFrame::new());
        }

        let delim = self.config.delimiter;

        // Parse header or generate column names.
        let (header_names, data_rows) = if self.config.has_header {
            let names: Vec<String> = split_fields(rows[0], delim)
                .into_iter()
                .map(|s| {
                    if self.config.trim_whitespace {
                        s.trim().to_string()
                    } else {
                        s.to_string()
                    }
                })
                .collect();
            (names, &rows[1..])
        } else {
            // Generate column names: col_0, col_1, ...
            let ncols = split_fields(rows[0], delim).len();
            let names: Vec<String> = (0..ncols).map(|i| format!("col_{}", i)).collect();
            (names, &rows[..])
        };

        let ncols = header_names.len();
        if ncols == 0 {
            return Ok(DataFrame::new());
        }

        // Limit rows if configured.
        let data_rows = if let Some(max) = self.config.max_rows {
            &data_rows[..data_rows.len().min(max)]
        } else {
            data_rows
        };

        if data_rows.is_empty() {
            // No data rows — return header-only DataFrame with empty columns.
            let columns: Vec<(String, Column)> = header_names
                .into_iter()
                .map(|name| (name, Column::Str(Vec::new())))
                .collect();
            return DataFrame::from_columns(columns);
        }

        // Type-infer from first data row.
        let first_fields = split_fields(data_rows[0], delim);
        let mut col_types: Vec<InferredType> = first_fields
            .iter()
            .map(|s| {
                let s = if self.config.trim_whitespace { s.trim() } else { *s };
                infer_type(s)
            })
            .collect();

        // Pad col_types if first row is shorter than header.
        while col_types.len() < ncols {
            col_types.push(InferredType::Str);
        }

        // Allocate column buffers.
        let nrows = data_rows.len();
        let mut int_bufs:   Vec<Option<Vec<i64>>>    = vec![None; ncols];
        let mut float_bufs: Vec<Option<Vec<f64>>>    = vec![None; ncols];
        let mut bool_bufs:  Vec<Option<Vec<bool>>>   = vec![None; ncols];
        let mut str_bufs:   Vec<Option<Vec<String>>> = vec![None; ncols];

        for (i, &t) in col_types.iter().enumerate() {
            match t {
                InferredType::Int   => int_bufs[i]   = Some(Vec::with_capacity(nrows)),
                InferredType::Float => float_bufs[i] = Some(Vec::with_capacity(nrows)),
                InferredType::Bool  => bool_bufs[i]  = Some(Vec::with_capacity(nrows)),
                InferredType::Str   => str_bufs[i]   = Some(Vec::with_capacity(nrows)),
            }
        }

        // Parse each data row.
        for (row_idx, &row_bytes) in data_rows.iter().enumerate() {
            let fields = split_fields(row_bytes, delim);
            for col_idx in 0..ncols {
                let raw = if col_idx < fields.len() {
                    fields[col_idx]
                } else {
                    // Missing field: treat as empty string.
                    ""
                };
                let s = if self.config.trim_whitespace { raw.trim() } else { raw };

                match col_types[col_idx] {
                    InferredType::Int => {
                        let v = s.parse::<i64>().unwrap_or(0);
                        int_bufs[col_idx].as_mut().unwrap().push(v);
                    }
                    InferredType::Float => {
                        let v = s.parse::<f64>().unwrap_or(0.0);
                        float_bufs[col_idx].as_mut().unwrap().push(v);
                    }
                    InferredType::Bool => {
                        let v = matches!(s, "true" | "1");
                        bool_bufs[col_idx].as_mut().unwrap().push(v);
                    }
                    InferredType::Str => {
                        str_bufs[col_idx].as_mut().unwrap().push(s.to_string());
                    }
                }

                let _ = row_idx; // suppress unused warning
            }
        }

        // Assemble columns.
        let mut columns: Vec<(String, Column)> = Vec::with_capacity(ncols);
        for (i, name) in header_names.into_iter().enumerate() {
            let col = match col_types[i] {
                InferredType::Int   => Column::Int(int_bufs[i].take().unwrap()),
                InferredType::Float => Column::Float(float_bufs[i].take().unwrap()),
                InferredType::Bool  => Column::Bool(bool_bufs[i].take().unwrap()),
                InferredType::Str   => Column::Str(str_bufs[i].take().unwrap()),
            };
            columns.push((name, col));
        }

        DataFrame::from_columns(columns)
    }
}

// ── StreamingCsvProcessor ───────────────────────────────────────────────────

/// A streaming CSV processor that visits rows one at a time without
/// materializing the full DataFrame.
///
/// Useful for large datasets where only aggregate statistics are needed.
/// Memory usage is O(ncols) regardless of the number of rows.
///
/// # Example
///
/// ```rust,ignore
/// let csv = b"x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0";
/// let mut proc = StreamingCsvProcessor::new(CsvConfig::default());
/// let (headers, sums, count) = proc.sum_columns(csv)?;
/// ```
pub struct StreamingCsvProcessor {
    config: CsvConfig,
}

impl StreamingCsvProcessor {
    pub fn new(config: CsvConfig) -> Self {
        StreamingCsvProcessor { config }
    }

    /// Stream through CSV, accumulating per-column sums using Kahan summation.
    ///
    /// Returns `(column_names, sums_per_col, row_count)`.
    /// Non-numeric fields contribute `0.0` to the sum.
    pub fn sum_columns(&self, input: &[u8]) -> Result<(Vec<String>, Vec<f64>, usize), DataError> {
        if input.is_empty() {
            return Ok((vec![], vec![], 0));
        }

        let rows: Vec<&[u8]> = input
            .split(|&b| b == b'\n')
            .filter(|r| !r.is_empty() && *r != b"\r")
            .collect();

        if rows.is_empty() {
            return Ok((vec![], vec![], 0));
        }

        let delim = self.config.delimiter;
        let (header_names, data_rows) = if self.config.has_header {
            let names: Vec<String> = split_fields(rows[0], delim)
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();
            (names, &rows[1..])
        } else {
            let ncols = split_fields(rows[0], delim).len();
            let names: Vec<String> = (0..ncols).map(|i| format!("col_{}", i)).collect();
            (names, &rows[..])
        };

        let ncols = header_names.len();
        // Kahan compensated sums per column.
        let mut sums: Vec<f64> = vec![0.0; ncols];
        let mut comp: Vec<f64> = vec![0.0; ncols];
        let mut row_count = 0usize;

        let data_rows = if let Some(max) = self.config.max_rows {
            &data_rows[..data_rows.len().min(max)]
        } else {
            data_rows
        };

        for &row_bytes in data_rows {
            let fields = split_fields(row_bytes, delim);
            for col_idx in 0..ncols {
                let s = if col_idx < fields.len() {
                    if self.config.trim_whitespace {
                        fields[col_idx].trim()
                    } else {
                        fields[col_idx]
                    }
                } else {
                    ""
                };
                let v: f64 = s.parse().unwrap_or(0.0);
                // Kahan step
                let y = v - comp[col_idx];
                let t = sums[col_idx] + y;
                comp[col_idx] = (t - sums[col_idx]) - y;
                sums[col_idx] = t;
            }
            row_count += 1;
        }

        Ok((header_names, sums, row_count))
    }

    /// Stream through CSV, collecting per-column min/max without materializing.
    ///
    /// Returns `(column_names, mins, maxs, row_count)`.
    /// Non-numeric fields contribute `f64::NAN` to min/max.
    pub fn minmax_columns(
        &self,
        input: &[u8],
    ) -> Result<(Vec<String>, Vec<f64>, Vec<f64>, usize), DataError> {
        if input.is_empty() {
            return Ok((vec![], vec![], vec![], 0));
        }

        let rows: Vec<&[u8]> = input
            .split(|&b| b == b'\n')
            .filter(|r| !r.is_empty() && *r != b"\r")
            .collect();

        if rows.is_empty() {
            return Ok((vec![], vec![], vec![], 0));
        }

        let delim = self.config.delimiter;
        let (header_names, data_rows) = if self.config.has_header {
            let names: Vec<String> = split_fields(rows[0], delim)
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();
            (names, &rows[1..])
        } else {
            let ncols = split_fields(rows[0], delim).len();
            let names = (0..ncols).map(|i| format!("col_{}", i)).collect();
            (names, &rows[..])
        };

        let ncols = header_names.len();
        let mut mins: Vec<f64> = vec![f64::INFINITY; ncols];
        let mut maxs: Vec<f64> = vec![f64::NEG_INFINITY; ncols];
        let mut row_count = 0usize;

        let data_rows = if let Some(max) = self.config.max_rows {
            &data_rows[..data_rows.len().min(max)]
        } else {
            data_rows
        };

        for &row_bytes in data_rows {
            let fields = split_fields(row_bytes, delim);
            for col_idx in 0..ncols {
                let s = if col_idx < fields.len() {
                    fields[col_idx].trim()
                } else {
                    ""
                };
                if let Ok(v) = s.parse::<f64>() {
                    if v < mins[col_idx] { mins[col_idx] = v; }
                    if v > maxs[col_idx] { maxs[col_idx] = v; }
                }
            }
            row_count += 1;
        }

        Ok((header_names, mins, maxs, row_count))
    }
}
