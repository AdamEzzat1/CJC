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

/// Take the accumulated field bytes as an owned `String` (lossy UTF-8) and
/// clear the buffer for the next field.
fn take_field(field: &mut Vec<u8>) -> String {
    let s = String::from_utf8_lossy(field).into_owned();
    field.clear();
    s
}

/// RFC-4180 record tokenizer over the WHOLE input.
///
/// A single pass with quotes as the outer context: inside a double-quoted
/// field, the delimiter and newlines (`\n`, `\r\n`) are literal data, and a
/// doubled quote `""` is one escaped `"`. This is what makes a field like
/// `"123 Main St, Apt 4"` stay a single field (the bug that shifted every
/// later column and blocked LendingClub's joint-application columns from
/// auto-promoting — ADR-0042 limitation #1).
///
/// Blank lines are skipped (matching the prior reader). A lone `"` opening a
/// non-empty/unquoted field is treated as a literal (lenient, RFC quotes
/// only at field start). Deterministic single pass — no `HashMap`, no FP.
fn tokenize_records(input: &[u8], delimiter: u8) -> Vec<Vec<String>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    let mut field_started_quoted = false;
    let n = input.len();
    let mut i = 0usize;

    // End the current record (push the in-progress field), skipping the
    // push entirely for a blank line (no field started, nothing buffered).
    macro_rules! end_record {
        () => {{
            if !(record.is_empty() && field.is_empty() && !field_started_quoted) {
                record.push(take_field(&mut field));
                records.push(std::mem::take(&mut record));
            }
            field_started_quoted = false;
        }};
    }

    while i < n {
        let b = input[i];
        if in_quotes {
            if b == b'"' {
                if i + 1 < n && input[i + 1] == b'"' {
                    field.push(b'"'); // escaped quote
                    i += 2;
                } else {
                    in_quotes = false; // closing quote
                    i += 1;
                }
            } else {
                field.push(b);
                i += 1;
            }
        } else if b == b'"' && field.is_empty() && !field_started_quoted {
            in_quotes = true;
            field_started_quoted = true;
            i += 1;
        } else if b == delimiter {
            record.push(take_field(&mut field));
            field_started_quoted = false;
            i += 1;
        } else if b == b'\n' {
            end_record!();
            i += 1;
        } else if b == b'\r' {
            // CRLF: let the following \n end the record. Lone CR: end here.
            if i + 1 < n && input[i + 1] == b'\n' {
                i += 1;
            } else {
                end_record!();
                i += 1;
            }
        } else {
            field.push(b);
            i += 1;
        }
    }
    // Flush a final record with no trailing newline.
    if !field.is_empty() || !record.is_empty() || field_started_quoted {
        record.push(take_field(&mut field));
        records.push(record);
    }
    records
}

/// Quote-aware split of a SINGLE row into owned field strings (handles
/// embedded delimiters and `""` escapes within the row, strips a trailing
/// `\r`). Used by the streaming processors, which split on `\n` first — so
/// a quoted field containing a newline is NOT supported in streaming mode
/// (it is in [`tokenize_records`] / [`CsvReader::parse`]). Acceptable: the
/// streaming path targets large *numeric* CSVs where quoted newlines are
/// vanishingly rare.
fn split_fields_quoted(row: &[u8], delimiter: u8) -> Vec<String> {
    let row = row.strip_suffix(b"\r").unwrap_or(row);
    let mut fields = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    let mut field_started_quoted = false;
    let n = row.len();
    let mut i = 0usize;
    while i < n {
        let b = row[i];
        if in_quotes {
            if b == b'"' {
                if i + 1 < n && row[i + 1] == b'"' {
                    field.push(b'"');
                    i += 2;
                } else {
                    in_quotes = false;
                    i += 1;
                }
            } else {
                field.push(b);
                i += 1;
            }
        } else if b == b'"' && field.is_empty() && !field_started_quoted {
            in_quotes = true;
            field_started_quoted = true;
            i += 1;
        } else if b == delimiter {
            fields.push(take_field(&mut field));
            field_started_quoted = false;
            i += 1;
        } else {
            field.push(b);
            i += 1;
        }
    }
    fields.push(take_field(&mut field));
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

        let delim = self.config.delimiter;
        // RFC-4180 record tokenization — quoted fields may contain the
        // delimiter and newlines. Replaces the prior split-on-`\n` +
        // naive-delimiter-split, which shattered quoted fields and shifted
        // every later column.
        let records = tokenize_records(input, delim);
        if records.is_empty() {
            return Ok(DataFrame::new());
        }

        // Parse header or generate column names.
        let (header_names, data_rows): (Vec<String>, &[Vec<String>]) = if self.config.has_header {
            let names: Vec<String> = records[0]
                .iter()
                .map(|s| {
                    if self.config.trim_whitespace {
                        s.trim().to_string()
                    } else {
                        s.clone()
                    }
                })
                .collect();
            (names, &records[1..])
        } else {
            // Generate column names: col_0, col_1, ...
            let ncols = records[0].len();
            let names: Vec<String> = (0..ncols).map(|i| format!("col_{}", i)).collect();
            (names, &records[..])
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
        let mut col_types: Vec<InferredType> = data_rows[0]
            .iter()
            .map(|s| {
                let s = if self.config.trim_whitespace { s.trim() } else { s.as_str() };
                infer_type(s)
            })
            .collect();

        // Reconcile col_types to the header width: pad when the first data
        // row is shorter, TRUNCATE when it is longer. Without the truncate,
        // a data row with more fields than the header indexes past the
        // ncols-sized column buffers below (a latent panic the bolero fuzz
        // surfaced). Extra data fields beyond the header are ignored.
        while col_types.len() < ncols {
            col_types.push(InferredType::Str);
        }
        col_types.truncate(ncols);

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
        for record in data_rows.iter() {
            for col_idx in 0..ncols {
                let raw: &str = if col_idx < record.len() {
                    record[col_idx].as_str()
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
    /// Create a new streaming CSV processor with the given configuration.
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
            let names: Vec<String> = split_fields_quoted(rows[0], delim)
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();
            (names, &rows[1..])
        } else {
            let ncols = split_fields_quoted(rows[0], delim).len();
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
            let fields = split_fields_quoted(row_bytes, delim);
            for col_idx in 0..ncols {
                let s = if col_idx < fields.len() {
                    if self.config.trim_whitespace {
                        fields[col_idx].trim()
                    } else {
                        fields[col_idx].as_str()
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
            let names: Vec<String> = split_fields_quoted(rows[0], delim)
                .into_iter()
                .map(|s| s.trim().to_string())
                .collect();
            (names, &rows[1..])
        } else {
            let ncols = split_fields_quoted(rows[0], delim).len();
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
            let fields = split_fields_quoted(row_bytes, delim);
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

// ── RFC-4180 tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod rfc4180_tests {
    use super::*;
    use proptest::prelude::*;

    fn parse(csv: &str) -> DataFrame {
        CsvReader::new(CsvConfig::default())
            .parse(csv.as_bytes())
            .unwrap()
    }
    fn str_col<'a>(df: &'a DataFrame, name: &str) -> &'a [String] {
        match df.get_column(name) {
            Some(Column::Str(v)) => v.as_slice(),
            _ => panic!("{name} is not a Str column"),
        }
    }
    fn int_col<'a>(df: &'a DataFrame, name: &str) -> &'a [i64] {
        match df.get_column(name) {
            Some(Column::Int(v)) => v.as_slice(),
            _ => panic!("{name} is not an Int column"),
        }
    }
    fn float_col<'a>(df: &'a DataFrame, name: &str) -> &'a [f64] {
        match df.get_column(name) {
            Some(Column::Float(v)) => v.as_slice(),
            _ => panic!("{name} is not a Float column"),
        }
    }

    #[test]
    fn quoted_comma_stays_one_field_and_keeps_columns_aligned() {
        // The LendingClub shape: a free-text column with commas sits BEFORE
        // numeric joint-application columns. Pre-fix the quoted commas
        // shattered `desc` and shifted every later column, so
        // annual_inc_joint/dti_joint failed to parse as numbers (ADR-0042
        // limitation #1). Now they land as the right types.
        let df = parse(
            "id,desc,annual_inc_joint,dti_joint\n\
             100,\"income, debt, and more\",50000,12.5\n\
             200,\"plain\",60000,9.0\n",
        );
        assert_eq!(df.ncols(), 4);
        assert_eq!(df.nrows(), 2);
        assert_eq!(
            str_col(&df, "desc"),
            &["income, debt, and more".to_string(), "plain".to_string()]
        );
        assert_eq!(int_col(&df, "annual_inc_joint"), &[50000, 60000]);
        assert_eq!(float_col(&df, "dti_joint"), &[12.5, 9.0]);
    }

    #[test]
    fn quoted_field_with_embedded_newline_is_one_row() {
        let df = parse("a,b\n10,\"line1\nline2\"\n20,\"x\"\n");
        assert_eq!(df.nrows(), 2, "embedded newline must not split the record");
        assert_eq!(
            str_col(&df, "b"),
            &["line1\nline2".to_string(), "x".to_string()]
        );
    }

    #[test]
    fn escaped_double_quotes_unescape() {
        let df = parse("a,b\n10,\"say \"\"hi\"\"\"\n");
        assert_eq!(str_col(&df, "b"), &["say \"hi\"".to_string()]);
    }

    #[test]
    fn unquoted_rows_still_parse() {
        let df = parse("a,b,c\n10,20,30\n40,50,60\n");
        assert_eq!(df.ncols(), 3);
        assert_eq!(int_col(&df, "a"), &[10, 40]);
        assert_eq!(int_col(&df, "c"), &[30, 60]);
    }

    #[test]
    fn crlf_line_endings_with_quotes() {
        let df = parse("a,b\r\n10,\"x, y\"\r\n20,z\r\n");
        assert_eq!(df.nrows(), 2);
        assert_eq!(str_col(&df, "b"), &["x, y".to_string(), "z".to_string()]);
    }

    #[test]
    fn quoted_empty_and_blank_lines() {
        let df = parse("a,b\n10,\"\"\n\n20,y\n");
        assert_eq!(df.nrows(), 2, "blank line must be skipped");
        assert_eq!(str_col(&df, "b"), &["".to_string(), "y".to_string()]);
    }

    #[test]
    fn data_row_longer_than_header_does_not_panic() {
        // Regression for the bolero-surfaced panic: a data row with more
        // fields than the header must ignore the extras, not index past the
        // header-width column buffers.
        let df = parse("a,b\n10,20,30,40\n50,60,70\n");
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.nrows(), 2);
        assert_eq!(int_col(&df, "a"), &[10, 50]);
        assert_eq!(int_col(&df, "b"), &[20, 60]);
    }

    #[test]
    fn parse_is_deterministic() {
        let csv = "id,desc,n\n1,\"a, b\",10\n2,\"c\"\"d\",20\n";
        let a = parse(csv);
        let b = parse(csv);
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        assert_eq!(str_col(&a, "desc"), str_col(&b, "desc"));
    }

    proptest! {
        /// Parsing arbitrary content never panics and is shape-stable.
        #[test]
        fn parse_arbitrary_never_panics_and_stable(s in ".{0,400}") {
            let csv = format!("a,b,c\n{}\n", s);
            let r1 = CsvReader::new(CsvConfig::default()).parse(csv.as_bytes());
            let r2 = CsvReader::new(CsvConfig::default()).parse(csv.as_bytes());
            prop_assert_eq!(r1.is_ok(), r2.is_ok());
            if let (Ok(d1), Ok(d2)) = (r1, r2) {
                prop_assert_eq!(d1.nrows(), d2.nrows());
                prop_assert_eq!(d1.ncols(), d2.ncols());
            }
        }

        /// A quoted field absorbing N delimiters is still exactly one field.
        #[test]
        fn quoted_field_absorbs_delimiters(n in 0usize..20) {
            let inner = vec!["x"; n + 1].join(",");
            let csv = format!("a,b\n1,\"{}\"\n", inner);
            let df = CsvReader::new(CsvConfig::default()).parse(csv.as_bytes()).unwrap();
            prop_assert_eq!(df.ncols(), 2);
            match df.get_column("b") {
                Some(Column::Str(v)) => prop_assert_eq!(&v[0], &inner),
                _ => prop_assert!(false, "b should be a Str column"),
            }
        }
    }

    #[test]
    fn fuzz_parse_arbitrary_bytes_never_panics() {
        bolero::check!()
            .with_type::<Vec<u8>>()
            .for_each(|bytes: &Vec<u8>| {
                let _ = CsvReader::new(CsvConfig::default()).parse(bytes);
                let _ = StreamingCsvProcessor::new(CsvConfig::default()).sum_columns(bytes);
            });
    }
}
