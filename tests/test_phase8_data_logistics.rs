//! Phase 8: Data Logistics Engine — Full Test Suite
//!
//! Tests cover:
//!   1.  CsvReader — basic correctness (Rust level)
//!   2.  CsvReader — type inference (Int, Float, Bool, Str)
//!   3.  CsvReader — TSV / custom delimiter
//!   4.  CsvReader — max_rows cap, header-less, empty input, trailing newlines
//!   5.  CsvReader — CRLF line endings, whitespace trimming
//!   6.  DataFrame — nrows, ncols, column_names, from_columns
//!   7.  DataFrame::push_row — append rows from string values
//!   8.  DataFrame::to_tensor — numeric columns → flat f64 tensor
//!   9.  StreamingCsvProcessor — sum_columns (Kahan), minmax_columns
//!  10.  Csv.parse builtin — eval layer (AST interpreter)
//!  11.  Csv.parse builtin — MIR executor
//!  12.  Csv.parse_tsv builtin — eval + MIR parity
//!  13.  DataFrame instance methods — nrows, ncols, column_names, column, to_tensor
//!  14.  Csv.stream_sum builtin — eval + MIR parity
//!  15.  Csv.stream_minmax builtin — eval + MIR parity
//!  16.  End-to-end: parse → to_tensor → tensor ops
//!  17.  Determinism gate — three consecutive runs bit-identical

use cjc_data::{Column, CsvConfig, CsvReader, DataFrame, StreamingCsvProcessor};
use cjc_eval::Interpreter;
use cjc_parser::parse_source;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn eval_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let mut interp = Interpreter::new(42);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("eval error: {:?}", e),
    }
    interp.output.clone()
}

fn mir_output(src: &str) -> Vec<String> {
    let (program, diag) = parse_source(src);
    assert!(!diag.has_errors(), "Parse errors:\n{}", diag.render_all(src, "<test>"));
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, 42)
        .expect("MIR exec failed");
    executor.output.clone()
}

fn assert_parity(src: &str) {
    let ast_out = eval_output(src);
    let mir_out = mir_output(src);
    assert_eq!(
        ast_out, mir_out,
        "AST/MIR parity failure!\nAST: {:?}\nMIR: {:?}\nSource:\n{}",
        ast_out, mir_out, src
    );
}

// ---------------------------------------------------------------------------
// Section 1: CsvReader — basic correctness (Rust level)
// ---------------------------------------------------------------------------

#[test]
fn test_csv_reader_basic_three_columns() {
    let csv = b"name,age,score\nAlice,30,9.5\nBob,25,8.1";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert_eq!(df.nrows(), 2, "expected 2 data rows");
    assert_eq!(df.ncols(), 3, "expected 3 columns");
    assert_eq!(df.column_names(), vec!["name", "age", "score"]);
}

#[test]
fn test_csv_reader_single_column() {
    let csv = b"value\n1.0\n2.0\n3.0";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 1);
    match &df.columns[0].1 {
        Column::Float(v) => assert_eq!(v.len(), 3),
        _ => panic!("expected Float column"),
    }
}

#[test]
fn test_csv_reader_numeric_values_correct() {
    let csv = b"x,y\n1.0,2.0\n3.0,4.0";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    // x column
    match &df.columns[0].1 {
        Column::Float(v) => {
            assert!((v[0] - 1.0).abs() < 1e-12);
            assert!((v[1] - 3.0).abs() < 1e-12);
        }
        _ => panic!("expected Float column for x"),
    }
    // y column
    match &df.columns[1].1 {
        Column::Float(v) => {
            assert!((v[0] - 2.0).abs() < 1e-12);
            assert!((v[1] - 4.0).abs() < 1e-12);
        }
        _ => panic!("expected Float column for y"),
    }
}

#[test]
fn test_csv_reader_empty_input() {
    let df = CsvReader::new(CsvConfig::default()).parse(b"").unwrap();
    assert_eq!(df.nrows(), 0);
    assert_eq!(df.ncols(), 0);
}

#[test]
fn test_csv_reader_header_only() {
    let csv = b"a,b,c";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert_eq!(df.nrows(), 0);
    assert_eq!(df.ncols(), 3);
    assert_eq!(df.column_names(), vec!["a", "b", "c"]);
}

// ---------------------------------------------------------------------------
// Section 2: CsvReader — type inference
// ---------------------------------------------------------------------------

#[test]
fn test_csv_type_inference_float() {
    let csv = b"v\n1.5\n2.7\n3.14";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert!(matches!(df.columns[0].1, Column::Float(_)), "expected Float column");
}

#[test]
fn test_csv_type_inference_int() {
    let csv = b"n\n10\n20\n30";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert!(matches!(df.columns[0].1, Column::Int(_)), "expected Int column");
}

#[test]
fn test_csv_type_inference_bool() {
    let csv = b"flag\ntrue\nfalse\ntrue";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert!(matches!(df.columns[0].1, Column::Bool(_)), "expected Bool column");
    match &df.columns[0].1 {
        Column::Bool(v) => assert_eq!(v, &[true, false, true]),
        _ => unreachable!(),
    }
}

#[test]
fn test_csv_type_inference_str() {
    let csv = b"label\nhello\nworld\nfoo";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert!(matches!(df.columns[0].1, Column::Str(_)), "expected Str column");
    match &df.columns[0].1 {
        Column::Str(v) => assert_eq!(v, &["hello", "world", "foo"]),
        _ => unreachable!(),
    }
}

#[test]
fn test_csv_type_inference_mixed_columns() {
    let csv = b"name,age,weight,active\nAlice,30,65.5,true\nBob,25,72.3,false";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert!(matches!(df.columns[0].1, Column::Str(_)),   "name → Str");
    assert!(matches!(df.columns[1].1, Column::Int(_)),   "age → Int");
    assert!(matches!(df.columns[2].1, Column::Float(_)), "weight → Float");
    assert!(matches!(df.columns[3].1, Column::Bool(_)),  "active → Bool");
}

// ---------------------------------------------------------------------------
// Section 3: CsvReader — TSV / custom delimiter
// ---------------------------------------------------------------------------

#[test]
fn test_csv_reader_tsv() {
    let tsv = b"a\tb\tc\n1\t2\t3\n4\t5\t6";
    let config = CsvConfig { delimiter: b'\t', ..CsvConfig::default() };
    let df = CsvReader::new(config).parse(tsv).unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.ncols(), 3);
    assert_eq!(df.column_names(), vec!["a", "b", "c"]);
}

#[test]
fn test_csv_reader_custom_delimiter_pipe() {
    let csv = b"x|y\n10|20\n30|40";
    let config = CsvConfig { delimiter: b'|', ..CsvConfig::default() };
    let df = CsvReader::new(config).parse(csv).unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.ncols(), 2);
}

// ---------------------------------------------------------------------------
// Section 4: CsvReader — max_rows, header-less, trailing newlines
// ---------------------------------------------------------------------------

#[test]
fn test_csv_reader_max_rows() {
    let csv = b"x\n1\n2\n3\n4\n5";
    let config = CsvConfig { max_rows: Some(3), ..CsvConfig::default() };
    let df = CsvReader::new(config).parse(csv).unwrap();
    assert_eq!(df.nrows(), 3, "max_rows=3 should cap at 3 rows");
}

#[test]
fn test_csv_reader_max_rows_larger_than_data() {
    let csv = b"x\n1\n2";
    let config = CsvConfig { max_rows: Some(100), ..CsvConfig::default() };
    let df = CsvReader::new(config).parse(csv).unwrap();
    assert_eq!(df.nrows(), 2, "all 2 rows read when max_rows > actual rows");
}

#[test]
fn test_csv_reader_no_header() {
    let csv = b"1.0,2.0\n3.0,4.0";
    let config = CsvConfig { has_header: false, ..CsvConfig::default() };
    let df = CsvReader::new(config).parse(csv).unwrap();
    assert_eq!(df.nrows(), 2);
    assert_eq!(df.ncols(), 2);
    // Auto-generated column names: col_0, col_1
    assert_eq!(df.column_names(), vec!["col_0", "col_1"]);
}

#[test]
fn test_csv_reader_trailing_newline() {
    let csv = b"a,b\n1,2\n3,4\n";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert_eq!(df.nrows(), 2, "trailing newline should not create empty row");
}

// ---------------------------------------------------------------------------
// Section 5: CsvReader — CRLF, whitespace trimming
// ---------------------------------------------------------------------------

#[test]
fn test_csv_reader_crlf_endings() {
    let csv = b"a,b\r\n1,2\r\n3,4\r\n";
    let df = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    assert_eq!(df.nrows(), 2, "CRLF line endings handled");
}

#[test]
fn test_csv_reader_whitespace_trim() {
    let csv = b"x , y \n 1.0 , 2.0 \n 3.0 , 4.0 ";
    let config = CsvConfig { trim_whitespace: true, ..CsvConfig::default() };
    let df = CsvReader::new(config).parse(csv).unwrap();
    assert_eq!(df.column_names(), vec!["x", "y"]);
    assert_eq!(df.nrows(), 2);
    match &df.columns[0].1 {
        Column::Float(v) => assert!((v[0] - 1.0).abs() < 1e-12),
        _ => panic!("expected Float"),
    }
}

// ---------------------------------------------------------------------------
// Section 6: DataFrame — struct methods (Rust level)
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_from_columns() {
    let df = DataFrame::from_columns(vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("y".to_string(), Column::Int(vec![10, 20, 30])),
    ]).unwrap();
    assert_eq!(df.nrows(), 3);
    assert_eq!(df.ncols(), 2);
    assert_eq!(df.column_names(), vec!["x", "y"]);
}

#[test]
fn test_dataframe_from_columns_length_mismatch_errors() {
    let result = DataFrame::from_columns(vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0])),
        ("y".to_string(), Column::Int(vec![10])),
    ]);
    assert!(result.is_err(), "length mismatch should return DataError");
}

#[test]
fn test_dataframe_empty() {
    let df = DataFrame::new();
    assert_eq!(df.nrows(), 0);
    assert_eq!(df.ncols(), 0);
    assert!(df.column_names().is_empty());
}

// ---------------------------------------------------------------------------
// Section 7: DataFrame::push_row
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_push_row_basic() {
    let mut df = DataFrame::from_columns(vec![
        ("name".to_string(), Column::Str(vec!["Alice".to_string()])),
        ("age".to_string(),  Column::Int(vec![30])),
    ]).unwrap();
    df.push_row(&["Bob", "25"]).unwrap();
    assert_eq!(df.nrows(), 2);
    match &df.columns[1].1 {
        Column::Int(v) => assert_eq!(v[1], 25),
        _ => panic!("expected Int column"),
    }
}

#[test]
fn test_dataframe_push_row_wrong_arity_errors() {
    let mut df = DataFrame::from_columns(vec![
        ("x".to_string(), Column::Float(vec![1.0])),
    ]).unwrap();
    let result = df.push_row(&["1.0", "2.0"]);  // too many values
    assert!(result.is_err(), "wrong arity should error");
}

// ---------------------------------------------------------------------------
// Section 8: DataFrame::to_tensor
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_to_tensor_2cols() {
    let df = DataFrame::from_columns(vec![
        ("x".to_string(), Column::Float(vec![1.0, 2.0, 3.0])),
        ("y".to_string(), Column::Float(vec![4.0, 5.0, 6.0])),
    ]).unwrap();
    let t = df.to_tensor(&["x", "y"]).unwrap();
    assert_eq!(t.shape(), &[3, 2]);
    // Row-major: [1,4, 2,5, 3,6]
    let data = t.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-12);
    assert!((data[1] - 4.0).abs() < 1e-12);
    assert!((data[2] - 2.0).abs() < 1e-12);
    assert!((data[3] - 5.0).abs() < 1e-12);
}

#[test]
fn test_dataframe_to_tensor_int_col_coerced() {
    // Int columns should be coerced to f64 in the tensor
    let df = DataFrame::from_columns(vec![
        ("n".to_string(), Column::Int(vec![1, 2, 3])),
    ]).unwrap();
    let t = df.to_tensor(&["n"]).unwrap();
    assert_eq!(t.shape(), &[3, 1]);
    let data = t.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-12);
    assert!((data[1] - 2.0).abs() < 1e-12);
    assert!((data[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_dataframe_to_tensor_unknown_column_errors() {
    let df = DataFrame::from_columns(vec![
        ("x".to_string(), Column::Float(vec![1.0])),
    ]).unwrap();
    let result = df.to_tensor(&["x", "missing"]);
    assert!(result.is_err(), "unknown column should error");
}

// ---------------------------------------------------------------------------
// Section 9: StreamingCsvProcessor
// ---------------------------------------------------------------------------

#[test]
fn test_streaming_sum_columns() {
    let csv = b"x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0";
    let (names, sums, count) = StreamingCsvProcessor::new(CsvConfig::default())
        .sum_columns(csv)
        .unwrap();
    assert_eq!(count, 3);
    assert_eq!(names, vec!["x", "y"]);
    assert!((sums[0] - 9.0).abs() < 1e-10, "sum(x) = 9.0, got {}", sums[0]);
    assert!((sums[1] - 12.0).abs() < 1e-10, "sum(y) = 12.0, got {}", sums[1]);
}

#[test]
fn test_streaming_minmax_columns() {
    let csv = b"a,b\n1.0,10.0\n5.0,2.0\n3.0,7.0";
    let (names, mins, maxs, count) = StreamingCsvProcessor::new(CsvConfig::default())
        .minmax_columns(csv)
        .unwrap();
    assert_eq!(count, 3);
    assert_eq!(names, vec!["a", "b"]);
    assert!((mins[0] - 1.0).abs() < 1e-10, "min(a)=1.0, got {}", mins[0]);
    assert!((maxs[0] - 5.0).abs() < 1e-10, "max(a)=5.0, got {}", maxs[0]);
    assert!((mins[1] - 2.0).abs() < 1e-10, "min(b)=2.0, got {}", mins[1]);
    assert!((maxs[1] - 10.0).abs() < 1e-10, "max(b)=10.0, got {}", maxs[1]);
}

#[test]
fn test_streaming_sum_empty_input() {
    let (names, sums, count) = StreamingCsvProcessor::new(CsvConfig::default())
        .sum_columns(b"")
        .unwrap();
    assert_eq!(count, 0);
    assert!(names.is_empty());
    assert!(sums.is_empty());
}

// ---------------------------------------------------------------------------
// Section 10: Csv.parse builtin — eval layer
// ---------------------------------------------------------------------------

#[test]
fn test_eval_csv_parse_nrows() {
    let src = r#"
let csv_bytes = "x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0";
let df = Csv.parse(csv_bytes);
print(df.nrows());
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn test_eval_csv_parse_ncols() {
    let src = r#"
let csv_bytes = "a,b,c\n1,2,3";
let df = Csv.parse(csv_bytes);
print(df.ncols());
"#;
    let out = eval_output(src);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn test_eval_csv_parse_column_access() {
    let src = r#"
let csv_bytes = "name,score\nAlice,9.5\nBob,8.1";
let df = Csv.parse(csv_bytes);
let scores = df.column("score");
print(scores);
"#;
    let out = eval_output(src);
    assert_eq!(out.len(), 1, "expected 1 print line");
    // The column is an array of floats
    assert!(out[0].contains("9.5"), "expected 9.5 in output, got: {}", out[0]);
    assert!(out[0].contains("8.1"), "expected 8.1 in output, got: {}", out[0]);
}

// ---------------------------------------------------------------------------
// Section 11: Csv.parse builtin — MIR executor
// ---------------------------------------------------------------------------

#[test]
fn test_mir_csv_parse_nrows() {
    let src = r#"
let csv_bytes = "x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0";
let df = Csv.parse(csv_bytes);
print(df.nrows());
"#;
    let out = mir_output(src);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn test_mir_csv_parse_ncols() {
    let src = r#"
let csv_bytes = "a,b,c\n1,2,3";
let df = Csv.parse(csv_bytes);
print(df.ncols());
"#;
    let out = mir_output(src);
    assert_eq!(out, vec!["3"]);
}

// ---------------------------------------------------------------------------
// Section 12: Csv.parse_tsv — eval + MIR parity
// ---------------------------------------------------------------------------

#[test]
fn test_csv_parse_tsv_parity() {
    let src = r#"
let tsv = "x\ty\n1.0\t2.0\n3.0\t4.0";
let df = Csv.parse_tsv(tsv);
print(df.nrows());
print(df.ncols());
"#;
    assert_parity(src);
    let out = eval_output(src);
    assert_eq!(out, vec!["2", "2"]);
}

// ---------------------------------------------------------------------------
// Section 13: DataFrame instance methods — eval + MIR parity
// ---------------------------------------------------------------------------

#[test]
fn test_dataframe_nrows_parity() {
    let src = r#"
let csv = "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0";
let df = Csv.parse(csv);
print(df.nrows());
"#;
    assert_parity(src);
}

#[test]
fn test_dataframe_ncols_parity() {
    let src = r#"
let csv = "a,b,c\n1.0,2.0,3.0";
let df = Csv.parse(csv);
print(df.ncols());
"#;
    assert_parity(src);
    let out = eval_output(src);
    assert_eq!(out, vec!["3"]);
}

#[test]
fn test_dataframe_column_names_parity() {
    let src = r#"
let csv = "x,y,z\n1.0,2.0,3.0";
let df = Csv.parse(csv);
print(df.column_names());
"#;
    assert_parity(src);
}

#[test]
fn test_dataframe_column_method_parity() {
    let src = r#"
let csv = "name,age\nAlice,30\nBob,25";
let df = Csv.parse(csv);
let ages = df.column("age");
print(ages);
"#;
    assert_parity(src);
}

#[test]
fn test_dataframe_to_tensor_method_parity() {
    let src = r#"
let csv = "x,y\n1.0,4.0\n2.0,5.0\n3.0,6.0";
let df = Csv.parse(csv);
let t = df.to_tensor(["x", "y"]);
print(t.shape());
print(t.sum());
"#;
    assert_parity(src);
    let out = eval_output(src);
    assert_eq!(out[0], "[3, 2]");
    // sum = 1+4+2+5+3+6 = 21
    let sum_val: f64 = out[1].trim().parse().unwrap();
    assert!((sum_val - 21.0).abs() < 1e-9, "expected sum 21.0, got {}", sum_val);
}

// ---------------------------------------------------------------------------
// Section 14: Csv.stream_sum builtin — eval + MIR parity
// ---------------------------------------------------------------------------

#[test]
fn test_csv_stream_sum_eval() {
    let src = r#"
let csv = "x,y\n1.0,10.0\n2.0,20.0\n3.0,30.0";
let stats = Csv.stream_sum(csv);
print(stats.x);
print(stats.y);
"#;
    let out = eval_output(src);
    assert_eq!(out.len(), 2);
    let sx: f64 = out[0].trim().parse().unwrap();
    let sy: f64 = out[1].trim().parse().unwrap();
    assert!((sx - 6.0).abs() < 1e-9, "sum(x)=6.0, got {}", sx);
    assert!((sy - 60.0).abs() < 1e-9, "sum(y)=60.0, got {}", sy);
}

#[test]
fn test_csv_stream_sum_parity() {
    let src = r#"
let csv = "a,b\n1.0,2.0\n3.0,4.0";
let stats = Csv.stream_sum(csv);
print(stats.a);
print(stats.b);
"#;
    assert_parity(src);
}

// ---------------------------------------------------------------------------
// Section 15: Csv.stream_minmax builtin — eval + MIR parity
// ---------------------------------------------------------------------------

#[test]
fn test_csv_stream_minmax_eval() {
    let src = r#"
let csv = "v\n5.0\n1.0\n3.0\n9.0\n2.0";
let mm = Csv.stream_minmax(csv);
print(mm.v_min);
print(mm.v_max);
"#;
    let out = eval_output(src);
    assert_eq!(out.len(), 2);
    let mn: f64 = out[0].trim().parse().unwrap();
    let mx: f64 = out[1].trim().parse().unwrap();
    assert!((mn - 1.0).abs() < 1e-9, "min=1.0, got {}", mn);
    assert!((mx - 9.0).abs() < 1e-9, "max=9.0, got {}", mx);
}

#[test]
fn test_csv_stream_minmax_parity() {
    let src = r#"
let csv = "p,q\n1.0,4.0\n5.0,2.0\n3.0,6.0";
let mm = Csv.stream_minmax(csv);
print(mm.p_min);
print(mm.p_max);
print(mm.q_min);
print(mm.q_max);
"#;
    assert_parity(src);
    let out = eval_output(src);
    let p_min: f64 = out[0].trim().parse().unwrap();
    let p_max: f64 = out[1].trim().parse().unwrap();
    let q_min: f64 = out[2].trim().parse().unwrap();
    let q_max: f64 = out[3].trim().parse().unwrap();
    assert!((p_min - 1.0).abs() < 1e-9);
    assert!((p_max - 5.0).abs() < 1e-9);
    assert!((q_min - 2.0).abs() < 1e-9);
    assert!((q_max - 6.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Section 16: End-to-end: parse → to_tensor → tensor ops
// ---------------------------------------------------------------------------

#[test]
fn test_end_to_end_csv_to_tensor_sum_parity() {
    let src = r#"
let csv = "feature1,feature2,label\n1.0,2.0,0.0\n3.0,4.0,1.0\n5.0,6.0,0.0\n7.0,8.0,1.0";
let df = Csv.parse(csv);
let t = df.to_tensor(["feature1", "feature2"]);
print(t.shape());
print(t.sum());
"#;
    assert_parity(src);
    let out = eval_output(src);
    // shape [4,2], sum = 1+2+3+4+5+6+7+8 = 36
    assert_eq!(out[0], "[4, 2]");
    let s: f64 = out[1].trim().parse().unwrap();
    assert!((s - 36.0).abs() < 1e-9, "expected sum 36.0, got {}", s);
}

#[test]
fn test_end_to_end_csv_parse_max_rows_parity() {
    let src = r#"
let csv = "n\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10";
let df = Csv.parse(csv, 4);
print(df.nrows());
"#;
    assert_parity(src);
    let out = eval_output(src);
    assert_eq!(out, vec!["4"]);
}

// ---------------------------------------------------------------------------
// Section 17: Determinism gate — three consecutive runs bit-identical
// ---------------------------------------------------------------------------

#[test]
fn test_determinism_csv_parse_three_runs() {
    // Parsing the same CSV three times should yield identical results.
    let csv = b"x,y,z\n1.5,2.5,3.5\n4.5,5.5,6.5\n7.5,8.5,9.5";
    let df1 = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    let df2 = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();
    let df3 = CsvReader::new(CsvConfig::default()).parse(csv).unwrap();

    let t1 = df1.to_tensor(&["x", "y", "z"]).unwrap();
    let t2 = df2.to_tensor(&["x", "y", "z"]).unwrap();
    let t3 = df3.to_tensor(&["x", "y", "z"]).unwrap();

    assert_eq!(t1.to_vec(), t2.to_vec(), "run1 vs run2 mismatch");
    assert_eq!(t2.to_vec(), t3.to_vec(), "run2 vs run3 mismatch");
}

#[test]
fn test_determinism_streaming_sum_three_runs() {
    let csv = b"a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0";
    let r1 = StreamingCsvProcessor::new(CsvConfig::default()).sum_columns(csv).unwrap();
    let r2 = StreamingCsvProcessor::new(CsvConfig::default()).sum_columns(csv).unwrap();
    let r3 = StreamingCsvProcessor::new(CsvConfig::default()).sum_columns(csv).unwrap();
    // All f64 bits must be identical (Kahan is deterministic)
    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&r1.1), bits(&r2.1), "sum run1 vs run2");
    assert_eq!(bits(&r2.1), bits(&r3.1), "sum run2 vs run3");
}
