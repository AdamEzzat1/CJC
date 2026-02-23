// CJC Test Suite — Reference Kernels (ROLE 6)
//
// Exercises the three canonical ETL/NLP kernels through the AST evaluator:
// A) Tokenizer: split on whitespace, strip boundaries
// B) Vocab Counter: count tokens, sort deterministically
// C) CSV Scanner: split rows/fields, validate
//
// These tests prove the bytes-first primitives are sufficient for real workloads.

use cjc_eval::*;
use cjc_ast::*;
use cjc_runtime::{Value, murmurhash3, value_hash};

// -- Helpers ---------------------------------------------------------------

fn span() -> Span { Span::dummy() }
fn ident(name: &str) -> Ident { Ident::dummy(name) }
fn byte_string_expr(bytes: &[u8]) -> Expr {
    Expr { kind: ExprKind::ByteStringLit(bytes.to_vec()), span: span() }
}
fn byte_char_expr(b: u8) -> Expr {
    Expr { kind: ExprKind::ByteCharLit(b), span: span() }
}
fn call(callee: Expr, args: Vec<Expr>) -> Expr {
    let call_args: Vec<CallArg> = args.into_iter().map(|value| CallArg { name: None, value, span: span() }).collect();
    Expr { kind: ExprKind::Call { callee: Box::new(callee), args: call_args }, span: span() }
}
fn field_expr(object: Expr, name: &str) -> Expr {
    Expr { kind: ExprKind::Field { object: Box::new(object), name: ident(name) }, span: span() }
}
fn method_call(object: Expr, method: &str, args: Vec<Expr>) -> Expr {
    call(field_expr(object, method), args)
}

fn eval_expr_val(expr: &Expr) -> Value {
    let mut interp = Interpreter::new(42);
    interp.eval_expr(expr).unwrap()
}

// =========================================================================
// Kernel A: Tokenizer
// =========================================================================

#[test]
fn test_tokenizer_kernel_basic() {
    // Input: "  Hello, World!  The quick brown fox jumps.  "
    // Step 1: trim_ascii
    // Step 2: split_byte(b' ')
    // Step 3: filter non-empty tokens

    let input = byte_string_expr(b"  Hello, World!  The quick brown fox jumps.  ");

    // trim_ascii
    let trimmed = method_call(input, "trim_ascii", vec![]);
    let trimmed_val = eval_expr_val(&trimmed);
    match &trimmed_val {
        Value::ByteSlice(b) => assert_eq!(**b, b"Hello, World!  The quick brown fox jumps.".to_vec()),
        _ => panic!("expected ByteSlice"),
    }

    // split_byte on space
    let split = method_call(byte_string_expr(b"Hello, World!  The quick brown fox jumps."), "split_byte", vec![byte_char_expr(b' ')]);
    let split_val = eval_expr_val(&split);
    match &split_val {
        Value::Array(parts) => {
            // Filter non-empty
            let non_empty: Vec<&Value> = parts.iter().filter(|v| {
                match v {
                    Value::ByteSlice(b) => !b.is_empty(),
                    _ => false,
                }
            }).collect();
            assert_eq!(non_empty.len(), 7);
            // Verify first and last tokens
            match non_empty[0] {
                Value::ByteSlice(b) => assert_eq!(**b, b"Hello,".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
            match non_empty[6] {
                Value::ByteSlice(b) => assert_eq!(**b, b"jumps.".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Array"),
    }
}

#[test]
fn test_tokenizer_kernel_empty_input() {
    let input = byte_string_expr(b"   ");
    let trimmed = method_call(input, "trim_ascii", vec![]);
    let val = eval_expr_val(&trimmed);
    match &val {
        Value::ByteSlice(b) => assert!(b.is_empty()),
        _ => panic!("expected empty ByteSlice"),
    }
}

#[test]
fn test_tokenizer_kernel_single_token() {
    let input = byte_string_expr(b"hello");
    let split = method_call(input, "split_byte", vec![byte_char_expr(b' ')]);
    let val = eval_expr_val(&split);
    match &val {
        Value::Array(parts) => {
            assert_eq!(parts.len(), 1);
            match &parts[0] {
                Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Array"),
    }
}

#[test]
fn test_tokenizer_deterministic_hash() {
    // Tokenize and hash all tokens — must be deterministic
    let split = method_call(
        byte_string_expr(b"the quick brown fox"),
        "split_byte",
        vec![byte_char_expr(b' ')],
    );
    let val = eval_expr_val(&split);
    match &val {
        Value::Array(tokens) => {
            let hashes: Vec<u64> = tokens.iter().map(|t| value_hash(t)).collect();
            // Run again
            let val2 = eval_expr_val(&method_call(
                byte_string_expr(b"the quick brown fox"),
                "split_byte",
                vec![byte_char_expr(b' ')],
            ));
            match &val2 {
                Value::Array(tokens2) => {
                    let hashes2: Vec<u64> = tokens2.iter().map(|t| value_hash(t)).collect();
                    assert_eq!(hashes, hashes2);
                }
                _ => panic!("expected Array"),
            }
        }
        _ => panic!("expected Array"),
    }
}

// =========================================================================
// Kernel B: Vocab Counter
// =========================================================================

#[test]
fn test_vocab_counter_kernel() {
    // Simulate: split input, count tokens into a HashMap, sort output
    let input = b"the cat sat on the mat the cat on the mat";
    let split = method_call(
        byte_string_expr(input),
        "split_byte",
        vec![byte_char_expr(b' ')],
    );
    let tokens_val = eval_expr_val(&split);

    // Count tokens using Rust (simulating what the CJC program would do)
    let tokens = match &tokens_val {
        Value::Array(arr) => arr.clone(),
        _ => panic!("expected Array"),
    };

    let mut counts: std::collections::HashMap<Vec<u8>, i64> = std::collections::HashMap::new();
    for t in tokens.iter() {
        match t {
            Value::ByteSlice(b) => {
                *counts.entry(b.to_vec()).or_insert(0) += 1;
            }
            _ => panic!("expected ByteSlice"),
        }
    }

    // Sort: count desc, token bytes asc
    let mut entries: Vec<(Vec<u8>, i64)> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Verify
    assert_eq!(entries[0], (b"the".to_vec(), 4));
    assert_eq!(entries[1], (b"cat".to_vec(), 2));
    assert_eq!(entries[2], (b"mat".to_vec(), 2));
    assert_eq!(entries[3], (b"on".to_vec(), 2));
    assert_eq!(entries[4], (b"sat".to_vec(), 1));

    // Determinism: compute output hash
    let output_hash: u64 = entries.iter().fold(0u64, |acc, (tok, count)| {
        acc ^ murmurhash3(tok) ^ murmurhash3(&count.to_le_bytes())
    });

    // Run again — must produce same hash
    let tokens_val2 = eval_expr_val(&method_call(
        byte_string_expr(input),
        "split_byte",
        vec![byte_char_expr(b' ')],
    ));
    let tokens2 = match &tokens_val2 {
        Value::Array(arr) => arr.clone(),
        _ => panic!("expected Array"),
    };
    let mut counts2: std::collections::HashMap<Vec<u8>, i64> = std::collections::HashMap::new();
    for t in tokens2.iter() {
        match t {
            Value::ByteSlice(b) => { *counts2.entry(b.to_vec()).or_insert(0) += 1; }
            _ => {}
        }
    }
    let mut entries2: Vec<(Vec<u8>, i64)> = counts2.into_iter().collect();
    entries2.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let output_hash2: u64 = entries2.iter().fold(0u64, |acc, (tok, count)| {
        acc ^ murmurhash3(tok) ^ murmurhash3(&count.to_le_bytes())
    });

    assert_eq!(output_hash, output_hash2);
}

#[test]
fn test_vocab_counter_zipf() {
    // Skewed (Zipf-like) distribution
    let mut input = Vec::new();
    for _ in 0..100 { input.extend_from_slice(b"the "); }
    for _ in 0..50 { input.extend_from_slice(b"of "); }
    for _ in 0..25 { input.extend_from_slice(b"and "); }
    for _ in 0..12 { input.extend_from_slice(b"to "); }
    for _ in 0..6 { input.extend_from_slice(b"in "); }
    input.pop(); // remove trailing space

    let split = method_call(
        byte_string_expr(&input),
        "split_byte",
        vec![byte_char_expr(b' ')],
    );
    let val = eval_expr_val(&split);
    match &val {
        Value::Array(tokens) => {
            // Filter non-empty and count
            let non_empty: Vec<&Value> = tokens.iter().filter(|v| match v {
                Value::ByteSlice(b) => !b.is_empty(),
                _ => false,
            }).collect();
            assert_eq!(non_empty.len(), 193); // 100+50+25+12+6
        }
        _ => panic!("expected Array"),
    }
}

// =========================================================================
// Kernel C: CSV Scanner
// =========================================================================

#[test]
fn test_csv_scanner_kernel() {
    let csv = byte_string_expr(b"name,age,score\nAlice,30,95.5\nBob,25,87.3\nCharlie,35,91.2");

    // Split on newline
    let rows = method_call(csv, "split_byte", vec![byte_char_expr(b'\n')]);
    let rows_val = eval_expr_val(&rows);

    match &rows_val {
        Value::Array(row_arr) => {
            assert_eq!(row_arr.len(), 4);

            // Parse header
            match &row_arr[0] {
                Value::ByteSlice(header) => {
                    let header_split = method_call(
                        byte_string_expr(header),
                        "split_byte",
                        vec![byte_char_expr(b',')],
                    );
                    let header_fields = eval_expr_val(&header_split);
                    match &header_fields {
                        Value::Array(fields) => {
                            assert_eq!(fields.len(), 3);
                            match &fields[0] {
                                Value::ByteSlice(b) => assert_eq!(**b, b"name".to_vec()),
                                _ => panic!("expected ByteSlice"),
                            }
                            match &fields[1] {
                                Value::ByteSlice(b) => assert_eq!(**b, b"age".to_vec()),
                                _ => panic!("expected ByteSlice"),
                            }
                            match &fields[2] {
                                Value::ByteSlice(b) => assert_eq!(**b, b"score".to_vec()),
                                _ => panic!("expected ByteSlice"),
                            }
                        }
                        _ => panic!("expected Array"),
                    }
                }
                _ => panic!("expected ByteSlice"),
            }

            // Parse data row 1: Alice,30,95.5
            match &row_arr[1] {
                Value::ByteSlice(row) => {
                    let fields = eval_expr_val(&method_call(
                        byte_string_expr(row),
                        "split_byte",
                        vec![byte_char_expr(b',')],
                    ));
                    match &fields {
                        Value::Array(f) => {
                            assert_eq!(f.len(), 3);
                            match &f[0] {
                                Value::ByteSlice(b) => assert_eq!(**b, b"Alice".to_vec()),
                                _ => panic!("expected ByteSlice"),
                            }
                            // Validate age field is valid UTF-8
                            match &f[1] {
                                Value::ByteSlice(b) => {
                                    let utf8_result = eval_expr_val(&method_call(
                                        byte_string_expr(b),
                                        "as_str_utf8",
                                        vec![],
                                    ));
                                    match &utf8_result {
                                        Value::Enum { variant, .. } => assert_eq!(variant, "Ok"),
                                        _ => panic!("expected Ok"),
                                    }
                                }
                                _ => panic!("expected ByteSlice"),
                            }
                        }
                        _ => panic!("expected Array"),
                    }
                }
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Array of rows"),
    }
}

#[test]
fn test_csv_scanner_count_fields() {
    // Count commas to validate consistent field count per row
    let csv = byte_string_expr(b"a,b,c\n1,2,3\n4,5,6");
    let rows = eval_expr_val(&method_call(csv, "split_byte", vec![byte_char_expr(b'\n')]));

    match &rows {
        Value::Array(row_arr) => {
            for row in row_arr.iter() {
                match row {
                    Value::ByteSlice(b) => {
                        let count = eval_expr_val(&method_call(
                            byte_string_expr(&**b),
                            "count_byte",
                            vec![byte_char_expr(b',')],
                        ));
                        assert!(matches!(count, Value::Int(2)));
                    }
                    _ => panic!("expected ByteSlice"),
                }
            }
        }
        _ => panic!("expected Array"),
    }
}

#[test]
fn test_csv_scanner_strip_prefix() {
    // Parse a CSV where rows start with a BOM or prefix
    let row = byte_string_expr(b"\xef\xbb\xbfname,age");
    let stripped = eval_expr_val(&method_call(
        row,
        "strip_prefix",
        vec![byte_string_expr(&[0xef, 0xbb, 0xbf])], // UTF-8 BOM
    ));
    match &stripped {
        Value::Enum { variant, fields, .. } => {
            assert_eq!(variant, "Ok");
            match &fields[0] {
                Value::ByteSlice(b) => assert_eq!(**b, b"name,age".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Ok"),
    }
}

#[test]
fn test_csv_newline_count() {
    // Count lines in CSV data using count_byte
    let csv = byte_string_expr(b"h\nr1\nr2\nr3\nr4");
    let count = eval_expr_val(&method_call(csv, "count_byte", vec![byte_char_expr(b'\n')]));
    assert!(matches!(count, Value::Int(4)));
}
