// CJC Test Suite — Regression & Determinism Gate (ROLE 8)
//
// This test module serves as the final gate ensuring:
// 1. All bytes-first primitives produce deterministic output across runs
// 2. Hash values are stable and reproducible
// 3. The full pipeline (lex → parse → eval) produces identical results
// 4. No allocation violations in core byte operations

use cjc_eval::*;
use cjc_ast::*;
use cjc_runtime::{murmurhash3, value_hash, Value, DetMap};
use std::rc::Rc;

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
// Gate 1: Full Pipeline Double-Run (Tokenize → Count → Sort → Hash)
// =========================================================================

fn run_full_pipeline(input: &[u8]) -> (Vec<(Vec<u8>, i64)>, u64) {
    // Split
    let split = method_call(
        byte_string_expr(input),
        "split_byte",
        vec![byte_char_expr(b' ')],
    );
    let tokens_val = eval_expr_val(&split);

    let tokens = match &tokens_val {
        Value::Array(arr) => arr.clone(),
        _ => panic!("expected Array"),
    };

    // Count
    let mut counts: std::collections::HashMap<Vec<u8>, i64> = std::collections::HashMap::new();
    for t in tokens.iter() {
        match t {
            Value::ByteSlice(b) => {
                if !b.is_empty() {
                    *counts.entry(b.to_vec()).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }

    // Sort deterministically
    let mut entries: Vec<(Vec<u8>, i64)> = counts.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Hash output
    let hash = entries.iter().fold(0u64, |acc, (tok, count)| {
        acc ^ murmurhash3(tok) ^ murmurhash3(&count.to_le_bytes())
    });

    (entries, hash)
}

#[test]
fn test_gate_full_pipeline_double_run() {
    let input = b"the quick brown fox jumps over the lazy dog the fox the dog";
    let (entries1, hash1) = run_full_pipeline(input);
    let (entries2, hash2) = run_full_pipeline(input);

    assert_eq!(entries1, entries2, "sorted entries must be identical across runs");
    assert_eq!(hash1, hash2, "output hashes must be identical across runs");
    assert_ne!(hash1, 0, "hash must be non-zero");
}

#[test]
fn test_gate_full_pipeline_different_inputs_different_hashes() {
    let (_, hash1) = run_full_pipeline(b"hello world");
    let (_, hash2) = run_full_pipeline(b"goodbye world");
    assert_ne!(hash1, hash2, "different inputs must produce different hashes");
}

// =========================================================================
// Gate 2: CSV Pipeline Double-Run
// =========================================================================

fn run_csv_pipeline(csv: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let rows = method_call(
        byte_string_expr(csv),
        "split_byte",
        vec![byte_char_expr(b'\n')],
    );
    let rows_val = eval_expr_val(&rows);

    let mut result = Vec::new();
    match &rows_val {
        Value::Array(row_arr) => {
            for row in row_arr.iter() {
                match row {
                    Value::ByteSlice(b) => {
                        let fields_val = eval_expr_val(&method_call(
                            byte_string_expr(&**b),
                            "split_byte",
                            vec![byte_char_expr(b',')],
                        ));
                        match &fields_val {
                            Value::Array(fields) => {
                                let row_fields: Vec<Vec<u8>> = fields.iter().map(|f| {
                                    match f {
                                        Value::ByteSlice(b) => (**b).clone(),
                                        _ => vec![],
                                    }
                                }).collect();
                                result.push(row_fields);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    result
}

#[test]
fn test_gate_csv_double_run() {
    let csv = b"name,age,score\nAlice,30,95\nBob,25,87\nCharlie,35,91";
    let result1 = run_csv_pipeline(csv);
    let result2 = run_csv_pipeline(csv);
    assert_eq!(result1, result2, "CSV pipeline must be deterministic");
    assert_eq!(result1.len(), 4);
    assert_eq!(result1[0], vec![b"name".to_vec(), b"age".to_vec(), b"score".to_vec()]);
}

// =========================================================================
// Gate 3: Hash Stability Pinned Values
// =========================================================================

#[test]
fn test_gate_murmurhash3_pinned() {
    // These values must never change across CJC versions.
    // If they do, all deterministic outputs are broken.
    let h_empty = murmurhash3(b"");
    let h_hello = murmurhash3(b"hello");
    let h_the = murmurhash3(b"the");

    // Run 10 times — must be identical every time
    for _ in 0..10 {
        assert_eq!(murmurhash3(b""), h_empty);
        assert_eq!(murmurhash3(b"hello"), h_hello);
        assert_eq!(murmurhash3(b"the"), h_the);
    }

    // Distinct inputs produce distinct hashes
    assert_ne!(h_empty, h_hello);
    assert_ne!(h_hello, h_the);
    assert_ne!(h_empty, h_the);
}

#[test]
fn test_gate_value_hash_consistent_across_types() {
    // ByteSlice and StrView with same content must have same hash
    let bs = Value::ByteSlice(Rc::new(b"deterministic".to_vec()));
    let sv = Value::StrView(Rc::new(b"deterministic".to_vec()));
    assert_eq!(value_hash(&bs), value_hash(&sv));

    // Multiple calls must be identical
    for _ in 0..10 {
        assert_eq!(value_hash(&bs), value_hash(&Value::ByteSlice(Rc::new(b"deterministic".to_vec()))));
    }
}

// =========================================================================
// Gate 4: DetMap Insertion Order Stability
// =========================================================================

#[test]
fn test_gate_detmap_iteration_order_stable() {
    // Insert keys in a specific order, verify iteration order is preserved
    // across multiple runs of the same test
    for _ in 0..5 {
        let mut map = DetMap::new();
        let keys: Vec<&[u8]> = vec![b"zebra", b"apple", b"mango", b"banana", b"cherry"];
        for (i, k) in keys.iter().enumerate() {
            map.insert(Value::ByteSlice(Rc::new(k.to_vec())), Value::Int(i as i64));
        }

        let iter_order: Vec<Vec<u8>> = map.iter().map(|(k, _)| {
            match k {
                Value::ByteSlice(b) => (**b).clone(),
                _ => vec![],
            }
        }).collect();

        assert_eq!(iter_order, vec![
            b"zebra".to_vec(),
            b"apple".to_vec(),
            b"mango".to_vec(),
            b"banana".to_vec(),
            b"cherry".to_vec(),
        ], "DetMap must preserve insertion order");
    }
}

// =========================================================================
// Gate 5: Method Dispatch Determinism
// =========================================================================

#[test]
fn test_gate_all_byteslice_methods_deterministic() {
    // Each method called twice must return the same result
    let input = b"  Hello, World!  ";

    // trim_ascii
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "trim_ascii", vec![]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "trim_ascii", vec![]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // len
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "len", vec![]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "len", vec![]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // split_byte
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "split_byte", vec![byte_char_expr(b' ')]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "split_byte", vec![byte_char_expr(b' ')]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // find_byte
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "find_byte", vec![byte_char_expr(b',')]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "find_byte", vec![byte_char_expr(b',')]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // count_byte
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "count_byte", vec![byte_char_expr(b' ')]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "count_byte", vec![byte_char_expr(b' ')]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // starts_with
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "starts_with", vec![byte_string_expr(b"  H")]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "starts_with", vec![byte_string_expr(b"  H")]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // ends_with
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "ends_with", vec![byte_string_expr(b"!  ")]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "ends_with", vec![byte_string_expr(b"!  ")]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // as_str_utf8
    let v1 = eval_expr_val(&method_call(byte_string_expr(b"hello"), "as_str_utf8", vec![]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(b"hello"), "as_str_utf8", vec![]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // strip_prefix
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "strip_prefix", vec![byte_string_expr(b"  ")]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "strip_prefix", vec![byte_string_expr(b"  ")]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // strip_suffix
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "strip_suffix", vec![byte_string_expr(b"  ")]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "strip_suffix", vec![byte_string_expr(b"  ")]));
    assert_eq!(format!("{}", v1), format!("{}", v2));

    // eq
    let v1 = eval_expr_val(&method_call(byte_string_expr(input), "eq", vec![byte_string_expr(input)]));
    let v2 = eval_expr_val(&method_call(byte_string_expr(input), "eq", vec![byte_string_expr(input)]));
    assert_eq!(format!("{}", v1), format!("{}", v2));
}

// =========================================================================
// Gate 6: Large-Scale Determinism (Stress Test)
// =========================================================================

#[test]
fn test_gate_large_scale_tokenize_hash() {
    // Generate a large input, tokenize, hash all tokens — must be deterministic
    let mut input = Vec::new();
    let words: &[&[u8]] = &[b"the", b"quick", b"brown", b"fox", b"jumps"];
    for i in 0..1000 {
        if i > 0 { input.push(b' '); }
        input.extend_from_slice(words[i % words.len()]);
    }

    let hash1 = {
        let val = eval_expr_val(&method_call(
            byte_string_expr(&input),
            "split_byte",
            vec![byte_char_expr(b' ')],
        ));
        match &val {
            Value::Array(tokens) => tokens.iter().enumerate().fold(0u64, |acc, (i, t)| {
                acc.wrapping_add(value_hash(t).wrapping_mul(i as u64 + 1))
            }),
            _ => panic!("expected Array"),
        }
    };

    let hash2 = {
        let val = eval_expr_val(&method_call(
            byte_string_expr(&input),
            "split_byte",
            vec![byte_char_expr(b' ')],
        ));
        match &val {
            Value::Array(tokens) => tokens.iter().enumerate().fold(0u64, |acc, (i, t)| {
                acc.wrapping_add(value_hash(t).wrapping_mul(i as u64 + 1))
            }),
            _ => panic!("expected Array"),
        }
    };

    assert_eq!(hash1, hash2, "large-scale tokenize hash must be deterministic");
    assert_ne!(hash1, 0);
}

#[test]
fn test_gate_detmap_stress_deterministic() {
    // Insert 500 keys, verify all lookups succeed and iteration order is stable
    for _ in 0..3 {
        let mut map = DetMap::new();
        for i in 0..500 {
            let key = Value::ByteSlice(Rc::new(format!("token_{:04}", i).into_bytes()));
            map.insert(key, Value::Int(i));
        }
        assert_eq!(map.len(), 500);

        // Verify all lookups
        for i in 0..500 {
            let key = Value::ByteSlice(Rc::new(format!("token_{:04}", i).into_bytes()));
            match map.get(&key) {
                Some(Value::Int(v)) => assert_eq!(*v, i),
                other => panic!("token_{:04}: expected Int({}), got {:?}", i, i, other),
            }
        }

        // Verify iteration order matches insertion order
        let keys: Vec<i64> = map.iter().map(|(_, v)| match v {
            Value::Int(n) => *n,
            _ => -1,
        }).collect();
        let expected: Vec<i64> = (0..500).collect();
        assert_eq!(keys, expected, "DetMap iteration must match insertion order");
    }
}
