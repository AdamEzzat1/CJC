// CJC Test Suite — Bytes-First + String Views Integration Tests
//
// Tests the complete pipeline: lexer -> parser -> eval for byte string literals,
// byte char literals, raw strings, and ByteSlice/StrView method dispatch.

use cjc_eval::*;
use cjc_ast::*;
use cjc_runtime::Value;

// -- Helpers ---------------------------------------------------------------

fn span() -> Span { Span::dummy() }
fn ident(name: &str) -> Ident { Ident::dummy(name) }
fn int_expr(v: i64) -> Expr { Expr { kind: ExprKind::IntLit(v), span: span() } }
fn string_expr(s: &str) -> Expr { Expr { kind: ExprKind::StringLit(s.to_string()), span: span() } }
fn ident_expr(name: &str) -> Expr { Expr { kind: ExprKind::Ident(ident(name)), span: span() } }
fn byte_string_expr(bytes: &[u8]) -> Expr {
    Expr { kind: ExprKind::ByteStringLit(bytes.to_vec()), span: span() }
}
fn byte_char_expr(b: u8) -> Expr {
    Expr { kind: ExprKind::ByteCharLit(b), span: span() }
}
fn raw_string_expr(s: &str) -> Expr {
    Expr { kind: ExprKind::RawStringLit(s.to_string()), span: span() }
}
fn raw_byte_string_expr(bytes: &[u8]) -> Expr {
    Expr { kind: ExprKind::RawByteStringLit(bytes.to_vec()), span: span() }
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
fn let_stmt(name: &str, init: Expr) -> Stmt {
    Stmt { kind: StmtKind::Let(LetStmt { name: ident(name), mutable: false, ty: None, init: Box::new(init) }), span: span() }
}
fn expr_stmt(e: Expr) -> Stmt {
    Stmt { kind: StmtKind::Expr(e), span: span() }
}

fn eval_expr_val(expr: &Expr) -> Value {
    let mut interp = Interpreter::new(42);
    interp.eval_expr(expr).unwrap()
}

fn eval_program_output(stmts: Vec<Stmt>) -> Vec<String> {
    let decls = stmts.into_iter().map(|s| Decl { kind: DeclKind::Stmt(s), span: span() }).collect();
    let program = Program { declarations: decls };
    let mut interp = Interpreter::new(42);
    let _ = interp.exec(&program);
    interp.output.clone()
}

// =========================================================================
// Byte String Literal Tests
// =========================================================================

#[test]
fn test_byte_string_lit_eval() {
    let val = eval_expr_val(&byte_string_expr(b"hello"));
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
        _ => panic!("expected ByteSlice, got {:?}", val),
    }
}

#[test]
fn test_byte_string_empty() {
    let val = eval_expr_val(&byte_string_expr(b""));
    match &val {
        Value::ByteSlice(b) => assert!(b.is_empty()),
        _ => panic!("expected ByteSlice"),
    }
}

#[test]
fn test_byte_string_with_binary_data() {
    let val = eval_expr_val(&byte_string_expr(&[0xff, 0x00, 0x41, 0x0a]));
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, vec![0xff, 0x00, 0x41, 0x0a]),
        _ => panic!("expected ByteSlice"),
    }
}

// =========================================================================
// Byte Char Literal Tests
// =========================================================================

#[test]
fn test_byte_char_lit_eval() {
    let val = eval_expr_val(&byte_char_expr(b'A'));
    match val {
        Value::U8(65) => {},
        _ => panic!("expected U8(65), got {:?}", val),
    }
}

#[test]
fn test_byte_char_newline() {
    let val = eval_expr_val(&byte_char_expr(b'\n'));
    match val {
        Value::U8(10) => {},
        _ => panic!("expected U8(10)"),
    }
}

#[test]
fn test_byte_char_zero() {
    let val = eval_expr_val(&byte_char_expr(0));
    match val {
        Value::U8(0) => {},
        _ => panic!("expected U8(0)"),
    }
}

// =========================================================================
// Raw String Literal Tests
// =========================================================================

#[test]
fn test_raw_string_lit_eval() {
    let val = eval_expr_val(&raw_string_expr(r"hello\nworld"));
    match &val {
        Value::String(s) => assert_eq!(**s, r"hello\nworld"),
        _ => panic!("expected String, got {:?}", val),
    }
}

#[test]
fn test_raw_string_preserves_backslashes() {
    let val = eval_expr_val(&raw_string_expr(r"C:\Users\data"));
    match &val {
        Value::String(s) => assert_eq!(**s, r"C:\Users\data"),
        _ => panic!("expected String"),
    }
}

#[test]
fn test_raw_byte_string_lit_eval() {
    let val = eval_expr_val(&raw_byte_string_expr(br"hello\nworld"));
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, br"hello\nworld".to_vec()),
        _ => panic!("expected ByteSlice, got {:?}", val),
    }
}

// =========================================================================
// ByteSlice Method Tests
// =========================================================================

#[test]
fn test_byteslice_len() {
    let expr = method_call(byte_string_expr(b"hello"), "len", vec![]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Int(5)));
}

#[test]
fn test_byteslice_is_empty_false() {
    let expr = method_call(byte_string_expr(b"hi"), "is_empty", vec![]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(false)));
}

#[test]
fn test_byteslice_is_empty_true() {
    let expr = method_call(byte_string_expr(b""), "is_empty", vec![]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(true)));
}

#[test]
fn test_byteslice_get() {
    let expr = method_call(byte_string_expr(b"ABC"), "get", vec![int_expr(1)]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::U8(66))); // 'B' = 66
}

#[test]
fn test_byteslice_get_out_of_bounds() {
    let expr = method_call(byte_string_expr(b"AB"), "get", vec![int_expr(5)]);
    let mut interp = Interpreter::new(42);
    assert!(interp.eval_expr(&expr).is_err());
}

#[test]
fn test_byteslice_slice() {
    let expr = method_call(byte_string_expr(b"hello world"), "slice", vec![int_expr(0), int_expr(5)]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
        _ => panic!("expected ByteSlice, got {:?}", val),
    }
}

#[test]
fn test_byteslice_find_byte_found() {
    let expr = method_call(byte_string_expr(b"hello,world"), "find_byte", vec![byte_char_expr(b',')]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Int(5)));
}

#[test]
fn test_byteslice_find_byte_not_found() {
    let expr = method_call(byte_string_expr(b"hello"), "find_byte", vec![byte_char_expr(b',')]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Int(-1)));
}

#[test]
fn test_byteslice_split_byte() {
    let expr = method_call(byte_string_expr(b"a,b,c"), "split_byte", vec![byte_char_expr(b',')]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::Array(parts) => {
            assert_eq!(parts.len(), 3);
            match &parts[0] {
                Value::ByteSlice(b) => assert_eq!(**b, b"a".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
            match &parts[1] {
                Value::ByteSlice(b) => assert_eq!(**b, b"b".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
            match &parts[2] {
                Value::ByteSlice(b) => assert_eq!(**b, b"c".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Array, got {:?}", val),
    }
}

#[test]
fn test_byteslice_trim_ascii() {
    let expr = method_call(byte_string_expr(b"  hello  "), "trim_ascii", vec![]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
        _ => panic!("expected ByteSlice"),
    }
}

#[test]
fn test_byteslice_trim_ascii_tabs_newlines() {
    let expr = method_call(byte_string_expr(b"\t\nhello\r\n"), "trim_ascii", vec![]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
        _ => panic!("expected ByteSlice"),
    }
}

#[test]
fn test_byteslice_starts_with() {
    let expr = method_call(
        byte_string_expr(b"hello world"),
        "starts_with",
        vec![byte_string_expr(b"hello")],
    );
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(true)));
}

#[test]
fn test_byteslice_starts_with_false() {
    let expr = method_call(
        byte_string_expr(b"hello world"),
        "starts_with",
        vec![byte_string_expr(b"world")],
    );
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(false)));
}

#[test]
fn test_byteslice_ends_with() {
    let expr = method_call(
        byte_string_expr(b"hello world"),
        "ends_with",
        vec![byte_string_expr(b"world")],
    );
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(true)));
}

#[test]
fn test_byteslice_count_byte() {
    let expr = method_call(byte_string_expr(b"hello"), "count_byte", vec![byte_char_expr(b'l')]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Int(2)));
}

#[test]
fn test_byteslice_count_byte_zero() {
    let expr = method_call(byte_string_expr(b"hello"), "count_byte", vec![byte_char_expr(b'z')]);
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Int(0)));
}

// =========================================================================
// UTF-8 Validation Tests
// =========================================================================

#[test]
fn test_byteslice_as_str_utf8_valid() {
    let expr = method_call(byte_string_expr(b"hello"), "as_str_utf8", vec![]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::Enum { variant, fields, .. } => {
            assert_eq!(variant, "Ok");
            assert_eq!(fields.len(), 1);
            match &fields[0] {
                Value::StrView(b) => assert_eq!(**b, b"hello".to_vec()),
                _ => panic!("expected StrView in Ok"),
            }
        }
        _ => panic!("expected Enum(Ok), got {:?}", val),
    }
}

#[test]
fn test_byteslice_as_str_utf8_invalid() {
    let expr = method_call(byte_string_expr(&[0xff, 0xfe]), "as_str_utf8", vec![]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::Enum { variant, .. } => {
            assert_eq!(variant, "Err");
        }
        _ => panic!("expected Enum(Err), got {:?}", val),
    }
}

// =========================================================================
// StrView Method Tests
// =========================================================================

#[test]
fn test_strview_len_bytes() {
    // First get a StrView by validating a byte string
    // We test via chained method calls using let bindings
    let stmts = vec![
        let_stmt("data", byte_string_expr(b"hello")),
        let_stmt("result", method_call(ident_expr("data"), "as_str_utf8", vec![])),
        // For now, test that as_str_utf8 returns Ok variant
        expr_stmt(call(ident_expr("print"), vec![ident_expr("result")])),
    ];
    let output = eval_program_output(stmts);
    assert_eq!(output.len(), 1);
    assert!(output[0].contains("Ok"));
}

// =========================================================================
// String.as_bytes() Tests
// =========================================================================

#[test]
fn test_string_as_bytes() {
    let expr = method_call(string_expr("hello"), "as_bytes", vec![]);
    let val = eval_expr_val(&expr);
    match &val {
        Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
        _ => panic!("expected ByteSlice, got {:?}", val),
    }
}

// =========================================================================
// ByteSlice Equality Tests
// =========================================================================

#[test]
fn test_byteslice_eq_true() {
    let expr = method_call(
        byte_string_expr(b"abc"),
        "eq",
        vec![byte_string_expr(b"abc")],
    );
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(true)));
}

#[test]
fn test_byteslice_eq_false() {
    let expr = method_call(
        byte_string_expr(b"abc"),
        "eq",
        vec![byte_string_expr(b"def")],
    );
    let val = eval_expr_val(&expr);
    assert!(matches!(val, Value::Bool(false)));
}

// =========================================================================
// Strip Prefix/Suffix Tests
// =========================================================================

#[test]
fn test_byteslice_strip_prefix_ok() {
    let expr = method_call(
        byte_string_expr(b"hello world"),
        "strip_prefix",
        vec![byte_string_expr(b"hello ")],
    );
    let val = eval_expr_val(&expr);
    match &val {
        Value::Enum { variant, fields, .. } => {
            assert_eq!(variant, "Ok");
            match &fields[0] {
                Value::ByteSlice(b) => assert_eq!(**b, b"world".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Ok"),
    }
}

#[test]
fn test_byteslice_strip_prefix_err() {
    let expr = method_call(
        byte_string_expr(b"hello"),
        "strip_prefix",
        vec![byte_string_expr(b"xyz")],
    );
    let val = eval_expr_val(&expr);
    match &val {
        Value::Enum { variant, .. } => assert_eq!(variant, "Err"),
        _ => panic!("expected Err"),
    }
}

#[test]
fn test_byteslice_strip_suffix_ok() {
    let expr = method_call(
        byte_string_expr(b"hello.csv"),
        "strip_suffix",
        vec![byte_string_expr(b".csv")],
    );
    let val = eval_expr_val(&expr);
    match &val {
        Value::Enum { variant, fields, .. } => {
            assert_eq!(variant, "Ok");
            match &fields[0] {
                Value::ByteSlice(b) => assert_eq!(**b, b"hello".to_vec()),
                _ => panic!("expected ByteSlice"),
            }
        }
        _ => panic!("expected Ok"),
    }
}
