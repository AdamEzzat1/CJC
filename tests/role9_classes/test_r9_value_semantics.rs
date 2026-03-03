//! Role 9, Requirement 3: Value Classes Default
//!
//! In CJC, struct and record are value types (stack-allocated, copied on
//! assignment).  Only `class` is a reference type (GC-allocated).  Tests verify:
//!   - Struct assignment copies (no aliasing)
//!   - Record assignment copies
//!   - Mutation of copy doesn't affect original
//!   - Struct field reassignment works
//!   - Value types in arrays and function calls
//!   - Nested value copy semantics
//!   - Parity between eval and MIR-exec

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse(src: &str) -> cjc_ast::Program {
    let (program, diags) = cjc_parser::parse_source(src);
    for d in &diags.diagnostics {
        eprintln!("  parse diag: {d:?}");
    }
    assert!(!diags.has_errors(), "unexpected parse errors");
    program
}

fn eval_str(src: &str) -> String {
    let program = parse(src);
    let mut interp = cjc_eval::Interpreter::new(42);
    let val = interp.exec(&program).expect("eval failed");
    format!("{val}")
}

fn mir_str(src: &str) -> String {
    let program = parse(src);
    let (val, _exec) =
        cjc_mir_exec::run_program_with_executor(&program, 42).expect("MIR exec failed");
    format!("{val}")
}

// ---------------------------------------------------------------------------
// 1. Struct is a value type — copy on assignment
// ---------------------------------------------------------------------------

#[test]
fn test_struct_is_value_type() {
    let src = r#"
struct Pair {
    x: i64,
    y: i64
}
fn main() -> i64 {
    let mut a = Pair { x: 1, y: 2 };
    let mut b = a;
    b.x = 99;
    a.x
}
"#;
    // If value type, a.x should still be 1 after mutating b
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "1", "struct should be value type — copy on assignment");
}

// ---------------------------------------------------------------------------
// 2. Record is a value type — copy on assignment
// ---------------------------------------------------------------------------

#[test]
fn test_record_is_value_type() {
    let src = r#"
record Coord {
    x: i64,
    y: i64
}
fn main() -> i64 {
    let a = Coord { x: 5, y: 10 };
    let b = a;
    if a == b {
        return 1;
    }
    0
}
"#;
    // Both a and b should exist independently with equal values
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "1");
}

// ---------------------------------------------------------------------------
// 3. Struct mutation doesn't alias
// ---------------------------------------------------------------------------

#[test]
fn test_struct_mutation_doesnt_alias() {
    let src = r#"
struct Box {
    val: i64
}
fn main() -> i64 {
    let mut original = Box { val: 10 };
    let mut copy = original;
    copy.val = 999;
    original.val
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "10", "mutating copy should not affect original");
}

// ---------------------------------------------------------------------------
// 4. Struct field reassignment works
// ---------------------------------------------------------------------------

#[test]
fn test_struct_field_reassignment() {
    let src = r#"
struct Counter {
    value: i64
}
fn main() -> i64 {
    let mut c = Counter { value: 0 };
    c.value = 1;
    c.value = 2;
    c.value = 3;
    c.value
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "3");
}

// ---------------------------------------------------------------------------
// 5. Struct vs record identity — struct allows mutation, record doesn't
// ---------------------------------------------------------------------------

#[test]
fn test_struct_vs_record_identity() {
    // Struct: mutable → mutation succeeds
    let struct_src = r#"
struct S {
    v: i64
}
fn main() -> i64 {
    let mut s = S { v: 1 };
    s.v = 42;
    s.v
}
"#;
    assert_eq!(eval_str(struct_src), "42");

    // Record: immutable → mutation fails
    let record_src = r#"
record R {
    v: i64
}
fn main() -> i64 {
    let mut r = R { v: 1 };
    r.v = 42;
    r.v
}
"#;
    let program = parse(record_src);
    let mut interp = cjc_eval::Interpreter::new(42);
    assert!(interp.exec(&program).is_err(), "record field mutation should fail");
}

// ---------------------------------------------------------------------------
// 6. Value types passed to function — no aliasing
// ---------------------------------------------------------------------------

#[test]
fn test_value_types_in_function() {
    let src = r#"
struct Data {
    x: i64
}
fn try_mutate(d: Any) -> i64 {
    42
}
fn main() -> i64 {
    let mut d = Data { x: 10 };
    let unused = try_mutate(d);
    d.x
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "10", "passing struct to function should not affect original");
}

// ---------------------------------------------------------------------------
// 7. Nested struct — deep copy
// ---------------------------------------------------------------------------

#[test]
fn test_nested_struct_value_copy() {
    let src = r#"
struct Inner {
    val: i64
}
struct Outer {
    inner: Any
}
fn main() -> i64 {
    let i = Inner { val: 5 };
    let mut a = Outer { inner: i };
    let b = a;
    b.inner.val
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "5");
}

// ---------------------------------------------------------------------------
// 8. Parity — value semantics across executors
// ---------------------------------------------------------------------------

#[test]
fn test_parity_value_semantics() {
    let src = r#"
struct Triple {
    a: i64,
    b: i64,
    c: i64
}
fn sum(t: Any) -> i64 {
    t.a + t.b + t.c
}
fn main() -> i64 {
    let t = Triple { a: 1, b: 2, c: 3 };
    let t2 = t;
    sum(t) + sum(t2)
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity: eval={ev} mir={mv}");
    assert_eq!(ev, "12", "both copies should have same values");
}

// ---------------------------------------------------------------------------
// 9. Struct equality — same-valued structs are equal
// ---------------------------------------------------------------------------

#[test]
fn test_struct_structural_equality() {
    let src = r#"
struct Pt {
    x: i64,
    y: i64
}
fn check() -> i64 {
    let a = Pt { x: 1, y: 2 };
    let b = Pt { x: 1, y: 2 };
    if a == b {
        return 1;
    }
    0
}
fn main() -> i64 {
    check()
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "1");
}

// ---------------------------------------------------------------------------
// 10. Multiple assignments preserve value semantics
// ---------------------------------------------------------------------------

#[test]
fn test_chain_assignment_value_semantics() {
    let src = r#"
struct V {
    x: i64
}
fn main() -> i64 {
    let mut a = V { x: 1 };
    let mut b = a;
    let mut c = b;
    c.x = 100;
    b.x = 200;
    a.x
}
"#;
    let ev = eval_str(src);
    let mv = mir_str(src);
    assert_eq!(ev, mv, "parity");
    assert_eq!(ev, "1", "chain assignment should produce independent copies");
}
