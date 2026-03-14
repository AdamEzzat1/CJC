//! Hardening tests: Deterministic scope iteration after BTreeMap migration.
//!
//! Verifies that struct field ordering, variable enumeration, and scope
//! operations produce identical results across runs — guaranteed by
//! BTreeMap's lexicographic key ordering.

use cjc_parser::parse_source;

fn eval_program(src: &str) -> String {
    let (program, diags) = parse_source(src);
    assert!(!diags.has_errors(), "Parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(42);
    interp.exec(&program).expect("Eval failed");
    interp.output.join("\n")
}

fn mir_program(src: &str) -> String {
    let (program, diags) = parse_source(src);
    assert!(!diags.has_errors(), "Parse errors: {:?}", diags.diagnostics);
    let (_val, exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    exec.output.join("\n")
}

// ---------------------------------------------------------------------------
// Struct field ordering is deterministic (BTreeMap: alphabetical)
// ---------------------------------------------------------------------------

#[test]
fn test_struct_field_order_deterministic() {
    let src = r#"
struct Point {
    z: f64,
    a: f64,
    m: f64,
}

fn main() {
    let p = Point { z: 3.0, a: 1.0, m: 2.0 };
    print(p);
}
"#;
    let out1 = eval_program(src);
    let out2 = eval_program(src);
    assert_eq!(out1, out2, "Struct print must be deterministic across runs");
}

#[test]
fn test_struct_field_order_parity() {
    let src = r#"
struct Config {
    zebra: i64,
    apple: i64,
    mango: i64,
}

fn main() {
    let c = Config { zebra: 3, apple: 1, mango: 2 };
    print(c);
}
"#;
    let eval_out = eval_program(src);
    let mir_out = mir_program(src);
    assert_eq!(eval_out, mir_out, "Struct field order must match between eval and MIR");
}

// ---------------------------------------------------------------------------
// Multiple variables in scope produce deterministic output
// ---------------------------------------------------------------------------

#[test]
fn test_many_variables_deterministic() {
    let src = r#"
fn main() {
    let z = 26;
    let a = 1;
    let m = 13;
    let b = 2;
    let y = 25;
    print(a);
    print(b);
    print(m);
    print(y);
    print(z);
}
"#;
    let out1 = eval_program(src);
    let out2 = eval_program(src);
    assert_eq!(out1, out2);
}

#[test]
fn test_many_variables_parity() {
    let src = r#"
fn main() {
    let z = 26;
    let a = 1;
    let m = 13;
    let b = 2;
    let y = 25;
    print(a + b + m + y + z);
}
"#;
    let eval_out = eval_program(src);
    let mir_out = mir_program(src);
    assert_eq!(eval_out, mir_out);
}

// ---------------------------------------------------------------------------
// Nested struct fields remain ordered
// ---------------------------------------------------------------------------

#[test]
fn test_nested_struct_determinism() {
    let src = r#"
struct Inner {
    zz: i64,
    aa: i64,
}

struct Outer {
    bb: Inner,
    aa: i64,
}

fn main() {
    let inner = Inner { zz: 99, aa: 11 };
    let outer = Outer { bb: inner, aa: 42 };
    print(outer.aa);
    print(outer.bb.aa);
    print(outer.bb.zz);
}
"#;
    let out1 = eval_program(src);
    let out2 = eval_program(src);
    assert_eq!(out1, out2);

    let mir_out = mir_program(src);
    assert_eq!(out1, mir_out, "Nested struct access parity");
}

// ---------------------------------------------------------------------------
// Repeated struct creation produces identical results
// ---------------------------------------------------------------------------

#[test]
fn test_struct_creation_loop_determinism() {
    let src = r#"
struct Pair {
    x: i64,
    y: i64,
}

fn main() {
    let mut i = 0;
    while i < 100 {
        let p = Pair { x: i, y: i * 2 };
        if i == 99 {
            print(p.x);
            print(p.y);
        }
        i = i + 1;
    }
}
"#;
    let out1 = eval_program(src);
    let out2 = eval_program(src);
    assert_eq!(out1, out2);

    let mir_out = mir_program(src);
    assert_eq!(out1, mir_out);
}
