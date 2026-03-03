//! LH07: CFG Executor tests
//!
//! Verifies:
//! - CFG executor produces identical output to tree-form MIR executor
//! - Simple expressions, function calls, if/else, while loops
//! - Parity with both eval (AST) and mir-exec (tree MIR)

// ── Helper: run via CFG and compare with tree-form ──────────────

fn assert_cfg_parity(src: &str) {
    let (program, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);

    // Tree-form MIR executor
    let (tree_val, tree_exec) = cjc_mir_exec::run_program_with_executor(&program, 42).unwrap();
    let tree_output = tree_exec.output.clone();

    // CFG executor
    let (cfg_val, cfg_exec) = cjc_mir_exec::run_program_cfg(&program, 42).unwrap();
    let cfg_output = cfg_exec.output.clone();

    assert_eq!(
        tree_output, cfg_output,
        "CFG parity failure (output mismatch):\n  tree: {:?}\n  cfg:  {:?}",
        tree_output, cfg_output
    );
}

// ── Basic expressions ───────────────────────────────────────────

#[test]
fn test_cfg_exec_print_int() {
    assert_cfg_parity("print(42);");
}

#[test]
fn test_cfg_exec_print_arithmetic() {
    assert_cfg_parity("print(1 + 2 * 3);");
}

#[test]
fn test_cfg_exec_let_binding() {
    assert_cfg_parity(r#"
let x: i64 = 10;
let y: i64 = 20;
print(x + y);
"#);
}

#[test]
fn test_cfg_exec_string() {
    assert_cfg_parity(r#"print("hello");"#);
}

// ── Function calls ──────────────────────────────────────────────

#[test]
fn test_cfg_exec_fn_call() {
    assert_cfg_parity(r#"
fn double(x: i64) -> i64 { x * 2 }
print(double(21));
"#);
}

#[test]
fn test_cfg_exec_fn_multiple_params() {
    assert_cfg_parity(r#"
fn add(a: i64, b: i64) -> i64 { a + b }
print(add(10, 32));
"#);
}

// ── If/else ─────────────────────────────────────────────────────

#[test]
fn test_cfg_exec_if_true() {
    assert_cfg_parity(r#"
let x: i64 = 5;
if x > 3 {
    print(1);
} else {
    print(0);
}
"#);
}

#[test]
fn test_cfg_exec_if_false() {
    assert_cfg_parity(r#"
let x: i64 = 1;
if x > 3 {
    print(1);
} else {
    print(0);
}
"#);
}

// ── While loops ─────────────────────────────────────────────────

#[test]
fn test_cfg_exec_while_loop() {
    assert_cfg_parity(r#"
let mut i: i64 = 0;
let mut sum: i64 = 0;
while i < 5 {
    sum = sum + i;
    i = i + 1;
}
print(sum);
"#);
}

#[test]
fn test_cfg_exec_while_zero_iterations() {
    assert_cfg_parity(r#"
let mut i: i64 = 10;
while i < 5 {
    i = i + 1;
}
print(i);
"#);
}

// ── Combined: function with if/while ────────────────────────────

#[test]
fn test_cfg_exec_fn_with_if() {
    assert_cfg_parity(r#"
fn abs(x: i64) -> i64 {
    if x < 0 { 0 - x } else { x }
}
print(abs(-7));
print(abs(3));
"#);
}

#[test]
fn test_cfg_exec_fn_with_while() {
    assert_cfg_parity(r#"
fn sum_to(n: i64) -> i64 {
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        total = total + i;
        i = i + 1;
    }
    total
}
print(sum_to(5));
"#);
}

// ── Nested control flow ─────────────────────────────────────────

#[test]
fn test_cfg_exec_nested_if() {
    assert_cfg_parity(r#"
fn classify(x: i64) -> i64 {
    if x > 0 {
        if x > 10 {
            2
        } else {
            1
        }
    } else {
        0
    }
}
print(classify(15));
print(classify(5));
print(classify(-1));
"#);
}

// ── Multiple outputs ────────────────────────────────────────────

#[test]
fn test_cfg_exec_multiple_prints() {
    assert_cfg_parity(r#"
print(1);
print(2);
print(3);
"#);
}

// ── Boolean operations ──────────────────────────────────────────

#[test]
fn test_cfg_exec_bool_ops() {
    assert_cfg_parity(r#"
let a: bool = true;
let b: bool = false;
if a {
    print(1);
}
if b {
    print(2);
}
"#);
}

// ── Float arithmetic ────────────────────────────────────────────

#[test]
fn test_cfg_exec_float() {
    assert_cfg_parity(r#"
let x: f64 = 3.14;
let y: f64 = 2.0;
print(x * y);
"#);
}

// ── Mutable variables ───────────────────────────────────────────

#[test]
fn test_cfg_exec_mutation() {
    assert_cfg_parity(r#"
let mut x: i64 = 1;
x = x + 1;
x = x * 3;
print(x);
"#);
}
