//! Wiring tests — `fused_matmul_dot` runs byte-identically through both executors.
//!
//! Each test parses a CJC-Lang program that calls `fused_matmul_dot(a, w, v)`
//! and verifies that cjc-eval and cjc-mir-exec produce the exact same stdout.
//! This is the load-bearing AST↔MIR parity gate for the new primitive.

fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let mut interp = cjc_eval::Interpreter::new(seed);
    interp
        .exec(&prog)
        .unwrap_or_else(|e| panic!("eval failed: {e:?}"));
    interp.output
}

fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (prog, diags) = cjc_parser::parse_source(src);
    assert!(!diags.has_errors(), "parse errors: {:?}", diags.diagnostics);
    let (_val, executor) = cjc_mir_exec::run_program_with_executor(&prog, seed)
        .unwrap_or_else(|e| panic!("mir-exec failed: {e:?}"));
    executor.output
}

fn run_parity(src: &str, seed: u64) -> Vec<String> {
    let a = run_eval(src, seed);
    let b = run_mir(src, seed);
    assert_eq!(
        a, b,
        "parity violation between cjc-eval and cjc-mir-exec\neval: {a:?}\nmir: {b:?}"
    );
    a
}

#[test]
fn fused_matmul_dot_basic_call_runs_in_both_executors() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([1.0, 2.0], [2]);
    let w = Tensor.from_vec([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let v = Tensor.from_vec([1.0, 1.0, 1.0], [3]);
    let s = fused_matmul_dot(a, w, v);
    print(s);
}
"#;
    let out = run_parity(src, 42);
    assert_eq!(out, vec!["36"]);
}

#[test]
fn fused_matmul_dot_negative_values_byte_identical() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([-1.0, 2.0, -3.0], [3]);
    let w = Tensor.from_vec([1.0, -1.0, -2.0, 2.0, 3.0, -3.0], [3, 2]);
    let v = Tensor.from_vec([-1.0, 1.0], [2]);
    let s = fused_matmul_dot(a, w, v);
    print(s);
}
"#;
    let out = run_parity(src, 42);
    // a @ W = [-1*1 + 2*-2 + -3*3, -1*-1 + 2*2 + -3*-3] = [-14, 14]
    // sum(. * v) = -14*-1 + 14*1 = 14 + 14 = 28
    assert_eq!(out, vec!["28"]);
}

#[test]
fn fused_matmul_dot_1x1_byte_identical() {
    let src = r#"
fn main() {
    let a = Tensor.from_vec([7.0], [1]);
    let w = Tensor.from_vec([3.0], [1, 1]);
    let v = Tensor.from_vec([2.0], [1]);
    let s = fused_matmul_dot(a, w, v);
    print(s);
}
"#;
    let out = run_parity(src, 0);
    // 7 * 3 * 2 = 42
    assert_eq!(out, vec!["42"]);
}

#[test]
fn fused_matmul_dot_seed_independent() {
    // The fused primitive uses no RNG, so different seeds yield identical output.
    let src = r#"
fn main() {
    let a = Tensor.from_vec([0.5, 0.25], [2]);
    let w = Tensor.from_vec([2.0, 4.0, 8.0, 16.0], [2, 2]);
    let v = Tensor.from_vec([1.0, 0.5], [2]);
    let s = fused_matmul_dot(a, w, v);
    print(s);
}
"#;
    let out_42 = run_parity(src, 42);
    let out_7 = run_parity(src, 7);
    let out_zero = run_parity(src, 0);
    assert_eq!(out_42, out_7);
    assert_eq!(out_42, out_zero);
}
