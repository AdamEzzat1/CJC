//! Phase 3 numerical computing tests — integration, differentiation,
//! constrained optimization.
//!
//! All tests run through BOTH executors (eval and MIR-exec) and verify
//! parity of outputs.

/// Run CJC source through eval, return output lines.
fn run_eval(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        let rendered = diag.render_all(src, "<test>");
        panic!("Parse errors:\n{rendered}");
    }
    let mut interp = cjc_eval::Interpreter::new(seed);
    match interp.exec(&program) {
        Ok(_) => {}
        Err(e) => panic!("Eval failed: {e:?}"),
    }
    interp.output
}

/// Run CJC source through MIR-exec, return output lines.
fn run_mir(src: &str, seed: u64) -> Vec<String> {
    let (program, diag) = cjc_parser::parse_source(src);
    if diag.has_errors() {
        let rendered = diag.render_all(src, "<test>");
        panic!("Parse errors:\n{rendered}");
    }
    let (_, executor) = cjc_mir_exec::run_program_with_executor(&program, seed)
        .unwrap_or_else(|e| panic!("MIR-exec failed: {e}"));
    executor.output
}

/// Assert parity between eval and MIR-exec outputs.
fn assert_parity(src: &str) {
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(
        eval_out, mir_out,
        "Parity failure:\n  eval: {:?}\n  mir:  {:?}",
        eval_out, mir_out
    );
}

// ─────────────────────────────────────────────────────────────────
// 3.1  Numerical Integration
// ─────────────────────────────────────────────────────────────────

#[test]
fn trapezoid_sin_integral() {
    // Integral of sin(x) from 0 to pi ≈ 2.0
    let src = r#"
let n: i64 = 1000;
let pi: f64 = 3.14159265358979323846;
let xs: Any = [];
let ys: Any = [];
let i: i64 = 0;
while i <= n {
    let x: f64 = float(i) * pi / float(n);
    xs = array_push(xs, x);
    ys = array_push(ys, sin(x));
    i = i + 1;
}
let result: f64 = trapz(xs, ys);
print(result);
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out);

    // Verify numerical accuracy
    let val: f64 = eval_out[0].parse().unwrap();
    assert!(
        (val - 2.0).abs() < 1e-5,
        "trapezoid of sin(x) from 0 to pi: expected ~2.0, got {}",
        val
    );
}

#[test]
fn simpson_x_squared_integral() {
    // Integral of x^2 from 0 to 1 ≈ 1/3
    let src = r#"
let n: i64 = 100;
let xs: Any = [];
let ys: Any = [];
let i: i64 = 0;
while i <= n {
    let x: f64 = float(i) / float(n);
    xs = array_push(xs, x);
    ys = array_push(ys, x * x);
    i = i + 1;
}
let result: f64 = simps(xs, ys);
print(result);
"#;
    let eval_out = run_eval(src, 42);
    let mir_out = run_mir(src, 42);
    assert_eq!(eval_out, mir_out);

    let val: f64 = eval_out[0].parse().unwrap();
    assert!(
        (val - 1.0 / 3.0).abs() < 1e-8,
        "simpson of x^2 from 0 to 1: expected ~0.333, got {}",
        val
    );
}

#[test]
fn cumtrapz_returns_running_integral() {
    let src = r#"
let xs: Any = [0.0, 1.0, 2.0, 3.0, 4.0];
let ys: Any = [0.0, 1.0, 2.0, 3.0, 4.0];
let result: Any = cumtrapz(xs, ys);
print(array_len(result));
print(result[0]);
print(result[3]);
"#;
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    assert_eq!(eval_out[0], "4");
    let v0: f64 = eval_out[1].parse().unwrap();
    let v3: f64 = eval_out[2].parse().unwrap();
    assert!((v0 - 0.5).abs() < 1e-12);
    assert!((v3 - 8.0).abs() < 1e-12);
}

// ─────────────────────────────────────────────────────────────────
// 3.2  Numerical Differentiation
// ─────────────────────────────────────────────────────────────────

#[test]
fn diff_central_of_x_squared() {
    // f(x) = x^2, f'(2) = 4.0
    let src = r#"
let xs: Any = [];
let ys: Any = [];
let i: i64 = 0;
while i <= 40 {
    let x: f64 = float(i) * 0.1;
    xs = array_push(xs, x);
    ys = array_push(ys, x * x);
    i = i + 1;
}
let derivs: Any = diff_central(xs, ys);
print(derivs[19]);
"#;
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    let val: f64 = eval_out[0].parse().unwrap();
    assert!(
        (val - 4.0).abs() < 1e-8,
        "diff_central at x=2: expected ~4.0, got {}",
        val
    );
}

#[test]
fn diff_forward_of_linear() {
    // f(x) = 3x + 1, f'(x) = 3
    let src = r#"
let xs: Any = [0.0, 1.0, 2.0, 3.0, 4.0];
let ys: Any = [1.0, 4.0, 7.0, 10.0, 13.0];
let derivs: Any = diff_forward(xs, ys);
print(derivs[0]);
print(derivs[2]);
"#;
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    let v0: f64 = eval_out[0].parse().unwrap();
    let v2: f64 = eval_out[1].parse().unwrap();
    assert!((v0 - 3.0).abs() < 1e-12);
    assert!((v2 - 3.0).abs() < 1e-12);
}

#[test]
fn gradient_1d_quadratic() {
    let src = r#"
let ys: Any = [0.0, 1.0, 4.0, 9.0, 16.0];
let grad: Any = gradient_1d(ys, 1.0);
print(array_len(grad));
print(grad[0]);
print(grad[2]);
print(grad[4]);
"#;
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    assert_eq!(eval_out[0], "5");
    let g0: f64 = eval_out[1].parse().unwrap();
    let g2: f64 = eval_out[2].parse().unwrap();
    let g4: f64 = eval_out[3].parse().unwrap();
    assert!((g0 - 1.0).abs() < 1e-12);
    assert!((g2 - 4.0).abs() < 1e-12);
    assert!((g4 - 7.0).abs() < 1e-12);
}

// ─────────────────────────────────────────────────────────────────
// 3.4  Constrained Optimization
// ─────────────────────────────────────────────────────────────────

#[test]
fn project_box_clamps_correctly() {
    let src = r#"
let x: Any = [-5.0, 3.0, 10.0];
let lo: Any = [0.0, 0.0, 0.0];
let hi: Any = [1.0, 1.0, 1.0];
let result: Any = project_box(x, lo, hi);
print(result[0]);
print(result[1]);
print(result[2]);
"#;
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    let v0: f64 = eval_out[0].parse().unwrap();
    let v1: f64 = eval_out[1].parse().unwrap();
    let v2: f64 = eval_out[2].parse().unwrap();
    assert!((v0 - 0.0).abs() < 1e-12);
    assert!((v1 - 1.0).abs() < 1e-12);
    assert!((v2 - 1.0).abs() < 1e-12);
}

#[test]
fn penalty_objective_adds_penalty() {
    let src = r#"
let f_val: f64 = 5.0;
let violations: Any = [2.0, -1.0, 3.0];
let penalty: f64 = 10.0;
let result: f64 = penalty_objective(f_val, violations, penalty);
print(result);
"#;
    // penalty = 10 * (max(0,2)^2 + max(0,-1)^2 + max(0,3)^2) = 10 * (4 + 0 + 9) = 130
    // total = 5 + 130 = 135
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    let val: f64 = eval_out[0].parse().unwrap();
    assert!(
        (val - 135.0).abs() < 1e-10,
        "penalty_objective: expected 135.0, got {}",
        val
    );
}

#[test]
fn projected_gd_step_respects_bounds() {
    let src = r#"
let x: Any = [0.5, 0.5];
let grad: Any = [10.0, -10.0];
let lr: f64 = 1.0;
let lo: Any = [0.0, 0.0];
let hi: Any = [1.0, 1.0];
let result: Any = projected_gd_step(x, grad, lr, lo, hi);
print(result[0]);
print(result[1]);
"#;
    // x - lr*grad = [0.5 - 10, 0.5 + 10] = [-9.5, 10.5]
    // projected to [0,1]: [0.0, 1.0]
    assert_parity(src);
    let eval_out = run_eval(src, 42);
    let v0: f64 = eval_out[0].parse().unwrap();
    let v1: f64 = eval_out[1].parse().unwrap();
    assert!((v0 - 0.0).abs() < 1e-12);
    assert!((v1 - 1.0).abs() < 1e-12);
}

// ─────────────────────────────────────────────────────────────────
// Determinism tests
// ─────────────────────────────────────────────────────────────────

#[test]
fn trapezoid_deterministic_across_runs() {
    let src = r#"
let n: i64 = 500;
let pi: f64 = 3.14159265358979323846;
let xs: Any = [];
let ys: Any = [];
let i: i64 = 0;
while i <= n {
    let x: f64 = float(i) * pi / float(n);
    xs = array_push(xs, x);
    ys = array_push(ys, sin(x));
    i = i + 1;
}
let result: f64 = trapz(xs, ys);
print(result);
"#;
    let r1 = run_eval(src, 42);
    let r2 = run_eval(src, 42);
    assert_eq!(r1, r2, "Determinism failure: eval outputs differ across runs");

    let m1 = run_mir(src, 42);
    let m2 = run_mir(src, 42);
    assert_eq!(m1, m2, "Determinism failure: MIR outputs differ across runs");
}
