//! GC-06 Phase 3a — wiring tests: the fused builtins are byte-identical across
//! cjc-eval (AST) and cjc-mir-exec (MIR), and reject shape mismatches the same
//! way in both.

#![allow(clippy::needless_raw_string_hashes)]

#[derive(Clone, Copy, Debug)]
enum Backend {
    Eval,
    Mir,
}

fn run_result(backend: Backend, body: &str, seed: u64) -> Result<Vec<String>, String> {
    let src = format!("fn main() {{\n{body}\n}}\n");
    let (program, diags) = cjc_parser::parse_source(&src);
    assert!(
        !diags.has_errors(),
        "parse errors:\n{:#?}\nsource:\n{src}",
        diags.diagnostics,
    );
    match backend {
        Backend::Eval => {
            let mut interp = cjc_eval::Interpreter::new(seed);
            match interp.exec(&program) {
                Ok(_) => Ok(interp.output),
                Err(e) => Err(format!("{e:?}")),
            }
        }
        Backend::Mir => match cjc_mir_exec::run_program_with_executor(&program, seed) {
            Ok((_v, exec)) => Ok(exec.output),
            Err(e) => Err(format!("{e:?}")),
        },
    }
}

fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
    run_result(backend, body, seed)
        .unwrap_or_else(|e| panic!("{backend:?} failed for:\n{body}\nerror: {e}"))
}

fn assert_parity_contains(label: &str, body: &str, needle: &str) {
    let e = run(Backend::Eval, body, 42);
    let m = run(Backend::Mir, body, 42);
    assert_eq!(e, m, "[{label}] AST↔MIR parity violation\n eval: {e:?}\n mir: {m:?}");
    let joined = e.join("\n");
    assert!(joined.contains(needle), "[{label}] expected `{needle}` in:\n{joined}");
}

fn assert_both_error(label: &str, body: &str) {
    assert!(run_result(Backend::Eval, body, 42).is_err(), "[{label}] eval should error");
    assert!(run_result(Backend::Mir, body, 42).is_err(), "[{label}] mir should error");
}

#[test]
fn wiring_fused_axpy() {
    assert_parity_contains(
        "fused_axpy",
        r#"
        let x: Tensor = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        let y: Tensor = Tensor.from_vec([10.0, 20.0, 30.0], [3]);
        print(fused_axpy(2.0, x, y));
        "#,
        "12.0", // 2*1 + 10
    );
}

#[test]
fn wiring_fused_mul_sub() {
    assert_parity_contains(
        "fused_mul_sub",
        r#"
        let a: Tensor = Tensor.from_vec([2.0, 3.0], [2]);
        let b: Tensor = Tensor.from_vec([4.0, 5.0], [2]);
        let c: Tensor = Tensor.from_vec([1.0, 1.0], [2]);
        print(fused_mul_sub(a, b, c));
        "#,
        "7.0", // 2*4 - 1
    );
}

#[test]
fn wiring_fused_sub_sq() {
    assert_parity_contains(
        "fused_sub_sq",
        r#"
        let a: Tensor = Tensor.from_vec([5.0, 3.0, 8.0], [3]);
        let b: Tensor = Tensor.from_vec([1.0, 1.0, 2.0], [3]);
        print(fused_sub_sq(a, b));
        "#,
        "36.0", // (8-2)^2
    );
}

#[test]
fn wiring_fused_axpy_accepts_int_alpha() {
    assert_parity_contains(
        "fused_axpy int alpha",
        r#"
        let x: Tensor = Tensor.from_vec([1.0, 2.0], [2]);
        let y: Tensor = Tensor.from_vec([5.0, 5.0], [2]);
        print(fused_axpy(3, x, y));
        "#,
        "8.0", // 3*1 + 5
    );
}

#[test]
fn wiring_fused_shape_mismatch_errors_in_both() {
    assert_both_error(
        "shape mismatch",
        r#"
        let a: Tensor = Tensor.from_vec([1.0, 2.0, 3.0], [3]);
        let b: Tensor = Tensor.from_vec([1.0, 2.0], [2]);
        print(fused_sub_sq(a, b));
        "#,
    );
}
