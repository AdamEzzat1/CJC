//! Test-harness helpers shared across the state_space_tests suite.

use cjc_runtime::state_space::dispatch_state_space;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Eval,
    Mir,
}

/// Run a CJC-Lang snippet wrapped in `fn main() { ... }`.
pub fn run(backend: Backend, body: &str, seed: u64) -> Vec<String> {
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
            interp
                .exec(&program)
                .unwrap_or_else(|e| panic!("eval failed:\nsource:\n{src}\nerror: {e:?}"));
            interp.output
        }
        Backend::Mir => {
            let (_v, exec) = cjc_mir_exec::run_program_with_executor(&program, seed)
                .unwrap_or_else(|e| panic!("MIR-exec failed:\nsource:\n{src}\nerror: {e:?}"));
            exec.output
        }
    }
}

/// Run on both backends and assert the printed output is byte-identical.
pub fn assert_parity(label: &str, body: &str) {
    let eval_out = run(Backend::Eval, body, 42);
    let mir_out = run(Backend::Mir, body, 42);
    assert_eq!(
        eval_out, mir_out,
        "[{label}] AST↔MIR parity violation\n  eval: {eval_out:?}\n  mir : {mir_out:?}",
    );
}

// -- Direct-dispatch helpers (no parser) -----------------------------------

#[allow(dead_code)]
pub fn t(data: &[f64], shape: &[usize]) -> Value {
    Value::Tensor(Tensor::from_vec(data.to_vec(), shape).unwrap())
}

#[allow(dead_code)]
pub fn unwrap_int(v: Option<Value>) -> i64 {
    match v {
        Some(Value::Int(i)) => i,
        other => panic!("expected Int, got {:?}", other),
    }
}

#[allow(dead_code)]
pub fn unwrap_tensor(v: Option<Value>) -> Tensor {
    match v {
        Some(Value::Tensor(t)) => t,
        other => panic!("expected Tensor, got {:?}", other),
    }
}

#[allow(dead_code)]
pub fn clear() {
    let _ = dispatch_state_space("state_space_clear", &[]).unwrap();
}

#[allow(dead_code)]
pub fn ssm_init(input_dim: i64, hidden_dim: i64, output_dim: i64, seed: i64) -> i64 {
    unwrap_int(
        dispatch_state_space(
            "state_space_init",
            &[
                Value::Int(input_dim),
                Value::Int(hidden_dim),
                Value::Int(output_dim),
                Value::Int(seed),
            ],
        )
        .unwrap(),
    )
}

#[allow(dead_code)]
pub fn ssm_step(handle: i64, x: Tensor) -> Tensor {
    unwrap_tensor(
        dispatch_state_space(
            "state_space_step",
            &[Value::Int(handle), Value::Tensor(x)],
        )
        .unwrap(),
    )
}

#[allow(dead_code)]
pub fn ssm_scan(handle: i64, xs: Tensor) -> Tensor {
    unwrap_tensor(
        dispatch_state_space(
            "state_space_scan",
            &[Value::Int(handle), Value::Tensor(xs)],
        )
        .unwrap(),
    )
}

#[allow(dead_code)]
pub fn ssm_state(handle: i64) -> Tensor {
    unwrap_tensor(
        dispatch_state_space("state_space_state", &[Value::Int(handle)])
            .unwrap(),
    )
}

#[allow(dead_code)]
pub fn ssm_set_state(handle: i64, h: Tensor) {
    let _ = dispatch_state_space(
        "state_space_set_state",
        &[Value::Int(handle), Value::Tensor(h)],
    )
    .unwrap();
}

#[allow(dead_code)]
pub fn ssm_reset(handle: i64) {
    let _ = dispatch_state_space("state_space_reset", &[Value::Int(handle)]).unwrap();
}

#[allow(dead_code)]
pub fn ssm_snapshot(handle: i64) -> Tensor {
    unwrap_tensor(
        dispatch_state_space("state_space_snapshot", &[Value::Int(handle)])
            .unwrap(),
    )
}

#[allow(dead_code)]
pub fn ssm_restore(handle: i64, h: Tensor) {
    let _ = dispatch_state_space(
        "state_space_restore",
        &[Value::Int(handle), Value::Tensor(h)],
    )
    .unwrap();
}

#[allow(dead_code)]
pub fn ssm_readout(handle: i64) -> Tensor {
    unwrap_tensor(
        dispatch_state_space("state_space_readout", &[Value::Int(handle)])
            .unwrap(),
    )
}
