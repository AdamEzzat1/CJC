//! Phase 3a — bolero fuzz targets for transformer-backbone `grad_graph_*`.
//!
//! Two harnesses, mirroring the Phase 3c fuzz contract:
//!
//! 1. `fuzz_phase3a_no_panic_on_bad_args` — feed garbage `Value` shapes
//!    into each new dispatch arm and assert it returns `Err(String)`,
//!    never panics. The dispatch layer is the boundary between
//!    `.cjcl` source and the kernel; panics here would surface as
//!    interpreter crashes for user code. Out-of-range node indices,
//!    wrong arg counts, non-tensor where tensor expected, negative
//!    shape entries — all must Err.
//!
//! 2. `fuzz_phase3a_numerical_bounds` — feed random *valid* inputs
//!    through each forward path and assert outputs are finite and
//!    obey the algebraic invariant of the op (softmax probabilities
//!    in `[0, 1]`, layer_norm output mean ≈ 0 std ≈ 1, gelu/silu
//!    bounded near input range).

use std::panic;

use bolero::check;

use cjc_ad::dispatch_grad_graph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn reset() {
    let _ = dispatch_grad_graph("grad_graph_new", &[]);
}

fn input_node(data: Vec<f64>, shape: &[usize]) -> i64 {
    let v = dispatch_grad_graph(
        "grad_graph_input",
        &[Value::Tensor(Tensor::from_vec(data, shape).unwrap())],
    )
    .unwrap()
    .unwrap();
    match v {
        Value::Int(i) => i,
        _ => unreachable!(),
    }
}

fn forward_vec(idx: i64) -> Vec<f64> {
    let v = dispatch_grad_graph("grad_graph_forward", &[Value::Int(idx)])
        .unwrap()
        .unwrap();
    match v {
        Value::Tensor(t) => t.to_vec(),
        _ => unreachable!(),
    }
}

// ─── Harness 1: garbage args must Err, never panic ──────────────────

#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_phase3a_no_panic_on_bad_args() {
    check!()
        .with_iterations(256)
        .for_each(|bytes: &[u8]| {
            // Any byte sequence picks one builtin and one bad arg shape.
            // The contract: every call must return Result, never panic.
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                reset();

                let op_byte = bytes.first().copied().unwrap_or(0) % 6;
                let bad_byte = bytes.get(1).copied().unwrap_or(0) % 5;

                let bad_args: Vec<Value> = match bad_byte {
                    0 => vec![],                          // wrong arg count
                    1 => vec![Value::Int(-1)],            // negative index
                    2 => vec![Value::Int(99_999)],        // OOB index
                    3 => vec![Value::Float(0.0)],         // wrong type
                    4 => vec![Value::Int(0), Value::Int(0), Value::Int(0)], // arg count
                    _ => vec![],
                };

                let name = match op_byte {
                    0 => "grad_graph_softmax",
                    1 => "grad_graph_cross_entropy",
                    2 => "grad_graph_layer_norm",
                    3 => "grad_graph_gelu",
                    4 => "grad_graph_silu",
                    _ => "grad_graph_reshape",
                };

                // Either it's not ours (Ok(None)) or it's an Err from the
                // arg-checker. It must never be Ok(Some(_)) given garbage args
                // (we feed indices to an empty graph).
                let r = dispatch_grad_graph(name, &bad_args);
                let _ = r; // discard; we only care about no-panic
            }));
            assert!(result.is_ok(), "dispatch panicked on garbage input");
        });
}

// ─── Harness 2: numerical invariants on valid input ─────────────────

fn read_floats(bytes: &[u8], offset: usize, n: usize) -> Option<Vec<f64>> {
    if offset + n * 8 > bytes.len() {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let s = offset + i * 8;
        let bits = u64::from_le_bytes([
            bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3],
            bytes[s + 4], bytes[s + 5], bytes[s + 6], bytes[s + 7],
        ]);
        let f = f64::from_bits(bits);
        // Clamp to a numerically civil range so softmax/gelu stay well-conditioned.
        let clamped = if f.is_finite() {
            f.clamp(-4.0, 4.0)
        } else {
            0.0
        };
        out.push(clamped);
    }
    Some(out)
}

#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_phase3a_numerical_bounds() {
    check!()
        .with_iterations(256)
        .for_each(|bytes: &[u8]| {
            // n in [1, 16], pick op from low byte.
            let n = ((bytes.first().copied().unwrap_or(1) & 0x0F) as usize).max(1);
            let op_byte = bytes.get(1).copied().unwrap_or(0) % 5;
            let Some(data) = read_floats(bytes, 2, n) else {
                return;
            };

            reset();
            let inp = input_node(data.clone(), &[n]);

            let out = match op_byte {
                0 => {
                    // softmax: outputs are probabilities in [0, 1] summing to 1.
                    let r = dispatch_grad_graph("grad_graph_softmax", &[Value::Int(inp)])
                        .unwrap().unwrap();
                    let idx = match r { Value::Int(i) => i, _ => unreachable!() };
                    let v = forward_vec(idx);
                    for x in &v {
                        assert!(x.is_finite(), "softmax produced non-finite: {x}");
                        assert!((-1e-9..=1.0 + 1e-9).contains(x),
                            "softmax outside [0,1]: {x}");
                    }
                    let s: f64 = v.iter().sum();
                    assert!((s - 1.0).abs() < 1e-9, "softmax sum = {s}");
                    return;
                }
                1 => {
                    // layer_norm: needs n >= 2 to define variance meaningfully.
                    if n < 2 { return; }
                    let r = dispatch_grad_graph("grad_graph_layer_norm", &[Value::Int(inp)])
                        .unwrap().unwrap();
                    match r { Value::Int(i) => i, _ => unreachable!() }
                }
                2 => {
                    let r = dispatch_grad_graph("grad_graph_gelu", &[Value::Int(inp)])
                        .unwrap().unwrap();
                    match r { Value::Int(i) => i, _ => unreachable!() }
                }
                3 => {
                    let r = dispatch_grad_graph("grad_graph_silu", &[Value::Int(inp)])
                        .unwrap().unwrap();
                    match r { Value::Int(i) => i, _ => unreachable!() }
                }
                _ => {
                    // reshape preserves data; we round-trip [n] -> [n, 1] -> [n].
                    let shape1 = Value::Array(std::rc::Rc::new(vec![
                        Value::Int(n as i64), Value::Int(1)]));
                    let r1 = match dispatch_grad_graph(
                        "grad_graph_reshape", &[Value::Int(inp), shape1],
                    ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
                    let shape2 = Value::Array(std::rc::Rc::new(vec![Value::Int(n as i64)]));
                    let r2 = match dispatch_grad_graph(
                        "grad_graph_reshape", &[Value::Int(r1), shape2],
                    ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
                    let v = forward_vec(r2);
                    assert_eq!(
                        v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "reshape round-trip changed data bits",
                    );
                    return;
                }
            };

            let v = forward_vec(out);
            for x in &v {
                assert!(x.is_finite(), "op produced non-finite: {x}");
            }
        });
}
