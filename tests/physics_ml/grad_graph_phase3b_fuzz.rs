//! Phase 3b — bolero fuzz targets for the array-arg & state-recovery
//! `grad_graph_*` builtins. Same two-harness shape as Phase 3a:
//!
//! 1. `fuzz_phase3b_no_panic_on_bad_args` — feed garbage `Value` shapes
//!    into each new dispatch arm and assert every call returns
//!    `Result`, never panics. Out-of-range axis, OOB gather indices,
//!    empty cat input list, non-array where array expected, non-param
//!    nodes in `backward_collect`, inverted reforward range — all must
//!    Err.
//!
//! 2. `fuzz_phase3b_array_round_trip` — for each op that produces a
//!    valid forward pass, assert the output is finite and (where
//!    applicable) shape-consistent with the inputs.

use std::panic;

use bolero::check;

use cjc_ad::dispatch_grad_graph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn reset() {
    let _ = dispatch_grad_graph("grad_graph_new", &[]);
}

fn input_node(data: Vec<f64>, shape: &[usize]) -> i64 {
    match dispatch_grad_graph(
        "grad_graph_input",
        &[Value::Tensor(Tensor::from_vec(data, shape).unwrap())],
    )
    .unwrap()
    .unwrap()
    {
        Value::Int(i) => i,
        _ => unreachable!(),
    }
}

fn forward_vec(idx: i64) -> Vec<f64> {
    match dispatch_grad_graph("grad_graph_forward", &[Value::Int(idx)])
        .unwrap()
        .unwrap()
    {
        Value::Tensor(t) => t.to_vec(),
        _ => unreachable!(),
    }
}

fn int_array(values: &[i64]) -> Value {
    Value::Array(std::rc::Rc::new(values.iter().map(|&i| Value::Int(i)).collect()))
}

// ─── Harness 1: garbage args must Err, never panic ──────────────────

#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_phase3b_no_panic_on_bad_args() {
    check!()
        .with_iterations(256)
        .for_each(|bytes: &[u8]| {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                reset();

                let op_byte = bytes.first().copied().unwrap_or(0) % 5;
                let bad_byte = bytes.get(1).copied().unwrap_or(0) % 7;

                // Pre-populate the graph with one node so some ops have
                // something valid to point at, while the *bad* arg below
                // is guaranteed to fail validation.
                let _ = input_node(vec![1.0, 2.0, 3.0], &[3]);

                let bad_args: Vec<Value> = match bad_byte {
                    0 => vec![],                                // wrong arg count
                    1 => vec![Value::Int(-1)],                  // negative idx
                    2 => vec![Value::Int(99_999)],              // OOB idx
                    3 => vec![Value::Int(0), Value::Int(99)],   // OOB axis / 2nd
                    4 => vec![Value::Int(0), int_array(&[99]), Value::Int(0)], // OOB index in array
                    5 => vec![int_array(&[]), Value::Int(0)],   // empty cat input
                    _ => vec![Value::Int(0), Value::Float(1.0)], // wrong type
                };

                let name = match op_byte {
                    0 => "grad_graph_batch_norm",
                    1 => "grad_graph_gather",
                    2 => "grad_graph_cat",
                    3 => "grad_graph_reforward",
                    _ => "grad_graph_backward_collect",
                };

                // Either Err, or (if the bad-byte combination happens to be
                // valid) Ok — but never a panic.
                let _ = dispatch_grad_graph(name, &bad_args);
            }));
            assert!(result.is_ok(), "dispatch panicked on garbage Phase-3b input");
        });
}

// ─── Harness 2: valid input → finite, shape-consistent output ──────

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
        let clamped = if f.is_finite() { f.clamp(-4.0, 4.0) } else { 0.0 };
        out.push(clamped);
    }
    Some(out)
}

#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_phase3b_array_round_trip() {
    check!()
        .with_iterations(256)
        .for_each(|bytes: &[u8]| {
            let n = ((bytes.first().copied().unwrap_or(2) & 0x0F) as usize).max(2);
            let op_byte = bytes.get(1).copied().unwrap_or(0) % 4;
            let Some(data) = read_floats(bytes, 2, n) else {
                return;
            };

            reset();
            let inp = input_node(data.clone(), &[n]);

            let out_idx = match op_byte {
                0 => {
                    // batch_norm — output should have same shape, all finite.
                    match dispatch_grad_graph(
                        "grad_graph_batch_norm",
                        &[Value::Int(inp)],
                    ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() }
                }
                1 => {
                    // gather — pick first half.
                    let k = (n / 2).max(1);
                    let indices: Vec<i64> = (0..k as i64).collect();
                    let r = match dispatch_grad_graph(
                        "grad_graph_gather",
                        &[Value::Int(inp), int_array(&indices), Value::Int(0)],
                    ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
                    let v = forward_vec(r);
                    assert_eq!(v.len(), k, "gather output length mismatch");
                    for x in &v { assert!(x.is_finite()); }
                    return;
                }
                2 => {
                    // cat — concatenate the input with itself.
                    let r = match dispatch_grad_graph(
                        "grad_graph_cat",
                        &[int_array(&[inp, inp]), Value::Int(0)],
                    ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
                    let v = forward_vec(r);
                    assert_eq!(v.len(), 2 * n, "cat output length mismatch");
                    // First half should byte-equal the input.
                    let first_half: Vec<u64> = v[..n].iter().map(|x| x.to_bits()).collect();
                    let inp_bits: Vec<u64> = data.iter().map(|x| x.to_bits()).collect();
                    assert_eq!(first_half, inp_bits, "cat first half ≠ input bits");
                    return;
                }
                _ => {
                    // backward_collect — wrap input as a parameter, sum, collect.
                    reset();
                    let p = match dispatch_grad_graph(
                        "grad_graph_param",
                        &[Value::Tensor(Tensor::from_vec(data.clone(), &[n]).unwrap())],
                    ).unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
                    let s = match dispatch_grad_graph("grad_graph_sum", &[Value::Int(p)])
                        .unwrap().unwrap() { Value::Int(i) => i, _ => unreachable!() };
                    let arr = match dispatch_grad_graph(
                        "grad_graph_backward_collect",
                        &[Value::Int(s), int_array(&[p])],
                    ).unwrap().unwrap() {
                        Value::Array(rc) => rc,
                        _ => unreachable!(),
                    };
                    assert_eq!(arr.len(), 1);
                    // dL/dp where L = sum(p) is all-ones.
                    let g = match &arr[0] {
                        Value::Tensor(t) => t.to_vec(),
                        _ => unreachable!(),
                    };
                    assert_eq!(g.len(), n);
                    for x in &g {
                        assert!((*x - 1.0).abs() < 1e-12, "expected dL/dp = 1, got {x}");
                    }
                    return;
                }
            };

            let v = forward_vec(out_idx);
            assert_eq!(v.len(), n, "shape-preserving op changed length");
            for x in &v { assert!(x.is_finite()); }
        });
}
