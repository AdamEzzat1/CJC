//! Phase 3c — bolero fuzz targets for the `grad_graph_*` dispatch surface.
//!
//! Two harnesses:
//!
//! 1. `fuzz_grad_graph_structural` — decode a random byte sequence into a
//!    sequence of dispatch calls (input/param/const/add/mul/sub/scalar_mul
//!    /tanh/sum/mean/forward/zero_grad/backward/reset). The contract being
//!    fuzzed is *graceful state recovery*: after any individual op, the
//!    ambient graph must remain usable. The dispatch layer itself never
//!    panics on out-of-range indices or non-scalar backward (we Err those).
//!    Underlying `GradGraph` / `Tensor` kernels can still panic on shape
//!    mismatches (`add(scalar, [1,3])` etc.) — that's a documented internal
//!    invariant, and the fuzzer treats those as recoverable: a per-op
//!    `catch_unwind` resets the ambient graph and we continue. What's
//!    actually being verified is that no panic *corrupts the dispatch
//!    layer's bookkeeping* in a way that affects subsequent calls.
//!
//! 2. `fuzz_grad_graph_numerical` — feed bounded-finite f64 inputs (range
//!    [-4, 4]) through a fixed pipeline (input → tanh → sum → forward) and
//!    assert the output is finite and within the analytically expected
//!    range |sum| ≤ n. This is a smoke target catching ULP-level blow-ups
//!    in dispatch's domain handling: tanh ∘ sum maps to (-n, n), well
//!    inside f64 range, so any NaN/Inf or |out|>n is a bug.
//!
//! Both harnesses use `bolero::check!` which compiles to proptest on
//! Windows/macOS (default) and to libfuzzer/AFL under `cargo bolero`.

use std::panic;

use bolero::check;

use cjc_ad::dispatch_grad_graph;
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

/// Apply one dispatch call decoded from a byte. The first byte selects the
/// op (mod 14), subsequent bytes pick node indices. `nodes` is the running
/// list of created node ids that ops can reference.
fn step_one(bytes: &[u8], cursor: &mut usize, nodes: &mut Vec<i64>) {
    let take = |c: &mut usize| -> u8 {
        if *c < bytes.len() {
            let b = bytes[*c];
            *c += 1;
            b
        } else {
            0
        }
    };
    let pick = |idx: u8, nodes: &[i64]| -> Option<i64> {
        if nodes.is_empty() { None } else { Some(nodes[idx as usize % nodes.len()]) }
    };

    let op = take(cursor) % 14;
    match op {
        0 => {
            let _ = dispatch_grad_graph("grad_graph_new", &[]);
            nodes.clear();
        }
        1 => {
            let t = Tensor::from_vec(vec![0.5, -0.25, 1.0], &[1, 3]).unwrap();
            if let Ok(Some(Value::Int(i))) =
                dispatch_grad_graph("grad_graph_input", &[Value::Tensor(t)])
            {
                nodes.push(i);
            }
        }
        2 => {
            let t = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3]).unwrap();
            if let Ok(Some(Value::Int(i))) =
                dispatch_grad_graph("grad_graph_param", &[Value::Tensor(t)])
            {
                nodes.push(i);
            }
        }
        3 => {
            let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
            if let Ok(Some(Value::Int(i))) =
                dispatch_grad_graph("grad_graph_const", &[Value::Tensor(t)])
            {
                nodes.push(i);
            }
        }
        4 => {
            if let (Some(a), Some(b)) =
                (pick(take(cursor), nodes), pick(take(cursor), nodes))
            {
                if let Ok(Some(Value::Int(i))) = dispatch_grad_graph(
                    "grad_graph_add",
                    &[Value::Int(a), Value::Int(b)],
                ) {
                    nodes.push(i);
                }
            }
        }
        5 => {
            if let (Some(a), Some(b)) =
                (pick(take(cursor), nodes), pick(take(cursor), nodes))
            {
                if let Ok(Some(Value::Int(i))) = dispatch_grad_graph(
                    "grad_graph_mul",
                    &[Value::Int(a), Value::Int(b)],
                ) {
                    nodes.push(i);
                }
            }
        }
        6 => {
            if let (Some(a), Some(b)) =
                (pick(take(cursor), nodes), pick(take(cursor), nodes))
            {
                if let Ok(Some(Value::Int(i))) = dispatch_grad_graph(
                    "grad_graph_sub",
                    &[Value::Int(a), Value::Int(b)],
                ) {
                    nodes.push(i);
                }
            }
        }
        7 => {
            if let Some(a) = pick(take(cursor), nodes) {
                if let Ok(Some(Value::Int(i))) = dispatch_grad_graph(
                    "grad_graph_scalar_mul",
                    &[Value::Int(a), Value::Float(0.5)],
                ) {
                    nodes.push(i);
                }
            }
        }
        8 => {
            if let Some(a) = pick(take(cursor), nodes) {
                if let Ok(Some(Value::Int(i))) =
                    dispatch_grad_graph("grad_graph_tanh", &[Value::Int(a)])
                {
                    nodes.push(i);
                }
            }
        }
        9 => {
            if let Some(a) = pick(take(cursor), nodes) {
                if let Ok(Some(Value::Int(i))) =
                    dispatch_grad_graph("grad_graph_sum", &[Value::Int(a)])
                {
                    nodes.push(i);
                }
            }
        }
        10 => {
            if let Some(a) = pick(take(cursor), nodes) {
                if let Ok(Some(Value::Int(i))) =
                    dispatch_grad_graph("grad_graph_mean", &[Value::Int(a)])
                {
                    nodes.push(i);
                }
            }
        }
        11 => {
            if let Some(a) = pick(take(cursor), nodes) {
                let _ = dispatch_grad_graph("grad_graph_forward", &[Value::Int(a)]);
            }
        }
        12 => {
            let _ = dispatch_grad_graph("grad_graph_zero_grad", &[]);
        }
        13 => {
            if let Some(a) = pick(take(cursor), nodes) {
                // Backward on a non-scalar should Err, not panic.
                let _ = dispatch_grad_graph("grad_graph_backward", &[Value::Int(a)]);
            }
        }
        _ => unreachable!(),
    }
}

/// Structural fuzzer: replay byte-decoded op sequences and assert that the
/// *dispatch layer* recovers from any shape-mismatch panic in the underlying
/// `GradGraph`/`Tensor` kernels. After each panic we reset the ambient graph
/// and keep going; what we're testing is that no panic corrupts dispatch
/// bookkeeping (the `RefCell`, the thread-local, etc.) in a way that breaks
/// subsequent valid calls.
#[test]
fn fuzz_grad_graph_structural() {
    check!().with_type::<Vec<u8>>().for_each(|input: &Vec<u8>| {
        let _ = dispatch_grad_graph("grad_graph_new", &[]);
        let mut nodes: Vec<i64> = Vec::new();
        let mut cursor = 0usize;
        let max = input.len().min(48);
        let slice = &input[..max];

        while cursor < slice.len() {
            let cursor_before = cursor;
            let nodes_snapshot = nodes.clone();
            let unwound = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let mut local_cursor = cursor_before;
                let mut local_nodes = nodes_snapshot.clone();
                step_one(slice, &mut local_cursor, &mut local_nodes);
                (local_cursor, local_nodes)
            }));
            match unwound {
                Ok((c, n)) => {
                    cursor = c;
                    nodes = n;
                }
                Err(_) => {
                    // Underlying kernel panicked (shape mismatch, division by
                    // zero, etc.). Reset and continue — the dispatch layer
                    // itself must still work for subsequent calls.
                    let _ = dispatch_grad_graph("grad_graph_new", &[]);
                    nodes.clear();
                    // Advance at least one byte so we don't infinite-loop on
                    // a deterministically-panicking op.
                    cursor = (cursor_before + 1).min(slice.len());

                    // Smoke-test that dispatch is still alive after the panic.
                    let alive = dispatch_grad_graph("grad_graph_len", &[]);
                    assert!(
                        matches!(alive, Ok(Some(Value::Int(0)))),
                        "dispatch corrupted after kernel panic at byte {}: {:?}",
                        cursor_before, alive,
                    );
                }
            }
        }
    });
}

/// Numerical fuzzer: tanh∘sum∘input produces finite output for any
/// bounded in-domain input vector. tanh ∈ (-1, 1) ⇒ sum ∈ (-n, n).
#[test]
fn fuzz_grad_graph_numerical() {
    check!().with_type::<[f64; 5]>().for_each(|arr: &[f64; 5]| {
        // Skip non-finite or large inputs — we're testing the numerical
        // pipeline, not error handling. f64 NaN/Inf are tested separately
        // by the `numerical_input_handles_extremes` wiring test.
        if arr.iter().any(|x| !x.is_finite() || x.abs() > 4.0) {
            return;
        }

        let _ = dispatch_grad_graph("grad_graph_new", &[]);
        let t = Tensor::from_vec(arr.to_vec(), &[1, 5]).unwrap();
        let i = match dispatch_grad_graph("grad_graph_input", &[Value::Tensor(t)]) {
            Ok(Some(Value::Int(i))) => i,
            other => panic!("grad_graph_input returned {:?}", other),
        };
        let th = match dispatch_grad_graph("grad_graph_tanh", &[Value::Int(i)]) {
            Ok(Some(Value::Int(i))) => i,
            other => panic!("grad_graph_tanh returned {:?}", other),
        };
        let s = match dispatch_grad_graph("grad_graph_sum", &[Value::Int(th)]) {
            Ok(Some(Value::Int(i))) => i,
            other => panic!("grad_graph_sum returned {:?}", other),
        };
        let out = match dispatch_grad_graph("grad_graph_forward", &[Value::Int(s)]) {
            Ok(Some(Value::Tensor(t))) => t,
            other => panic!("grad_graph_forward returned {:?}", other),
        };
        for v in out.to_vec() {
            assert!(v.is_finite(), "tanh∘sum produced non-finite: {v}");
            // tanh ∈ (-1, 1), sum of 5 elements ⇒ |out| < 5 + slack.
            assert!(v.abs() <= 5.0 + 1e-9, "tanh∘sum out of range: {v}");
        }
    });
}
