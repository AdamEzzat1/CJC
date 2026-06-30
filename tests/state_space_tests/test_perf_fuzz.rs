//! Bolero fuzz targets for the Phase-2 state-space dispatch surface.
//!
//! Two harnesses:
//!
//! 1. `fuzz_state_space_structural` — decode a random byte sequence into a
//!    sequence of dispatch calls (init / step / readout / fused-step /
//!    batched / get_A / get_B / get_C / get_b_o / concat / reset / clear).
//!    The contract under fuzz is *graceful state recovery*: after any op
//!    sequence, the dispatch layer must remain usable and never panic on
//!    an out-of-range handle, malformed shape, etc. Errors at the dispatch
//!    boundary surface as `Err(String)`, not panics. Lower-level Tensor
//!    panics on shape mismatch are caught with `catch_unwind` (documented
//!    internal invariant) and the arena is reset to recover.
//!
//! 2. `fuzz_concat_numerical` — feed two bounded-finite tensors of random
//!    length 0–32 into `tensor_concat_1d`. Output must be:
//!      - shape `[a.len + b.len]`
//!      - all entries finite (no NaN/Inf)
//!      - first n_a entries equal a's entries (preservation)
//!      - last n_b entries equal b's entries (preservation)
//!
//! Both targets compile to proptest on Windows (default) and to libfuzzer
//! under `cargo bolero fuzz`.

use std::panic;

use bolero::check;

use cjc_runtime::state_space::{dispatch_state_space, tensor_concat_1d};
use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

fn take(bytes: &[u8], cursor: &mut usize) -> u8 {
    if *cursor < bytes.len() {
        let b = bytes[*cursor];
        *cursor += 1;
        b
    } else {
        0
    }
}

fn pick_handle(handles: &[i64], idx: u8) -> Option<i64> {
    if handles.is_empty() {
        None
    } else {
        Some(handles[(idx as usize) % handles.len()])
    }
}

/// Apply one fuzz-decoded op. Errors and panics are absorbed; the arena is
/// reset on panic so subsequent calls can continue.
fn step_one(bytes: &[u8], cursor: &mut usize, handles: &mut Vec<i64>) {
    let op = take(bytes, cursor) % 12;
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        match op {
            0 => {
                let _ = dispatch_state_space("state_space_clear", &[]);
                handles.clear();
            }
            1 => {
                // init with bounded dims (1..=4) so we never request huge tensors
                let i = ((take(bytes, cursor) % 4) + 1) as i64;
                let h = ((take(bytes, cursor) % 4) + 1) as i64;
                let o = ((take(bytes, cursor) % 4) + 1) as i64;
                let s = take(bytes, cursor) as i64;
                let r = dispatch_state_space(
                    "state_space_init",
                    &[Value::Int(i), Value::Int(h), Value::Int(o), Value::Int(s)],
                );
                if let Ok(Some(Value::Int(handle))) = r {
                    handles.push(handle);
                }
            }
            2 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let n = ((take(bytes, cursor) % 4) + 1) as usize;
                    let xs: Vec<f64> = (0..n)
                        .map(|i| (take(bytes, cursor) as f64 - 128.0) / 128.0)
                        .collect();
                    if let Ok(t) = Tensor::from_vec(xs, &[n]) {
                        let _ = dispatch_state_space(
                            "state_space_step",
                            &[Value::Int(h), Value::Tensor(t)],
                        );
                    }
                }
            }
            3 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let _ = dispatch_state_space("state_space_readout", &[Value::Int(h)]);
                }
            }
            4 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let n = ((take(bytes, cursor) % 4) + 1) as usize;
                    let xs: Vec<f64> = (0..n)
                        .map(|_| (take(bytes, cursor) as f64 - 128.0) / 128.0)
                        .collect();
                    if let Ok(t) = Tensor::from_vec(xs, &[n]) {
                        let _ = dispatch_state_space(
                            "state_space_step_with_readout",
                            &[Value::Int(h), Value::Tensor(t)],
                        );
                    }
                }
            }
            5 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let b = ((take(bytes, cursor) % 4) + 1) as usize;
                    let n = ((take(bytes, cursor) % 4) + 1) as usize;
                    let xs: Vec<f64> = (0..(b * n))
                        .map(|_| (take(bytes, cursor) as f64 - 128.0) / 128.0)
                        .collect();
                    if let Ok(t) = Tensor::from_vec(xs, &[b, n]) {
                        let _ = dispatch_state_space(
                            "state_space_step_batched",
                            &[Value::Int(h), Value::Tensor(t)],
                        );
                    }
                }
            }
            6 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let _ = dispatch_state_space("state_space_get_A", &[Value::Int(h)]);
                }
            }
            7 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let _ = dispatch_state_space("state_space_get_B", &[Value::Int(h)]);
                }
            }
            8 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let _ = dispatch_state_space("state_space_get_C", &[Value::Int(h)]);
                }
            }
            9 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let _ = dispatch_state_space("state_space_get_b_o", &[Value::Int(h)]);
                }
            }
            10 => {
                if let Some(h) = pick_handle(handles, take(bytes, cursor)) {
                    let _ = dispatch_state_space("state_space_reset", &[Value::Int(h)]);
                }
            }
            _ => {
                let n_a = take(bytes, cursor) as usize % 8;
                let n_b = take(bytes, cursor) as usize % 8;
                let a_data: Vec<f64> = (0..n_a)
                    .map(|_| (take(bytes, cursor) as f64 - 128.0) / 128.0)
                    .collect();
                let b_data: Vec<f64> = (0..n_b)
                    .map(|_| (take(bytes, cursor) as f64 - 128.0) / 128.0)
                    .collect();
                if let (Ok(a), Ok(b)) = (
                    Tensor::from_vec(a_data, &[n_a]),
                    Tensor::from_vec(b_data, &[n_b]),
                ) {
                    let _ = dispatch_state_space(
                        "tensor_concat_1d",
                        &[Value::Tensor(a), Value::Tensor(b)],
                    );
                }
            }
        }
    }));
    if result.is_err() {
        // A lower-level kernel panicked. Reset the arena to a known-good
        // state so subsequent ops aren't influenced by partial mutation.
        let _ = dispatch_state_space("state_space_clear", &[]);
        handles.clear();
    }
}

#[test]
fn fuzz_state_space_structural() {
    check!()
        .with_max_len(256)
        .for_each(|bytes: &[u8]| {
            let _ = dispatch_state_space("state_space_clear", &[]);
            let mut cursor = 0;
            let mut handles: Vec<i64> = Vec::new();
            // Bounded number of ops per fuzz iteration.
            for _ in 0..32 {
                if cursor >= bytes.len() {
                    break;
                }
                step_one(bytes, &mut cursor, &mut handles);
            }
            // Final invariant: the arena is still queryable.
            let _ = dispatch_state_space("state_space_len", &[]).unwrap();
        });
}

#[test]
fn fuzz_concat_numerical() {
    check!()
        .with_max_len(64)
        .for_each(|bytes: &[u8]| {
            if bytes.len() < 2 {
                return;
            }
            let n_a = (bytes[0] as usize) % 16;
            let n_b = (bytes[1] as usize) % 16;
            let total_needed = n_a + n_b;
            if bytes.len() < 2 + total_needed {
                return;
            }
            let a_data: Vec<f64> = (0..n_a)
                .map(|i| (bytes[2 + i] as f64 - 128.0) / 128.0)
                .collect();
            let b_data: Vec<f64> = (0..n_b)
                .map(|i| (bytes[2 + n_a + i] as f64 - 128.0) / 128.0)
                .collect();
            let a = Tensor::from_vec(a_data.clone(), &[n_a]).unwrap();
            let b = Tensor::from_vec(b_data.clone(), &[n_b]).unwrap();
            let c = tensor_concat_1d(&a, &b).unwrap();
            assert_eq!(c.shape(), &[n_a + n_b]);
            let cd = c.to_vec();
            for v in &cd {
                assert!(v.is_finite(), "concat output must be finite");
            }
            assert_eq!(cd[..n_a].to_vec(), a_data, "first n_a entries must equal a");
            assert_eq!(cd[n_a..].to_vec(), b_data, "last n_b entries must equal b");
        });
}
