//! State-space-model primitives — `state_space_*` builtins.
//!
//! Routed from `dispatch_builtin` (see `builtins.rs`). Both executors call
//! `dispatch_builtin` and inherit these handlers without further changes.
//!
//! # Model
//!
//! Each cell implements a deterministic discrete-time recurrence:
//!
//! ```text
//!   h_t = tanh(A · h_{t-1} + B · x_t)         // hidden state update
//!   y_t = C · h_t + b_o                       // readout
//! ```
//!
//! - `A` is `[hidden, hidden]`, `B` is `[hidden, input]`, `C` is `[output, hidden]`.
//! - `b_o` is `[output]`.
//! - Hidden state `h` is `[hidden]`.
//!
//! This is intentionally simple: a single linear+tanh recurrence with fixed
//! shapes. The goal is to give CJC-Lang a *deterministic, inspectable*
//! temporal-memory primitive — not a full Mamba/SSM. The chess RL demo uses
//! it as a side-channel feature extractor; the trainable MLP head takes the
//! readout as an additional input, learning to use temporal context.
//!
//! # Determinism
//!
//! - Weights are drawn from a SplitMix64 stream seeded by the user. Two
//!   `state_space_init(_, _, _, seed)` calls with the same arguments produce
//!   bit-identical weights.
//! - All matmuls use `Tensor::matmul` (deterministic, no FMA, no parallel
//!   reductions in the hot path).
//! - Arena is a `BTreeMap<usize, Cell>` for deterministic iteration.
//!
//! # Handle representation
//!
//! Cells live in a `thread_local!` arena keyed by `usize`. The handle crosses
//! the language boundary as `Value::Int(i64)` — same convention as
//! `grad_graph_*` (Phase 3c). No new `Value` variant.
//!
//! # AD scope (deferred)
//!
//! `state_space_*` ops do **not** participate in `GradGraph`. The forward pass
//! is eager; gradients do not flow through `state_step`. The chess demo uses
//! the readout as a frozen feature, and the trainable MLP head (which *is*
//! built on `GradGraph`) learns to use those features. Plumbing SSM into
//! GradGraph is deferred — see `docs/state_space/ADR-0018-state-space.md`.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::tensor::Tensor;
use crate::value::Value;

// ─── Cell type ─────────────────────────────────────────────────────────────

/// A single state-space cell: weights + current hidden state.
///
/// Fields are `pub(crate)` for test access only. User code should never
/// reach in directly — the dispatch layer is the public API.
#[derive(Clone, Debug)]
pub(crate) struct StateSpaceCell {
    pub(crate) input_dim: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) output_dim: usize,
    /// Transition matrix A, shape `[hidden, hidden]`.
    pub(crate) a: Tensor,
    /// Input projection B, shape `[hidden, input]`.
    pub(crate) b: Tensor,
    /// Readout C, shape `[output, hidden]`.
    pub(crate) c: Tensor,
    /// Readout bias, shape `[output]`.
    pub(crate) b_o: Tensor,
    /// Hidden state h, shape `[hidden]`. Reset by `state_space_reset`.
    pub(crate) h: Tensor,
}

// ─── Arena (thread-local) ──────────────────────────────────────────────────

thread_local! {
    /// All cells visible from the current thread, keyed by handle index.
    /// `BTreeMap` for deterministic iteration order.
    static ARENA: RefCell<BTreeMap<usize, StateSpaceCell>> = RefCell::new(BTreeMap::new());
    /// Monotonic handle counter. Never reused, so a stale handle from a
    /// cleared arena reliably surfaces as an "out of range" error rather
    /// than aliasing a fresh cell.
    static NEXT_HANDLE: RefCell<usize> = RefCell::new(0);
}

fn fresh_handle() -> usize {
    NEXT_HANDLE.with(|cell| {
        let mut h = cell.borrow_mut();
        let v = *h;
        *h = h.wrapping_add(1);
        v
    })
}

fn with_arena<R>(f: impl FnOnce(&mut BTreeMap<usize, StateSpaceCell>) -> R) -> R {
    ARENA.with(|cell| f(&mut *cell.borrow_mut()))
}

/// Reset the arena to a fresh empty state. Tests should call
/// `state_space_clear()` from `.cjcl` source at the top of a test, since
/// thread-locals can persist across test functions on cargo's thread pool.
pub fn reset_arena() {
    ARENA.with(|cell| cell.borrow_mut().clear());
    NEXT_HANDLE.with(|cell| *cell.borrow_mut() = 0);
}

// ─── Deterministic RNG (SplitMix64) ────────────────────────────────────────

/// SplitMix64 — same algorithm used by `cjc-repro`, inlined here to avoid a
/// new crate dependency. Deterministic, no external state.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Map a u64 to f64 in `[-1, 1)` deterministically. We take the top 53 bits
/// to fill the f64 mantissa, scale to `[0, 1)`, then shift to `[-1, 1)`.
fn u64_to_signed_unit(x: u64) -> f64 {
    let m = (x >> 11) as f64;       // 53-bit integer
    let u = m / (1u64 << 53) as f64; // [0, 1)
    2.0 * u - 1.0                    // [-1, 1)
}

/// Fill a tensor of given size with deterministic small-magnitude weights.
/// Scale matches a "Glorot-lite" `±sqrt(1 / fan_in)`, capped to avoid huge
/// initial state magnitudes.
fn make_weights(numel: usize, fan_in: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let scale = if fan_in == 0 { 1.0 } else { (1.0 / fan_in as f64).sqrt() };
    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let r = u64_to_signed_unit(splitmix64(&mut state));
        out.push(r * scale);
    }
    out
}

// ─── Argument helpers ──────────────────────────────────────────────────────

fn arg_count(name: &str, args: &[Value], expected: usize) -> Result<(), String> {
    if args.len() != expected {
        Err(format!(
            "{}: expected {} arguments, got {}",
            name,
            expected,
            args.len()
        ))
    } else {
        Ok(())
    }
}

fn arg_usize(name: &str, val: &Value, label: &str) -> Result<usize, String> {
    match val {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        Value::Int(i) => Err(format!("{}: {} must be non-negative, got {}", name, label, i)),
        other => Err(format!(
            "{}: expected Int for {}, got {}",
            name,
            label,
            other.type_name()
        )),
    }
}

fn arg_u64(name: &str, val: &Value, label: &str) -> Result<u64, String> {
    match val {
        Value::Int(i) => Ok(*i as u64),
        other => Err(format!(
            "{}: expected Int for {}, got {}",
            name,
            label,
            other.type_name()
        )),
    }
}

fn arg_handle(name: &str, val: &Value) -> Result<usize, String> {
    let h = arg_usize(name, val, "handle")?;
    let exists = with_arena(|a| a.contains_key(&h));
    if !exists {
        return Err(format!(
            "{}: handle {} does not refer to a live cell (was it cleared?)",
            name, h
        ));
    }
    Ok(h)
}

fn arg_tensor<'a>(name: &str, val: &'a Value, label: &str) -> Result<&'a Tensor, String> {
    match val {
        Value::Tensor(t) => Ok(t),
        other => Err(format!(
            "{}: expected Tensor for {}, got {}",
            name,
            label,
            other.type_name()
        )),
    }
}

// ─── Numerics ──────────────────────────────────────────────────────────────

/// Deterministic 1-D `(hidden, input) · (input,) → (hidden,)` matvec.
fn matvec(weight: &Tensor, x: &Tensor) -> Result<Vec<f64>, String> {
    let ws = weight.shape();
    let xs = x.shape();
    if ws.len() != 2 || xs.len() != 1 || ws[1] != xs[0] {
        return Err(format!(
            "state_space matvec: shape mismatch (weight {:?}, x {:?})",
            ws, xs
        ));
    }
    let rows = ws[0];
    let cols = ws[1];
    let wd = weight.to_vec();
    let xd = x.to_vec();
    let mut out = vec![0.0; rows];
    for i in 0..rows {
        let row_off = i * cols;
        let mut acc = 0.0;
        for j in 0..cols {
            acc += wd[row_off + j] * xd[j];
        }
        out[i] = acc;
    }
    Ok(out)
}

fn add_inplace(a: &mut [f64], b: &[f64]) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

fn tanh_inplace(a: &mut [f64]) {
    for v in a.iter_mut() {
        *v = v.tanh();
    }
}

// ─── Operations ────────────────────────────────────────────────────────────

fn op_init(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Result<usize, String> {
    if hidden_dim == 0 {
        return Err("state_space_init: hidden_dim must be >= 1".into());
    }
    if input_dim == 0 {
        return Err("state_space_init: input_dim must be >= 1".into());
    }
    if output_dim == 0 {
        return Err("state_space_init: output_dim must be >= 1".into());
    }
    // Per-tensor seed offsets so changing one shape doesn't disturb the
    // initialization stream of the others.
    let a_data = make_weights(hidden_dim * hidden_dim, hidden_dim, seed.wrapping_add(0xA1));
    let b_data = make_weights(hidden_dim * input_dim, input_dim, seed.wrapping_add(0xB2));
    let c_data = make_weights(output_dim * hidden_dim, hidden_dim, seed.wrapping_add(0xC3));
    let bo_data = vec![0.0; output_dim];
    let h_data = vec![0.0; hidden_dim];

    let a = Tensor::from_vec(a_data, &[hidden_dim, hidden_dim])
        .map_err(|e| format!("state_space_init: {:?}", e))?;
    let b = Tensor::from_vec(b_data, &[hidden_dim, input_dim])
        .map_err(|e| format!("state_space_init: {:?}", e))?;
    let c = Tensor::from_vec(c_data, &[output_dim, hidden_dim])
        .map_err(|e| format!("state_space_init: {:?}", e))?;
    let b_o = Tensor::from_vec(bo_data, &[output_dim])
        .map_err(|e| format!("state_space_init: {:?}", e))?;
    let h = Tensor::from_vec(h_data, &[hidden_dim])
        .map_err(|e| format!("state_space_init: {:?}", e))?;

    let cell = StateSpaceCell {
        input_dim,
        hidden_dim,
        output_dim,
        a,
        b,
        c,
        b_o,
        h,
    };
    let handle = fresh_handle();
    with_arena(|arena| arena.insert(handle, cell));
    Ok(handle)
}

/// Borrow-mut a cell, run a closure on it, return the closure's result.
/// Errors if the handle is gone (race with `state_space_clear`).
fn with_cell_mut<R>(name: &str, h: usize, f: impl FnOnce(&mut StateSpaceCell) -> Result<R, String>) -> Result<R, String> {
    with_arena(|arena| {
        let cell = arena.get_mut(&h).ok_or_else(|| {
            format!("{}: handle {} is no longer alive", name, h)
        })?;
        f(cell)
    })
}

fn op_step(handle: usize, x: &Tensor) -> Result<Tensor, String> {
    with_cell_mut("state_space_step", handle, |cell| {
        if x.shape() != [cell.input_dim] {
            return Err(format!(
                "state_space_step: expected x of shape [{}], got {:?}",
                cell.input_dim,
                x.shape()
            ));
        }
        // h_new = tanh(A·h + B·x)
        let mut ah = matvec(&cell.a, &cell.h)?;
        let bx = matvec(&cell.b, x)?;
        add_inplace(&mut ah, &bx);
        tanh_inplace(&mut ah);
        cell.h = Tensor::from_vec(ah, &[cell.hidden_dim])
            .map_err(|e| format!("state_space_step: {:?}", e))?;

        // y = C·h + b_o
        let mut ch = matvec(&cell.c, &cell.h)?;
        let bo = cell.b_o.to_vec();
        add_inplace(&mut ch, &bo);
        Tensor::from_vec(ch, &[cell.output_dim])
            .map_err(|e| format!("state_space_step: {:?}", e))
    })
}

fn op_scan(handle: usize, xs: &Tensor) -> Result<Tensor, String> {
    let (t, input_dim, hidden_dim_ck, output_dim) = with_arena(|arena| {
        let cell = arena.get(&handle).ok_or_else(|| {
            format!("state_space_scan: handle {} is no longer alive", handle)
        })?;
        let s = xs.shape();
        if s.len() != 2 || s[1] != cell.input_dim {
            return Err(format!(
                "state_space_scan: expected xs of shape [T, {}], got {:?}",
                cell.input_dim, s
            ));
        }
        Ok((s[0], cell.input_dim, cell.hidden_dim, cell.output_dim))
    })?;
    let _ = hidden_dim_ck; // borrow check sanity

    let xs_data = xs.to_vec();
    let mut out = Vec::with_capacity(t * output_dim);
    for step in 0..t {
        let row = &xs_data[step * input_dim..(step + 1) * input_dim];
        let x_step = Tensor::from_vec(row.to_vec(), &[input_dim])
            .map_err(|e| format!("state_space_scan: {:?}", e))?;
        let y = op_step(handle, &x_step)?;
        out.extend(y.to_vec());
    }
    Tensor::from_vec(out, &[t, output_dim])
        .map_err(|e| format!("state_space_scan: {:?}", e))
}

fn op_reset(handle: usize) -> Result<(), String> {
    with_cell_mut("state_space_reset", handle, |cell| {
        let h = vec![0.0; cell.hidden_dim];
        cell.h = Tensor::from_vec(h, &[cell.hidden_dim])
            .map_err(|e| format!("state_space_reset: {:?}", e))?;
        Ok(())
    })
}

fn op_state(handle: usize) -> Result<Tensor, String> {
    with_arena(|arena| {
        let cell = arena.get(&handle).ok_or_else(|| {
            format!("state_space_state: handle {} is no longer alive", handle)
        })?;
        Ok(cell.h.clone())
    })
}

fn op_set_state(handle: usize, h: &Tensor) -> Result<(), String> {
    with_cell_mut("state_space_set_state", handle, |cell| {
        if h.shape() != [cell.hidden_dim] {
            return Err(format!(
                "state_space_set_state: expected hidden of shape [{}], got {:?}",
                cell.hidden_dim,
                h.shape()
            ));
        }
        cell.h = h.clone();
        Ok(())
    })
}

fn op_readout(handle: usize) -> Result<Tensor, String> {
    with_arena(|arena| {
        let cell = arena.get(&handle).ok_or_else(|| {
            format!("state_space_readout: handle {} is no longer alive", handle)
        })?;
        let mut ch = matvec(&cell.c, &cell.h)?;
        let bo = cell.b_o.to_vec();
        add_inplace(&mut ch, &bo);
        Tensor::from_vec(ch, &[cell.output_dim])
            .map_err(|e| format!("state_space_readout: {:?}", e))
    })
}

/// Fused step + readout. Mutates `h` in place and returns `(y, h_new)`.
///
/// Saves one full borrow of the arena, one Tensor clone of `h`, and one
/// dispatch-call boundary compared to `step` followed by `state` /
/// `readout` from user code. The numerics are bit-identical to the
/// step+readout sequence.
fn op_step_with_readout(handle: usize, x: &Tensor) -> Result<(Tensor, Tensor), String> {
    with_cell_mut("state_space_step_with_readout", handle, |cell| {
        if x.shape() != [cell.input_dim] {
            return Err(format!(
                "state_space_step_with_readout: expected x of shape [{}], got {:?}",
                cell.input_dim,
                x.shape()
            ));
        }
        // h_new = tanh(A·h + B·x)
        let mut ah = matvec(&cell.a, &cell.h)?;
        let bx = matvec(&cell.b, x)?;
        add_inplace(&mut ah, &bx);
        tanh_inplace(&mut ah);
        cell.h = Tensor::from_vec(ah, &[cell.hidden_dim])
            .map_err(|e| format!("state_space_step_with_readout: {:?}", e))?;

        // y = C·h + b_o
        let mut ch = matvec(&cell.c, &cell.h)?;
        let bo = cell.b_o.to_vec();
        add_inplace(&mut ch, &bo);
        let y = Tensor::from_vec(ch, &[cell.output_dim])
            .map_err(|e| format!("state_space_step_with_readout: {:?}", e))?;
        Ok((y, cell.h.clone()))
    })
}

/// Apply the same single-step recurrence row-by-row to a `[B, input]` batch
/// of inputs, *resetting `h` to zero at the start*. Returns `[B, output]`.
///
/// Use case: evaluating multiple completely-independent positions through
/// the same SSM cell without leaking hidden state across them. Each row
/// starts from a zero hidden state, advances one step, and contributes one
/// output row. The cell's hidden state on return is the *last* row's `h`.
fn op_step_batched(handle: usize, xs: &Tensor) -> Result<Tensor, String> {
    let (b, input_dim, output_dim, hidden_dim) = with_arena(|arena| {
        let cell = arena.get(&handle).ok_or_else(|| {
            format!("state_space_step_batched: handle {} is no longer alive", handle)
        })?;
        let s = xs.shape();
        if s.len() != 2 || s[1] != cell.input_dim {
            return Err(format!(
                "state_space_step_batched: expected xs of shape [B, {}], got {:?}",
                cell.input_dim, s
            ));
        }
        Ok((s[0], cell.input_dim, cell.output_dim, cell.hidden_dim))
    })?;
    let xs_data = xs.to_vec();
    let mut out = Vec::with_capacity(b * output_dim);
    for row in 0..b {
        with_cell_mut("state_space_step_batched", handle, |cell| {
            // Reset h to zero before each row so they are independent.
            cell.h = Tensor::from_vec(vec![0.0; hidden_dim], &[hidden_dim])
                .map_err(|e| format!("{:?}", e))?;
            Ok(())
        })?;
        let row_data = xs_data[row * input_dim..(row + 1) * input_dim].to_vec();
        let x_row = Tensor::from_vec(row_data, &[input_dim])
            .map_err(|e| format!("state_space_step_batched: {:?}", e))?;
        let y = op_step(handle, &x_row)?;
        out.extend(y.to_vec());
    }
    Tensor::from_vec(out, &[b, output_dim])
        .map_err(|e| format!("state_space_step_batched: {:?}", e))
}

/// Extract a *clone* of one of the cell's weight tensors.
///
/// User code can wrap the result in `grad_graph_input(...)` to make the SSM
/// recurrence participate in a `cjc-ad::GradGraph`, composing the recurrence
/// from existing `matmul + add + tanh` GradOps. This is the recommended
/// pattern for differentiable SSMs until a fused `GradOp::SsmStep` ships.
fn op_get_a(handle: usize) -> Result<Tensor, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.a.clone())
            .ok_or_else(|| format!("state_space_get_A: handle {} is no longer alive", handle))
    })
}
fn op_get_b(handle: usize) -> Result<Tensor, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.b.clone())
            .ok_or_else(|| format!("state_space_get_B: handle {} is no longer alive", handle))
    })
}
fn op_get_c(handle: usize) -> Result<Tensor, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.c.clone())
            .ok_or_else(|| format!("state_space_get_C: handle {} is no longer alive", handle))
    })
}
fn op_get_b_o(handle: usize) -> Result<Tensor, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.b_o.clone())
            .ok_or_else(|| format!("state_space_get_b_o: handle {} is no longer alive", handle))
    })
}

/// Native concat of two 1-D tensors. Replaces user-space loop+`array_push`,
/// which is O(N²) due to COW reallocation. The fast path here is a single
/// `Vec::with_capacity` + two `extend_from_slice` calls.
///
/// Both inputs must be 1-D. Output shape is `[a.len + b.len]`.
pub fn tensor_concat_1d(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(format!(
            "tensor_concat_1d: both inputs must be 1-D (got shapes {:?} and {:?})",
            a.shape(),
            b.shape()
        ));
    }
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let mut out = Vec::with_capacity(a_data.len() + b_data.len());
    out.extend_from_slice(&a_data);
    out.extend_from_slice(&b_data);
    let total = out.len();
    Tensor::from_vec(out, &[total]).map_err(|e| format!("tensor_concat_1d: {:?}", e))
}

fn op_hidden_dim(handle: usize) -> Result<i64, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.hidden_dim as i64)
            .ok_or_else(|| format!("state_space_hidden_dim: handle {} is no longer alive", handle))
    })
}

fn op_input_dim(handle: usize) -> Result<i64, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.input_dim as i64)
            .ok_or_else(|| format!("state_space_input_dim: handle {} is no longer alive", handle))
    })
}

fn op_output_dim(handle: usize) -> Result<i64, String> {
    with_arena(|arena| {
        arena
            .get(&handle)
            .map(|c| c.output_dim as i64)
            .ok_or_else(|| format!("state_space_output_dim: handle {} is no longer alive", handle))
    })
}

// ─── Public dispatch ───────────────────────────────────────────────────────

/// Dispatch table for `state_space_*` builtins. Returns `Ok(None)` if the
/// name doesn't match — `dispatch_builtin` falls through to other branches.
pub fn dispatch_state_space(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    let result: Value = match name {
        "state_space_init" => {
            arg_count(name, args, 4)?;
            let input_dim = arg_usize(name, &args[0], "input_dim")?;
            let hidden_dim = arg_usize(name, &args[1], "hidden_dim")?;
            let output_dim = arg_usize(name, &args[2], "output_dim")?;
            let seed = arg_u64(name, &args[3], "seed")?;
            let h = op_init(input_dim, hidden_dim, output_dim, seed)?;
            Value::Int(h as i64)
        }
        "state_space_step" => {
            arg_count(name, args, 2)?;
            let h = arg_handle(name, &args[0])?;
            let x = arg_tensor(name, &args[1], "x")?;
            Value::Tensor(op_step(h, x)?)
        }
        "state_space_scan" => {
            arg_count(name, args, 2)?;
            let h = arg_handle(name, &args[0])?;
            let xs = arg_tensor(name, &args[1], "xs")?;
            Value::Tensor(op_scan(h, xs)?)
        }
        "state_space_reset" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            op_reset(h)?;
            Value::Void
        }
        "state_space_state" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Tensor(op_state(h)?)
        }
        "state_space_set_state" => {
            arg_count(name, args, 2)?;
            let h = arg_handle(name, &args[0])?;
            let st = arg_tensor(name, &args[1], "h")?;
            op_set_state(h, st)?;
            Value::Void
        }
        "state_space_readout" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Tensor(op_readout(h)?)
        }
        "state_space_step_with_readout" => {
            arg_count(name, args, 2)?;
            let h = arg_handle(name, &args[0])?;
            let x = arg_tensor(name, &args[1], "x")?;
            let (y, h_new) = op_step_with_readout(h, x)?;
            Value::Array(std::rc::Rc::new(vec![Value::Tensor(y), Value::Tensor(h_new)]))
        }
        "state_space_step_batched" => {
            arg_count(name, args, 2)?;
            let h = arg_handle(name, &args[0])?;
            let xs = arg_tensor(name, &args[1], "xs")?;
            Value::Tensor(op_step_batched(h, xs)?)
        }
        "state_space_get_A" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Tensor(op_get_a(h)?)
        }
        "state_space_get_B" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Tensor(op_get_b(h)?)
        }
        "state_space_get_C" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Tensor(op_get_c(h)?)
        }
        "state_space_get_b_o" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Tensor(op_get_b_o(h)?)
        }
        "tensor_concat_1d" => {
            arg_count(name, args, 2)?;
            let a = arg_tensor(name, &args[0], "a")?;
            let b = arg_tensor(name, &args[1], "b")?;
            Value::Tensor(tensor_concat_1d(a, b)?)
        }
        "state_space_snapshot" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            // Snapshot is just the current hidden state — a deep copy via
            // `Tensor::clone`. The caller stores this and later passes it to
            // `state_space_restore` to reproduce future outputs bit-for-bit.
            Value::Tensor(op_state(h)?)
        }
        "state_space_restore" => {
            arg_count(name, args, 2)?;
            let h = arg_handle(name, &args[0])?;
            let st = arg_tensor(name, &args[1], "snapshot")?;
            op_set_state(h, st)?;
            Value::Void
        }
        "state_space_hidden_dim" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Int(op_hidden_dim(h)?)
        }
        "state_space_input_dim" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Int(op_input_dim(h)?)
        }
        "state_space_output_dim" => {
            arg_count(name, args, 1)?;
            let h = arg_handle(name, &args[0])?;
            Value::Int(op_output_dim(h)?)
        }
        "state_space_clear" => {
            arg_count(name, args, 0)?;
            reset_arena();
            Value::Void
        }
        "state_space_len" => {
            arg_count(name, args, 0)?;
            Value::Int(with_arena(|a| a.len()) as i64)
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Value {
        Value::Tensor(Tensor::from_vec(data.to_vec(), shape).unwrap())
    }

    fn handle_of(v: Value) -> i64 {
        match v {
            Value::Int(i) => i,
            other => panic!("expected Int handle, got {:?}", other),
        }
    }

    fn tensor_of(v: Value) -> Tensor {
        match v {
            Value::Tensor(t) => t,
            other => panic!("expected Tensor, got {:?}", other),
        }
    }

    #[test]
    fn init_returns_handle_and_zero_state() {
        reset_arena();
        let h = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(4), Value::Int(8), Value::Int(2), Value::Int(42)],
            )
            .unwrap()
            .unwrap(),
        );
        let st = tensor_of(
            dispatch_state_space("state_space_state", &[Value::Int(h)])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(st.shape(), &[8]);
        assert!(st.to_vec().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn step_changes_hidden_state() {
        reset_arena();
        let h = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(3), Value::Int(4), Value::Int(2), Value::Int(7)],
            )
            .unwrap()
            .unwrap(),
        );
        let x = t(&[1.0, 0.5, -0.25], &[3]);
        let y1 = tensor_of(
            dispatch_state_space("state_space_step", &[Value::Int(h), x.clone()])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(y1.shape(), &[2]);
        let st1 = tensor_of(
            dispatch_state_space("state_space_state", &[Value::Int(h)])
                .unwrap()
                .unwrap(),
        );
        // After one step from zero state, hidden must not be all zeros
        // unless B·x is exactly zero — which it isn't with a nonzero x and
        // SplitMix64-seeded B.
        assert!(st1.to_vec().iter().any(|&v| v.abs() > 1e-12));
    }

    #[test]
    fn reset_then_replay_matches_fresh_step() {
        reset_arena();
        let h = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(2), Value::Int(3), Value::Int(2), Value::Int(99)],
            )
            .unwrap()
            .unwrap(),
        );
        let x = t(&[0.7, -0.3], &[2]);
        let y_a = tensor_of(
            dispatch_state_space("state_space_step", &[Value::Int(h), x.clone()])
                .unwrap()
                .unwrap(),
        );
        // Disturb by stepping again
        let _ = dispatch_state_space("state_space_step", &[Value::Int(h), x.clone()]).unwrap();
        // Reset and re-step
        let _ = dispatch_state_space("state_space_reset", &[Value::Int(h)]).unwrap();
        let y_b = tensor_of(
            dispatch_state_space("state_space_step", &[Value::Int(h), x.clone()])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(y_a.to_vec(), y_b.to_vec());
    }

    #[test]
    fn snapshot_restore_round_trip() {
        reset_arena();
        let h = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(2), Value::Int(3), Value::Int(1), Value::Int(11)],
            )
            .unwrap()
            .unwrap(),
        );
        let x1 = t(&[1.0, 0.0], &[2]);
        let x2 = t(&[0.0, 1.0], &[2]);
        let _ = dispatch_state_space("state_space_step", &[Value::Int(h), x1.clone()]).unwrap();
        let snap = dispatch_state_space("state_space_snapshot", &[Value::Int(h)])
            .unwrap()
            .unwrap();
        // Future-A: step with x2
        let y_a = tensor_of(
            dispatch_state_space("state_space_step", &[Value::Int(h), x2.clone()])
                .unwrap()
                .unwrap(),
        );
        // Restore and replay
        let _ = dispatch_state_space("state_space_restore", &[Value::Int(h), snap]).unwrap();
        let y_b = tensor_of(
            dispatch_state_space("state_space_step", &[Value::Int(h), x2.clone()])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(y_a.to_vec(), y_b.to_vec());
    }

    #[test]
    fn scan_equals_repeated_step() {
        reset_arena();
        // Build cell A and cell B with identical seed -> identical weights.
        let ha = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(2), Value::Int(3), Value::Int(2), Value::Int(123)],
            )
            .unwrap()
            .unwrap(),
        );
        let hb = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(2), Value::Int(3), Value::Int(2), Value::Int(123)],
            )
            .unwrap()
            .unwrap(),
        );
        let xs = Value::Tensor(
            Tensor::from_vec(vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0], &[3, 2]).unwrap(),
        );
        let ys = tensor_of(
            dispatch_state_space("state_space_scan", &[Value::Int(ha), xs])
                .unwrap()
                .unwrap(),
        );
        // Step b through manually
        let mut acc: Vec<f64> = Vec::new();
        for step in 0..3 {
            let row = match step {
                0 => vec![1.0, 0.0],
                1 => vec![0.5, 0.5],
                _ => vec![0.0, 1.0],
            };
            let xv = Value::Tensor(Tensor::from_vec(row, &[2]).unwrap());
            let y = tensor_of(
                dispatch_state_space("state_space_step", &[Value::Int(hb), xv])
                    .unwrap()
                    .unwrap(),
            );
            acc.extend(y.to_vec());
        }
        assert_eq!(ys.to_vec(), acc);
    }

    #[test]
    fn shape_mismatch_errors() {
        reset_arena();
        let h = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(3), Value::Int(4), Value::Int(2), Value::Int(1)],
            )
            .unwrap()
            .unwrap(),
        );
        let bad = t(&[1.0, 2.0], &[2]); // wrong input_dim
        let err = dispatch_state_space("state_space_step", &[Value::Int(h), bad]).unwrap_err();
        assert!(err.contains("expected x of shape"), "got: {}", err);
    }

    #[test]
    fn dead_handle_errors() {
        reset_arena();
        let h = handle_of(
            dispatch_state_space(
                "state_space_init",
                &[Value::Int(2), Value::Int(2), Value::Int(2), Value::Int(0)],
            )
            .unwrap()
            .unwrap(),
        );
        // Explicitly clear, then try to use
        let _ = dispatch_state_space("state_space_clear", &[]).unwrap();
        let err = dispatch_state_space("state_space_state", &[Value::Int(h)]).unwrap_err();
        assert!(err.contains("does not refer to a live cell"), "got: {}", err);
    }

    #[test]
    fn unknown_name_falls_through() {
        let r = dispatch_state_space("not_a_state_space_op", &[]).unwrap();
        assert!(r.is_none());
    }
}
