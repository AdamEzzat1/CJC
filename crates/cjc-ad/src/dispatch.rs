//! Phase 3c — language-level dispatch for `grad_graph_*` builtins.
//!
//! Routed from both `cjc-eval` and `cjc-mir-exec` *after* the shared
//! `cjc_runtime::dispatch_builtin` and after `cjc_quantum::dispatch_quantum`,
//! so user `.cjcl` code can construct GradGraph nodes, run forward/backward,
//! and read tensors — all from the language.
//!
//! # Handle representation
//!
//! Phase 3c picks **Option B** from the brief: a single ambient `GradGraph`
//! per execution thread, addressed by `usize` node index. The graph lives in
//! a `thread_local!` so both executors share the same instance within a
//! single test thread, which is exactly what AST↔MIR parity testing needs.
//!
//! Node indices cross the language boundary as `Value::Int(i64)`. This avoids
//! introducing a new `Value` variant, preserving `Value` enum layout and the
//! MIR register layout (HARD RULE #1 from the Phase 3c brief).
//!
//! # Why a satellite dispatch and not `cjc-runtime/builtins.rs`?
//!
//! `cjc-ad` already depends on `cjc-runtime`. Adding the reverse dependency
//! would create a cycle. The `cjc-quantum::dispatch_quantum` precedent shows
//! satellite dispatch is the canonical pattern for this exact case.

use std::cell::RefCell;
use std::rc::Rc;

use cjc_runtime::value::Value;
use cjc_runtime::tensor::Tensor;

use crate::GradGraph;
use crate::pinn::Activation;

thread_local! {
    /// Ambient graph for the current thread. Reset by `grad_graph_new()`.
    static AMBIENT: RefCell<GradGraph> = RefCell::new(GradGraph::new());
}

/// Run a closure with mutable access to the ambient graph.
///
/// Borrow is held only for the duration of the closure, which never re-enters
/// `dispatch_grad_graph` — so nested-borrow panics are structurally impossible.
pub fn with_ambient<R>(f: impl FnOnce(&mut GradGraph) -> R) -> R {
    AMBIENT.with(|cell| f(&mut *cell.borrow_mut()))
}

/// Reset the ambient graph to a fresh empty state.
///
/// Called by `grad_graph_new()`. Tests should call this at the top of any
/// `.cjcl` source that constructs a graph, since thread-locals can outlive
/// individual test functions when cargo runs tests on a thread pool.
pub fn reset_ambient() {
    AMBIENT.with(|cell| {
        *cell.borrow_mut() = GradGraph::new();
    });
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

fn arg_idx(name: &str, val: &Value) -> Result<usize, String> {
    match val {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        Value::Int(i) => Err(format!("{}: node index must be non-negative, got {}", name, i)),
        other => Err(format!(
            "{}: expected Int node index, got {}",
            name,
            other.type_name()
        )),
    }
}

/// Like `arg_idx`, but additionally asserts that the index is in range for
/// the current ambient graph. Returning `Err` here is what makes the
/// dispatch layer fuzz-safe: out-of-range indices surface as language-level
/// errors rather than panicking deep inside `GradGraph::*` index ops.
fn arg_idx_checked(name: &str, val: &Value) -> Result<usize, String> {
    let idx = arg_idx(name, val)?;
    let len = with_ambient(|g| g.len());
    if idx >= len {
        return Err(format!(
            "{}: node index {} out of range (graph has {} nodes)",
            name, idx, len
        ));
    }
    Ok(idx)
}

fn arg_tensor<'a>(name: &str, val: &'a Value) -> Result<&'a Tensor, String> {
    match val {
        Value::Tensor(t) => Ok(t),
        other => Err(format!(
            "{}: expected Tensor, got {}",
            name,
            other.type_name()
        )),
    }
}

fn arg_f64(name: &str, val: &Value) -> Result<f64, String> {
    match val {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        other => Err(format!(
            "{}: expected number, got {}",
            name,
            other.type_name()
        )),
    }
}

/// Decode `Value::Array` of `Value::Int` (each ≥ 1) into a `Vec<usize>`.
/// Used by shape arguments — `grad_graph_reshape` and (Phase 3b) `gather`,
/// `cat`. Rejects negative or non-integer entries with a clean Err.
fn arg_usize_array(name: &str, val: &Value) -> Result<Vec<usize>, String> {
    match val {
        Value::Array(elems) => {
            let mut out = Vec::with_capacity(elems.len());
            for (i, e) in elems.iter().enumerate() {
                match e {
                    Value::Int(n) if *n >= 0 => out.push(*n as usize),
                    Value::Int(n) => {
                        return Err(format!(
                            "{name}: shape element [{i}] must be non-negative, got {n}"
                        ))
                    }
                    other => {
                        return Err(format!(
                            "{name}: shape element [{i}] must be Int, got {}",
                            other.type_name()
                        ))
                    }
                }
            }
            Ok(out)
        }
        other => Err(format!(
            "{name}: expected Array of Int (shape), got {}",
            other.type_name()
        )),
    }
}

fn arg_str<'a>(name: &str, val: &'a Value) -> Result<&'a str, String> {
    match val {
        Value::String(s) => Ok(s.as_str()),
        other => Err(format!(
            "{}: expected String, got {}",
            name,
            other.type_name()
        )),
    }
}

fn parse_activation(name: &str, s: &str) -> Result<Activation, String> {
    match s {
        "tanh" => Ok(Activation::Tanh),
        "sigmoid" => Ok(Activation::Sigmoid),
        "relu" => Ok(Activation::Relu),
        "none" => Ok(Activation::None),
        "gelu" => Ok(Activation::Gelu),
        "silu" => Ok(Activation::Silu),
        "elu" => Ok(Activation::Elu),
        "selu" => Ok(Activation::Selu),
        "sin" => Ok(Activation::SinAct),
        other => Err(format!(
            "{}: unknown activation {:?} (expected one of: tanh sigmoid relu none gelu silu elu selu sin)",
            name, other
        )),
    }
}

fn idx_value(idx: usize) -> Value {
    Value::Int(idx as i64)
}

fn tensor_value(t: Tensor) -> Value {
    Value::Tensor(t)
}

// ─── Dispatch ─────────────────────────────────────────────────────────────

/// Dispatch table for language-level `grad_graph_*` builtins.
///
/// Returns `Ok(None)` if `name` is not one of ours, so callers can fall
/// through to other dispatch tables.
pub fn dispatch_grad_graph(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    let result: Value = match name {
        // ── Construction ────────────────────────────────────────────
        "grad_graph_new" => {
            arg_count(name, args, 0)?;
            reset_ambient();
            Value::Void
        }
        "grad_graph_param" => {
            arg_count(name, args, 1)?;
            let t = arg_tensor(name, &args[0])?.clone();
            let idx = with_ambient(|g| g.parameter(t));
            idx_value(idx)
        }
        "grad_graph_input" => {
            arg_count(name, args, 1)?;
            let t = arg_tensor(name, &args[0])?.clone();
            let idx = with_ambient(|g| g.input(t));
            idx_value(idx)
        }
        // const = non-trainable input; same node behavior, kept as a separate
        // name so user code documents intent.
        "grad_graph_const" => {
            arg_count(name, args, 1)?;
            let t = arg_tensor(name, &args[0])?.clone();
            let idx = with_ambient(|g| g.input(t));
            idx_value(idx)
        }

        // ── Pointwise ops ───────────────────────────────────────────
        "grad_graph_add" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let b = arg_idx_checked(name, &args[1])?;
            idx_value(with_ambient(|g| g.add(a, b)))
        }
        "grad_graph_sub" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let b = arg_idx_checked(name, &args[1])?;
            idx_value(with_ambient(|g| g.sub(a, b)))
        }
        "grad_graph_mul" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let b = arg_idx_checked(name, &args[1])?;
            idx_value(with_ambient(|g| g.mul(a, b)))
        }
        "grad_graph_div" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let b = arg_idx_checked(name, &args[1])?;
            idx_value(with_ambient(|g| g.div(a, b)))
        }
        "grad_graph_neg" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.neg(a)))
        }
        "grad_graph_scalar_mul" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let s = arg_f64(name, &args[1])?;
            idx_value(with_ambient(|g| g.scalar_mul(a, s)))
        }
        "grad_graph_pow" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            // Brief specifies integer power, but cjc-ad's `pow` takes f64.
            // Accept either i64 or f64; both narrow to the same call.
            let n = arg_f64(name, &args[1])?;
            idx_value(with_ambient(|g| g.pow(a, n)))
        }
        "grad_graph_exp" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.exp(a)))
        }
        "grad_graph_ln" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.ln(a)))
        }
        "grad_graph_sqrt" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.sqrt(a)))
        }
        "grad_graph_sin" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.sin(a)))
        }
        "grad_graph_cos" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.cos(a)))
        }
        "grad_graph_tanh" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.tanh_act(a)))
        }

        // ── Reductions / matmul ─────────────────────────────────────
        "grad_graph_sum" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.sum(a)))
        }
        "grad_graph_mean" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.mean(a)))
        }
        "grad_graph_matmul" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let b = arg_idx_checked(name, &args[1])?;
            idx_value(with_ambient(|g| g.matmul(a, b)))
        }

        // ── Fused MLP layer ────────────────────────────────────────
        // Signature: grad_graph_mlp_layer(input, weight, bias, activation_str) -> NodeIdx
        "grad_graph_mlp_layer" => {
            arg_count(name, args, 4)?;
            let inp = arg_idx_checked(name, &args[0])?;
            let w = arg_idx_checked(name, &args[1])?;
            let b = arg_idx_checked(name, &args[2])?;
            let act_str = arg_str(name, &args[3])?;
            let act = parse_activation(name, act_str)?;
            idx_value(with_ambient(|g| g.mlp_layer(inp, w, b, act)))
        }

        // ── Forward / state read ────────────────────────────────────
        "grad_graph_forward" => {
            // The graph is forward-evaluated eagerly on each op call, so
            // forward() at a node is just a tensor read. Kept under this name
            // for spec parity with the brief.
            arg_count(name, args, 1)?;
            let idx = arg_idx_checked(name, &args[0])?;
            tensor_value(with_ambient(|g| g.tensor(idx)))
        }
        "grad_graph_set_tensor" => {
            arg_count(name, args, 2)?;
            let idx = arg_idx_checked(name, &args[0])?;
            let t = arg_tensor(name, &args[1])?.clone();
            with_ambient(|g| g.set_tensor(idx, t));
            Value::Void
        }
        "grad_graph_param_grad" => {
            arg_count(name, args, 1)?;
            let idx = arg_idx_checked(name, &args[0])?;
            let g_opt = with_ambient(|g| g.grad(idx));
            match g_opt {
                Some(t) => tensor_value(t),
                None => {
                    return Err(format!(
                        "grad_graph_param_grad: node {} has no gradient (not a parameter, or backward not run)",
                        idx
                    ))
                }
            }
        }

        // ── Backward / optimizer support ────────────────────────────
        "grad_graph_zero_grad" => {
            arg_count(name, args, 0)?;
            with_ambient(|g| g.zero_grad());
            Value::Void
        }
        "grad_graph_backward" => {
            arg_count(name, args, 1)?;
            let loss_idx = arg_idx_checked(name, &args[0])?;
            // Backward requires a scalar leaf. If the loss node is non-scalar
            // (e.g., an unreduced add/mul), the inner GradGraph would assert.
            // Surface that as a clean Err instead of unwinding.
            let shape = with_ambient(|g| g.tensor(loss_idx).shape().to_vec());
            let total: usize = shape.iter().product();
            if total != 1 {
                return Err(format!(
                    "grad_graph_backward: loss node {} has shape {:?} (numel={}), but backward requires a scalar (numel=1). Apply grad_graph_sum or grad_graph_mean first.",
                    loss_idx, shape, total
                ));
            }
            with_ambient(|g| g.backward(loss_idx));
            Value::Void
        }
        "grad_graph_clip_grad_norm" => {
            arg_count(name, args, 1)?;
            let max_norm = arg_f64(name, &args[0])?;
            let actual = with_ambient(|g| g.clip_grad_norm(max_norm));
            Value::Float(actual)
        }

        // ── Phase 3a: transformer-backbone activations & ops ────────
        // All map directly to existing `cjc_ad::GradGraph` Rust methods.
        // Determinism contract is unchanged: `softmax`, `cross_entropy`,
        // and `layer_norm` use Kahan accumulators internally; `gelu`,
        // `silu`, `reshape` are pointwise / shape-only and inherit the
        // tensor kernel's bit-exact behavior.

        "grad_graph_softmax" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.softmax(a)))
        }
        "grad_graph_cross_entropy" => {
            arg_count(name, args, 2)?;
            let logits = arg_idx_checked(name, &args[0])?;
            let targets = arg_idx_checked(name, &args[1])?;
            idx_value(with_ambient(|g| g.cross_entropy(logits, targets)))
        }
        "grad_graph_layer_norm" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.layer_norm(a)))
        }
        "grad_graph_gelu" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.gelu(a)))
        }
        "grad_graph_silu" => {
            arg_count(name, args, 1)?;
            let a = arg_idx_checked(name, &args[0])?;
            idx_value(with_ambient(|g| g.silu(a)))
        }
        // grad_graph_reshape(node, shape_array) -> NodeIdx
        // shape_array is a Value::Array of Value::Int (positive).
        "grad_graph_reshape" => {
            arg_count(name, args, 2)?;
            let a = arg_idx_checked(name, &args[0])?;
            let shape = arg_usize_array(name, &args[1])?;
            // Materialize new tensor; shape mismatch surfaces as a clean Err
            // rather than the inner `expect()` panic.
            let cur_numel: usize = with_ambient(|g| g.tensor(a).shape().iter().product());
            let new_numel: usize = shape.iter().product();
            if new_numel != cur_numel {
                return Err(format!(
                    "grad_graph_reshape: cannot reshape tensor with {cur_numel} elements into shape {shape:?} ({new_numel} elements)"
                ));
            }
            idx_value(with_ambient(|g| g.reshape(a, &shape)))
        }

        // ── Introspection ───────────────────────────────────────────
        "grad_graph_len" => {
            arg_count(name, args, 0)?;
            Value::Int(with_ambient(|g| g.len()) as i64)
        }

        // Not one of ours — let the caller fall through.
        _ => return Ok(None),
    };

    Ok(Some(result))
}

// Make `Rc` reachable to suppress unused-import warning when no Rc paths
// are exercised in this module today; it stays available for future arms
// that need to wrap collections.
#[allow(dead_code)]
fn _rc_marker() -> Rc<()> {
    Rc::new(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cjc_runtime::tensor::Tensor;

    fn t(data: &[f64], shape: &[usize]) -> Value {
        Value::Tensor(Tensor::from_vec(data.to_vec(), shape).unwrap())
    }

    #[test]
    fn new_resets_ambient() {
        // Build something, reset, confirm len drops to 0.
        let _ = dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let _ = dispatch_grad_graph(
            "grad_graph_param",
            &[t(&[1.0, 2.0], &[2])],
        )
        .unwrap();
        let len_before = match dispatch_grad_graph("grad_graph_len", &[])
            .unwrap()
            .unwrap()
        {
            Value::Int(n) => n,
            _ => panic!("expected Int"),
        };
        assert!(len_before >= 1);
        let _ = dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let len_after = match dispatch_grad_graph("grad_graph_len", &[])
            .unwrap()
            .unwrap()
        {
            Value::Int(n) => n,
            _ => panic!("expected Int"),
        };
        assert_eq!(len_after, 0);
    }

    #[test]
    fn add_two_params_then_backward_returns_unit_grads() {
        let _ = dispatch_grad_graph("grad_graph_new", &[]).unwrap();
        let a = match dispatch_grad_graph("grad_graph_param", &[t(&[3.0], &[1])])
            .unwrap()
            .unwrap()
        {
            Value::Int(i) => i,
            _ => panic!(),
        };
        let b = match dispatch_grad_graph("grad_graph_param", &[t(&[4.0], &[1])])
            .unwrap()
            .unwrap()
        {
            Value::Int(i) => i,
            _ => panic!(),
        };
        let s = match dispatch_grad_graph(
            "grad_graph_add",
            &[Value::Int(a), Value::Int(b)],
        )
        .unwrap()
        .unwrap()
        {
            Value::Int(i) => i,
            _ => panic!(),
        };
        // Forward value: 3 + 4 = 7
        let s_t = match dispatch_grad_graph("grad_graph_forward", &[Value::Int(s)])
            .unwrap()
            .unwrap()
        {
            Value::Tensor(t) => t,
            _ => panic!(),
        };
        assert_eq!(s_t.to_vec(), vec![7.0]);
        // Backward: dS/da = dS/db = 1
        let _ = dispatch_grad_graph("grad_graph_zero_grad", &[]).unwrap();
        let _ = dispatch_grad_graph("grad_graph_backward", &[Value::Int(s)]).unwrap();
        let ga = match dispatch_grad_graph("grad_graph_param_grad", &[Value::Int(a)])
            .unwrap()
            .unwrap()
        {
            Value::Tensor(t) => t,
            _ => panic!(),
        };
        let gb = match dispatch_grad_graph("grad_graph_param_grad", &[Value::Int(b)])
            .unwrap()
            .unwrap()
        {
            Value::Tensor(t) => t,
            _ => panic!(),
        };
        assert_eq!(ga.to_vec(), vec![1.0]);
        assert_eq!(gb.to_vec(), vec![1.0]);
    }

    #[test]
    fn unknown_name_returns_none() {
        let r = dispatch_grad_graph("not_a_grad_graph_builtin", &[]).unwrap();
        assert!(r.is_none());
    }
}
