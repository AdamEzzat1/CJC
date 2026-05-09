//! Phase 0.1 — language-level dispatch for `abng_*` builtins.
//!
//! Routed from both `cjc-eval` and `cjc-mir-exec` *after*
//! `cjc_runtime::dispatch_builtin`, `cjc_quantum::dispatch_quantum`, and
//! `cjc_ad::dispatch_grad_graph`, so user `.cjcl` code can construct
//! ABNG graphs, observe values, snapshot, and replay.
//!
//! # Handle representation
//!
//! Phase 0.1 uses a **multi-graph** arena: one process-wide
//! `thread_local!` `BTreeMap<i64, AdaptiveBeliefGraph>` keyed by an
//! integer `graph_id` returned from `abng_new`. This differs from
//! `cjc-ad`'s single-ambient pattern because:
//!
//! * Tests need independent graphs without resetting global state.
//! * Future `cjcl abng diff a.snap b.snap` needs two graphs in one
//!   process.
//! * The CJC-Lang user-facing API is naturally `let g = abng_new(seed)`,
//!   not `abng_new()` with hidden state.
//!
//! `graph_id` crosses the language boundary as `Value::Int(i64)`. This
//! preserves the `Value` enum layout (HARD RULE #1).
//!
//! # Why a satellite dispatch
//!
//! Adding `cjc-abng` to `cjc-runtime`'s dispatch table would introduce
//! a `cjc-runtime → cjc-abng` dependency, while `cjc-abng` already
//! depends on `cjc-runtime`. That's a cycle. The
//! `cjc_quantum::dispatch_quantum` and `cjc_ad::dispatch_grad_graph`
//! precedents establish satellite dispatch as the canonical pattern.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use cjc_runtime::tensor::Tensor;
use cjc_runtime::value::Value;

use crate::graph::{AdaptiveBeliefGraph, GraphError};
use crate::leaf_head::{encode_activation_tag, parse_activation};
use crate::serialize;

thread_local! {
    /// Per-thread arena of named graphs. Keyed by `graph_id` from
    /// `abng_new`; `next_id` is the monotonically-increasing counter.
    static ARENA: RefCell<Arena> = RefCell::new(Arena::default());
}

#[derive(Default)]
struct Arena {
    graphs: BTreeMap<i64, AdaptiveBeliefGraph>,
    next_id: i64,
}

/// Reset the per-thread arena. Tests should call this when running on a
/// shared thread pool to avoid leaking state across tests.
pub fn reset_arena() {
    ARENA.with(|cell| *cell.borrow_mut() = Arena::default());
}

fn with_arena<R>(f: impl FnOnce(&mut Arena) -> R) -> R {
    ARENA.with(|cell| f(&mut *cell.borrow_mut()))
}

/// Read a graph by id, returning a friendly error if it isn't there.
fn with_graph<R>(
    name: &str,
    graph_id: i64,
    f: impl FnOnce(&mut AdaptiveBeliefGraph) -> R,
) -> Result<R, String> {
    with_arena(|a| match a.graphs.get_mut(&graph_id) {
        Some(g) => Ok(f(g)),
        None => Err(format!("{name}: no graph with id {graph_id}")),
    })
}

// ─── Argument decoders ─────────────────────────────────────────────────────

fn arg_count(name: &str, args: &[Value], expected: usize) -> Result<(), String> {
    if args.len() != expected {
        Err(format!(
            "{name}: expected {expected} arguments, got {}",
            args.len()
        ))
    } else {
        Ok(())
    }
}

fn arg_i64(name: &str, val: &Value) -> Result<i64, String> {
    match val {
        Value::Int(i) => Ok(*i),
        other => Err(format!(
            "{name}: expected Int, got {}",
            other.type_name()
        )),
    }
}

fn arg_u32_node(name: &str, val: &Value) -> Result<u32, String> {
    let i = arg_i64(name, val)?;
    if i < 0 {
        return Err(format!("{name}: node id must be non-negative, got {i}"));
    }
    if i > u32::MAX as i64 {
        return Err(format!("{name}: node id {i} exceeds u32::MAX"));
    }
    Ok(i as u32)
}

fn arg_f64(name: &str, val: &Value) -> Result<f64, String> {
    match val {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        other => Err(format!(
            "{name}: expected number, got {}",
            other.type_name()
        )),
    }
}

fn arg_tensor<'a>(name: &str, val: &'a Value) -> Result<&'a Tensor, String> {
    match val {
        Value::Tensor(t) => Ok(t),
        other => Err(format!(
            "{name}: expected Tensor, got {}",
            other.type_name()
        )),
    }
}

/// Decode a `Value::Bytes` (Rc<RefCell<Vec<u8>>>) into a fresh `Vec<u8>`
/// snapshot. We clone because `replay` is called on a borrowed slice and
/// the borrow path is simpler if we own the bytes.
fn arg_bytes(name: &str, val: &Value) -> Result<Vec<u8>, String> {
    match val {
        Value::Bytes(b) => Ok(b.borrow().clone()),
        Value::ByteSlice(b) => Ok((**b).clone()),
        other => Err(format!(
            "{name}: expected Bytes, got {}",
            other.type_name()
        )),
    }
}

/// Decode a `Tensor` of integer-valued `f64`s in `[0, 255]` to a `Vec<u8>`
/// (prefix bytes). Used by [`abng_descend`](dispatch_abng) and
/// [`abng_route_path`](dispatch_abng); their inputs are the output of
/// `abng_encode_prefix`, which produces integer-valued bin indices in a
/// `Tensor`.
fn arg_prefix_bytes(name: &str, val: &Value) -> Result<Vec<u8>, String> {
    let t = arg_tensor(name, val)?;
    let v = t.to_vec();
    let mut out = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() || x.fract() != 0.0 || x < 0.0 || x > 255.0 {
            return Err(format!(
                "{name}: prefix[{i}] = {x} is not an integer byte in [0, 255]"
            ));
        }
        out.push(x as u8);
    }
    Ok(out)
}

/// Decode a `Tensor` argument that's expected to be 1-D `f64` data.
/// Returns the flat `Vec<f64>` and the length.
fn arg_tensor_1d_f64(name: &str, val: &Value) -> Result<Vec<f64>, String> {
    let t = arg_tensor(name, val)?;
    if t.shape().len() != 1 {
        return Err(format!(
            "{name}: expected 1-D Tensor, got shape {:?}",
            t.shape()
        ));
    }
    Ok(t.to_vec())
}

/// Decode a 1-D Tensor of integer-valued positive `f64`s as a `Vec<u32>`
/// — used for `hidden_dims` when configuring the leaf head.
fn arg_dims_tensor(name: &str, val: &Value) -> Result<Vec<u32>, String> {
    let t = arg_tensor(name, val)?;
    if t.shape().len() != 1 {
        return Err(format!(
            "{name}: expected 1-D Tensor of dim sizes, got shape {:?}",
            t.shape()
        ));
    }
    let v = t.to_vec();
    let mut out = Vec::with_capacity(v.len());
    for (i, &x) in v.iter().enumerate() {
        if !x.is_finite() || x.fract() != 0.0 || x < 1.0 || x > u32::MAX as f64 {
            return Err(format!(
                "{name}: hidden_dims[{i}] = {x} must be a positive integer"
            ));
        }
        out.push(x as u32);
    }
    Ok(out)
}

fn arg_string<'a>(name: &str, val: &'a Value) -> Result<&'a str, String> {
    match val {
        Value::String(s) => Ok(s.as_str()),
        other => Err(format!(
            "{name}: expected String, got {}",
            other.type_name()
        )),
    }
}

fn arg_u32_param_idx(name: &str, val: &Value) -> Result<u32, String> {
    let i = arg_i64(name, val)?;
    if i < 0 {
        return Err(format!("{name}: param index must be non-negative, got {i}"));
    }
    if i > u32::MAX as i64 {
        return Err(format!("{name}: param index {i} exceeds u32::MAX"));
    }
    Ok(i as u32)
}

/// Decode a 2-D Tensor of shape `[n_dims, n_bins-1]` into the flat row-major
/// `Vec<f64>` plus the inferred dimensions. Used by `abng_set_codebook`.
fn arg_codebook_tensor(name: &str, val: &Value) -> Result<(usize, u16, Vec<f64>), String> {
    let t = arg_tensor(name, val)?;
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(format!(
            "{name}: expected 2-D Tensor [n_dims, n_bins-1], got shape {shape:?}"
        ));
    }
    let n_dims = shape[0];
    let per_dim = shape[1];
    if per_dim == 0 || per_dim >= 256 {
        return Err(format!(
            "{name}: per-dim boundary count {per_dim} must be in [1, 255]"
        ));
    }
    let n_bins = (per_dim + 1) as u16;
    Ok((n_dims, n_bins, t.to_vec()))
}

// ─── Return-value helpers ─────────────────────────────────────────────────

fn graph_err_to_string(name: &str, err: GraphError) -> String {
    format!("{name}: {err}")
}

fn hex_string(hash: &[u8; 32]) -> String {
    cjc_snap::hash::hex_string(hash)
}

fn bytes_value(bytes: Vec<u8>) -> Value {
    Value::Bytes(Rc::new(RefCell::new(bytes)))
}

// ─── Dispatch ────────────────────────────────────────────────────────────

/// Dispatch table for language-level `abng_*` builtins.
///
/// Returns `Ok(None)` if `name` is not one of ours, so callers can fall
/// through to other dispatch tables.
pub fn dispatch_abng(name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    let result: Value = match name {
        // ── Construction / lifecycle ──────────────────────────────
        "abng_new" => {
            arg_count(name, args, 1)?;
            let seed = arg_i64(name, &args[0])?;
            let graph = AdaptiveBeliefGraph::new(seed as u64);
            let id = with_arena(|a| {
                let id = a.next_id;
                a.next_id = a.next_id.wrapping_add(1);
                a.graphs.insert(id, graph);
                id
            });
            Value::Int(id)
        }
        "abng_drop" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_arena(|a| {
                a.graphs.remove(&id);
            });
            Value::Void
        }

        // ── Inspection ────────────────────────────────────────────
        "abng_root" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            // Phase 0.1: root is always node 0; surface as a builtin so
            // user code doesn't have to hardcode that fact (and so the
            // Phase 0.2+ multi-root story has a built-in to extend).
            with_graph(name, id, |g| {
                debug_assert!(g.node_count() >= 1);
                Value::Int(0)
            })?
        }
        "abng_node_count" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| Value::Int(g.node_count() as i64))?
        }
        "abng_audit_len" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| Value::Int(g.audit_len() as i64))?
        }
        "abng_chain_head" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| {
                Value::String(Rc::new(hex_string(&g.chain_head)))
            })?
        }
        "abng_node_stats" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                if node_id >= g.node_count() {
                    return Err(graph_err_to_string(
                        name,
                        GraphError::NodeOutOfRange {
                            node_id,
                            n_nodes: g.node_count(),
                        },
                    ));
                }
                let node = &g.nodes[node_id as usize];
                let n = node.stats.n_seen as f64;
                let mean = node.stats.mean;
                let var = node.stats.variance();
                let t = Tensor::from_vec(vec![n, mean, var], &[3])
                    .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
                Ok(Value::Tensor(t))
            })??
        }
        "abng_node_stats_version" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                if node_id >= g.node_count() {
                    return Err(graph_err_to_string(
                        name,
                        GraphError::NodeOutOfRange {
                            node_id,
                            n_nodes: g.node_count(),
                        },
                    ));
                }
                Ok(Value::Int(g.nodes[node_id as usize].stats_version as i64))
            })??
        }

        // ── Mutation ──────────────────────────────────────────────
        "abng_observe" => {
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let value = arg_f64(name, &args[2])?;
            with_graph(name, id, |g| {
                g.observe(node_id, value)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_observe_batch" => {
            // Phase 0.6 Item 4 (v13) — emits ONE `BeliefUpdateBatch`
            // audit event covering all N values, instead of N per-row
            // `BeliefUpdate` events. Post-batch
            // `NodeStats::canonical_bytes` is bit-identical to the
            // per-row equivalent (Welford folds in row order with
            // Kahan compensation); the chain head WILL differ (one
            // chain advance vs N). For the legacy "loop observe"
            // semantics, use `abng_observe_slice`.
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let values = arg_tensor(name, &args[2])?.to_vec();
            with_graph(name, id, |g| {
                g.observe_batch(node_id, &values)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_observe_slice" => {
            // Phase 0.6 Item 4 — explicit per-row loop semantics
            // (N `BeliefUpdate` events, N chain advances). Provided
            // as a stable opt-in for callers that need the legacy
            // chain history (e.g. parity comparisons against pre-v13
            // logs).
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let values = arg_tensor(name, &args[2])?.to_vec();
            with_graph(name, id, |g| {
                g.observe_slice(node_id, &values)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }

        // ── Audit / replay ────────────────────────────────────────
        "abng_verify_chain" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| Value::Bool(g.verify_chain().is_ok()))?
        }
        "abng_serialize" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            let blob = with_graph(name, id, |g| serialize::serialize(g))?;
            bytes_value(blob)
        }
        "abng_replay" => {
            arg_count(name, args, 1)?;
            let blob = arg_bytes(name, &args[0])?;
            let graph = serialize::replay(&blob).map_err(|e| format!("{name}: {e}"))?;
            let id = with_arena(|a| {
                let id = a.next_id;
                a.next_id = a.next_id.wrapping_add(1);
                a.graphs.insert(id, graph);
                id
            });
            Value::Int(id)
        }

        // ── Phase 0.2: structural mutation ────────────────────────
        "abng_add_node" => {
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let parent = arg_u32_node(name, &args[1])?;
            let key_byte_i = arg_i64(name, &args[2])?;
            if !(0..=255).contains(&key_byte_i) {
                return Err(format!(
                    "{name}: key_byte must be in [0, 255], got {key_byte_i}"
                ));
            }
            let key_byte = key_byte_i as u8;
            let new_id = with_graph(name, id, |g| {
                g.add_node(parent, key_byte)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Int(new_id as i64)
        }

        // ── Phase 0.2: structural inspection ──────────────────────
        "abng_node_parent" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                if node_id >= g.node_count() {
                    return Err(graph_err_to_string(
                        name,
                        GraphError::NodeOutOfRange {
                            node_id,
                            n_nodes: g.node_count(),
                        },
                    ));
                }
                let parent = g.nodes[node_id as usize].parent;
                Ok(Value::Int(match parent {
                    Some(p) => p as i64,
                    None => -1,
                }))
            })??
        }
        "abng_node_kind" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.node_kind(node_id)
                    .map(|k| Value::Int(k as u8 as i64))
                    .map_err(|e| graph_err_to_string(name, e))
            })??
        }
        "abng_node_child_count" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                if node_id >= g.node_count() {
                    return Err(graph_err_to_string(
                        name,
                        GraphError::NodeOutOfRange {
                            node_id,
                            n_nodes: g.node_count(),
                        },
                    ));
                }
                Ok(Value::Int(g.nodes[node_id as usize].children.len() as i64))
            })??
        }
        "abng_node_child" => {
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let key_byte_i = arg_i64(name, &args[2])?;
            if !(0..=255).contains(&key_byte_i) {
                return Err(format!(
                    "{name}: key_byte must be in [0, 255], got {key_byte_i}"
                ));
            }
            let key_byte = key_byte_i as u8;
            with_graph(name, id, |g| {
                if node_id >= g.node_count() {
                    return Err(graph_err_to_string(
                        name,
                        GraphError::NodeOutOfRange {
                            node_id,
                            n_nodes: g.node_count(),
                        },
                    ));
                }
                let child = g.nodes[node_id as usize].children.get(key_byte);
                Ok(Value::Int(match child {
                    Some(id) => id as i64,
                    None => -1,
                }))
            })??
        }

        // ── Phase 0.2: codebook ───────────────────────────────────
        "abng_set_codebook" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let (n_dims, n_bins, flat) = arg_codebook_tensor(name, &args[1])?;
            with_graph(name, id, |g| {
                g.set_codebook(n_dims, n_bins, &flat)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_codebook_dims" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| match &g.codebook {
                Some(cb) => Value::Int(cb.n_dims as i64),
                None => Value::Int(0),
            })?
        }
        "abng_codebook_hash" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| {
                let s = match &g.codebook {
                    Some(cb) => hex_string(&cb.frozen_hash),
                    None => String::new(),
                };
                Value::String(Rc::new(s))
            })?
        }
        "abng_encode_prefix" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let x = arg_tensor_1d_f64(name, &args[1])?;
            let bytes = with_graph(name, id, |g| {
                g.encode_prefix(&x).map_err(|e| graph_err_to_string(name, e))
            })??;
            // Return as a 1-D Tensor of f64 bin indices.
            let data: Vec<f64> = bytes.iter().map(|&b| b as f64).collect();
            let len = data.len();
            let t = Tensor::from_vec(data, &[len])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }

        // ── Phase 0.2: routing ───────────────────────────────────
        "abng_descend" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let prefix = arg_prefix_bytes(name, &args[1])?;
            let evidence = with_graph(name, id, |g| g.descend(&prefix))?;
            // Return [matched_prefix, leaf_id] as a 2-element f64 Tensor.
            let t = Tensor::from_vec(
                vec![evidence.matched_prefix as f64, evidence.leaf_id as f64],
                &[2],
            )
            .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_descend_traced" => {
            // Phase 0.4 Track A — descend + emit one Routed audit event
            // (tag 0x1B) per call. Same return shape as abng_descend.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let prefix = arg_prefix_bytes(name, &args[1])?;
            let evidence = with_graph(name, id, |g| g.descend_traced(&prefix))?;
            let t = Tensor::from_vec(
                vec![evidence.matched_prefix as f64, evidence.leaf_id as f64],
                &[2],
            )
            .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_route_to_leaf" => {
            // Phase 0.6 Item 7 — fused native kernel for the
            // `encode_prefix → descend → extract_leaf` pattern that
            // shows up in every per-row training inner loop. The
            // existing 3-builtin sequence pays an interpreter
            // dispatch + Tensor allocation + extraction cost for
            // each row. This fused builtin runs the same Rust path
            // in one dispatch.
            //
            // Bit-equivalent to:
            //   let prefix = abng_encode_prefix(g, x);
            //   let evidence = abng_descend(g, prefix);
            //   let leaf = int(evidence.get([1]));
            //
            // Returns the leaf node id directly as `Value::Int(i64)`.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let x = arg_tensor_1d_f64(name, &args[1])?;
            let leaf = with_graph(name, id, |g| {
                let bytes = g
                    .encode_prefix(&x)
                    .map_err(|e| graph_err_to_string(name, e))?;
                Ok::<i64, String>(g.descend(&bytes).leaf_id as i64)
            })??;
            Value::Int(leaf)
        }
        "abng_route_to_leaf_batch" => {
            // Phase 0.6 Item 8 — TidyView-discipline "chunked
            // dispatch" applied to ABNG. Takes a 2-D `[n, d]` input
            // tensor and returns a 1-D `[n]` tensor of leaf ids.
            // Bit-equivalent to N calls of `abng_route_to_leaf` over
            // each row, but pays ONE interpreter dispatch + ONE
            // output allocation across the whole batch.
            //
            // The win comes from amortizing per-call overhead — see
            // `Graph::route_to_leaf_batch` for the implementation
            // details.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let xs_t = arg_tensor(name, &args[1])?;
            if xs_t.shape().len() != 2 {
                return Err(format!(
                    "{name}: input must be 2-D [n, d], got shape {:?}",
                    xs_t.shape()
                ));
            }
            let n = xs_t.shape()[0];
            let xs = xs_t.to_vec();
            let leaf_ids = with_graph(name, id, |g| {
                g.route_to_leaf_batch(&xs, n)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            // Wrap as a 1-D Tensor of f64 (the canonical numeric
            // tensor type — same convention as abng_descend's
            // [matched, leaf_id] return).
            let data: Vec<f64> = leaf_ids.iter().map(|&l| l as f64).collect();
            let len = data.len();
            let t = Tensor::from_vec(data, &[len])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_predict_snap" => {
            // Phase 0.4 Track A — pack a predict + lineage tuple into a
            // self-contained Bytes blob using the dedicated PRED_MAGIC
            // format in `cjc_abng::predict_snap`. Drives `cjcl abng
            // explain`.
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let phi = arg_tensor_1d_f64(name, &args[2])?;
            let blob = with_graph(name, id, |g| {
                crate::predict_snap::pack(g, node_id, &phi)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            bytes_value(blob)
        }
        "abng_compact_log" => {
            // Phase 0.4 Track A — emit one StatsSnapshot audit event
            // (tag 0x1A) per distinct node touched in [0, until_seq).
            // Returns the count emitted.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let until_seq = arg_i64(name, &args[1])?;
            if until_seq < 0 {
                return Err(format!(
                    "{name}: until_seq must be non-negative, got {until_seq}"
                ));
            }
            let emitted = with_graph(name, id, |g| {
                g.compact_log(until_seq as u64)
            })?;
            Value::Int(emitted as i64)
        }
        "abng_route_path" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let prefix = arg_prefix_bytes(name, &args[1])?;
            let evidence = with_graph(name, id, |g| g.descend(&prefix))?;
            let data: Vec<f64> = evidence.path.iter().map(|&n| n as f64).collect();
            let len = data.len();
            let t = Tensor::from_vec(data, &[len])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        // ── Phase 0.3a: per-node MLP head ─────────────────────────
        "abng_set_leaf_head" => {
            arg_count(name, args, 5)?;
            let id = arg_i64(name, &args[0])?;
            let input_dim_i = arg_i64(name, &args[1])?;
            if !(1..=u32::MAX as i64).contains(&input_dim_i) {
                return Err(format!(
                    "{name}: input_dim must be a positive int, got {input_dim_i}"
                ));
            }
            let hidden_dims = arg_dims_tensor(name, &args[2])?;
            let output_dim_i = arg_i64(name, &args[3])?;
            if !(1..=u32::MAX as i64).contains(&output_dim_i) {
                return Err(format!(
                    "{name}: output_dim must be a positive int, got {output_dim_i}"
                ));
            }
            let act_str = arg_string(name, &args[4])?;
            let act = parse_activation(act_str).map_err(|e| format!("{name}: {e}"))?;
            with_graph(name, id, |g| {
                g.set_leaf_head(
                    input_dim_i as u32,
                    hidden_dims.clone(),
                    output_dim_i as u32,
                    act,
                )
                .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_leaf_head_dims" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            let v: Vec<f64> = with_graph(name, id, |g| match &g.head {
                Some(head) => {
                    let mut v = Vec::with_capacity(3 + head.hidden_dims.len());
                    v.push(head.input_dim as f64);
                    v.push(head.hidden_dims.len() as f64);
                    for &h in &head.hidden_dims {
                        v.push(h as f64);
                    }
                    v.push(head.output_dim as f64);
                    v.push(encode_activation_tag(head.activation) as f64);
                    v
                }
                None => Vec::new(),
            })?;
            let len = v.len();
            let t = Tensor::from_vec(v, &[len])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_leaf_param_count" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.leaf_param_count(node_id)
                    .map(|n| Value::Int(n as i64))
                    .map_err(|e| graph_err_to_string(name, e))
            })??
        }
        "abng_leaf_param" => {
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let k = arg_u32_param_idx(name, &args[2])?;
            with_graph(name, id, |g| {
                g.leaf_param(node_id, k)
                    .map(Value::Tensor)
                    .map_err(|e| graph_err_to_string(name, e))
            })??
        }
        "abng_leaf_set_param" => {
            arg_count(name, args, 4)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let k = arg_u32_param_idx(name, &args[2])?;
            let t = arg_tensor(name, &args[3])?.clone();
            with_graph(name, id, |g| {
                g.leaf_set_param(node_id, k, t)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_leaf_set_params_batch" => {
            // abng_leaf_set_params_batch(graph_id, node_id, params: Tensor[])
            //   → Void
            // Phase 0.4 Track C-2.3.6 — single audit event for a whole-
            // vector param writeback (collapses 2(L+1) LeafParamsUpdated
            // events into one LeafParamsUpdatedBatch).
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let params = match &args[2] {
                Value::Array(arr) => {
                    let mut out: Vec<Tensor> = Vec::with_capacity(arr.len());
                    for (i, v) in arr.iter().enumerate() {
                        match v {
                            Value::Tensor(t) => out.push(t.clone()),
                            other => {
                                return Err(format!(
                                    "{name}: params[{i}] must be Tensor, got {}",
                                    other.type_name()
                                ));
                            }
                        }
                    }
                    out
                }
                other => {
                    return Err(format!(
                        "{name}: expected Array of Tensor for params, got {}",
                        other.type_name()
                    ));
                }
            };
            with_graph(name, id, |g| {
                g.leaf_set_params_batch(node_id, params)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_leaf_params_hash" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let h = with_graph(name, id, |g| {
                g.leaf_params_hash(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::String(Rc::new(hex_string(&h)))
        }
        "abng_leaf_forward" => {
            // abng_leaf_forward(graph_id, node_id, x_grad_idx) -> Array[Int]
            // Returns [y_idx, p_0, p_1, ..., p_{n-1}] where y_idx is the
            // output GradGraph node and p_* are the param indices for
            // backward_collect.
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let x_idx_i = arg_i64(name, &args[2])?;
            if x_idx_i < 0 {
                return Err(format!(
                    "{name}: x_grad_idx must be a non-negative GradGraph index, got {x_idx_i}"
                ));
            }
            let x_idx = x_idx_i as usize;
            let (y_idx, params) = with_graph(name, id, |g| {
                g.leaf_forward(node_id, x_idx)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            let mut out: Vec<Value> = Vec::with_capacity(1 + params.len());
            out.push(Value::Int(y_idx as i64));
            for p in params {
                out.push(Value::Int(p as i64));
            }
            Value::Array(Rc::new(out))
        }

        // ── Phase 0.3b: Bayesian linear regression head ───────────
        "abng_set_blr_prior" => {
            arg_count(name, args, 4)?;
            let id = arg_i64(name, &args[0])?;
            let precision = arg_f64(name, &args[1])?;
            let a = arg_f64(name, &args[2])?;
            let b = arg_f64(name, &args[3])?;
            with_graph(name, id, |g| {
                g.set_blr_prior(precision, a, b)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_blr_features" => {
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let x_idx_i = arg_i64(name, &args[2])?;
            if x_idx_i < 0 {
                return Err(format!(
                    "{name}: x_grad_idx must be non-negative, got {x_idx_i}"
                ));
            }
            let phi_idx = with_graph(name, id, |g| {
                g.blr_features(node_id, x_idx_i as usize)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Int(phi_idx as i64)
        }
        "abng_blr_update" => {
            // abng_blr_update(graph_id, node_id, features_2d, y_1d) -> Void
            arg_count(name, args, 4)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let features_t = arg_tensor(name, &args[2])?;
            let y_t = arg_tensor(name, &args[3])?;
            if features_t.shape().len() != 2 {
                return Err(format!(
                    "{name}: features must be 2-D [n, d], got shape {:?}",
                    features_t.shape()
                ));
            }
            if y_t.shape().len() != 1 {
                return Err(format!(
                    "{name}: y must be 1-D [n], got shape {:?}",
                    y_t.shape()
                ));
            }
            let features = features_t.to_vec();
            let y = y_t.to_vec();
            with_graph(name, id, |g| {
                g.blr_update(node_id, &features, &y)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_blr_predict" => {
            // abng_blr_predict(graph_id, node_id, phi_1d) -> Tensor[3]
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let phi = arg_tensor_1d_f64(name, &args[2])?;
            let (mean, epi, ale) = with_graph(name, id, |g| {
                g.blr_predict(node_id, &phi)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            let t = Tensor::from_vec(vec![mean, epi, ale], &[3])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_reset_blr" => {
            // Phase 0.4 Track C-2.3.5 — reset BLR posterior to prior +
            // refresh feature_version_hash to current MLP params hash.
            // Recovers from `BlrError::FeatureVersionStale`.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.reset_blr(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_blr_predict_with_fallback" => {
            // Phase 0.4 Track C-2.3.8 — predict at the nearest ancestor
            // (incl. self) with n_seen >= 1; walks the parent chain.
            // Returns Tensor[4] = [mean, epistemic_leverage,
            // aleatoric_var, source_node_id_as_f64]. The source node id
            // is encoded as f64 (NodeId is u32, so the cast is exact for
            // every node id < 2^53). Errors with
            // BlrError::NoEvidence { walked } if no ancestor has any
            // observations.
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let phi = arg_tensor_1d_f64(name, &args[2])?;
            let (mean, lev, ale, source) = with_graph(name, id, |g| {
                g.blr_predict_with_fallback(node_id, &phi)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            let t = Tensor::from_vec(vec![mean, lev, ale, source as f64], &[4])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_blr_state_hash" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let h = with_graph(name, id, |g| {
                g.blr_state_hash(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::String(Rc::new(hex_string(&h)))
        }
        "abng_blr_n_seen" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let n = with_graph(name, id, |g| {
                g.blr_n_seen(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Int(n as i64)
        }

        // ── Phase 0.3c: density tracker ───────────────────────────
        "abng_set_density_tracker" => {
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            with_graph(name, id, |g| {
                g.set_density_tracker()
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_density_observe" => {
            // (graph_id, node_id, features_2d) -> Void
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let features_t = arg_tensor(name, &args[2])?;
            if features_t.shape().len() != 2 {
                return Err(format!(
                    "{name}: features must be 2-D [n, d], got shape {:?}",
                    features_t.shape()
                ));
            }
            let features = features_t.to_vec();
            with_graph(name, id, |g| {
                g.density_observe(node_id, &features)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_density_score" => {
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let phi = arg_tensor_1d_f64(name, &args[2])?;
            let s = with_graph(name, id, |g| {
                g.density_score(node_id, &phi)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Float(s)
        }
        "abng_density_n_seen" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let n = with_graph(name, id, |g| {
                g.density_n_seen(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Int(n as i64)
        }

        // ── Phase 0.3c: calibration bins ──────────────────────────
        "abng_set_calibration" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let n_bins_i = arg_i64(name, &args[1])?;
            if !(2..=100).contains(&n_bins_i) {
                return Err(format!("{name}: n_bins must be in [2, 100], got {n_bins_i}"));
            }
            with_graph(name, id, |g| {
                g.set_calibration(n_bins_i as u8)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_calibration_observe" => {
            // (graph_id, node_id, predicted_prob: f64, was_correct: bool) -> Void
            arg_count(name, args, 4)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let p = arg_f64(name, &args[2])?;
            let was_correct = match &args[3] {
                Value::Bool(b) => *b,
                Value::Int(i) => *i != 0,
                other => {
                    return Err(format!(
                        "{name}: was_correct must be Bool, got {}",
                        other.type_name()
                    ));
                }
            };
            with_graph(name, id, |g| {
                g.calibration_observe(node_id, p, was_correct)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_calibration_ece" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let ece = with_graph(name, id, |g| {
                g.calibration_ece(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Float(ece)
        }
        "abng_calibration_n_seen" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let n = with_graph(name, id, |g| {
                g.calibration_n_seen(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Int(n as i64)
        }

        // ── Phase 0.3c: drift detector ────────────────────────────
        "abng_freeze_drift_baseline" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.freeze_drift_baseline(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_drift_score" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let s = with_graph(name, id, |g| {
                g.drift_score(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Float(s)
        }

        // ── Phase 0.3c: composite OOD score ───────────────────────
        "abng_ood_score" => {
            // (graph_id, node_id, phi_1d, matched_prefix: i64, prefix_max: i64) -> Float
            arg_count(name, args, 5)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let phi = arg_tensor_1d_f64(name, &args[2])?;
            let matched_prefix_i = arg_i64(name, &args[3])?;
            let prefix_max_i = arg_i64(name, &args[4])?;
            if !(0..=255).contains(&matched_prefix_i) {
                return Err(format!(
                    "{name}: matched_prefix must be in [0, 255], got {matched_prefix_i}"
                ));
            }
            if !(0..=255).contains(&prefix_max_i) {
                return Err(format!(
                    "{name}: prefix_max must be in [0, 255], got {prefix_max_i}"
                ));
            }
            let s = with_graph(name, id, |g| {
                g.ood_score(
                    node_id,
                    &phi,
                    matched_prefix_i as u8,
                    prefix_max_i as u8,
                )
                .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Float(s)
        }

        "abng_node_stats_chain_head" => {
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                if node_id >= g.node_count() {
                    return Err(graph_err_to_string(
                        name,
                        GraphError::NodeOutOfRange {
                            node_id,
                            n_nodes: g.node_count(),
                        },
                    ));
                }
                Ok(Value::String(Rc::new(hex_string(
                    &g.nodes[node_id as usize].stats_chain_head,
                ))))
            })??
        }

        // ── Phase 0.3d-4: decision engine + unfreeze ─────────────
        "abng_decide_step" => {
            // (graph_id) -> Tensor[6] — counts per ActionKind in
            //   index order [Grow, Split, Merge, Prune, Compress, Freeze].
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            let counts = with_graph(name, id, |g| g.decide_step())?;
            let data: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
            let t = Tensor::from_vec(data, &[crate::graph::N_ACTION_KINDS])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_unfreeze" => {
            // (graph_id, node_id) -> Void
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.unfreeze(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }

        // ── Phase 0.3d-3: decision policy + structural mutations ─
        "abng_set_decision_policy" => {
            // (graph_id, thresholds: Tensor[11]) -> Void
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let thresholds = arg_tensor_1d_f64(name, &args[1])?;
            with_graph(name, id, |g| {
                g.set_decision_policy(&thresholds)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_decision_policy_hash" => {
            // (graph_id) -> String — empty when no policy is installed.
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            let hex = with_graph(name, id, |g| {
                g.decision_policy_hash()
                    .map(|h| hex_string(&h))
                    .unwrap_or_default()
            })?;
            Value::String(Rc::new(hex))
        }
        "abng_force_grow" => {
            // (graph_id, parent, key_byte: i64) -> Int (new child node id)
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let parent = arg_u32_node(name, &args[1])?;
            let key_byte_i = arg_i64(name, &args[2])?;
            if !(0..=255).contains(&key_byte_i) {
                return Err(format!(
                    "{name}: key_byte must be in [0, 255], got {key_byte_i}"
                ));
            }
            let child = with_graph(name, id, |g| {
                g.force_grow(parent, key_byte_i as u8)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Int(child as i64)
        }
        "abng_force_split" => {
            // (graph_id, parent) -> Tensor[2]: [child_a, child_b]
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let parent = arg_u32_node(name, &args[1])?;
            let (a, b) = with_graph(name, id, |g| {
                g.force_split(parent)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            let t = Tensor::from_vec(vec![a as f64, b as f64], &[2])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_force_merge" => {
            // (graph_id, absorbed, into) -> Void
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let absorbed = arg_u32_node(name, &args[1])?;
            let into = arg_u32_node(name, &args[2])?;
            with_graph(name, id, |g| {
                g.force_merge(absorbed, into)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_force_prune" => {
            // (graph_id, node_id) -> Void
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.force_prune(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_force_compress" => {
            // (graph_id, node_id) -> Void
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.force_compress(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_force_freeze" => {
            // (graph_id, node_id) -> Void
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            with_graph(name, id, |g| {
                g.force_freeze(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_is_frozen" => {
            // (graph_id, node_id) -> Bool
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let b = with_graph(name, id, |g| {
                g.is_frozen(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Bool(b)
        }
        "abng_action_count" => {
            // (graph_id, kind: i64 in 0..=5) -> Int
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let kind_i = arg_i64(name, &args[1])?;
            if !(0..=(crate::graph::N_ACTION_KINDS as i64 - 1)).contains(&kind_i) {
                return Err(format!(
                    "{name}: kind index {kind_i} out of range [0, {}]",
                    crate::graph::N_ACTION_KINDS - 1
                ));
            }
            let kind = crate::graph::ActionKind::from_index(kind_i as u8)
                .ok_or_else(|| format!("{name}: unknown action kind {kind_i}"))?;
            let count = with_graph(name, id, |g| g.action_count(kind))?;
            Value::Int(count as i64)
        }
        "abng_unfreeze_count" => {
            // Phase 0.4-extended (v11) — observability counter for
            // Unfreeze events (manual + drift-trip auto-Unfreeze).
            // Distinct from action_counts to keep the 6-element
            // ActionKind indexing convention valid.
            arg_count(name, args, 1)?;
            let id = arg_i64(name, &args[0])?;
            let count = with_graph(name, id, |g| g.unfreeze_count)?;
            Value::Int(count as i64)
        }
        "abng_stamp_provenance" => {
            // Phase 0.5 Item 1 — (graph_id, node_id, hex_hash: String) -> Void.
            // hex_hash MUST be exactly 64 lowercase hex chars (the
            // ergonomic .cjcl-side encoding of a [u8; 32]). Idempotent:
            // re-stamping with the same hash is a no-op.
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let hex = arg_string(name, &args[2])?;
            if hex.len() != 64 {
                return Err(format!(
                    "{name}: expected 64-char hex hash, got {} chars",
                    hex.len()
                ));
            }
            let mut hash = [0u8; 32];
            for i in 0..32 {
                let lo = hex.as_bytes()[2 * i];
                let hi = hex.as_bytes()[2 * i + 1];
                let nybble = |b: u8| -> Result<u8, String> {
                    match b {
                        b'0'..=b'9' => Ok(b - b'0'),
                        b'a'..=b'f' => Ok(b - b'a' + 10),
                        b'A'..=b'F' => Ok(b - b'A' + 10),
                        other => Err(format!(
                            "{name}: hex hash contains non-hex byte {other:#04x}"
                        )),
                    }
                };
                hash[i] = (nybble(lo)? << 4) | nybble(hi)?;
            }
            with_graph(name, id, |g| {
                g.stamp_provenance(node_id, hash)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_provenance_stamp" => {
            // Phase 0.5 Item 1 — (graph_id, node_id) -> String (64-char
            // lowercase hex). Returns the all-zero hash for unstamped
            // nodes. Useful for explain / verify flows from .cjcl.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let hash = with_graph(name, id, |g| {
                if (node_id as usize) >= g.nodes.len() {
                    return Err(format!(
                        "{name}: node_id {node_id} out of range (n_nodes = {})",
                        g.nodes.len()
                    ));
                }
                Ok(g.nodes[node_id as usize].provenance_stamp_hash)
            })??;
            let mut s = String::with_capacity(64);
            for b in hash.iter() {
                s.push_str(&format!("{:02x}", b));
            }
            Value::String(s.into())
        }

        // ── Phase 0.3d-2: per-node expected_epistemic capture ────
        "abng_set_expected_epistemic" => {
            // (graph_id, node_id, value: f64) -> Void
            arg_count(name, args, 3)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let value = arg_f64(name, &args[2])?;
            with_graph(name, id, |g| {
                g.set_expected_epistemic(node_id, value)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Void
        }
        "abng_expected_epistemic" => {
            // (graph_id, node_id) -> Float. Returns the captured value
            // when present, else `-1.0` as the "not captured" sentinel
            // (since `set_expected_epistemic` requires `value > 0`,
            // `-1.0` cannot collide with a real reference).
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let v = with_graph(name, id, |g| {
                g.expected_epistemic(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Float(v.unwrap_or(-1.0))
        }
        "abng_force_recapture_expected_epistemic" => {
            // Phase 0.4 Track C-2.3.12 — overwrite the captured
            // expected_epistemic with a fresh deterministic capture
            // from the current BLR posterior. Returns the new value.
            // Use after `abng_reset_blr` to keep `ood_score`'s
            // calibrated ratio aligned with the post-reset posterior
            // shape.
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let value = with_graph(name, id, |g| {
                g.force_recapture_expected_epistemic(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            Value::Float(value)
        }

        // ── Phase 0.3d-1: maturity + signature (lazy / read-only) ─
        "abng_node_maturity" => {
            // (graph_id, node_id) -> Tensor[4]
            //   [samples_seen f64, calibration_stable f64, uncertainty_stable f64, trust_level f64]
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let m = with_graph(name, id, |g| {
                g.node_maturity(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            let data = vec![
                m.samples_seen as f64,
                m.calibration_stable as u8 as f64,
                m.uncertainty_stable as u8 as f64,
                m.trust_level as f64,
            ];
            let t = Tensor::from_vec(data, &[4])
                .map_err(|e| format!("{name}: tensor build failed: {e:?}"))?;
            Value::Tensor(t)
        }
        "abng_node_signature" => {
            // (graph_id, node_id) -> Bytes[32]
            arg_count(name, args, 2)?;
            let id = arg_i64(name, &args[0])?;
            let node_id = arg_u32_node(name, &args[1])?;
            let sig = with_graph(name, id, |g| {
                g.node_signature(node_id)
                    .map_err(|e| graph_err_to_string(name, e))
            })??;
            bytes_value(sig.canonical_bytes().to_vec())
        }

        // Not one of ours — let the caller fall through.
        _ => return Ok(None),
    };

    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &[Value]) -> Value {
        dispatch_abng(name, args).unwrap().unwrap()
    }

    fn call_err(name: &str, args: &[Value]) -> String {
        dispatch_abng(name, args).unwrap_err()
    }

    #[test]
    fn unknown_name_falls_through() {
        let r = dispatch_abng("not_an_abng_builtin", &[]).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn new_drop_lifecycle() {
        reset_arena();
        let id = match call("abng_new", &[Value::Int(7)]) {
            Value::Int(i) => i,
            _ => panic!(),
        };
        // Drop is a no-op if id absent.
        let _ = call("abng_drop", &[Value::Int(id)]);
        // After drop, lookup fails.
        let err = call_err("abng_chain_head", &[Value::Int(id)]);
        assert!(err.contains("no graph"));
    }

    #[test]
    fn observe_then_stats() {
        reset_arena();
        let id = match call("abng_new", &[Value::Int(0)]) {
            Value::Int(i) => i,
            _ => panic!(),
        };
        let _ = call("abng_observe", &[Value::Int(id), Value::Int(0), Value::Float(1.0)]);
        let _ = call("abng_observe", &[Value::Int(id), Value::Int(0), Value::Float(3.0)]);
        let stats = match call("abng_node_stats", &[Value::Int(id), Value::Int(0)]) {
            Value::Tensor(t) => t.to_vec(),
            _ => panic!(),
        };
        assert_eq!(stats[0], 2.0);
        assert_eq!(stats[1], 2.0);
        assert_eq!(stats[2], 2.0);
    }

    #[test]
    fn serialize_replay_round_trip() {
        reset_arena();
        let id = match call("abng_new", &[Value::Int(11)]) {
            Value::Int(i) => i,
            _ => panic!(),
        };
        for v in [0.5, 1.5, 2.5, 3.5] {
            let _ = call(
                "abng_observe",
                &[Value::Int(id), Value::Int(0), Value::Float(v)],
            );
        }
        let head1 = match call("abng_chain_head", &[Value::Int(id)]) {
            Value::String(s) => (*s).clone(),
            _ => panic!(),
        };
        let blob = match call("abng_serialize", &[Value::Int(id)]) {
            Value::Bytes(b) => b.borrow().clone(),
            _ => panic!(),
        };
        let id2 = match call(
            "abng_replay",
            &[Value::Bytes(Rc::new(RefCell::new(blob)))],
        ) {
            Value::Int(i) => i,
            _ => panic!(),
        };
        let head2 = match call("abng_chain_head", &[Value::Int(id2)]) {
            Value::String(s) => (*s).clone(),
            _ => panic!(),
        };
        assert_eq!(head1, head2);
    }
}
