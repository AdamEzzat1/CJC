---
title: "ADR-0016: Language-Level GradGraph Primitives"
tags: [adr, ml, autodiff, dispatch]
status: Accepted
date: 2026-04-26
---

# ADR-0016: Language-Level GradGraph Primitives

## Status

Accepted

## Context

Before Phase 3c, every PINN demo in CJC-Lang was a thin script over the
baked-in Rust trainer (`pinn_heat_1d_nn_train`, `pinn_burgers_1d_nn_train`,
…). The user-visible language ratio was roughly **5% CJC-Lang / 95% Rust**:
the script picked hyperparameters and compared metrics, but the entire
forward pass, residual, backward, and Adam loop ran inside Rust. This made
CJC-Lang look like a configuration shell for a hidden ML engine, not a
language you could *write* models in.

The existing `cjc-ad::GradGraph` arena (Phase 2.4 → v2.5) is already a
deterministic first-order reverse-mode autodiff engine with 34 `GradOp`
variants, parameter/input nodes, and a flat-array tape. It just wasn't
addressable from `.cjcl` source.

## Decision

### 1. Expose `grad_graph_*` builtins via a satellite dispatch crate

We add a new dispatch surface — `cjc_ad::dispatch_grad_graph(name, args)` —
routed from both executors *after* `cjc_runtime::dispatch_builtin` and
`cjc_quantum::dispatch_quantum`. The satellite-dispatch pattern was already
established for `cjc-quantum` (ADR-0015 implicitly) and is the only viable
placement: `cjc-runtime` cannot depend on `cjc-ad` (cycle), but both
executors do.

24 builtins ship in this phase (rounded up from the brief's "~20"):

- **Construction:** `grad_graph_new`, `grad_graph_param`, `grad_graph_input`, `grad_graph_const`
- **Pointwise:** `grad_graph_add`, `grad_graph_sub`, `grad_graph_mul`, `grad_graph_div`, `grad_graph_neg`, `grad_graph_scalar_mul`, `grad_graph_pow`, `grad_graph_exp`, `grad_graph_ln`, `grad_graph_sqrt`, `grad_graph_sin`, `grad_graph_cos`, `grad_graph_tanh`
- **Reductions / matmul:** `grad_graph_sum`, `grad_graph_mean`, `grad_graph_matmul`
- **Fused:** `grad_graph_mlp_layer(input, weight, bias, activation_str)`
- **State / backward:** `grad_graph_forward`, `grad_graph_set_tensor`, `grad_graph_param_grad`, `grad_graph_zero_grad`, `grad_graph_backward`, `grad_graph_clip_grad_norm`, `grad_graph_len`

### 2. Handle representation: ambient thread-local graph (Option B)

Node indices cross the language boundary as `Value::Int(node_idx)`. The
graph itself lives in a `thread_local!` `RefCell<GradGraph>` reset by
`grad_graph_new()`. This avoids:

- A new `Value::GradNode` variant (would change `Value` enum layout — HARD
  RULE #1 violation)
- A handle-table indirection (extra hashmap, non-deterministic if not
  carefully ordered)

The trade-off: only one graph per thread at a time. For PINN training and
the workloads the brief targets, this is exactly the natural pattern (build
a graph per epoch, train, discard) — the same pattern the Rust trainer
already uses internally.

### 3. Higher-order AD: ship FD fallback, defer to Phase 3d

PINN residuals need `u_t`, `u_xx`, etc. Native higher-order AD requires a
graph-of-graphs rewrite of the existing tape, which is a major change
affecting every `GradOp` backward closure. The brief explicitly authorized
**central finite differences** (ε=1e-3, 3-point stencil) as the Phase 3c
fallback, with the understanding that:

- Truncation error at ε=1e-3 is O(ε²) ≈ 1e-6, well inside PINN smoke
  thresholds
- Bit-equality vs the Rust trainer is therefore *relaxed* to RMSE<1e-6
  rather than enforced byte-equal — and within phase, the AST↔MIR contract
  is still enforced byte-equal (since both backends take the same FD path)
- A future ADR-0017 can introduce native `grad_graph_grad_of` without
  breaking any user code that ships in Phase 3c

### 4. Activation as `&str` at the boundary

`grad_graph_mlp_layer(input, weight, bias, "tanh")` parses the activation
string at dispatch time. Alternatives considered:

- `Value::Activation` enum variant — would change `Value` layout
- Integer encoding — opaque and brittle across versions
- String — self-documenting, version-stable, parsed once per call (not
  in a hot loop)

The activation set matches `cjc_ad::pinn::Activation`'s 9 variants:
`tanh`, `sigmoid`, `relu`, `none`, `gelu`, `silu`, `elu`, `selu`, `sin`.

### 5. Bounds checking at the dispatch boundary

Every node index argument is validated against `graph.len()` before
forwarding to `GradGraph::*`. The bolero structural fuzzer found that
out-of-range indices and non-scalar `backward()` calls would otherwise
panic deep inside the arena; we now `Err` cleanly. Shape mismatches in
the underlying tensor kernels (`add(scalar, [1,3])`) remain a documented
internal invariant — the dispatch layer recovers state cleanly via
`grad_graph_new()` after such panics, but does not pre-validate shapes.

## Consequences

### Positive

- **Language ratio flips.** The flagship demo
  (`examples/physics_ml/pinn_heat_1d_pure.cjcl`, ~200 LOC) builds the
  forward pass, residual, backward, and Adam update entirely in CJC-Lang
  source — Rust now provides only primitives (`adam_step`, tensor ops,
  graph backward), exactly mirroring how Chess RL v2.5 splits.
- **No `Value` enum changes.** MIR register layout, parser, type checker
  all untouched.
- **AST↔MIR byte-equal.** Verified across 27 wiring tests + the full
  flagship demo (50 epochs of training, every reported loss bit-identical
  across executors).
- **Fuzz-hardened dispatch.** 2 bolero targets (structural + numerical),
  >256 cases each. Out-of-range indices, non-scalar backward, and
  shape-mismatched binary ops all surface as recoverable errors rather
  than panics.
- **Future-proof.** When Phase 3d adds native higher-order AD, the same
  `grad_graph_*` surface gains `grad_graph_grad_of(node, wrt)` without
  breaking existing code.

### Negative

- One ambient graph per thread. Code that wants to construct two graphs
  concurrently must use threads; this is acceptable for the target
  workloads.
- 24 new builtins to maintain; the wiring pattern (`dispatch.rs` arm +
  routes from both executors) is mechanical but adds dispatch surface
  area.
- FD residuals double the forward-pass count per collocation point (5
  evaluations per 2D point) compared to native AD. Phase 3d will close
  that gap.

### Builtin count

`451 → 475` user-facing builtins (24 new from `grad_graph_*`).

## Related

- [[ADR-0015 PINN PDE Problem Suite]]
- [[Autodiff]]
- [[ML Primitives]]
- [[PINN in Pure CJC-Lang]]
