# Phase 3c — Design Note

**Date:** 2026-04-26
**Companion to:** `PHASE_3C_API_AUDIT.md`, `PHASE_3C_PROMPT.md`
**Owner:** Lead Language Architect + Runtime Systems Engineer + Numerical Computing
Engineer (joint sign-off)

This is the "stop and think" deliverable per STEP 2 of the brief. It locks the
two architectural decisions and explains the rationale so future readers
(`ADR-0016`) can audit the trade-offs.

---

## Decision 1 — Handle representation: **Option B (ambient graph)**

The brief offered two options:

| | Option A: opaque GC handle | **Option B: ambient (graph_id, node_idx)** |
|---|---|---|
| `Value` enum | new `Value::Handle(usize)` | unchanged — `i64` for node idx |
| Storage | `Rc<RefCell<GradGraph>>` in handle table | one `RefCell<GradGraph>` thread-local in `cjc-runtime` |
| NoGC | breaks (handle table is GC) | clean (thread-local is static) |
| Determinism | iteration order tied to handle assignment | trivial — single graph, deterministic ops vec |
| MIR layout | new register kind needed | existing `i64` register reuses |
| HARD RULE #1 | risks Value enum layout change | preserved |

**Choice: Option B.** Three reasons compound:

1. The audit confirms `GradGraph` already behaves like a *region*, not a value:
   `pinn.rs` builds one per epoch, drops it, builds the next. There is no
   meaningful "two graphs alive at once" use case in any existing trainer.
2. **HARD RULE #1** — modifying `Value` enum layout breaks `cjc-mir`'s register
   layout. Option A was a stop-and-explain candidate; Option B sidesteps it
   entirely.
3. Chess RL precedent (`adam_step`) keeps optimizer *state* outside the language
   on the user side (`Vec<Vec<f64>>`), and ships a primitive that reads/writes
   detached tensors. Same pattern fits `grad_graph_*`: the graph is the
   primitive's hidden state, not a first-class value.

### Implementation shape — *and a dependency-graph correction*

The brief assumed `grad_graph_*` arms would land in
`crates/cjc-runtime/src/builtins.rs`. They cannot: `cjc-ad` already depends
on `cjc-runtime` (for `Tensor`), so adding `cjc-runtime → cjc-ad` would
introduce a cycle. Precedent for this exact problem already exists with
`cjc-quantum::dispatch_quantum`, which is a satellite dispatch crate routed
into both executors *after* the shared dispatch.

**Revised wiring:**

```rust
// crates/cjc-ad/src/dispatch.rs   (new file)
use std::cell::RefCell;
thread_local! {
    static AMBIENT: RefCell<crate::GradGraph> = RefCell::new(crate::GradGraph::new());
}
pub fn dispatch_grad_graph(name: &str, args: &[Value])
    -> Result<Option<Value>, String> { … }
```

```rust
// crates/cjc-eval/src/lib.rs and cjc-mir-exec/src/lib.rs
// (one new arm each, mirroring the existing cjc-quantum arm)
match cjc_ad::dispatch_grad_graph(name, &args) {
    Ok(Some(value)) => return Ok(value),
    Err(msg) => return Err(/* runtime error variant */(msg)),
    Ok(None) => {} // fall through
}
```

Both arms re-export through `cjc_ad::dispatch_grad_graph`. Builtin count goes
up in **`cjc-ad`'s satellite dispatch**, not in
`cjc-runtime/builtins.rs`'s 368.

Returned `usize` node indices are surfaced to the user as `Value::Int(i64)`.
The `(graph_id, node_idx)` tuple from the brief collapses to just `node_idx`
because there is exactly one graph per execution context. If a future Phase 3d
needs multi-graph support (e.g. independent optimizers in parallel), a graph
id can be added by widening the return value to a packed `i64` or to a small
struct value. Not now.

### Ambient state across the parity boundary

Both `cjc-eval::Interpreter` and `cjc-mir-exec::Executor` route builtins
through `cjc_runtime::dispatch_builtin`. Because `AMBIENT` is a
`thread_local!`, the same thread-local is touched whether the test runs the
AST or the MIR executor — guaranteeing *the same graph instance* on a single
thread.

Cross-test isolation: every test should call `grad_graph_new()` at the top of
its `.cjcl` source. Tests that share a process (cargo's test harness uses one
thread per test by default; we don't enable `--test-threads=1`) get a fresh
graph because each new test thread initializes its own thread-local slot.
Within a single thread (back-to-back invocations) the explicit `reset` keeps
isolation.

## Decision 2 — Higher-order AD: **finite-difference fallback**

The audit confirms there is no `GradOp::GradOf` today. To ship
`grad_graph_grad_of(out_idx, input_idx) -> NodeIdx`, we'd need:

1. A new variant `GradOp::GradOf { out: usize, input: usize }`.
2. Forward evaluation: trigger an inner reverse-mode pass on the existing tape
   from `out`, with seed at `input`. Result = `∂out/∂input`.
3. Backward evaluation: a vector-Jacobian product *of a* vector-Jacobian product
   — reverse-on-reverse. The math is `∂²out/∂input∂θ` for every parameter θ
   touched by the inner pass.
4. Reachability + dead-code rules updated to keep inner-pass intermediates alive.
5. New op-level tests across all 34 existing op variants, since each has a
   distinct backward rule that now also needs a "derivative of backward" rule.

Realistic LOC budget: ~600–900 in `cjc-ad/src/lib.rs`, plus ~200 in tests, plus
a non-trivial determinism review (FMA, BTreeMap iteration, parallel reduction
all become potential leakage points). Brief authorizes a fallback if this
exceeds 500 LOC. **It does.**

**Choice: ship FD fallback.** The PINN residual `u_t + α·u_xx` is computed in
`.cjcl` via:

```text
u_t  = (NN(x, t+ε) - NN(x, t-ε)) / (2ε)
u_xx = (NN(x+ε, t) - 2·NN(x, t) + NN(x-ε, t)) / ε²
```

with `ε = 1e-3`. Phase 1 smoke thresholds (L2 < 0.20, max < 0.40) have ~3×
headroom which absorbs central-difference truncation error
O(ε²·max|u_tt|) ≈ 1e-6 here.

### Bit-equality target — relaxed

| | original target | **Phase 3c target** |
|---|---|---|
| `bit_hash_f64(rust_params) == bit_hash_f64(cjcl_params)` | required | dropped |
| `(rust_l2 - cjcl_l2).abs() < 1e-6` | required | required |
| Smoke thresholds `L2 < 0.20`, `max < 0.40` | required | required |
| AST↔MIR bit-equal on the `.cjcl` PINN | required | **required** |

The first row drops because the Rust trainer uses analytic backward through
graph nodes for `u_xx` (it has access to `GradGraph` internals); the `.cjcl`
trainer uses central differences. This is a *deliberate* numerical
divergence, not a determinism bug. AST↔MIR equality is preserved because both
executors call the same FD helper through the same builtin path.

### Why not forward-mode duals?

Considered. Forward-mode AD over the input axis would give bit-identical
`u_x`, `u_xx` matching the Rust trainer. But it requires a `Tensor` type that
carries dual numbers (or two parallel forward passes per input dim) and rolls
into the same 600+ LOC budget as reverse-on-reverse. Same conclusion: not in
Phase 3c.

## Decision 3 — Activation parameter at the language boundary

`mlp_layer`'s activation is a Rust enum (`pinn::Activation`). CJC-Lang doesn't
expose Rust enums. Option A: add a CJC-Lang enum (large pipeline change).
Option B: pass an `&str`. **Option B.**

Accepted strings (lowercase, exact match): `"tanh"`, `"sigmoid"`, `"relu"`,
`"none"`, `"gelu"`, `"silu"`, `"elu"`, `"selu"`, `"sin"`. Any other value is
a runtime error. The string→enum match lives once in `cjc-runtime/builtins.rs`
inside the `grad_graph_mlp_layer` arm; both executors hit it through
`dispatch_builtin`.

## Decision 4 — Out-of-scope for this phase

Listed in the brief; reaffirmed here:
- No `GradOp::GradOf`. No reverse-on-reverse.
- No `grad_graph_*` for wave/burgers/allen-cahn/kdv `.cjcl` ports.
- No new optimizers (`adamw_step` etc.).
- No JIT path. Both executors run interpreted.

## Mapping to HARD RULES (brief §HARD RULES)

| HARD RULE | Status |
|---|---|
| #1 Value enum unchanged | ✅ Option B preserves layout |
| #2 `GradGraph::backward` keeps `&mut self` | ✅ unchanged |
| #3 Bit-equality (Rust↔.cjcl) | Replaced by RMSE<1e-6, documented |
| #4 cjc-eval ↔ cjc-mir-exec parity | ✅ both go through `dispatch_builtin` |

Rule #3's relaxation is the only deviation; it is the explicitly-authorized
fallback path, not a stop condition.

---

## Implementation order (next)

1. Add `crates/cjc-runtime/src/grad_graph_ctx.rs` with the thread-local and
   `with_ambient`/`reset_ambient` helpers.
2. Add 24 dispatch arms in `crates/cjc-runtime/src/builtins.rs`.
3. Smoke-compile both executors. They route through `dispatch_builtin` already.
4. Wiring tests (`tests/physics_ml/grad_graph_wiring.rs`) — one per builtin.
5. Proptest + bolero.
6. `examples/physics_ml/pinn_heat_1d_pure.cjcl` flagship demo.
7. `tests/physics_ml/heat_1d_pure_cjcl_parity.rs`.
8. Vault: ADR-0016, Showcase, Autodiff.md.

End of design.
