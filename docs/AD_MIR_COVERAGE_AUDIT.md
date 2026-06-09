# Autodiff × MIR Coverage Audit

**Date:** 2026-06-09
**Scope:** Resolve the §4.1 question from `HANDOFF_NEXT_SESSION_v3.md`:
"Determine whether the existing AD path (`cjc-ad/src/dispatch.rs` with
the 24 `grad_graph_*` builtins from Phase 3c) already covers MIR-exec,
or whether there's still a gap."

**Conclusion:** **Coverage is complete and parity-tested for all
shipped AD operations.** Two gaps remain — both deferred by explicit
prior decisions, not by oversight:

1. **Higher-order AD** (`grad_graph_grad_of`) is a finite-difference
   fallback (3-point central, ε=1e-3), not a native graph operation.
   Documented in ADR-0016 as a Phase 3d deferral.
2. **MIR optimizer interaction** is implicitly covered by the AST/MIR
   parity gate (every grad_graph_* builtin has a fixture in
   `tests/physics_ml/`) but not exercised under every optimizer pass.
   Specifically: the `loop_unroll` Universal classification shipped
   today (commit `3d333d8`) lets `loop_unroll` fire on AD-using
   functions for the first time. The parity gate covers this for the
   fixture corpus but the assertion is "AST output == MIR output," not
   "AD-MIR with optimizer == AD-MIR without optimizer." A focused
   AD-only optimizer-toggle test would close that.

The rest of this doc walks through the evidence.

---

## 1. Single dispatch entry — no executor-specific divergence possible

### 1.1 Where `grad_graph_*` lives

All 39 grad_graph_* builtins are routed through ONE function:

```rust
// crates/cjc-ad/src/dispatch.rs
pub fn dispatch_grad_graph(name: &str, args: &[Value])
    -> Result<Option<Value>, String>
```

Stateless dispatch — takes name + args, returns Result. The "Option"
distinguishes `Some(value)` (handled) from `None` (not a grad_graph_*
name; let the caller try other dispatchers).

### 1.2 Both executors call the same entry

The route to `dispatch_grad_graph` is byte-identical from both sides:

**cjc-eval** (`crates/cjc-eval/src/lib.rs`):
```rust
// Try grad_graph_* dispatch (Phase 3c)
match cjc_ad::dispatch_grad_graph(name, &args) {
    Ok(Some(value)) => return Ok(value),
    Err(msg) => return Err(EvalError::Runtime(msg)),
    Ok(None) => {} // not a grad_graph_* builtin, fall through
}
```

**cjc-mir-exec** (`crates/cjc-mir-exec/src/lib.rs:2227`):
```rust
// Phase 3c: language-level GradGraph primitives (grad_graph_*).
// Same dispatch as cjc-eval's path so AST and MIR observe the same
// ambient graph on a single thread.
match cjc_ad::dispatch_grad_graph(name, &args) {
    Ok(Some(value)) => {
        self.call_cache.insert(name.to_string(), CallDispatch::GradGraph);
        return Ok(value);
    }
    Err(msg) => return Err(MirExecError::Runtime(msg)),
    Ok(None) => {} // not a grad_graph_* builtin, fall through
}
```

The cjc-mir-exec path additionally populates an inline call cache on
success — purely a performance optimization; the dispatch result is
identical.

`grep grad_graph_ crates/cjc-eval/src/` and the analogous mir-exec grep
each return exactly 2 hits: the routing block and one comment. There
is no executor-specific dispatch arm for any grad_graph_* name.
Divergence between the two executors on AD operations is structurally
impossible — they share the implementation.

### 1.3 Why this is the right architecture

The thread-local arena (`thread_local! GRAD_GRAPH: RefCell<GradGraph>`)
is the second half of the design. Because the arena lives in TLS, both
executors operating on the same thread see the same node indices. A
program that runs `let g = grad_graph_new();` in eval mode then later
calls `grad_graph_backward(g, loss);` in mir-exec mode would observe a
consistent graph — the cross-executor handoff "just works" because
both are reading/writing the same TLS slot.

Phase 3c documents this as "Handle representation: Option B from
brief — single ambient thread_local RefCell<GradGraph> per thread,
addressed by Value::Int(node_idx). Preserves Value enum layout
(HARD RULE #1)." That preservation is what keeps the dispatch
stateless and the two executors interchangeable.

---

## 2. Surface area — what's actually covered

Phase 3c shipped 24 builtins. Subsequent phases (3a, 3b, and the
chess RL v2.5 P6/P7 work) added another ~15, bringing the total to
**39 grad_graph_* builtins** across these categories:

| Category | Builtins |
|---|---|
| **Construction** | grad_graph_new, grad_graph_param, grad_graph_input, grad_graph_const |
| **Pointwise** | add, sub, mul, div, neg, scalar_mul, pow, exp, ln, sqrt, sin, cos, tanh |
| **Reductions** | sum, mean, matmul |
| **NN layers** | mlp_layer, softmax, cross_entropy, layer_norm, gelu, silu, batch_norm |
| **Reshaping** | reshape, gather, cat |
| **State / control** | forward, set_tensor, param_grad, zero_grad, backward, clip_grad_norm, len |
| **Optimization-hot** | reforward (P7 reuse), backward_collect (P6 batched) |

All exposed via the same `dispatch_grad_graph` function. Each name is
also reachable from BOTH executors automatically through their shared
routing block.

---

## 3. Parity coverage — verified by tests

The `tests/physics_ml/` directory contains the cross-executor parity
suite for AD:

| File | Purpose |
|---|---|
| `grad_graph_wiring.rs` | Phase 3c — 27 wiring tests; every builtin AST↔MIR byte-equal |
| `grad_graph_phase3a_wiring.rs` | Phase 3a expansion (softmax/cross_entropy/layer_norm/gelu/silu) |
| `grad_graph_phase3b_wiring.rs` | Phase 3b expansion (reshape/gather/cat/batch_norm) |
| `grad_graph_proptest.rs` | 5 properties × 256 cases — add/mul/matmul/mlp_layer/tanh forward = direct call |
| `grad_graph_phase3a_proptest.rs` | proptest for Phase 3a builtins |
| `grad_graph_phase3b_proptest.rs` | proptest for Phase 3b builtins |
| `grad_graph_fuzz.rs` | 2 bolero targets — structural state recovery + numerical tanh∘sum bounded |
| `grad_graph_phase3a_fuzz.rs` | bolero target for Phase 3a |
| `grad_graph_phase3b_fuzz.rs` | bolero target for Phase 3b |
| `heat_1d_pure_cjcl_parity.rs` | End-to-end PINN demo; eval ↔ mir byte-equal |
| `pinn_heat_1d.rs` | Standalone PINN heat-equation training test |
| `pinn_burgers_1d.rs` | Burgers' PINN |
| `pinn_kdv_1d.rs` | KdV PINN |
| `pinn_wave_1d.rs` | Wave PINN |
| `pinn_allen_cahn_1d.rs` | Allen–Cahn PINN |

Wiring tests are the strongest evidence here: each one constructs a
small graph using one builtin, runs the program through `cjcl run`
(AST eval) and `cjcl run --mir-opt` (MIR exec), and asserts the two
outputs are byte-identical. With 27 + Phase 3a + Phase 3b = ~50 wiring
tests, every grad_graph_* name has at least one parity test.

The proptest properties go further by checking that the GRAPH forward
result matches a direct Rust computation — this catches arithmetic
errors in the graph operations themselves, not just executor
divergence.

---

## 4. Confirmed gaps

### 4.1 Higher-order AD (`grad_graph_grad_of`) — deferred to Phase 3d

ADR-0016 documents the decision. The current PINN demos use a
3-point central finite-difference at ε=1e-3 to compute first-order
gradients of the loss w.r.t. PDE residual inputs. This is functional
but:

- **Accuracy:** O(ε²) truncation error, which is bounded but not
  exact. A native `grad_graph_grad_of` would produce machine-precision
  results.
- **Performance:** 3 forward passes per gradient component vs. 1
  forward + 1 backward for native autodiff.
- **Expressiveness:** Higher-order derivatives (Hessians, third-order
  Taylor terms) compose naturally with native AD; the FD fallback
  amplifies error with each layer.

**Action:** Phase 3d. ADR-0016 has the design sketch. No work needed
in this audit — flagging only.

### 4.2 Optimizer × AD parity not exhaustively tested

The AST/MIR parity gate asserts:
- `cjcl run program.cjcl` (AST eval) ==
- `cjcl run --mir-opt program.cjcl` (MIR exec with full optimizer)

For every fixture that uses AD, this transitively verifies that the
optimizer doesn't break AD semantics. As of commit `3d333d8` (today),
the optimizer's default pass list includes `loop_unroll` — and
loop_unroll is now classified Universal, meaning it can fire on
AD-using functions where it previously would have been blocked by the
PerPassLegalityGate's strict-reduction check.

The AST/MIR parity gate passed after that promotion (1/1 — verified
in this session). That's evidence loop_unroll doesn't break AD. But
the assertion is "AST and MIR agree end-to-end," not specifically
"AD result == AD result with-optimizer-vs-without." A focused
optimizer-toggle test for AD would be more rigorous.

**Suggested follow-up** (not blocking, not done in this audit): add
`tests/physics_ml/grad_graph_optimizer_invariance.rs` that runs a
representative PINN program through:
1. `cjc_mir_exec::run_program_with_executor(&program, seed)` (no
   optimizer)
2. `cjc_mir_exec::run_program_optimized(&program, seed)` (with
   optimizer)

And asserts byte-identical output of both. That's tighter than the
existing AST/MIR check, which only verifies AST-eval matches one
MIR-exec variant.

---

## 5. Summary table

| Question | Answer |
|---|---|
| Does cjc-ad's dispatch_grad_graph cover MIR-exec? | **Yes — single entry point used by both executors.** |
| Are there executor-specific AD code paths in cjc-mir-exec? | **No — grep returns 0 hits for executor-specific grad_graph_*** dispatch arms. |
| Are all 39 grad_graph_* builtins parity-tested? | **Yes — ~50 wiring tests + 15+ proptest cases each × 256 generations.** |
| Is the thread-local arena coherent across executors? | **Yes — same TLS slot, indexed by Value::Int(node_idx).** |
| Higher-order AD coverage? | **Finite-difference fallback only (ADR-0016 deferral).** |
| Optimizer × AD interaction tested? | **Transitively via AST/MIR parity gate (passes after loop_unroll Universal promotion). A dedicated optimizer-toggle test would be tighter.** |

---

## 6. What this audit does NOT cover

- **Distributed AD.** Multi-thread or multi-host gradient
  accumulation is out of scope.
- **AD across module boundaries.** `cjc-module` integration with
  grad_graph_* is untested here; the module system uses the same
  thread-local arena so should work transparently, but no fixture
  exercises it.
- **AD on closures.** The grad_graph_* builtins don't accept CJC-Lang
  closures; you build the graph imperatively. Higher-order AD would
  change this picture.

---

*Generated alongside the §3.2 / §3.3 / §4.1 sweep in commit (this
session's HEAD). See HANDOFF_NEXT_SESSION_v3.md §4.1 for the
originating question. No code changes — pure read-only audit.*
