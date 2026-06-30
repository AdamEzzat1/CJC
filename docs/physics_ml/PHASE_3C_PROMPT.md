# Phase 3c — Expose `grad_graph_*` Primitives to CJC-Lang

**Status:** brief / not started.
**Goal:** flip the PINN suite's language ratio from ~5% CJC-Lang / 95% Rust to a
"primitives in Rust, algorithm in CJC-Lang" split, mirroring Chess RL v2.5's
relationship to `adam_step` (`adam_step` is a Rust primitive; the optimizer
*driver* lives in the Chess PRELUDE).

The deliverable is a `pinn_heat_1d.cjcl` written entirely against newly-exposed
`grad_graph_*` builtins, producing **bit-identical** L2/max/residual to the
existing Rust `pinn_heat_1d_nn_train` baked-in trainer at the same seed and
config — proving the language can express the same algorithm using only
language-level primitives.

---

## ROLE

You are a stacked systems team inside the CJC-Lang compiler repository at
`C:\Users\adame\CJC`.

1. **Lead Language Architect** — owns the `grad_graph_*` surface design, opaque
   handle semantics, and value/object boundary.
2. **Compiler Pipeline Engineer** — owns Lexer → Parser → AST → HIR → MIR → Exec
   flow for the new opaque types.
3. **Runtime Systems Engineer** — owns memory model for `GradGraph` arenas
   crossing the language boundary, the GC/NoGC story for graph handles, and
   builtin dispatch wiring.
4. **Numerical Computing Engineer** — owns Kahan/Binned correctness through the
   exposed primitives, FD stencil reusability, and higher-order AD plumbing
   (`du/dx` not just `dL/dparam`).
5. **Determinism & Reproducibility Auditor** — enforces bit-identical output
   between a Rust-trained PINN and a `.cjcl`-trained PINN at the same seed.
6. **QA Automation Engineer** — owns wiring tests, proptest, bolero fuzz, and
   regression gates against the existing `tests/physics_ml/` suite.

---

## PRIME DIRECTIVES (inherited verbatim from CLAUDE.md)

1. Do not break the compiler pipeline `Lexer → Parser → AST → [TC] → HIR → MIR
   → [Optimize] → Exec`.
2. Do not introduce hidden allocations or GC usage in NoGC-verified paths.
3. Maintain deterministic execution — same seed = bit-identical output.
4. Preserve backward compatibility unless explicitly impossible.
5. Never silently refactor unrelated systems.
6. Language primitives must stay minimal — higher-level functionality belongs
   in libraries (Bastion, Vizor) or, in this case, **in user `.cjcl` code**.
7. Both executors must agree — every new builtin must work in `cjc-eval` AND
   `cjc-mir-exec`, and parity must be checked.

**Phase 3c-specific addition:** the bit-equality target is *across executors
and across languages*: `Rust pinn_heat_1d_nn_train(seed=42)` ==
`.cjcl user-written PINN at seed=42` to the last ULP, in both AST and MIR.

---

## CONTEXT — what already exists

- `crates/cjc-ad/src/pinn.rs` (4,855 LOC) — Rust trainers for heat / wave /
  burgers / allen_cahn / kdv. Each builds a `GradGraph`, runs Adam, returns a
  `PinnResult`. **These stay. Do not rewrite them.**
- `crates/cjc-ad/src/lib.rs` — arena-based `GradGraph` with `ops: Vec<GradOp>`,
  `tensors: Vec<Tensor>`, `param_grads: Vec<Option<Tensor>>`.
- `cjc-runtime::dispatch_builtin` — single dispatch table. ~451 builtins today.
- The wiring pattern is **three places**:
  1. `crates/cjc-runtime/src/builtins.rs` — shared dispatch.
  2. `crates/cjc-eval/src/lib.rs` — AST executor call routing.
  3. `crates/cjc-mir-exec/src/lib.rs` — MIR executor call routing.
- `tests/physics_ml/` — 34 existing tests across 5 PDEs, plus a
  `bench/physics_ml_bench` harness that gates 10 benchmarks.

---

## FEATURE SCOPE — Phase 3c

### 1. API audit (Lead Language Architect)

Walk `crates/cjc-ad/src/lib.rs` and `pinn.rs` and produce a one-page note:
which `GradGraph` methods are *primitive* (no algorithmic content beyond an op
node) vs. which are *driver* (loops, control flow, optimizer state). Only
primitives get exposed. Save as `docs/physics_ml/PHASE_3C_API_AUDIT.md`.

### 2. Builtin surface (target ~20 names)

Minimum viable set:

```
grad_graph_new()                               -> GradGraphHandle
grad_graph_param(g, tensor)                    -> NodeIdx
grad_graph_input(g, tensor)                    -> NodeIdx
grad_graph_const(g, tensor)                    -> NodeIdx
grad_graph_matmul(g, a, b)                     -> NodeIdx
grad_graph_add(g, a, b)                        -> NodeIdx
grad_graph_sub(g, a, b)                        -> NodeIdx
grad_graph_mul(g, a, b)                        -> NodeIdx
grad_graph_pow(g, a, n_i64)                    -> NodeIdx     // integer power only
grad_graph_neg(g, a)                           -> NodeIdx
grad_graph_tanh(g, a)                          -> NodeIdx
grad_graph_sin(g, a)                           -> NodeIdx
grad_graph_cos(g, a)                           -> NodeIdx
grad_graph_exp(g, a)                           -> NodeIdx
grad_graph_mlp_layer(g, x, w, b, activation_str)  -> NodeIdx  // fused, matches Rust
grad_graph_sum(g, a)                           -> NodeIdx     // → scalar
grad_graph_mean(g, a)                          -> NodeIdx
grad_graph_forward(g, out_idx)                 -> Tensor
grad_graph_reforward(g, out_idx)               -> Tensor
grad_graph_set_tensor(g, idx, tensor)          -> ()
grad_graph_zero_grad(g)                        -> ()
grad_graph_backward(g, loss_idx)               -> ()
grad_graph_param_grad(g, idx)                  -> Tensor
grad_graph_clip_grad_norm(g, max_norm: f64)    -> ()
```

Optional, only if the heat 1D demo needs them:

```
grad_graph_grad_of(g, output_idx, input_idx)   -> NodeIdx
       // Higher-order: derivative of `output` w.r.t. `input` as a new graph node.
       // Required for u_x, u_xx, u_t in the residual.
       // Implementation: reverse-on-reverse, reuses the existing tape.
```

If `grad_graph_grad_of` proves invasive, fall back to a finite-difference
helper exposed at the language level (`fd_first(f, x, eps)`), and document the
deviation. The .cjcl heat demo can use FD u_xx with a 4-point stencil at
`ε = 1e-3` — Phase 1's smoke threshold has 3× headroom, which absorbs the
4th-order FD error.

### 3. The handle type (Runtime Systems Engineer)

`GradGraph` cannot live in `Value` as a clone-on-read scalar. Two options:

- **Option A — opaque GC handle**: wrap `Rc<RefCell<GradGraph>>` in a new
  `Value::Handle(usize)` variant + handle table. NoGC-incompatible.
- **Option B — index-as-value**: keep one ambient `GradGraph` per execution
  context, address by `(graph_id, node_idx)` integer tuple. NoGC-clean.

**Recommend Option B.** The Chess RL precedent is `Tensor` values passed by
`Rc`-cloning the underlying `Buffer` — but `GradGraph` is more like a *region*
than a value, and exposing it as ambient state matches how PINN trainers
already use it internally. Single ambient context per program also makes
deterministic ordering trivial.

If Option B is chosen, the actual builtin signatures don't take `g`:

```
grad_graph_new()                          -> ()      // resets ambient graph
grad_graph_param(tensor)                  -> i64     // node idx
...
```

The `g` argument in the surface above is for spec clarity. Pick A or B in the
design note (Step 2 of the workflow) and stick with it.

### 4. Higher-order AD — the hard part (Numerical Computing Engineer)

A PINN residual needs `u_t + α·u_xx`, where `u = NN(x, t)`. Today the Rust
trainer does this by building extra graph nodes for `u_x = ∂u/∂x` etc. via a
specialized internal path. To expose this to `.cjcl`, you need either:

- **Forward-mode dual numbers on the input axis** (cleanest, matches the
  trainer): `grad_graph_grad_of(out_idx, input_idx) -> NodeIdx` builds a new
  graph node whose value at forward time is `∂out/∂input`. Reverse-on-reverse.
- **FD fallback**: cheaper to ship, costs a constant in accuracy. The
  trainer's KdV `u_xxx` already uses 4-point FD — there is precedent.

**Decision rule:** if `grad_graph_grad_of` adds <500 LOC to `cjc-ad` and
preserves bit-equality with the trainer, ship it. Otherwise fall back to FD
and document deviation in the bit-equality test (target becomes "match within
1e-6 RMSE", not bit-identical).

### 5. Reference implementation

Write `examples/physics_ml/pinn_heat_1d_pure.cjcl` (~150-250 LOC) that:

- Initializes weights with seeded SplitMix64 (use `rng_uniform` builtin)
- Builds a 2-20-20-1 Tanh MLP via `grad_graph_mlp_layer`
- Constructs the residual `u_t - α·u_xx` using either `grad_graph_grad_of` or
  the FD helper
- Runs a hand-written Adam loop for 500 epochs (lr=1e-3) — Adam state
  management lives in `.cjcl`, calls `adam_step` builtin per parameter
- Prints final L2/max against the analytical solution
- Gates on the same Phase 1 smoke thresholds (L2 < 0.20, max < 0.40)

This is the flagship demo. It should be the file pointed to in talks/blog
posts. Keep the LOC budget tight — a 700-line PINN is not impressive; a
~200-line one written against minimal primitives is.

---

## TESTS — all under `tests/physics_ml/`

Co-locate with the existing PINN tests. Do **not** spread across multiple
test directories.

### 5a. Wiring tests (`tests/physics_ml/grad_graph_wiring.rs`)

One file, one `#[test]` per builtin. Each test:
1. Builds a tiny graph (one or two nodes).
2. Runs forward in `cjc-eval`, captures result.
3. Runs forward in `cjc-mir-exec`, captures result.
4. Asserts bit-equality of both `Vec<f64>` outputs via the existing
   `crate::common::bit_hash_f64` helper.

Coverage target: **one test per exposed builtin.** ~20-25 tests.

### 5b. Proptest (`tests/physics_ml/grad_graph_proptest.rs`)

Use `proptest` (already in Cargo.lock):

```rust
proptest! {
    #[test]
    fn matmul_node_matches_direct_call(
        m in 1usize..8, k in 1usize..8, n in 1usize..8,
        seed in any::<u64>(),
    ) {
        // Build (a @ b) via grad_graph_matmul; build same product via
        // cjc_runtime::matmul. Assert bit-equal.
    }

    #[test]
    fn add_then_backward_returns_unit_grad(/* ... */) { /* ... */ }

    #[test]
    fn mlp_layer_forward_matches_unfused_path(/* ... */) { /* ... */ }
}
```

Five proptest cases minimum: `add`, `mul`, `matmul`, `mlp_layer`, `tanh`.
Each runs ≥256 cases.

### 5c. Bolero fuzz (`tests/physics_ml/grad_graph_fuzz.rs`)

```rust
bolero::check!()
    .with_arbitrary::<GraphProgram>()
    .for_each(|prog| {
        // Build a randomly-shaped graph from `prog`. Run forward.
        // Invariants:
        //   1. No panic.
        //   2. No NaN/Inf in finite-input branches.
        //   3. AST↔MIR bit-equal.
    });
```

Two fuzz targets: structural (random shape combinations) and numerical
(random finite f64 inputs into a fixed graph).

### 5d. Bit-equality regression test
(`tests/physics_ml/heat_1d_pure_cjcl_parity.rs`)

The flagship test. Runs `pinn_heat_1d_nn_train(epochs=500, seed=42)` (Rust)
and the `.cjcl` reference at the same seed. Asserts:

- `bit_hash_f64(rust_final_params) == bit_hash_f64(cjcl_final_params)` if
  `grad_graph_grad_of` shipped, OR
- `(rust_l2 - cjcl_l2).abs() < 1e-6` if FD fallback shipped.

Either way, smoke thresholds (L2 < 0.20, max < 0.40) must hold for both.

### 5e. Regression gate (no new file, just a hard rule)

After Phase 3c lands, **all 34 existing `tests/physics_ml/` tests must
continue passing without modification**. Run:

```bash
cargo test --test physics_ml --release
```

Expected: `34 passed; 0 failed; 2 ignored` plus the new wiring/proptest/fuzz
counts. Total target: ~70+ tests in `tests/physics_ml/`.

---

## DEVELOPMENT WORKFLOW (6 steps, inherited from CLAUDE.md)

### STEP 1 — Codebase analysis
Audit `cjc-ad` API surface (Step §1 above). Output: `PHASE_3C_API_AUDIT.md`.

### STEP 2 — Safe design
Pick handle representation (Option A vs B). Pick higher-order AD strategy
(`grad_graph_grad_of` vs FD fallback). Output:
`docs/physics_ml/PHASE_3C_DESIGN.md`. **Stop and reconsider** if either
decision implies modifying `Value` enum layout or `GradGraph::backward`
signature — those are pipeline-level changes that need a separate ADR.

### STEP 3 — Implementation
Wire all builtins through the canonical three-place pattern. Hold to one
commit per logical group: graph construction, op nodes, forward/backward,
parameter access. Do not refactor unrelated dispatch arms.

### STEP 4 — Test creation
Write tests in the order: wiring → proptest → fuzz → bit-equality demo.

### STEP 5 — Regression gate
Run, in order:
1. `cargo test --test physics_ml --release` — must show 34 prior tests pass
   plus all new tests.
2. `cargo run --release -p physics-ml-bench` — 10/10 must still pass with
   identical L2/max/residual values to today.
3. AST↔MIR diff on **all 6** `.cjcl` examples (5 existing + new
   `pinn_heat_1d_pure.cjcl`).
4. `python scripts/vault_audit.py` — must exit 0.

If any step fails, stop and fix root cause. Never bypass with `#[ignore]`.

### STEP 6 — Documentation
See "Vault changes" below. **The vault update is part of the deliverable, not
a follow-up.**

---

## VAULT CHANGES (Obsidian) — required, not optional

Update `CJC-Lang_Obsidian_Vault/`:

1. **`06_Tensors_ML_AD/Autodiff.md`** — add a section "Language-level GradGraph
   primitives" listing the new builtins with a 1-line semantic each.
2. **`06_Tensors_ML_AD/ML Primitives.md`** — link to the new pure-`.cjcl`
   reference implementation.
3. **`13_ADRs/ADR-0016 Language-Level GradGraph Primitives.md`** — new ADR.
   Decision: which handle representation, which higher-order AD strategy,
   why. List the alternative considered. Include the bit-equality test
   result.
4. **`13_ADRs/ADR Index.md`** — add ADR-0016 row.
5. **`09_Showcase/`** — add a new note `PINN in Pure CJC-Lang.md` that walks
   through the `pinn_heat_1d_pure.cjcl` flagship demo. Cross-link from
   `Autodiff.md`. This is the user-facing deliverable.
6. **`10_Roadmap_and_Open_Questions/`** — mark the "PINN-in-language"
   roadmap item complete; add a new open question for Phase 3d (do we expose
   `grad_graph_*` as a Bastion-style high-level wrapper, or keep raw?).
7. Run `python scripts/vault_audit.py` — must exit 0. Fix any broken
   wikilinks before declaring done.

Update root project memory:

8. `C:\Users\adame\.claude\projects\C--Users-adame-CJC\memory\MEMORY.md` — add
   one line under "Completed Milestones" for Phase 3c.
9. Update the "Builtin Count" memory entry — was 451, will be ~471.

---

## OUTPUT FORMAT

For every file changed:

```
FILE: path/to/file.rs
<reason for change in one sentence>
<actual code or diff>
```

End with:

```
Test Summary:
  Wiring tests:    XX passed
  Proptest cases:  YY × 256 passed
  Bolero fuzz:     ZZ targets, no panics in N iterations
  Bit-equality:    Rust vs .cjcl PINN — <bit-identical | RMSE 1.4e-7>
  Regression:      34 prior physics_ml tests, 10/10 bench gates — all green
  Vault audit:     1,622+/1,622+ wikilinks resolve

Builtin count:    451 → ~471
LOC delta:        +X CJC-Lang flagship demo, +Y Rust dispatch wiring
```

---

## HARD RULES

If any of the following becomes true, **stop and explain**, do not push through:

1. The new builtins require modifying `Value` enum layout in a way that
   breaks `cjc-mir`'s register layout.
2. Higher-order AD requires breaking `GradGraph::backward`'s `&mut self`
   signature established in v2.4.
3. Bit-equality between Rust and `.cjcl` PINN cannot be achieved even with
   identical seeds — that means a hidden non-determinism (BTreeMap iteration
   order, FMA, parallel reduction) leaked through the new surface.
4. Proptest finds a counterexample where `cjc-eval` and `cjc-mir-exec`
   disagree on a `grad_graph_*` op output by more than 0 ULP.

In every "stop" case: write up the obstacle in `PHASE_3C_DESIGN.md` and
propose Option B (or Option C — usually a smaller scope cut).

---

## OUT-OF-SCOPE for Phase 3c

These are tempting and you must not do them in this phase:

- Wiring `grad_graph_*` to JIT or ahead-of-time compilation (Phase 3d+).
- Writing pure-`.cjcl` versions of wave/burgers/allen-cahn/kdv (one flagship
  demo is enough; reusable patterns can graduate to a Bastion library later).
- Adding new optimizers (`adamw_step`, `lion_step`). Adam is already wired.
- Refactoring `pinn.rs`'s internal `GradGraph` calls to go through the new
  builtin surface. The Rust trainers stay direct — that's the regression
  oracle.
- Touching the heat / wave / burgers / allen-cahn / kdv `.cjcl` files. They
  are the regression baselines.

---

## DELIVERABLES CHECKLIST

- [ ] `docs/physics_ml/PHASE_3C_API_AUDIT.md`
- [ ] `docs/physics_ml/PHASE_3C_DESIGN.md`
- [ ] ~20 new builtins wired in `cjc-runtime`/`cjc-eval`/`cjc-mir-exec`
- [ ] `examples/physics_ml/pinn_heat_1d_pure.cjcl` (≤ 250 LOC)
- [ ] `tests/physics_ml/grad_graph_wiring.rs` (~20-25 tests)
- [ ] `tests/physics_ml/grad_graph_proptest.rs` (≥5 proptests, ≥256 cases each)
- [ ] `tests/physics_ml/grad_graph_fuzz.rs` (≥2 bolero targets)
- [ ] `tests/physics_ml/heat_1d_pure_cjcl_parity.rs` (bit-equality vs Rust)
- [ ] All 34 prior `tests/physics_ml/` tests still pass
- [ ] All 10 bench gates still pass with identical numerics
- [ ] AST↔MIR diff = 0 on all 6 `.cjcl` examples
- [ ] `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0016 ...md` shipped
- [ ] `09_Showcase/PINN in Pure CJC-Lang.md` shipped
- [ ] `scripts/vault_audit.py` exit 0
- [ ] `MEMORY.md` updated

---

**End of brief.** Open-question routing: implementation choices land in
`PHASE_3C_DESIGN.md`; cross-cutting architecture lands in `ADR-0016.md`.
