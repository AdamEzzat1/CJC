# Physics-informed learning demos

Tiny, verifiable physics-informed examples written entirely against the
existing CJC-Lang surface. No compiler/runtime changes, no new builtins.

The companion to these demos is the production-grade pure-CJC-Lang PINN at
[`examples/physics_ml/pinn_heat_1d_pure.cjcl`][pure]. The demos here are
deliberately the smallest possible programs that still falsify a real
claim about the language.

[pure]: ../physics_ml/pinn_heat_1d_pure.cjcl

| File | What it shows | Closed-form check |
| --- | --- | --- |
| `01_harmonic_oscillator_residual.cjcl` | Plug the analytic solution `u(t)=cos(ωt)` into `u″ + ω²u`, with a 3-point central FD for `u″`. | `L_∞` residual = 1.33e-6, well within the analytic FD bound `h²ω⁴/12 ≈ 1.33e-6` at h=1e-3, ω=2. |
| `02_heat_1d_analytic_residual.cjcl` | Mixed-derivative residual `u_t − α u_xx` on the analytic solution `u(x,t) = sin(πx)·exp(−απ²t)`, plus IC and BC verification. | Interior PDE residual `≤ 8.1e-8`, IC and BCs exact (no FD truncation; pure `sin`/`exp` arithmetic). |
| `03_grad_graph_one_step_descent.cjcl` | One full forward / backward / SGD step on `L(a) = (a−c)²` using only `grad_graph_*` builtins. Verifies analytic gradient and post-step loss. | At a=0: `L=4`, `dL/da=−4`. After lr=0.1 step: `a=0.4`, `L=2.56`. All exact to 1e-12. |

## Run

```bash
cjcl run examples/physics_informed_learning/01_harmonic_oscillator_residual.cjcl
cjcl run examples/physics_informed_learning/02_heat_1d_analytic_residual.cjcl
cjcl run examples/physics_informed_learning/03_grad_graph_one_step_descent.cjcl
```

## Parity gate

```bash
cjcl parity examples/physics_informed_learning/01_harmonic_oscillator_residual.cjcl
cjcl parity examples/physics_informed_learning/02_heat_1d_analytic_residual.cjcl
cjcl parity examples/physics_informed_learning/03_grad_graph_one_step_descent.cjcl
# All three: Verdict: IDENTICAL
```

## What this proves about CJC-Lang today

- **The math primitives compose to express physics laws.** The harmonic
  oscillator and 1D heat-equation residuals on their analytic solutions
  drop to FD-truncation order (≤ 8e-8 to ≤ 1.4e-6 depending on grid),
  not "language noise." That is the prerequisite for any PINN-style
  workflow being meaningful.
- **`grad_graph_*` (Phase 3c) is honest.** Reverse-mode AD returns the
  analytic gradient on the simplest non-trivial case (`(a−c)²`), and a
  hand-written SGD step reduces the loss to the closed-form post-step
  value. This is the substrate the bigger PINN demos rely on.
- **Determinism extends to AD.** Both the AST and MIR executors agree
  byte-equally on every printed value across all three demos.

## What this does **not** prove

- That CJC-Lang is competitive with JAX, PyTorch, or Julia SciML on PINN
  training quality, throughput, or numerical convergence at scale.
- That a neural network *trained* on these residuals reaches publishable
  accuracy. The flagship trained PINN demo
  ([`examples/physics_ml/pinn_heat_1d_pure.cjcl`][pure]) intentionally
  ships at a 50-epoch demo budget, not a converged-research budget.
- That higher-order autodiff (true `∂²u/∂x²` from a network rather than
  FD) is supported. ADR-0016 documents the deferral; demo 03 only uses
  first-order reverse mode.

## Honest tolerance choices

`01_harmonic_oscillator_residual.cjcl` uses tol = 5e-6, **not** 1e-12.
The 3-point central FD has truncation error `h²/12 · u⁗(ξ)`, so for
`u(t) = cos(ωt)` with ω=2, h=1e-3 the bound is ≈ 1.33e-6. This is
finite-difference physics, not a language defect. The tolerance is
documented inline in the demo.

`02_heat_1d_analytic_residual.cjcl` uses tol = 5e-4 for the interior
mixed-derivative residual; the realised L_∞ is ~8e-8 (much tighter than
the bound, because the residual involves cancellation between two FD
errors of the same order).
