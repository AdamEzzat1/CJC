---
title: PINN in Pure CJC-Lang
tags: [showcase, ml, pinn, autodiff]
status: Implemented (Phase 3c)
date: 2026-04-26
---

# PINN in Pure CJC-Lang

A 1D heat-equation physics-informed neural network, written end-to-end in
`.cjcl` source. Every step — graph construction, MLP forward pass, FD
residual, backward, Adam parameter update — goes through language-level
primitives. The Rust side ships only the underlying tensor/autodiff/Adam
kernels.

**Source:** `examples/physics_ml/pinn_heat_1d_pure.cjcl` (~200 LOC)

## What's solved

$$ u_t = \alpha \, u_{xx}, \quad x \in [0,1], \; t \in [0,1] $$

with IC $u(x,0) = \sin(\pi x)$ and Dirichlet BCs $u(0,t) = u(1,t) = 0$.
The analytical solution $u(x,t) = e^{-\alpha \pi^2 t} \sin(\pi x)$ provides
the ground truth for accuracy gating.

## Architecture

```
2-input (x, t) → tanh(20) → tanh(20) → linear(1) → u(x, t)
```

- **Forward pass:** three calls to `grad_graph_mlp_layer(input, weight, bias, activation_str)`. The fused MLP node collapses transpose + matmul + bias-add + activation into one graph node (P8 from v2.5).
- **Residual:** central finite differences at ε=1e-3 against the network's own output. Each residual term consumes 5 forward passes (`u(x±ε, t)`, `u(x, t±ε)`, `u(x,t)`) and produces $r^2 = (u_t - \alpha u_{xx})^2$.
- **Loss:** $L = w_{\text{phys}} \cdot \overline{r^2} + w_{\text{bc}} (\overline{e_{IC}^2} + \overline{e_{BC}^2})$
- **Backward:** `grad_graph_zero_grad()` then `grad_graph_backward(total_loss)` — the same arena-based pass that powers the Rust trainer.
- **Optimizer:** hand-written Adam loop in CJC-Lang. Per parameter, calls the native `adam_step` builtin (9-arg fused element-wise update) and stores returned moment buffers as `Tensor`s.

## Language ratio

| Component | Phase 3b (pre) | Phase 3c (post) |
|---|---|---|
| Forward pass | Rust (`pinn_mlp_eval`) | CJC-Lang (`grad_graph_mlp_layer`×3) |
| Residual | Rust (FD inside trainer) | CJC-Lang (FD via `grad_graph_*`) |
| Backward | Rust (called by trainer) | CJC-Lang (`grad_graph_backward`) |
| Adam loop | Rust (`pinn_train` driver) | CJC-Lang (`for epoch` + `adam_step`) |
| Loss | Rust | CJC-Lang |
| **Effective ratio** | ~5% CJC / 95% Rust | ~85% CJC / 15% Rust |

The 15% Rust is now exactly what it should be: tensor kernels, the AD tape's
backward sweep, and `adam_step` — primitives at the same level as `matmul`,
not algorithms.

## Verified properties

1. **AST↔MIR byte-equal output.** Every reported loss line, every final
   metric, byte-identical between `cjcl run` and `cjcl run --mir-opt`.
   Tested by [[Determinism Contract]] gate
   `pure_cjcl_demo_eval_mir_byte_equal`.
2. **Loss monotonically non-increasing across reported epochs.** Catches
   any regression where the optimizer stops descending.
3. **Final metrics finite and within demo thresholds** (L2 < 0.65, max <
   1.0 at the 50-epoch demo budget; tighter gates at 500 epochs).

## Run it

```bash
# AST tree-walk
cjcl run examples/physics_ml/pinn_heat_1d_pure.cjcl

# MIR register machine (same output, byte-identical)
cjcl run --mir-opt examples/physics_ml/pinn_heat_1d_pure.cjcl
```

Sample output (50 epochs, seed=42):

```
PINN-in-pure-CJC-Lang: 1D heat equation
=======================================
epochs=50 lr=0.001 alpha=0.01

epoch=10  loss=2.701811766863513
epoch=20  loss=2.4218832403243233
epoch=30  loss=2.228239226814673
epoch=40  loss=2.0840799798704515
epoch=50  loss=2.010906592541508

Final metrics (vs analytical at t=0.5):
  L2 RMSE  : 0.5034091658245224
  Max abs  : 0.7632689643164735
```

## Why this matters

Before Phase 3c, every CJC-Lang PINN demo was a configuration shell over a
black-box Rust trainer. After Phase 3c, the *algorithm* is in CJC-Lang —
which means a user can:

- Modify the loss function without touching Rust
- Try alternative optimizers (SGD, RMSprop) by writing the update in CJC-Lang
- Mix neural and symbolic terms in the residual
- Inspect intermediate gradients via `grad_graph_param_grad`

…all while the determinism contract holds bit-equal across executors.

## Related

- [[ADR-0016 Language-Level GradGraph Primitives]]
- [[Autodiff]]
- [[ML Primitives]]
- [[PINN Support]]
- [[Chess RL v2]] — same primitives-vs-algorithm split applied to RL
