# ABNG Phase 0.3b — Per-node Bayesian Linear Regression head (Design Note)

**Date:** 2026-05-06
**Builds on:** [Phase 0.3a](PHASE_0_3a_DESIGN.md)
**Scope:** Attach a per-node Bayesian linear regression head on top of the MLP's penultimate features. Normal-Inverse-Gamma conjugate update. Epistemic + aleatoric uncertainty as separate quantities.

> **Naming note (Phase 0.4 Track C-2.3.7):** the BLR head is per-*node*,
> not per-*leaf*. `set_blr_prior` initializes the root, and every
> `add_node` / `force_grow` / `force_split` initializes a fresh
> `BlrState` from the prior on the new node. Pre-0.4 this doc and the
> code said "per-leaf" — the code was always per-node; only the name
> was wrong. The body of this note has been updated to match.

## Why this slice

Phase 0.3a made ABNG a neural architecture. Phase 0.3b makes it
*Bayesian-inspired* in a defensible way — the only Bayesian object in
the model is the last linear layer's weights, given a Normal-Inverse-Gamma
prior. The MLP supplies penultimate features; the BLR head provides a
closed-form predictive distribution with separate epistemic and
aleatoric uncertainty.

This is the **"Bayesian last layer"** architecture (Snoek et al. 2015):
~90% of the calibration benefit at <1% of the compute cost compared to
a fully variational BNN. It's the right scope for a system whose
deterministic guarantees must remain bit-identical.

What 0.3b *does* deliver:

1. `BlrPrior { precision, a, b }` — graph-wide one-shot install. A
   simple isotropic prior `N(0, σ²/precision · I)` with
   `σ² ~ InvGamma(a, b)`.
2. `BlrState` per node — full posterior `(mean, precision_matrix,
   a, b, n_seen)` over the penultimate-features → output regression.
3. Deterministic NIG conjugate update with Cholesky decomposition.
4. `blr_features(node_id, x_idx)` — penultimate features as a
   GradGraph node, ready for either backward (training the MLP) or
   `blr_update` (training the BLR posterior).
5. `blr_update(node_id, features, y)` — closed-form NIG update.
6. `blr_predict(node_id, features) → [mean, epistemic_var, aleatoric_var]`.
7. Snapshot **v4** (clean break from v3).
8. ~6 new builtins (506 → 512).

What 0.3b explicitly does **not** deliver:

* **Multi-output BLR** — Phase 0.3c+. For chess RL value head and other
  scalar regression, single-output is enough.
* **Streaming Sherman-Morrison rank-1 update** — the batch update form
  is fine for the typical "update on a minibatch" pattern; SM is an
  opt for the streaming case.
* **OOD scoring, calibration bins, drift detector** — Phase 0.3c.
* **Structural decisions** — Phase 0.3d. They depend on the *evidence*
  BLR + calibration provide, so they have to come after.

## Design

### `BlrPrior`

```rust
pub struct BlrPrior {
    pub precision: f64,    // λ_0 — isotropic prior precision (1/variance)
    pub a:         f64,    // a_0 — InverseGamma shape  (a_0 > 0)
    pub b:         f64,    // b_0 — InverseGamma scale  (b_0 > 0)
    pub config_hash: [u8; 32],
}
```

The prior is `w ~ N(0, σ²/λ_0 · I)`, `σ² ~ InvGamma(a_0, b_0)`. Setting
`λ_0` very small produces a near-flat prior on the weights; setting it
moderate (e.g. 1.0) gives mild regularization.

### `BlrState` per node

```rust
pub struct BlrState {
    pub d:          u32,       // penultimate-feature dimensionality
    pub mean:       Tensor,    // shape [d] — posterior mean of weights
    pub precision:  Tensor,    // shape [d, d] — posterior precision matrix Λ
    pub a:          f64,       // posterior IG shape
    pub b:          f64,       // posterior IG scale
    pub n_seen:     u64,       // total observations applied
}
```

Initial state at install time: `mean = 0`, `precision = λ_0 · I`,
`a = a_0`, `b = b_0`, `n_seen = 0`.

`d` is determined by the penultimate-features dimension at install
time, which is `head.hidden_dims.last()` if non-empty else
`head.input_dim`. **Phase 0.3b requires the leaf head to be configured
before BLR**, so `d` is unambiguous.

### NIG conjugate update

Given a batch `X: [n, d]` and `y: [n]`:

```
Λ_new = Λ + X^T X
m_new = Λ_new^(-1) (Λ μ + X^T y)
a_new = a + n / 2
b_new = b + 0.5 · (μ^T Λ μ + y^T y - m_new^T Λ_new m_new)
n_new = n_seen + n
```

The matrix inversion in `m_new` is implemented via Cholesky
decomposition of `Λ_new`: solve `L L^T m = (Λ μ + X^T y)` by forward +
back substitution.

### Prediction at point φ

```
mean = μ^T φ
epistemic_var = φ^T Λ^(-1) φ                     # variance of the posterior mean
aleatoric_var = b / (a - 1)        if a > 1      # mean of InvGamma noise variance
              = +∞                 otherwise
total = aleatoric_var · (1 + epistemic_var)      # full predictive variance
```

We expose epistemic and aleatoric *separately* — the caller can combine
them if they want the full predictive variance. This is what makes ABNG
defensibly transparent about uncertainty decomposition.

### Determinism contract

* Cholesky decomposition uses plain `+`, `-`, `*`, `/`, `sqrt` on
  `f64` — no FMA. Bit-deterministic for fixed input order.
* Sums in NIG update use `KahanAccumulatorF64` from `cjc-repro`.
* Cholesky uses the lower-triangular form computed in column-major
  order; the loop ordering is fixed.

### Numerical safeguards

* `a > 1` check before computing aleatoric mean — return `+∞` (`f64::INFINITY`)
  when `a ≤ 1` rather than divide-by-zero.
* **No silent Cholesky regularization.** An earlier draft of this note
  claimed `f64::EPSILON` would be added to the diagonal of `Λ_new`
  before decomposition. This was never shipped — the as-built code
  errors with `BlrError::NonPositiveDefinite` on any non-positive
  pivot (`crates/cjc-abng/src/blr.rs::cholesky`). The hard-error
  contract is intentional: silent regularization would mask data
  corruption and break the bit-determinism guarantee. With the
  posterior NIG update, `Λ_new = Λ + XᵀX` is positive-definite by
  construction whenever `Λ` is — so a non-PD pivot signals a corrupt
  state, not a numerically-tight one. Phase 0.4 Track C-2.3.10
  brought this note in line with the code; the architecture doc
  (`ABNG_CURRENT_ARCHITECTURE.md` §6.5) now also calls out the
  no-regularization contract explicitly.
* `b > 0` invariant maintained by the NIG update math. If numerical
  drift produces `b < f64::EPSILON`, the value is clamped to
  `f64::EPSILON` to keep the InverseGamma posterior well-defined
  (Phase 0.4 Track C-2.3.4); on clamp, `BlrState::update` returns
  `Ok(Some(b_pre_clamp))` and the graph layer's `blr_update` appends
  a `BlrNumericalRescue` audit event (tag `0x18`) immediately after
  the corresponding `BlrUpdated`. The clamped state is bit-identical
  with or without the audit event.

### Snapshot format v4

```
magic           "ABNG\x04"     (5)
... [v3 layout up to per-node section] ...
per node:
  ... [v3 per-node fields] ...
  blr_present  u8              (1)        0x00 if no BLR, 0x01 if present
  if blr_present:
    d            u32 BE        (4)
    mean         f64 BE × d
    precision    f64 BE × d²
    a            f64 BE
    b            f64 BE
    n_seen       u64 BE
n_events ...
```

Header gains a `blr_prior_present u8` flag right after the leaf-head
section; if present, encodes `precision, a, b` (8 bytes each) plus
`config_hash` (32 bytes).

### Audit kinds (extending Phase 0.3a)

| Tag  | Kind | Payload |
|------|------|---------|
| 0x08 | `BlrPriorConfigured` | `config_hash: [u8; 32]` |
| 0x09 | `BlrInitialized` | `state_hash: [u8; 32]` (per-node) |
| 0x0A | `BlrUpdated` | `state_hash: [u8; 32]` (per-node, post-update) |

### New builtins (6)

| Name | Args | Returns | Purpose |
|---|---|---|---|
| `abng_set_blr_prior` | `g, precision: f64, a: f64, b: f64` | `Void` | install + freeze prior; init all existing nodes' BLR state |
| `abng_blr_features` | `g, node_id, x_grad_idx: i64` | `Int` | penultimate-features GradGraph idx |
| `abng_blr_update` | `g, node_id, features_2d_tensor, y_1d_tensor` | `Void` | NIG update from a batch |
| `abng_blr_predict` | `g, node_id, features_1d_tensor` | `Tensor[3]` | `[mean, epistemic_var, aleatoric_var]` |
| `abng_blr_state_hash` | `g, node_id` | `String` (hex) | SHA-256 of BLR state canonical bytes |
| `abng_blr_n_seen` | `g, node_id` | `Int` | observations applied to this node's BLR |

Total surface after 0.3b: **512 dispatch arms** (506 + 6).

## Tests (estimate)

* In-crate: 77 → ~95 (+18) — Cholesky correctness, NIG update math
  vs closed-form for d=2, predict, prior-installation guard,
  before-leaf-head ordering, snapshot v4.
* Integration: 122 → ~145 (+23) — every new builtin, AST↔MIR parity,
  end-to-end "deep features + BLR predict" round-trip with
  uncertainty quantities verified bit-identical across executors.

## Risks

1. **Cholesky implementation hand-roll.** ~30 LOC. Mitigated by the
   "vs closed-form" property test for small d.
2. **Snapshot break v3 → v4.** Same justification; ABNG still in pre-1.0
   dev cycle.
3. **`a > 1` discipline.** Aleatoric mean is undefined when `a ≤ 1`.
   Phase 0.3b returns `f64::INFINITY` instead — a calibration-aware
   user should treat that as "abstain". Documented.
4. **`d` is fixed at BLR install time.** If the user later changes the
   MLP head's hidden_dims (which is impossible — it's frozen too), `d`
   would be wrong. Frozen-head + frozen-prior + d=fn(head) keeps the
   contract closed.
