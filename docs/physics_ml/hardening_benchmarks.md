# Physics ML Hardening Benchmarks

Phase-staged benchmark suite that proves the post-hardening physics-ML
stack (v0.1.5+) produces accurate, deterministic, reproducible PINN
results across the AST and MIR executors. Every benchmark includes:

- Governing equation, domain, IC, BC, analytical/reference solution.
- L2 relative error and max absolute error against the reference.
- Mean PDE residual at the final parameters.
- Bit-identical replay verification (same seed → same `final_params`).
- AST/MIR agreement statement (structural for builtins, runtime for
  `.cjcl` script paths).
- Pass/fail thresholds with explicit numerical bounds.

## Phase Plan

| Phase | Scope | Status |
|---|---|---|
| 1 | 1D heat equation — examples, tests, bench harness, docs | **shipped** |
| 2 | 1D wave + Burgers (smooth shock-formation regime) | **shipped** |
| 3 | Allen-Cahn + KdV + external reference inference (`pinn_mlp_eval_grid`) | **shipped** |
| 3b | Wire `pinn_eval_at` as a `cjcl` builtin so .cjcl examples can do their own L2; Allen-Cahn implicit-FD reference for full-domain L2 | planned |
| 4 | Property tests (residual-decreases, no-NaN) + Bolero fuzz | planned |
| 5 | Vault notes + accuracy/determinism reports + deferred-features doc | planned |

## Phase 1 — 1D Heat Equation

### Equation

```
u_t = α · u_xx              on x ∈ [0, 1], t ∈ [0, 1]
IC:   u(x, 0) = sin(π x)
BCs:  u(0, t) = u(1, t) = 0
α     = 0.01
```

### Analytical Solution

```
u(x, t) = exp(-α π² t) · sin(π x)
```

Built into the trainer (`cjc_ad::pinn::pinn_heat_1d_nn_train`); the
returned `PinnResult.l2_error` and `PinnResult.max_error` are computed
against this expression on a deterministic evaluation grid.

### Configuration

| Parameter | Smoke | Convergence | Long (ignored) |
|---|---|---|---|
| Epochs | 500 | 2 000 | 5 000 |
| Collocation points | 64 | 64 | 128 |
| IC points | 50 | 50 | 100 |
| BC points | 25 | 25 | 50 |
| Hidden layers | 2 × 20 (Tanh) | same | same |
| Optimizer | Adam, lr=1e-3, cosine decay | same | same |
| Seed | 42 | 42 | 42 |

### Pass/Fail Thresholds

| Metric | Smoke | Convergence | Long-converge |
|---|---|---|---|
| L2 relative error | < 0.20 | < 0.05 | < 0.01 |
| Max abs error | < 0.40 | < 0.10 | < 0.02 |

Calibration note: observed values at seed=42 are L2 ≈ 0.07 / max ≈ 0.13
(smoke), L2 ≈ 0.0072 / max ≈ 0.0125 (convergence). Thresholds carry
~3× headroom over observed values so the gates catch real regressions
without flaking on platform-level variance. Long-converge bounds are
extrapolated assuming continued convergence; revisit after the first
real run.
| Determinism | bit-identical replay required | required | required |
| Residual / loss-history | finite, downward trend | same | same |

Thresholds are calibrated to the default network (2-20-20-1, Tanh).
Tightening requires either deeper networks or longer training budgets.

### Files

- Example: [`examples/physics_ml/pinn_heat_1d.cjcl`](../../examples/physics_ml/pinn_heat_1d.cjcl)
- Tests: [`tests/physics_ml/pinn_heat_1d.rs`](../../tests/physics_ml/pinn_heat_1d.rs)
- Bench harness: [`bench/physics_ml_bench/src/main.rs`](../../bench/physics_ml_bench/src/main.rs)
- Output: `target/physics_ml_bench/results.json`, `target/physics_ml_bench/summary.md`

### Running

```bash
# Fast tests (smoke + determinism, ~seconds)
cargo test --test physics_ml --release

# Long-running convergence test (~minutes)
cargo test --test physics_ml --release -- --ignored

# Bench harness — writes results.json + summary.md
cargo run --release -p physics-ml-bench

# Run the .cjcl example through both executors and diff
cjcl run examples/physics_ml/pinn_heat_1d.cjcl > /tmp/ast.out
cjcl run --mir-opt examples/physics_ml/pinn_heat_1d.cjcl > /tmp/mir.out
diff /tmp/ast.out /tmp/mir.out  # should be empty
```

## Determinism Guarantee

Every PINN trainer in `cjc-ad/src/pinn.rs` uses:

- `cjc_repro::Rng` (SplitMix64) seeded from `config.seed` for all
  random draws (Xavier weight init, collocation-point sampling).
- `BTreeMap`/`Vec` for any iterated state — no `HashMap` random ordering.
- Kahan / binned accumulators for floating-point reductions
  (see `cjc-runtime/src/ml.rs` and `cjc-runtime/src/tensor.rs`).
- No FMA in SIMD kernels (preserves bit-identical results vs. scalar).

Bit-identical replay is verified per-benchmark by hashing the final
parameter vector with a SplitMix64-based mix and comparing across two
runs with the same seed (`tests/physics_ml/common.rs::bit_hash_f64`).

## AST/MIR Equivalence

The 12 `pinn_train_*` builtins are wired identically in both executors
([`cjc-eval/src/lib.rs:2459`](../../crates/cjc-eval/src/lib.rs) and
[`cjc-mir-exec/src/lib.rs:1941`](../../crates/cjc-mir-exec/src/lib.rs))
and dispatch into the *same* Rust function in `cjc-ad`. Numerical
divergence between executors is structurally impossible for these
builtins; the bench harness reports this as `trivial`.

For benchmarks composed in pure CJC-Lang (where the script itself
performs autodiff and gradient updates), the `.cjcl` parity gate
applies: the script is run through both `cjcl run` and
`cjcl run --mir-opt` and outputs are diffed line-for-line. This gate
is wired in Phase 4.

## Limitations

- Higher-order spatial derivatives (∂²u/∂x², ∂³u/∂x³) inside the
  trainer use **central finite differences**, not symbolic autodiff.
  This is intentional (cheaper, deterministic) but limits the maximum
  attainable accuracy for stiff problems. See
  [`docs/physics_ml/deferred_physics_features.md`](deferred_physics_features.md)
  (Phase 5) for the full deferred list.
- Schrödinger-style benchmarks rely on real+imaginary channels because
  `GradGraph` does not yet support complex autodiff.

## Phase 2 — 1D Wave Equation

### Equation

```
u_tt = c² · u_xx                on x ∈ [0, 1], t ∈ [0, 1]
IC:    u(x, 0) = sin(π x),  u_t(x, 0) = 0
BCs:   u(0, t) = u(1, t) = 0
c      = 1.0
```

### Analytical Solution

```
u(x, t) = sin(π x) · cos(c π t)
```

Built into the trainer (`cjc_ad::pinn::pinn_wave_train`). **Important
caveat:** the trainer evaluates `l2_error` and `max_error` at the
**mid-time slice t=0.5 only** (50 spatial points). This is a
representative metric — not a full space-time L2. A full-domain
evaluator is on the Phase 3 list.

### Configuration

| Parameter | Smoke | Convergence | Long (ignored) |
|---|---|---|---|
| Epochs | 500 | 2 000 | 5 000 |
| Collocation | 64 | 64 | 128 |
| Hidden layers | 2 × 20 (Tanh) → 1 (3 hidden) | same | same |
| Optimizer | Adam, lr=1e-3, cosine decay | same | same |
| Seed | 42 | 42 | 42 |

### Files

- Example: [`examples/physics_ml/pinn_wave_1d.cjcl`](../../examples/physics_ml/pinn_wave_1d.cjcl)
- Tests: [`tests/physics_ml/pinn_wave_1d.rs`](../../tests/physics_ml/pinn_wave_1d.rs)

## Phase 2 — 1D Viscous Burgers' Equation

### Equation

```
u_t + u·u_x = ν · u_xx          on x ∈ [-1, 1], t ∈ [0, 1]
IC:           u(x, 0) = -sin(π x)
BCs:          u(-1, t) = u(1, t) = 0
ν             = 0.01 / π ≈ 0.00318  (kinematic viscosity)
```

### Reference Solution

**No closed-form analytical solution exists.** The Cole-Hopf transform
yields a series solution but the trainer does not compute it. Trainer-
reported `l2_error` / `max_error` measure **IC reproduction at t=0**
against `-sin(π x)` (50 spatial points) — this is _not_ a global L2.

The most meaningful global convergence indicator for Burgers is
`mean_residual` (RMS of the PDE residual at final params).

Burgers is the canonical "shock formation" PDE: the smooth IC
`-sin(π x)` steepens into a near-discontinuity at `x ≈ 0` by `t ≈ 1/π`.
PINNs handle the smooth regime well; near-shock accuracy degrades
monotonically with viscosity.

### Configuration

| Parameter | Smoke | Convergence |
|---|---|---|
| Epochs | 500 | 2 000 |
| Collocation | 64 | 64 |
| Hidden layers | 2 × 20 (Tanh) → 1 (3 hidden) | same |
| ν | 0.01/π | same |
| Seed | 42 | 42 |

### Files

- Example: [`examples/physics_ml/pinn_burgers_1d.cjcl`](../../examples/physics_ml/pinn_burgers_1d.cjcl)
- Tests: [`tests/physics_ml/pinn_burgers_1d.rs`](../../tests/physics_ml/pinn_burgers_1d.rs)

### Phase 2 Pass/Fail Thresholds

Calibrated post-run against observed seed=42 values; see
`bench/physics_ml_bench/src/main.rs` and `tests/physics_ml/common.rs`.

| Benchmark | Budget | L2 | Max | Residual |
|---|---|---|---|---|
| wave 1D (mid-slice) | smoke (500 ep) | < 0.50 | < 1.00 | < 0.50 |
| wave 1D (mid-slice) | conv (2000 ep) | < 0.30 | < 0.50 | < 0.30 |
| burgers 1D (IC repro) | smoke (500 ep) | < 0.30 | < 0.55 | < 0.80 |
| burgers 1D (IC repro) | conv (2000 ep) | < 0.20 | < 0.40 | < 0.50 |

Calibration note: observed values at seed=42 are
- **wave smoke**: L2 ≈ 0.236, max ≈ 0.362, residual ≈ 0.236
- **wave conv**: L2 ≈ 0.142, max ≈ 0.210, residual ≈ 0.142
- **burgers smoke**: IC L2 ≈ 0.095, IC max ≈ 0.173, residual ≈ 0.252
- **burgers conv**: IC L2 ≈ 0.056, IC max ≈ 0.113, residual ≈ 0.147

Wave thresholds carry only ~2× headroom (vs heat 1D's 3×) because
wave converges slower under the same epoch budget — tightening
further requires more epochs, not a fix to the gate. Burgers
carries ~3× headroom across all metrics. Both wave's `l2_error`
and `mean_residual` are evaluated on the same mid-time slice and
happen to coincide numerically; for Burgers the two are distinct
(IC slice vs full collocation set), and `residual` is the more
meaningful global signal.

## Important Note on Trend Tests

Phase 2 explicitly **does not** include head-vs-tail loss trend
assertions for wave or burgers. The default trainer schedule uses
`boundary_weight=10` with a startup ramp that pushes `total_loss`
*up* during the first ~100 epochs (the network is fitting the BC
constraint, which dwarfs IC and physics terms early). A naive
"loss decreased" gate flakes against this. Heat 1D *does* keep its
trend gate because heat's trajectory is monotone under the same
schedule.

The accuracy thresholds (L2, max, residual) plus history NaN/Inf
checks plus bit-identical replay collectively prove training
worked, so the trend gate is redundant where it doesn't apply.

## Phase 3 — Allen-Cahn + KdV (external reference inference)

Both trainers return `l2_error: None`. Phase 3 ships option (1) from
the previous plan: a pure-Rust `cjc_ad::pinn::pinn_mlp_eval_grid`
helper that evaluates a trained MLP at arbitrary `(x, t)` points
without needing a `GradGraph`. This unblocks full-domain L2 against
external references for all PDEs, including the closed-form
soliton solution for KdV.

### Phase 3 Helper API

```rust
pub fn pinn_mlp_eval_grid(layer_sizes: &[usize], flat_params: &[f64], inputs: &[f64]) -> Vec<f64>;
pub fn pinn_l2_max_errors(pred: &[f64], target: &[f64]) -> (f64, f64);

// Allen-Cahn: IC reference (no closed-form full solution).
pub fn allen_cahn_ic_reference(x: f64) -> f64;        // x²·cos(π·x)
pub fn allen_cahn_ic_grid(n_x: usize) -> (Vec<f64>, Vec<f64>);

// KdV: analytical 1-soliton.
pub fn kdv_soliton_reference(x: f64, t: f64, c: f64) -> f64;
pub fn kdv_reference_grid(x_min: f64, x_max: f64, n_x: usize,
                          t_min: f64, t_max: f64, n_t: usize, c: f64)
                          -> (Vec<f64>, Vec<f64>);
```

The inference helper uses Kahan-stable scalar dot products and a
canonical loop order; outputs are bit-identical to the trainer's
internal graph forward, and bit-identical across runs. Both the
test harness and bench harness compute external L2/max via this
helper.

### Phase 3 — 1D Allen-Cahn Equation

```
u_t = ε²·u_xx + u - u³            on x ∈ [-1, 1], t ∈ [0, 1]
IC:   u(x, 0) = x²·cos(π·x)
BCs:  periodic, u(-1, t) = u(1, t)
ε     = 0.01
```

**Reference**: no closed-form full space-time solution exists at
ε=0.01. The harness gates on (a) IC reproduction at t=0 against
`x²·cos(π·x)` evaluated externally on 50 uniform points, and
(b) `mean_residual`. A high-resolution implicit-FD reference for
full-domain L2 is deferred to Phase 3b — Allen-Cahn at this ε is
stiff (cubic reaction + small diffusion), so a fair reference solver
needs implicit timestepping or a spectral method, which is heavier
than the rest of the suite.

### Phase 3 — 1D KdV Equation (single soliton)

```
u_t + 6·u·u_x + u_xxx = 0         on x ∈ [-5, 5], t ∈ [0, 1]
IC:    u(x, 0) = 0.5·sech²(x/2)   (single soliton)
Exact: u(x, t) = 0.5·sech²(0.5·(x - t))   (1-soliton, c=1)
```

KdV with this IC has an exact analytical soliton solution. The
harness builds a 50×11 grid in (x, t) ∈ [-5, 5] × [0, 1] and
computes full-domain RMSE/max via `pinn_mlp_eval_grid`. KdV is
the suite's stiffest test for high-order dispersion: the `u_xxx`
term is computed via 4-point central FD on the network during
training (`(u(x+2ε) − 2u(x+ε) + 2u(x−ε) − u(x−2ε)) / (2ε³)`).

### Phase 3 Pass/Fail Thresholds

Allen-Cahn (IC reproduction at t=0 over 50 uniform x ∈ [-1,1], plus
mean residual at final params):

| Metric | Smoke (500 ep) | Convergence (2000 ep) |
|---|---|---|
| IC RMSE         | < 0.40 | < 0.05 |
| IC max abs      | < 0.85 | < 0.10 |
| Mean residual   | < 0.10 | < 0.02 |

KdV (full-domain RMSE/max over 50×11 (x,t) grid against the analytical
1-soliton, plus mean residual):

| Metric | Smoke (500 ep) | Convergence (2000 ep) |
|---|---|---|
| RMSE            | < 0.15  | < 0.02   |
| Max abs error   | < 0.40  | < 0.06   |
| Mean residual   | < 0.02  | < 1.0e-3 |

Calibration note: observed values at seed=42 are
**Allen-Cahn smoke** IC L2 ≈ 0.129 / IC max ≈ 0.280 / res ≈ 0.0265,
**Allen-Cahn conv** IC L2 ≈ 0.0097 / IC max ≈ 0.0177 / res ≈ 0.0034,
**KdV smoke** RMSE ≈ 0.0380 / max ≈ 0.118 / res ≈ 3.14e-3,
**KdV conv** RMSE ≈ 0.0052 / max ≈ 0.0160 / res ≈ 1.73e-4.
Thresholds carry ~3× headroom over observed values, matching the
Phase 1/2 convention. KdV residual headroom is wider (~6×) because
the IC matches the analytical soliton at t=0 — the trainer starts
with a near-zero residual signal, so absolute-scale variance bites
harder than relative drift.

### Files (Phase 3)

- Examples:
  [`examples/physics_ml/pinn_allen_cahn_1d.cjcl`](../../examples/physics_ml/pinn_allen_cahn_1d.cjcl),
  [`examples/physics_ml/pinn_kdv_1d.cjcl`](../../examples/physics_ml/pinn_kdv_1d.cjcl)
- Tests:
  [`tests/physics_ml/pinn_allen_cahn_1d.rs`](../../tests/physics_ml/pinn_allen_cahn_1d.rs),
  [`tests/physics_ml/pinn_kdv_1d.rs`](../../tests/physics_ml/pinn_kdv_1d.rs)
- Bench: same harness, now ships 10 benchmarks (5 PDEs × {smoke, conv})

## Next Steps

Phase 3b — wire `pinn_eval_at(final_params, layer_sizes, x, t)` as
a `cjcl` builtin so .cjcl examples can do their own external L2
gating, and write a high-resolution implicit-FD reference for
Allen-Cahn full-domain comparison. Phase 4 then layers proptest
property tests (residual decreases under more epochs, no-NaN
invariants) and Bolero fuzz on the inference helper.
