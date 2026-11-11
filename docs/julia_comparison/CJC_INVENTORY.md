# CJC-Lang inventory: AD, SciML, numerics, quantum

**Commit:** `31058a8` (branch `claude/cjc-quantum-audit-5a7f9c`, same tree as
`origin/master`). **Date:** 2026-11-09.

**Evidence labels**

| Label | Meaning |
|---|---|
| **probe** | A `.cjcl` program shows the claim; see [probes/README.md](probes/README.md) |
| **read** | I opened the cited code myself |
| **audit** | One of the four read-only sub-audits opened the code and cited it; I did not re-open it |

**Surface labels**

| Label | Meaning |
|---|---|
| **cjcl** | Reachable from `.cjcl` source |
| **crate** | Rust only |
| **stub** | The name resolves but always errors or does nothing |

How a builtin becomes reachable: a name that neither executor handles itself
falls through to `cjc_runtime::builtins::dispatch_builtin`
(`crates/cjc-eval/src/lib.rs:2867`, `crates/cjc-mir-exec/src/lib.rs:2979`).
So in practice a match arm there that returns `Ok(Some)` means **cjcl**
(audit). The PINN trainers and the `GradGraph` object methods are the
exceptions: each executor implements them separately.

Test notation: "unit" means in-crate `#[cfg(test)]`; "parity" means eval ≡ MIR
is compared by printed output; "prop"/"bolero" mean property and fuzz tests.

---

## 1. Language and numeric core

| Capability | Where | Surface | Evidence | Notes |
|---|---|---|---|---|
| Generics, traits, `impl Trait for T`, `[T;N]`, shape/const type args | `cjc-ast/src/lib.rs:151-431` | cjcl (syntax) | audit | At runtime, values are dynamic. Monomorphisation is an opt-in MIR pass (`--mir-mono`, `cjc-mir/src/monomorph.rs`) |
| Runtime numeric values | `cjc-runtime/src/value.rs:130-256` | cjcl | audit | `Int(i64)`, `Float(f64)`, `U8`, `Bf16`, `F16`, `Complex(ComplexF64)`, `Tensor` (**f64 only**), `SparseTensor`. There is no f32, i32 or Dual value |
| Tensor element type | `cjc-runtime/src/tensor.rs:74-76` (`Buffer<f64>`) | cjcl | audit | `DType` / `TypedStorage` (`tensor_dtype.rs:32-53`) exist but **no code uses them** |
| Complex scalars | `cjc-runtime/src/complex.rs:42` | cjcl | audit | `+ - * / ==` in both executors. No Complex×Float arm, no complex tensors; `rfft` returns `(re, im)` tuples |
| Operator dispatch | `cjc-eval/src/lib.rs:1552-1750` (+ MIR mirror) | cjcl | audit | Hand-written match on runtime tags. User structs cannot overload operators. A second `fn f` replaces the first |
| Multiple dispatch engine | `cjc-dispatch/src/lib.rs` | **crate, unused** | audit | Specificity ranking plus a coherence checker; only `tests/test_dispatch.rs` calls it |
| Implicit NumPy-style broadcasting | `tensor.rs:396-487` | cjcl | audit, probe (forward, P2) | Works in the forward pass. Scalar–tensor arms are asymmetric (`Tensor + Int` is an error) |
| Explicit fused elementwise ops | ADR-0027; `fused_axpy`, `fused_mul_sub`, `fused_sub_sq`, `broadcast_fma` | cjcl | audit | No automatic fusion; the one rewrite (`fusion_rewrite.rs`) is opt-in |
| Decorators | `cjc-parser/src/lib.rs:647-666`; runtime `cjc-eval/src/lib.rs:4377-4396`, `cjc-mir-exec/src/lib.rs:5858` | cjcl | audit | Only `@memoize`, `@trace` and `@nogc` mean anything. Other names are probably ignored without a warning [unverified]. User-defined decorators: no |
| Default parameters, variadics | `cjc-ast/src/lib.rs:292-300`; eval `:4351-4416`; MIR `:5829-5927` | cjcl | audit | **Already shipped**: CLAUDE.md §2 is out of date. No named arguments [unverified] |
| Macros / quoting / `@generated` | — | ✗ | audit | None |
| Units of measure | — | ✗ | audit | None |
| Elementary functions | `cjc-runtime/src/builtins.rs:869-1035` → `cjc_repro::dmath` | cjcl | audit, read (commit `0160044` is in HEAD) | fdlibm/musl ports with Payne–Hanek reduction and no FMA. The external harness claims < 1 ulp for most functions and ≤ ~2 ulp for tanh/sinh/atanh. In-repo tests compare against platform libm within 2 ulp (`dmath.rs:1753`) |
| `randn` / normal sampling | `cjc-repro/src/lib.rs:129` (`next_normal_f64`) | cjcl | **read** | **Calls the platform's `ln` and `cos`**, so the last bit depends on the OS. Feeds `Tensor.randn`, the distribution samplers, NSS and ABNG |
| bf16 / f16 conversion | `value.rs:55-61`; `f16.rs:87-139` | cjcl | audit | bf16 truncates (and a NaN can become ±Inf). f16 rounds ties away from zero, overflows at 65504 instead of 65520, and mishandles the subnormal carry |
| Wider precision (double-double, BigFloat, rational, interval) | — | ✗ | audit | Double-double exists only inside dmath's argument reduction |
| Kahan accumulator | `cjc-repro` `KahanAccumulatorF64` | crate (used inside builtins) | audit | Classic Kahan, not Neumaier |
| Binned accumulator | `BinnedAccumulatorF64` (2048 exponent bins) | used by `sum`, `mean`, `dot`, `sum_axis` | audit, **probe P5** | Plain `+` within a bin, so it is **not order-invariant** (P5: `[1,b,b]` gives 3, `[b,b,1]` gives 3+2⁻⁵¹) |
| Executor parity tests | `tests/pinn_parity.rs` (`assert_parity`), `cjcl parity` CLI | test | audit, probe | Compares printed output. `tests/fixtures/runner.rs` runs **only MIR** |

## 2. Automatic differentiation (`crates/cjc-ad`)

| Capability | Where | Surface | Evidence | Exact or FD | Notes |
|---|---|---|---|---|---|
| Forward-mode `Dual{value, deriv}` | `lib.rs:35-195` | crate | audit | exact, first order only | Scalar f64; can't be nested; no multiple partials; nothing outside `cjc-ad` uses it |
| Reverse-mode tape `GradGraph` | `lib.rs:380`; `backward` `:1409` | cjcl (`GradGraph.new()` methods; 39 `grad_graph_*` builtins via `dispatch.rs:233-623`) | audit | exact | About 40 `GradOp` variants (`lib.rs:204-364`), including fused `MlpLayer` |
| Gradient accumulation | `accumulate_grad` `lib.rs:3658` → `add_assign_unchecked` | — | audit | — | Plain `+=` in a fixed order: deterministic but not compensated |
| **Broadcast-aware backward** | `lib.rs:1448-1480` | cjcl | **read, probe P2** | — | **Missing.** `Add`/`Mul`/`Div` pass the upstream gradient through unchanged; only `MlpLayer` sums its bias gradient |
| `jacobian` | `lib.rs:2305` | cjcl (method) | audit | **exact** (one reverse sweep per output) | — |
| `hessian_diag(loss, p, eps)` | `lib.rs:2343` | cjcl (method) | **read, probe P1** | **FD, and wrong**: it never re-runs the forward pass | Returns 4 instead of 12 for p⁴ |
| `hessian(loss, p)` | `lib.rs:2392` | crate | read | **FD** (eps = 1e-5, re-runs the forward pass) | The forward re-run falls back to stale values for Softmax, LayerNorm, BatchNorm, CE, Where, Cat, Gather and Log2 (`lib.rs:2690`) |
| `double_backward` | `lib.rs:2706` | crate | audit | **FD** (diagonal only) | The name suggests AD-over-AD; the docstring says FD |
| Taylor mode / jets / nested AD | — | ✗ | audit | — | The backward pass works on concrete tensors and cannot be differentiated again |
| Custom rules (`rrule`/`frule`) | — | ✗ | audit | — | `GradOp` is a closed enum. `stop_gradient` and `grad_checkpoint` are identity functions |
| `vmap_forward` | `lib.rs:2769` | crate | audit | — | A loop over the batch |
| Gradient checks | `check_grad_finite_diff` (crate); `tests/pinn_correctness.rs:157` | test | audit | — | `tests/chess_rl_advanced/test_ad_verify.rs:167` only checks that the FD value is finite. The `grad_graph_*` proptests test **forward** values |
| `symbolic_derivative` | `builtins.rs:5760` | **stub** | audit | — | Always errors |

## 3. SciML: ODE, PDE, PINN, optimisation, linear algebra

| Capability | Where | Surface | Evidence | Notes |
|---|---|---|---|---|
| `ode_step_euler`, `ode_step_rk4`, `ode_solve_rk4` | `cjc-runtime/src/ode.rs:24,44,293` | crate | audit | RK4 combination uses Kahan |
| `ode_solve_rk45` (Dormand–Prince 5(4)) | `ode.rs:395` | crate | audit | Real embedded error estimate and step control. Not FSAL (7 RHS evaluations per step). **If the step falls below 1e-14 it silently returns a truncated trajectory** (`:520,:527`). Error norm summed with plain `+=` |
| `adjoint_solve` | `ode.rs:561` | crate | audit | **Broken**: the adjoint starts at zero, there is no way to pass dL/dy(T), and the parameter gradient is discarded. Its test bypasses the function |
| `ode_step`, `pde_step` | `builtins.rs:5728,5744` | **stub** | audit | Reserved names. They can't be built yet because builtins can't call closures (`builtins.rs:5711-5718`) |
| Stiff, symplectic, SDE, DAE, events, dense output | — | ✗ | audit | — |
| `pde_laplacian_1d`, `pde_step_diffusion` | `ode.rs:86,110` | crate | audit | 1-D explicit Euler with no CFL check |
| PINN trainers (14) | `cjc-ad/src/pinn.rs` (heat-poly :591, harmonic :735, Burgers :1891, Poisson-2D :2189, heat-NN :2438, wave :2710, Helmholtz :2946, diffusion–reaction :3134, Allen–Cahn :3320, KdV :3496, Schrödinger :3666, Navier–Stokes :3845, Burgers-2D :4060, inverse diffusion :4237) | cjcl for 10 (`pinn_train_*`, implemented separately in each executor) | audit | **All PDE derivatives use finite-difference stencils inside the graph** (default ε = 1e-4); the only exception is the polynomial heat model |
| Root finding (`bisect`, `brentq`, `newton_scalar`, `secant`) | `cjc-runtime/src/optimize.rs:51-253` | crate; `bisect` is a **stub** | audit, **probe P7** | — |
| Optimisers (GD, BFGS, L-BFGS, Nelder–Mead) | `optimize.rs:362-603` | crate | audit | Armijo line search only. Reachable: `adam_step`, `Adam.new`, `Sgd.new`, projected GD |
| `solve`, `det`, `lstsq` | `linalg.rs:369,315,428` | cjcl | audit | LU with partial pivoting; QR via Householder; `solve`'s substitution uses plain `-=` |
| LU / QR / Cholesky / inverse as values | `linalg.rs:55,128,266,804` | **crate** | audit | Only through MIR opcodes that the lowering never emits |
| `eigh` | `linalg.rs:527` | cjcl | audit, probe (correct in P3) | Householder tridiagonalisation plus implicit QR with Wilkinson shift |
| `svd` | `linalg.rs:1162` | cjcl | **probe P3** | **Computed as `eigh(AᵀA)`** although the header says Golub–Kahan. σ_min comes out as 0 for a matrix with σ_min = 5e-11. `pinv` and `matrix_rank` are built on it |
| `schur` | `linalg.rs:958` | cjcl | audit | Single real shift, so complex-conjugate eigenvalue pairs may converge badly [unverified] |
| `cond`, `matrix_exp` (Padé 13), norms, `kron` | `linalg.rs:908,1394` | cjcl | audit | — |
| Matmul accumulation | `tensor.rs:~953-968` | cjcl | audit | Kahan when every dimension is < 64; **plain tiled sum when any dimension is ≥ 64** |
| Sparse CSR/COO, `matvec` | `sparse.rs` | cjcl ops, but **no way to build a `SparseTensor` from `.cjcl`** | audit | — |
| CG / GMRES / BiCGSTAB | `sparse_solvers.rs:81,165,364` | crate | audit | Binned sums; no preconditioners |
| Lanczos / Arnoldi | `sparse_eigen.rs:38,258` | cjcl (`sparse_lanczos`, `sparse_arnoldi`), but no way to build the input | audit | Lanczos has **no reorthogonalisation** |
| Quadrature (`trapz`, `simpson`, `cumtrapz`) | `integrate.rs` | cjcl | audit | Kahan. No Gauss or adaptive quadrature |
| `erf`, `erfc`, `normal_cdf` | `distributions.rs:183,195,235` | cjcl | **probe P4** | A&S 7.1.26: relative error 2.7e-7 near 0, **6.7e-3 at erfc(5)** |
| `ln_gamma` | `distributions.rs:17` | crate | audit | Lanczos g = 7. No `gamma`/`lgamma` builtin; no Bessel or zeta functions |

## 4. Quantum (`crates/cjc-quantum`)

This extends [SURFACE_AUDIT.md](../quantum_simulation_research_stack/SURFACE_AUDIT.md),
which remains the authority for the full builtin list. Caps (verified earlier):
dense 26 qubits, density 14, MPS 100,000, stabilizer 32,768.

| Capability | Where | Surface | Evidence | Notes |
|---|---|---|---|---|
| Kraus channels | `density.rs:363,398` | crate (general); cjcl for 3 fixed 1-qubit channels | audit | No CPTP check; panics if dimensions mismatch |
| Lindblad / master equation / MC wavefunction / steady state | — | ✗ | audit (`lib.rs:30-31` says so) | — |
| Time-dependent Hamiltonians | — | ✗ | audit | — |
| MPS | `mps.rs:440` | cjcl | audit, plus the MPS fixes in this session | The general 2-qubit gate path is private |
| DMRG (two-site) | `dmrg.rs:729`; Hamiltonian enum `{Ising, Heisenberg}` `:42` | cjcl (`dmrg_ising`, `dmrg_heisenberg`; energy only) | **probe P6** | **About 8% above the exact ground state at n = 4, 6, 8**. Tests check only 2 sites (tolerance 0.1) and `E < 0` at 4 |
| MPO / OpSum, quantum-number symmetry, TEBD/TDVP, MPS entanglement entropy | — | ✗ | audit | — |
| Circuit IR | `circuit.rs:21-30` | cjcl | audit | A flat `Vec<Gate>` over 13 gates. No block IR, no circuit adjoint, no optimisation passes |
| QASM 2.0 import/export | `qasm.rs` | cjcl | read (this session) | `if` is rejected |
| Adjoint differentiation | `adjoint.rs:51` | **crate** | audit | Diagonal observables only; gradients only for Rx/Ry/Rz. Checked against FD and parameter shift |
| Parameter shift | `wirtinger.rs:194`; `vqe.rs:456,586`; `qaoa.rs:401` | inside closed-loop builtins only | audit | — |
| Wirtinger dual | `wirtinger.rs:32` | crate, **unused** | audit | — |
| FD gradients in QML | `qml.rs:323-348` (ε = 1e-4) | inside `qml_train` | audit | — |
| VQE / QAOA | `vqe.rs:484,638`; `qaoa.rs:457` | cjcl | audit | Spin chains only. No molecular VQE |
| Jordan–Wigner, H₂/LiH Hamiltonians | `fermion.rs:227,320` | cjcl (builder) | audit | `jw_two_body` is **untested**. No Bravyi–Kitaev |
| Trotter | `trotter.rs:31,120` | cjcl | audit | Orders 1 and 2 only; uses only `coeff.re`; tested against 1000-step Trotter, not an exact exponential |
| Stabilizer tableau, QEC | `stabilizer.rs:95`; `qec.rs` | cjcl | audit | No CZ, S†, reset or Pauli frames; repetition-code decoder only |
| `fidelity` | `density.rs:469` | crate | **read** | Computes Tr(ρσ), which equals Uhlmann fidelity only if one state is pure (its docstring says so) |
| Partial trace, purity, von Neumann entropy | `density.rs:514,444,490` | crate / cjcl / cjcl | audit | `trace`, `purity`, `fidelity`, `partial_trace` and `apply_kraus` use plain sums, which contradicts the module's "Kahan everywhere" doc |
| Gate trigonometry | `gates.rs:212-226` and others | — | audit, read | dmath. Angles from `.cjcl` come through dmath-based builtins since `0160044` |

## 5. Corrections to existing docs found during this inventory

| Doc | Says | Code says |
|---|---|---|
| `CLAUDE.md` §2 "Default parameters" | parser does not accept `=` after a type | implemented in both executors |
| `CLAUDE.md` §5 "Decorators" | a feature to implement | parser and `@memoize`/`@trace` exist; user-defined decorators do not |
| `crates/cjc-quantum/src/lib.rs:21-23`; MISSING_FEATURES P1-8 | `.cjcl` math still uses platform libm | dmath since `0160044` (except `randn`, §1) |
| `linalg.rs:1144` | "SVD via Golub-Kahan Bidiagonalization" | `eigh(AᵀA)` |
| `linalg.rs:5-6`, `accumulator` header | every reduction is Kahan/binned; binned is "commutative + associative" | `solve`'s substitution and matmul ≥ 64 use plain sums; within a bin the sum is not associative (P5) |
| `cjc-ad/src/lib.rs:3-5` | "Supports `grad()`, `jacobian()`" | no free `grad()` function |
| `docs/AD_MIR_COVERAGE_AUDIT.md` | the proptests are strong evidence of gradient correctness; PINN FD ε = 1e-3 | the proptests check forward values; the Rust PINN default is ε = 1e-4 |
| `cjc-runtime/src/lib.rs:140`, `ode.rs:1-5` | only Euler and RK4; adaptive solvers will be library code | RK45 exists in Rust (crate-only) |
</content>
</invoke>
