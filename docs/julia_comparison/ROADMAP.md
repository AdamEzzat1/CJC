# Prioritised roadmap

**Priority**

| Priority | Meaning |
|---|---|
| **P0** | Correctness: CJC gives a wrong answer today, labels a finite-difference result as exact, or gives different results on different OSes |
| **P1** | Unlocks SciML or quantum work, or removes a major accuracy limit |
| **P2** | Valuable but not urgent |
| **Won't-do** | Rejected, with the reason given |

**Scores**

| Column | Values |
|---|---|
| **Acc** (accuracy impact) | H / M / L |
| **Unlock** (SciML/quantum capability it enables) | H / M / L |
| **Det risk** (risk to determinism) | H / M / L / none |
| **Effort** | S / M / L |

Evidence for each item is in [probes/](probes/) and
[CJC_INVENTORY.md](CJC_INVENTORY.md). The issue texts are in [ISSUES.md](ISSUES.md).

## P0: correctness

| # | Gap | Evidence | Acc | Unlock | Det risk | Effort | Crates | Issue |
|---|---|---|---|---|---|---|---|---|
| P0-1 | `hessian_diag` never re-runs the forward pass, so it is wrong for any composed loss | probe P1 (4 vs 12) | H | M | none | S | cjc-ad | I-01 |
| P0-2 | The backward pass does not sum broadcast gradients back to the input shape; `examples/01_mlp_xor.cjcl` silently grows its bias to [4,4] | probe P2 plus the example run | H | H | L (the reduction must be binned and in a fixed order) | S–M | cjc-ad | I-02 |
| P0-3 | `svd` is computed as `eigh(AᵀA)`: small singular values are lost and reported as exactly 0; `pinv` and `matrix_rank` inherit this | probe P3 | H | M | L | M | cjc-runtime | I-03 |
| P0-4 | Two-site DMRG converges to an energy about 8% above the ground state for n ≥ 4 | probe P6 (−5.950 vs −6.464) | H | H | none | M (diagnose first) | cjc-quantum | I-04 |
| P0-5 | The binned accumulator adds values within a bin with plain `+`, so `sum` depends on input order | probe P5 | M (1 ulp) | — | **H**: the determinism contract promises independence from chunking and thread count | M | cjc-repro (+ every caller) | I-05 |
| P0-6 | `randn` and normal samplers use platform `ln`/`cos`, so they can differ across OSes | read: `cjc-repro/src/lib.rs:129` | L per sample, H for cross-OS reproducibility | — | fixes a cross-OS determinism hole | S | cjc-repro | I-06 |
| P0-7 | Rust `hessian` and `double_backward` are finite differences; `double_backward`'s name implies exact AD; `reforward` silently leaves stale values for 8 op kinds | read (`lib.rs:2392,2706,2690`) | M | — | none | S (rename and document, plus an error on unsupported ops); the exact replacement is P1-1 | cjc-ad | I-07 |
| P0-8 | `erf`/`erfc`/`normal_cdf` reach 6.7e-3 relative error in the tail | probe P4 | H for statistics | M | none (built on dmath) | S | cjc-runtime | I-08 |
| P0-9 | `ode_solve_rk45` silently returns a truncated trajectory when the step underflows; `adjoint_solve` starts from a zero adjoint and throws away the parameter gradient | audit (`ode.rs:520,527,582,600`) | H (crate-only today) | — | none | S | cjc-runtime | I-09 |

**Hypothesis to test first (not a finding):** P0-6 may explain the open
cross-platform canary failures (`pinn_chain_head_canary_locked`,
`v2_6_*_canary_locked`, `primitive_master_hash_golden`; see task chip
`task_d04e49df`).
- Those canaries hash models whose initial weights come from `randn`.
- **Test:** route `next_normal_f64` through dmath, re-lock the canaries on Windows, and see whether ubuntu and macOS CI agree.
- **If they still differ, the cause is elsewhere.** Record which canaries remain.

## P1: capability and accuracy unlocks

| # | Gap | Julia reference | Acc | Unlock | Det risk | Effort | Crates | ADR |
|---|---|---|---|---|---|---|---|---|
| P1-1 | Exact higher-order AD: forward-over-reverse Hessian-vector products and full Hessians; Taylor-mode jets for the n-th derivative | ForwardDiff nesting, TaylorDiff, NeuralPDE `EnzymeForwardDerivative` | H | **H** (exact PINN residuals) | L | L | cjc-ad, executors (new `GradGraph` methods) | **ADR-B** |
| P1-2 | Builtins that call back into the executor (closures as arguments to runtime functions) | — (native in Julia) | — | **H**: needed for ODE solvers, root finding, quadrature and custom AD rules from `.cjcl` | M: both executors need one shared callback trait | M | cjc-runtime, cjc-eval, cjc-mir-exec | **ADR-C** |
| P1-3 | ODE family reachable from `.cjcl`: RK45 with FSAL, dense output and events; Rosenbrock (Rodas5P-class) and BDF for stiff problems; Velocity-Verlet; ODE adjoint (Interpolating/Gauss-style with a fixed checkpoint schedule) | OrdinaryDiffEq, SciMLSensitivity | H | H | M (Newton iteration policy, deterministic LU) | L | cjc-runtime (+ Bastion wrappers) | **ADR-D** |
| P1-4 | A `DetScalar` trait with a double-double (`DD`) type, complex tensors, and dtype-generic kernels | DoubleFloats, generic linear algebra | H | M | M (memory layout) | L | cjc-runtime, cjc-repro | **ADR-E** |
| P1-5 | Custom AD rules (`g.custom(fwd, bwd, inputs)`) | ChainRules | H | M | L | M (after ADR-C) | cjc-ad, executors | under ADR-C |
| P1-6 | Open quantum systems: Lindblad `mesolve` (deterministic integrator), MC-wavefunction trajectories with seeded jumps averaged in a fixed order, steady state | QuantumToolbox, QuantumOptics | — | H | L | M–L | cjc-quantum | **ADR-G** |
| P1-7 | Quantum AD from `.cjcl`: adjoint method for any Hermitian observable (a Pauli sum) and every parametrised gate, uncomputing instead of storing a tape | Yao `expect'` | M | H | none | M | cjc-quantum | — |
| P1-8 | General MPO (OpSum-like builder) plus DMRG validated against exact diagonalisation, with truncation error reported | ITensorMPS | H | H | none | M | cjc-quantum | **ADR-H** |
| P1-9 | Expose LU/QR/Cholesky as `.cjcl` values; build `SparseTensor` from triplets in `.cjcl`; wire CG/GMRES/BiCGSTAB with Jacobi/ILU(0) preconditioners | LinearSolve | M | M | none | M | cjc-runtime | — |
| P1-10 | Lanczos with full or selective reorthogonalisation | KrylovKit/Arpack [verify] | M | M | none | S | cjc-runtime | — |
| P1-11 | Uhlmann fidelity, plus making the plain sums in the density module compensated | QuantumOptics | M | L | none | S | cjc-quantum | — |

## P2

| # | Gap | Notes |
|---|---|---|
| P2-1 | IEEE-correct f16/bf16 conversion (round to nearest even, overflow threshold, carry from subnormals, NaN preservation) | A bug class, but reaching it requires half precision on purpose |
| P2-2 | `bisect` dead stub (probe P7): wire it after ADR-C, or delete the arm | Trivial |
| P2-3 | Gauss–Kronrod adaptive quadrature; `gamma`/`lgamma`/Bessel builtins | Deterministic ports |
| P2-4 | Symbolic differentiation library (`Sym`, `diff`, `simplify`, `lambdify`) | Bastion; replaces the `symbolic_derivative` stub |
| P2-5 | Units (`Quantity` record) | Bastion |
| P2-6 | Interval type (round-to-nearest plus 1-ulp widening) | Bastion; not tight |
| P2-7 | Stabilizer: CZ, S†, reset, Pauli frames; MWPM/union-find decoder | QuantumClifford parity |
| P2-8 | Block circuit IR | Yao parity; a nicer API, not an accuracy gain |
| P2-9 | Higher-order (Suzuki) Trotter, and an exact-exponential test oracle | — |
| P2-10 | `schur` with a double shift (Francis) | Complex-conjugate pairs [unverified failure] |
| P2-11 | Fixing doc drift (CJC_INVENTORY §5) | Includes CLAUDE.md §2 and §5 |

## Won't-do

| Item | Why |
|---|---|
| Copying Julia's `Base.Math` kernels | They use FMA, and CJC forbids FMA. dmath (fdlibm/musl, no FMA) is already cross-OS bit-identical |
| Calling LAPACK/BLAS for accuracy | Adds a dependency, and threaded BLAS is not deterministic. Port the algorithm instead (one-sided Jacobi SVD, Golub–Kahan) |
| BigFloat via MPFR | Adds a dependency. Double-double covers most of the "check in higher precision" need; Julia stays the BigFloat oracle offline |
| A macro system / `@generated` | Would add a pipeline stage, and expansion order would become a new source of nondeterminism. LANGUAGE_MECHANISMS §3 |
| StaticArrays-style types | Their benefit comes from JIT specialisation; no accuracy gain |
| Dot-broadcast fusion syntax | No accuracy gain, since Julia's broadcast does not contract to FMA either. ADR-0027 already chose explicit fused builtins |
| User-level multiple dispatch (wiring `cjc-dispatch` in) | Changes how every call resolves; no accuracy need. Revisit only after ADR-E |
| ModelingToolkit-style symbolic index reduction in the core | Belongs to a library; the core only needs index-1 mass-matrix support in ADR-D |
| GPU, QPU or Rydberg hardware paths | Out of scope; Bloqade.jl itself is in maintenance mode |

## Proposed ADRs

The ADR numbers continue from ADR-0048 in `CJC-Lang_Obsidian_Vault/13_ADRs/`.
Each is a *proposal*: the number is reserved only when the ADR is written.

| Letter | Proposed number | Title | Decision to make |
|---|---|---|---|
| ADR-A | 0049 | Order-invariant binned summation | Make each bin exact: per-bin TwoSum with a carry, or a fixed-point long accumulator (ExBLAS-style), so the sum is independent of order and chunking. Measure the cost on the `sum`/`dot` hot paths |
| ADR-B | 0050 | Exact higher-order AD | Forward-over-reverse HVP plus a `Jet<N>` Taylor type; remove FD from the `hessian*` APIs or rename them `*_fd` |
| ADR-C | 0051 | Runtime callbacks into executors | One `CallbackHost` trait implemented by both executors, so runtime builtins (ODE, root finding, quadrature, custom AD rules) can call `.cjcl` closures. Calls happen in a deterministic order, with no re-entrancy into the GC from NoGC paths |
| ADR-D | 0052 | Deterministic ODE solver family | Tableau-driven RK (DP5 with FSAL and dense output); Rodas5P-class Rosenbrock and BDF with a fixed Newton policy; symplectic; event location; adjoint with a fixed checkpoint schedule. All norms compensated; `powf` via dmath |
| ADR-E | 0053 | `DetScalar` trait and typed tensors | The trait, the `DD` type, complex tensors; `Tensor` storage gains a dtype tag; reductions become generic over an accumulator |
| ADR-F | amend 0046 | dmath for RNG transforms | Box–Muller via dmath; document that the bits of the random stream change |
| ADR-G | 0054 | Open-system quantum simulation | Lindblad via a deterministic integrator on vectorised ρ; MCWF with seeded jump records |
| ADR-H | 0055 | General MPO and the DMRG correctness gate | An OpSum builder; DMRG must match exact diagonalisation to 1e-10 for n ≤ 14 at full rank |
</content>
</invoke>
