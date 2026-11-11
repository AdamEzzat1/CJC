# CJC-Lang vs Julia: feature and accuracy matrix

The legend is the one used in [EXTERNAL_COMPARISON.md](../quantum_simulation_research_stack/EXTERNAL_COMPARISON.md):

| Mark | Meaning |
|---|---|
| ✓ | Implemented |
| ◐ | Partial or limited |
| ✗ | Absent |
| — | Not applicable |
| ? | Not established |

- **CJC column:** ✓ means exposed to `.cjcl`. A feature that exists only in the crate is ◐ and says "crate". ✓\* means exposed but **wrong**, with a probe to show it. Evidence is in [CJC_INVENTORY.md](CJC_INVENTORY.md).
- **Julia column:** sources and versions are in [JULIA_SURVEY.md](JULIA_SURVEY.md).
- **Det-compatible?** asks whether CJC can adopt the *mechanism* without breaking bit-identical output, executor parity or the zero-dependency rule.
  - **yes:** fits as is.
  - **yes, with cost:** possible; the cost is stated.
  - **no:** do not adopt; the reason is stated.

> **No performance claims appear in this document.** Accuracy entries marked
> "probe" were measured with the scripts in [probes/](probes/). Everything
> else is a *planned measurement* ([ACCURACY_HARNESS_PLAN.md](ACCURACY_HARNESS_PLAN.md)).

## A. Language and numeric core

| Capability | CJC | Julia | Julia's accuracy mechanism | Det-compatible? |
|---|---|---|---|---|
| Elementary functions that give the same bits on every OS | ✓ dmath (fdlibm/musl, no FMA) | ◐ `Base.Math` is pure Julia, but its kernels use `fma`, and `muladd` can differ between machines | Table-driven methods and polynomial kernels | CJC is **ahead on reproducibility** here. Do not copy Julia's kernels, which rely on FMA |
| The same bits for random normals | ✓\* `randn` calls the platform's `ln`/`cos` (read, `cjc-repro/src/lib.rs:129`) | — (Julia's Random has its own algorithm) | — | yes: switch to dmath; this changes stream bits |
| Published ulp error per function | ◐ claimed in an external harness; in-repo test only checks ≤ 2 ulp against libm | ✗ no official table [verify] | — | yes: harness slice S1 |
| Special functions (erf, gamma, Bessel) | ✓\* erf/erfc relative error up to 6.7e-3 (probe P4); no gamma/Bessel builtins | ✓ SpecialFunctions.jl [version: verify] | Minimax rational approximations, near ulp-level accuracy | yes: port a deterministic implementation (e.g. fdlibm `s_erf.c`), built on dmath |
| Order-invariant summation | ✓\* binned is not associative within a bin (probe P5) | ◐ `sum` uses pairwise summation, which depends on order; `Base.sum_kbn` was removed from Base [verify] | Pairwise summation | yes: exact per-bin accumulation (§ROADMAP P0-5) |
| Double-double / multiple precision | ✗ | ✓ DoubleFloats, MultiFloats, BigFloat | Error-free transformations; MPFR | **yes** for double-double (pure ±, ×, no FMA: TwoSum plus Dekker TwoProd); BigFloat would add an MPFR dependency → **no**. A pure-Rust arbitrary-precision type is P2 at most |
| Interval arithmetic | ✗ | ✓ IntervalArithmetic (IEEE 1788) | Directed rounding, correctly rounded libm | yes, with cost: emulate directed rounding without the FPU rounding mode (round-to-nearest plus a 1-ulp widening using dmath bounds). Not tight. P2 |
| Uncertainty propagation | ✗ | ✓ Measurements.jl | Linear propagation that tracks correlations | yes: a library (Bastion) type |
| Generic numeric code over number types | ◐ dynamic `Value` tags; `Tensor` is f64 only | ✓ multiple dispatch with specialisation | One routine runs on any `<:Number` | yes, with cost: see LANGUAGE_MECHANISMS §1 |
| Complex tensors | ✗ (scalar `Complex` only) | ✓ | — | yes |
| f16/bf16 conversion that follows IEEE | ✓\* truncating / ties-away / wrong overflow threshold (audit) | ✓ round-to-nearest-even | — | yes: a bug fix |
| Units of measure | ✗ | ✓ Unitful | Dimensions in the type system | yes, in a library: LANGUAGE_MECHANISMS §5 |

## B. Automatic differentiation

| Capability | CJC | Julia | Julia's accuracy mechanism | Det-compatible? |
|---|---|---|---|---|
| First-order reverse mode | ✓ `GradGraph` (cjcl) | ✓ Zygote, Enzyme, Mooncake, ReverseDiff | Exact | — |
| Forward mode | ◐ crate `Dual` (scalar, not nestable) | ✓ ForwardDiff (chunked, tagged) | Generic dual numbers; tags prevent perturbation confusion | yes |
| Exact Jacobian | ✓ `jacobian` (reverse, one sweep per row) | ✓ | — | — |
| Exact Hessian | ✗ **FD only**; the cjcl `hessian_diag` is also **wrong** (probe P1) | ✓ ForwardDiff nesting; forward-over-reverse; Mooncake/Enzyme | Nested duals or forward-over-reverse | yes: a forward-over-reverse tape is deterministic |
| Derivatives of order 3 and above (PINNs) | ✗ | ✓ TaylorDiff (alpha), TaylorSeries; NeuralPDE `EnzymeForwardDerivative` (order ≤ 4) | Taylor mode (truncated polynomials) | **yes**: jet arithmetic is a fixed order of ± and ×; coefficient recurrences use dmath |
| Broadcast-correct backward pass | ✗\* gradients are not reduced to the input shape (probe P2) | ✓ (ChainRules `unbroadcast`) | Reduce back to the input shape | yes: a fixed-order binned reduction |
| Custom derivative rules | ✗ | ✓ ChainRules `rrule`/`frule` | Hand-written, numerically stable adjoints (e.g. for `logsumexp`) | yes: LANGUAGE_MECHANISMS §6 |
| Gradients through mutation and control flow | ◐ the tape records whatever ran (eager) | ✓ Enzyme, Mooncake | — | — |
| AD over complex numbers | ◐ Wirtinger dual (crate, unused) | ✓ ChainRules complex conventions | — | yes |
| Gradient-check utility | ◐ crate `check_grad_finite_diff`; one test asserts finiteness only | ✓ e.g. Yao `gradcheck`, FiniteDifferences.jl | Richardson extrapolation of FD | yes |

## C. SciML

| Capability | CJC | Julia | Julia's accuracy mechanism | Det-compatible? |
|---|---|---|---|---|
| ODE solving reachable from the language | ✗ (`ode_step`/`pde_step` are stubs; RK4/RK45 crate-only) | ✓ | — | yes, with cost: builtins must be able to call closures (ROADMAP ADR-C) |
| Adaptive non-stiff RK | ◐ crate RK45 (not FSAL; returns a truncated trajectory silently) | ✓ Tsit5, Vern6–9, Feagin | Embedded error estimators, PI step control, high-order interpolants | yes: step control is deterministic if `powf` comes from dmath and the norms are compensated |
| Stiff solvers | ✗ | ✓ Rosenbrock/Rodas, BDF/FBDF, Radau, SDIRK | L-stable implicit methods; Newton with AD Jacobians | yes, with cost: the Newton iteration needs a fixed policy (iteration cap, deterministic LU, no threaded BLAS) |
| Symplectic integrators | ✗ | ✓ Verlet, McAte, KahanLi, RKN | Energy preservation over long runs | yes (cheap) |
| SDE | ✗ | ✓ (StochasticDiffEq sublibrary) | Adaptive strong-order schemes | yes: seeded SplitMix64 Brownian increments |
| DAE / index reduction | ✗ | ✓ (Rodas with mass matrices, IDA, DFBDF; ModelingToolkit Pantelides) | Symbolic structural analysis | ◐ index-1 mass-matrix Rosenbrock: yes. Symbolic index reduction: that is ModelingToolkit's scope → Won't-do in the core |
| Events / callbacks | ✗ | ✓ | Root finding on interpolants | yes |
| ODE adjoint sensitivities | ✗\* crate `adjoint_solve` is broken (audit) | ✓ Backsolve, Interpolating, Quadrature, Gauss | Continuous adjoints with checkpointing | yes: a fixed checkpoint schedule |
| PINN derivatives | ◐ FD stencils for all trainers | ◐ NeuralPDE **also defaults to FD**; exact mode is opt-in | Nested forward-mode AD | yes, via Taylor mode (ROADMAP P1) |
| Nonlinear solvers | ◐ crate scalar root finders; the `bisect` builtin is dead (probe P7) | ✓ NonlinearSolve (Newton, TR, LM, bracketing) | Globalisation; guaranteed brackets | yes |
| Optimisers | ◐ Adam/SGD (cjcl); BFGS/L-BFGS/NM crate-only | ✓ Optimization.jl | Wolfe line searches | yes |
| Dense linear algebra | ✓ `solve`/`det`/`lstsq`/`eigh`/`cond`/`expm`; ✓\* `svd` (probe P3); ◐ LU/QR/Cholesky crate-only | ✓ LAPACK and pure-Julia generic versions | Backward-stable algorithms (Golub–Kahan SVD, pivoted QR) | yes: one-sided Jacobi SVD is deterministic and accurate (the MPS SVD in cjc-quantum already does it). Calling LAPACK → **no** (dependency, threads) |
| Sparse direct / iterative | ◐ CG/GMRES/BiCGSTAB crate-only, no preconditioners; no way to build sparse input from cjcl | ✓ LinearSolve (KLU, UMFPACK, Krylov, preconditioners) | Preconditioning | yes |
| Sparse eigen | ◐ Lanczos (no reorthogonalisation), Arnoldi | ✓ KrylovKit/Arpack [verify package choice] | Implicit restarts; reorthogonalisation | yes |
| Quadrature | ◐ trapz/simpson | ✓ QuadGK, HCubature [verify] | Adaptive Gauss–Kronrod with error estimates | yes |

## D. Quantum

| Capability | CJC | Julia | Julia's accuracy mechanism | Det-compatible? |
|---|---|---|---|---|
| Dense statevector | ✓ ≤ 26 qubits | ✓ Yao | — | — |
| Block circuit IR (chain/put/control/dagger) | ✗ flat gate list | ✓ Yao | Composable, symbolic | yes |
| AD through circuits for any observable | ◐ crate adjoint (diagonal observables, Rx/Ry/Rz only) | ✓ Yao `expect'` (uncomputes instead of storing a tape) | Exploits reversibility | yes: the same uncompute method is deterministic |
| Parameter shift | ◐ inside closed-loop builtins | ✓ via Yao or by hand | — | yes |
| Density matrix | ✓ ≤ 14 qubits | ✓ QuantumOptics, QuantumToolbox | — | — |
| Lindblad master equation | ✗ | ✓ `mesolve`, `master` | Adaptive DiffEq integrators | yes: fixed-step or deterministic adaptive integrator |
| MC wavefunction trajectories | ✗ | ✓ `mcsolve`, `mcwf` | Jump times found by root finding | yes: seeded jumps; trajectories averaged in a fixed order |
| Steady state | ✗ | ✓ | Direct or iterative sparse solves | yes |
| Fidelity of mixed states | ◐ crate, Tr(ρσ) (read) | ✓ Uhlmann | Matrix square root | yes (via eigh) |
| MPS / DMRG | ✓\* DMRG is about 8% off at n ≥ 4 (probe P6); only Ising/Heisenberg | ✓ ITensorMPS | Variational sweeps; truncation-error control; Krylov expansion | yes |
| General MPO (OpSum) | ✗ | ✓ | Automatic MPO compression | yes |
| Quantum-number conserving tensors | ✗ | ✓ ITensors | Block sparsity | yes, with cost: large (L) |
| TDVP / TEBD | ✗ | ✓ | — | yes |
| Stabilizer tableau | ✓ CHP | ✓ QuantumClifford (destabilisers, Pauli frames, graph states) | Exact GF(2) arithmetic | yes; CJC is also exact |
| QEC decoders | ◐ repetition only | ✓ QuantumClifford ECC ecosystem [verify which decoders] | — | yes |
| Rydberg / analog | ✗ | ◐ Bloqade.jl (maintenance) | — | yes, but low priority |
| Higher-order Trotter | ◐ orders 1–2 | ✓ (through Yao/QuantumOptics composition) [verify] | Suzuki recursion | yes |
| QPU execution | — | Bloqade can submit to Aquila | — | out of scope (not scored) |

## Reading the matrix

1. **Where CJC is wrong, and not just missing something, is the first priority.**
   Six entries have probes showing a wrong answer reachable from `.cjcl`
   (P1–P6). None of them needs anything from Julia to fix.
2. **Where Julia is ahead, its advantage usually comes from *generic types plus
   one more mechanism*:** Taylor mode, adaptive error control, implicit
   methods. Nearly all of those mechanisms are deterministic in themselves.
   What CJC cannot copy is the *infrastructure* Julia uses to get them cheaply:
   JIT specialisation, FMA, LAPACK/BLAS, threads.
3. **CJC is ahead on cross-OS bit reproducibility:** dmath with no FMA, golden
   hashes checked on Windows, Linux and macOS in CI. The `randn` libm call is
   the last known hole.
</content>
</invoke>
