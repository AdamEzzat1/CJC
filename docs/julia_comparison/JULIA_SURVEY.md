# Julia survey: the packages that set the bar

**Sources accessed:** 2026-11-09. Versions and dates come from the GitHub
Releases API (`gh api`), checked against `JuliaRegistries/General` `Versions.toml`.
Capability notes come from READMEs, the docs, and source at the tagged version.
Some docs sites (docs.sciml.ai, juliadiff.org) refused the summarising fetch
tool, so they were read with `curl`. Anything not confirmed is marked
**[verify]**. All descriptions are paraphrases.

**Local Julia:** `julia 1.10.13` (the LTS line) is on this machine's PATH. The
current stable release is 1.13.3. The harness pins its own version; see
ACCURACY_HARNESS_PLAN §2.

> **Monorepo note.** OrdinaryDiffEq, ModelingToolkit, Lux, Optimization,
> NonlinearSolve and DifferentiationInterface are now monorepos, and their
> newest GitHub "release" is often a sub-package. The versions below are the
> umbrella package's own tags.

## 1. Language core and number types

| Package | Latest (date) | Status | What matters for accuracy | Caveat for a deterministic CPU target | Source |
|---|---|---|---|---|---|
| Julia | 1.13.3 (2026-10-26) | active | Generic code is specialised per argument type, so one routine runs on Float64, BigFloat, Double64, Interval, Dual, Measurement or Complex | Relies on JIT specialisation | https://github.com/JuliaLang/julia/releases |
| `Base.Math` | ships with 1.13.3 | active | Written in Julia, with no `ccall` into libm in `base/math.jl`. Trig and inverse trig are translated from openlibm, `cbrt` and hyperbolics come from FDLIBM, and `log` uses Tang's table method. **No published ulp table [verify]** | **The `exp`/`log` kernels use `fma`**, and the docs say `muladd` may give different results on different machines | https://github.com/JuliaLang/julia/tree/v1.13.3/base/special |
| BigFloat (MPFR) | in Base | active | Arbitrary precision, correctly rounded; precision and rounding mode can be set | Heap allocation | https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/ |
| DoubleFloats.jl | 1.12.0 (2026-10-23) | active | `Double64`: about 106 significand bits; the README reports about 1e-31 relative error for elementary functions on [0, 1] | Same exponent range as Float64 | https://github.com/JuliaMath/DoubleFloats.jl |
| MultiFloats.jl | 3.3.2 (2026-09-03) | active | `Float64x2`…`x4`, branch-free renormalisation; multiplication is commutative from v3 | Needs round-to-nearest; treats ±Inf as NaN; transcendentals go through MPFR | https://github.com/dzhang314/MultiFloats.jl |
| IntervalArithmetic.jl | 1.0.13 (2026-11-08) | active | The README says it complies with IEEE 1788-2015; rigorous enclosures via directed rounding (CRlibm, CoreMath, RoundingEmulator) | Breaks if subnormals are flushed to zero; `configure` triggers recompilation; the `:fast` matmul is threaded | https://github.com/JuliaIntervals/IntervalArithmetic.jl |
| Measurements.jl | 2.14.1 (2025-10-18) | active | First-order error propagation that tracks correlations (`x - x` has zero uncertainty) | — | https://github.com/JuliaPhysics/Measurements.jl |
| StaticArrays.jl | 1.9.24 (2026-10-21) | active | Fixed-size stack arrays, unrolled kernels | Its whole value comes from specialising on the size | https://github.com/JuliaArrays/StaticArrays.jl |
| Unitful.jl | 1.29.0 (2026-09-04) | active (moved to JuliaPhysics) | Dimensional correctness checked by the type system (not numerical accuracy) | Resolved at compile time | https://github.com/JuliaPhysics/Unitful.jl |

## 2. Automatic differentiation

| Package | Latest (date) | Status | Accuracy mechanism | Caveat | Source |
|---|---|---|---|---|---|
| ForwardDiff.jl | 1.4.7 (2026-11-02) | active | Generic `Dual{Tag}`. Tags stop perturbation confusion in nested derivatives (`InvalidTagException`). Chunked partials; an opt-in NaN-safe mode. Exact Hessians by nesting forward over forward | Chunk size chosen per call | https://github.com/JuliaDiff/ForwardDiff.jl |
| ReverseDiff.jl | 1.18.8 (2026-11-09) | maintained (released on the access date); Lux lists it as tier IV | Tape-based reverse mode; supports nested derivatives | A compiled tape is only valid if control flow does not branch | https://github.com/JuliaDiff/ReverseDiff.jl |
| Zygote.jl | 0.7.16 (2026-11-06) | active, no deprecation. The view that it is "being superseded by Enzyme/Mooncake" is **community opinion, not official [verify]** | Source-to-source reverse mode over ChainRules; no mutation | Hooks into the compiler | https://github.com/FluxML/Zygote.jl |
| Enzyme.jl | 0.13.219 (2026-11-07) | very active | AD on optimised LLVM IR; forward and reverse mode; mutation | LLVM/JIT; GPU; thread determinism **[verify]** | https://github.com/EnzymeAD/Enzyme.jl |
| Mooncake.jl | 0.5.69 (2026-11-07) | active (repo moved to `chalk-lab`) | Reverse and forward mode on Julia IR; Hessians; a correctness-first project policy | Caches tied to input types | https://github.com/chalk-lab/Mooncake.jl |
| TaylorDiff.jl | 0.3.5 (2025-11-27) | **"early alpha"** per its README | Taylor mode: truncated polynomials give the n-th derivative at a cost linear in n, so PINN-style high derivatives are exact | Order is a type parameter | https://github.com/JuliaDiff/TaylorDiff.jl |
| TaylorSeries.jl | 0.23.3 (2026-10-23) | active | Taylor series in one or many variables over generic coefficients (BigFloat, Interval); the basis of Taylor-method ODE integrators | — | https://github.com/JuliaDiff/TaylorSeries.jl |
| ChainRules / ChainRulesCore | 1.74.13 (2026-11-09) / 1.26.8 (2026-10-30) | active | `rrule`/`frule`: hand-written, numerically stable adjoints shared by many AD systems | — | https://github.com/JuliaDiff/ChainRules.jl |
| DifferentiationInterface.jl | 0.7.22 (2026-11-06) | active | One API over many AD backends | — | https://github.com/JuliaDiff/DifferentiationInterface.jl |

## 3. SciML

| Package | Latest (date) | Status | Accuracy mechanism | Caveat | Source |
|---|---|---|---|---|---|
| DifferentialEquations.jl | 8.1.1 (2026-08-27) | active umbrella | ODE, SDE, DDE, DAE, jump and BVP problems; callbacks and events | — | https://github.com/SciML/DifferentialEquations.jl |
| OrdinaryDiffEq.jl | 7.8.1 (2026-08-26) | active monorepo | Adaptive error control. **Stiff:** Rosenbrock23, Rodas4/5/5P, Rosenbrock-W, TRBDF2, KenCarp/Kvaerno, FBDF/QNDF/ABDF2/DFBDF, RadauIIA, ROCK/RKC. **High order:** Vern6–9, Feagin12/14 (tolerances down to 1e-30). **Symplectic:** VelocityVerlet, McAte, KahanLi, RKN. Automatic stiffness switching. Generic number types (BigFloat, ArbNumerics). In v7 only Tsit5, Verner, Rosenbrock23, Rodas5P and FBDF are re-exported | Jacobians via ForwardDiff; GPU variants; threaded ensembles | https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/ |
| StochasticDiffEq.jl | 7.2.0 (2026-09-10) | its old repo is archived; it now lives inside the OrdinaryDiffEq monorepo (no functional change) | Adaptive strong-order SDE schemes | — | https://github.com/SciML/StochasticDiffEq.jl |
| SciMLSensitivity.jl | 7.119.14 (registry; date ≈2026-10-30 **[verify]**) | active | Continuous adjoints: Backsolve, Interpolating, Quadrature and Gauss (the docs prefer GaussAdjoint; Backsolve can be unstable on stiff problems). Forward sensitivity. Discrete adjoints through AD. Shadowing (LSS, NILSS) for chaotic systems | Vector-Jacobian products come from Enzyme, Zygote, ReverseDiff or Mooncake | https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/ |
| NeuralPDE.jl | 7.5.0 (2026-10-30) | active | **The default derivative is `FiniteDifferenceDerivative()`**: central stencils with step `eps(T)^(1/(2+order))`. The opt-in `EnzymeForwardDerivative()` uses nested forward mode for total order ≤ 4 | GPU via Lux/Reactant | https://github.com/SciML/NeuralPDE.jl (`src/pinn_types.jl` at v7.5.0) |
| DiffEqFlux.jl | 4.12.0 (2026-09-29) | active | Neural-ODE layers built on SciMLSensitivity | GPU | https://github.com/SciML/DiffEqFlux.jl |
| ModelingToolkit.jl | 11.46.1 (2026-10-27) | active | Symbolic index reduction (Pantelides, dummy derivatives) and tearing before any numerics run; symbolic Jacobians | Generates code at runtime | https://github.com/SciML/ModelingToolkit.jl |
| Symbolics.jl | 7.48.0 (2026-10-28) | active | Computer algebra, exact symbolic derivatives, `build_function` | Generates code at runtime | https://github.com/JuliaSymbolics/Symbolics.jl |
| Lux.jl | 1.31.5 (2026-10-27) | active | Explicit-parameter neural networks; AD tier list | Reactant/XLA | https://lux.csail.mit.edu/stable/manual/autodiff |
| Optimization.jl | 5.9.1 (2026-09-12) | active | One front end over many optimisers | — | https://github.com/SciML/Optimization.jl |
| LinearSolve.jl | 5.18.3 (2026-10-30) | active | Picks the method automatically; LU in pure Julia; KLU/UMFPACK; Sparspak (generic types); Krylov methods; mixed-precision LU **[verify details]** | BLAS threading is not deterministic | https://github.com/SciML/LinearSolve.jl |
| NonlinearSolve.jl | 4.32.0 (2026-09-14) | active | Newton, trust region, Levenberg–Marquardt; bracketing methods that guarantee an enclosure | — | https://github.com/SciML/NonlinearSolve.jl |

**Reference-solution convention (SciMLBenchmarks).** Stiff benchmarks (ROBER,
HIRES, OREGO, VDPOL) use as reference a pair of solutions, `CVODE_BDF` and
`Rodas5P`, both at `abstol = reltol = 1e-14`. POLLU uses `Rodas5` at 1e-14.
Non-stiff benchmarks use `Vern9` (or `Vern7`/`Vern8`) at 1e-14 or 1e-12. Errors
are computed with `WorkPrecisionSet(...; appxsol=…)`. No BigFloat reference
was found in the files sampled **[verify]**.
Sources: https://github.com/SciML/SciMLBenchmarks.jl/tree/master/benchmarks/StiffODE ,
…/NonStiffODE.
**Standard stiff test set:** the CWI/Bari "Test Set for IVP Solvers" (Lioen,
de Swart, van der Veen 1996; Mazzia and Magherini, release 2.4, 2008) and
Hairer–Wanner's test set: HIRES (8 equations), POLLU (20), VDPOL, ROBER, OREGO.
https://archimede.uniba.it/~testset/report/hires.pdf · https://unige.ch/~hairer/testset/testset.html

## 4. Quantum

| Package | Latest (date) | Status | Accuracy and capability mechanism | Caveat | Source |
|---|---|---|---|---|---|
| Yao.jl | 0.9.3 (2026-03-01); YaoBlocks 0.14.3 | active but slow; the README says "early-release beta" | Block IR (chain, put, control, kron, repeat, dagger). **Reverse-mode AD built in:** `expect'(H, reg => circuit)` gets gradients by un-computing the circuit instead of storing a tape. `gradcheck`; rules ported to ChainRules; YaoSym; YaoToEinsum | GPU through CuYao | https://github.com/QuantumBFS/Yao.jl |
| QuantumOptics.jl | 1.2.10 (2026-09-11) | active | Schrödinger and master equations, MC wavefunction, Bloch–Redfield, stochastic master equations, steady state, correlations and spectra, bosonic/spin/composite bases | Time evolution uses DiffEq adaptive integrators | https://github.com/qojulia/QuantumOptics.jl |
| QuantumToolbox.jl | 0.51.1 (2026-10-16) | active; QuTiP-like API (paper in Quantum 9, 1866, 2025) | `sesolve`, `mesolve`, `mcsolve`, `ssesolve`/`smesolve`, `brmesolve`, low-rank `mesolve`, `steadystate`, `spectrum`, correlations; differentiable | GPU; distributed trajectories | https://github.com/qutip/QuantumToolbox.jl |
| ITensors.jl | 0.9.32 (2026-09-27) | active (Flatiron Institute) | Tensors with labelled indices; **block-sparse storage that conserves quantum numbers**; SVD truncation with a cutoff | — | https://github.com/ITensor/ITensors.jl |
| ITensorMPS.jl | 0.4.2 (2026-09-27) | active. Split out of ITensors at v0.7 (2024-10-25) | `OpSum` → MPO, `dmrg`, `tdvp`, `linsolve`, `dmrg_x`, `expand`; truncation-error and `maxdim` control | — | https://github.com/ITensor/ITensorMPS.jl |
| QuantumClifford.jl | 0.11.8 (2026-09-06) | active | Tableaux with destabilisers, mixed stabilisers, graph states, Pauli frames, canonicalisation, QEC tooling. **Exact GF(2) arithmetic** | "Vectorized and multithreaded" | https://github.com/QuantumSavory/QuantumClifford.jl |
| Bloqade.jl | 0.2.5 (2025-04-04) | **maintenance**: the README recommends bloqade-python | Rydberg Hamiltonians; emulation through DiffEq/Krylov | Submitting to QPU hardware is out of scope | https://github.com/QuEraComputing/Bloqade.jl |

## 5. Items still marked [verify]

1. Per-function ulp bounds for `Base.Math`: there is no official table, so the harness measures them (ACCURACY_HARNESS_PLAN, slice S1).
2. "Zygote is being superseded": informal, not official.
3. Whether Enzyme is thread-safe and deterministic.
4. SciMLSensitivity 7.119.14 release date (GitHub's latest release object is 7.112.1, dated 2026-06-25).
5. LinearSolve's mixed-precision LU: only the file names were confirmed.
6. Whether any SciMLBenchmarks reference solution uses BigFloat.
7. The version of SpecialFunctions.jl (the erf/gamma oracle for slice S1b) was not surveyed. Pin it when the harness's `Manifest.toml` is created.
</content>
</invoke>
