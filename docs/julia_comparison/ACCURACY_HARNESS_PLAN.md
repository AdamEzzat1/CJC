# Accuracy harness plan: `bench/julia_compare/`

**Status:** designed in this session, to be built next session. Julia is the
**oracle**: it runs offline and only produces golden files. CJC is the
**system under test**. CJC never links, calls or depends on Julia at build
time or at run time.

The structure follows `bench/quantum_compare/`, which already works this way
with Aer and Stim: a Rust runner, external drivers under `externals/`, and
results in `bench_results/`.

## 1. Principles

1. **References carry more precision than f64.** A reference value is a
   double-double pair `(hi, lo)` with `hi + lo` within about 2⁻¹⁰⁶ relative of
   the true value. Error is computed in double-double, so a 0.5-ulp error and
   a 0.9-ulp error can be told apart.
2. **Inputs are exact.** Every input is stored as its IEEE-754 bit pattern in
   hex and also as a 17-significant-digit decimal. The runner checks that the
   two agree.
3. **Two oracles where cheap.** Elementary and special functions are computed
   in Julia `BigFloat` (MPFR) and also in Python `mpmath`. The probes already
   use mpmath, which is installed here. If the two disagree by more than 2⁻¹⁰⁰
   relative, the row is dropped and listed in the report.
4. **Hashed, committed goldens.** Each slice directory holds a `SHA256SUMS`
   and a `META.json` recording:
   - the Julia version and the `Manifest.toml` hash;
   - the Python and mpmath versions;
   - the machine, the CJC commit and the generation date.

   The runner refuses to run if a hash does not match.
5. **Both executors, every row.** The runner evaluates every CJC expression
   in both `cjc-eval` and `cjc-mir-exec`. An executor mismatch is a separate,
   **blocking** failure, reported before any accuracy number.
6. **Report first, gate later.** The first build only reports. Thresholds
   (§5) become CI gates one slice at a time, after the P0 fixes land, so the
   harness never needs `#[ignore]` to stay green.
7. **Known bugs are positive controls.** Every probe in [probes/](probes/)
   becomes a harness row. Until it is fixed, the harness must report it as
   failing. A harness that passes P1–P6 is broken.
8. **No timing.** This harness measures accuracy only. Performance against
   Julia is out of scope.

## 2. Layout

```
bench/julia_compare/
  Cargo.toml                 # bin `julia_compare`; depends on cjc-eval, cjc-mir-exec, cjc-parser
  src/main.rs                # `run --slice S1 --out <dir>`, `verify-goldens`, `report`
  src/dd.rs                  # double-double error arithmetic (TwoSum, Dekker TwoProd, no FMA)
  src/slices/*.rs            # one module per slice: builds the .cjcl program, reads the outputs
  externals/
    Project.toml, Manifest.toml        # pinned Julia environment
    gen_s1_elementary.jl  gen_s1b_special.jl  gen_s2_linalg.jl  gen_s3_ad.jl
    gen_s4_ode.jl  gen_s5_pinn.jl  gen_s6_quantum.jl
    cross_mpmath.py                     # second oracle for S1/S1b
  golden/<slice>/*.csv, SHA256SUMS, META.json
bench_results/julia_compare/<date>_<machine>/
  <slice>.jsonl  report.md
```

**Julia version.** Pin **1.10 LTS**, which is already installed (1.10.13).
Upgrading to 1.13 would mean regenerating every golden file and re-pinning
the Manifest. Record the choice in `META.json`.

## 3. The CSV contract

```
# slice=S1 fn=sin grid=edge seed=0x5EED0001 julia=1.10.13 manifest=<sha256>
x_bits,x_dec,ref_hi,ref_lo,ref_hi_bits,ref_lo_bits
3ff0000000000000,1.0,0.8414709848078965,3.1...e-17,3feaed548f090cee,...
```

- **H0, a precondition (first task of the build):** a round-trip test. Write
  10⁵ random bit patterns, read them back through CJC's own `read_csv` /
  `df_read_csv` in both executors, and compare the bits.
  - If the decimal path is not correctly rounded, that is a P0 finding in its own right.
  - In that case the harness falls back to building inputs from `x_bits` through the Rust runner.
  - This session did not settle whether CJC's CSV float parsing is correctly rounded.
- **Error metrics (computed by the runner in double-double):**
  - `ulp_err = |cjc − (hi + lo)| / ulp(hi)`;
  - `rel_err = |cjc − (hi + lo)| / |hi + lo|`;
  - for vectors and matrices, also the norm-wise relative error.

## 4. Slices

**S1: elementary functions** (the negative control: dmath should pass)
- **Functions:** the dmath set exposed as builtins: `sin cos tan asin acos atan atan2 sinh cosh tanh exp expm1 log log1p log2 log10 pow sqrt hypot`.
- **Grids** (each seeded with SplitMix64; the seed is recorded):
  1. 10⁴ points log-uniform over each function's domain.
  2. Edge cases: ±0, the smallest subnormal, the largest subnormal, the smallest normal, `f64::MAX`, ±Inf, NaN; integer multiples kπ/2 for k up to 10⁶ (nearest f64 values); the hard cases for argument reduction from the Kahan–McDonald list (`x = 6381956970095103·2^797` and similar); `exp` overflow and underflow boundaries; `pow` with integer and half-integer exponents; `pow(x, y)` with x near 1 and large y.
  3. 10³ points within 64 ulp of the zeros of `sin`, `cos` and `log`.
- **Report:** maximum and mean ulp error, and the arguments of the worst five.
- **Also report:** Julia `Base.Math`'s own ulp error on the same grid, as context only. It gives the "Julia accuracy" figure no source publishes [verify item 1]. It is computed by the Julia script itself, with no CJC numbers placed beside timings.

**S1b: special functions** (the positive control: P4)
- **Functions:** `erf`, `erfc`, `normal_cdf`, `normal_ppf`, plus the crate's `ln_gamma` via a Rust-side runner row, labelled crate.
- **Grid:** 10⁴ points on [−6, 27] for `erfc` (the tail is where A&S 7.1.26 breaks down).
- **Oracle:** `SpecialFunctions.erf` on `BigFloat` (MPFR `mpfr_erf`), cross-checked against mpmath.
- **Expected today:** a maximum relative error around 1e-2 in the `erfc` tail.

**S2: reductions and linear algebra** (positive controls: P3, P5)
- **Reductions:**
  - `sum`, `mean`, `dot` and `matmul` on seeded random vectors, using cancellation-heavy sequences (Ogita–Rump–Oishi generated dot products with condition numbers 10⁰ to 10³²) and matrix sizes 63/64/65, which straddle the matmul Kahan → tiled switch.
  - Report the relative error against the exact BigFloat value, as a function of the condition number.
  - **Permutation test (P5):** for each input, 16 seeded permutations plus 3 chunkings. Report how many distinct bit patterns each produces. The contract requires exactly 1.
- **Solve / QR / SVD / eig:**
  - **Matrices:** Hilbert n = 2…14, Vandermonde on equispaced and Chebyshev nodes, `diag(1, …, 10⁻ᵏ)` with random orthogonal mixing, and graded matrices.
  - **Report:**
    - for `solve`: normwise backward error ‖Ax̂ − b‖ / (‖A‖‖x̂‖ + ‖b‖) and forward error ‖x̂ − x‖ / ‖x‖, against κ(A);
    - for `svd`: per-σ relative error against BigFloat singular values from `GenericLinearAlgebra.svdvals` [verify package];
    - for `eigh`: eigenvalue error relative to ‖A‖.
  - **Pass shape:** backward error ≤ c·n·u (u = 2⁻⁵³), and forward error ≤ c·κ·u.
  - **Expected today:** `svd` fails the σ-relative check from about κ ≈ 10⁸ on, because squaring κ loses half the digits.

**S3: automatic differentiation** (positive controls: P1, P2)
- **Functions:** a fixed list of 20 scalar and vector functions: polynomials, `exp(sin(x))`, `log(1 + x²)`, `tanh` chains, softmax-CE, layer norm, a two-layer MLP with a broadcast bias, Rosenbrock.
- **Report:**
  - gradient against ForwardDiff (exact);
  - Hessian (full and diagonal) against ForwardDiff nested;
  - third and fourth derivatives of scalar functions against TaylorDiff, and also TaylorSeries for redundancy, since TaylorDiff is alpha.
- **Third reference:** a BigFloat central difference with Richardson extrapolation, to detect oracle bugs.
- **Metric:** relative error per component.
- **Expected today:**
  - gradients agree to about 1e-15;
  - `hessian_diag` is wrong by O(1) on any composed function (P1);
  - Rust `hessian` has an error of about 1e-6 to 1e-10 (from FD);
  - the broadcast-bias gradient has the wrong shape (P2).

**S4: ODEs**
- **Problems:**
  - Non-stiff: exponential, harmonic oscillator, Kepler two-body, Lotka–Volterra.
  - Stiff: Robertson, Van der Pol μ = 10³, HIRES, all from the CWI/Bari test set.
- **Reference:** following the SciMLBenchmarks convention, the solution at the final time and at 20 fixed times from `Vern9` (non-stiff) or from `Rodas5P` and `CVODE_BDF` (stiff), all at `abstol = reltol = 1e-14`. Agreement between the two references sets the reference's own error bar. For the linear and Kepler problems, use the closed-form solution as well.
- **Report:**
  - **Observed order:** fixed-step RK4 and Euler at h = 2⁻ᵏ, k = 4…14; fit the slope of log error against log h, expecting 4 and 1.
  - **Work–precision (accuracy only):** for crate RK45, error against requested `rtol` from 10⁻³ to 10⁻¹².
  - **Stiff problems:** what crate RK45 does on Robertson, i.e. the number of steps and whether it **silently truncates** (positive control for the RK45 h-underflow finding).
- **Surface:**
  - ODE solvers are crate-only today, so S4 runs through a Rust runner and its rows are labelled `crate`.
  - Once ADR-C lands, the same rows go through `.cjcl` in both executors.

**S5: PINNs**
- **Problems:** the 10 `.cjcl`-reachable `pinn_train_*` trainers, run with a fixed seed and fixed epochs.
- **Reference:** analytic solutions where they exist (heat, wave, Helmholtz, Poisson manufactured solution, harmonic). For Burgers, Allen–Cahn and KdV, a Julia spectral/MOL reference from `OrdinaryDiffEq` at 1e-12 on a 1024-point grid.
- **Metric:** relative L2 error on a fixed 256-point evaluation grid.
- **Context only, not compared:** NeuralPDE.jl's published examples.
- **Second metric:** the PDE residual computed **exactly**, by evaluating the trained network's derivatives with ForwardDiff in Julia on exported weights. This separates "the network is wrong" from "FD stencils hid the residual".

**S6: quantum** (positive control: P6)
- **Workloads:**
  - QFT n = 3…12;
  - random circuits n ≤ 20, depth 20, seeded;
  - Trotterised Heisenberg chains n ≤ 12, order 1 and 2, against the exact `exp(-iHt)`;
  - VQE energy landscape points for H₂ (fixed parameters) against the exact ground state;
  - DMRG ground-state energies for Ising and Heisenberg n = 4…40 against ITensorMPS `dmrg` and exact diagonalisation for n ≤ 14;
  - noisy density-matrix evolution under the three `.cjcl` channels against QuantumOptics/QuantumToolbox.
- **Report:** state fidelity |⟨ψ_ref|ψ⟩|² (dense), relative energy error, and the trace-distance error for density matrices.
- **Oracles:** Yao (circuits), ITensorMPS (DMRG), QuantumToolbox (density matrix, with Lindblad as context for future work). States are exported as interleaved (re, im) double-double pairs. Statevectors above n = 16 are compared through ⟨Z_i⟩ and ⟨Z_iZ_j⟩ only, to keep the golden files small.
- **Expected today:** DMRG relative energy error of about 8% at n ≥ 4.

## 5. Proposed thresholds (they become gates only after the matching P0 fix)

| Slice | Metric | Gate |
|---|---|---|
| S1 | maximum ulp error per function | ≤ the bound dmath claims (1 ulp for most; 2 for tanh, sinh, atanh) |
| S1b | maximum relative error | ≤ 1e-15 on [−6, 6]; ≤ 1e-13 relative in the `erfc` tail |
| S2 sums | distinct bit patterns across permutations and chunkings | = 1 |
| S2 solve | backward error | ≤ 10·n·u |
| S2 svd | σ relative error for σ ≥ σ_max·u·n | ≤ 10·n·κ_σ·u |
| S3 | AD gradient and Hessian relative error | ≤ 1e-13 (exact AD); FD-based entries are reported as such, never gated as exact |
| S4 | observed order | within ±0.1 of the theoretical order |
| S4 | adaptive solver at tolerance `tol` | error ≤ 10·tol, and **never** a silently truncated trajectory |
| S6 | DMRG energy relative error at χ that covers the full rank | ≤ 1e-10 |

## 6. Cost and order of construction

1. **H0** plus the runner skeleton plus **S1/S1b**: S effort, about one session. This is the first slice (README §5).
2. **S2:** M. **S3:** M, and needs TaylorDiff/ForwardDiff in the Manifest. **S6:** M.
3. **S4/S5:** M–L. S4 becomes `.cjcl`-level only after ADR-C.
</content>
</invoke>
