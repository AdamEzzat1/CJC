# Issue drafts, ready for GitHub

Each draft is ready to paste into a GitHub issue. "Both executors" means the
test runs the same `.cjcl` through `cjc_eval::Interpreter` and
`cjc_mir_exec::run_program_with_executor` and asserts identical output
(`tests/pinn_parity.rs::assert_parity` is the pattern to follow).

Every fix must also turn its probe in [probes/](probes/) from wrong to
right. No `#[ignore]`, and no loosening of tolerances to make a test pass.

---

### I-01 · `GradGraph.hessian_diag` returns wrong values for composed losses (P0-1)
**Problem.** `hessian_diag` (`crates/cjc-ad/src/lib.rs:2343`) changes the
parameter and calls `backward`, but never re-runs the forward pass, so the
intermediate nodes keep their old values. For `loss = (p*p)*(p*p)` at p = 1 it
returns 4.000000000004; the exact value is 12 (probe `p1_hessian_diag_stale.cjcl`).
**Acceptance criteria**
- Probe P1 prints 12 to within 1e-6 (FD) now, and to within 1e-13 once ADR-B lands.
- `hessian_diag` either re-runs the forward pass, or returns an error for any op whose `reforward` support is missing (see I-07).
- Harness S3: `hessian_diag` relative error ≤ 1e-6 (FD) on all 20 functions.

**Tests (both executors)**
- p⁴, `exp(sin(p))`, and an MLP loss, compared with analytic second derivatives.
- A regression case: the composed loss must differ from the stale value.

### I-02 · The backward pass does not sum gradients of broadcast ops back to the input shape (P0-2)
**Problem.** `GradOp::Add/Sub/Mul/Div` (`lib.rs:1448-1480`) pass the upstream
gradient through without reducing the broadcast dimensions.
- For `sum(x[2,3] + b[3])`, `grad(b)` comes back with shape `[2,3]`, all ones. It should be `[2,2,2]` with shape `[3]` (probe P2).
- `examples/01_mlp_xor.cjcl` silently turns its `[1,4]` bias into `[4,4]`.

**Acceptance criteria**
- An `unbroadcast(grad, input_shape)` helper sums over the broadcast axes with a fixed-order binned reduction. It is applied in every binary op's backward.
- Probe P2 prints shape `[3]` and `[1,3]` with values `[2,2,2]`.
- Running `01_mlp_xor` keeps `b1.shape() == [1, 4]` on every epoch.
- Where two gradients of different shapes meet at one node, there is no panic.

**Tests (both executors)**
- Every combination of `{add, sub, mul, div} × {[n,m]+[m], [n,m]+[1,m], [n,m]+[n,1], [n,m]+scalar}`, checked against the reduced FD gradient.
- A determinism test: 100 repeated runs give identical bits.

### I-03 · `svd` is computed as `eigh(AᵀA)`: small singular values are lost (P0-3)
**Problem.** `linalg.rs:1162-1180` squares the condition number.
- For `[[1,1],[1,1+1e-10]]`, `svd` returns σ_min = **0.0**. The true value is 5.000000413576855e-11 (mpmath), and `eigh` gives 5.000000413701855e-11 (probe P3).
- `pinv` and `matrix_rank` are built on `svd`.

**Acceptance criteria**
- A one-sided Jacobi SVD (cjc-quantum's MPS SVD, with this session's sign fix, is a model) or Golub–Kahan bidiagonalisation with implicit QR. All sweeps happen in a fixed order, sums are compensated, and there is no FMA.
- `U` and `V` are orthonormal to ≤ 10·n·u even when a σ is 0 (complete the basis deterministically).
- Probe P3 gives σ_min with absolute error ≤ c·n·u·σ_max (normwise backward stability). *Corrected 2026-11-10:* the original "relative error ≤ 1e-12" was more than any normwise-stable SVD promises. On this matrix LAPACK `gesdd` (numpy) has relative error 1.2e-7, and the one-sided Jacobi fix has 3.5e-7. Both are below 0.04·u·σ_max.
- Harness S2: for Hilbert matrices n ≤ 12, the relative error of each σ ≤ 10·n·κ_σ·u.
- The `linalg.rs:1144` header comment is corrected.

**Tests (both executors)**
- Hilbert, Vandermonde and graded matrices against hard-coded mpmath references.
- Rank-deficient matrices.
- `pinv` and `matrix_rank` on near-singular inputs.

### I-04 · `dmrg_heisenberg` and `dmrg_ising` converge to the wrong energy for n ≥ 4 (P0-4)
**Problem.**
- `dmrg_heisenberg(4, 8, 10, 1e-10)` returns −5.950264027437909. Exact diagonalisation of the Hamiltonian in `dmrg.rs:77-85` gives −6.464101615137755 (`probes/p6_exact.py`). n = 6 and n = 8 are also about 8% too high.
- χ = 8 already covers the full rank at n = 4, so truncation is not the cause.
- The existing 4-site test only asserts `E < 0`.

**Acceptance criteria**
- Diagnose the cause first, recording it in the issue. Candidates:
  - the boundary-operator environments;
  - the Lanczos iteration cap of 30 (`dmrg.rs:778`);
  - the effective-Hamiltonian assembly;
  - the energy function used for convergence.
- DMRG matches exact diagonalisation to ≤ 1e-10 relative for n = 4…12 at full-rank χ, for both Hamiltonians.
- Harness S6 against ITensorMPS for n ≤ 40.

**Tests (both executors)**
- Replace `test_dmrg_4_qubit_heisenberg_energy_negative` with exact-value tests: hard-coded numpy values for n = 4, 6, 8.
- Monotone energy decrease over sweeps.

### I-05 · The binned accumulator is not order-invariant (P0-5, ADR-A)
**Problem.** Inside one exponent bin, `bins[e] += v` uses plain IEEE addition.
- `Tensor.from_vec([1, b, b]).sum()` gives 3.0, while `[b, b, 1]` gives 3.0000000000000004, with b = 1 + 2⁻⁵² (probe P5).
- The accumulator's header says it is "commutative + associative". The determinism contract relies on that to give the same results at any thread count and with any chunking.

**Acceptance criteria**
- ADR-A is written and accepted, with a measured cost on `sum`/`dot`/`mean` (interleaved A/B, allocation counts).
- The accumulator becomes exact within each bin: TwoSum with error carry, or an integer long accumulator.
- Probe P5 prints `true`, and both orders give 3.0000000000000004, the exact result.
- Harness S2 permutation test: exactly 1 distinct bit pattern across 16 permutations × 3 chunkings for all inputs, including the Ogita–Rump–Oishi sequences.

**Tests**
- A proptest: random vectors, random permutation, bit-equal sums.
- A parity test in both executors.
- Golden hashes that change must be re-locked **with an explanation in the commit message**.

### I-06 · `randn` and normal samplers use the platform libm (P0-6, ADR-0046 amendment)
**Problem.** `Rng::next_normal_f64` (`crates/cjc-repro/src/lib.rs:129`) calls
`f64::ln` and `f64::cos`. Every normal draw (`Tensor.randn`, the distribution
samplers, NSS and ABNG weight initialisation) can therefore differ in the last
bit between Windows, Linux and macOS.

**Acceptance criteria**
- Use `dmath::ln` and `dmath::cos`.
- A cross-platform golden for 10⁴ `randn` draws passes on all three CI operating systems.
- Report whether the open cross-platform canary failures (task `task_d04e49df`) are resolved by this change. Do not claim they are without the CI run.

**Tests**
- A `randn` hash golden in both executors.
- Add it to `cross_platform_golden.rs`-style CI coverage.

### I-07 · Finite-difference higher derivatives are presented as exact (P0-7)
**Problem.**
- `hessian` and `double_backward` (`lib.rs:2392`, `2706`) are central differences with ε = 1e-5. The name `double_backward` implies AD-over-AD.
- `reforward`'s catch-all (`lib.rs:2690`) keeps stale values for Softmax, LayerNorm, BatchNorm, CrossEntropy, Where, Cat, Gather and Log2.

**Acceptance criteria**
- Until ADR-B lands: rename the functions to `hessian_fd`/`hessian_diag_fd`/`second_derivative_fd`, keep deprecated aliases, and document the FD error.
- `reforward` returns an error for ops it cannot re-run, instead of silently using stale values.
- After ADR-B: the exact versions take the original names.

**Tests**
- An FD Hessian through a softmax must either match FD-of-FD or return an error.
- It must never return the stale value.

### I-08 · `erf`/`erfc`/`normal_cdf` are accurate to only about 1e-7 absolute (P0-8)
**Problem.** The Abramowitz & Stegun 7.1.26 formula (`distributions.rs:183,195`) gives:
- relative error 2.7e-7 for `erf(0.5)`;
- 6.6e-4 for `erfc(3)`;
- 6.7e-3 for `erfc(5)` (probe P4, mpmath reference).

**Acceptance criteria**
- Port fdlibm `s_erf.c` (the same family as dmath) using dmath's `exp`.
- Harness S1b: maximum relative error ≤ 1e-15 on [−6, 6] and ≤ 1e-13 in the `erfc` tail, checked against both Julia `BigFloat` and mpmath.
- `normal_cdf` uses `erfc` for x < 0, so the tail does not cancel.

**Tests (both executors)**
- 50 hard-coded mpmath reference points.

### I-09 · Silent failures in the crate ODE solvers (P0-9)
**Problem.**
- `ode_solve_rk45` silently `break`s when h < 1e-14 and returns a shorter trajectory (`ode.rs:520,527`).
- `adjoint_solve` starts from a zero adjoint, has no way to take dL/dy(T), and discards the parameter gradient (`ode.rs:582,600-618`).
- Its finite-difference test does not call it.

**Acceptance criteria**
- RK45 returns `Err(StepSizeUnderflow { t, h })`.
- `adjoint_solve` takes `dl_dy_final` and returns `(dL/dy0, dL/dθ)`. Its test compares the function itself against FD on a linear ODE (with an analytic gradient) and on Lotka–Volterra.
- Harness S4: Robertson with RK45 either completes or errors; it never truncates silently.

**Tests**
- Rust-level tests today; parity tests once ADR-C exposes the solvers.

---

### I-10 · ADR-C: builtins that call back into the executor (P1-2)
**Acceptance criteria**
- An ADR defines a `CallbackHost` trait implemented by both executors.
- A first consumer is wired end to end: `ode_solve(f, y0, t_span, opts)` with RK45. The same consumer can replace the `ode_step`/`pde_step` stubs and the dead `bisect` arm.
- Callback order is identical in both executors, checked by a recording test.
- The NoGC verifier rejects callback builtins inside `@nogc` functions.

### I-11 · ADR-B: exact Hessians and Taylor-mode derivatives (P1-1)
**Acceptance criteria**
- Exact Hessian-vector products (forward-over-reverse) and full Hessians.
- A `Jet<N>` for scalar n-th derivatives.
- Harness S3: relative error ≤ 1e-13 against ForwardDiff and TaylorDiff for orders 2–4.
- A PINN option `derivative: "exact"` for the heat and Burgers trainers. On the S5 grid its residual error is compared with FD's.

### I-12 · Quantum adjoint gradients from `.cjcl` (P1-7)
**Acceptance criteria**
- A `q_adjoint_grad(circuit, pauli_sum)` builtin covers every parametrised gate and any Pauli-sum observable.
- Its results agree with parameter shift to ≤ 1e-12 on random circuits n ≤ 10.
- Harness S6 against Yao `expect'`.
- Parity tests in both executors.

### I-13 · Open-system quantum simulation (P1-6, ADR-G)
**Acceptance criteria**
- `density_mesolve(rho, H, c_ops, t, opts)` with a deterministic integrator.
- An MCWF trajectory builtin, seeded.
- Harness S6: trace-distance error against QuantumToolbox `mesolve` ≤ 1e-8 on a driven damped qubit and on a 4-qubit dephasing chain.
- The trajectory average converges to the `mesolve` result as 1/√N (statistical test, fixed seed).
</content>
</invoke>
