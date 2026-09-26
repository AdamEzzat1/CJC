# CJC-Lang Quantum Simulation — Verification Plan

**Status:** PLAN. §2 records **current** coverage (counted from source on
2026-09-24; pass/fail status from the runs recorded in SURFACE_AUDIT §8).
§3 onward is **proposed** coverage — nothing in §3–§8 exists yet unless it's
explicitly marked "exists".
**Companion docs:** [SURFACE_AUDIT.md](SURFACE_AUDIT.md) · [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) ·
[MISSING_FEATURES.md](MISSING_FEATURES.md)

> Tests establish correctness claims; benchmarks establish performance claims.
> A feature is **"tested"** only if at least one test would fail if the feature
> were broken. A feature reachable from `.cjcl` is **"language-tested"** only if
> a test drives it through *both* `cjc-eval` and `cjc-mir-exec`.

> **Status update (2026-09-24):**
> - T-1…T-8 are fixed on this branch.
> - Implemented from this plan:
>   - §6 dispatch fuzz sweep: `tests/beta_tests/fuzz/test_fuzz_quantum_dispatch.rs`.
>   - §7 Bolero `fuzz_quantum_dispatch_no_panic`, plus the two repaired targets.
>   - §5 properties (8): `tests/beta_tests/quantum_prop/test_quantum_properties.rs`.
>   - §3.6 QAOA oracle and gradient tests.
>   - §3.8 chemistry reference tests.
>   - §8 negative cases: all probes return errors.
> - §4.1 (per-builtin parity table) and §4.3 (statistical tests) are still open.

---

## 1. Principles

1. **Bit patterns, not tolerances, where bit-identity is promised.** The crate
   promises "same seed = bit-identical" (`lib.rs:4`, `measure.rs:25`,
   `pure.rs` header). Replay tests compare `f64::to_bits()` / exact integer
   vectors, not `abs() < 1e-12`.
2. **Tolerances only against mathematics.** Comparisons with analytic values or
   other simulators use explicit tolerances with the justification written next
   to them (ulp budget or accumulated-rounding argument).
3. **Negative tests are first-class.** Every `.cjcl` builtin gets at least one
   test proving malformed input returns `Err` (a runtime error value), **not** a
   Rust panic and not a silent default.
4. **Every assertion must be able to fail.** Fuzz/Bolero bodies must not wrap
   assertions in `catch_unwind` and then discard the result (see §2.4 finding).
5. **Parity is two-dimensional:** eval ≡ MIR (executor parity) and, where both
   exist, Rust backend ≡ `"pure"` backend (implementation parity). They are
   tracked separately.

## 2. Current coverage (exists today)

### 2.1 Counts

| Location | `#[test]` markers | Notes |
|---|---|---|
| `crates/cjc-quantum/src/*.rs` (unit) | 280 | `docs/QUANTUM_SIMULATION.md` says 224 — stale |
| `tests/beta_tests/quantum/` (13 files) | 311 | Integration + parity. Doc says "342 in beta_tests/" — not reproducible |
| `tests/beta_tests/hardening/` quantum files | 88 | **46 are byte-identical duplicates**: `hardening/test_quantum_hardening.rs` ≡ `quantum/test_quantum_hardening.rs` (35) and `hardening/test_vqe.rs` ≡ `quantum/test_vqe.rs` (11); both copies compile and run |
| `tests/beta_tests/quantum_prop/` | **0 compiled** | `mod.rs` is a comment only; the `test_quantum_determinism.rs` inside is a byte-identical, **uncompiled** copy of `quantum/test_quantum_determinism.rs`. Doc claims "Property Tests (8 in beta_tests/quantum_prop/)" |
| `tests/beta_tests/fuzz/test_fuzz_quantum.rs` | 10 | Seeded loops over **valid** inputs (e.g. skips `q == q2` for CNOT); crate-API only, no `.cjcl` |
| `tests/bolero_fuzz/mod.rs` quantum targets | 2 | `fuzz_fermion_expectation_determinism`, `fuzz_zne_richardson_determinism` — **vacuous**, see §2.4 |
| `tests/bench_50q.rs`, `tests/bench_dual_mode.rs` | 14 + 17 | Timed smoke tests (see BENCHMARK_PLAN §1) |
| `tests/quantum_and_piml_demos_parity.rs` | 5 | Demos 01–03 only; 04–06 "verified out-of-band" (not in CI) |
| `proptest!` in any quantum test | **0** | — |

### 2.2 What is language-tested (eval ≡ MIR)

Only three `tests/beta_tests/quantum/` files drive `.cjcl` source through both
executors: `test_quantum_integration.rs` (58), `test_quantum_native_types.rs`
(35), `test_quantum_pure_backend.rs` (28). Parity is checked by comparing
`format!("{}", value)` strings. For finite `f64` this is close to bit-exact
because Rust's `Display` prints the shortest round-trip representation. It still
isn't the same as a `to_bits()` comparison: it misses NaN payloads, and it
depends on `Value`'s `Display` impl.

**Builtins with zero executor-level test coverage** (text search of the three
files above plus `bench_50q.rs` — `bench_dual_mode.rs` is eval-only and excluded; 27 of 78 dispatch names; text search, so an upper bound on coverage):

`q_run`, `q_amplitudes`, `q_rz`, `q_s`, `q_t`, `q_y`, `q_z`\*, `q_swap`, `q_toffoli`, `q_ccx`,
`q_fermion_h2`\*, `q_fermion_lih`, `q_fermion_new`, `q_fermion_n_terms`\*,
`q_fermion_expectation`\*, `q_trotter_evolve`, `q_trotter_error`, `q_zne_mitigate`\*,
`q_zne_linear`\*, `q_scale_noise`, `mps_energy`, `mps_left_canonicalize`,
`mps_right_canonicalize`, `mps_mixed_canonicalize`, `mps_swap`, `qec_surface_code`,
`qml_train`.

\* exercised only by demos 04–06, which are not in the CI parity test.

### 2.3 What is unit-tested only (crate API, not `.cjcl`)

Density (28 integration + 23 unit), stabilizer (23 + 23), DMRG, QAOA, QEC, QML,
VQE, fermion, Trotter, mitigation, adjoint (12 unit), Wirtinger (11 unit),
SIMD kernels (6 unit — kernels themselves are dead code in production),
`HybridCircuit` mid-circuit measurement (unit only; not reachable from `.cjcl`).

### 2.4 Defects found in existing tests

| ID | Defect | Evidence | Fix |
|---|---|---|---|
| T-1 | Bolero targets cannot fail | `tests/bolero_fuzz/mod.rs:171-195, 200-225`: body wrapped in `let _ = panic::catch_unwind(\|\| { … assert_eq!(…) })` — assertion panics are caught and discarded | Remove `catch_unwind`; if panic-freedom is the property, assert `catch_unwind(...).is_ok()` |
| T-2 | Duplicate test files double-count | `diff -q` identical pairs listed in §2.1 | Delete the `hardening/` copies or the `quantum/` copies; keep one |
| T-3 | Empty `quantum_prop` module + orphan file | §2.1 | Either populate with real `proptest!` suites (§5) or delete |
| T-4 | Fuzz tests only use valid inputs | `test_fuzz_quantum.rs:55` (`if q != q2`) | Add invalid-input fuzzing through `dispatch_quantum` (§6) |
| T-5 | Statistical tests use ad-hoc bands | `measure.rs:201-207` accepts 0.4–0.6 on 1000 shots | Replace with a chi-squared / binomial CI test at a documented α (§4.3) |
| T-6 | Bench name mismatch | `bench_20q_dmrg` runs 8 qubits | **Fixed:** renamed `bench_8q_dmrg` |
| T-7 | Demos 04–06 not gated | `quantum_and_piml_demos_parity.rs:24-31` | Add to `DEMOS` |
| T-8 | **Failing test on this branch** | `test_qaoa.rs:165` `qaoa_4_cycle_finds_good_cut` fails deterministically (run 2026-09-24). The last SVD change (`66b65bd`) was verified with `--lib` only | Bisect; add `--test test_beta_tests` to the `cjc-quantum` pre-merge gate (MISSING_FEATURES P0-9) |

## 3. Unit tests (proposed, per simulator family)

Each row lists the property and the exact oracle. "Bits" = `to_bits()` equality.

### 3.1 Gates / statevector (`gates.rs`, `statevector.rs`)
- Every gate matrix is unitary: ‖U†U − I‖_max ≤ 2 ulp·dim (all 13 gates, plus Rx/Ry/Rz at θ ∈ {0, π/7, π, −π, 2π, 1e-300, 1e300}).
- Involutions: X², Y², Z², H², SWAP², CX², CZ², CCX² = I on random states (≤ 4 ulp L∞).
- Algebraic identities: HZH = X, HXH = Z, S² = Z, T² = S, CZ = (I⊗H)CX(I⊗H), SWAP = CX·CX'·CX.
- `Rz(θ)` vs `Rz(θ+4π)` bits-equal? (Records whether angle reduction is exact — expected **not** bits-equal; the test documents it.)
- `from_amplitudes` rejects non-power-of-2 **and** 0-length (exists) **and** NaN amplitudes (proposed: currently accepted).
- `normalize` on the zero vector leaves it unchanged (current behaviour, `statevector.rs:93`) — document and test.

### 3.2 Measurement / sampling / PRNG (`measure.rs`, `lib.rs`)
- `rand_f64` ∈ [0,1), never 1.0: exhaustive over the top-53-bit boundary (`u64::MAX`).
- `splitmix64` golden vector: first 16 outputs for seeds {0, 1, 42, u64::MAX} pinned as constants (guards against accidental constant edits).
- `sample_basis_state` on a distribution with trailing zeros never returns a zero-probability index. Current fallback returns `n-1` (`measure.rs:99`), even when `P(n-1) = 0`. Test and fix.
- `measure_qubit` on a state with `prob0` computed as `1.0 + ε` → outcome 0 always; with prob exactly 0 → outcome 1 always.
- Negative `.cjcl` seeds: `q_sample(c, n, -1)` maps to `u64::MAX` via `as u64` — pin the mapping so it cannot silently change.

### 3.3 MPS (`mps.rs`, `pure.rs::PureMps`)
- `svd_sign_stabilized`: U·Σ·V† reconstructs A (≤ 1e-12 rel), U/V orthonormal, and the sign convention holds (largest-|·| entry of each left singular vector is positive).
- SVD determinism: same input → bits-equal U, Σ, V over 100 random 8×8 complex matrices.
- Degenerate singular values (identity, rank-1, all-zeros matrix) — no NaN, sign rule still deterministic.
- `apply_cnot_adjacent` both orientations vs dense statevector (n ≤ 10).
- `mixed_canonicalize(c)` preserves `to_statevector()` (≤ 1e-12) and yields left/right isometries.
- Truncation: with χ_max < exact bond, record discarded weight; state norm after truncation — decide & document whether renormalised.

### 3.4 Density / noise (`density.rs`, `PureDensity`)
- Kraus completeness Σ K†K = I for all three channels over p ∈ {0, 1e-12, 0.5, 1}.
- Channel trace preservation (≤ 1e-13), hermiticity ρ = ρ† (bits-equal after symmetric ops?), PSD (min eigenvalue ≥ −1e-12).
- Depolarizing convention: test pins `ρ → (1−p)ρ + (p/3)(XρX+YρY+ZρZ)` and documents that this equals "replace with I/2 w.p. 4p/3" (the docstring currently says "w.p. p", `density.rs:575-577`).
- `purity`, `von_neumann_entropy` on maximally mixed n=1..4: purity = 2⁻ⁿ, S = n·ln 2 (≤ 1e-12).
- `partial_trace` of a Bell state = I/2.
- `fidelity` symmetric, = 1 for identical states, = |⟨ψ|φ⟩|² for pure states.

### 3.5 Stabilizer (`stabilizer.rs`, `PureStabilizer`)
- Tableau commutation invariants after every gate: stabilizers mutually commute; destabilizer i anticommutes with stabilizer i only.
- `to_statevector()` (n ≤ 10) vs dense `Circuit` for random Clifford circuits (≤ 1e-12 up to global phase).
- Deterministic measurement (Z-basis on |0…0⟩, after X) is seed-independent — assert across 64 seeds.
- Word-boundary cases n ∈ {63, 64, 65, 127, 128, 129} for bitpacked `rowmult`.

### 3.6 VQE / QAOA / QML
- `mps_energy`: n ≤ 10 vs dense ⟨ψ|H|ψ⟩ for Ising and Heisenberg (≤ 1e-10).
- `vqe_heisenberg` energy ≥ exact ground energy (variational bound) for n ∈ {4, 6, 8}.
- `qaoa_maxcut`: `cut_value` ≤ analytic optimum; `energy_history` monotone non-increasing? (document if not).
- QML: `qml_gradient` (finite-difference, ε=1e-4, `qml.rs:~335`) vs parameter-shift on bias parameters; module doc currently says "parameter-shift" (`qml.rs:8`) — test must pin which one is used.

### 3.7 QEC
- Repetition code d ∈ {3, 5, 7}: every single-qubit X error is corrected (exhaustive).
- Surface code: `qec_decode` on a surface code currently routes to `decode_repetition_code` (`dispatch.rs:556`) although `qec.rs:14,274-275` says no 2D decoder exists — test must assert **Err** (after fix), not a plausible-looking vector.
- `estimate_logical_error_rate` is monotone in p for d=3 over p ∈ {0.001, 0.01, 0.05, 0.1} within binomial CI.

### 3.8 Fermion / Trotter / ZNE
- `PauliTerm::multiply` full 4×4 Pauli table with phases (exists partially).
- H₂: basis-state energies (−1.8302, −0.2738, 0.1824, 0 — from demo 05) and matrix minimum −1.851199 (≤ 1e-6). Also assert the independent FCI reference from `verification/h2_sto3g_check.py`: electronic ground −1.851024 Ha within 5e-4 Ha, and total −1.137270 Ha once 1/R = 0.713754 is added. Do **not** assert excited levels, which are off by 19–24 mHa (VERIFY_FOLLOWUPS §1).
- LiH: the current matrix is **not** LiH (VERIFY_FOLLOWUPS §1). After regeneration, assert the PySCF references: FCI total −7.882762 Ha, or the 2e/2o active-space total −7.863374 Ha, whichever encoding ships.
- Trotter: 2nd-order error ≤ 1st-order error at equal steps; measured error ≤ `trotter_error_bound` for H₂ at t ∈ {0.5, 1, 2}; commuting-term Hamiltonian → exact at 1 step.
- Richardson: exact on polynomials of degree ≤ k−1; `[1,2,3]` → coefficients `[3,−3,1]` bits-equal; duplicate scales → `Err` (verify current behaviour, probe p23).

### 3.9 Adjoint / Wirtinger
- `adjoint_differentiation` vs central finite differences (h = 1e-6) and vs parameter-shift for Rx/Ry/Rz on random 1–4 qubit circuits (≤ 1e-7).
- Non-parametric gates in the reverse sweep: S/T inverted as `Rz(−π/2)` / `Rz(−π/4)` (`adjoint.rs:147-148`) — correct up to global phase; test that gradients are unaffected.
- Wirtinger: ∂|z|²/∂z* = z (exists); chain rule on `norm_sq ∘ mul` vs numeric.

### 3.10 Pure backend
- For every operation pair present in both backends: Rust ≡ pure on the same inputs. Compare with bits where the algorithms are claimed identical, and with a stated tolerance where they differ. The docs contradict each other here ("bit-identical" at `QUANTUM_SIMULATION.md:417` vs "match to 1e-10" at `:430`), so the test result settles the question.

## 4. Integration / parity tests (proposed)

### 4.1 Executor parity (eval ≡ MIR)
- One parity test **per dispatch name** (78), generated from a table so a new
  builtin without a row fails a meta-test that diffs the table against
  `dispatch.rs` match arms.
- Compare values structurally with `f64::to_bits` (not `Display`).
- Include `cjcl run --mir-opt` as a third column, recorded but not gated.

### 4.2 Backend parity (Rust ≡ pure)
- `qubits(n)` vs `qubits(n,"pure")` for n ≤ 12 on W2-style random circuits: probabilities bits-equal or documented tolerance.
- MPS, stabilizer, density: same op sequence on both backends; compare `mps_z_expectation`, measurement outcomes, `density_probs`.
- **Backend-mismatch tests:** each op that the pure backend lacks (`q_sample`, `q_amplitudes`, `q_toffoli`, `q_run`→consumer, `mps_energy`, `mps_*canonicalize`, `mps_swap`, `q_fermion_lih`, `q_trotter_*`) must return a clear `Err` naming the backend. Current behaviour is recorded in SURFACE_AUDIT §7 (probe p14).

### 4.3 Statistical tests (seeded, reproducible)
- H|0⟩ and Ry(θ)|0⟩ for θ ∈ {π/3, π/2, 2π/3}: 10⁴ shots, two-sided binomial
  test at α = 1e-6, with seed fixed. Because the seed is fixed, the test is
  deterministic. Its role is to catch a biased sampler, not to recheck
  randomness on every run.
- Bell/GHZ n ∈ {2..8}: zero probability mass on forbidden outcomes over 10⁴ shots (exact).
- Chi-squared goodness-of-fit on a 3-qubit random state vs `q_probs` (α = 1e-6).
- Different seeds are non-degenerate: 64 seeds × 100 shots → ≥ 60 distinct sample vectors.

### 4.4 Composability tests (language-level)
- `q_trotter_evolve` result fed to `q_probs` / `q_fermion_expectation` — currently **not possible** (SURFACE_AUDIT §5); test is written as expected-Err now, flipped to expected-value when fixed.
- Circuit aliasing: `let b = q_x(a, 0); q_probs(a)` — pins whether gate builtins mutate their input in place (probe p10). Whichever semantics is chosen must be documented and tested.

### 4.5 Demo gate
- Add demos 04–06 to `tests/quantum_and_piml_demos_parity.rs` (T-7).

## 5. Property tests (`proptest!`, proposed location `tests/beta_tests/quantum_prop/`)

| Property | Generator | Oracle |
|---|---|---|
| Normalization preserved | random gate sequences (len ≤ 64, n ≤ 10, all 13 gates, angles ∈ [−4π, 4π]) | Kahan Σ\|α\|² within 64·n_gates ulp of 1 |
| Inverse recovers state | random sequence S then S† | L∞ ≤ 1e-12 vs original |
| Probabilities sum to one | same | `q_probs` sum ≤ 1e-12 from 1 |
| Replay bits-equal | random circuit + seed | two runs `to_bits` equal (probs, samples, measure) |
| Seed sensitivity | random circuit with ≥1 H; seeds s≠t | sample vectors differ for ≥ 1 of 16 seed pairs |
| eval ≡ MIR | random small `.cjcl` programs (gate chains + one observable) | bits-equal |
| Rust ≡ pure | random circuits n ≤ 8 | per §3.10 |
| MPS ≡ statevector | random {H, X, Ry, adjacent CX} n ≤ 10, χ = 2^(n/2) (exact) | amplitudes ≤ 1e-10 up to global phase |
| Density trace/PSD | random gates + channels (p ∈ [0,1]) n ≤ 5 | trace ≤ 1e-12 from 1; min eig ≥ −1e-12 |
| Density ≡ statevector (noise-free) | random gates n ≤ 6 | ρ = \|ψ⟩⟨ψ\| ≤ 1e-12 |
| Stabilizer ≡ dense | random Clifford n ≤ 10 | statevector up to global phase |
| Bell/GHZ correlation | n ∈ 2..10, random seed | all measured bits equal |
| Trotter order | random 2–3 qubit Pauli Hamiltonians (≤ 6 terms) | err₂ ≤ err₁ + 1e-12 at same steps |

## 6. Fuzz tests (seeded loops or `proptest`, proposed `tests/beta_tests/fuzz/`)

All go through `dispatch_quantum` (or `.cjcl` source) so the language boundary is exercised:

- **Random valid circuits** (gate + qubit + angle from full range incl. ±inf/NaN angles) → no panic; NaN angle → `Err` or documented NaN propagation.
- **Random invalid circuits:** out-of-range qubits, negative qubits, `usize`-overflow ints, duplicate operands (`q_cx(c,0,0)`, `q_toffoli(c,1,1,2)`, `q_swap(c,2,2)`) → `Err`.
- **Random API argument shapes:** for each of 78 builtins, arity 0..(k+2) with values drawn from {Int, Float, Bool, String, Array, nested Array, QuantumState of the *wrong* kind} → never panics.
- **Measurement seeds:** i64::MIN, −1, 0, i64::MAX → deterministic, no panic.
- **Random small Hamiltonians** (Pauli strings, n ≤ 4) → expectation real within 1e-12, Trotter no panic.
- **Random graph/QAOA inputs** once `Graph::new` is exposed; today: `qaoa_graph_cycle(n)` for n ∈ {−1, 0, 1, 2, 3, 1e6}.
- **Random noise parameters:** p ∈ {−1, −0.0, 0, 1, 1+ε, NaN, ∞} for all three channels → `Err` for out-of-domain (today: panic, probe p01).
- **Malformed backend objects:** pure object into Rust-only op and vice versa; `Graph` into `mps_h`; `SurfaceCode` into `stabilizer_measure` → `Err`.

## 7. Bolero targets (proposed `tests/bolero_fuzz/quantum_*.rs`, registered in `tests/bolero_fuzz/mod.rs`)

Each target must **not** swallow assertion failures (T-1).

| Target | Input | Property |
|---|---|---|
| `bolero_dispatch_no_panic` | (builtin index, arg vector of arbitrary `Value`s) | `std::panic::catch_unwind(dispatch_quantum(..)).is_ok()` — panic ⇒ fail |
| `bolero_circuit_replay` | byte string → decoded circuit (n ≤ 8) + seed | two executions: `probs`, `sample(16)`, `measure` bits-equal |
| `bolero_normalization` | same | Σp within 1e-12 of 1; no NaN/Inf unless an input angle was non-finite |
| `bolero_density_trace` | decoded gate+channel program (n ≤ 4) | trace ≈ 1, hermitian, min eig ≥ −1e-12; no NaN |
| `bolero_mps_vs_dense` | decoded {H,X,Ry,adj CX} program (n ≤ 8) | amplitudes agree (exact χ) |
| `bolero_stabilizer_replay` | Clifford program (n ≤ 128) + seed | replay bits-equal; tableau invariants hold |
| `bolero_eval_mir_parity` | decoded tiny `.cjcl` program (≤ 12 statements from a grammar of quantum builtins) | eval output ≡ MIR output (bits) or both `Err` |
| `bolero_bounded_alloc` | constructor args (n) | constructors refuse n beyond documented caps *before* allocating (measure: tracking allocator high-water mark < cap-derived bound) |
| `bolero_zne` (fix existing) | scales/values | `Err` on duplicate/non-finite scales; deterministic bits otherwise |
| `bolero_fermion` (fix existing) | amplitudes | expectation real, finite, deterministic bits |

## 8. Negative / error tests — required list

Every row below must end in `Err(..)` surfaced as a CJC runtime error (not a
process abort). The "today" column is from the probe results in
SURFACE_AUDIT §7.

| Case | Example | Today |
|---|---|---|
| Invalid qubit (circuit) | `q_h(qubits(2), 7)` then `q_probs` | see §7 p09 |
| Invalid qubit (MPS/stabilizer/density) | `stabilizer_measure(stabilizer_new(2), 5, 1)` | see §7 p03 |
| Duplicate operands | `q_cx(c, 0, 0)` | see §7 p08 |
| Non-adjacent MPS CNOT | `mps_cnot(m, 0, 2)` | see §7 p02 |
| Invalid probability | `density_depolarize(d, 0, 1.5)` | see §7 p01 |
| Size caps | `density_new(15)`, `mps_new(0)`, `mps_new(-1)`, `density_new(20,"pure")` | see §7 p04/p05/p17/p22 |
| Unsupported gate / string enum | `density_gate(d,"Rx",0)`, `mps_energy(m,"heisenbrg")`, `q_scale_noise(..,"amplitude-damping")`, `q_trotter_error(h,1.0,4,3)` | see §7 p16/p20/p24 |
| Malformed circuit/data objects | `qml_train(..., [[0.1,"x"],...], ...)`, `qec_decode([2,"a"], code)` | see §7 p19 |
| Wrong arity | `q_h(c)`, `mps_ry(m, 0)` | see §7 p11/p12 |
| Hamiltonian/state size mismatch | `q_trotter_evolve(q_fermion_h2(), qubits(3), ...)` | see §7 p06 |
| Degenerate parameters | `q_trotter_evolve(..., n_steps = 0, ...)`, `q_zne_mitigate([1,1],[..])` | see §7 p07/p23 |
| Seed handling | negative seeds, float seeds (`extract_int` truncates floats silently, `dispatch.rs:1477`) | see §7 p21 |
| Backend mismatch | `q_sample(qubits(2,"pure"), 5, 1)` | see §7 p14 |
| Surface-code decode | `qec_decode(syn, qec_surface_code(3))` | see §7 p15 |

## 9. Exit criteria before any public claim

| Claim type | Required gates |
|---|---|
| "Feature X is available in CJC-Lang" | ≥1 eval≡MIR parity test for every builtin of X + negative tests of §8 for X |
| "Deterministic / bit-identical" | §5 replay property + Bolero replay target green. "Across platforms" additionally requires an identical `verification/libm_bits.rs` hash on Windows, Linux, and macOS; this **fails** today: Windows and Linux differ on 6.0% of angles (VERIFY_FOLLOWUPS §2) |
| "Correct" for a simulator family | §3 unit oracle tests + §5 cross-representation property (MPS≡dense, density≡dense, stabilizer≡dense) |
| "Pure backend equivalent" | §3.10 + §4.2 green with the tolerance stated in the doc |
| Any performance statement | BENCHMARK_PLAN §5 schema-valid record with `replay_ok = true` |
