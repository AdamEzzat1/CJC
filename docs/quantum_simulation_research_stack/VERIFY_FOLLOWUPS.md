# Audit Follow-ups — Verification Results (2026-09-24)

This file resolves three open items from the first audit pass:

1. Chemistry reference energies (H₂, and also LiH).
2. Cross-platform bit-identity of rotation gates.
3. The external facts marked **[verify]** in EXTERNAL_COMPARISON.

All scripts, inputs, and outputs are in [`verification/`](verification/).

**Environments**

- **Windows:** Windows 11 x86-64, `rustc 1.97.1` (`x86_64-pc-windows-msvc`),
  Python 3.11.3, numpy 2.2.6, scipy 1.16.1, mpmath 1.3.0.
- **Linux:** Docker containers on the same machine.
  - `rust:1-slim-bookworm`: `rustc 1.98.1`, `x86_64-unknown-linux-gnu`, glibc 2.36.
  - `python:3.11-slim`: PySCF 2.14.0, OpenFermion 1.8.1, numpy 2.4.6.
  - PySCF is not buildable on Windows: pip's CMake step failed, and that also
    blocked OpenFermion.

---

## 1. Chemistry Hamiltonians (`fermion.rs`)

### Method

The CJC matrices are **extracted from CJC's own code**, not transcribed.
[`verification/ham_dump/`](verification/ham_dump/) is a throwaway Rust binary
that calls `FermionicHamiltonian::expectation` (the path `q_fermion_expectation`
uses) and recovers every matrix element by polarization:

- ⟨i|H|i⟩ directly.
- Re H_ij from (|i⟩+|j⟩)/√2.
- Im H_ij from (|i⟩+i|j⟩)/√2.

The output is [`cjc_hamiltonians.txt`](verification/cjc_hamiltonians.txt), and
both matrices are Hermitian to ≤ 7×10⁻¹⁵.

Two independent references were used:

1. [`h2_sto3g_check.py`](verification/h2_sto3g_check.py): from-scratch STO-3G
   integrals (Szabo & Ostlund), JW, and exact diagonalisation. It uses numpy and
   scipy only and runs on Windows.
2. [`pyscf_openfermion_check.py`](verification/pyscf_openfermion_check.py): PySCF
   RHF + FCI, with integrals transformed and JW-mapped by OpenFermion (the
   openfermionpyscf recipe), in Linux.
   Output: [`pyscf_openfermion_check.out.txt`](verification/pyscf_openfermion_check.out.txt).

For H₂ the two references agree to all printed digits.

### H₂ at R = 0.7414 Å

| Quantity | PySCF / OpenFermion | From-scratch | CJC (extracted) | Δ (CJC − ref) |
|---|---|---|---|---|
| E_nuc = 1/R | 0.713754 | 0.713754 | not included | — |
| RHF total | −1.116684 | −1.116684 | — | — |
| FCI ground, electronic | −1.851024 | −1.851024 | −1.851199 | **−1.75 × 10⁻⁴ Ha** |
| FCI ground, total | −1.137270 | −1.137270 | −1.137445 (min + E_nuc) | −1.75 × 10⁻⁴ Ha |
| Other levels | N=2: −1.246233 (×3), −0.883655, −0.233918 | same | −0.252801, 0.0 (N=0), 0.1824 (N=4) | N=2 excited: +1.9×10⁻²; N=4: −2.4×10⁻² |

**Verdict: H₂ is correct for the ground state only.**

- The coefficients form an electronic-only Hamiltonian. The docstring's −1.1373
  Ha is the total energy, so the `g0` comment "Nuclear repulsion + constant"
  (`fermion.rs:378`) is wrong.
- The ground state is within 0.175 mHa of FCI, about 9× inside chemical
  accuracy.
- The excited and N=4 levels are off by 19–24 mHa. Don't use this matrix for
  Trotter-dynamics or excited-state accuracy claims.

### LiH at R = 1.546 Å — **new finding: the Hamiltonian does not describe LiH**

| Quantity | PySCF | CJC (extracted, `lih_hamiltonian()`) |
|---|---|---|
| E_nuc | **+1.026864** | constant term **−7.4983** (labelled "nuclear_repulsion") |
| RHF total | −7.863134 | — |
| FCI total, full space (12 qubits) | **−7.882762** (matches docstring −7.8825) | — |
| FCI, 4-qubit active space (2e/2o, frozen Li 1s), total | −7.863374 | N=2 ground **−10.009166** (−2.15 Ha vs active space) |
| Lowest eigenvalue any N | — | −11.137500 (N=4) |
| Vacuum (N=0) | E_nuc = +1.027 | −7.4983 |

The CJC spectrum by particle number is in
[`cjc_lih_sectors.out.txt`](verification/cjc_lih_sectors.out.txt). The sectors
don't mix (inter-sector couplings are ≤ 5×10⁻¹⁵).

**Verdict: `q_fermion_lih` is not a LiH Hamiltonian.**
- Every relevant eigenvalue is 2–3 Ha below the true FCI energy. That is
  unphysical: it violates the variational bound against the real molecule.
- The "nuclear repulsion" constant has the wrong sign and the wrong magnitude.
- The one-body diagonal (−1.2528, −0.4760) is essentially H₂'s MO energies
  (−1.2525, −0.4759).
- The docstring's −7.8825 Ha is the true LiH FCI energy, but this matrix doesn't
  reproduce it.
- **Recommendation:** remove `q_fermion_lih`, or regenerate it from PySCF
  integrals (active space documented, script committed), and assert the
  reference energies in tests.

## 2. Cross-platform bit-identity of rotation gates — **fails**

### Method

- [`verification/libm_bits.rs`](verification/libm_bits.rs) evaluates the exact
  expressions the gates use, `(θ/2).cos()` and `(θ/2).sin()`
  (`gates.rs::rx_matrix/ry_matrix/rz_matrix`). It covers 20 fixed angles, 200,000
  seeded angles in [−4π, 4π], and 20,000 in [−10⁶, 10⁶], and prints the bit
  patterns. It was run on Windows (MSVC) and Linux (glibc).
- [`libm_check.py`](verification/libm_check.py) compares each dump with the
  correctly rounded values (mpmath, 300 bits).
- [`libm_diff.py`](verification/libm_diff.py) compares the two dumps directly.
- The 11 MB dumps are not committed; the emitter regenerates them. SHA-256:
  - Windows: `213e435b6ac1e429698aa29d5eecb0ebe89dc0f09e460660b3fcb112eedabbcb`
  - Linux: `943b70e8235988a39446fe34882d74880a11a82528ee021b5a37daa222503306`

### Results

| | Windows (MSVC CRT) | Linux (glibc 2.36) |
|---|---|---|
| Evaluations not correctly rounded (of 440,040) | 13,650 (3.1%) | 582 (0.13%) |
| Max error | 1 ulp | 1 ulp |

Windows vs Linux ([`libm_diff_windows_vs_linux.out.txt`](verification/libm_diff_windows_vs_linux.out.txt)):
**13,208 of 220,020 angles (6.0%) produce at least one different bit.** Of
those, 12,030 are in the gate-realistic range |θ| ≤ 4π. One of them is
`θ = π/2`: `sin(π/4)` is `0.7071067811865476` on Windows and
`0.7071067811865475` on Linux.

**End-to-end CJC output:** [`ry_half_pi.cjcl`](verification/ry_half_pi.cjcl),
`q_probs(q_ry(qubits(1), 0, π/2))`.

| Platform | `cjcl` output |
|---|---|
| Windows, release build of this branch | `[0.5000000000000001, 0.5000000000000001]` |
| Linux (container, glibc 2.36), release build of `HEAD` (`671dfeb`) | `[0.5000000000000001, 0.4999999999999999]` |

### Conclusions

1. The "same seed = bit-identical **across platforms**" claim (`lib.rs:4`,
   `measure.rs:25`, `docs/QUANTUM_SIMULATION.md`) is **false for circuits with
   rotation gates**. The platform math libraries return different bits for ~6%
   of gate angles, including π/2.
2. **Correction to the first pass.** I had noted that glibc's sin/cos are
   "widely reported to be correctly rounded". glibc 2.36 is **not** perfectly
   correctly rounded here (0.13% of evaluations are off), though it is ~24×
   closer than MSVC. Neither platform's libm can serve as the determinism
   anchor.
3. Same-platform, same-toolchain determinism is unaffected.

**Recommendation (MISSING_FEATURES P1-8):**
- Implement a pure-Rust, fixed-algorithm sin/cos in CJC, ideally correctly
  rounded, and use it for all gate construction.
- CI gate: `libm_bits.rs` rewritten against that implementation must produce
  the same hash on Windows, Linux, and macOS.
- Until then, the docs must say "same platform and toolchain".

## 3. External **[verify]** items

| Item | Result | Evidence |
|---|---|---|
| QuEST v4.3.0 release date | **Confirmed**: published 2026-09-24T15:18:53Z (same day as access); v4.2.0 2025-10-14 | `api.github.com/repos/QuEST-Kit/QuEST/releases` |
| TensorCircuit `to_qir`/`from_qir` is LLVM QIR? | **Refuted.** Returns `List[Dict[str, Any]]` of gate objects and tensors, an internal Python IR. No LLVM/pyqir reference | `tensorcircuit-ng/tensorcircuit/abstractcircuit.py:375-431` (master) |
| qsim multi-GPU | **Resolved.** Released v0.22.1: `gpu_mode` is only 0 (CUDA) or 1 (cuStateVec), so multi-GPU means the cuQuantum Appliance. On `main` (unreleased), `cirq_interface.md:166-171` adds cuStateVecEx native multi-device/multi-node (Open MPI), while `choose_hw.md:54-57,87-88` still says the Appliance is required, so `main`'s docs disagree with each other | `raw.githubusercontent.com/quantumlib/qsim/{v0.22.1,main}/docs/…`, `qsimcirq/qsim_simulator.py@v0.22.1` |
| TensorCircuit-NG paper arXiv:2602.14167 | **Confirmed**: "TensorCircuit-NG: A Universal, Composable, and Scalable Platform for Quantum Computing and Quantum Simulation", Zhang, Chen, Li et al., submitted 2026-02-15 | `arxiv.org/abs/2602.14167` |
| TensorCircuit-NG latest release | **Confirmed**: v1.9.1, 2026-08-11 | GitHub releases API |
| qsim latest release | **Confirmed**: v0.22.1, 2026-09-01 | GitHub releases API |

---

## Linux end-to-end `cjcl` run

- **Build:** `cargo build --release --bin cjcl` in `rust:1-slim-bookworm` (rustc 1.98.1) on `git archive HEAD` (`671dfeb`).
- **Output:** `cjcl run ry_half_pi.cjcl` prints `[0.5000000000000001, 0.4999999999999999]`. Windows prints `[0.5000000000000001, 0.5000000000000001]`.
- **Executor parity:** `cjcl parity` on Linux reports **IDENTICAL**, so eval and MIR agree on each platform while the platforms disagree with each other.
- **Conclusion:** end-to-end proof that the cross-platform bit-identity claim is false as implemented.
