# CJC-Lang Quantum Simulation — External Comparison (classical simulation only)

**Date of external-source access:** 2026-09-24 · **CJC branch:** `claude/cjc-quantum-audit-5a7f9c` (`671dfeb`)
**Framing:** CJC-Lang targets classical simulation on conventional hardware
(CPU-first; commodity GPU/distributed as future work). QPU/provider execution is
**out of scope** and is not scored. Circuit interchange *is* scored, because it
enables cross-validation against other classical simulators.
**Companion docs:** [SURFACE_AUDIT.md](SURFACE_AUDIT.md) · [MISSING_FEATURES.md](MISSING_FEATURES.md) ·
[BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) · [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md)

> **No performance comparison is made in this document.** Nothing here has been
> measured against CJC. External scale and speed figures are the projects' own
> claims, quoted with sources. They must not be placed next to CJC numbers until
> the harness in BENCHMARK_PLAN produces same-machine measurements.

> **Source-quality caveat.** External facts were gathered from official docs,
> READMEs, release API data, and papers (URLs below). Several pages were read
> through a summarising fetch tool, and PyPI pages were unreachable, so release
> data comes from the GitHub releases API. Verify quoted phrases against the
> source before quoting them publicly. The items originally flagged **[verify]**
> have since been checked; see [VERIFY_FOLLOWUPS §3](VERIFY_FOLLOWUPS.md#3-external-verify-items).

---

## 1. Feature matrix

Legend: ✓ documented/implemented · ◐ partial or limited (see note) · ✗ not
present · — not applicable · ? not found in docs consulted.
For CJC, "✓" means **exposed to `.cjcl`**; crate-only is marked ◐ with a note
(evidence in SURFACE_AUDIT).

| Capability | CJC-Lang (`cjc-quantum`) | Qiskit Aer | Cirq + qsim | PennyLane Lightning | QuEST v4 | Qulacs | Stim | TensorCircuit(-NG) | cuQuantum (GPU lib) |
|---|---|---|---|---|---|---|---|---|---|
| Dense statevector | ✓ ≤26 q (hard cap), single-thread scalar | ✓ | ✓ (qsim) | ✓ (qubit/kokkos/gpu) | ✓ | ✓ | ✗ | ✓ | ✓ cuStateVec |
| Density matrix | ✓ ≤14 q (Rust), 6 1q gates + CNOT | ✓ | ✓ (Cirq DM sim) | ? (not documented for qubit/tensor) | ✓ | ✓ | ✗ | ✓ `DMCircuit` | ✓ cuDensityMat (Lindblad) |
| Noise channels | ◐ 3 fixed 1q channels from `.cjcl`; general Kraus crate-only | ✓ broad library + `NoiseModel`, readout error | ✓ Cirq channels; qsim via trajectories | ? | ✓ dephasing, depolarising, damping, Pauli, Kraus, superop | ✓ broad incl. CPTP, instruments | ◐ Pauli noise only (by design) | ✓ incl. readout | ✓ (cuDensityMat, cuPauliProp) |
| Trajectory (MC wavefunction) noise | ✗ | ✓ | ✓ (qsim) | ? | ? | ✓ (probabilistic gates) | — | ✓ | ? |
| MPS / tensor network | ◐ 1D MPS; H, X, Ry, adjacent CNOT, SWAP only; DMRG builtin | ✓ `matrix_product_state` (+ GPU `tensor_network`) | ◐ qsim MPS mentioned; quimb/qFlex external | ✓ `lightning.tensor` (cuTensorNet) | ✗ | ✗ | ✗ | ✓ `MPSCircuit`, TN core | ✓ cuTensorNet |
| Stabilizer / Clifford | ✓ CHP tableau (H S X Y Z CNOT, 1q measure) | ✓ + extended stabilizer | ✓ `CliffordSimulator` | ✗ | ✗ | ✗ | ✓ tableau + Pauli-frame batch sampling | ✓ `StabilizerCircuit` (stim backend) | ◐ cuStabilizer (details not retrievable) |
| QEC tooling (DEM, decoders) | ◐ repetition + surface layout; repetition decoder only; code-capacity noise | ? | ✗ | ✗ | ✗ | ✗ | ✓ detector error models, sinter, PyMatching integration | ◐ via stim | ✗ |
| Gradients | ◐ adjoint (crate-only, diagonal obs); param-shift inside VQE/QAOA; FD in QML | ? (ecosystem) | ? | ✓ adjoint native; + PennyLane param-shift/backprop | ? | ✓ `backprop`, `GradCalculator` | — | ✓ JAX/TF/PyTorch autodiff | ✓ cuTensorNet/cuDensityMat/cuPauliProp backward |
| VQE / QAOA / QML | ◐ closed-loop builtins, final value only; QAOA cycle graphs only | ◐ via Qiskit ecosystem | ◐ via Cirq ecosystem | ✓ via PennyLane | ? | ◐ parametric circuits | — | ✓ | — (library) |
| Fermion / chemistry | ◐ H₂ (ground state verified vs PySCF); "LiH" matrix is not LiH (VERIFY_FOLLOWUPS §1); JW helpers crate-only | ◐ ecosystem | ◐ ecosystem (OpenFermion) | ◐ PennyLane qchem | ✗ | ✗ | — | ◐ | — |
| Time evolution | ◐ Trotter 1st/2nd order (output not consumable from `.cjcl`) | ◐ ecosystem | ◐ | ◐ | ✓ Trotter, imaginary-time, Lindbladian (v4.2) | ◐ `NoisyEvolution` | — | ✓ `AnalogCircuit` | ✓ cuDensityMat |
| Error mitigation (ZNE) | ✓ Richardson/linear extrapolation helpers | ✗ (ecosystem) | ✗ (ecosystem) | ◐ PennyLane transforms | ✗ | ✗ | — | ? | — |
| Multithreading | ✗ none in crate | ✓ OpenMP | ✓ OpenMP | ✓ OpenMP | ✓ OpenMP | ✓ OpenMP | ◐ multi-process via sinter | ◐ backend-dependent | — |
| SIMD | ◐ AVX2 kernels exist but **unused** | ? | ✓ AVX/FMA | ✓ AVX2/AVX512 | ◐ BMI2 option (v4.3) | ✓ AVX2 | ✓ 256-bit AVX | backend | — |
| GPU | ✗ | ✓ CUDA (Linux x86_64 wheels) | ✓ CUDA / cuStateVec | ✓ CUDA, HIP, Kokkos | ✓ CUDA, HIP | ✓ `qulacs-gpu` | ✗ | ✓ | ✓ (is the GPU layer) |
| Multi-GPU / distributed | ✗ | ✓ MPI + cache blocking | ◐ released v0.22.1: multi-GPU only via cuQuantum Appliance; native cuStateVecEx on unreleased `main` (verified) | ✓ MPI (gpu, kokkos) | ✓ MPI, multi-GPU | ✓ MPI | ✗ | ✓ multi-node multi-GPU | ✓ MPI (cuStateVec Ex, cuTensorNet) |
| OpenQASM import/export | ✗ | ◐ via Qiskit core (not Aer) | ✓ 2.0/3.0 (importer "experimental", subset) | ✓ `from_qasm`, `from_qasm3`, `to_openqasm` | ✗ | ◐ `qulacs.converter` (unverified) | ✗ | ✓ `from_openqasm` | — |
| QIR (LLVM) | ✗ | ? | ? | ? | ? | ? | ? | ✗ (`to_qir` is an internal Python gate-list IR, not LLVM QIR — verified in source) | ? |
| Native circuit format | ✗ (no serialization; QuantumState non-serializable) | Qiskit objects | Cirq JSON | PennyLane tapes | — | JSON/pickle | ✓ `.stim`, `.dem` | JSON, from_qiskit/cirq | — |
| Seeded sampling | ✓ explicit `seed` arg on every stochastic builtin | ✓ `seed_simulator` | ✓ `seed` | ✓ `seed` (NumPy RNG semantics) | ✓ `setQuESTSeeds` | ✓ `random_seed` | ✓ `--seed` | ◐ backend RNG + `status` | ◐ caller-supplied randoms |
| Documented reproducibility contract | ◐ **claims** bit-identical same-seed across runs *and platforms*; same-process replay tested; cross-platform claim **fails as implemented** (Windows vs Linux sin/cos differ on 6.0% of gate angles, VERIFY_FOLLOWUPS §2) | ✗ none found; open issues on seed handling | ✗ | ✗ | ◐ seeds broadcast from root under MPI; default seeding non-reproducible | ✗ | ✓ explicit: same version + flags + architecture only; not across versions; may differ AVX vs SSE | ✗ | ✗ |
| Executor/language integration | ✓ builtins in a language with two executors (AST-eval, MIR); eval≡MIR parity-tested for 51/78 names | Python API | Python API | Python API | C/C++ API | Python/C++ | Python/C++/CLI | Python | C/C++/Python |
| License | (repo license) | Apache-2.0 | Apache-2.0 | Apache-2.0 | MIT | MIT | Apache-2.0 | Apache-2.0 | BSD-3 (repo); **proprietary** SDK binaries/headers |
| Latest release seen | 0.1.11 workspace (CLI reports 0.1.4) | 0.17.2 (2025-09-17) | qsim 0.22.1 (2026-09-01); Cirq 1.7.0 (2026-06-30) | 0.45.0 (2026-05-11) | 4.3.0 (2026-09-24, verified) | 0.6.14 (2026-07-29) | 1.16.0 (2026-05-22) | NG 1.9.1 (2026-08-11) | 26.09.0 (2026-09-10) |

## 2. Reading the matrix honestly

**Where CJC-Lang is differentiated (narrow, evidence-backed):**
1. **Determinism is a design contract, not an option.**
   - Every stochastic builtin takes an explicit seed.
   - Reductions use Kahan summation.
   - Complex multiplication avoids FMA (`mul_fixed`).
   - Iteration order is fixed.
   - Among the externals, only Stim writes down an equally explicit reproducibility rule. That rule is *weaker by admission*: it covers the same version, flags, and architecture only. CJC's stronger cross-platform claim **does not hold as implemented**. Rotation gates use the platform libm, and Windows and Linux return different bits for 6.0% of gate angles (VERIFY_FOLLOWUPS §2). Today CJC's honest contract is "same platform and toolchain", comparable to Stim's.
2. **Language-level integration with dual-executor parity.**
   - Quantum builtins run identically under AST-eval and MIR-exec because both call one dispatch function.
   - Every non-crashing probe in this audit agreed across executors.
   - No external simulator has an equivalent concept. It matters for reproducible scientific workflows, but not for simulation capability.
3. **Breadth of algorithm families in one dependency-free crate.**
   - Statevector, MPS/DMRG, stabilizer, density, QEC, VQE/QAOA/QML, fermion/Trotter, and ZNE all live in one crate with zero external dependencies.
   - Breadth is not depth, though. Most families are **partial** at the language level (SURFACE_AUDIT §4).

**Where CJC-Lang is behind (every row is a documented gap, not a guess):**
- **No parallelism of any kind.** No threads or GPU, and the SIMD kernels are unused. Every mature dense simulator ships OpenMP and SIMD, and most ship GPU. Expect a large throughput gap on dense workloads; the size is **unknown until measured**.
- **No circuit interchange.** This blocks cross-validation against every other simulator here: comparison circuits must be hand-transcribed or generated twice (BENCHMARK_PLAN §3).
- **Noise modelling:**
  - Three fixed channels from `.cjcl`.
  - No noise-model object, readout error, or trajectories.
  - The depolarizing parameterisation differs from Qiskit's by a 4/3 factor (SURFACE_AUDIT §6.3).
- **QEC:**
  - No detector error model, circuit-level noise, or matching decoder.
  - The surface-code decode path gives wrong answers.
  - The relevant bar is Stim + PyMatching; the gap is structural, not incremental.
- **Differentiable workflows:** the adjoint method exists in Rust but is not exposed. PennyLane Lightning (native adjoint) and TensorCircuit (JAX/TF/PyTorch) are the reference points.
- **Robustness:** nine probed malformed inputs crash the process (SURFACE_AUDIT §7). Mature simulators raise language-level exceptions for all equivalent inputs. This comes from general knowledge of Python API conventions and was **not probed** for the externals.

**Not a gap under this audit's framing:** no QPU/provider backends; no pulse-level control.

## 3. Per-project notes (sources)

All URLs accessed 2026-09-24.

### Qiskit Aer
- Methods: automatic, statevector, density_matrix, stabilizer, extended_stabilizer, matrix_product_state, unitary, superop, tensor_network (GPU/cuTensorNet). — https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
- Noise: `NoiseModel`, `QuantumError`, `ReadoutError`, `PauliLindbladError`, plus depolarizing, amplitude/phase damping, thermal relaxation, Pauli, Kraus, coherent-unitary, and reset errors. — https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html
- Parallelism:
  - OpenMP; `statevector_parallel_threshold` defaults to 14.
  - GPU and `cuStateVec_enable`.
  - MPI with cache blocking. — https://qiskit.github.io/qiskit-aer/howtos/running_gpu.html
- Status: the README says Aer is in "reduced maintenance". — https://github.com/Qiskit/qiskit-aer
- Seeding: `seed_simulator` exists, but open issues report inconsistent seeding in the V2 primitives. — https://github.com/Qiskit/qiskit-aer/issues/2453, https://github.com/Qiskit/qiskit-aer/issues/1916
- Docs inconsistency: the release-notes page is titled 0.17.1 while 0.17.2 is released. — https://qiskit.github.io/qiskit-aer/release_notes.html, https://api.github.com/repos/Qiskit/qiskit-aer/releases

### Cirq + qsim
- Cirq has pure-state, density-matrix, and Clifford simulators. — https://quantumai.google/cirq/simulate/simulation, https://quantumai.google/reference/python/cirq/CliffordSimulator
- qsim:
  - Schrödinger statevector simulator with AVX/FMA, OpenMP, and gate fusion. — https://github.com/quantumlib/qsim
  - Noise is simulated as trajectories. — https://quantumai.google/qsim/tutorials/noisy_qsimcirq
- Hardware guidance (the project's own claims):
  - About 30 qubits in about 16 GB of RAM. — https://quantumai.google/qsim/overview
  - 32–36 qubits on A100 configurations. — https://quantumai.google/qsim/choose_hw
- **Docs disagree on multi-GPU:**
  - `choose_hw` says it requires the cuQuantum Appliance.
  - `docs/cirq_interface.md` on `main` describes native cuStateVecEx multi-device support.
  - I could not confirm that support in a tagged release.
- Interchange: OpenQASM 2.0/3.0 import and export; the importer is "experimental" and a subset. — https://quantumai.google/cirq/build/interop

### PennyLane Lightning
- Devices: lightning.qubit, lightning.kokkos, lightning.gpu, lightning.amdgpu, and lightning.tensor. — https://docs.pennylane.ai/projects/lightning/en/stable/
- SIMD and threads: AVX2/AVX512 and OpenMP. — https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html
- Gradients:
  - Native adjoint on qubit, kokkos, and gpu; lightning.tensor supports parameter-shift and finite differences only. — https://docs.pennylane.ai/projects/lightning/en/stable/lightning_tensor/device.html
  - Interfaces: Autograd, PyTorch, JAX, and TensorFlow (deprecated as of v0.44). — https://docs.pennylane.ai/en/stable/introduction/interfaces.html
- MPI: lightning.gpu and lightning.kokkos. — https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html, https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html
- QASM: `from_qasm`, `from_qasm3`, `to_openqasm`. — https://docs.pennylane.ai/en/stable/code/api/pennylane.from_qasm.html
- Seeding: NumPy-generator semantics; the default draws from the global generator. — https://docs.pennylane.ai/projects/lightning/en/stable/code/api/pennylane_lightning.lightning_qubit.LightningQubit.html

### QuEST (v4)
- Methods and noise: statevector and density-matrix `Qureg`; exact channels (dephasing, depolarising, damping, Pauli, Kraus maps, superoperators). — https://quest-kit.github.io/QuEST/, https://github.com/QuEST-Kit/QuEST
- Acceleration: OpenMP, MPI, CUDA, HIP, and cuQuantum.
- Seeding: `setQuESTSeeds`, a Mersenne Twister. Under MPI only the root's seeds are used. The default seeding uses `std::random_device` and is not reproducible. — https://quest-kit.github.io/QuEST/group__debug__seed.html
- Releases:
  - v4.2.0 added Trotter, imaginary-time, and Lindbladian functions.
  - v4.3.0 published 2026-09-24T15:18:53Z, the access date (verified via the releases API). — https://api.github.com/repos/QuEST-Kit/QuEST/releases
- The README says the docs are "still under construction".

### Qulacs
- State types: statevector, density matrix, `CausalConeSimulator`, and `NoiseSimulator`. — http://docs.qulacs.org/en/latest/pyRef/qulacs/index.html
- Noise: a broad gate library (CPTP, instruments, and more). — http://docs.qulacs.org/en/latest/pyRef/qulacs/gate/index.html
- Gradients: `ParametricQuantumCircuit.backprop` and `GradCalculator`. — http://docs.qulacs.org/en/latest/api/classParametricQuantumCircuit.html
- Acceleration: OpenMP, AVX2, `qulacs-gpu`, and MPI. — https://github.com/qulacs/qulacs
- Seeding: a `random_seed` argument on sampling.
- Benchmarks and paper: Suzuki et al., *Quantum* 5, 559 (2021), arXiv:2011.13524.

### Stim
- Simulation:
  - Stabilizer only: tableau simulation plus Pauli-frame batch sampling.
  - Pauli noise only, by design. — https://github.com/quantumlib/Stim, https://github.com/quantumlib/Stim/blob/main/doc/gates.md
- QEC tooling:
  - Detector error models.
  - `sinter` for parallel Monte Carlo. — https://github.com/quantumlib/Stim/tree/main/glue/sample
  - PyMatching integration via `Matching.from_detector_error_model`. — https://github.com/oscarhiggott/PyMatching
- Paper claim (its own measurement): a distance-100 surface code with 20k qubits and 8M gates is analysed in 15 s, then sampled at about 1 kHz. — https://arxiv.org/abs/2103.02202
- Reproducibility rule: `--seed` gives exact reproducibility only with the same flags, version, and architecture. Results are not consistent across versions and may differ between AVX and SSE machines. — https://github.com/quantumlib/Stim/blob/main/doc/usage_command_line.md

### TensorCircuit / TensorCircuit-NG
- Project status: TensorCircuit-NG presents itself as the maintained successor. The original repo's last release is v0.12.0 (2024-03-15). — https://github.com/tensorcircuit/tensorcircuit-ng, https://api.github.com/repos/tencent-quantum-lab/tensorcircuit/releases
- Circuit types: statevector/TN, DM, MPS, stabilizer (stim-backed), qudit, analog, and fermionic Gaussian.
- Gradients: autodiff through JAX, TF, and PyTorch.
- Scale: GPU and multi-node multi-GPU (README).
- Speed claim: "10 to 10^6+ times acceleration" is self-reported.
- Paper: arXiv:2602.14167 (verified: "TensorCircuit-NG: A Universal, Composable, and Scalable Platform…", submitted 2026-02-15).
- Interchange: `from_openqasm`, `from_qiskit`, and `from_cirq`. `to_qir`/`from_qir` use TensorCircuit's own IR (a `List[Dict]` of gates), **not** LLVM QIR (verified in `abstractcircuit.py:375-431`). — https://tensorcircuit-ng.readthedocs.io/en/stable/api/circuit.html

### NVIDIA cuQuantum (reference for commodity-GPU acceleration)
- Components: cuStateVec (incl. multi-process "Ex" API), cuTensorNet (MPS/TN, MPI, gradients), cuDensityMat (Lindblad), cuPauliProp, and cuStabilizer. — https://docs.nvidia.com/cuda/cuquantum/latest/index.html
- The sampler takes **caller-supplied random numbers**. This pattern would let a CJC GPU backend keep CJC's SplitMix64 as the single source of randomness. — https://docs.nvidia.com/cuda/cuquantum/26.03.2/python/bindings/generated/cuquantum.bindings.custatevec.sampler_sample.html
- **Not fully open source:** SDK headers and binaries are under an NVIDIA proprietary license; the GitHub repo is BSD-3. — https://github.com/NVIDIA/cuQuantum

## 4. Open research questions raised by the comparison

1. Can CJC keep its bit-identical guarantee after adding threads and SIMD?
   - Stim's caveat (AVX vs SSE may differ) shows the difficulty.
   - A viable design needs a fixed reduction tree and a fixed partitioning that doesn't depend on thread count. CLAUDE.md already requires this for other crates.
2. Does `f64::sin/cos` from the platform libm break cross-platform bit-identity for rotation gates? (BENCHMARK_PLAN §6.4.)
3. Would a GPU backend based on cuStateVec (caller-supplied randoms) or a portable layer (Kokkos-style, as in lightning.kokkos) fit CJC's zero-external-dependency policy? It probably doesn't without a feature-gated optional crate. This is an architectural decision, not a gap to close silently.
4. Which QASM dialect to adopt first? OpenQASM 2.0 is the common denominator across Qiskit, Cirq, PennyLane, and TensorCircuit, and it's enough for W1–W7 benchmarks.
