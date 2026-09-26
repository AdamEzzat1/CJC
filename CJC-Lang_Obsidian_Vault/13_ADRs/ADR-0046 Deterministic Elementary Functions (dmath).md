# ADR-0046 Deterministic Elementary Functions (`cjc_repro::dmath`)

- **Status:** Accepted (2026-09-24). Scope: `cjc-quantum`. `cjc-runtime` builtins deferred (see "Not decided here").
- **Crates:** `cjc-repro` (new module `dmath`), `cjc-quantum` (59 call sites switched)
- **Companion docs:** `docs/quantum_simulation_research_stack/VERIFY_FOLLOWUPS.md` (the 6.0% Windows/Linux divergence measurement), `verification/dmath_check.py`, `verification/dmath_check/`
- **Related:** [[ADR-0004 SplitMix64 RNG]], [[ADR-0002 Kahan Accumulator]], [[ADR-0044 Quantum Value Semantics]]

## Context

CJC-Lang promises "same seed ⇒ bit-identical output". Every rotation gate
(`q_rx`, `q_ry`, `q_rz`, `mps_ry`, the QAOA/VQE/QML/Trotter ansätze, density
rotations) called `f64::sin` / `f64::cos`, which Rust forwards to the platform
C math library.

The quantum audit measured the consequence:

- Windows (UCRT) vs Linux (glibc 2.36) disagree on **6.0% of gate angles**
  (`sin`/`cos` of θ/2 over a 100k-angle sweep), including `sin(π/2 · k)` for
  some `k`. Hashes: Windows `213e435b…`, Linux `943b70e8…`.
- glibc is not correctly rounded either: it is off in 0.13% of evaluations.

So the same `.cjcl` program with the same seed produced different amplitudes
on different operating systems. Kahan summation, `mul_fixed` (no FMA), and
SplitMix64 were all in place. The libm was the one component that was not
deterministic across platforms.

IEEE 754 requires correct rounding only for `+ - * / sqrt`. Any function
built from those operations alone, evaluated in a fixed order with no fused
multiply-add, depends only on its input bits. Rust never contracts
`a * b + c` into an FMA unless the code calls `mul_add`.

## Decision

1. **Add `cjc_repro::dmath`** with `sin`, `cos`, `sin_cos`, `exp`, `ln`, `pow`, and `powi`, built from IEEE `+ - * /` and integer bit operations only.
   - `sin`/`cos`/`exp`/`ln`: fdlibm algorithms (Sun, 1993, as ported by musl; notice preserved in the module docs).
     - Polynomial kernels on [-π/4, π/4].
     - Cody–Waite 3-stage reduction for |x| < 2²⁰·π/2.
   - Large-argument reduction (|x| ≥ 2²⁰·π/2): new Payne–Hanek code.
     - It uses `u128` limb arithmetic in place of fdlibm's 24-bit float chunks.
     - The 1280-bit 2/π table is generated with mpmath. Its first words match fdlibm's published `ipio2` table.
   - `pow(x, y)` = `exp(y · ln x)`, with exact special cases and integer exponents routed to `powi`. Its error is about `|y ln x|` ulps. This is documented, and it is used only for noise-scale factors.
   - `powi`: fixed-order binary exponentiation.
2. **Switch every non-test transcendental call in `cjc-quantum` to `dmath`.** That is 59 sites: gates, density, dispatch (`mps_ry`), dmrg, mitigation, pure, qaoa, qml, trotter, and vqe. Test code keeps `std` with tolerances, because it computes expected values rather than simulator output.
3. **Accuracy target: < 1 ulp, not correct rounding.** Correct rounding (CORE-MATH style) would cost 3–10× in code size and speed. It is also unnecessary for determinism: any fixed algorithm is deterministic.

## Evidence

| Check | Result |
|---|---|
| mpmath comparison, 104k `sin` + 104k `cos` (gate range, kernel range, every exponent to 2¹⁰²³, k·π/2 for \|k\| ≤ 2000, the 6381956970095103·2⁷⁹⁷ hard case) | max error 0.72 ulp (sin), 0.75 ulp (cos) |
| mpmath comparison, 60k `exp` over [-745, 745] incl. subnormal results | max 0.83 ulp; overflow → `inf` correct |
| mpmath comparison, 80k `ln` (all exponents incl. subnormals, plus [0.5, 1.5]) | max 0.75 ulp |
| Bit-identity: 348,004-line output dump | Windows 11 (UCRT) and Debian 12 (glibc 2.36) **identical**: SHA-256 `e6a7c8ed77535c48…` |
| `golden_hash_is_platform_independent` unit test | passes on Windows and Linux; CI runs it on ubuntu/windows/macos-latest |
| Agreement with platform libm | ≤ 2 ulp everywhere (both sides < 1 ulp) |
| `cjc-quantum` unit tests after the switch | 280/280 pass unchanged |

Known fdlibm results reproduced exactly: `exp(1) = 2.7182818284590455`, which is one ulp above `E` at 0.67 ulp error, and `sin(π/6) = 0.5`, which is 0.90 ulp from 0.49999999999999995. These confirm the transcription is faithful; they are not bugs.

## Consequences

- **Positive:** Quantum simulation is now bit-identical across Windows, Linux, and macOS for the same seed and inputs. The earlier caveat in `docs/QUANTUM_SIMULATION.md` ("same platform and toolchain") can be lifted for `cjc-quantum`.
- **Behaviour change:** Quantum outputs on a given OS may change in the last bit. For example, Windows used to return `0.49999999999999994` for `sin(π/6)`. No quantum test pinned such bits.
- **Performance:** fdlibm kernels are roughly as fast as UCRT/glibc for |x| < 2²⁰. The rotation-gate cost is dominated by the O(2ⁿ) amplitude update anyway.
- **`cjc-repro` stays zero-dependency.**

## Not decided here

- **`cjc-runtime` builtins** (`sin`, `cos`, `exp`, `log`, `pow`, `tanh`, … exposed to `.cjcl`, plus `cjc-ad`'s dual and tape ops) still use platform libm. Switching them changes the last bits of every existing golden hash in the repo (chess RL weight hashes, PINN, ABNG canaries). That needs its own decision, a hash-migration plan, and `tan`/`atan2`/`tanh`/`asin`/… implementations. Tracked as a follow-up.
- `sqrt` needs nothing: IEEE requires it to be correctly rounded.
