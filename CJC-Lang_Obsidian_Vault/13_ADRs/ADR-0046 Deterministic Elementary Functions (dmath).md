# ADR-0046 Deterministic Elementary Functions (`cjc_repro::dmath`)

- **Status:** Accepted (2026-09-24). Scope: `cjc-quantum`. **Amended 2026-09-26:** extended to everything a `.cjcl` program can observe (see "Amendment 2026-09-26").
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

- ~~`cjc-runtime` builtins still use platform libm~~ — decided in the amendment below.
- `sqrt` needs nothing: IEEE requires it to be correctly rounded.
- ~~**Analytics crates** still call platform libm~~ — moved in the second amendment below.

## Amendment 2026-09-26 — runtime math builtins

**Decision.** Every transcendental a `.cjcl` program can observe now comes
from `dmath`: `cjc-runtime` (builtins, tensor ops, activations,
distributions incl. `randn`, stats), `cjc-ad` (autodiff forward and backward
ops, so they match the forward builtins), both executors' `**`, both MIR
constant folders, and `cjc-data` `DExpr` functions. 387 call sites, migrated
by renaming `x.exp()` → `x.det_exp()` through a new `DetMath` extension
trait (inherent `f64` methods would shadow same-named trait methods). The
trait exists only for `f64`, so every compiled rename is on an `f64`; the 12
renames that hit `GradGraph` node builders (`g.exp(node)`) failed to compile
and were reverted.

**New functions** (all needed by the runtime): `tan`, `asin`, `acos`,
`atan`, `atan2`, `sinh`, `cosh`, `tanh`, `atanh`, `exp_m1`, `ln_1p`, `log2`,
`log10`, `hypot`, and an accurate `pow`. The old `pow = exp(y · ln x)` lost
~|y · ln x| ulps (e.g. tens of ulps for `10.0 ** 20`) — acceptable for
quantum noise factors, not for the language's `**`. It is replaced by
fdlibm `e_pow.c`. Sources: musl (fdlibm ports and musl's own
`tanh`/`sinh`/`cosh`/`atanh`/`hypot`, MIT), FreeBSD `msun` for `log2` and
`pow` (musl replaced both with table-driven code).

**Evidence.**

| Check | Result |
|---|---|
| mpmath, 609,860 evaluations (`verification/dmath_ext_check.py`), inputs concentrated on each algorithm's branch thresholds | < 1 ulp: `tan asin acos atan exp_m1 ln_1p log2 log10 pow hypot` (pow 0.81, log2 0.74, tan 0.75). Above 1 ulp: `atan2` 1.20, `cosh` 1.24, `atanh` 1.49, `sinh` 1.67, `tanh` 1.95 |
| Bit-compare with musl's own C code (`verification/musl_bitcompare/`, MinGW gcc, no FMA) | Bit-identical on every comparable input (377,655 across 10 functions; `sinh`/`cosh` except paths that call musl's table-driven `exp`). So the >1 ulp cases are musl's algorithms, not transcription loss |
| Platform-libm cross-check | Found Windows UCRT `atanh` off by up to 8.4 ulp near ±1 (mpmath-confirmed; dmath 0.41 ulp) — an example of the inconsistency this removes |
| `dmath` golden hashes | `GOLDEN_HASH` 0xa92df4d4e1fb935e → 0x8ae8ded20ae0ffde (pow replaced; sin/cos/exp/ln unchanged); new `GOLDEN_HASH_EXTENDED` 0xa7a13cc95309466a. Enforced on Linux/Windows/macOS CI |

**Hash migration.** Full workspace after the switch: 5 of 12,218 tests moved,
each attributed to a `.cjcl` transcendental builtin feeding the hashed
output and re-locked with its pre-dmath value recorded in a comment:
`primitive_master_hash_golden` (fixture calls `sin cos exp log randn`) and
the ABNG `.cjcl` chain-head canaries `pinn_cjcl`, `pinn_scaled`,
`compact_scaled` (sources call `sin`/`cos`/`exp`); every functional
assertion in those programs is unchanged. Three property tests used the
platform libm as an **exact** oracle for runtime output, which made them
seed-dependent after the switch: `tanh_forward_matches_direct` (failed on
the first run), `prop_adam_step_matches_oracle` (bias correction via
`powf`; passed on the first run, failed on the second), and
`reconstructed_first_step_from_extracted_b` (tanh; in
`tests/state_space_tests/`, which no test target currently compiles). All
now use `dmath`. The `fused_matmul_norm` p-norm references were
aligned to `dmath::pow` too (their exact comparisons cover only L1/L2 today,
so they were not failing). The chess RL weight hash did not move.

**Not verified locally:** Linux bit-identity (no Docker daemon on the dev
machine). The golden-hash tests run in the three-OS CI matrix, which is the
cross-platform check.

## Amendment 2026-09-26 (2) — analytics crates

**Decision.** The analytics crates move to `dmath` as well: `cjc-vizor`
(layout, render, stats), `cjc-nss`, `cjc-cana`, `cjc-cana-compress`,
`cjc-abng`, `cjc-locke`, `cjc-cronos-gan`. That is 91 method-call sites via
the same `DetMath` rename (`cjc-cana` gains a `cjc-repro` dependency). Four
renames hit `GradGraph::ln` node builders in `cjc-nss/cluster_grad.rs`,
failed to compile, and were reverted, as in the first amendment. No
`f64::exp`-style path calls or other libm methods (`log`, `exp2`, `cbrt`,
`asinh`, `acosh`) exist in non-test code, so after this amendment no
workspace crate calls platform libm transcendentals outside tests and
`#[cfg(test)]` items. `sqrt` stays on std (correctly rounded by IEEE).

**Also found.** `cjc-runtime/src/state_space.rs` (ADR-0020/0021) had never
been compiled: the commit that added it (`671dfeb`) carried only new files,
so the `mod` declaration and the dispatch fallback were lost, and none of
the `state_space_*` builtins were reachable. Its first-amendment dmath
migration was therefore dead code until the module was wired in alongside
this amendment.
