# Bruchion native kernels (`bruchion-kernels`)

Milestone 1 of the Bruchion-for-CJC integration, as planned in the Bruchion
repository's `docs/cjc-integration-plan.md`: `cjc-runtime` can link a prebuilt,
C-ABI kernel pack written in Bruchion and route a few hot loops to it behind a
runtime switch, with the existing Rust bodies as the fallback and as the reference.

## What is in this crate

| piece | where |
|---|---|
| the cargo feature `bruchion-kernels` (off by default; no crate dependency) | `crates/cjc-runtime/Cargo.toml` |
| the build script that links `libkernels_f64.a` from `BRUCHION_KERNELS_DIR` | `crates/cjc-runtime/build.rs` |
| the generated FFI skeleton, `include!`d from that directory | `crates/cjc-runtime/src/bruchion/mod.rs` |
| one safe function per kernel, its Rust fallback, and the parity tests | `crates/cjc-runtime/src/bruchion/dispatch.rs` |
| the runtime switch `RuntimePolicy::bruchion_kernels` (+ `set_bruchion_kernels`) | `crates/cjc-runtime/src/runtime_policy.rs` |
| the routed call sites: `kernel::relu_raw`, `kernel::matmul_raw`, `ml::mse_loss`, `ml::adam_step` | `kernel_bridge.rs`, `ml.rs` |

Nothing changes unless **both** the feature is compiled in and the switch is on. The
default build has no build-script work, no foreign symbol and no new dependency, so
CI's `cargo test --workspace` is unaffected; the `without_the_feature` test asserts
the switch is inert there.

## Building the kernel directory

In the Bruchion repository:

```bash
./target/release/bruchionc build-kernel examples/cjc/kernels_f64.bru --require no_alloc,no_os --emit-archive
```

prints `.bruchion-kernels/kernels_f64-<hash16>/`. Its name is the content hash of the
kernel C, its flags and its header; pinning it pins the ABI, and `bruchionc abi-check
<old>/metadata.json <new>/metadata.json` classifies any change between two of them.
Then, here:

```powershell
$env:BRUCHION_KERNELS_DIR = "<that directory>"
cargo test -p cjc-runtime --features bruchion-kernels bruchion
```

## What the parity tests assert

`dispatch::parity` (compiled only with the feature; 9 tests, 2 of them
platform-ignored, plus the timing probe) compares every kernel with its
Rust body **bit for bit** (`to_bits`), on SplitMix64 inputs spanning 2^-8 to 2^8 with
exact zeros and negative zeros mixed in, plus the edge cases each kernel has:

- `axpy`, `mse`, `dot_kahan` at 0, 1, 7 and 4093 elements; `dot_kahan` also on the
  input `[2^52+1, 0.1, 0, 2^53+2, 0]`, where a Kahan recurrence without CJC's
  zero skip differs in the last bit (both sides must give `13510798882111492`);
- `relu` on `-0.0`, NaN and the infinities;
- `matmul` at seven shapes including `k = 0` and `m = 0`;
- `adam_step` over five steps with `1 − β^t` from `powf` on this side;
- `powi` against a Rust transcription of compiler-builtins' `__powidf2`
  (`powi_reference`) for exponents −12..=12 over 200 bases plus the edge cases; and,
  separately, against `f64::powi` itself — the test that found the disagreement below;
- the heat-equation residual and its gradient at six `(n_colloc, n_params)` shapes
  with the source term from `sin` on this side, against the same loop with
  `powi_reference`; and, separately, against CJC's own loop with `f64::powi`;
- and the routed public functions (`mse_loss`, `matmul_raw`, `adam_step`) with the
  switch off versus on.

A disagreement in any of these is the deliverable, not a failure to hide: it names a
place where "the same algorithm" was not "the same arithmetic".

## The first disagreement: `f64::powi` on MSVC targets

Found 2026-09-22 by the first run of these tests on `x86_64-pc-windows-msvc`
(rustc 1.97.1): two of nine failed, `powi` and the heat-equation gradient.

**What it is.** `f64::powi` is documented as LLVM's `llvm.powi`, which on every target
with a `__powidf2` libcall is binary exponentiation (`r *= a` per set bit of `|n|`,
`a *= a` between bits, `1 / r` for a negative `n`). On MSVC targets LLVM has no
`__powidf2` libcall and lowers a `powi` with a runtime exponent to the C runtime's
`pow(x, (double) n)`. The probe binary imports `pow` from
`api-ms-win-crt-math-l1-1-0.dll`. Evidence, one machine:

| probe (200 bases in [0.5, 4), exponents −12..=12, exponent hidden from the optimizer) | mismatches vs `__powidf2` |
|---|---:|
| x86_64-pc-windows-msvc, rustc 1.97.1, `-C opt-level=0` | 2728 / 5000 |
| x86_64-pc-windows-msvc, rustc 1.97.1, `-C opt-level=3` | 2728 / 5000 |
| x86_64-unknown-linux-gnu, rustc 1.98.1 (Docker `rust:1-bookworm`), `-C opt-level=0` and `3` | 0 / 5000 |

Witness: `x = 3.8511975033176533`, `n = −11` — `__powidf2` and the kernel give
`4510433485740570282`, `f64::powi` on MSVC gives `4510433485740570284` (the `pow`
result). A *constant* exponent at `-O` is expanded inline by LLVM into the same
multiplication tree as `__powidf2`, so on MSVC the same source computes different bits
in a debug build (`cargo test`) and a release build, and different bits from Linux in
both.

**What it touches in CJC.** Every `powi` whose exponent is not a compile-time constant
after inlining, on Windows: `pinn.rs`'s `piml_heat_1d_train` (`x.powi(i as i32 - 2)`
in the physics gradient — the second failing test is exactly this, one ulp in
`grad[7]` at 1000 collocation points × 9 parameters), `ml.rs`'s learning-rate decay
(`decay_rate.powi((epoch / step_size) as i32)`), and, in debug builds, every
`.powi(2)` in `hypothesis.rs`, `sparse_eigen.rs`, `f16.rs`, `linalg.rs`. None of this
is visible to a test that runs on one platform at one profile.

**What the tests now assert.** The kernel equals `powi_reference` bit for bit
everywhere (`powi_kernel_is_compiler_builtins_binary_exponentiation_bit_for_bit`);
the gradient kernel equals the reference loop everywhere. The two tests against
`f64::powi` and CJC's own loop are `#[ignore]`d **with the reason** on
`target_env = "msvc"` and run everywhere else; `dispatch::f64_powi_is_binary_exponentiation()`
is the runtime probe they and any CJC code can ask.

**What it means for the switch.** With the kernels on, Windows CJC computes the
Linux bits for these paths; with them off, it does not. That is an argument for a
CJC-owned `powi` (the eleven lines of `powi_reference`, in `cjc-repro`) used at every
site, independent of this integration — proposed, not done here.

## Linking on an MSVC toolchain

The archive is built by MinGW gcc; the host Rust toolchain here is
`x86_64-pc-windows-msvc`. Two things make that link work, both handled:

- `___chkstk_ms` (libgcc's stack probe, used by the binned accumulator's 16 KiB
  frame) — `--emit-archive` bundles libgcc's `_chkstk_ms.o` into the archive;
- `fprintf` in the Bruchion runtime's assert path — `build.rs` adds
  `legacy_stdio_definitions.lib`, which every MSVC toolchain ships.

On a GNU toolchain (Linux, MinGW) neither is needed and `build.rs` adds nothing.

## Not done in this milestone

- No CJC-source builtin toggles the switch (`runtime_policy::set_bruchion_kernels`
  is the Rust API); the builtin and its AST/MIR wiring are the next step.
- `cjc-ad`'s `piml_heat_1d_train` still runs its own physics loop; the kernels for
  it exist and are parity-tested, the call site is not yet routed.
- No registered timing record exists on the CJC side. `dispatch::timing_probe`
  prints nanoseconds per element for kernel versus fallback when run with
  `--ignored` (`cargo test -p cjc-runtime --release --features bruchion-kernels
  timing_probe -- --ignored --nocapture`); it is a probe, not a benchmark record, and
  milestone 1's "a record shows the timing" clause is therefore **not** met. What
  the probe said on 2026-09-22 (one machine, no interleaving, no A/A):

| `timing_probe`, release, 2^16 elements, min of 25 after a warm-up | kernel | Rust body |
|---|---:|---:|
| `axpy` (ns per element) | 0.4532 | 0.2075 |
| `dot_kahan` | 3.0075 | 3.0273 |
| `relu` | 0.3052 | 0.1266 |
| `mse` | 2.8870 | 2.9037 |

  The Kahan kernels tie (a serial dependency chain either way). The elementwise
  kernels are about 2.2–2.4x **slower** than the Rust bodies, because the release
  Rust loop is auto-vectorized by LLVM and the kernel is gcc's scalar `-O2` loop
  (`-ffp-contract=off`, no `-O3`). Switching the kernels on for `relu_raw` today
  is therefore a slowdown; the Bruchion side's SIMD step 2 (an explicit,
  bit-identical lane loop for `axpy`/`relu`) is the prerequisite for routing the
  elementwise kernels at all.
- `powi` in CJC's own code still lowers to `pow` on MSVC targets (above); the fix
  belongs in `cjc-repro`, not behind this feature.
