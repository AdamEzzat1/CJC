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

`dispatch::parity` (compiled only with the feature; 15 tests, two of them not run by default:
the timing probe (`#[ignore]`) and one platform-ignored statement about Rust's own
`f64::powi`) compares every kernel with its Rust body **bit for bit** (`to_bits`), on
SplitMix64 inputs spanning 2^-8 to 2^8 with exact zeros and negative zeros mixed in,
plus the edge cases each kernel has:

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
- `sum_expbinned` (the pack's transcription of `BinnedAccumulatorF64`) against
  `accumulator::binned_sum_f64` at the four sizes, on a NaN, on each infinity and on
  both, on `[0, -0, the smallest subnormal]`, and on the three-value witness below;
- the fused `mse_grad` against its Rust twin at the four sizes, loss and every
  gradient element, plus the gradient witness (`n = 3`, `d = 2.9`: `h + h` with
  `h = d / 3` is not `(2 d) / 3`, and the kernel gives the former);
- and the routed public functions (`mse_loss`, `mse_loss_grad`, `matmul_raw`,
  `adam_step`) with the switch off versus on.

`crates/cjc-ad/tests/mse_grad_parity.rs` holds `ml::mse_loss_grad` to the `GradGraph`
chain it stands in for (`sub -> mul -> mean -> backward`) at six sizes, and with the
feature runs the same comparison through the switch. `crates/cjc-ad/tests/
pinn_routing_bits.rs` pins `piml_heat_1d_train`'s bits (see below).

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
site, independent of this integration — **done** (the performance stack, 2026-09-22):
`cjc_repro::powi_f64` is the one copy of `__powidf2`'s algorithm, every runtime-exponent
site calls it (`pinn.rs`'s heat gradient, `ml::lr_step_decay`, `cjc-nss`'s Adam bias
correction, `cjc-quantum`'s Vandermonde rows, `cjc-vizor`'s log-axis ticks, and this
file's fallback), `powi_reference` delegates to it, and spec tests pin the bits with
hard-coded values, the MSVC witness among them, on every platform. With that,
`heat1d_gradient_matches_cjcs_own_loop_bit_for_bit` runs on MSVC and passes: CJC's own
loop and the kernel agree there now, and `piml_heat_1d_train` computes the Linux bits
on Windows. `f64_powi_is_binary_exponentiation_on_this_target` stays ignored on MSVC as
a statement about Rust's `f64::powi`, which no CJC arithmetic depends on any more.

## The second disagreement: two "binned" accumulators

The fused kernel's first version summed the squared errors with the pack's own
`core.binned_*` accumulator, on the pack's claim that it and `BinnedAccumulatorF64`
give the same bits. The parity test disagreed at `n = 4093` by one ulp. They are
different algorithms: core's keeps integer significands per exponent with carries and
rounds once at the end (a correctly rounded sum); CJC's keeps one `f64` per biased
exponent, adds into it with a plain `+=` (a rounding on every same-exponent add) and
Kahan-folds the touched bins in ascending order. The smallest witness is three values:
`{1, 1+2^-52, 1+2^-52}` sums to `3 + 2^-51` exactly, which is representable, which
core's returns, and which CJC's gives as `3.0`. The kernel now carries CJC's accumulator
(`cjc_sum_expbinned_f64` is that accumulator alone, exported so it can be held to parity
directly); the pack's `cjc_sum_binned_f64` stays as the correctly rounded sum, documented
as not CJC's, and nothing here routes to it. The lesson is the powi one again: "the same
algorithm" is a claim to measure, and the parity test is where it gets measured.

## The fused loss and gradient, and the PINN routing

`ml::mse_loss_grad(pred, target, grad) -> Result<f64>` is the loss of
`mean((pred - target)^2)` and its gradient with respect to `pred` in one pass into a
caller buffer — the bits `GradGraph` produces for the chain, including the gradient's
spelling `h + h` with `h = (1/n) d` (the backward of `mul(diff, diff)` accumulates through
both operands). It goes through `dispatch::mse_grad` and so the switch. No CJC-language
builtin reaches it yet: a program's `mean((pred - target)^2)` with a gradient still
builds the graph and its buffers; the record's `mse_loss_grad` row and its
`status quo` neighbour show the two costs side by side.

`cjc-ad`'s `piml_heat_1d_train` routes its physics loss through
`dispatch::heat1d_residual_grad` when the feature is built and the switch is on; the loop
that defines it runs otherwise. The source term at the collocation points is computed
once per training rather than once per point per epoch, and the per-epoch buffers are
allocated once and zeroed — both bit-neutral, both held to digests of the training
captured before the change (`tests/pinn_routing_bits.rs`: four shapes, the final
parameters, every epoch's five logged values and the summary numbers; the tests also
assert that a flipped bit, another seed and one more epoch are rejected, so they can
fail). With the feature, the routed training gives the same digests.

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
- `piml_heat_1d_train` is routed (above); the other PINN problems (Burgers, the
  harmonic oscillator) go through `GradGraph` and are not.
- `ml::mse_loss_grad` and the switch itself have no CJC-language entry point.
- `adam_step` is **not routed any more**: the record has the kernel 17.85x slower
  (a software `sqrt` in the libm-free pack against `sqrtsd`), so `adam_step_raw` takes
  the fallback regardless of the switch; the kernel, its parity test and its bench row
  stay.
- A registered timing record now exists: `bench/bruchion_kernels_bench` (run through
  `bench/bruchion_kernels_bench/run.ps1`, which refuses to record on a loaded machine
  and stamps the gate readings, the kernel hash and the tree state into the
  provenance), writing `bench_results/bruchion_kernels/{REPORT.md, rows.jsonl,
  phases.csv, provenance.txt}` and archiving the previous record under `history/`.
  Three clean-tree records, at `abe6b21`, `a24189b` and `dab5f6c` (the last is on
  disk, the others under `history/`; 2^16 elements, five interleaved phases of one
  second, an A/A arm, the gate open each time), agree: `relu` slower 1.27x and `axpy`
  slower 1.31–1.37x, whole band above 1; `adam_step` slower 15–18x; `matmul 64x17x33`
  **faster about 1.5x**, whole band below 1 on all three (median ratios 0.658, 0.658,
  0.639) — the one kernel win, one shape, one machine, reproduced; `matmul 128^3`,
  `dot_kahan`, `mse`, both `heat1d_residual_grad` shapes and `mse_loss_grad` within
  the A/A spread every time, i.e. the runs cannot tell those arms apart. (The
  `a24189b` record's provenance says `dirty tree: true` on a clean tree: a parsing
  bug in the runner, fixed in `dab5f6c`; the archived file is left as written.) `mse_loss_grad` against its unrouted
  status quo (the `GradGraph` chain): 2.84 against 14.69 ns per element and 0 against
  122 allocations per call — a CJC-side comparison of two Rust paths, not a kernel
  win. The elementwise kernels are still slower than CJC's release Rust bodies, and
  role 6's flag sweep (`-O3`, AVX2, gcc 16, in the C harness) found no setting that
  runs them faster than the locked `-O2` does. `dispatch::timing_probe` remains as a
  probe. What the probe said on 2026-09-22 (one machine,
  no interleaving, no A/A):

| `timing_probe`, release, 2^16 elements, min of 25 after a warm-up | kernel (scalar, session 3) | kernel (2-lane `@simd`, SIMD step 2) | kernel (unrolled, `restrict`) | Rust body | kernel, A/A re-run under a runaway service host | Rust body, same re-run | kernel, A/A re-run, quiet (gated) | Rust body, quiet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `axpy` (ns per element) | 0.4532 | 0.4547 | 0.3815–0.5066 | 0.2075–0.2975 | 0.6638–0.9003 | 0.3555–0.5035 | 0.3830–0.3998 | 0.2075–0.2533 |
| `dot_kahan` | 3.0075 | 2.8885 | 3.0075–3.5248 | 3.0289–3.6331 | 5.1559–5.5618 | 5.0186–5.3955 | 2.8931–3.0075 | 2.9068–3.0273 |
| `relu` | 0.3052 | 0.2213 | 0.1663–0.2579 | 0.1282–0.1953 | 0.2975–0.4532 | 0.2106–0.3647 | 0.1663 | 0.1221–0.1282 |
| `mse` | 2.8870 | 2.8870 | 3.0075–3.6087 | 2.9205–3.6377 | 4.9820–5.4596 | 5.1636–5.5984 | 2.8870–3.0075 | 2.9037–3.0258 |

  The last two columns are three samples in a row (the machine's state drifted
  between them, so the ratio within a sample is the number: `axpy` 1.70–1.86x,
  `relu` 1.30–1.32x, the Kahan kernels 0.97–0.99x). The Kahan kernels tie (a serial
  dependency chain either way). The elementwise kernels are **slower** than the
  Rust bodies: with the scalar kernels 2.2–2.4x; after the Bruchion side's SIMD
  step 2 (two-lane `@simd` slice twins) 2.2x for `axpy` and 1.8x for `relu`; after
  unrolling the lane head and `restrict`-qualifying its pointers about 1.8x and
  1.3x. The same bits throughout — the parity tests above passed unchanged
  against each archive. The release Rust loop is auto-vectorized by LLVM on the
  same SSE2 baseline; the rest of the gap is scheduling, which gcc 8 at the locked
  `-O2` flags does conservatively. Switching the kernels on for `relu_raw` today
  is therefore still a slowdown, by less.
  **The A/A re-run** (last two columns, later the same day, after the Bruchion side
  mirrored these forms into its self-hosted compiler — a change that emits no
  different C): the archive is byte-identical to the previous column's
  (`abi-check` reports the same `kernel_sha256`, the harness hash is
  `15a6e852fe202fdc` again, the parity tests pass unchanged, 794 / 3 ignored), so
  this run measures the probe, not the kernels. Every row, kernel and Rust alike,
  came out about 1.7x slower in absolute terms — machine state, the run followed a
  twenty-minute test ladder — which is why only the ratio within a sample is read.
  Those ratios, three samples in a row: `axpy` 1.87x, 1.79x, 1.60x; `relu` 1.84x,
  1.24x, 1.31x; `dot_kahan` 1.03x, 1.03x, 0.99x; `mse` 0.96x, 0.98x, 0.99x. The
  previous column's bands (`axpy` 1.70–1.86x, `relu` 1.30–1.32x) reproduce, and
  widen: on identical bits this probe's `relu` ratio spread 1.24–1.84x across three
  runs, so a `relu` reading from it is good to about ±0.3x, an `axpy` reading to
  about ±0.15x, and a change smaller than that is not something this probe can see.
  The conclusion is unchanged: the kernels are slower than the release Rust bodies,
  by about 1.3x (`relu`) and about 1.8x (`axpy`), and the Kahan kernels tie.
  **The quiet re-run** (last two columns) resolves what "machine state" was. The
  first A/A re-run happened under a runaway system service host (`svchost` hosting
  DcomLaunch, Power, PlugPlay and the brokers, at about 130% of one core for over
  an hour, driven by a user-mode driver host resetting the power scheme to itself
  up to once a second); a gate waiting for the CPU to average under 10% never
  opened while it ran. It stopped on its own — the machine was not rebooted in
  between, which `LastBootUpTime` and the event log both said, Fast Startup having
  turned a "shut down" into a resume — and the gate then read 12.5% average, 19.4%
  peak, the host at 0%, the residual being the desktop app. Three samples in a row
  on the same archive bits: `axpy` 1.85x, 1.58x, 1.86x; `relu` 1.31x, 1.36x, 1.30x;
  `dot_kahan` 0.99x, 1.00x, 0.99x; `mse` 0.99x, 0.99x, 0.99x — and the absolute
  numbers are the unrolled column's again, tight across the three. So the 1.7x in
  the loaded columns was the host, the ratios were sound even then, and the probe's
  own spread on a quiet machine is about ±0.03x for `relu` and ±0.15x for `axpy`
  (one sample's Rust body ran slower, not the kernel). The conclusion stands: the
  kernels are slower than the release Rust bodies, by about 1.3x (`relu`) and
  about 1.8x (`axpy`), and the Kahan kernels tie. Any future probe is gated on the
  measured load, and "rebooted" is checked against `LastBootUpTime` first.
  With `restrict` the kernels' `x` and `y` must not overlap; the dispatch
  functions take `&[f64]` and `&mut [f64]`, so that holds by construction.
- ~~`powi` in CJC's own code still lowers to `pow` on MSVC targets~~ — done, see above:
  `cjc_repro::powi_f64` at every runtime-exponent site.
