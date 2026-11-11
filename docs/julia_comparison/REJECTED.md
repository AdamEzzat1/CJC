# Rejected and downgraded findings

Findings removed by the falsification gate, or reduced to a planned
measurement. Each entry names the gate that removed it:
- **code**: the code says otherwise;
- **surface**: the claim depends on whether something is reachable from `.cjcl`;
- **currency**: the Julia fact is not current or not official;
- **fair**: the comparison is not like for like;
- **determinism**: CJC cannot adopt it without breaking its guarantees;
- **evidence**: there is no runnable proof yet.

| # | Candidate finding | Gate | Why it was rejected or downgraded |
|---|---|---|---|
| R-1 | "The general `sin`/`cos`/`exp`/`log` builtins call platform libm" (a known lead) | code | Commit `0160044` routes every `.cjcl`-observable elementary function through `cjc_repro::dmath` (`builtins.rs:869-1035`). The remaining platform-libm calls are in tests only, except `randn`, which became **P0-6**. The stale statements are listed in CJC_INVENTORY §5 |
| R-2 | "Default parameters are missing" (CLAUDE.md §2) | code | Implemented in both executors (`cjc-eval/src/lib.rs:4351-4416`, `cjc-mir-exec/src/lib.rs:5829-5927`) |
| R-3 | "Decorators are missing" (CLAUDE.md §5) | code | The parser and the `@memoize`/`@trace` runtimes exist. Only *user-defined* decorators are missing, which is not an accuracy gap and is not ranked |
| R-4 | "`jacobian` is finite differences" | code | It is exact: one reverse sweep per output element with a one-hot seed (`lib.rs:2305`) |
| R-5 | "Using FD for PINN derivatives is a CJC-specific weakness compared with Julia" | fair | NeuralPDE.jl v7.5.0 **also defaults to `FiniteDifferenceDerivative()`**. Julia's real advantage is the *option* of exact derivatives (`EnzymeForwardDerivative`, order ≤ 4). The gap is restated as P1-1 (exact option), not "CJC uses FD" |
| R-6 | "Julia's libm is more accurate than CJC's" | evidence, currency, determinism | Julia publishes no per-function ulp table [verify]. dmath's own bounds come from external harnesses. Julia's kernels use FMA, which CJC forbids. Downgraded to a planned measurement (harness S1, which also measures `Base.Math` on the same grid) |
| R-7 | "Zygote is deprecated or superseded" | currency | Release 0.7.16 on 2026-11-06, with no deprecation notice. That it is being superseded is community opinion only. Not used in any ranking |
| R-8 | "Lanczos produces ghost eigenvalues" | evidence | No reorthogonalisation was found in the code (audit), but no run has shown ghosts. Kept as P1-10 (a cheap fix) with measurement in harness S2. Not called a wrong answer |
| R-9 | "`schur` fails on complex-conjugate eigenvalue pairs" | evidence | Single-shift QR is a known weakness, but no failure has been shown. P2-10 [unverified] |
| R-10 | "The `grad_graph_*` proptests prove gradient correctness" (`docs/AD_MIR_COVERAGE_AUDIT.md`) | code | The cited proptests check **forward** values (`add_forward_matches_direct`). Turned into an inventory correction, not a capability claim |
| R-11 | "Fused ops hide the broadcasting gap" (a known lead) | — | **Confirmed, not rejected:** `MlpLayer` sums its bias gradient, but the plain ops do not (probe P2). Listed here only to close the lead |
| R-12 | "Bloqade-style Rydberg simulation is a major gap" | fair, currency | Bloqade.jl is in maintenance mode and recommends its Python successor. Hardware submission is out of scope. Only P2-level interest remains, and it is not ranked above P2 |
| R-13 | "CJC should adopt StaticArrays / dot-fusion / macros to match Julia" | fair, determinism | Their benefit comes from JIT specialisation, or they would add a pipeline stage. No accuracy gain. Won't-do (LANGUAGE_MECHANISMS §2–4) |
| R-14 | "BigFloat support is needed for accuracy" | determinism (zero-dependency rule) | MPFR would be a dependency. Double-double (ADR-E) covers the "rerun in higher precision" need, and Julia remains the BigFloat oracle offline |
| R-15 | "`fidelity` is wrong" | surface, code | It is crate-only, and its docstring says it is exact only when one state is pure. Downgraded from a wrong answer to a capability gap (P1-11) |
| R-16 | "Executor parity proves these numerics are right" | evidence | Every probe was IDENTICAL across executors and still wrong, because both executors call the same runtime function. Parity is necessary but tells you nothing about accuracy. This is why the harness has positive controls |
| R-17 | "The bf16 NaN → Inf conversion happens in practice" | evidence | Inferred from the truncation code; not demonstrated, since no `.cjcl` path produces a bf16 NaN with a low-bit-only payload. Kept as part of P2-1 |
| R-18 | Any performance comparison with Julia | scope | Out of scope by design. No timings were taken or quoted next to CJC |
</content>
</invoke>
