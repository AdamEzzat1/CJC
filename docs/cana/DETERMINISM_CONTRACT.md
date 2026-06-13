# Determinism Contract for Runtime/Executor Optimizations

**Status:** standing reference — every speed or memory optimization to
the runtime, executor, or numerical kernels is gated against this list
BEFORE it lands. Produced by the determinism-auditor pass of the
stacked-role optimization arc (2026-06-13); file:line evidence verified
against the tree at that time.

The CJC-Lang promise is **bit-identical output for the same seed, on
every run, platform, and thread count.** An optimization that is faster
or smaller but violates any invariant below is not an optimization — it
is a regression, and must be redesigned, not forced.

## The 10 invariants

| # | Invariant | Enforced at | Optimization category that violates it | Test that catches a violation |
|---|---|---|---|---|
| 1 | FP reductions use Kahan/Binned compensated summation, never naive `+=` | `cjc-repro/src/kahan.rs`, `cjc-runtime/src/accumulator.rs` | Vectorized/parallel reductions with ad-hoc merge order | `cjc-repro` kahan byte-stable tests; parity stress |
| 2 | No FMA (`mul_add`/`_mm*_fmadd*`) — separate mul+add (two roundings) | `cjc-runtime/src/tensor_simd.rs`, `cjc-repro/src/lib.rs` | Auto-vectorized matmul/dot targeting FMA hardware | double-run output determinism gate |
| 3 | Reductions marked Strict/Kahan/Unknown must NOT be reassociated | `cjc-mir/src/reduction.rs`, `cjc-mir/src/verify.rs` | Loop unroll / strength-reduce / CSE over accumulators | `tests/test_parity_stress.rs` (50 seeds) |
| 4 | BTreeMap/BTreeSet everywhere — never HashMap/HashSet in decision or output paths | `cjc-mir-exec/src/lib.rs` (scopes, call_cache, trace_node_ids), `cjc-eval/src/lib.rs` | Swapping to HashMap "for speed" | parity stress; any double-run hash test |
| 5 | Decision/output iteration order is deterministic (BTreeMap key order) | `cjc-mir-exec/src/trace.rs`, `cjc-cana-nss/src/lib.rs` | Unordered field/collection iteration feeding branches or output | parity gate (divergent output) |
| 6 | RNG is SplitMix64 seeded from the top-level seed; fork by `next_u64` | `cjc-repro/src/lib.rs` | Removing the seed, `thread_rng`, clock-seeding | `test_rng_deterministic`, `stress_tensor_randn_determinism` |
| 7 | AST-eval ≡ MIR-exec byte-identical output for every (program, seed) | `cjc-cli/src/commands/parity.rs`, `tests/fixtures` | Any executor change that perturbs eval order or dispatch | `tests/test_parity_stress.rs`, `test_builtin_parity.rs` |
| 8 | Content hashes use FNV-1a (`CanaHasher`), never seeded `DefaultHasher` | `cjc-cana/src/hash.rs` | Swapping to blake3/SHA for speed; `std::hash` | report-hash byte-identity tests |
| 9 | Float ordering uses `f64::total_cmp`, never `<`/`partial_cmp` | `cjc-runtime/src/stats.rs` | Replacing `total_cmp` with `partial_cmp` in sorts | quantile/rank tests with NaN |
| 10 | No wall-clock/sensors in decision paths or hashes (diagnostics are post-hoc, byte-equality-gated first) | `bench/cana_diagnostics` (hard wall) | Feeding timing back into a decision/label/hash | row-hash double-run stability |

## Pre-commit checklist (run before every optimization lands)

- [ ] grep the diff for `mul_add`, `_fmadd`, `fma` — none introduced (inv. 2)
- [ ] grep the diff for `HashMap`, `HashSet` — none in decision/output paths (inv. 4, 5)
- [ ] marked reductions still use Kahan/Binned, not plain `+=` (inv. 1, 3)
- [ ] any float sort uses `.total_cmp()` (inv. 9)
- [ ] no `Instant::now`/`SystemTime`/clock in a decision or hash path (inv. 10)
- [ ] `cargo test --test fixtures` (parity gate) green (inv. 7)
- [ ] `cargo test --test test_parity_stress --test test_builtin_parity` green (inv. 3, 7)
- [ ] a double-run canary on a representative program: same seed → byte-identical output AND (where applicable) byte-identical trace events
- [ ] `bench/cana_diagnostics` determinism gates green if the change is in an executor path the harness times (inv. 10)

## The rule

An optimization is approved to land **iff** it passes the pre-commit
checklist AND shows a measured improvement (wall-clock via
`bench/cana_diagnostics`, or bytes via the `alloc_bytes_in_window`
instrumentation) that justifies the change. Faster-but-nondeterministic
fails. Smaller-but-inaccurate fails. When in doubt, the legality/parity
gate is the final authority — redesign rather than force.
