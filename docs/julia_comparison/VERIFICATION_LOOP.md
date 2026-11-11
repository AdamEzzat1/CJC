# Verification loop for the P0 fixes

The probes showed that the two executors agreeing does not mean the answer is
right. So a fix only counts once evidence that **did not come from the agent
that wrote it** has confirmed it. Each step below rules out a different way
of fooling ourselves.

| # | Step | Who runs it | What it rules out |
|---|---|---|---|
| V0 | **Baseline captured before any fix**: probe outputs and the full `cargo test --workspace --no-fail-fast` result on unfixed `31058a8` | lead | Counting a failure that was already there as a regression, or a pass that was never failing as a fix |
| V1 | **Red first**: the new test fails on unfixed code | fixing agent, then the lead reruns | Tests that pass no matter what, i.e. tests that don't test the bug |
| V2 | **Independent oracle**: expected values come from mpmath, numpy dense ED, `fractions.Fraction` or analytic derivatives, never from the fixed code's own output | fixing agent; the lead re-derives a sample | "Golden" values that just record whatever the new code does |
| V3 | **Probe rerun by the lead** on a fresh release build of the merged tree, diffed against the V0 outputs | lead | Trusting the agent's account of its own results |
| V4 | **Mutation check**: revert the core of the fix and the new tests must fail again | fixing agent, then the lead repeats at least one per fix | A test that passes because of something other than the fix |
| V5 | **Invariant checks** that need no reference value: UΣVᵀ = A and orthonormality (SVD); erf + erfc = 1 and odd symmetry (erf); ⟨ψ\|H\|ψ⟩ of the returned MPS equal to the reported energy, plus monotone sweeps (DMRG); bit-identical sums over every permutation and chunking (binned) | in the tests | An oracle that is wrong, or tested at too few points |
| V6 | **Both executors and determinism**: new tests go through `cjc-eval` and `cjc-mir-exec`; repeat runs give identical bits | in the tests | Executor divergence, and nondeterminism introduced by the fix |
| V7 | **Full regression**: `cargo test --workspace --no-fail-fast` on the merged tree, diffed against V0. Every new failure is explained, never silenced | lead | Breaking something else |
| V8 | **Golden fallout table**: every golden or canary value that changes is listed with its old and new value and the reason. It is re-locked only with the user's approval and a commit message that explains it | lead and user | Hiding an unintended change as "expected" churn |
| V9 | **Independent review**: a fresh reviewer gets the diff and the bug description, but not the fixing agent's report | reviewer agent | Being anchored on the author's story |
| V10 | **Cross-OS evidence** (randn, binned): only CI on ubuntu and macOS can show that the cross-platform output is identical. A result from Windows alone is reported as "not verified cross-platform" | CI | Claiming cross-platform determinism from a single machine |

## Results: 2026-11-10, branch `claude/p0-accuracy-fixes` (base `31058a8`), Windows

| Fix | V3 probe: before → after (lead's merged release build) | Oracle checked by the lead | V4 mutation (agent) | V9 review |
|---|---|---|---|---|
| P1 `hessian_diag` | 4.000000000004 → 12.000000000389 | Central-difference theory: 12 + 4h² = 12.0000000004 | revert → 9 + 4 + 1 tests fail | clean |
| P2 broadcast gradient | shape [2,3] of ones → [3] / [1,3] of 2.0 | analytic | revert → 9 + 7 fail | clean |
| P3 svd | σ_min 0.0 → 4.99999868699e-11 | mpmath on the exact f64 input: 5.00000041357685e-11. LAPACK gets 1.2e-7 relative, ours 3.5e-7; both are backward stable | revert → 11 fail | clean |
| P4 erf/erfc | erfc(5) 6.7e-3 relative → 1.42 ulp | mpmath: 0.17 / 0.22 / 1.05 / 1.42 ulp on the probe points | revert → 7 fail | clean (doc claim corrected to "max 2.46 ulp") |
| P5 binned | `false` → `true`; both orders give 3.0000000000000004 | Fraction oracle (agent); reviewer's independent Python port matched on 3000 random multisets | revert → 10 / 10 fail | clean |
| P6 DMRG | −5.950 / −9.160 / −12.522 → −6.464101615137755 / −9.974308535551721 / −13.499730394751616 | Lead's own ED (S⁺S⁻ form) for n=10/12 agrees to ~1e-14 | revert one index → 3 + 2 fail | clean |
| P0-6 randn | (no probe) 476 of 10⁴ draws differ from libm on Windows | golden hash | revert → 3 fail | clean |

- **V6:** every probe is IDENTICAL across the two executors, and three repeat runs of P5 and P6 are bit-identical.
- **V7:** baseline 12,398 passed / 0 failed. After the fixes: 12,483 passed / 4 failed. Those are exactly the four tests the determinism agent predicted, with exactly the hashes it predicted.
- **V8 golden fallout:**
  - Three canaries change because of the `randn` fix: `pinn_scaled_chain_head_canary_locked`, `tabular_scaled_chain_head_canary_locked` and `primitive_master_hash_golden`. These are the same canaries that fail on Linux and macOS CI. **Re-locked 2026-11-10 with the user's approval**; each test's comment keeps the pre-fix (Windows libm) hash. New values: `1361407d…`, `e2b49c4a…`, `eff68faf…`.
  - `test_acf_sinusoidal_periodicity` asserted `> 0.9` on a quantity whose exact value is 0.9. It now asserts `|r − 0.9| < 1e-12`, which is stricter.
- **V10:** not yet run. Cross-OS identity of `randn` and the sums can only be shown by ubuntu and macOS CI.
- **Behaviour change:** with correct gradients, `examples/01_mlp_xor.cjcl` no longer converges in 200 epochs. The old low loss came from a per-sample bias that the bug created. **Retuned:** the example now runs 2000 epochs, with the same init and learning rate. Loss reaches 0.0045 by epoch 1600; predictions are 0.033 / 0.938 / 0.951 / 0.064 for targets 0 / 1 / 1 / 0; both executors are IDENTICAL. It converges for all 8 seeds tried, and so do five other settings in the sweep.
- **Follow-ups fixed in the same session**, each through red, green and a mutation check:
  - **`vmap_forward`** now reuses `reforward_masked`, recomputing only the nodes that depend on the input. It returns `Err` for a dependent node that can't be recomputed (`StructField`, `MapLookup`).
    - Red: 12 of 14 failed. Mutation: 13 failed.
    - Oracle: a fresh graph built per batch element, compared bit for bit, plus plain-f64 softmax and log2.
  - **`cjc-data` vectorised `log`/`exp`/`sin`/`cos`/`tan`** now use dmath.
    - Red: 7 of 8 failed, with 1.5–9% of elements differing from dmath. Mutation: both the public `mutate` test and the in-crate fast-path-vs-row test fail.
  - **`cjc-vizor` log-scale mapping** now uses dmath `log10`/`log2`, which are exact at decades.
    - Red: 2 of 4 failed. `map_y(10)` on [1, 1e6] was 1 ulp off its tick. Mutation: 2 failed.
    - `log_ticks` output was already correct and is unchanged.
- **Known gaps:**
  - The `1 - normal_cdf(|z|)` p-value tails in `hypothesis.rs` still cancel.
  - `cond()` still uses `eigh(AᵀA)`.
  - Binned `sum` is about 4× slower (release, noisy machine).
  - dmath's `exp(1.0)` = 2.7182818284590455 is 1 ulp above the correctly rounded `e`. That is the known fdlibm result, not a regression.

## What does not count as evidence
- An agent's statement that a test passes, unless the lead reran it.
- Executor parity on its own.
- A test whose expected value was taken from the new code's output.
- A probe run on a stale binary. Every rerun records the commit and the build time.
