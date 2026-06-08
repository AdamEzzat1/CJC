# CANA Cost Model Training — First Run Findings

**Date:** 2026-06-07 (TIER B #4 from handoff)
**Status:** Infrastructure shipped; first-pass training data captured.
**Output:** `LinearCostModel::trained()` constructor + `trained_ranker()` helper.

---

## 1. What shipped

### Infrastructure
- `bench/cana_train_cost_model/` — new training-only benchmark crate
  - `programs.rs`: 18-program training corpus spanning all five
    canonical passes (CF, SR, DCE, CSE, LICM)
  - `main.rs`: leave-one-out measurement + gradient-descent OLS fit +
    Rust-source code generation

### API additions
- `cjc_cana::CoefficientSource` — enum: `Default` (hand-tuned) | `Trained`
- `cjc_cana::LinearCostModel::trained()` — const constructor
- `cjc_cana::trained_ranker()` — PassRanker using trained coefficients

### Tests (7 new, in `linear_cost_model.rs`)
- `trained_constructor_returns_distinct_source`
- `trained_model_resolves_known_passes`
- `trained_model_rejects_unknown_passes`
- `trained_predictions_stay_in_normalized_range`
- `trained_predictions_are_deterministic`
- `trained_and_default_differ_on_at_least_one_query`
- `trained_confidence_reflects_data_quality`

---

## 2. How to regenerate

```bash
cargo run --release -p cana-train-cost-model
```

The binary:
1. Parses + lowers all 18 corpus programs to MIR.
2. For each (program, pass) pair, runs the program twice (with and without
   the pass), measures wall-clock run_us, takes median of 5 iters.
3. Labels: `benefit = (run_us_without - run_us_with) / max(run_us_without, 1)`,
   clamped to `[0, 0.5]`.
4. Per pass, fits 4 weights (over expr_count, loop_depth, branch_count,
   alloc_sites) via gradient descent (lr=0.05, 5000 steps, normalized
   features).
5. Emits Rust source ready to paste into `linear_cost_model.rs`'s
   `trained_pass_coefficients` function.

Confidence values are derived from training RMSE: lower RMSE → higher
confidence, capped at [0.1, 0.95].

---

## 3. First-run results

### Per-pass fit quality

| Pass             | Mean benefit | Train RMSE | Confidence |
|------------------|--------------|------------|------------|
| constant_fold    | 0.1229       | 0.1817     | 0.10       |
| strength_reduce  | 0.0728       | 0.1295     | 0.32       |
| dce              | 0.1049       | 0.1663     | 0.16       |
| cse              | 0.0609       | 0.1369     | 0.29       |
| licm             | 0.1237       | 0.1687     | 0.14       |

### Per-program leaderboard (highest measured benefit per program)

| Program         | Winning pass    | Benefit |
|-----------------|-----------------|---------|
| arith_heavy     | licm            | 0.5000  |
| arith_med       | dce             | 0.5000  |
| arith_tiny      | constant_fold   | 0.2581  |
| cse_in_loop     | licm            | 0.0556  |
| cse_repeat      | constant_fold   | 0.5000  |
| dce_branchy     | licm            | 0.0588  |
| dce_dead        | licm            | 0.0000  |
| float           | cse             | 0.2692  |
| large           | strength_reduce | 0.0678  |
| loop_invariant  | licm            | 0.0000  |
| loop_nested2    | licm            | 0.0000  |
| loop_nested3    | dce             | 0.4690  |
| loop_nested4    | constant_fold   | 0.0980  |
| many_fn         | constant_fold   | 0.3208  |
| mixed           | constant_fold   | 0.5000  |
| recursive       | licm            | 0.0000  |
| sr_in_loop      | cse             | 0.5000  |
| sr_pow2         | licm            | 0.3103  |

---

## 4. Honest assessment

**The leaderboard is half-noise.** A few representative weirdnesses:

- `arith_heavy` "won" with LICM, but it has zero loops.
- `sr_in_loop` "won" with CSE, but its shared subexpressions are trivial.
- `loop_invariant` and `loop_nested2` show LICM benefit at 0.0000 — i.e.,
  the measurement found no improvement.

**Root cause:** the programs run in 1-100 μs on modern hardware. OS
scheduler and cache noise contribute ±5-20 μs of variance per
measurement. The actual per-pass benefit on a microbenchmark is in the
same order as the noise floor. Median-of-5 only partially helps.

**Consequences for the fit:**
- LICM's fitted `w_loop_depth` is **negative** (-0.038). It shouldn't be.
- CF's fitted `w_loop_depth` is also negative. Same story.
- All confidences are below 0.5 — the model knows it's uncertain.

The clamp in `predict_pass_gain` (output to `[0.0, 0.5]`) prevents
nonsense predictions like negative benefit, but a sign-flipped weight
still distorts the relative ranking of programs.

---

## 5. Why ship it anyway

1. **The infrastructure is the value.** A future maintainer with a
   bigger corpus, longer-running programs, or a cleaner signal (e.g.
   instruction-count proxy) can regenerate trained coefficients in
   ~2 minutes by running the binary and pasting the output.

2. **The default is unchanged.** `default_ranker()` keeps the hand-tuned
   coefficients that were known-good in Phase 2. `trained_ranker()` is
   opt-in for users who want to experiment.

3. **The confidence floor protects callers.** Trained coefficients all
   carry confidence ≤ 0.32. Any downstream consumer that weighs by
   confidence will naturally distrust the trained model and fall back
   to the canonical default sequence when uncertainty is high.

4. **Determinism is preserved.** The trained coefficients are baked-in
   constants; predictions remain bit-identical across runs and
   platforms. Same as the hand-tuned values.

---

## 6. What would move the needle

In rough effort order:

| Improvement | Effort | Expected RMSE reduction |
|---|---|---|
| Larger workloads (multiply each loop's `n` by 10-100×) | 1 hr | ~2× |
| More corpus programs (18 → 60) targeting feature corners | 1 day | ~1.5× |
| Switch label from wall-clock to MIR-instruction-count delta for CF/DCE/CSE/SR | 1 day | ~3× (but only on these 4 passes) |
| Hybrid label: instruction-count for size-shrinking passes, wall-clock for LICM | 1 day | ~2× overall |
| Multi-seed averaging at compile time (run each (prog, pass) 100x not 5x) | small | ~4× (40 min benchmark) |
| Constrained OLS (enforce non-negative `w_loop_depth` for LICM) | half day | Stops sign-flips, doesn't fix variance |

None of these were done in this session — that's deliberate scope.
The first-run findings are the deliverable; further refinement is
follow-up work guided by what we now know about the noise floor.

---

## 7. Where to read the code

| File | Purpose |
|---|---|
| `bench/cana_train_cost_model/programs.rs` | 18-program training corpus |
| `bench/cana_train_cost_model/main.rs` | Measurement + OLS fit + code generation |
| `crates/cjc-cana/src/linear_cost_model.rs` | `CoefficientSource` enum + `trained_pass_coefficients` fn |
| `crates/cjc-cana/src/pass_ranker.rs` | `trained_ranker()` helper |

---

**Bottom line:** the cost-model training loop closes. The numbers it
produces today reflect microbenchmark reality more than they reflect the
"true" per-pass benefit. The architecture stays clean; the corpus and
signal can be improved when the marginal benefit justifies the effort.
