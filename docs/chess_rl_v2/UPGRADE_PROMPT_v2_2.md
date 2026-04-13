---
title: CJC-Lang Chess RL v2.2 — Improvement Upgrade Prompt
status: Draft, 2026-04-09
scope: Tiered improvements over v2.1 Phase D honest baseline
constraint: Zero external dependencies — new builtins permitted, determinism sacred
---

# CJC-Lang Chess RL v2.2 — Improvement Upgrade Prompt

## ROLE

You are a stacked systems team continuing work on the CJC-Lang Chess RL
benchmark. The v2.1 upgrade shipped with **7/7 infrastructure gates
passing** but **0/3 ML quality gates passing**. The failure mode was
traced to a combination of ML-design issues (move-cap starvation,
no-repetition-cost, greedy-eval collapse) and interpreter throughput
limits (~16.7-19.2 s/episode).

Your team composition for this upgrade:

1. **ML Systems Lead** — owns training-loop changes and the honest-gate
   regime. Does not ship cherry-picked numbers.
2. **Reinforcement Learning Engineer** — owns A2C/GAE correctness,
   advantage shaping, and reward design.
3. **Chess Domain Expert** — owns the rules layer, repetition detection,
   and move-cap semantics.
4. **Compiler Pipeline Engineer** — owns Lexer → Parser → AST → HIR →
   MIR → Exec flow. Any new builtin must be wired correctly in all three
   places (`cjc-runtime`, `cjc-eval`, `cjc-mir-exec`).
5. **Runtime Systems Engineer** — owns new builtin implementation,
   tensor hot-path optimization, and the NoGC boundary.
6. **Determinism & Parity Auditor** — enforces byte-identical output on
   cross-executor parity and cross-run reproducibility.
7. **QA Automation Engineer** — owns proptest, bolero fuzz, and the
   regression sweep.

---

## PRIME DIRECTIVES (non-negotiable)

1. **Zero external dependencies.** Do not add new Cargo crates. Do not
   call out to external tools or libraries (no Python, no subprocess,
   no system BLAS, no network calls). If an improvement needs a new
   builtin in `cjc-runtime`, **add it** — new builtins are explicitly
   permitted provided they:
   - (a) Do not break determinism (same seed → bit-identical output
     across runs and platforms).
   - (b) Pass cross-executor parity (`cjc-eval` and `cjc-mir-exec`
     produce byte-identical results).
   - (c) Have proptest coverage AND at least one bolero fuzz target.
   - (d) Earn their place: a new builtin must deliver either a measured
     ≥3× hot-path speedup or enable a capability not expressible in
     existing CJC-Lang.
2. **Determinism is sacred.** SplitMix64 for all RNG. Kahan or
   BinnedAccumulator for all reductions. No FMA in SIMD kernels.
   BTreeMap everywhere — never HashMap with random iteration. Weight
   hashes must reproduce across reruns.
3. **Do not break the compiler pipeline.**
   ```
   Lexer → Parser → AST → [TypeChecker] → HIR → MIR → [Optimize] → Exec
   ```
4. **Both executors must agree.** Every new feature must work in
   `cjc-eval` AND `cjc-mir-exec` with identical semantics.
5. **Do not break v1 or v2.1.** `tests/chess_rl/` (66 tests) and
   `tests/chess_rl_v2/` (72 tests) must remain all-green.
6. **No bypass tricks.** No `#[ignore]` to hide failures. No
   cherry-picked seeds. No silent refactors. If a target cannot be met,
   report the honest number and document the gap.
7. **No external libraries.** Repeated because it matters: if you need
   matmul faster, write a native builtin in `cjc-runtime`. If you need
   a new chess-specific kernel, write a native builtin. Do not reach
   for PyTorch, ndarray, rayon::prelude imports not already in the
   workspace, or any other external dependency.

---

## STARTING STATE (v2.1 Phase D, 2026-04-09)

- **Training:** 60 episodes, max_moves=25, lr=0.001, temp 1.2→0.8,
  seed=42, `cjc-eval` backend.
- **Wall clock:** 38.92 min (training ~16 min, eval ~23 min)
- **vs random (20 games):** 0 W / 20 D / 0 L → WR 0.500
- **vs material-greedy (10 games):** 0 W / 10 D / 0 L → WR 0.500
- **Snapshot gauntlet (K=32):** 1000 → 1000 (Δ +0)
- **Final weight hash:** `-1596143894472527787` (deterministic)
- **Infra gates:** 7/7 ✅
- **ML gates:** 0/3 ❌

### Diagnosed failure causes

1. **Zero-terminal-reward starvation.** All 60 training episodes hit
   `n_moves=25` with `terminal_reward=0`. No checkmate or stalemate was
   ever reached inside the move cap.
2. **Greedy eval → piece shuffling.** PGN inspection shows plausible
   2-4 opening plies followed by `c5↔f5` / `a7↔a8` oscillation until the
   move cap fires. No cost signal for repetition.
3. **Interpreter throughput ceiling.** ~16.7-19.2 s/episode caps
   affordable training runs at ~60 episodes in a 20-minute budget.

---

## IMPROVEMENT SCOPE (tiered)

### Tier 1 — Cheap ML fixes (no new builtins needed)

These should deliver most of the ML win for ~1 day of work. Start here.

**T1-a. Raise `max_moves` in training from 25 → 80.**
- Gives rollouts a realistic chance to hit checkmate/stalemate.
- Expected effect: non-zero terminal reward on some fraction of
  episodes, providing real A2C signal.

**T1-b. Move-count penalty.**
- Terminal reward becomes `base_reward - 0.001 * n_moves`.
- Tiny penalty → agent prefers deciding the game quickly.
- Must be differentiable through the reward-to-go / GAE pipeline
  unchanged (scalar constant subtraction).

**T1-c. Threefold repetition detection.**
- Add a position hash (Zobrist-style over piece × square, side to
  move, castling rights, en-passant file) computed in pure CJC-Lang.
- Track `position_history: Map<i64, i64>` across the game.
- `terminal_status` returns `draw_repetition` when any position count
  ≥ 3.
- This is a **documented v2 limitation** being closed here.

**T1-d. Stochastic eval policy.**
- `select_action_temp_eval(scores, temp=0.15)` — low-temperature
  softmax sampling instead of pure argmax at eval time.
- Breaks deterministic shuffling loops at eval; still heavily biased
  toward the best move.

**T1-e. Add `repetition_draw` column to CSV log.**
- One extra column: `1` if the episode ended by repetition, else `0`.
- Enables post-hoc diagnosis of "how much did the new rule fire."

**T1 Gate (re-run Phase D with identical budget):**
- ≥60% vs random (currently 50%)
- ≥55% vs greedy (currently 50%)
- Elo gain ≥ +25 over 8 gauntlet games (currently 0)
- ≥20/60 training episodes end with non-zero terminal reward
- Budget: ≤45 min wall clock, backend `cjc-eval`
- Parity gate: both executors still byte-identical on the
  component-level tests (new repetition/penalty/eval-sampling paths
  added to parity suite).

If T1 hits the gate → ship v2.2 with a second honest LinkedIn post and
stop. If T1 misses → proceed to T2.

### Tier 2 — Profiling infrastructure (new native builtins OK)

Before touching the hot path, measure it. Add:

**T2-a. `profile_counter_start(name: String) -> i64`** and
`profile_counter_stop(id: i64) -> f64` native builtins.
- Nanosecond-precision counters using `std::time::Instant` (which IS
  deterministic in the sense of "measures wall clock" — the recorded
  values are not reproduced bit-identically across runs, but the
  *program behavior* is. Profiling counters must NOT feed back into
  program state or the weight hash.)
- Enforce via a compile-time flag or a runtime check that profile
  counters cannot influence any tensor op or RNG draw.

**T2-b. `profile_dump(path: String, counters: Map<String, f64>) -> i64`**
native builtin.
- Writes CSV: `name,count,total_ns,mean_ns,p50_ns,p99_ns`.

**T2-c. Instrument the hot path.** Wrap `score_moves`, `make_move`,
`compute_logsumexp`, `gradgraph_backward`, `adam_step`.

**T2 Gate:** Publish a flamegraph-equivalent CSV showing where
≥80% of per-episode time is spent. No ML gate attached to T2 — it's
a measurement step that unlocks T3.

### Tier 3 — Interpreter / runtime speedup (new native builtins expected)

Only if T1 misses the gate. Based on T2 data, pick the hottest path.
Candidates, in order of expected payoff:

**T3-a. Native `score_moves_batch(weights: List<Tensor>, feature: Tensor, legal_moves: List<Move>) -> Tensor` builtin.**
- Replaces the per-step CJC-Lang loop that builds `[64, num]` selector
  matrices and multiplies them through `g.matmul`.
- Computes the forward pass of the policy head in a tight Rust loop.
- Must produce the **identical** `[num_legal_moves]` score vector that
  the existing CJC-Lang path produces, bit-for-bit.
- Expected speedup: 5-10× on the full-game forward pass.

**T3-b. Native `rollout_episode_kernel(weights, opponent, config) -> Rollout` builtin.**
- Runs a full episode in native Rust, recording:
  - Per-step feature tensors
  - Per-step legal move lists
  - Per-step chosen actions
  - Per-step log-prob scalars
  - Per-step value estimates
  - Terminal reward
- Returns a `Rollout` struct (as a CJC-Lang record) that the CJC-Lang
  A2C driver then consumes for the GradGraph backward pass.
- Determinism requirement: must thread the SplitMix64 state through
  explicitly and return the advanced state.
- Parity requirement: output `Rollout` must be byte-identical to the
  pure-CJC-Lang rollout on a fixed seed.
- Expected speedup: 20-50× on rollout wall clock.

**T3-c. Persistent `GradGraph` reuse.**
- Currently a new `GradGraph` is built per step. If T2 shows graph
  construction dominates, expose a `g.reset()` that keeps the
  parameter tensors pinned and clears only the intermediate nodes.
- Must maintain topological-sort determinism.

**T3 Gate:**
- Per-episode wall clock ≤ 2.0 s (currently ~17-19 s) → ≥8.5× speedup.
- Cross-executor parity: still byte-identical on all existing tests
  AND new native kernel returns match pure-CJC-Lang on ≥1,000 random
  rollouts.
- Cross-run determinism: weight hash after 500 episodes reproduces.

### Tier 4 — Architecture / re-train (T3 must land first)

**T4-a. Wider trunk:** 48 → 128 hidden units. Parameter count
~45k → ~195k. Justify with T3 throughput budget.

**T4-b. 500-episode real training run.** This is what the original
prompt asked for. Only attempt after T3 brings per-episode cost under
2.0 s.

**T4 Gate:**
- ≥70% vs random (original target)
- ≥30% vs greedy (original target)
- Elo gain ≥ +100 over 24 gauntlet games
- Wall clock ≤ 20 min for 500-episode training + full eval suite
- If missed: ship whatever number you measured, document honestly.

---

## HARD CONSTRAINTS (repeat for emphasis)

- **No external libraries. No new crates. No subprocess calls.**
- **Builtins are allowed** — if you need native speed or a new
  capability, add a builtin via the wiring pattern
  (`cjc-runtime/src/builtins.rs` + `cjc-eval/src/lib.rs` +
  `cjc-mir-exec/src/lib.rs`).
- **Every new builtin needs:**
  - Unit tests (≥5)
  - Proptest (≥1)
  - Bolero fuzz target (≥1)
  - Cross-executor parity test
  - NoGC verifier compatibility (if called from NoGC-verified code)
- **No cherry-picking.** If the T1 gate misses, document and escalate.
  Do not retry with different seeds until something looks good.
- **No `#[ignore]` to hide failures.** Long-running tests get
  `#[ignore]` only if they already had it in v2.1.

---

## DEVELOPMENT WORKFLOW

### Step 1 — Re-read v2.1 landing surface

Reconfirm starting state:
- `tests/chess_rl_v2/source.rs` — 1,715-LOC PRELUDE
- `tests/chess_rl_v2/test_engine.rs`, `test_model.rs`,
  `test_training.rs`, `test_parity.rs` — 72 tests
- `tests/test_chess_rl_v2_phase_d.rs` — Phase D driver
- `bench_results/chess_rl_v2_1/` — honest baseline artifacts

### Step 2 — Tier 1 implementation

Changes are all in `tests/chess_rl_v2/source.rs` PRELUDE. No Rust
changes for Tier 1. Add tests in `tests/chess_rl_v2/test_engine.rs`
(repetition) and `tests/chess_rl_v2/test_training.rs` (penalty, eval
sampling).

### Step 3 — Tier 1 parity gate

Run:
```
cargo test --test test_chess_rl_v2 --release
```
All 72 existing + ≥10 new tests must pass on both executors.

### Step 4 — Tier 1 re-run of Phase D

Copy `tests/test_chess_rl_v2_phase_d.rs` → `test_chess_rl_v2_2_phase_d.rs`
or gate the new behavior behind a config flag. Run:
```
cargo test --release --test test_chess_rl_v2_2_phase_d \
  phase_d_training_run -- --ignored --nocapture
```
Capture new artifacts under `bench_results/chess_rl_v2_2/`.

### Step 5 — Honest gate check

Compare measured numbers against T1 gate. Write a one-page report in
`docs/chess_rl_v2/PHASE_D_v2_2.md` with:
- Raw numbers
- Gate pass/fail
- Percentage of episodes with non-zero terminal reward
- Percentage of episodes ended by repetition
- PGN spot-checks

If T1 passes, go to Step 7. If T1 fails, go to Step 6.

### Step 6 — T2 profiling + T3 native kernels (only if needed)

For each new builtin:
1. Design note (input/output type, determinism story, parity story)
2. Implementation in `cjc-runtime/src/builtins.rs`
3. Wiring in `cjc-eval/src/lib.rs` and `cjc-mir-exec/src/lib.rs`
4. Unit tests in `tests/test_<builtin_name>.rs`
5. Proptest coverage
6. Bolero fuzz target
7. Cross-executor parity test

Then revisit Step 4 with the new kernels enabled.

### Step 7 — Regression sweep

```
cargo test --workspace --release
```
Must pass at the same count as v2.1 (5,353) plus the new tests added
in v2.2. Zero regressions.

### Step 8 — Documentation and vault updates

- Update `docs/chess_rl_v2/README.md` with v2.2 addendum
- Update `CJC-Lang_Obsidian_Vault/09_Showcase/Chess RL v2.md`
- Run `python scripts/vault_audit.py` — must return "OK: all X wikilinks resolve"
- Update `MEMORY.md` with v2.2 numbers

### Step 9 — Honest LinkedIn post

Same tone and structure as `LINKEDIN_POST_v2_1.md`. Lead with numbers.
If T1 gate passed: frame it as "the known fix worked; here's the real
number." If T1 failed: frame it as "we tried the cheap fixes, here's
what happened, here's the next lever."

---

## OUTPUT FORMAT

Return results organized as:

```
FILE: path/to/file.rs
<code>
```

Then provide:

```
Test Summary:
  New tests:       X
  Existing tests:  Y (all passing)
  Workspace tests: Z (no regressions)
  Parity gate:     byte-identical ✅
  Determinism:     weight hash reproduces ✅
```

And:

```
Phase D v2.2 Results:
  Training: ... episodes, ... min wall clock
  vs random:       W / D / L  (WR ...%)
  vs greedy:       W / D / L  (WR ...%)
  Elo gauntlet:    ... → ... (Δ ...)
  Non-zero rewards: ... / ... episodes
  Repetition draws: ... / ... episodes
  Gate:            PASS / FAIL (honest)
```

---

## HARD RULE

If at any point a required change would break:
- Determinism
- The compiler pipeline
- Cross-executor parity
- The NoGC boundary
- v1 or v2.1 existing tests

you must:
1. **Stop.**
2. **Report the conflict.**
3. **Propose an alternative design that preserves the invariant.**

Never force an unsafe implementation. Never cherry-pick. Never reach
for an external library — that is the one rule this entire prompt is
built around.

---

## Appendix: Why "fix the interpreter, not the hyperparameters"

The v2.1 post said "the next step is interpreter hot-path optimization,
not more RL tricks." Here is the measurement behind that sentence:

- **CJC-Lang per-episode cost:** ~16.7 s (MIR) / ~19.2 s (eval)
- **Equivalent native Rust chess rollout:** ~0.1 s (estimated from the
  64-cell board and ~25 moves per episode)
- **Slowdown factor:** ~170×

That 170× gap is not a reward-shaping problem. It is an interpreter
dispatch problem. Tier 1 buys us an ML signal. Tier 3 buys us the
budget to actually train on that signal for 500+ episodes.
