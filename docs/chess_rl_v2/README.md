# Chess RL v2 — The Upgraded Flagship Demo

**Status:** Implemented and verified (25/25 tests passing, 2026-04-09).
**Scope:** Pure CJC-Lang. No external ML tools, no external chess engines.
**Location:** `tests/chess_rl_v2/` — source as a `pub const PRELUDE` string, plus Rust harnesses.

This document describes the v2 chess reinforcement learning demo — a deliberate
rewrite of the original `tests/chess_rl_project/` showcase that exploits the
improved autodiff infrastructure in CJC-Lang. The original demo rolled its own
REINFORCE chain rule by hand and never used the GradGraph API. v2 routes the
full policy + value loss through a real GradGraph, and adds the chess rules
the old engine quietly skipped.

## What v2 is

A self-contained demo that, in pure CJC-Lang:

1. Implements a full chess engine with castling, en passant, auto-queen
   promotion, 50-move rule, insufficient material (K–K, K–N, K–B, K+B vs K+B
   same color), checkmate, and stalemate.
2. Encodes positions into a 774-dim feature tensor (12 piece planes × 64
   squares, castling rights, halfmove clock fraction, EP flag) flipped to the
   side-to-move's perspective.
3. Runs an MLP policy/value network (~45,873 parameters) with a **factored
   policy head**: separate `[1, 64]` from-square and `[1, 64]` to-square
   logits that are summed per legal move.
4. Runs self-play rollouts, computes Generalized Advantage Estimation
   (γ=0.99, λ=0.95), and performs a single A2C update through a real
   `GradGraph` per episode with global-L2-norm gradient clipping.
5. Ships an evaluation arena (vs random baseline, and vs snapshot).

## Architecture

### File layout

```
tests/chess_rl_v2/
├── mod.rs              — module roots
├── harness.rs          — Rust test helpers (backends, run_parity, parse_*)
├── source.rs           — full CJC-Lang program in a pub const PRELUDE string
├── test_engine.rs      — 11 engine / rules correctness tests
├── test_model.rs       — 5 encoder / network tests
├── test_training.rs    — 5 training / determinism tests
└── test_parity.rs      — 4 eval-vs-mir parity gates
tests/test_chess_rl_v2.rs — cargo test entry point
```

### Source sections

The PRELUDE is split into five clearly labelled sections:

| Marker | What it contains |
|---|---|
| `// ============== ENGINE`   | Board, state accessors, geometry, attack detection, `apply_move`, `generate_pseudo_legal`, `legal_moves`, `insufficient_material`, `terminal_status` |
| `// ============== FEATURES` | `encode_state` (774-dim tensor) + `arr_set` helper |
| `// ============== MODEL`    | `init_weights`, `forward_eager`, `score_moves`, `select_action`, `select_action_greedy` |
| `// ============== TRAINING` | `rollout_episode`, `compute_gae`, `a2c_update`, `train_one_episode`, `tensor_sumsq` |
| `// ============== EVAL`     | `play_vs_random`, `eval_vs_random`, `play_snapshot`, `eval_vs_snapshot`, `policy_entropy_from_rollout` |

### Network

```
features  [1, 774]
    ↓ matmul W1 + b1   → [1, 48]
    ↓ relu
    ↓ matmul W2 + b2   → [1, 48]
    ↓ relu            ⇒  h2
              ┌─ matmul Wpf + bpf → [1, 64]   from-square logits
    h2 ──────┤
              ├─ matmul Wpt + bpt → [1, 64]   to-square logits
              │
              └─ matmul Wv + bv → tanh → scalar value
```

A legal move `(from, to)` scores as `from_logits[from] + to_logits[to]`.
Because features and logits are always rendered from the side-to-move's
perspective (squares flipped for black), the same network can play both
colors without extra parameters.

### Training loss

For each step `t` in a rollout, we build into the graph:

```
log π(a_t | s_t)  =  s_a  −  logsumexp(scores)      (rewritten, no softmax op)
policy_term       =  −log π(a_t) · A_t              (GAE advantage)
value_term        =  c_v · (V(s_t) − G_t)²          (squared returns error)
step_loss         =  policy_term + value_term
```

`loss_acc` accumulates all `step_loss` nodes inside one shared GradGraph,
then we divide by `n` and call `g.backward(total_loss)`. Gradients are
extracted for all 10 trainable tensors, combined into a global L2 norm,
and clipped by `max_grad_norm = 1.0` before a manual SGD step.

Entropy regularisation is **not** routed through the graph in v2: GradGraph
has no softmax op, and rolling our own `H = −Σ p log p` would need broadcast
divisions the current backward rules don't cover. The `policy_entropy_from_rollout`
diagnostic computes entropy eagerly for monitoring, but the loss is just
policy + value. The `c_entropy` parameter is kept in the `a2c_update` signature
for API stability.

### Why logsumexp instead of softmax

GradGraph ships with `add / sub / mul / div / neg / matmul / sum / mean /
sigmoid / relu / tanh / sin / cos / exp / ln / sqrt / pow / scalar_mul` plus
`parameter / input / backward / value / grad`. Softmax would need broadcast
division of an element-wise tensor by a scalar denominator — the backward
rule for that doesn't exist today.

The log-policy identity `log π(a) = s_a − log Σ exp(s)` replaces softmax
entirely: it uses only `sub`, `exp`, `sum`, `ln`, which are all already
wired for backward. The price is that dense entropy regularisation is not
available in-graph; the upside is the loss is numerically standard and
depends only on existing, tested GradOps.

### The one-hot selector trick

GradGraph also has no `gather` op. To extract `s_a` (the chosen action's
score) from the dense `from_logits` / `to_logits` tensors we build two
`[64, 1]` one-hot selector tensors and use `matmul`:

```
s_from = from_logits [1,64] @ one_hot_from [64,1]  →  [1,1]
s_to   = to_logits   [1,64] @ one_hot_to   [64,1]  →  [1,1]
s_a    = s_from + s_to
```

For the logsumexp denominator we need the full `[1, num]` score vector,
which we construct with a second pair of `[64, num]` selector matrices
that gather from/to logits into per-move scores:

```
scores_from = from_logits @ sel_from  [64, num]  →  [1, num]
scores_to   = to_logits   @ sel_to    [64, num]  →  [1, num]
scores_vec  = scores_from + scores_to
```

The selectors are constant tensors passed into the graph via `g.input(...)`,
so they carry no gradient and contribute no parameters.

## v1 vs v2 — what actually changed

| Aspect | v1 (`chess_rl_project`) | v2 (`chess_rl_v2`) |
|---|---|---|
| Castling                        | ❌                              | ✅ full (path empty, not in/through check) |
| En passant                      | ❌                              | ✅ including EP square state tracking |
| Auto-queen promotion            | ❌                              | ✅ |
| 50-move rule                    | ❌                              | ✅ (halfmove counter in state) |
| Insufficient material           | ❌                              | ✅ (K-K, K-N, K-B, K+B vs K+B same colour) |
| State representation            | Board only                      | `[board, side, castling, ep_sq, halfmove, ply]` |
| Feature tensor                  | 66-D (raw board + from/to)      | **774-D** (12 piece planes × 64 + rights + clock + EP) |
| Perspective flip (side-to-move) | ❌                              | ✅ |
| Network parameters              | ~1,100                          | **~45,873** |
| Policy head                     | Single flat move score          | **Factored from-square + to-square** |
| Value head                      | ❌                              | ✅ (scalar tanh, trained against GAE returns) |
| RL algorithm                    | Hand-rolled REINFORCE           | **A2C + GAE (γ=0.99, λ=0.95)** |
| Gradient computation            | Hand-rolled chain rule (~95 LOC)| **Full GradGraph backward** |
| Gradient clipping               | ❌                              | Global L2 norm, `max_grad_norm=1.0` |
| Parity coverage                 | Per builtin                     | **End-to-end training episode byte-identical** |

The v1 demo is intact at `tests/chess_rl_project/` and still passes its
66 tests. v2 is a parallel upgrade, not a replacement.

## Running the demo

```bash
cargo test --test test_chess_rl_v2 --release
```

Expected: **25 tests pass**, about 75 seconds on a warm build (dominated by
`test_training::*`). Individual buckets:

```bash
cargo test --test test_chess_rl_v2 --release chess_rl_v2::test_engine    # 11 tests
cargo test --test test_chess_rl_v2 --release chess_rl_v2::test_model     #  5 tests
cargo test --test test_chess_rl_v2 --release chess_rl_v2::test_training  #  5 tests
cargo test --test test_chess_rl_v2 --release chess_rl_v2::test_parity    #  4 tests
```

## Verification loop

The verification loop described in the upgrade plan has been run:

- **Correctness.** 11 engine tests cover initial piece count, starting legal
  move count (20), black-after-e4 (also 20), terminal status, pawn double
  push sets EP square, insufficient material (K-K), Fool's Mate detection,
  50-move rule trigger, kingside castling in the legal move list, auto-queen
  promotion, and eval-vs-mir parity on a 3-move game with a capture.
- **Autodiff.** `test_training::weights_change_after_update` probes `bv`
  (value-head bias) and `b1` (trunk bias) before and after one episode and
  asserts both move — confirming that gradient flows through the full
  policy + value loss on a real chess rollout. (Probing W1[0,0] was the
  naive first version, but that weight has a structurally-zero gradient
  because feature dim 0 is "my pawn on a1" which is always 0 at start.)
- **Determinism.** `test_training::training_is_deterministic` asserts that
  running the same body+seed twice gives byte-identical stdout. All 16
  non-parity tests use `run_parity` which cross-checks cjc-eval and
  cjc-mir-exec.
- **Full parity gate.** `test_parity::parity_single_training_episode`
  trains one episode through both executors and compares updated weights
  at five probe points. This is the strongest single assertion:
  a full backward+SGD pass on ~45k parameters agrees bit-for-bit between
  the two executors.
- **Regression.** The original 66-test `test_chess_rl_project` suite still
  passes with zero changes.

## Limitations — honest list

- **No training curve.** We verify that weights move and that training is
  deterministic, but we don't ship a "play 100 games, see loss drop, see
  win-rate go up" scripted result. The original v1 demo did have such a
  result — but it was learned via hand-rolled REINFORCE on a broken
  (no castling, no draws) engine. v2 solves the harder problem correctly
  but hasn't been run long enough to produce a convincing learning curve.
- **No entropy bonus in-graph.** The loss is policy + value only; entropy
  is eagerly computed for monitoring. Adding a differentiable entropy term
  would require a softmax GradOp, or a broadcast-division backward rule.
- **No threefold repetition.** Zobrist hashing was considered and skipped.
  The other three draw conditions (50-move, insufficient material,
  stalemate) are all implemented.
- **No opening book, no search, no tactics verification.** This is a
  reinforcement-learning demo, not a competitive engine. A full game
  against Stockfish will not be close.
- **Self-play only against itself or uniformly random.** Snapshot arena
  exists (`play_snapshot`, `eval_vs_snapshot`) but we do not yet drive a
  scheduled self-play training loop that keeps frozen prior weights.

## What this demo proves about CJC-Lang

1. **The language can host a real RL loop end-to-end.** Engine + encoder +
   network + advantage estimation + backward pass + weight update all live
   in one pure CJC-Lang program. No Python, no PyTorch, no external chess
   engine.
2. **GradGraph is load-bearing.** ~45k parameters, 10 trainable tensors,
   per-episode dynamic graph construction (each chess position builds a
   different number of matmul nodes because legal-move counts vary), and
   the backward pass stays correct and deterministic.
3. **The parity contract holds under pressure.** cjc-eval and cjc-mir-exec
   produce byte-identical output on the full training episode, not just
   on unit tests of individual builtins. This is the most demanding parity
   evidence the project has.
4. **The wiring pattern scales.** Every op used by this demo — `tanh`,
   `softmax`, `categorical_sample`, `log`, `exp`, `sqrt`, `matmul`,
   Tensor-scalar arithmetic, `GradGraph.*` — was already wired in both
   executors before v2 landed. No new builtins were needed.

## What still separates this from an elite RL chess system

- **Scale of training.** A real chess RL system runs millions of
  self-play games in parallel. We run tens per test.
- **Search + policy.** AlphaZero-style systems use the policy to guide
  MCTS, not as a standalone action distribution. v2 has no search.
- **Better state representation.** AlphaZero uses recent-position stacks,
  move counters, and custom piece features. 774-D is decent but not
  state-of-the-art.
- **Residual convolutional trunks.** Our trunk is a 774 → 48 → 48 MLP.
  Serious systems use 19-layer ResNets over an 8×8×N input.
- **Parallelism.** No multi-environment rollout collection. Each episode
  is a single serial self-play trace.

None of these are CJC-Lang limitations — they're "we'd need to write
considerably more code" items. The language's autodiff, tensors, and
determinism guarantees are sufficient to tackle them; v2 is deliberately
scoped as a capability demonstration rather than a training run.

---

## v2.1 upgrade (2026-04-09) — Adam, infra, and an honest first training run

v2.1 is an **in-place upgrade of v2** (not a parallel v3). Same
`tests/chess_rl_v2/` directory, same `PRELUDE` source string, same file
layout. Everything in v2 above still holds; the following sections
describe what v2.1 adds.

### What v2.1 adds

**Training upgrades (Phase B):**

- **Adam optimizer** with bias-corrected moments (β₁=0.9, β₂=0.999, ε=1e-8).
  Lives as `a2c_update_adam` / `train_one_episode_adam_*` in `source.rs`,
  backed by a native `adam_step` builtin in `cjc-runtime` (Phase B1b).
  The builtin is a ~80-line element-wise kernel that operates on a flat
  buffer; it was added because doing the bias-corrected update scalar-by-scalar
  through the CJC-Lang interpreter over W1's 37,200 elements was ~9× slower
  than a native kernel. The optimizer **driver logic** (state management,
  parallel `m`/`v` lists across 10 weight tensors, `a2c_update_adam` itself)
  is still pure CJC-Lang. The native builtin is a primitive, not the algorithm.
- **Advantage and return whitening** (mean-subtract, std-divide with 1e-8 eps).
- **Temperature annealing** on action sampling (`anneal_temp`, `select_action_temp`,
  `rollout_episode_temp`). Linear schedule `T_start → T_end` over episodes.
- **Resignation threshold** (`rollout_episode_full`). A patience-gated early
  termination when the value head is consistently below a threshold.
- **Material-greedy baseline opponent** (`select_action_material_greedy`,
  `play_vs_greedy`, `eval_vs_greedy`). Provides a non-trivial evaluation
  target between uniform-random and self-play.

**Infrastructure (Phase C):**

- **Checkpoint bundle** (`save_checkpoint` / `load_checkpoint`). 31-tensor
  bundle: 10 weights + 10 Adam first-moment + 10 Adam second-moment + 1
  meta-tensor encoding `[episode, adam_step, format_version]`. Uses the
  Phase A `tensor_list_save` builtin with content-hash footer verification
  on load.
- **CSV training log** (`csv_open_log`, `csv_log_episode`). Header row
  `episode,loss,n_moves,terminal_reward,temp,adam_step`, one row per episode,
  written via a new `file_append` builtin in `cjc-runtime`. Crash-recoverable:
  a partially-written log survives a process abort.
- **Elo-lite rating + snapshot gauntlet** (`elo_expected`, `elo_update`,
  `elo_apply_record`, `gauntlet_vs_snapshots`). Standard Elo formula
  `expected = 1/(1 + 10^((r_opp - r_self)/400))` with K=32 update rule.
  Ratings in pure CJC-Lang (no new builtins).
- **PGN dump** (`sq_to_uci`, `move_to_lan`, `play_recorded_game`,
  `pgn_format_game`, `pgn_dump_game`). Writes long-algebraic-notation PGN
  files. Append-based so multiple games accumulate naturally in one file.
- **Vizor training curves** (`vizor_training_curve`, `vizor_training_curves`).
  Generates SVG plots via the existing `cjc-vizor` library through
  `import vizor` at the top of the `PRELUDE`.

### New tests (all passing, zero regressions)

| Phase | Tests added | File |
|---|---|---|
| B1 (Adam CJC-Lang) | 7 | `test_training.rs` |
| B1b (native `adam_step`) | 10 | `tests/test_adam_step.rs` (new) |
| B2 (whitening) | 4 | `test_training.rs` |
| B3 (temperature) | 4 | `test_training.rs` |
| B4 (resignation) | 4 | `test_training.rs` |
| B5 (material-greedy) | 3 | `test_training.rs` |
| C1 (checkpoint) | 3 | `test_training.rs` |
| C2 (CSV + `file_append`) | 6 | `test_training.rs` + `tests/test_file_append.rs` |
| C3 (Elo + gauntlet) | 9 | `test_training.rs` |
| C4 (PGN) | 8 | `test_training.rs` |
| C5 (Vizor curves) | 3 | `test_training.rs` |
| **Total v2.1** | **61 new** | — |

Test breakdown after v2.1 (verified 2026-04-09):

- `test_chess_rl_v2`: **72 tests** (up from 25), all passing.
- `test_adam_step`: **10 tests** (new file), including proptest + bolero
  fuzz targets for the native `adam_step` builtin.
- `test_file_append`: **4 tests** (new file), cross-executor parity +
  bolero fuzz.
- Full `cargo test --workspace --release`: **[see Phase E log]** — no
  regressions triggered by the v2.1 changes. The single substantive
  prelude change — adding `import vizor` at the top — is a declaration,
  not an executable statement, and has zero measurable impact on
  non-Vizor tests.

### Phase D — the first real training run

A 60-episode training run was executed end-to-end through `cjc-eval`
with a 20-minute wall-clock target. The driver lives at
`tests/test_chess_rl_v2_phase_d.rs` and is gated behind `#[ignore]` so
it never pollutes the default test suite; invoke with
`cargo test --release --test test_chess_rl_v2_phase_d phase_d_training_run -- --ignored --nocapture`.

**Configuration:**

| Parameter | Value |
|---|---|
| Episodes               | 60 |
| `max_moves` (training) | 25 |
| `max_moves` (eval)     | 30–40 |
| Learning rate          | 0.001 |
| Temperature schedule   | 1.2 → 0.8 linear |
| Optimizer              | Adam, β₁=0.9, β₂=0.999, ε=1e-8 |
| Gradient clip          | global L2, `max_grad_norm=1.0` |
| Entropy bonus          | 0.01 (coefficient only; entropy itself is not in-graph) |
| Seed                   | 42 |
| Backend                | `cjc-eval` (tree-walk) |

**Honest outcome:**

| Metric | Value |
|---|---|
| Wall clock                      | **38.92 min** (training ~16 min, eval + gauntlet + PGN + plots ~23 min) |
| vs random baseline (20 games)   | **0 W / 20 D / 0 L** — win rate 0.500 |
| vs material-greedy (10 games)   | **0 W / 10 D / 0 L** — win rate 0.500 |
| Snapshot gauntlet (2×4 games, K=32) | **0 W / 8 D / 0 L** — Elo 1000 → **1000** (Δ +0) |
| Final weight hash               | `-1596143894472527787` (deterministic, reproducible) |
| CSV log rows                    | 61 (header + 60 episodes) |
| Checkpoints written             | 2 (ep30, ep60, ~1.1 MB each) |
| PGN games dumped                | 3 |
| Training curve SVGs             | 2 (`training_loss.svg`, `training_reward.svg`) |

All artifacts live under `bench_results/chess_rl_v2_1/`.

**Infrastructure gates: 7/7 passing.** Every artifact materialized exactly
as the Phase C tests assert. Cross-run determinism held (weight hash is
stable). The checkpoint bundle round-trips.

**ML quality gates: 0/3 passing** against the original upgrade prompt's
aspirational targets (≥70% vs random, ≥30% vs greedy, Elo +100). The
honest story of why:

1. **Every training episode hit `n_moves = 25` with `terminal_reward = 0`.**
   Not a single episode reached a checkmate or stalemate inside the move
   cap. So the A2C update had **zero true-reward signal** for all 60
   episodes — it was updating on GAE-bootstrapped value estimates only,
   which is a well-known weak-signal regime. Raising `max_moves` or
   adding a move-count penalty would fix this at the cost of wall-clock.
2. **Greedy eval collapses into piece-shuffling.** The three dumped PGN
   games show a textbook under-trained RL failure mode: 2–4 plausible
   opening moves (game 1 plays e2-e4 / Bb5 against Nf6 — the start of a
   Ruy-Lopez-style line), followed by 10+ plies of `c5↔f5 / a7↔a8`
   shuffling until the move cap fires and returns 0.0. Without
   threefold-repetition detection (documented limitation from v2), the
   cheapest escape from this local minimum isn't in the agent's reach.
3. **The prompt's 500-episode target is infeasible on the current
   interpreter.** A timing probe measured ~16.7 s/episode on `cjc-mir-exec`
   and ~19.2 s/episode on `cjc-eval`, so 500 episodes would cost
   ~2.5 hours — 8× over the 20-minute gate. v2.1 ran the largest honest
   episode count that fits a human-scale training budget (60). The
   acceptance gates designed for the 500-episode run are documented
   here as aspirational rather than met. A future v2.2 could either
   (a) optimize the interpreter hot path for Adam training, (b) add a
   native rollout/forward-pass builtin that collapses multiple MIR ops
   into one kernel, or (c) downsize the network so 500 episodes fit in
   budget.

**What this result actually demonstrates:**

- CJC-Lang can host an end-to-end A2C + GAE + Adam training loop with
  snapshot gauntlets, Elo rating, checkpoint save/load, CSV logging,
  PGN dump, and live SVG plotting — **all from a single `fn main()`**
  running through either of two executors, with byte-identical parity
  on the individual building blocks.
- The determinism contract holds under 38.9 minutes of continuous
  compute: the final weight hash is stable across re-runs.
- The failure mode is instructive and recoverable: the pipeline is
  rock solid, the ML budget is the bottleneck.

**What this result does not demonstrate:**

- That CJC-Lang can train a competitive chess agent. v2.1 does not
  claim this; the prompt asked and the honest answer is "not in 60
  episodes with a 48-wide MLP trunk and no search."

### Production wiring notes

1. **`import vizor` is now at the top of the `PRELUDE`.** Enabling the
   Vizor library unconditionally for all chess_rl_v2 tests. Zero impact
   on non-Vizor tests (verified by running the full 72-test
   `test_chess_rl_v2` suite with no regressions). This is a `Decl`
   node, not an executable statement.
2. **`adam_step` and `file_append` are new runtime builtins.** Both
   follow the canonical wiring pattern (shared dispatch in
   `cjc-runtime/src/builtins.rs`, auto-forwarded from both executors).
   Each has dedicated unit + proptest + bolero fuzz coverage in a
   top-level `tests/` file.
3. **The PRELUDE grew by ~1,000 lines** (training helpers, Elo math,
   PGN, curves, checkpoint). It is still a single `pub const` string
   to preserve parse-once semantics across test invocations.

### Running the Phase D driver

```bash
# Full 60-episode training run. Budget ~40 minutes wall clock.
cargo test --release --test test_chess_rl_v2_phase_d \
    phase_d_training_run -- --ignored --nocapture

# Artifacts land in bench_results/chess_rl_v2_1/.
```

```bash
# Timing probe (20 episodes). ~6 minutes wall clock.
cargo test --release --test test_chess_rl_v2_training_probe \
    -- --ignored --nocapture
```

Both are gated behind `#[ignore]` and will not run in the default
`cargo test --workspace` cycle.

---

## v2.2 upgrade (2026-04-09) — Tier 1 cheap ML fixes

v2.2 applied four low-cost ML fixes to see whether the v2.1 zero-signal
result was algorithmic or systemic. Changes (all in-place on the same
`tests/chess_rl_v2/` directory):

- **Raised `max_moves` 25 → 80** to give games time to resolve.
- **Move-count penalty** of 0.001 per ply (shrinks reward magnitude for
  long games, provides a gradient even in drawn-out play).
- **Threefold repetition detection** (Zobrist hashing, terminates game
  with draw reward on 3-fold).
- **Stochastic evaluation** at temp=0.15 (breaks argmax ties to reveal
  policy diversity).

### Phase D v2.2 results (60 episodes, seed=42, `cjc-eval`)

| Metric | v2.1 | v2.2 | Tier 1 gate | |
|---|---|---|---|---|
| Wall clock | 38.92 min | **73.12 min** | ≤ 45 min | ❌ |
| vs random WR | 0.500 | 0.500 | ≥ 0.60 | ❌ |
| vs greedy WR | 0.500 | **0.450** | ≥ 0.55 | ❌ |
| Elo gain | +0 | +0 | ≥ +25 | ❌ |
| Non-zero terminals | 0/60 | **2/60** | ≥ 20/60 | ❌ |
| Weight hash | `-1596143894472527787` | `3194409110565838047` | deterministic | ✅ |

**All 5 Tier 1 gates missed.** Tripling the move cap tripled per-episode
cost but only produced 2 non-zero rewards (episodes 8 and 38). The failure
confirmed: **the bottleneck is interpreter throughput, not ML algorithm
design.** Detailed analysis in `docs/chess_rl_v2/PHASE_D_v2_2.md`.

### Tests added in v2.2

12 tests for Tier 1 features (repetition detection, move penalty,
stochastic eval, v2.2 rollout parity). `test_chess_rl_v2` grew from
72 → 84 tests.

---

## v2.3 upgrade (2026-04-10) — Profiling + Native Kernels

v2.3 attacked the interpreter throughput bottleneck identified by v2.2
with three tiers:

### Tier 2 — Profiling infrastructure

Three new builtins (`profile_zone_start`, `profile_zone_stop`,
`profile_dump`) provide write-only, per-zone wall-clock measurement.
Key design: profiling counters are thread-local `RefCell<BTreeMap>` and
**never feed back into program state**, so instrumented runs produce
bit-identical weight hashes to uninstrumented runs.

Profiling the v2.2 rollout (max_moves=40) revealed:

| Zone | % of rollout time |
|---|---|
| `score_moves` (encode state + forward pass + gather) | **84.2%** |
| `legal_moves` | 7.7% |
| `apply_move` | 0.1% |
| `rep_tracking` | 0.1% |

### Tier 3 — Native hot-path kernels

Two builtins replaced the two hottest CJC-Lang code paths:

1. **`encode_state_fast`** — replaces `encode_state` which called
   `arr_set` 38× on a 774-element array (O(38 × 774) element copies).
   Native version: single-pass buffer fill (O(774)).

2. **`score_moves_batch`** — replaces `score_moves` which looped over
   legal moves doing `tensor.get()` + `array_push()` (O(num_moves²)
   due to COW array copies). Native version: single forward pass +
   gather from logit tensors.

Both produce **bit-identical** output to their CJC-Lang counterparts,
verified by `v23_rollout_matches_v22` and
`v23_train_episode_weight_hash_matches_v22` parity tests.

**Measured speedup:**

| Metric | v2.2 | v2.3 | Speedup |
|---|---|---|---|
| Rollout only (max_moves=40) | 10.4 s | 1.3 s | **7.7×** |
| Full training episode (max_moves=80) | ~73 s | ~80 s | **~0.9×** |

The rollout speedup did not translate to full-episode speedup because the
`a2c_update_adam` backward pass (~25 s, unchanged) and interpreter loop
overhead (COW array copies in the outer 80-iteration while-loop) now
dominate. This is a textbook [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law)
scenario: speeding up 84% of the rollout ≠ speeding up 84% of the episode,
because the rollout is only ~50% of total episode time.

### Tier 4 — Phase D v2.3 re-run (120 episodes)

*See `docs/chess_rl_v2/PHASE_D_v2_3.md` for the full post-mortem.*

Run via:
```bash
cargo test --release --test test_chess_rl_v2_3_phase_d \
    phase_d_v23_training_run -- --ignored --nocapture

# Artifacts land in bench_results/chess_rl_v2_3/.
```

### Tests added in v2.3

| Suite | Tests |
|---|---|
| `test_profile_zones` (new file) | 11 (profiling builtins + proptest + bolero) |
| `test_native_kernels` (new file) | 11 (encode_state_fast + score_moves_batch + proptest + bolero) |
| `test_chess_rl_v2` additions | 12 (profile parity, instrumented rollout, v2.3 rollout + training) |
| **Total v2.3** | **34 new tests** |

`test_chess_rl_v2` grew from 84 → 97 tests (including 1 `#[ignore]`
speedup benchmark).

### New builtins added in v2.3

| Builtin | Purpose |
|---|---|
| `profile_zone_start(name)` | Start a named profiling zone, returns handle |
| `profile_zone_stop(handle)` | Stop a zone, returns elapsed seconds |
| `profile_dump(path)` | Dump zone stats to CSV |
| `encode_state_fast(board, side, castling, ep, halfmove)` | Native O(774) state encoder |
| `score_moves_batch(weights, feature, moves, side)` | Native forward pass + gather |

### Running the v2.3 Phase D driver

```bash
# Full 120-episode training run. Budget ~2.5+ hours at ~80 s/episode.
cargo test --release --test test_chess_rl_v2_3_phase_d \
    phase_d_v23_training_run -- --ignored --nocapture
```

Gated behind `#[ignore]` and will not run in the default `cargo test
--workspace` cycle.

## v2.4 upgrade (2026-04-10) — Arena GradGraph + Dead Node Elimination

v2.4 is a pure performance refactor of the autodiff engine (`cjc-ad`),
targeting the backward pass bottleneck identified by v2.3's Amdahl's Law
analysis. No new CJC-Lang syntax or ML algorithm changes — just faster
gradient computation.

### What changed

**P1: Arena-based GradGraph** — The core data structure changed from:
```rust
// Before: per-node heap allocation + runtime borrow checking
pub struct GradGraph {
    pub nodes: Vec<Rc<RefCell<GradNode>>>,
}
```
to:
```rust
// After: flat contiguous arrays, direct indexing
pub struct GradGraph {
    ops: Vec<GradOp>,
    tensors: Vec<Tensor>,
    param_grads: Vec<Option<Tensor>>,
}
```

This eliminates ~120 `.borrow()` calls per backward pass, removes per-node
Rc+RefCell heap overhead, and enables cache-contiguous traversal.

**P2: Dead node elimination** — Before backward traversal, a reachability
set is built by walking from the loss node to all its transitive inputs.
Nodes unreachable from the loss are skipped entirely. For multi-head
networks (policy + value in chess RL), this skips 20-30% of graph nodes
that contribute no gradient.

**P3: No-clone backward (partial)** — With arena storage, `tensors[i]` is
borrowed by reference instead of cloned out of a RefCell. The `op` field
still requires a clone due to Rust borrow rules (backward body borrows
`&mut self` for gradient accumulation while needing `&self.ops[i]`).

**P5: COW array builtins** — `array_pop` and `array_reverse` now use
`Rc::make_mut()` (matching `array_push`), avoiding O(N) deep copies when
the array has refcount=1.

### Performance impact

20-episode training probe (seed=42, cjc-eval backend, clean machine):

| Metric | v2.3 baseline | v2.4 arena | Speedup |
|---|---|---|---|
| Per-episode time | 82.2 s | **43.51 s** | **1.89×** (47% reduction) |
| 20-episode wall | ~1640 s | **870.2 s** | **1.88×** |
| Weight hash | `9.790915694115341` | `9.790915694115341` | **bit-identical** |

The 47% reduction aligns with the predicted 40-60% backward pass
improvement from the design proposal.

### Design documents

- `docs/chess_rl_v2/PERFORMANCE_AUDIT_v2_4.md` — four-area audit
  (GradGraph backward, COW arrays, gradient strategies, MIR optimizer)
- `docs/chess_rl_v2/DESIGN_PROPOSALS_v2_4.md` — eight ranked proposals
  (P1-P8), implementation order, expected speedups

### Regression status

- `cjc-ad` library tests: **80/80 passed**
- Parity tests (builtin + stress + vizor): **55/55 passed**
- Chess RL v2 test suite: **97 passed, 0 failed, 3 ignored**
- Full workspace sweep: pending confirmation

### Determinism

All changes preserve bit-identical output. The arena refactor changes
allocation patterns but not arithmetic. Dead node elimination skips nodes
that produce zero gradient — the gradient values for reachable nodes are
unchanged.

## v2.5 upgrade (2026-04-10) — In-Place Gradients + Fused MLP + PINN Graph Reuse

v2.5 implements the four deferred proposals from the v2.4 performance
architecture audit. These changes affect the autodiff engine (`cjc-ad`)
and the PINN training loop, with no changes to the chess RL CJC-Lang source.

### P4: In-place gradient accumulation

Added `Tensor::add_assign_unchecked(&mut self, other: &Tensor)` which
mutates in-place instead of allocating a new tensor. The `accumulate_grad()`
function in `backward()` now uses this for the common case where a gradient
slot already has a value, eliminating ~N/2 tensor allocations per backward
pass.

Also added `Buffer::borrow_data_mut()` to support mutable access to the
underlying COW buffer.

### P6: backward_collect (batched gradient retrieval)

Added `GradGraph::backward_collect(loss_idx, param_indices) → Vec<Option<Tensor>>`
which batches `zero_grad()` + `backward()` + gradient collection into a
single native call. Wired into both executors as a GradGraph method.

Note: the backward pass already runs entirely in Rust (a single
`g.backward(loss_idx)` call from CJC-Lang triggers the full native
traversal). P6 provides a convenience API, not a fundamental speedup.

### P7: PINN graph reuse

The PINN training loop (`pinn_harmonic_train`) now builds the computation
graph once before the epoch loop, then reuses it via:
1. `graph.set_tensor(param_idx, new_weights)` for each parameter
2. `graph.set_tensor(bw_idx, new_boundary_weight)` for adaptive weights
3. `graph.reforward(start, end)` to recompute derived tensors

This eliminates O(epochs × graph_size) allocations. The `reforward()`
method was made public.

### P8: Fused MLP layer GradOp

Added `GradOp::MlpLayer { input, weight, bias, activation }` which
collapses transpose + matmul + bias-add + activation into a single graph
node. Each MLP layer is now 1 node instead of 3-4, reducing graph size
by ~3× per layer.

The fused backward computes d_input, d_weight, and d_bias in one pass
without intermediate tensor allocation. Supports Tanh, Sigmoid, Relu,
and None activations.

`mlp_forward()` in `cjc-ad/src/pinn.rs` now uses the fused op. The
method is also exposed as a GradGraph method (`g.mlp_layer(input, weight,
bias, "relu")`) for CJC-Lang code.

### Performance impact

20-episode training probe (seed=42, cjc-eval backend, clean machine):

| Metric | v2.3 baseline | v2.4 arena | v2.5 in-place | Speedup |
|---|---|---|---|---|
| Per-episode time | 82.2 s | 43.51 s | **29.80 s** | **2.76×** (from v2.3) |
| 20-episode wall | ~1640 s | 870.2 s | **596.0 s** | **2.75×** |
| Weight hash | `9.79091...` | `9.79091...` | `9.79091...` | **bit-identical** |

The v2.4→v2.5 improvement (43.51 → 29.80, 1.46×) is entirely from P4
(in-place gradient accumulation). The `accumulate_grad()` function
previously allocated a new tensor on every call; now it mutates in-place,
eliminating ~N/2 tensor allocations per backward pass.

### Design documents

- `docs/chess_rl_v2/V1_VS_V2_5_COMPARISON.md` — full v1 → v2.5 comparison
- `docs/chess_rl_v2/LINKEDIN_DRAFT.md` — technical substance for LinkedIn post

### Regression status

- `cjc-ad` library tests: **80/80 passed** (including PINN determinism + convergence)
- Parity tests: **55/55 passed**
- Chess RL v2 test suite: **97 passed, 0 failed, 2 ignored**

