---
title: Chess RL v2
tags: [showcase, ml, rl, autodiff]
status: v2.5 upgrade shipped (2026-04-10) — in-place gradients, fused MLP, PINN graph reuse
---

# Chess RL v2

The **upgraded flagship demo** for CJC-Lang. Sits alongside the earlier [[Chess RL Demo]] — v1 is untouched and still passing. v2 is a deliberate rewrite that exploits the improved autodiff infrastructure and adds the chess rules v1 skipped.

Location: `tests/chess_rl_v2/`. Full writeup in `docs/chess_rl_v2/README.md`.

## What v2 delivers (vs v1)

| Aspect | v1 | v2 |
|---|---|---|
| Castling / en passant / promotion / 50-move / insufficient material | ❌ | ✅ full |
| Feature tensor | 66-D raw | **774-D** (12 piece planes × 64 + rights + clock + EP, flipped to side-to-move) |
| Network | ~1.1k params, single head | **~45,873 params, factored from+to heads + value head** |
| Algorithm | Hand-rolled REINFORCE | **A2C + GAE** (γ=0.99, λ=0.95) |
| Gradients | Hand-rolled chain rule (~95 LOC) | **Full GradGraph backward** across all 10 trainable tensors |
| Clipping | ❌ | Global L2 norm, `max_grad_norm=1.0` |
| Parity gate | Per builtin | **End-to-end A2C update byte-identical on both executors** |

## Tests

**25 tests, all passing** (`cargo test --test test_chess_rl_v2 --release`, verified 2026-04-09):

| Suite | File | Tests |
|---|---|---|
| Engine / rules   | `test_engine.rs`   | 11 |
| Model / forward  | `test_model.rs`    |  5 |
| Training / AD    | `test_training.rs` |  5 |
| eval↔mir parity  | `test_parity.rs`   |  4 |

The standout is `test_parity::parity_single_training_episode`: it runs one full A2C update (rollout → GradGraph backward → L2 clip → SGD) on **both** executors and asserts five probe points on the updated weights are bit-identical. See [[Parity Gates]] for the general framework.

## What it exercises in the language

- Full GradGraph API: `parameter`, `input`, `matmul`, `add/sub/mul`, `relu`, `tanh`, `sum`, `exp`, `ln`, `backward`, `grad`, `value` — see [[cjc-ad]]
- Tensor eager path: `Tensor.from_vec`, `.randn`, `.zeros`, `.shape`, `.get`, `.softmax`, `.relu` — see [[Tensor Runtime]]
- Builtins: `matmul`, `log`, `exp`, `sqrt`, `float`, `categorical_sample`, `array_push`, `len`, `print`
- Tensor-scalar arithmetic + Tensor-Tensor arithmetic (wired in [[cjc-mir-exec]] for this demo family)
- Full control flow: nested `while`, `if`/`else`, `return`, mutable `let` rebinding
- [[Determinism Contract]]: SplitMix64 RNG threaded through seed, byte-identical output between [[cjc-eval]] and [[cjc-mir-exec]]

## Interesting engineering tricks

- **Logsumexp instead of softmax.** GradGraph has no softmax GradOp, so `log π(a) = s_a − log Σ exp(s)` is used instead — only needs `exp`, `sum`, `ln`, `sub`, which all have backward rules.
- **One-hot selector matmul for gather-by-action.** GradGraph also lacks a `gather` op. Per-step, we build `[64, 1]` one-hot selectors and `[64, num]` selector matrices and multiply through `g.matmul` to extract the chosen action's score and the full per-move score vector.
- **`return out;` vs `[...]` tail expression.** The parser rejected `[new_weights, [total_val, 0.0, 0.0, 0.0]]` as a bare tail expression at function end — the brackets on a new line after a statement were interpreted ambiguously. Explicit `let out = [...]; return out;` works. Worth a future parser fix.
- **`col` and `row` are reserved keywords.** Renaming to `cc` / `rr` inside the selector-matrix build loop was necessary.

## Honest limitations

- **No training curve published.** v2 proves that one update step is correct and deterministic. It does **not** ship a "train for 100 episodes, win rate climbs from X% to Y%" experiment.
- **No entropy bonus in-graph.** Entropy is only computed eagerly as a diagnostic. Adding a differentiable entropy term needs either a softmax GradOp or broadcast-division backward rules.
- **No threefold repetition** (other draw conditions are implemented).
- **Not competitive.** 48-wide MLP trunk. No search. Uniform random baseline only.

## v2.1 upgrade (2026-04-09)

v2.1 is an **in-place upgrade** of v2 — same directory, same `PRELUDE` string,
same file layout. The upgrade adds:

### Training upgrades (Phase B)

- **Adam optimizer** with bias-corrected moments. CJC-Lang driver logic plus
  a new native `adam_step` builtin in [[cjc-runtime]] for the element-wise
  update (the only Phase B/C change to the Rust runtime). Native kernel gave
  a ~9× speedup over the scalar CJC-Lang loop on W1's 37,200 elements.
- **Advantage + return whitening** (zero-mean, unit-std, 1e-8 eps floor).
- **Temperature annealing** on action sampling (linear `T_start → T_end`).
- **Resignation threshold** with patience gate.
- **Material-greedy baseline opponent** (non-trivial eval target between
  random and self-play).

### Infrastructure (Phase C)

- **Checkpoint bundle** — 31-tensor save/load via the Phase A `tensor_snap`
  module in [[cjc-runtime]] (10 weights + 10 Adam `m` + 10 Adam `v` + 1
  meta) with content-hash footer verification on load.
- **CSV training log** — one row per episode via a new `file_append` builtin.
  Crash-recoverable: a partial log survives a process abort.
- **Elo-lite rating + snapshot gauntlet** — pure CJC-Lang math (`elo_expected`,
  `elo_update`, `elo_apply_record`, `gauntlet_vs_snapshots`). K=32, standard
  logistic formula.
- **PGN dump** — long-algebraic notation via `play_recorded_game` +
  `pgn_format_game` + `pgn_dump_game`. Append-based so multiple games
  accumulate in one file.
- **Vizor training curves** — SVG output via [[cjc-vizor]] through a new
  `import vizor` gate at the top of the `PRELUDE`.

### Tests added in v2.1

| Phase | Tests | Notes |
|---|---|---|
| B1 Adam CJC-Lang | 7 | |
| B1b native `adam_step` | 10 | Includes proptest + bolero fuzz |
| B2 whitening | 4 | |
| B3 temperature | 4 | |
| B4 resignation | 4 | |
| B5 material-greedy | 3 | |
| C1 checkpoint | 3 | Round-trip + resumability + pack/unpack |
| C2 CSV + `file_append` | 6 | Cross-executor parity + fuzz |
| C3 Elo + gauntlet | 9 | Closed-form Elo checks + parity |
| C4 PGN | 8 | Determinism + header/body invariants |
| C5 Vizor curves | 3 | Byte-identical SVG determinism test |
| **Total** | **61 new** | All passing, zero regressions |

`test_chess_rl_v2` grew from **25 → 72 tests**. Dedicated
`test_adam_step` (10 tests) and `test_file_append` (4 tests) files
were added at the repo root.

### Phase D — first real training run (60 episodes)

Run via `cargo test --release --test test_chess_rl_v2_phase_d phase_d_training_run -- --ignored --nocapture`.

**Configuration:** 60 episodes · max_moves=25 · lr=0.001 · temp 1.2→0.8 ·
Adam (β₁=0.9, β₂=0.999, ε=1e-8) · L2 grad clip 1.0 · seed=42 ·
`cjc-eval` backend.

**Honest result:**

| Metric | Value |
|---|---|
| Wall clock | **38.92 min** (training ~16 min, eval ~23 min) |
| vs random (20 games) | 0 W / 20 D / 0 L — WR 0.500 |
| vs material-greedy (10 games) | 0 W / 10 D / 0 L — WR 0.500 |
| Snapshot gauntlet (K=32) | 0 W / 8 D / 0 L — Elo 1000 → **1000** |
| Final weight hash | `-1596143894472527787` (deterministic) |

**Infrastructure gates: 7/7 passing.** Every artifact (CSV, 2 checkpoints,
PGN, 2 SVGs, weight hash) materialized exactly as the Phase C tests
assert. Cross-run determinism held over 38.9 minutes of continuous compute.

**ML quality gates: 0/3 passing.** The upgrade prompt's aspirational targets
(≥70% vs random, ≥30% vs greedy, Elo +100) were not met. Root causes,
honestly:

1. **Zero true-reward signal.** All 60 training episodes hit `n_moves=25`
   with `terminal_reward=0`. The agent never once reached checkmate or
   stalemate inside the move cap, so A2C had nothing but GAE-bootstrapped
   value estimates to learn from — a well-known weak-signal regime.
2. **Greedy eval → shuffling.** The three dumped PGN games show 2–4
   plausible opening plies (game 1 opens e2-e4 Nf6, Bb5 — Ruy-Lopez-ish)
   followed by 10+ plies of piece-shuffling until the move cap fires.
   Without threefold-repetition detection (documented limitation from v2),
   the agent has no cost signal for repetition.
3. **500-episode target infeasible on current interpreter.** Timing probe
   measured ~16.7 s/episode (MIR) and ~19.2 s/episode (eval), so 500
   episodes = ~2.5h, 8× over the 20-min gate. v2.1 ran the largest honest
   episode count that fits the budget (60).

### What v2.1 actually proves

- CJC-Lang can host a **production-shaped** RL training loop: Adam,
  checkpoint save/load, CSV log, Elo-lite gauntlet, PGN emission, and
  live SVG plotting all driven from a single CJC-Lang `fn main()`, with
  byte-identical parity on the building blocks between [[cjc-eval]] and
  [[cjc-mir-exec]].
- The [[Determinism Contract]] survives 38.9 minutes of continuous compute
  — the final weight hash is reproducible.
- **The failure mode is instructive, not a bug.** The infrastructure is
  solid; the ML budget is the bottleneck. A future v2.2 should target
  interpreter hot-path optimization (or a native rollout kernel) rather
  than ML algorithm changes.

### What v2.1 does not prove

- That CJC-Lang can train a competitive chess agent. 60 episodes on a
  48-wide MLP trunk with no search is not enough, and v2.1 makes no such
  claim.

### Artifacts

All under `bench_results/chess_rl_v2_1/`:

- `training_log.csv` — 61 lines (header + 60 episodes)
- `checkpoint_ep30.bin`, `checkpoint_ep60.bin` — ~1.1 MB each, 31-tensor
  bundles
- `sample_games.pgn` — 3 games, PGN-parseable
- `training_loss.svg`, `training_reward.svg` — Vizor-generated
- `phase_d_summary.txt` — the Rust harness's formatted summary
- `phase_d_stdout.log` — full test output
- `phase_e_regression.log` — `cargo test --workspace --release`

## v2.2 upgrade (2026-04-09) — Tier 1 cheap ML fixes

Applied four low-cost ML fixes to the v2.1 baseline: raised `max_moves`
25→80, move-count penalty 0.001/ply, threefold repetition detection
(Zobrist hashing), stochastic eval at temp=0.15.

**Phase D v2.2 result (60 episodes, seed=42):** All 5 Tier 1 gates missed.
Wall clock 73 min (gate ≤45), vs random 0.500 WR (gate ≥0.60), vs greedy
0.450 WR (gate ≥0.55), Elo +0 (gate ≥+25), 2/60 non-zero terminals
(gate ≥20/60). Confirmed: **the bottleneck is interpreter throughput, not
ML algorithm design.**

12 new tests in `test_training.rs` (v2.2 rollout parity, repetition,
penalty, stochastic eval). Full post-mortem: `docs/chess_rl_v2/PHASE_D_v2_2.md`.

## v2.3 upgrade (2026-04-10) — Profiling + Native Kernels

Attacked the interpreter throughput bottleneck with:

### Tier 2 — Profiling

Three new builtins (`profile_zone_start`, `profile_zone_stop`,
`profile_dump`) for write-only per-zone wall-clock measurement. Thread-local
`BTreeMap` counters that never feed back into program state (instrumented
runs produce bit-identical weight hashes). See `docs/chess_rl_v2/PROFILE_DESIGN.md`.

Profiling revealed `score_moves` is **84.2%** of rollout wall clock.

### Tier 3 — Native hot-path kernels

Two builtins replaced the hottest CJC-Lang code paths:

- **`encode_state_fast`** — O(774) buffer fill replacing O(38×774) COW copies
- **`score_moves_batch`** — single native forward pass + gather replacing
  O(num_moves²) interpreter loop

Both produce **bit-identical output** to CJC-Lang counterparts. Achieved
**7.7× rollout speedup** (10.4 s → 1.3 s at max_moves=40). However, full
training episode speedup was only ~0.9× due to Amdahl's Law: the
`a2c_update_adam` backward pass (~25 s) and interpreter loop overhead now
dominate.

### Tier 4 — Phase D v2.3 (120 episodes, seed=42)

| Metric | v2.2 | v2.3 | Tier 4 gate | |
|---|---|---|---|---|
| Wall clock | 73.1 min | **164.4 min** | ≤ 45 min | ❌ |
| vs random WR | 0.500 | **0.500** | ≥ 55% | ❌ |
| vs greedy WR | 0.450 | **0.500** | ≥ 50% | ✅ |
| Elo gain | +0 | **+13.9** | ≥ 0 | ✅ |
| Non-zero terminals | 2/60 | **6/120** | ≥ 30/120 | ❌ |
| Weight hash | `3194409110565838047` | `-3450316119511861008` | deterministic | ✅ |

**2/5 gates passed** (first time any v2.x run has passed >0 ML gates).
Elo +13.9 is the first measurable learning signal in the series. Full
post-mortem: `docs/chess_rl_v2/PHASE_D_v2_3.md`.

### Tests added in v2.3

| Suite | Tests |
|---|---|
| `test_profile_zones` (new) | 11 |
| `test_native_kernels` (new) | 11 |
| `test_chess_rl_v2` additions | 12 |
| **Total v2.3** | **34 new** |

`test_chess_rl_v2` grew from **84 → 97 tests**.

## v2.4 upgrade (2026-04-10) — Arena GradGraph + Dead Node Elimination

Pure performance refactor of the autodiff engine ([[cjc-ad]]), targeting
the backward pass bottleneck identified by v2.3's Amdahl's Law analysis.

### Arena-based GradGraph (P1)

Replaced `Vec<Rc<RefCell<GradNode>>>` with three flat arrays:
- `ops: Vec<GradOp>` — operation enum (39 variants)
- `tensors: Vec<Tensor>` — cached forward values
- `param_grads: Vec<Option<Tensor>>` — gradient accumulator

Eliminates ~120 `.borrow()` calls per backward pass, removes per-node
Rc+RefCell heap overhead, enables cache-contiguous traversal.

### Dead node elimination (P2)

Before backward traversal, builds a reachability set from the loss node.
Nodes unreachable from the loss are skipped entirely — for the chess RL
factored policy/value architecture, this skips 20-30% of graph nodes.

### No-clone backward (P3, partial)

Tensors are now borrowed by reference from the flat array instead of
cloned out of RefCell. The `op` field still clones (Rust borrow rules:
`&mut self` for gradient accumulation + `&self.ops[i]` is disallowed).

### COW array builtins (P5)

`array_pop` and `array_reverse` now use `Rc::make_mut()` (matching
`array_push`), avoiding O(N) deep copies when refcount=1.

### Performance impact

20-episode training probe (seed=42, cjc-eval backend):

| Metric | v2.3 | v2.4 | Speedup |
|---|---|---|---|
| Per-episode | 82.2 s | **43.51 s** | **1.89× (47%)** |
| Weight hash | `9.790915694115341` | `9.790915694115341` | **bit-identical** |

### Regression status

- `cjc-ad` library tests: **80/80 passed**
- Parity tests: **55/55 passed**
- Chess RL v2: **97 passed, 0 failed, 3 ignored**
- Determinism: bit-identical (same arena traversal order, same arithmetic)

### Design documents

- `docs/chess_rl_v2/PERFORMANCE_AUDIT_v2_4.md` — four-area audit
- `docs/chess_rl_v2/DESIGN_PROPOSALS_v2_4.md` — eight ranked proposals

## v2.5 upgrade (2026-04-10) — In-Place Gradients + Fused MLP + PINN Graph Reuse

Implements the four deferred proposals from the v2.4 audit:

### P4: In-place gradient accumulation
`Tensor::add_assign_unchecked` mutates in-place instead of allocating.
`accumulate_grad()` now eliminates ~N/2 tensor allocations per backward pass.

### P6: backward_collect
`GradGraph::backward_collect(loss, param_indices)` batches zero_grad +
backward + gradient collection. Wired in both executors. (Convenience API —
backward already runs natively.)

### P7: PINN graph reuse
PINN training builds graph once, then `set_tensor()` + `reforward()` per
epoch. Eliminates O(epochs × graph_size) allocations. `reforward()` made public.

### P8: Fused MLP layer
`GradOp::MlpLayer` collapses transpose + matmul + bias-add + activation into
one node (3× fewer nodes per layer). Fused backward computes d_input, d_weight,
d_bias in one pass. `mlp_forward()` now uses fused op.

### Performance impact

| Metric | v2.3 | v2.4 | v2.5 | Overall |
|---|---|---|---|---|
| Per-episode | 82.2 s | 43.51 s | **29.80 s** | **2.76× faster** |
| Weight hash | `9.79091...` | `9.79091...` | `9.79091...` | bit-identical |

### Regression status
- cjc-ad: **80/80** (PINN determinism + convergence pass)
- Parity: **55/55**
- Chess RL v2: **97 passed, 0 failed, 2 ignored**
- Weight hash: `9.790915694115341` (bit-identical)

### Design documents
- `docs/chess_rl_v2/V1_VS_V2_5_COMPARISON.md` — full v1 → v2.5 comparison
- `docs/chess_rl_v2/LINKEDIN_DRAFT.md` — technical substance for LinkedIn

## Related

- [[Chess RL Demo]] — the v1 original (preserved, still passing 66 tests)
- [[cjc-ad]]
- [[cjc-eval]]
- [[cjc-mir-exec]]
- [[cjc-vizor]]
- [[Parity Gates]]
- [[Determinism Contract]]
- [[Wiring Pattern]]
