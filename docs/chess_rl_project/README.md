# Chess RL Benchmark for CJC

A complete chess reinforcement learning benchmark implemented entirely in the CJC programming language, demonstrating CJC's viability for scientific computing and RL workloads.

## Overview

This benchmark implements:
- A simplified chess environment (board representation, move generation, legality checking, terminal detection)
- A REINFORCE policy gradient agent with a per-move scoring MLP
- Self-play training with manual backpropagation
- Evaluation against a uniform random opponent

All game logic and neural network code is written in CJC. Only 3 general-purpose builtins were added to the runtime (`log`, `exp`, `categorical_sample`).

## Project Structure

```
tests/chess_rl_project/
  cjc_source.rs          -- All CJC source: CHESS_ENV, RL_AGENT, TRAINING constants
  mod.rs                 -- Module declarations
  test_debug.rs          -- Parser/syntax diagnostic tests
  test_01_board_invariants.rs   -- Board representation (11 tests)
  test_02_movegen_determinism.rs -- Move generation (7 tests)
  test_03_legal_move_sanity.rs   -- Legal move correctness (10 tests)
  test_04_rollout_determinism.rs -- Rollout determinism (5 tests)
  test_05_training_smoke.rs      -- Training basics (9 tests)
  test_06_training_determinism.rs -- Training determinism (4 tests)
  test_07_eval_vs_random.rs      -- Evaluation pipeline (5 tests)

docs/chess_rl_project/
  README.md              -- This file
  DESIGN.md              -- Architecture and design decisions
  DETERMINISM.md         -- Determinism proof and methodology
  RESULTS.md             -- Final results and CJC viability assessment
```

## Running

```bash
# Run all chess RL tests
cargo test --test test_chess_rl_project

# Run specific test suite
cargo test --test test_chess_rl_project test_01

# Run with output
cargo test --test test_chess_rl_project -- --nocapture
```

## Simplifications

The chess implementation makes documented simplifications:
- No castling
- No en passant
- Pawns auto-promote to queen
- 200-halfmove draw limit (no 50-move rule)
- Per-move scoring MLP (not a full policy over all positions)

These are intentional: the goal is to stress-test CJC's computational capabilities, not to build a competitive chess engine.

## New Runtime Primitives

| Builtin | Signature | Purpose |
|---------|-----------|---------|
| `log` | `f64 -> f64` | Natural logarithm (for log-probability computation) |
| `exp` | `f64 -> f64` | Exponential function (for softmax numerics) |
| `categorical_sample` | `Tensor -> i64` | Sample from categorical distribution (seeded RNG) |

Additionally, `.transpose()` and Tensor-scalar arithmetic were wired into the MIR-exec path (they already existed in the AST-eval path).

## Test Count

- **49 chess RL tests**: All passing
- **2186 total workspace tests**: 0 failures, 16 ignored (13 pre-existing + 3 diagnostic)
