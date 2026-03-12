# Chess RL Fuzzing

## Overview

Bolero-based fuzz tests for the chess RL subsystem. On Windows, these run as proptest during `cargo test`. On Linux, use `cargo bolero test <target>` for coverage-guided fuzzing.

## Fuzz Targets (`fuzz/test_chess_fuzz.rs`, 5 targets)

| Target | Input | Invariant |
|--------|-------|-----------|
| fuzz_legal_moves_no_panic | u64 seed | Board init + legal_moves never panics |
| fuzz_rollout_no_panic | u64 seed | Full game rollout never panics |
| fuzz_select_action_no_panic | u64 seed | Action selection never panics |
| fuzz_train_episode_no_panic | u64 seed | Training episode never panics |
| fuzz_eval_random_no_panic | u64 seed | Eval vs random never panics |

## Design Decisions

- **Seed-based fuzzing**: All fuzz inputs are `u64` seeds rather than raw byte arrays. This is because the chess RL system is a CJC program, and the only meaningful fuzz dimension is the RNG seed which controls weight initialization and categorical sampling.
- **catch_unwind**: All targets use `catch_unwind` to catch panics without aborting the test process.
- **Short episodes**: Max moves capped at 8-10 to keep fuzz throughput high.

## Windows Instructions

```bash
# Run as proptest (automatic on Windows)
cargo test --test test_chess_rl_hardening fuzz

# For coverage-guided fuzzing on Linux:
cargo bolero test fuzz_legal_moves_no_panic
cargo bolero test fuzz_rollout_no_panic
```
