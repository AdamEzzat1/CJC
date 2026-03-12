# Chess RL Regression Report

## Baseline (Before Hardening)

| Metric | Count |
|--------|-------|
| Chess RL tests passing | 66 |
| Chess RL tests ignored | 3 |
| Total workspace tests | 3,287 |
| Total ignored | 27 |
| Total failures | 0 |

## New Tests Added

| Category | File | Tests |
|----------|------|-------|
| Board hardening | test_board_hardening.rs | 21 |
| Movegen hardening | test_movegen_hardening.rs | 10 |
| Game logic hardening | test_game_logic_hardening.rs | 10 |
| Agent hardening | test_agent_hardening.rs | 11 |
| Training hardening | test_training_hardening.rs | 6 |
| Determinism hardening | test_determinism_hardening.rs | 12 |
| Property: board | property/test_board_props.rs | 4 |
| Property: movegen | property/test_movegen_props.rs | 3 |
| Property: agent | property/test_agent_props.rs | 3 |
| Property: determinism | property/test_determinism_props.rs | 3 |
| Fuzz: chess | fuzz/test_chess_fuzz.rs | 5 |
| **Subtotal new** | | **88** |
| Existing chess RL (re-run) | chess_rl_project/ | 66 |
| **Total in harness** | | **149** (+ 5 existing filtered) |

## Final Results

| Metric | Count |
|--------|-------|
| Chess RL hardening tests passing | 149 |
| Chess RL hardening tests ignored | 3 |
| Total workspace tests passing | 3,436 |
| Total workspace ignored | 30 |
| Total failures | **0** |

## Ignored Tests (3)

All from `chess_rl_project/test_debug.rs` — known parser limitation:
1. `debug_fn_body_while_with_semi` — `};` after while in fn body
2. `debug_fn_while_eval_path` — same limitation
3. `debug_function_no_params_no_return` — same limitation

These are pre-existing and unrelated to the hardening work.

## Determinism Verification

- 12 dedicated determinism tests run each program 2-5 times with same seed
- All produce bit-identical output
- Property tests verify determinism across 30-50 random seeds
- No non-deterministic behavior detected

## Regression Discipline

- Zero existing tests modified
- Zero existing tests broken
- Zero new `#[ignore]` annotations added
- All new tests pass on first run
