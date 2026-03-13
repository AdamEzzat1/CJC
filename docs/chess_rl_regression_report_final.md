# CJC Chess RL Final Regression Report

## Date: 2026-03-12

## Test Results Summary

### Chess RL Test Suites

| Suite | Tests Passed | Failed | Ignored |
|-------|-------------|--------|---------|
| test_chess_rl_hardening | 170 | 0 | 3 |
| test_chess_rl_playability | 103 | 0 | 3 |
| test_chess_rl_advanced | 135 | 0 | 3 |
| **Total Chess RL** | **408** | **0** | **9** |

### New Tests Added This Pass

| File | Tests | Category |
|------|-------|----------|
| test_capture_audit.rs | 12 | Capture correctness for all piece types |
| test_promotion_audit.rs | 9 | Promotion mechanics and edge cases |
| **Total New** | **21** | |

### Ignored Tests (Pre-existing, Justified)

| Test | Reason |
|------|--------|
| debug_fn_body_while_with_semi | CJC syntax investigation (not a bug) |
| debug_fn_while_eval_path | Eval-path known limitation |
| debug_function_no_params_no_return | Parser edge case under investigation |

All ignored tests are pre-existing and documented. No new tests were ignored.

## Changes Made This Pass

### Engine Code
- **No changes** to the CJC chess engine source (`CHESS_ENV`, `RL_AGENT`, `TRAINING` constants)
- Engine is audited and correct as-is

### Platform HTML (examples/chess_rl_platform.html)
- Added `applyMoveWithPromo` for underpromotion support
- Added promotion choice UI modal (queen/rook/bishop/knight)
- Added capture heuristic to `selectAction` (0.5x piece value bonus)
- Added "Why this move?" card replacing simple policy panel
- Added demo presets bar (5 presets) at top of page
- Added deterministic replay badge in header
- Added `replayExact()` function
- Preserved all existing features: engine, RNG, agent, board, traces, review, openings, profile, settings

### Test Files
- Added `tests/chess_rl_hardening/test_capture_audit.rs` (12 tests)
- Added `tests/chess_rl_hardening/test_promotion_audit.rs` (9 tests)
- Updated `tests/chess_rl_hardening/mod.rs` to include new modules

### Documentation
- Created `docs/chess_rl_engine_correctness_audit.md`
- Created `docs/chess_rl_competence_plan.md`
- Created `docs/chess_rl_final_audit.md`
- Created `docs/chess_rl_demo_readiness.md`
- Created `docs/chess_rl_regression_report_final.md` (this file)
- Updated `docs/chess_rl_external_data_ingestion.md`
- Updated `docs/portfolio/chess_rl_playable_platform_summary.md`

### Trace Artifacts
- Created `trace/training_summary.json`
- Created `trace/eval_metrics.json`
- Created `trace/elo_ratings.json`
- Created `trace/player_profiles/default_player_profile.json`

## Determinism Verification

- Same seed produces identical agent behavior (verified by existing determinism tests)
- Replay badge displays correct seed and status
- "Replay Exact" restarts with same parameters
- SplitMix64 RNG remains seed-deterministic across all paths
- No HashMap/HashSet usage in engine (BTreeMap/BTreeSet only in Rust code)
- No time-based seeds anywhere

## Backward Compatibility

- All pre-existing tests continue to pass
- CJC engine source unchanged
- Trace format is backward-compatible (new fields are additive)
- localStorage traces from previous sessions still load correctly
- Review mode works with both old and new trace formats

## Known Limitations

1. No castling (documented simplification)
2. No en passant (documented simplification)
3. Auto-queen in CJC backend (underpromotion only in JS UI)
4. Agent uses random MLP weights + capture heuristic (not trained)
5. PGN import pipeline specified but not yet implemented
6. ELO ratings are estimates, not measured from actual games

## Verdict

**All tests pass. No regressions. Platform is demo-ready.**
