# CJC Chess RL Final Regression Report (v2 — Post-Professionalization)

## Date: 2026-03-12

## Test Results Summary

### Chess RL Test Suites

| Suite | Tests Passed | Failed | Ignored |
|-------|-------------|--------|---------|
| test_chess_rl_playability | 128 | 0 | 3 |
| test_chess_rl_hardening | 170 | 0 | 3 |
| test_chess_rl_advanced | 135 | 0 | 3 |
| test_chess_rl_project | 66 | 0 | 3 |
| **Total Chess RL** | **499** | **0** | **12** |

### Ignored Tests (Pre-existing, Justified)

| Test | Reason |
|------|--------|
| debug_fn_body_while_with_semi | CJC syntax investigation (not a bug) |
| debug_fn_while_eval_path | Eval-path known limitation |
| debug_function_no_params_no_return | Parser edge case under investigation |

All ignored tests are pre-existing and documented. No new tests were ignored.

## Changes Made This Pass (Professionalization)

### Engine Code (CJC)
- **No changes** to CJC engine source, compiler crates, or Rust test infrastructure

### Platform HTML (examples/chess_rl_platform.html)

#### Draw Rules (Phase 4)
- Added threefold repetition detection (position hashing)
- Added fifty-move rule (halfmove clock, triggers at 100 half-moves)
- Added insufficient material detection (K vs K, K+N vs K, K+B vs K, same-color K+B vs K+B)
- Draw offer with agent decision-making (declines if winning by >2 material)

#### RL Stabilization (Phase 5)
- REINFORCE baseline subtraction (EMA, alpha=0.1)
- Learning rate schedule (0.01 / (1 + episode * 0.05))
- Weight persistence to localStorage (1MB bound)
- Weight loading on page load
- 1-ply tactical lookahead in evaluateMove() (hanging piece detection, free capture bonus)

#### Replay Determinism (Phase 6)
- Honest "Replay Stable" badge (yellow "Weights Modified" warning when trained)
- Clean RNG state management

#### UX Polish (Phases 8-9)
- Board: 48px → 56px squares, 30px → 34px font
- Panel spacing improved
- Undo/takeback button added

#### Explorer & Profile (Phase 10)
- Opening Explorer: flat table → expandable hierarchical tree (6 plies deep)
- Style Profile: threshold 2 games → 1 game
- localStorage bounded with auto-eviction

### Documentation Created/Updated
- `docs/chess_rl_change_control.md` — New: change-control audit
- `docs/chess_rl_replay_contract.md` — New: replay determinism contract
- `docs/chess_rl_final_audit.md` — Updated to v2
- `docs/chess_rl_demo_readiness.md` — Updated to v2
- `docs/chess_rl_regression_report_final.md` — Updated (this file)
- `docs/portfolio/chess_rl_playable_platform_summary.md` — Updated counts/features

## Determinism Verification

- Same seed produces identical agent behavior (with fresh weights)
- Honest badge warns when training has modified weights
- SplitMix64 RNG remains seed-deterministic
- Position hashing for draw detection is deterministic (string-based, no randomness)
- No HashMap/HashSet usage in engine
- No time-based seeds anywhere

## Backward Compatibility

- All pre-existing Rust tests pass without modification
- CJC engine source unchanged
- Trace format backward-compatible (new fields additive)
- localStorage traces from previous sessions still load
- New weight persistence field (`cjc_chess_weights`) is separate from traces

## Verdict

**All 499 tests pass. Zero regressions. Zero CJC changes. Platform is demo-ready (A-).**
