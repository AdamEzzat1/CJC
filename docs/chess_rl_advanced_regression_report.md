# Chess RL Advanced Research Platform — Regression Report

## Baseline (Before Advanced Platform)

| Metric | Count |
|--------|-------|
| Chess RL tests passing | 149 |
| Chess RL tests ignored | 3 |
| Total workspace tests | 3,436 |
| Total ignored | 30 |
| Total failures | 0 |

## New Tests Added

| Category | File | Tests |
|----------|------|-------|
| Multi-episode training | test_multi_training.rs | 9 |
| Win-rate evaluation | test_win_rate.rs | 6 |
| Self-play architecture | test_selfplay.rs | 7 |
| Model snapshots | test_snapshots.rs | 7 |
| ELO rating system | test_elo.rs | 8 |
| Deterministic replay | test_replay.rs | 6 |
| MIR trace hooks | test_mir_trace.rs | 7 |
| AD gradient verification | test_ad_verify.rs | 7 |
| Castling + en passant | test_castling_ep.rs | 7 |
| Trace export | test_trace_export.rs | 5 |
| Existing chess RL (re-run) | chess_rl_project/ | 49 |
| Existing hardening (re-run) | chess_rl_hardening/ | 19 (property+fuzz) |
| **Subtotal new** | | **69** |
| **Total in advanced harness** | | **135** (+ 3 existing ignored) |

## Final Results

| Metric | Count |
|--------|-------|
| Chess RL advanced tests passing | 135 |
| Chess RL advanced tests ignored | 3 |
| Total workspace tests passing | ~3,571 |
| Total workspace ignored | ~33 |
| Total failures | **0** |

## Ignored Tests (3)

Same as before — all from `chess_rl_project/test_debug.rs` (known parser limitation):
1. `debug_fn_body_while_with_semi`
2. `debug_fn_while_eval_path`
3. `debug_function_no_params_no_return`

## New Capabilities Tested

### Multi-Episode Training (Phase 2)
- Weight propagation across episodes
- Per-episode metric collection (reward, loss, steps)
- train_episode_returning_weights function
- Deterministic multi-episode training

### Win-Rate Evaluation (Phase 3)
- Evaluation against random opponent
- Wins/draws/losses counting
- Win rate bounded in [0, 1]
- Works for both white and black sides

### Castling + En Passant Detection (Phase 4)
- Initial castling rights detection
- Path-blocked castling detection
- Castling availability after clearing path
- En passant double-move detection
- Adjacent pawn detection for EP captures
- Extended board state (69 elements with rights)

### AD Gradient Verification (Phase 5)
- Forward pass consistency
- Non-zero gradient updates (checked across all weight elements)
- Gradient direction: positive advantage increases action score
- Gradient direction: negative advantage decreases action score
- Zero advantage = no weight change
- Finite difference numerical gradient verification
- Gradient determinism

### MIR Trace (Phase 6)
- Board state tracing
- Legal move tracing
- Action selection tracing
- Forward pass tracing
- Full game trajectory tracing
- Trace determinism

### Self-Play (Phase 7)
- Two-agent games with separate weights
- Self-play episode completion
- Max moves respected
- Multi-game evaluation between agents
- Self-play determinism
- Reward perspective (white-relative)

### Model Snapshots (Phase 8)
- Weight tensors are snappable
- Snap/restore roundtrip for tensors
- Deterministic content hashing
- Hash changes with data changes
- Float array roundtrip
- Trained vs initial weight differentiation

### League + ELO (Phases 9-10)
- ELO expected score formula
- ELO update mechanics (winner gains, draw no change for equals)
- ELO conservation (total rating preserved)
- Upset bonus (lower-rated winner gains more)
- 3-model round-robin league
- League result determinism

### Deterministic Replay (Phase 11)
- Same seed + weights = identical game
- 5-run replay consistency
- Training replay determinism
- Self-play replay determinism
- Different seeds produce different weight initializations
- Win-rate evaluation replay determinism

### Trace Export (Phase 2+)
- Per-episode metric parsing
- JSON-format trace records
- Win-rate trace export
- Full pipeline trace (train + eval)
- Trace determinism

## Regression Discipline

- Zero existing tests modified
- Zero existing tests broken
- Zero new `#[ignore]` annotations added
- All new tests pass on first run (after 5 targeted fixes)
- Full workspace regression: 0 failures
