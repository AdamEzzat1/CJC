# Chess RL Playable Platform — Regression Report

## Baseline (Before Playable Platform)

| Metric | Count |
|--------|-------|
| Chess RL advanced tests passing | 135 |
| Chess RL advanced tests ignored | 3 |
| Total workspace tests | ~3,571 |
| Total ignored | ~33 |
| Total failures | 0 |

## New Tests Added

| Category | File | Tests |
|----------|------|-------|
| Human move validation | test_human_move.rs | 11 |
| Trace replay determinism | test_replay_trace.rs | 6 |
| Player profile generation | test_player_profile.rs | 4 |
| PGN import / normalization | test_pgn_import.rs | 5 |
| Opening tree construction | test_opening_tree.rs | 5 |
| Style-conditioned adaptation | test_adaptation.rs | 6 |
| **Subtotal new** | | **37** |
| **Total in playability harness** | | **103** (incl. existing chess RL + 3 ignored) |

## Final Results

| Metric | Count |
|--------|-------|
| Playability harness tests passing | 103 |
| Playability harness tests ignored | 3 |
| Total workspace tests passing | 3,674 |
| Total workspace ignored | 36 |
| Total failures | **0** |

## Ignored Tests (3)

Same as before — all from `chess_rl_project/test_debug.rs` (known parser limitation):
1. `debug_fn_body_while_with_semi`
2. `debug_fn_while_eval_path`
3. `debug_function_no_params_no_return`

## New Capabilities Tested

### Human Move Validation
- Initial position legal move count (20 for white, 20 for black)
- e2-e4 legality verification
- e2-e5 illegality verification
- Board state update correctness (functional, immutable)
- Move count changes after opening moves
- Terminal status detection
- Check detection
- King finding
- Pawn promotion to queen
- Move generation determinism (seed-independent)

### Trace Replay Determinism
- Move sequence replay produces identical board
- Agent game replay deterministic
- Board encoding deterministic (seed-independent)
- Encoding normalization correctness (piece * side / 6.0)
- Incremental apply_move consistency
- Board size always 64

### Player Profile Generation
- Agent produces valid game results (reward, moves)
- Piece count tracking after game
- Profile metrics deterministic
- Different seeds produce different trajectories

### PGN Import / Board Normalization
- Board reconstruction from move index sequences
- Encoding tensor shape [1, 64]
- Encoding side symmetry (white = -black)
- All generated moves verified as legal (no check after move)
- Coordinate roundtrip: sq_of(rank_of(sq), file_of(sq)) == sq

### Opening Tree Construction
- First move set stable across seeds
- Black response set stable
- Opening sequence replay (Italian Game)
- Move enumeration order (sq 0..63)
- Capture replaces piece correctly

### Style-Conditioned Adaptation
- Untrained agent selects valid move index
- Different weight initializations produce different move choices
- Forward pass produces finite scores
- Action probabilities sum to 1.0
- Training changes weights (non-trivial training)
- Random vs agent produces different trajectory

## Interactive Platform Capabilities

The `examples/chess_rl_platform.html` file provides:

| Feature | Implementation |
|---------|---------------|
| JS chess engine | Exact mirror of CJC engine (same piece encoding, move gen order) |
| SplitMix64 RNG | BigInt-based port for u64 parity |
| MLP agent | [66→16→1] with softmax + categorical sampling |
| vs Agent mode | Trained weights with policy visualization |
| vs Random mode | Uniform random move selection |
| Trace recording | JSONL format compatible with existing replay |
| Post-game review | Ply-by-ply navigation with auto-play |
| Policy explainability | Top-5 candidates, confidence, move count |
| Opening explorer | Move frequency from stored traces |
| Material summary | Piece tracking with advantage display |
| Style profile | Games, win rate, capture rate, piece usage |
| Demo presets | Quick, Standard, Endurance, Seed Battle |
| Session export | JSON summary of all games |

## Regression Discipline

- Zero existing tests modified
- Zero existing tests broken
- Zero new `#[ignore]` annotations added
- All 37 new tests pass on first run
- Full workspace regression: 0 failures
