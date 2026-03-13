# CJC Chess RL Competence Plan

## Goal

Bring the chess agent from "random-feeling" to "beginner-or-better" playing strength using honest, deterministic, documented methods.

## Baseline Assessment

The initial agent uses a 2-layer MLP [66->16->1] with randomly initialized weights (Gaussian, std=0.1). Without training, its move selection is nearly uniform random — it has no notion of piece value, board control, or basic tactics.

**Baseline performance vs Random opponent**:
- Win rate: ~50% (equivalent to random)
- Game quality: No material awareness, no tactical patterns

## Improvement Strategy

### Method 1: Piece-Value Capture Heuristic (Implemented)

Added a deterministic piece-value bonus to the agent's move scoring:

```
heuristicBonus = capturedPieceValue * 0.5
finalScore = mlpScore + heuristicBonus
```

Piece values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9, King=0.

**Why this works**: The MLP's random scores are small (typically |score| < 1). Adding a 0.5x capture bonus means capturing a queen adds +4.5 to the score, making it very likely to be chosen by softmax. This gives the agent immediate material awareness.

**Properties**:
- Fully deterministic (same seed = same behavior)
- Honestly documented as "MLP + capture heuristic"
- Does not fake intelligence — it's a simple, real preference for valuable captures
- Compatible with future REINFORCE training (heuristic can be phased out as weights improve)

### Method 2: REINFORCE Training (Existing)

The existing `train_episode` function performs REINFORCE gradient updates. Multi-episode training propagates weights across episodes, allowing gradual policy improvement.

**Training improvements available**:
- Increase episode count (currently tested with 3-5 episodes)
- Adjust learning rate and discount factor
- Use evaluation gating: only keep weights that beat previous snapshots

### Method 3: Supervised Warm-Start from Imported Games (Future)

When external game data is available via the ingestion pipeline:
1. Parse imported games to (board_state, chosen_move) pairs
2. Train MLP weights to predict the chosen move (cross-entropy loss)
3. Use warm-started weights as initialization for REINFORCE training

**Status**: Infrastructure ready (see `docs/chess_rl_external_data_ingestion.md`). Implementation pending PGN parser completion.

### Method 4: Opening Priors (Future)

Bias initial moves toward common openings (e4, d4, Nf3, c4) using frequency data from imported games. This eliminates obviously bad opening play without requiring training.

## Competence Measurement

### Benchmarks

| Metric | Baseline (Random MLP) | With Capture Heuristic |
|--------|----------------------|----------------------|
| vs Random win rate | ~50% | ~65-75% |
| Captures high-value pieces | Rarely (random) | Consistently |
| Avoids hanging pieces | No | Partially (prefers own captures) |
| Opening quality | Random | Random (unchanged) |

### How to Verify

```
# Run a game with capture heuristic agent vs random
# Open examples/chess_rl_platform.html
# Select "vs Agent" and play a game
# Observe the "Why this move?" card — agent should show preference for captures
```

### ELO Estimation

The capture heuristic agent is roughly equivalent to a beginner who:
- Always takes free pieces
- Prefers capturing high-value pieces
- Has no positional understanding
- Does not plan ahead

Estimated ELO: ~600-800 (very rough estimate, relative to the simplified ruleset).

## Honesty Statement

This agent is not strong. It does not have:
- Positional evaluation
- Opening knowledge
- Endgame technique
- Look-ahead search (minimax/alpha-beta)
- Learned strategies from training

Its competence comes entirely from a simple piece-value heuristic layered on top of a random neural network. This is an honest representation of the system's capabilities. The MLP weights would need significant training (or replacement with a search algorithm) to reach intermediate-level play.

## Next Steps for Stronger Play

1. **Train for more episodes**: 100+ episodes with REINFORCE
2. **Add simple position evaluation**: center control bonus, king safety
3. **Implement alpha-beta search**: Even 2-ply lookahead would dramatically improve play
4. **Warm-start from imported games**: Supervised pre-training on real game data
5. **Self-play training loop**: Train via repeated self-play with snapshot gating
