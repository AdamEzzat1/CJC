# Chess RL — Replay Determinism Contract

## What "Replay Stable" Means

When the "Replay Stable" badge is displayed, the following guarantees hold:

1. **Same seed** produces same RNG sequence (SplitMix64 with BigInt for exact u64 parity)
2. **Same initial weights** (fresh, untrained) produce same agent behavior
3. **Same human moves** (if replaying a recorded game) produce same board states
4. **Same opponent type** (agent vs random) is used

## When Replay Is NOT Stable

The badge changes to "Weights Modified" (yellow warning) when:

- The agent has been trained via REINFORCE (weights differ from initialization)
- Trained weights are loaded from localStorage (weights differ from seed-derived initialization)

In these cases, the agent's policy distribution has changed, so replaying with the same seed produces **different agent moves** even though the RNG sequence is identical.

## RNG Management

- Each game creates a fresh `SplitMix64` instance from the user-specified seed
- Weight initialization consumes a deterministic number of RNG calls
- When using trained weights, the RNG still advances past the weight-init calls to maintain RNG parity for move sampling
- Agent move sampling uses `rng.nextF64()` for categorical sampling

## Position Tracking for Draw Detection

- Threefold repetition uses string-based position hashing (board + side + castling + EP)
- This is deterministic: same game state always produces the same hash
- 50-move rule uses a halfmove clock reset on pawn moves and captures

## What Is NOT Guaranteed

- Cross-browser reproducibility (different JS engines may have floating-point differences)
- Exact parity with CJC backend (JS engine has castling + EP, CJC backend does not)
- Replay stability across code versions (internal changes may shift RNG consumption)

## How to Verify Replay

1. Play a game with seed N, opponent "agent", NO prior training (fresh weights)
2. Click "Replay Exact"
3. Make the same human moves
4. Agent should make identical moves (badge shows "Replay Stable")

If you've trained the agent, click "Reset Agent" first to restore fresh weights.
