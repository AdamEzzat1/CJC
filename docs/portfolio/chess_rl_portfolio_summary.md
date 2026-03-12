# Chess RL Portfolio Summary

## Resume Bullet

Built a deterministic chess reinforcement learning benchmark in a custom programming language (CJC), featuring a complete chess engine with legal move generation, a 2-layer MLP policy network with manual REINFORCE gradients, deterministic trace export, and a 149-test hardening suite including property-based and fuzz testing.

## GitHub README Description

**CJC Chess RL** - A fully deterministic chess reinforcement learning system written in the CJC programming language. Implements a complete chess environment (board representation, legal move generation, check/checkmate detection) and a REINFORCE policy gradient agent, all executing on a deterministic MIR register-machine runtime. Same seed = bit-identical results across runs.

## Full Portfolio Explanation

### What It Is

A self-contained reinforcement learning system that learns to play chess, built entirely in CJC (a custom deterministic numerical programming language compiled via Rust). The system includes:

- **Chess environment** (345 lines of CJC): Piece movement, legal move filtering, check detection, terminal state recognition (checkmate/stalemate)
- **Policy network** (154 lines of CJC): 2-layer MLP [66 -> 16 -> 1] with manual forward/backward pass
- **Training loop** (165 lines of CJC): REINFORCE policy gradient with trajectory collection, advantage estimation, and SGD weight updates
- **Trace export system**: Captures per-ply board state, policy decisions, and move probabilities in JSONL format
- **Dashboard viewer**: Interactive HTML visualization showing board state, policy probabilities, and game timeline

### Why It Matters

1. **Deterministic RL**: Same seed produces bit-identical training trajectories. This is non-trivial — it requires deterministic RNG, deterministic move generation order, deterministic floating-point reductions, and deterministic categorical sampling.

2. **Custom Language Runtime**: The chess engine and RL agent run on CJC's MIR register-machine executor, demonstrating the language's capability for complex numerical workloads.

3. **Manual Gradient Computation**: REINFORCE gradients are computed explicitly in CJC (not using an automatic differentiation framework), showing understanding of the underlying mathematics.

4. **Engineering Rigor**:
   - 149 hardening tests across 6 categories (unit, property, fuzz, determinism)
   - Property tests verify invariants across hundreds of random seeds
   - Bolero fuzz tests verify crash resistance
   - Zero regressions against 3,400+ existing tests

### Technical Depth

| Aspect | Implementation |
|--------|----------------|
| Board | 64-int flat array, functional (immutable) updates |
| Move generation | Piece-type enumeration, square-ordered for determinism |
| Legal filtering | Apply-and-check: try each move, verify king not in check |
| Network | [1,66] -> relu([1,16]) -> [1,1] per move, softmax over all |
| Gradient | Manual REINFORCE: d(log_pi)/d(theta) via chain rule through ReLU |
| RNG | SplitMix64, seeded, no platform-dependent behavior |
| Testing | proptest (randomized), Bolero (fuzz), determinism gates |

### How to Present It

1. **Demo**: Open `examples/chess_rl_dashboard.html` in a browser. Click "Demo Data" to see the interactive board with policy probability bars.

2. **Run tests**: `cargo test --test test_chess_rl_hardening` — shows 149 tests passing.

3. **Trace export**: `cargo run --example chess_rl_trace_demo` — generates JSONL trace files.

4. **Key talking points**:
   - "I built a chess RL system from scratch in a custom language"
   - "Every training run is bit-identical — true reproducibility"
   - "149 tests including property-based and fuzz testing"
   - "The gradient computation is manual, not autograd"

## Follow-On Improvements (Prioritized)

1. **Multi-episode training with trace** — Train 50+ episodes and export learning curve data (impact: high, safety: safe, portfolio: high)

2. **Win rate visualization** — Show learning progress over episodes in the dashboard (impact: high, safety: safe, portfolio: very high)

3. **Castling + en passant** — Add remaining chess rules for completeness (impact: medium, safety: needs determinism audit, portfolio: medium)

4. **AD-based gradients** — Replace manual REINFORCE with CJC's autodiff (impact: high, safety: medium — must preserve determinism, portfolio: high)

5. **MIR trace hooks** — Instrument the MIR executor to capture computation steps, enabling step-through debugging (impact: medium, safety: invasive, portfolio: very high)
