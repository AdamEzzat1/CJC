# CJC Chess RL — Portfolio Summary

## What This Is

A chess-playing reinforcement learning agent built entirely in **CJC**, a custom deterministic programming language I designed and implemented from scratch. The chess engine, neural network, training algorithm, and evaluation framework all run in CJC with zero external libraries — no TensorFlow, no PyTorch, no Stockfish, no ONNX. Every matrix multiply, every gradient, every random number is mine.

The interactive demo runs in a single HTML file with a JavaScript engine that mirrors the CJC backend exactly, allowing browser-based play, training, and evaluation.

## The Language (CJC)

CJC is a deterministic numerical programming language implemented in Rust across 20 workspace crates (~75K LOC). It has:

- **Two execution backends:** AST tree-walk interpreter and MIR register-machine executor, verified to produce bit-identical output
- **Type system** with inference, closures with capture analysis, pattern matching
- **Tensor runtime** with BLAS-style operations, AD (forward + reverse mode), loss functions, LR schedules
- **Deterministic execution:** SplitMix64 RNG, BTreeMap everywhere (never HashMap), Kahan summation, no FMA
- **MIR optimizer:** Constant folding, dead code elimination, NoGC static verifier
- **2200+ workspace tests**, all passing

The chess RL demo exists to prove CJC handles real numerical workloads — not just Fibonacci.

## The Chess Engine

Written in pure CJC. No libraries. No opening books. No endgame tables.

- Flat 64-int board representation (piece encoding: +1..+6 white, -1..-6 black)
- Square-ordered pseudo-legal move generation with king-safety filtering
- Castling, en passant, pawn promotion (all four pieces)
- Draw detection: threefold repetition, 50-move rule, insufficient material
- Terminal detection: checkmate, stalemate

The JavaScript mirror adds no new chess logic — same encoding, same move generation order, same piece values.

## The Agent

### Architecture (V2 — Actor-Critic with Residual Blocks)

```
Input: 288 features
  Board squares (64) + Attack maps (128) + Piece-square tables (64)
  + Material (6) + Mobility (2) + King pos (4) + Pawn structure (8)
  + Game phase (2) + Side (1) + Move encoding (2)

  -> Linear(288, 128) + GELU                    [shared trunk]
  -> ResBlock(128): Linear->GELU->Linear + skip  [pattern extraction]
  -> ResBlock(128): Linear->GELU->Linear + skip  [deeper features]
  |
  |-> Policy Head: Linear(128,64)->GELU->Linear(64,1)    [move score]
  |-> Value Head:  Linear(128,32)->GELU->Linear(32,1)->tanh  [position value]

Parameters: ~110,000
Activation: GELU (Gaussian Error Linear Unit)
Initialization: He (sqrt(2/fan_in))
```

### Training Algorithm: A2C with GAE

- **Advantage Actor-Critic (A2C):** Policy gradient with learned value baseline
- **Generalized Advantage Estimation (GAE):** lambda=0.95, gamma=0.99
- **Entropy bonus:** 0.01 coefficient for exploration
- **Gradient clipping:** Global L2 norm bounded to 0.5
- **Batch training:** 8 trajectories per update from experience replay buffer (100 games)
- **Learning rate:** 0.0003 with slow decay (0.0003 / (1 + episode * 0.002))

### Curriculum Learning

| Phase | Episodes | Opponent | Purpose |
|---|---|---|---|
| A | 0-50 | Random | Build initial value function from easy wins |
| B | 50-200 | Heuristic | Learn to beat positional play |
| C | 200+ | Self-play | Improve against itself |

### Evaluation Framework

Four baseline agents for comparison:
1. **Random:** Uniform random legal moves
2. **Heuristic:** Greedy piece-value + center control + development + 1-ply lookahead
3. **1-ply Minimax:** Maximize worst-case material balance
4. **Untrained network:** Same architecture, random weights

Evaluation protocol: Every 10 training episodes, play 20 games against each baseline with fixed seeds. Results displayed as win rate charts in the Training dashboard.

### Tactical Puzzle Suite

5 predefined positions with known best moves (back-rank mate, knight fork, free capture, promotion, hanging piece). Agent is tested with greedy selection (temperature=0). Puzzle score tracks improvement over training.

## What the Demo Shows

The interactive platform (`chess_rl_platform.html`) includes:

### Play Tab
- Interactive board with legal move highlighting
- "Why this move?" card showing agent confidence, top-5 candidates, and policy distribution
- Move history, material display, draw/resign controls, undo

### Training Tab
- **Train N Episodes** button with async loop (UI stays responsive)
- **Learning curves:** Reward, policy loss, value loss, entropy, gradient norm — all over episodes
- **Win rate charts:** vs Random and vs Heuristic, evaluated periodically
- **Curriculum phase indicator** with auto-progression
- **Replay buffer** and batch training status
- **Model checkpoints** saved at intervals

### Evaluation Tab
- One-click evaluation against all baselines
- Win/Draw/Loss breakdown per opponent
- Tactical puzzle score

### Review Tab
- Post-game board replay with per-ply policy visualization
- Agent decision analysis at each move

### Opening Explorer
- Hierarchical tree of opening moves across all played games
- Win/draw/loss statistics per opening line

### Style Profile
- Player statistics: win rate, capture rate, favorite piece, piece usage distribution

## Technical Decisions

**Why A2C over PPO?** Simpler (no clipping ratio), sufficient for 110K parameters, easier to explain. PPO is a future upgrade path.

**Why manual gradients, not autodiff?** CJC has reverse-mode AD (GradGraph), but the JS platform uses manual backprop to demonstrate understanding of the math. Every gradient is derived by hand — no `.backward()` magic.

**Why a single HTML file?** Zero dependencies, zero build tools. Open the file, play chess, train the agent. The entire platform — engine, network, training, UI, charts — is self-contained.

**Why deterministic execution?** Same seed = bit-identical game. This isn't just an academic property — it enables reproducible debugging, replay verification, and honest training metrics.

## What It Can't Do

- **It won't beat Stockfish.** A 110K parameter network with REINFORCE-family training can't compete with 100M+ parameter engines using MCTS. That's not the point.
- **It won't learn opening theory.** The network has no memory of past games during play — it only sees the current board. Opening preferences emerge from weight updates, not memorization.
- **Training curves take dozens of games to show improvement.** Policy gradient methods have high variance; visible learning requires 50+ episodes against random opponents.

## Test Coverage

| Suite | Tests | Status |
|---|---|---|
| Chess RL Project | 49 | PASS |
| Chess RL Playability | ~128 | PASS |
| Chess RL Hardening | ~170 | PASS |
| Chess RL Advanced (incl. ML upgrade) | ~153 | PASS |
| Full Workspace | 2200+ | PASS |

15 new tests specifically for the ML upgrade:
- Network determinism, finite outputs, weight initialization variance
- Training determinism, reward bounds, loss finiteness
- Game engine regression (encoding, legal moves, termination, action validity)
- Fuzz tests across 10+ random seeds
- MIR execution parity

## Zero External Dependencies

The CJC compiler: Rust, 20 crates, zero `extern crate` beyond std.
The chess engine: Pure CJC source, zero imports.
The demo platform: Single HTML file, zero npm/CDN/library dependencies.

Every line of code — from the lexer to the learning rate schedule — is written from scratch.
