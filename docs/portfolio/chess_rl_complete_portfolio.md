# CJC Chess RL Platform — Complete Portfolio Document

## One-Liner

A fully playable chess reinforcement learning platform — engine, agent, training, and interactive dashboard — built from scratch in CJC, a custom deterministic programming language, with zero external libraries.

---

## What Is This?

A chess-playing AI that learns from every game it plays against you, built entirely in a custom programming language I designed called CJC. The system spans three layers:

1. **CJC Chess Engine** — A complete chess environment written in pure CJC (my custom language), executing on CJC's MIR register-machine runtime. Handles board representation, legal move generation, check/checkmate detection, and terminal state recognition.

2. **REINFORCE Policy Agent** — A neural network (2-layer MLP) that evaluates chess positions and selects moves, trained via REINFORCE policy gradients with manual analytical backpropagation. No autograd framework — every gradient is computed by hand through the chain rule.

3. **Interactive Research Platform** — A browser-based dashboard where you play against the agent, watch it explain its reasoning in real-time, train it through gameplay, and analyze its learning with research-grade tooling.

---

## What Language Is the Engine Written In?

**CJC** — a deterministic numerical programming language I designed and implemented in Rust.

### Why I Built a Custom Language

CJC exists to solve a specific problem: **reproducible computation**. In mainstream languages and ML frameworks, the same code with the same inputs can produce different results across runs due to nondeterministic thread scheduling, floating-point reduction ordering, hash-map iteration order, and platform-dependent RNG behavior. CJC eliminates all of these:

- **Deterministic execution**: Same seed = bit-identical output, every time, on every platform
- **No hidden nondeterminism**: BTreeMap/BTreeSet everywhere (never HashMap/HashSet), explicit seed threading, no FMA instructions, Kahan/Binned accumulators for floating-point reductions
- **Custom RNG**: SplitMix64 with BigInt precision for exact u64 parity across Rust and JavaScript
- **Two execution backends**: AST tree-walk interpreter (v1) and MIR register-machine executor (v2), both producing identical results for the same program

CJC is implemented as a Rust workspace with 20 crates spanning lexer, parser, AST, type system, HIR/MIR lowering, two executors, automatic differentiation, a data DSL, and more — roughly 40,000 lines of Rust.

### Why Use CJC for Chess RL?

The chess RL benchmark is the stress test that proves CJC works for real numerical workloads:

- It exercises tensors, matrix multiplication, softmax, ReLU, categorical sampling
- It requires correct gradient computation through multiple layers
- It demands determinism across hundreds of training episodes
- It demonstrates that a custom language can handle a non-trivial ML pipeline end-to-end

If CJC can train a chess agent with reproducible results, it can handle the scientific computing workloads it was designed for.

---

## Does the Engine Use Any Libraries or Tools?

**No.** The CJC chess engine uses zero external libraries, zero frameworks, and zero third-party dependencies.

### What's in the engine (all CJC):

| Component | Lines of CJC | External Dependencies |
|-----------|-------------|----------------------|
| Chess environment (board, moves, legality, terminal states) | ~345 | None |
| Policy network (2-layer MLP, forward pass) | ~154 | None |
| Training loop (REINFORCE, backprop, SGD) | ~165 | None |
| Evaluation pipeline (vs random opponent) | ~50 | None |
| Trace export (JSONL) | ~45 | None |

### Runtime primitives (3 total):

Only three general-purpose mathematical functions were added to the CJC runtime to support this project:

1. **`log`** — Natural logarithm (for log-probability computation)
2. **`exp`** — Exponential function (for softmax numerics)
3. **`categorical_sample`** — Sample from a categorical distribution using seeded RNG

These are standard math primitives, not chess-specific or ML-specific. All tensor operations (`matmul`, `transpose`, `softmax`, `relu`), array operations, and control flow were pre-existing CJC features.

### The JavaScript platform layer:

The interactive browser platform (`examples/chess_rl_platform.html`) contains a JavaScript mirror of the CJC engine. This JS engine is a faithful port — same piece encoding, same board representation, same move generation order — extended with castling, en passant, and advanced draw rules for a complete playing experience. It uses no frameworks, no npm packages, no build tools. It's a single self-contained HTML file.

---

## What Can the Demo Do?

### Play Chess Against an AI Agent
- Click pieces to select them, click highlighted squares to move
- Agent responds with real policy network reasoning
- Full chess rules: castling, en passant, promotion (including underpromotion UI)
- Legal move highlighting with visual feedback

### Watch the Agent Think
- **"Why this move?" card** shows the agent's top-5 candidate moves with actual probability scores
- Policy data is real — computed from the MLP forward pass, not fabricated
- You can see which moves the agent considered and why it chose what it did

### Train the Agent Through Play
- After every game (win, loss, or draw), REINFORCE training updates the agent's neural network weights
- **Baseline subtraction** (exponential moving average) reduces gradient variance
- **Learning rate schedule** decays over episodes for stable convergence
- **Weight persistence** — trained weights survive page refresh via localStorage
- The agent genuinely adapts to your play style over time

### Verify Determinism
- Seed displayed prominently in the header
- **"Replay Exact"** — same seed with fresh weights produces identical agent behavior
- **Honest badge** — shows "Weights Modified" warning when training has changed weights, so you know replay won't match
- SplitMix64 RNG with BigInt for exact cross-platform parity

### Analyze Games
- **Post-game review** — step through any completed game ply by ply
- **Opening Explorer** — expandable hierarchical tree showing your opening tendencies (6 plies deep)
- **Style Profile** — statistics on your play patterns, available after just 1 game
- **JSONL trace export** — download full game traces for external analysis

### Complete Draw Detection
- **Threefold repetition** — position hashing detects repeated positions
- **Fifty-move rule** — halfmove clock triggers at 100 half-moves
- **Insufficient material** — auto-draws K vs K, K+N vs K, K+B vs K, same-color K+B vs K+B
- **Draw offers** — agent evaluates material before accepting (declines if winning by >2 points)

### Undo Mistakes
- **Undo/takeback** button lets you take back your last move pair during gameplay
- Game state is fully reconstructed from move history

---

## What Can't the Demo Do?

### Engine Limitations (CJC Backend)
- **No castling or en passant** in the CJC backend — these exist only in the JS platform layer
- **Auto-promotion to queen** in CJC backend (JS layer supports underpromotion)
- **200-halfmove draw limit** in CJC backend (JS layer has proper 50-move rule)

### Agent Limitations
- **Beginner-to-intermediate strength** — the MLP is intentionally small (66 inputs → 16 hidden → 1 output) to keep test execution fast
- **1-ply tactical lookahead only** — the agent detects hanging pieces and free captures one move ahead, but has no deep search
- **Slow learning** — REINFORCE with a tiny network and sparse reward (one signal per game) requires many games to show dramatic improvement
- **No opening book** — the agent has no pre-programmed openings; all opening preferences emerge from training
- **No endgame tablebase** — the agent has no perfect endgame play

### Platform Limitations
- **No drag-and-drop** — pieces are moved by click (select piece, then click destination)
- **No move sounds** — visual feedback only
- **No clock/timer** — untimed games only
- **No online multiplayer** — single-player against the agent or random opponent
- **No cross-browser determinism guarantee** — different JS engines may have floating-point differences
- **Single HTML file** — not a production web application with routing, state management, etc.

### Data Limitations
- **PGN import pipeline exists but is not yet wired to the browser** — the Rust-side PGN parser can parse master games into traces, but there's no UI to feed them into the agent's training
- **No supervised pre-training** — the agent starts from random weights every time (unless previously trained weights are in localStorage)
- **localStorage bounded at 1MB for weights** — sufficient for the current MLP (~17KB) but limits future network scaling

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| Chess RL Playability | 128 | All pass |
| Chess RL Hardening | 170 | All pass |
| Chess RL Advanced | 135 | All pass |
| Chess RL Project | 66 | All pass |
| **Total Chess RL** | **499** | **0 failures** |
| **CJC Workspace Total** | **2186+** | **0 failures, 16 ignored** |

Test categories include: board invariants, move generation determinism, legal move correctness, rollout determinism, training smoke tests, training determinism, evaluation pipelines, PGN parser validation, property-based testing, fuzz testing, self-play verification, ELO conservation, gradient verification, and replay determinism.

---

## Architecture

```
CJC Source Code (chess env + MLP agent + REINFORCE training)
     |
     v
CJC Compiler Pipeline
     Lexer → Parser → AST → TypeChecker → HIR → MIR → Optimizer
     |
     ├── cjc-eval (AST tree-walk interpreter)
     └── cjc-mir-exec (MIR register-machine executor)
           |
           v
     Deterministic Execution (same seed = bit-identical output)
           |
           v
Rust Test Harness (499 tests)
     ├── Board invariants, movegen, legality
     ├── Training determinism, gradient verification
     ├── Self-play, ELO league, model snapshots
     └── PGN parser with board parity checks
           |
           v
Interactive Browser Platform (examples/chess_rl_platform.html)
     ├── JS chess engine (faithful CJC mirror + castling/EP)
     ├── Live play with real-time policy visualization
     ├── Post-game REINFORCE training with baseline subtraction
     ├── Weight persistence (localStorage)
     ├── Opening Explorer, Style Profile, Game Review
     └── Deterministic replay with honest badge
```

---

## Key Technical Decisions

### Why Manual Gradients Instead of Autograd?

CJC has an automatic differentiation system (`cjc-ad`), but the chess RL agent uses manual analytical gradients. This was intentional:

1. It demonstrates understanding of the REINFORCE algorithm at the mathematical level
2. The manual gradients are numerically verified against finite-difference approximations
3. It proves the training pipeline works without relying on AD infrastructure correctness

### Why a Flat 64-Integer Board?

The board is a simple array of 64 integers (rank * 8 + file indexing). No bitboards, no object-oriented piece hierarchy. This keeps the CJC code readable and the determinism guarantees simple — array indexing is trivially deterministic.

### Why SplitMix64?

SplitMix64 is fast, has good statistical properties, and is simple enough to implement identically in both Rust and JavaScript (using BigInt for exact u64 parity). This is what enables the "same seed = same game" guarantee across the CJC backend and the browser platform.

### Why a Single HTML File?

The entire interactive platform is one self-contained HTML file with no build step, no dependencies, no npm. You open it in a browser and it works. This makes it trivially portable and eliminates the "it worked on my machine" problem.

---

## How to Run the Demo

### Quick Start
```bash
# Start the server
python -m http.server 8765 --directory examples

# Open in browser
# Navigate to http://localhost:8765/chess_rl_platform.html
```

### Run the Tests
```bash
# All chess RL tests
cargo test --test test_chess_rl_project
cargo test --test test_chess_rl_playability
cargo test --test test_chess_rl_hardening
cargo test --test test_chess_rl_advanced

# Full workspace
cargo test
```

### 2-Minute Demo Script

1. "This is a chess RL research platform built on CJC, a deterministic numerical language I designed."
2. *Click Play Agent* — "One click starts a game with a seeded agent."
3. *Make 3-4 moves* — "Notice the agent explains each move with real policy data."
4. *Point to replay badge* — "The seed guarantees reproducible behavior."
5. *Click Replay Exact* — "Same seed, same game — determinism verified."
6. *Switch to Review tab* — "Post-game analysis shows every decision point."
7. *Open Opening Explorer* — "Click arrows to explore the move tree."
8. "The agent uses REINFORCE with baseline subtraction. It improves across games and persists weights to localStorage."

---

## Summary

This project demonstrates end-to-end systems engineering: designing a programming language, implementing its compiler and runtime, building a non-trivial application in that language, creating an interactive visualization layer, and backing everything with 499 tests and determinism guarantees. The chess RL platform is not a toy — it's a real reinforcement learning system with real gradients, real training, and real adaptation, all running on a custom language with zero external dependencies.
