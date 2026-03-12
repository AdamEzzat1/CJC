# Chess RL Subsystem Audit

## Architecture Summary

The chess RL subsystem is a complete deterministic reinforcement learning benchmark written in pure CJC (~685 lines of CJC source) with a Rust test harness (~2,009 lines).

### Components

| Component | Location | Lines | Purpose |
|-----------|----------|-------|---------|
| CHESS_ENV | cjc_source.rs:76-421 | 345 | Board, movegen, legality, terminal detection |
| RL_AGENT | cjc_source.rs:435-589 | 154 | 2-layer MLP, forward pass, REINFORCE |
| TRAINING | cjc_source.rs:596-760 | 165 | Rollout, training loop, evaluation |
| Test harness | 7 test files + debug | ~1,300 | 66 passing tests |

### Board Representation
- Flat 64-int array, row-major (rank*8 + file)
- Piece encoding: 0=empty, +1..+6=white (P,N,B,R,Q,K), -1..-6=black
- Functional model: `apply_move()` returns new board, no mutation

### RL Agent
- Input: [1, 66] = board features + normalized move coordinates
- Hidden: [1, 16] with ReLU
- Output: scalar score per move, softmaxed for categorical sampling
- REINFORCE gradient with manual backprop

### Determinism
- All randomness via seeded SplitMix64
- Move generation in strict square order (0-63)
- No HashMap, no time-based seeds, no platform dependencies

## Current Test Coverage

| Suite | Tests | Determinism Tests |
|-------|-------|-------------------|
| Board invariants | 11 | 0 |
| Movegen determinism | 8 | 2 |
| Legal move sanity | 9 | 1 |
| Rollout determinism | 5 | 3 |
| Training smoke | 8 | 0 |
| Training determinism | 5 | 4 |
| Eval vs random | 6 | 0 |
| Debug | 20 | 0 |
| **Total** | **72** | **10** |

## Gaps Identified

1. **Board mutation safety** — `apply_move` immutability not tested (fixed in hardening)
2. **Pawn promotion edge cases** — only tested via scholar's mate path
3. **Feature encoding symmetry** — side-perspective negation not verified (fixed in hardening)
4. **Agent numerical stability** — forward pass finite checks missing (fixed in hardening)
5. **Property-based coverage** — no randomized testing (fixed: 10 proptest suites added)
6. **Fuzz coverage** — no crash resistance testing (fixed: 5 Bolero targets added)
7. **Trace export** — no way to capture per-ply game state for visualization

## Simplifications (Documented)
- No castling
- No en passant
- Pawns auto-promote to queen only
- 200-halfmove draw limit (no 50-move rule)
