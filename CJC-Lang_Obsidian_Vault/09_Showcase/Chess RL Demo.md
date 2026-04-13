---
title: Chess RL Demo
tags: [showcase, ml, rl]
status: Implemented
---

# Chess RL Demo

The **capability benchmark** for CJC-Lang: a complete chess engine plus a REINFORCE-trained policy network, implemented in pure CJC-Lang and running end-to-end through both [[cjc-eval]] and [[cjc-mir-exec]].

## What's in the CJC-Lang side

From README.md's honest description:

- **Complete chess engine** — Board representation, legal move generation for all piece types, castling, en passant, pawn promotion, check and checkmate detection, stalemate, 50-move rule, insufficient material, threefold repetition
- **Neural network forward pass** — `forward_move()` with linear layers + tanh through CJC's tensor primitives
- **Training loop** — REINFORCE policy gradient with reward shaping, weight updates, gradient computation
- **Board encoding** — Board state → feature tensor
- **Action selection** — Softmax over legal moves with temperature-based exploration

Source: `tests/chess_rl_project/`, `tests/chess_rl_advanced/`.

## Tests

**319 tests** across 4 suites (verified by `#[test]` marker count on 2026-04-09):

| Suite | Path | Tests |
|---|---|---|
| Project | `tests/chess_rl_project/` | 69 |
| Advanced | `tests/chess_rl_advanced/` | 84 |
| Hardening | `tests/chess_rl_hardening/` | 104 |
| Playability | `tests/chess_rl_playability/` | 62 |
| **Total** |  | **319** |

Earlier drafts of this note claimed "216 dedicated tests" — that figure was stale. The suites cover:
- Move generation correctness (all piece types, all special moves)
- Training determinism (same seed → identical weights after N episodes)
- Network output validity (no NaN, bounded values)
- Parity between AST and MIR execution
- Fuzz testing across multiple random seeds

Suites: `tests/chess_rl_project/` (49 tests in CLAUDE.md memory; 150 per README), `tests/chess_rl_advanced/` (66 tests), `tests/chess_rl_hardening/`, `tests/chess_rl_playability/`. The discrepancy between 49 and 150 comes from different counting of sub-tests; **Needs verification** of exact total.

## What's in the JavaScript frontend

The browser demo `examples/chess_rl_platform.html` (~158KB, single HTML file) contains a JavaScript **mirror** of the chess engine for interactivity. CJC-Lang does not yet compile to WebAssembly, so the browser UI is JavaScript.

The JS frontend has a *more advanced* agent than the CJC-Lang side:
- Actor-Critic with GAE
- Residual blocks, GELU, He init
- ~110K parameters
- 288-dimensional feature vector
- Curriculum learning (vs random → heuristic → self-play)
- Training dashboard with live charts
- 4 baseline opponents
- Tactical puzzle suite

This split is honest: the CJC-Lang side proves the language can handle the *pipeline*; the JS side shows what the language *could* do if it had a browser target. The README is clear that this is not a grandmaster chess program.

## Training results (500 episodes, V1 network in CJC-Lang)

| Metric | Value |
|---|---|
| vs Random | 60% win rate (12W / 8D / 0L) |
| vs Heuristic | 20% (4W / 16D / 0L) |
| vs 1-ply Minimax | 20% (2W / 5D / 3L) |
| Tactical puzzles | 1/5 (pawn promotion) |
| Value loss | Decreased ~47% over training |
| Policy entropy | Decreased ~16% (policy specializing) |

The README's own framing: *"The demo proves the training pipeline works end-to-end, not that it produces a grandmaster."*

## What the demo proves

1. **CJC-Lang can express non-trivial algorithms** — full chess engine in CJC.
2. **Numerical computing works end-to-end** — NN forward pass + gradient + weight update through CJC tensor primitives.
3. **Determinism holds under load** — same seed, same weights, many episodes.
4. **Both backends agree** — every chess test identical in eval and mir-exec.
5. **Test infrastructure scales** — 200+ tests for a single domain.

## What the demo does NOT prove

- Production readiness (interpreter is slow).
- Module system wiring (demo is single-file).
- Browser compilation (JS frontend is a workaround).

## Running it

```
cargo test --test test_chess_rl_project
cargo test --test test_chess_rl_advanced
```

For the interactive browser demo, open `examples/chess_rl_platform.html` directly — no build step.

## Related

- [[What CJC-Lang Can Already Do]]
- [[ML Primitives]]
- [[Autodiff]]
- [[Determinism Contract]]
- [[Parity Gates]]
