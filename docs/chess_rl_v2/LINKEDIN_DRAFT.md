---
title: LinkedIn Post Draft — Chess RL in CJC-Lang
date: 2026-04-10
status: Ready to post (personal hook is yours to add)
---

# LinkedIn Post — Ready to Adapt

*Add your personal hook at the top. The rest is ready to post.*

---

I built a chess-playing AI from scratch — in a programming language I also built from scratch. No PyTorch. No NumPy. No Stockfish. Zero external dependencies.

The language is CJC-Lang (Computational Jacobian Core), a numerical programming language I wrote in Rust — 21 crates, ~96K lines of code. It's designed for ML and scientific computing where reproducibility matters: the same input always produces the same output, enforced through careful floating-point handling, deterministic data structures, and explicit randomness control.

The chess demo is ~1,715 lines of pure CJC-Lang. It includes:

→ A reinforcement learning agent (A2C + GAE) with 45,873 trainable parameters
→ A complete chess engine — castling, en passant, promotion, repetition detection, all of it
→ An Adam optimizer, self-play training loop, and evaluation suite with Elo ratings

The language provides the building blocks: built-in tensors, an automatic differentiation engine for computing gradients, and native performance kernels that I added as bottlenecks emerged during development.

That process was the most interesting part. Profiling showed 84% of training time in one function — so I wrote a native kernel (7.7× faster). Then the backward pass dominated instead, so I redesigned the autodiff engine: arena allocation, dead node elimination, fused layers. Total result: training went from 82 to 30 seconds per episode.

The agent learns modestly — Elo +13.9 after 120 episodes. It won't beat Stockfish. But that's not the point. The point is that a language I built from the ground up can express, train, and evaluate a real ML agent — complete game rules, gradient computation, optimization, evaluation — with nothing borrowed.

Every tensor op, every gradient, every optimizer step — code I own.

---

# Technical Reference (not for posting)

## The Numbers

| | Start (v1) | Now (v2.5) |
|---|---|---|
| Network | 1,072 params | **45,873 params** (43× larger) |
| Input | 66 features | **774 features** (12× richer) |
| Gradients | 95 lines of hand-rolled chain rule | **Automatic differentiation** with arena-based graph |
| Optimizer | Manual SGD | **Native Adam** (~9× faster than interpreted) |
| Chess rules | Basic (no castling) | **Complete** (castling, en passant, 50-move, Zobrist repetition) |
| Algorithm | REINFORCE | **A2C + GAE** |
| Training speed | N/A | **29.80 s/episode** (from 82.2 s after optimization — 2.76× faster) |
| Learning signal | None | **Elo +13.9** (first measurable learning) |
| Determinism | Yes | **Still yes** — same seed = same game, always |

## Key Stats
- Per-episode: 29.80 s (v2.5) vs 82.2 s (v2.3) = 2.76× faster
- Weight hash: `9.790915694115341` (determinism proof)
- 80/80 autodiff tests, 55/55 parity tests, 97/97 chess RL tests
- 0 external dependencies, 21 crates, ~96K LOC

## Two-Layer Architecture (for answering questions)
- **Rust layer:** CJC-Lang compiler + runtime + autodiff + tensor library (21 crates, ~96K LOC)
- **CJC-Lang layer:** Chess engine + RL agent + training loop + evaluation (~1,715 LOC)
- The chess demo is a CJC-Lang PROGRAM. CJC-Lang is a Rust PROJECT.
