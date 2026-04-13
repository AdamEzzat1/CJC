---
title: Chess RL — v1 vs v2.5 Full Comparison
date: 2026-04-10
purpose: Reference for LinkedIn post and project documentation
---

# Chess RL Demo: v1 → v2.5 Evolution

## The Journey

| Version | Date | Focus |
|---|---|---|
| **v1** | 2026-03 | Proof of concept — can CJC-Lang express RL? |
| **v2.0** | 2026-04-09 | Full rewrite — real autodiff, complete chess rules |
| **v2.1** | 2026-04-09 | Adam optimizer, checkpoints, Elo, PGN, Vizor curves |
| **v2.2** | 2026-04-09 | Cheap ML fixes (longer games, move penalty, Zobrist) |
| **v2.3** | 2026-04-10 | Profiling + native hot-path kernels (7.7× rollout speedup) |
| **v2.4** | 2026-04-10 | Arena GradGraph + dead node elimination (1.89× episode speedup) |
| **v2.5** | 2026-04-10 | In-place gradients, fused MLP, PINN graph reuse, backward_collect |

---

## Head-to-Head Comparison

| Dimension | v1 (original) | v2.5 (current) | Factor |
|---|---|---|---|
| **Network parameters** | ~1,072 | ~45,873 | **43×** larger |
| **Input features** | 66 (raw board + move) | 774 (12 piece planes × 64 + rights + clock + EP) | **12×** richer |
| **Gradient method** | Hand-rolled REINFORCE (~95 LOC) | Full GradGraph autodiff (arena-based, fused MLP) | Automatic |
| **Optimizer** | SGD (manual scalar loop) | Adam (native `adam_step` builtin, ~9× faster) | Native |
| **Chess rules** | Basic (no castling, no en passant, no 50-move) | Complete (castling, EP, promotion, 50-move, insufficient material, Zobrist repetition) | Full |
| **Algorithm** | REINFORCE | A2C + GAE (γ=0.99, λ=0.95) | State-of-art |
| **Network architecture** | 66→16→1 MLP (single scalar output) | 774→128→64→1 (factored from+to policy heads + value head) | Full A2C |
| **Tests** | 49 | 97+ (chess RL v2 suite alone) | **2×** |
| **Training signal** | None observed (tests capped at <60s) | Elo +13.9 (120 episodes, first measurable learning) | Measurable |
| **Determinism** | Bit-identical | Bit-identical | Preserved |
| **External dependencies** | 0 | 0 | Zero always |
| **Evaluation infrastructure** | Win-rate vs random | Elo-lite + snapshot gauntlet + PGN + Vizor SVG curves | Full suite |
| **Checkpoint system** | None | 31-tensor bundle + CSV training log | Production-grade |

---

## What Changed in the Autodiff Engine (v2.4 → v2.5)

### v2.4 (arena refactor)
- Replaced `Vec<Rc<RefCell<GradNode>>>` with flat arrays
- Dead node elimination (skip 20-30% unreachable nodes)
- Result: 82.2 → 43.51 s/episode (1.89× speedup)

### v2.5 additions
- **P4: In-place gradient accumulation** — `Tensor::add_assign_unchecked` eliminates N/2 allocations per backward pass
- **P6: backward_collect** — batched zero_grad + backward + gradient collection in one native call
- **P7: PINN graph reuse** — build graph once, `set_tensor` + `reforward` per epoch (eliminates O(epochs × graph_size) allocations)
- **P8: Fused MLP layer** — `GradOp::MlpLayer` collapses transpose + matmul + bias-add + activation into one node (3× fewer nodes per layer, fused backward)

---

## Language Features Exercised

The chess RL demo exercises nearly every CJC-Lang capability:

| Feature | How it's used |
|---|---|
| **Tensors** | Board encoding (774-D), weight matrices, gradient tensors |
| **Autodiff (GradGraph)** | Full A2C loss backward through policy + value heads |
| **Closures** | Callback functions for evaluation, training hooks |
| **Pattern matching** | Piece type dispatch, move classification |
| **Structs** | Board state, game config, training stats |
| **For-loops** | Rollout iteration, move generation, Elo computation |
| **While-loops** | Game simulation (until terminal state) |
| **If-expressions** | Legal move filtering, terminal detection |
| **Arrays (COW)** | Move lists, trajectory buffers, weight checkpoints |
| **Deterministic RNG** | SplitMix64 seed threading through all randomness |
| **File I/O** | CSV training logs, PGN game records |
| **Vizor (SVG)** | Training curve visualization |
| **Native builtins** | `adam_step`, `encode_state_fast`, `score_moves_batch`, `categorical_sample` |

---

## Performance Engineering Story

```
v1:   Hand-rolled gradients, ~1K params, <60s test cap
        ↓
v2.0: GradGraph autodiff, 45K params, ~120 s/episode
        ↓ (2.1-2.3: Adam, profiling, native kernels)
v2.3: Native rollout (7.7× faster), but backward pass dominates → 82.2 s/episode
        ↓ (Amdahl's Law insight)
v2.4: Arena GradGraph + dead node elimination → 43.51 s/episode (1.89×)
        ↓
v2.5: In-place gradients + fused MLP layers → 29.80 s/episode (2.76× from v2.3)
```

Each step was driven by honest profiling, not guesswork. The Amdahl's Law
moment (v2.3) was pivotal: 7.7× rollout speedup translated to only ~0.9×
episode speedup because the backward pass (which was untouched) dominated.
This redirected focus from the interpreter to the autodiff engine itself.

---

## What This Proves About CJC-Lang

1. **A language can grow to support its hardest demo.** Adam optimizer
   became a native builtin. The autodiff engine gained arena allocation
   and fused MLP layers. Hot-path kernels were added as native builtins.
   All driven by measured need, not speculation.

2. **Zero external dependencies is viable for real ML.** No PyTorch, no
   NumPy, no Stockfish. Every tensor operation, every gradient, every
   optimizer step runs in code owned by CJC-Lang.

3. **Determinism by design scales.** From a 1K-parameter toy to a
   45K-parameter A2C agent with Zobrist hashing, checkpoint serialization,
   and multi-episode training — same seed = same game, always.

4. **Honest measurement matters.** The demo doesn't claim to beat
   Stockfish. It reports Elo +13.9 from 120 episodes — the first
   measurable learning signal — and documents exactly where the
   remaining bottlenecks are.
