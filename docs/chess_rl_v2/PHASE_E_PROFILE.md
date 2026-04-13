---
title: CJC-Lang Chess RL v2.3 — Phase E Profile Report
date: 2026-04-10
scope: Tier 2 profiling results
predecessor: docs/chess_rl_v2/UPGRADE_PROMPT_v2_3.md
---

# Phase E Profile Report

## Methodology

Ran one instrumented training episode (`train_one_episode_adam_v22_instrumented`)
with profile zones around the 5 candidate hot sections:

- `rollout_total` — whole episode (outer envelope)
- `score_moves` — per-step forward pass (encode_state + forward_eager + score gathering)
- `legal_moves` — per-step move generation
- `apply_move` — per-step move application
- `rep_tracking` — per-step repetition map update

Configuration: max_moves=80, temp=1.0, penalty=0.001, seed=42, backend=cjc-eval.
The episode played 80 plies (no early termination).

## Results

| Zone | Count | Total (s) | % of episode | Mean (ms) | Min (ms) | Max (ms) | Stddev (ms) |
|------|-------|-----------|-------------|-----------|----------|----------|-------------|
| **score_moves** | 80 | 40.14 | **84.2%** | 502 | 342 | 1,411 | 179 |
| **legal_moves** | 80 | 3.66 | **7.7%** | 46 | 25 | 99 | 15 |
| apply_move | 80 | 0.07 | 0.1% | 0.8 | 0.5 | 2.3 | 0.3 |
| rep_tracking | 80 | 0.03 | 0.1% | 0.4 | 0.2 | 1.0 | 0.1 |
| rollout_total | 1 | 47.69 | 100% | — | — | — | — |

Unaccounted overhead: 47.69 - 40.14 - 3.66 - 0.07 - 0.03 = **3.79 s (7.9%)**
This covers `select_action_temp` wrapper (softmax, categorical_sample),
array_push bookkeeping, interpreter dispatch per loop iteration, and the
`a2c_update_adam` backward pass (the profiled episode dumped after
rollout, before a2c_update, so the a2c_update is in the second dump).

## Analysis

### 1. `score_moves` is 84% of wall clock

This is the per-step forward pass. It calls:
1. `encode_state(state)` — builds a 774-dim feature vector using `arr_set`
   (38 calls × 774-element array copy per call = ~29,400 element copies)
2. `forward_eager(weights, features)` — 4 matmuls + 2 adds + relu + tanh
3. `score_moves` gather loop — loops over ~25 legal moves, does
   `tensor.get()` + `array_push()` per move (~325 element copies)

The matmuls themselves are O(774×48 + 48×48 + 48×64×2 + 48×1) ≈ 50K FLOPs,
which in native Rust takes <1 ms. The ~500 ms per-call cost is dominated
by **interpreter dispatch overhead** (hundreds of function calls,
variable lookups, and COW array copies per encode_state call).

### 2. `legal_moves` is 7.7%

The move generator iterates all 64 squares, checks piece type, and
generates sliding/jumping moves. It's a tight interpreter loop but
doesn't have the quadratic array-copy problem of encode_state.

### 3. Everything else is noise

`apply_move` and `rep_tracking` together are <0.3% of wall clock.
Not worth optimizing.

## Tier 3 Kernel Candidates (ranked by expected speedup)

| Priority | Kernel | Expected speedup | Rationale |
|----------|--------|-----------------|-----------|
| **1** | `encode_state_fast` | 20-50× | Eliminates 38 × 774-element COW copies. Native: one pass over 64 squares, direct buffer writes. |
| **2** | `score_moves_batch` | 10-30× | Eliminates per-move tensor.get + array_push. Native: single forward pass + gather from logit tensors. |
| 3 | Combined (both above) | 50-100× on `score_moves` zone | The two kernels together replace the entire `score_moves` call chain. |

### Expected episode-level impact

Current: ~48 s/episode (80 plies)
- `score_moves` zone: 40.1 s → with native kernels: ~0.5-1.0 s
- `legal_moves` zone: 3.7 s → unchanged (not targeted)
- Everything else: 4.0 s → unchanged

**Predicted v2.3 episode time: ~8-9 s/episode** (5-6× speedup)

This comfortably clears the Tier 3 gate (≤30 s/episode) and likely
hits the stretch target (≤15 s/episode).

## Tier 2 Gate

- [x] Profile CSV published: `bench_results/chess_rl_v2_3/profile_hot_zones.csv`
- [x] Hot path identified: `score_moves` is 84% of wall clock
- [x] Top 5 zones ranked and analyzed
- [x] All Tier 2 tests passing (11 profile unit tests + 3 chess RL parity tests)
- [x] Workspace regression clean (pre-existing `test_cjc_v0_1_hardening` stack overflow only)

## Next Step

Proceed to Tier 3: implement `encode_state_fast` and `score_moves_batch`
native builtins, wire them into `rollout_episode_v23`, and measure the
actual speedup.
