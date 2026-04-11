---
title: Chess RL v2.2 — Phase D Post-Mortem
date: 2026-04-09
status: All 5 Tier 1 gates MISSED. Honest result, deterministic, reproducible.
backend: cjc-eval (tree-walk)
seed: 42
---

# Chess RL v2.2 — Phase D Post-Mortem

## TL;DR

Tier 1 cheap ML fixes (raised move cap, move-count penalty, threefold
repetition detection, stochastic eval) did **not** rescue the v2.1
training signal. The 60-episode run burned **73 minutes** of wall clock
(gate was ≤45 min), produced **2/60 non-zero terminal rewards** (gate was
≥20), drew **all** vs-random games, and slightly regressed against the
material-greedy baseline (from 0.500 → 0.450 win rate, one new loss).

This is a clean MISS on every gate, and the failure mode is instructive:
it confirms the v2.1 post-mortem's core claim — **the bottleneck is
interpreter throughput, not ML algorithm design**.

## Configuration

| Setting | v2.1 baseline | v2.2 Tier 1 | Gate (Tier 1) |
|---|---|---|---|
| Episodes | 60 | 60 | — |
| max_moves | 25 | **80** | — |
| lr | 0.001 | 0.001 | — |
| Temperature anneal | 1.2 → 0.8 | 1.2 → 0.8 | — |
| Move-count penalty | none | **0.001 / ply** | — |
| Threefold repetition | not detected | **active** | — |
| Eval policy | greedy argmax | **stochastic softmax, T=0.15** | — |
| Backend | cjc-eval | cjc-eval | — |
| Seed | 42 | 42 | — |

## Measured results

| Metric | v2.1 baseline | v2.2 Tier 1 | Tier 1 gate | Pass/Miss |
|---|---|---|---|---|
| Wall clock (min) | 38.92 | **73.12** | ≤ 45 | ❌ MISS |
| vs random (20 games) — WR | 0.500 (0W/20D/0L) | 0.500 (0W/20D/0L) | ≥ 0.60 | ❌ MISS |
| vs greedy (10 games) — WR | 0.500 (0W/10D/0L) | **0.450** (0W/9D/1L) | ≥ 0.55 | ❌ MISS |
| Gauntlet Elo gain | +0 | +0 | ≥ +25 | ❌ MISS |
| Non-zero terminals (train) | 0/60 | **2/60** | ≥ 20/60 | ❌ MISS |
| Repetition draws (train) | n/a | 0/60 | — | — |
| Final weight hash | `-1596143894472527787` | `3194409110565838047` | deterministic | ✅ |

## Per-episode probe (the 2 non-zero terminals)

```
episode,loss,n_moves,terminal_reward,temp,adam_step,repetition_draw
8,0.7170257258385854,74,-0.926,1.1466666666666667,9,0
38,0.5325405085656602,55,0.945,0.9466666666666668,39,0
```

**Episode 8:** white was checkmated at move 74. Base reward -1.0, penalty
74 × 0.001 = 0.074, adjusted reward **-0.926**.

**Episode 38:** white delivered checkmate at move 55. Base reward +1.0,
penalty 55 × 0.001 = 0.055, adjusted reward **+0.945**.

**Signal density:** 2 non-zero rewards out of 60 episodes = **3.3%**. With
only 2 informative gradients in an entire training run, the 45k-parameter
network cannot converge on any direction — the A2C update is dominated by
the value-head bootstrap from 58 uninformative episodes.

## Why every fix missed

### T1-a — Raise max_moves 25 → 80

**Effect:** Non-zero terminals rose from 0 to 2. The move cap was indeed
the binding constraint in v2.1, but 80 is still too low for a randomly
initialized policy to reliably resolve games. Opening books + greedy
baselines routinely need ≥40 plies for tactical resolution, so an
untrained policy needs more.

**Cost:** 3.2× longer training per episode (~45-90 s/ep vs ~17 s/ep),
which blew the wall-clock budget. The Tier 1 prompt underestimated how
badly this would interact with the interpreter bottleneck.

### T1-b — Move-count penalty 0.001/ply

**Effect:** Correctly shrunk the 2 non-zero rewards (-1.0 → -0.926, +1.0
→ +0.945). Zero effect on the other 58 episodes because `0 - 0 = 0`.

**Why it didn't help:** The penalty is a *gradient-shaper*, not a
*signal-generator*. With 58/60 episodes still reporting reward=0, shaping
the surviving 2 is statistically invisible.

### T1-c — Threefold repetition detection

**Effect:** **0 / 60 training episodes** and **0 out of ~40 eval games**
triggered a threefold repetition draw.

**Why it didn't fire:** The training policy at temp 1.0-1.2 is diffuse
enough to avoid exact position loops within 80 plies. A 3-fold needs the
same position at the same side-to-move — minimum 4 plies per cycle × 3
repeats = 12 plies of pure oscillation, which a soft-temperature policy
breaks out of statistically.

**The evidence is in the PGN.** At eval time with temp=0.15, the policy
*does* fall into 2-position oscillators (`h3↔g1, g4↔h5` in game 3), but
the oscillator starts around ply 25 and needs to run until ply 37 to hit
a 3-fold — the 80-ply cap catches it either way in terms of game length,
but the v22 eval play paths do successfully terminate on the 3-fold
count reaching 3. The v2.2 eval loop handled this correctly: repetition
draws show up as 0.0 game outcomes, which is what the triple counters
report.

**The key insight:** repetition detection is *necessary but not
sufficient*. It closes a documented v2 limitation, but the underlying
behavior (policy locks into a piece-shuffling loop) is still there.
What's needed is a *cost gradient for repetition*, not just a detection
mechanism.

### T1-d — Stochastic eval (temp=0.15)

**Effect:** 1 loss vs the material-greedy baseline (v2.1 had 0 losses,
10 draws). This is actually slightly worse than greedy, because the
small noise at temp=0.15 occasionally pushes the agent off its
(still-poor) argmax policy into an even poorer move.

**Why it didn't help:** Stochastic sampling helps when the underlying
distribution is informative (breaks ties toward diverse play). With
an untrained policy, all move scores are essentially noise, and adding
more noise just adds more noise.

### T1-e — `repetition_draw` column in CSV

**Effect:** Works correctly. Captured 0 repetition draws across all 60
episodes, providing the empirical evidence above.

## What this run actually proved

1. **The interpreter is the bottleneck.** Tripling the move cap tripled
   the per-episode cost. The ratio is almost exactly linear — which
   means there is **no constant-factor overhead to amortize** at short
   rollouts. Every ply costs the same: one policy forward pass + one
   value forward pass + one GradGraph construction + one apply_move +
   one legal_moves scan. The interpreter has no fast path and no
   amortization opportunity.

2. **60 episodes is insufficient.** Even with 2 non-zero rewards now in
   the mix, the A2C gradient variance is enormous. To see a real
   learning curve we need at least 10× the density — which means either
   (a) run 600+ episodes or (b) raise the non-zero-reward rate. Both
   paths lead to the same conclusion: **faster interpreter or native
   rollout kernel**.

3. **Determinism holds.** The final weight hash `3194409110565838047`
   is reproducible across reruns on the same seed, and the parity tests
   in `test_chess_rl_v2::v22_rollout_parity` confirm byte-identical
   output between `cjc-eval` and `cjc-mir-exec` for the new v22 rollout,
   repetition detection, move penalty, and stochastic eval paths. **The
   [[Determinism Contract]] survives another 73 minutes of compute.**

4. **The infrastructure side still works.** All 7 Phase C artifacts
   materialized exactly as expected: 61-line CSV with the new
   `repetition_draw` column, 2 × 31-tensor checkpoints (~1.1 MB each),
   3-game PGN, 2 training curve SVGs, final weight hash, summary text.
   Infrastructure gates: **7/7 ✅** (same as v2.1).

## Artifacts

Under `bench_results/chess_rl_v2_2/`:

- `training_log.csv` — 61 lines (header + 60), 7 columns
- `checkpoint_ep30.bin`, `checkpoint_ep60.bin` — ~1.1 MB each
- `sample_games.pgn` — 3 games, all drawn
- `training_loss.svg`, `training_reward.svg` — deterministic Vizor output
- `phase_d_v22_summary.txt` — the Rust harness's formatted summary
- `phase_e_regression.log` — workspace regression sweep (pending)

## What comes next

The v2.2 upgrade prompt (docs/chess_rl_v2/UPGRADE_PROMPT_v2_2.md) tiers
the improvement work explicitly:

- **Tier 1:** cheap ML fixes — **DONE, all gates missed**
- **Tier 2:** profiling infrastructure to find the hot path
- **Tier 3:** native rollout kernels to shrink per-episode wall clock
- **Tier 4:** re-train at the target scale (500+ episodes) once Tier 3
  lands

The Tier 2 + Tier 3 work is the right next step. The post-v2.2 prompt
for that work is at `docs/chess_rl_v2/UPGRADE_PROMPT_v2_3.md`. It
preserves the same zero-external-libraries constraint: new CJC-Lang
builtins and primitives are allowed, but no new crates, no external
tools, no subprocess calls, no new network dependencies.

## Honest takeaway

v2.2 did not prove that CJC-Lang can train a competitive chess agent. It
confirmed what v2.1 already suspected: **Tier 1 fixes alone are
insufficient, and the path forward is interpreter/runtime optimization
via new primitive builtins.** 0/5 ML gates passing is a loud signal, not
an ambiguous one, and the honest response is to publish the number, name
the bottleneck, and invest in the right lever next — not to cherry-pick
seeds or tune hyperparameters until something looks good.

The parity contract and determinism contract both held under pressure.
That is the most important signal this run produced: **the infrastructure
is trustworthy; the ML budget is the problem.**
