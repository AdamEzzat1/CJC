# Phase E — compression prototypes: before/after bytes at bounded error

All lossless measurements are roundtrip-verified byte-exactly (RLE and
motif codecs) and, for trace streams, decoded back to bit-exact events
— reconstruction error is ZERO by proof, not by assumption. The
checkpoint section is lossy-advisory: each matrix is compressed to the
smallest rank whose relative Frobenius error stays within the
tolerance, and matrices that cannot beat raw storage are kept raw.

## Prototype 1 — instrumented trace streams (lossless, zero error)

Streams from instrumented runs of 6 hash-pinned Phase-D subjects (seed 42); events capped at 250000 per stream — truncation, when it happens, is printed per row.

| subject | events (measured/total) | canonical bytes → codecs | delta-columnar bytes → codecs | best ratio |
|---|---|---|---|---|
| mem_grad_a1 | 258/258 | 11622 → RLE 5180 (2.24×) / motif 2510 (4.63×) | 11622 → RLE 2234 (5.20×) / motif 331 (35.11×) | **35.11×** |
| mem_grad_a5 | 65538/65538 | 2949222 → RLE 1440831 (2.05×) / motif 557487 (5.29×) | 2949222 → RLE 554055 (5.32×) / motif 68149 (43.28×) | **43.28×** |
| fp_hot | 3402/3402 | 153102 → RLE 70820 (2.16×) / motif 17152 (8.93×) | 153102 → RLE 18588 (8.24×) / motif 3621 (42.28×) | **42.28×** |
| grad_f90_d1_n1024 | 1026/1026 | 46182 → RLE 27457 (1.68×) / motif 5215 (8.86×) | 46182 → RLE 4776 (9.67×) / motif 1152 (40.09×) | **40.09×** |
| tensor_ew_n32_i200 | 2251/2251 | 101307 → RLE 47843 (2.12×) / motif 11359 (8.92×) | 101307 → RLE 10415 (9.73×) / motif 2447 (41.40×) | **41.40×** |
| holdout_alloc_pulse | 5002/5002 | 225102 → RLE 109275 (2.06×) / motif 42877 (5.25×) | 225102 → RLE 42336 (5.32×) / motif 5268 (42.73×) | **42.73×** |

## Prototype 2 — checkpoint low-rank (advisory, rel-Frobenius ≤ 0.05)

Source: `C:\Users\adame\CJC\.claude\worktrees\stupefied-liskov-83b258\bench\cana_diagnostics\..\..\../../../bench_results/chess_rl_v2_1/checkpoint_ep60.bin` (1101744 file bytes). Diagnostic checkpoints only — never training-resumption paths.

| tensor | shape | raw bytes | chosen rank | payload bytes | rel-Frobenius |
|---|---|---|---|---|---|
| ckpt[0] | 774×48 | 297216 | — (kept raw) | 297216 | — |
| ckpt[2] | 48×48 | 18432 | — (kept raw) | 18432 | — |
| ckpt[4] | 48×64 | 24576 | — (kept raw) | 24576 | — |
| ckpt[6] | 48×64 | 24576 | — (kept raw) | 24576 | — |
| ckpt[10] | 774×48 | 297216 | 33 | 217304 | 0.0498 |
| ckpt[12] | 48×48 | 18432 | 22 | 17104 | 0.0473 |
| ckpt[14] | 48×64 | 24576 | 19 | 17208 | 0.0495 |
| ckpt[16] | 48×64 | 24576 | 21 | 19016 | 0.0478 |
| ckpt[20] | 774×48 | 297216 | 15 | 98792 | 0.0500 |
| ckpt[22] | 48×48 | 18432 | 13 | 10120 | 0.0460 |
| ckpt[24] | 48×64 | 24576 | — (kept raw) | 24576 | — |
| ckpt[26] | 48×64 | 24576 | — (kept raw) | 24576 | — |

Tensor payload totals (incl. 6576 passthrough bytes for non-2-D tensors, kept raw): **1100976 → 800072 bytes (1.38×)**

## Prototype 3 — committed disk artifacts (lossless)

| artifact | bytes → codecs |
|---|---|
| bench_results/cana_ablation/profiles.cpdb | 1088931 → RLE 1138281 (0.96×) / motif 130517 (8.34×) |
| bench_results/cana_diagnostics/phases.csv | 21341 → RLE 22036 (0.97×) / motif 8405 (2.54×) |

Hard wall unchanged: these are diagnostics artifacts; nothing here
feeds compile decisions, hashes, or profile-row stable fields.
