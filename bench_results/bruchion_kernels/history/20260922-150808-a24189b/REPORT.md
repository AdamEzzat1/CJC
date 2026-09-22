# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: a24189b95f0a563a4b6ab10824629602938206e4
- dirty tree: true
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernels=false
- feature bruchion-kernels: true
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k10
- kernel_sha256: c14ff6d8c6140f5268e664a55efc5675976e64657560814828d484b131133c33
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~1000000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790114429
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 13.3% max 18.3% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): svchost 26%, claude 24%, wmiprvse 22%
- last boot: 9/21/2026 9:47:14 PM (uptime 17.2 h)
- peak RSS at exit: 20736 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 48076 | 0.1967 [0.1866, 0.2292] ns/elem | 0.1907 [0.1848, 0.2371] ns/elem | 0.2358 [0.2157, 0.3128] ns/elem | 0.806, 0.969, 1.271 | 0.941, 1.199, 1.676 | 0, 0 | inside band (within A/A) |
| axpy | 10319 | 0.4749 [0.4383, 0.4931] ns/elem | 0.4834 [0.4621, 0.5141] ns/elem | 0.6526 [0.6034, 0.6930] ns/elem | 0.937, 1.018, 1.173 | 1.224, 1.374, 1.581 | 0, 0 | kernel slower, 1.37x |
| dot_kahan | 4152 | 3.3237 [3.2552, 3.3474] ns/elem | 3.1975 [3.1252, 3.9750] ns/elem | 3.1969 [3.1918, 3.9623] ns/elem | 0.934, 0.962, 1.221 | 0.954, 0.962, 1.217 | 0, 0 | inside band (within A/A) |
| mse | 3739 | 3.2296 [3.1925, 4.0694] ns/elem | 3.2951 [3.0710, 3.7494] ns/elem | 3.3030 [3.1032, 3.8072] ns/elem | 0.755, 1.020, 1.174 | 0.763, 1.023, 1.193 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 14388 | 2.0536 [1.9671, 2.3208] ns/elem | 1.9868 [1.9446, 2.5571] ns/elem | 1.3517 [1.2637, 1.5108] ns/elem | 0.838, 0.967, 1.300 | 0.545, 0.658, 0.768 | 0, 0 | kernel faster, 1.52x |
| matmul 128x128x128 | 164 | 3.4590 [3.1811, 3.7899] ns/elem | 3.1269 [2.9843, 3.9795] ns/elem | 3.2253 [2.8885, 3.5673] ns/elem | 0.787, 0.904, 1.251 | 0.762, 0.932, 1.121 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 3937 | 3.5760 [3.0898, 3.8470] ns/elem | 3.5166 [3.2414, 3.6833] ns/elem | 55.1397 [54.1807, 58.0744] ns/elem | 0.843, 0.983, 1.192 | 14.084, 15.420, 18.796 | 0, 0 | kernel slower, 15.42x |
| heat1d_residual_grad 1000x9 | 41493 | 2.7189 [2.6946, 3.4241] ns/elem | 2.8958 [2.6379, 3.4726] ns/elem | 2.3072 [2.2262, 2.3215] ns/elem | 0.770, 1.065, 1.289 | 0.650, 0.849, 0.862 | 0, 0 | kernel faster, 1.18x (within A/A) |
| heat1d_residual_grad 8192x9 | 5521 | 3.1649 [2.7153, 3.5458] ns/elem | 3.2798 [2.6896, 3.6840] ns/elem | 2.4395 [2.1276, 2.7547] ns/elem | 0.759, 1.036, 1.357 | 0.600, 0.771, 1.015 | 0, 0 | inside band (within A/A) |
| mse_loss_grad | 5757 | 2.6007 [2.5323, 3.4451] ns/elem | 2.7029 [2.5456, 2.9433] ns/elem | 2.3113 [2.2923, 2.5685] ns/elem | 0.739, 1.039, 1.162 | 0.665, 0.889, 1.014 | 0, 0 | inside band (within A/A) |
| mse+grad via GradGraph (status quo) | 592 | 11.4504 [11.3711, 15.1398] ns/elem | 11.4175 [11.1665, 13.1521] ns/elem | - | 0.738, 0.997, 1.157 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
