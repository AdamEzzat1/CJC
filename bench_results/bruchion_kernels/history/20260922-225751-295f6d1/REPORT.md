# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: 295f6d16c165e5a39816d05243093e683f86bdf0
- dirty tree: false
- dirty paths: (none outside the record directory)
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernels=false
- feature bruchion-kernels: true
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k13
- kernel_sha256: 83208dd7b5cee97f112e278d59510ef30de63df0b3d951d0c21da3016ae8aec8
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~500000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790127317
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 11.9% max 19.5% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): claude 20%, svchost 12%, msmpeng 10%
- last boot: 9/21/2026 9:47:14 PM (uptime 20.8 h)
- peak RSS at exit: 25240 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 33333 | 0.2025 [0.1527, 0.2411] ns/elem | 0.2354 [0.1559, 0.2826] ns/elem | 0.2748 [0.1945, 0.2934] ns/elem | 0.647, 1.162, 1.851 | 0.807, 1.357, 1.922 | 0, 0 | inside band (within A/A) |
| axpy | 11520 | 0.5440 [0.4807, 0.7261] ns/elem | 0.6354 [0.5028, 0.6650] ns/elem | 0.7504 [0.6738, 0.8092] ns/elem | 0.692, 1.168, 1.384 | 0.928, 1.379, 1.683 | 0, 0 | inside band (within A/A) |
| dot_kahan | 1083 | 4.2009 [3.8177, 4.4939] ns/elem | 4.2835 [3.3270, 6.3567] ns/elem | 4.2120 [3.8251, 4.7911] ns/elem | 0.740, 1.020, 1.665 | 0.851, 1.003, 1.255 | 0, 0 | inside band (within A/A) |
| mse | 2219 | 4.4987 [4.1375, 5.2501] ns/elem | 4.4156 [3.4952, 4.7163] ns/elem | 4.4206 [3.7320, 5.2425] ns/elem | 0.666, 0.982, 1.140 | 0.711, 0.983, 1.267 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 5192 | 2.6821 [2.2889, 3.1873] ns/elem | 2.7263 [2.2111, 3.1508] ns/elem | 1.9337 [1.5352, 2.0924] ns/elem | 0.694, 1.017, 1.377 | 0.482, 0.721, 0.914 | 0, 0 | kernel faster, 1.39x (within A/A) |
| matmul 128x128x128 | 59 | 3.9336 [3.6715, 4.6935] ns/elem | 4.3463 [3.8805, 4.4222] ns/elem | 4.2692 [3.6477, 4.4493] ns/elem | 0.827, 1.105, 1.204 | 0.777, 1.085, 1.212 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 1630 | 4.8101 [4.4633, 5.1618] ns/elem | 4.9446 [4.3920, 5.2558] ns/elem | 8.0297 [6.4038, 8.7327] ns/elem | 0.851, 1.028, 1.178 | 1.241, 1.669, 1.957 | 0, 0 | kernel slower, 1.67x |
| heat1d_residual_grad 1000x9 | 8547 | 3.7888 [3.5119, 4.0054] ns/elem | 3.5896 [3.2031, 4.0414] ns/elem | 3.2759 [3.1693, 3.8937] ns/elem | 0.800, 0.947, 1.151 | 0.791, 0.865, 1.109 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 1802 | 3.5514 [3.5108, 4.2422] ns/elem | 3.9882 [3.4575, 4.7519] ns/elem | 3.5387 [3.0954, 3.6552] ns/elem | 0.815, 1.123, 1.353 | 0.730, 0.996, 1.041 | 0, 0 | inside band (within A/A) |
| matmul tiled 128x128x128 | 348 | 0.3600 [0.2972, 0.4106] ns/elem | 0.3238 [0.2837, 0.3874] ns/elem | 0.4864 [0.4235, 0.5949] ns/elem | 0.691, 0.899, 1.304 | 1.032, 1.351, 2.002 | 1, 0 | kernel slower, 1.35x |
| matmul tiled 256x256x256 | 98 | 0.3466 [0.3351, 0.3541] ns/elem | 0.3276 [0.2982, 0.3869] ns/elem | 0.5592 [0.4981, 0.6026] ns/elem | 0.842, 0.945, 1.154 | 1.407, 1.614, 1.798 | 1, 0 | kernel slower, 1.61x |
| relu call path: bare ffi vs kernel::relu_raw | 19230 | 0.3129 [0.2426, 0.3240] ns/elem | 0.2898 [0.2396, 0.3626] ns/elem | 0.3071 [0.2515, 0.3296] ns/elem | 0.740, 0.926, 1.494 | 0.776, 0.981, 1.359 | 0, 0 | inside band (within A/A) |
| relu loop: Rust body vs bare ffi | 6587 | 0.2489 [0.2225, 0.3152] ns/elem | 0.2307 [0.2088, 0.2741] ns/elem | 0.2877 [0.2492, 0.3223] ns/elem | 0.662, 0.927, 1.232 | 0.791, 1.156, 1.449 | 0, 0 | inside band (within A/A) |
| mse_loss_grad | 2672 | 3.0511 [2.9900, 3.8290] ns/elem | 2.8156 [2.4735, 3.5323] ns/elem | 2.1568 [1.7280, 2.4704] ns/elem | 0.646, 0.923, 1.181 | 0.451, 0.707, 0.826 | 0, 0 | kernel faster, 1.41x (within A/A) |
| mse+grad via GradGraph, fused mean | 276 | 21.8532 [18.7158, 24.5113] ns/elem | 20.9544 [18.4487, 25.8270] ns/elem | 22.2145 [18.9745, 24.7323] ns/elem | 0.753, 0.959, 1.380 | 0.774, 1.017, 1.321 | 97, 97 | inside band (within A/A) |
| mse+grad via GradGraph, unfused (status quo) | 302 | 31.5398 [26.4548, 36.5577] ns/elem | 30.4190 [28.1469, 33.8078] ns/elem | - | 0.770, 0.964, 1.278 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
