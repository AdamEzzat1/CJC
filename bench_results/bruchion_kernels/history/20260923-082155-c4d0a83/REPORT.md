# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: c4d0a83d5d3f0359b788b1f563bcb323765e9455
- dirty tree: false
- dirty paths: (none outside the record directory)
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernel_ops=0xfff bruchion_kernels=false
- feature bruchion-kernels: true
- routing set, kernel arm: axpy, dot_kahan, relu, mse, mse_grad, mse_grad_scaled, matmul, matmul_tiled, adam_step, powi, heat1d_residual, heat1d_residual_grad
- routing set, default: dot_kahan, mse, mse_grad, matmul, heat1d_residual_grad
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k13
- kernel_sha256: 83208dd7b5cee97f112e278d59510ef30de63df0b3d951d0c21da3016ae8aec8
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~500000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790143113
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 12.3% max 21.7% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): claude 29%, wmiprvse 15%, svchost 12%
- last boot: 9/21/2026 9:47:14 PM (uptime 25.2 h)
- peak RSS at exit: 91648 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 20833 | 0.2021 [0.1905, 0.2236] ns/elem | 0.2021 [0.1965, 0.2186] ns/elem | 0.2571 [0.2519, 0.2948] ns/elem | 0.879, 1.000, 1.147 | 1.127, 1.272, 1.547 | 0, 0 | kernel slower, 1.27x |
| axpy | 9041 | 0.6069 [0.5160, 0.6491] ns/elem | 0.5760 [0.5320, 0.6321] ns/elem | 0.7696 [0.6960, 0.8690] ns/elem | 0.820, 0.949, 1.225 | 1.072, 1.268, 1.684 | 0, 0 | kernel slower, 1.27x |
| dot_kahan | 1896 | 3.5750 [3.5189, 3.7933] ns/elem | 3.6055 [3.5590, 3.6919] ns/elem | 3.7099 [3.5157, 3.8863] ns/elem | 0.938, 1.009, 1.049 | 0.927, 1.038, 1.104 | 0, 0 | inside band (within A/A) |
| mse | 1828 | 3.6569 [3.5031, 4.3941] ns/elem | 3.6866 [3.4907, 3.9463] ns/elem | 3.6464 [3.5596, 3.7825] ns/elem | 0.794, 1.008, 1.127 | 0.810, 0.997, 1.080 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 4219 | 2.2962 [2.1224, 2.8101] ns/elem | 2.3533 [2.2441, 2.7987] ns/elem | 1.5641 [1.5114, 2.1008] ns/elem | 0.799, 1.025, 1.319 | 0.538, 0.681, 0.990 | 0, 0 | kernel faster, 1.47x |
| matmul 128x128x128 | 52 | 3.6045 [3.4626, 3.7565] ns/elem | 3.5925 [3.4749, 4.1075] ns/elem | 3.4746 [3.3719, 3.6059] ns/elem | 0.925, 0.997, 1.186 | 0.898, 0.964, 1.041 | 0, 0 | inside band (within A/A) |
| matmul 256x17x33 | 1091 | 2.4423 [2.2390, 2.7764] ns/elem | 2.4740 [2.3140, 2.9327] ns/elem | 1.6638 [1.5587, 1.9856] ns/elem | 0.833, 1.013, 1.310 | 0.561, 0.681, 0.887 | 0, 0 | kernel faster, 1.47x |
| matmul 33x64x17 | 4420 | 3.0129 [2.9581, 3.3066] ns/elem | 3.1234 [3.0358, 3.3980] ns/elem | 2.7298 [2.5898, 2.9203] ns/elem | 0.918, 1.037, 1.149 | 0.783, 0.906, 0.987 | 0, 0 | kernel faster, 1.10x (within A/A) |
| matmul 17x33x64 | 4854 | 3.1252 [2.8965, 3.5917] ns/elem | 3.0067 [2.9405, 3.6681] ns/elem | 2.4679 [2.3803, 2.7509] ns/elem | 0.819, 0.962, 1.266 | 0.663, 0.790, 0.950 | 0, 0 | kernel faster, 1.27x (within A/A) |
| matmul 32x32x32 | 4194 | 3.1561 [2.5698, 3.6900] ns/elem | 2.9973 [2.4971, 3.5731] ns/elem | 2.5854 [2.1011, 2.9327] ns/elem | 0.677, 0.950, 1.390 | 0.569, 0.819, 1.141 | 0, 0 | inside band (within A/A) |
| matmul 64x64x8 | 5353 | 2.7694 [2.7628, 2.9453] ns/elem | 2.8115 [2.7085, 2.8777] ns/elem | 2.6163 [2.4282, 2.9069] ns/elem | 0.920, 1.015, 1.042 | 0.824, 0.945, 1.052 | 0, 0 | inside band (within A/A) |
| matmul 16x16x1024 | 494 | 2.5000 [2.4592, 3.0886] ns/elem | 2.4964 [2.4253, 2.6548] ns/elem | 1.9006 [1.8694, 2.1220] ns/elem | 0.785, 0.999, 1.080 | 0.605, 0.760, 0.863 | 0, 0 | kernel faster, 1.32x |
| adam_step (t = 7) | 1858 | 3.5647 [3.3531, 4.0307] ns/elem | 3.4510 [3.3450, 4.1153] ns/elem | 5.7242 [5.6656, 6.8191] ns/elem | 0.830, 0.968, 1.227 | 1.406, 1.606, 2.034 | 0, 0 | kernel slower, 1.61x |
| heat1d_residual_grad 1000x9 | 18518 | 4.3065 [2.7781, 6.1674] ns/elem | 4.1384 [2.6653, 12.4809] ns/elem | 3.4168 [2.5996, 7.4485] ns/elem | 0.432, 0.961, 4.493 | 0.422, 0.793, 2.681 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 2072 | 2.6278 [2.5749, 3.0450] ns/elem | 2.6871 [2.6625, 2.7751] ns/elem | 2.2772 [2.2054, 2.5504] ns/elem | 0.874, 1.023, 1.078 | 0.724, 0.867, 0.990 | 0, 0 | kernel faster, 1.15x |
| matmul tiled 128x128x128 | 1067 | 0.2470 [0.2402, 0.2605] ns/elem | 0.2462 [0.2389, 0.2551] ns/elem | 0.3587 [0.3539, 0.3771] ns/elem | 0.917, 0.997, 1.062 | 1.359, 1.452, 1.570 | 1, 0 | kernel slower, 1.45x |
| matmul tiled 256x256x256 | 123 | 0.2654 [0.2547, 0.2894] ns/elem | 0.2591 [0.2572, 0.2666] ns/elem | 0.3731 [0.3722, 0.3761] ns/elem | 0.889, 0.976, 1.046 | 1.286, 1.406, 1.476 | 1, 0 | kernel slower, 1.41x |
| relu call path: bare ffi vs kernel::relu_raw | 27027 | 0.2077 [0.1948, 0.2208] ns/elem | 0.2113 [0.1961, 0.2164] ns/elem | 0.1999 [0.1988, 0.2326] ns/elem | 0.888, 1.017, 1.111 | 0.900, 0.962, 1.194 | 0, 0 | inside band (within A/A) |
| relu loop: Rust body vs bare ffi | 22831 | 0.1618 [0.1533, 0.1712] ns/elem | 0.1575 [0.1548, 0.1589] ns/elem | 0.1976 [0.1955, 0.2137] ns/elem | 0.904, 0.973, 1.037 | 1.142, 1.221, 1.395 | 0, 0 | kernel slower, 1.22x |
| mse_loss_grad | 3455 | 1.7533 [1.7251, 2.0415] ns/elem | 1.8156 [1.7206, 1.9901] ns/elem | 1.3640 [1.2708, 1.4475] ns/elem | 0.843, 1.036, 1.154 | 0.622, 0.778, 0.839 | 0, 0 | kernel faster, 1.29x |
| mse+grad via GradGraph, fused mean | 460 | 11.3385 [11.0037, 20.4165] ns/elem | 11.2786 [10.6202, 16.1492] ns/elem | 11.2338 [10.6287, 12.0402] ns/elem | 0.520, 0.995, 1.468 | 0.521, 0.991, 1.094 | 85, 85 | inside band (within A/A) |
| mse+grad via GradGraph, unfused (status quo) | 665 | 18.5762 [18.3849, 18.8864] ns/elem | 18.0082 [17.9004, 18.4837] ns/elem | - | 0.948, 0.969, 1.005 | - | 110, - | status quo, not routed |
| conv1d 4096x8x9 (kernel::conv1d_raw) | 1600 | 1.1817 [1.1287, 1.4979] ns/elem | 1.2508 [1.1495, 1.3835] ns/elem | - | 0.767, 1.058, 1.226 | - | 0, - | status quo, not routed |
| piml_heat_1d_train 1000x9, 20 epochs (end to end) | 2147 | 1.0223 [0.9523, 1.2464] ns/elem | 1.0484 [0.9802, 1.1992] ns/elem | 2.5241 [2.3867, 2.8606] ns/elem | 0.786, 1.026, 1.259 | 1.915, 2.469, 3.004 | 13, 13 | kernel slower, 2.47x |
| q_h on 20 qubits (Gate::apply, a branch per index) | 188 | 2.5635 [2.5287, 2.7355] ns/elem | 2.5719 [2.4937, 2.8334] ns/elem | - | 0.912, 1.003, 1.120 | - | 1, - | status quo, not routed |
| q_h on 20 qubits (strided pair loop, same bits) | 208 | 2.7533 [2.5024, 3.1354] ns/elem | 2.6981 [2.4142, 2.9531] ns/elem | - | 0.770, 0.980, 1.180 | - | 0, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
