# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: 1cf178b62619ddac1089df9814919a3b86bea36b
- dirty tree: false
- dirty paths: (none outside the record directory)
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernels=false
- feature bruchion-kernels: true
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k12
- kernel_sha256: f76281e378749f8f6097b0db0aafc0355995bf0a8af10daeae69f211b79f4298
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~1000000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790123770
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 12.4% max 16.5% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): claude 26%, svchost 12%, msmpeng 9%
- last boot: 9/21/2026 9:47:14 PM (uptime 19.8 h)
- peak RSS at exit: 24716 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 36630 | 0.1767 [0.1727, 0.2258] ns/elem | 0.1846 [0.1706, 0.2336] ns/elem | 0.2330 [0.2196, 0.2779] ns/elem | 0.755, 1.045, 1.353 | 0.973, 1.319, 1.609 | 0, 0 | inside band (within A/A) |
| axpy | 20703 | 0.4857 [0.4499, 0.5859] ns/elem | 0.4900 [0.4685, 0.5362] ns/elem | 0.6310 [0.6076, 0.7642] ns/elem | 0.800, 1.009, 1.192 | 1.037, 1.299, 1.699 | 0, 0 | kernel slower, 1.30x |
| dot_kahan | 4185 | 3.1564 [3.1222, 3.7780] ns/elem | 3.1378 [3.0468, 3.2087] ns/elem | 3.1253 [3.0211, 3.8771] ns/elem | 0.806, 0.994, 1.028 | 0.800, 0.990, 1.242 | 0, 0 | inside band (within A/A) |
| mse | 4965 | 3.0847 [3.0682, 3.9751] ns/elem | 3.1352 [3.0900, 3.4973] ns/elem | 3.1418 [3.0335, 3.2892] ns/elem | 0.777, 1.016, 1.140 | 0.763, 1.019, 1.072 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 9756 | 1.9215 [1.8417, 2.6013] ns/elem | 2.0537 [1.9173, 2.4062] ns/elem | 1.3714 [1.3130, 1.7088] ns/elem | 0.737, 1.069, 1.307 | 0.505, 0.714, 0.928 | 0, 0 | kernel faster, 1.40x (within A/A) |
| matmul 128x128x128 | 154 | 3.1530 [3.1296, 3.9452] ns/elem | 3.2939 [3.2351, 3.4443] ns/elem | 3.0538 [2.9694, 3.6242] ns/elem | 0.820, 1.045, 1.101 | 0.753, 0.969, 1.158 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 3749 | 3.3705 [3.2204, 3.9292] ns/elem | 3.3577 [3.2422, 3.3819] ns/elem | 5.4118 [5.2955, 6.5474] ns/elem | 0.825, 0.996, 1.050 | 1.348, 1.606, 2.033 | 0, 0 | kernel slower, 1.61x |
| heat1d_residual_grad 1000x9 | 43668 | 2.6961 [2.4741, 3.1607] ns/elem | 2.8903 [2.7522, 3.4468] ns/elem | 2.4934 [2.2748, 3.0448] ns/elem | 0.871, 1.072, 1.393 | 0.720, 0.925, 1.231 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 6009 | 2.7250 [2.5152, 3.3843] ns/elem | 2.5997 [2.5702, 2.7879] ns/elem | 2.3143 [2.2435, 2.6032] ns/elem | 0.759, 0.954, 1.108 | 0.663, 0.849, 1.035 | 0, 0 | inside band (within A/A) |
| matmul tiled 128x128x128 | 1975 | 0.2762 [0.2600, 0.2852] ns/elem | 0.2838 [0.2675, 0.3405] ns/elem | 0.4054 [0.3863, 0.4178] ns/elem | 0.938, 1.027, 1.310 | 1.355, 1.468, 1.607 | 1, 0 | kernel slower, 1.47x |
| matmul tiled 256x256x256 | 199 | 0.2726 [0.2638, 0.3410] ns/elem | 0.2784 [0.2622, 0.3308] ns/elem | 0.4206 [0.3978, 0.4299] ns/elem | 0.769, 1.021, 1.254 | 1.167, 1.543, 1.630 | 1, 0 | kernel slower, 1.54x |
| relu call path: bare ffi vs kernel::relu_raw | 47169 | 0.2315 [0.2186, 0.2589] ns/elem | 0.2221 [0.2153, 0.2815] ns/elem | 0.2246 [0.2122, 0.2851] ns/elem | 0.832, 0.960, 1.288 | 0.820, 0.970, 1.304 | 0, 0 | inside band (within A/A) |
| relu loop: Rust body vs bare ffi | 54644 | 0.1840 [0.1727, 0.2196] ns/elem | 0.1836 [0.1740, 0.2266] ns/elem | 0.2441 [0.2161, 0.2538] ns/elem | 0.793, 0.998, 1.312 | 0.984, 1.327, 1.469 | 0, 0 | inside band |
| mse_loss_grad | 5555 | 1.9176 [1.8540, 2.4998] ns/elem | 2.0132 [1.8670, 2.2216] ns/elem | 1.5005 [1.4875, 1.8094] ns/elem | 0.747, 1.050, 1.198 | 0.595, 0.782, 0.976 | 0, 0 | kernel faster, 1.28x (within A/A) |
| mse+grad via GradGraph (status quo) | 593 | 23.6569 [22.2967, 27.0963] ns/elem | 23.5025 [22.3323, 25.7040] ns/elem | - | 0.824, 0.993, 1.153 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
