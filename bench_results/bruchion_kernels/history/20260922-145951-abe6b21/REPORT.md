# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: abe6b21d8cc4917ac8edef096ab1e20915442612
- dirty tree: false
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
- unix time: 1790109699
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 12.5% max 24.4% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): claude 15%, svchost 10%, wmiprvse 9%
- last boot: 9/21/2026 9:47:14 PM (uptime 15.9 h)
- peak RSS at exit: 20740 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 63694 | 0.1594 [0.1524, 0.1742] ns/elem | 0.1537 [0.1517, 0.1831] ns/elem | 0.2018 [0.1925, 0.2044] ns/elem | 0.871, 0.964, 1.201 | 1.105, 1.266, 1.340 | 0, 0 | kernel slower, 1.27x |
| axpy | 27100 | 0.4078 [0.3892, 0.4270] ns/elem | 0.3970 [0.3894, 0.4435] ns/elem | 0.5337 [0.5227, 0.5879] ns/elem | 0.912, 0.973, 1.140 | 1.224, 1.309, 1.511 | 0, 0 | kernel slower, 1.31x |
| dot_kahan | 4997 | 3.1021 [3.0442, 3.1030] ns/elem | 3.0893 [3.0588, 3.1018] ns/elem | 3.0548 [3.0406, 3.1059] ns/elem | 0.986, 0.996, 1.019 | 0.980, 0.985, 1.020 | 0, 0 | inside band (within A/A) |
| mse | 4837 | 3.0645 [3.0292, 3.1479] ns/elem | 3.0644 [3.0358, 3.0948] ns/elem | 3.0564 [3.0124, 3.1025] ns/elem | 0.964, 1.000, 1.022 | 0.957, 0.997, 1.024 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 14619 | 1.9696 [1.7692, 2.0242] ns/elem | 1.8351 [1.7254, 1.9730] ns/elem | 1.2967 [1.2215, 1.3933] ns/elem | 0.852, 0.932, 1.115 | 0.603, 0.658, 0.788 | 0, 0 | kernel faster, 1.52x |
| matmul 128x128x128 | 157 | 2.9776 [2.8303, 3.0391] ns/elem | 2.9367 [2.8667, 3.0710] ns/elem | 2.7601 [2.7015, 2.8232] ns/elem | 0.943, 0.986, 1.085 | 0.889, 0.927, 0.997 | 0, 0 | kernel faster, 1.08x (within A/A) |
| adam_step (t = 7) | 4533 | 2.9488 [2.8554, 3.0211] ns/elem | 2.8419 [2.8148, 3.0923] ns/elem | 52.6276 [52.5364, 56.0754] ns/elem | 0.932, 0.964, 1.083 | 17.390, 17.847, 19.638 | 0, 0 | kernel slower, 17.85x |
| heat1d_residual_grad 1000x9 | 46728 | 2.8048 [2.4887, 3.0639] ns/elem | 2.6315 [2.4837, 3.4594] ns/elem | 2.4956 [2.3590, 3.1157] ns/elem | 0.811, 0.938, 1.390 | 0.770, 0.890, 1.252 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 6082 | 2.5638 [2.5326, 3.2039] ns/elem | 2.7484 [2.5235, 3.1562] ns/elem | 2.5027 [2.4125, 3.1896] ns/elem | 0.788, 1.072, 1.246 | 0.753, 0.976, 1.259 | 0, 0 | inside band (within A/A) |
| mse_loss_grad | 5319 | 2.8422 [2.5303, 2.9809] ns/elem | 2.7810 [2.5788, 3.1735] ns/elem | 2.4760 [2.3477, 2.9654] ns/elem | 0.865, 0.978, 1.254 | 0.788, 0.871, 1.172 | 0, 0 | inside band (within A/A) |
| mse+grad via GradGraph (status quo) | 585 | 14.6931 [11.5168, 15.9668] ns/elem | 13.8581 [11.8160, 14.9562] ns/elem | - | 0.740, 0.943, 1.299 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
