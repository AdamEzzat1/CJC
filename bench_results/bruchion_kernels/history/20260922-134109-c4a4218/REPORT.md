# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: c4a421860fb96faac1b9ad25f648240d2b11eb72
- dirty tree: true
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernels=false
- feature bruchion-kernels: true
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k8
- kernel_sha256: 87265ae8e7636392ab514b8895c8ce88bb71deaf923142fb62dab6fbf3a2ff91
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~500000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790106709
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 13.8% max 23.3% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): svchost 29%, wmiprvse 21%, claude 8%
- last boot: 9/21/2026 9:47:14 PM (uptime 15.1 h)
- peak RSS at exit: 14812 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| relu | 26041 | 0.1535 [0.1514, 0.1658] ns/elem | 0.1602 [0.1534, 0.1684] ns/elem | 0.1954 [0.1908, 0.2080] ns/elem | 0.925, 1.044, 1.112 | 1.151, 1.273, 1.374 | kernel slower, 1.27x |
| axpy | 14084 | 0.4942 [0.3885, 0.5821] ns/elem | 0.5128 [0.3876, 0.5850] ns/elem | 0.6261 [0.5547, 0.7649] ns/elem | 0.666, 1.038, 1.506 | 0.953, 1.267, 1.969 | inside band (within A/A) |
| dot_kahan | 2054 | 3.0876 [3.0680, 3.1340] ns/elem | 3.1434 [3.0452, 3.4388] ns/elem | 3.1543 [3.0572, 3.2567] ns/elem | 0.972, 1.018, 1.121 | 0.976, 1.022, 1.062 | inside band (within A/A) |
| mse | 2145 | 3.1100 [3.0902, 3.5672] ns/elem | 3.1761 [3.0889, 3.4874] ns/elem | 3.1007 [3.0563, 3.2945] ns/elem | 0.866, 1.021, 1.129 | 0.857, 0.997, 1.066 | inside band (within A/A) |
| matmul 64x17x33 | 7751 | 1.9575 [1.8142, 2.5207] ns/elem | 2.0679 [1.8635, 2.4339] ns/elem | 1.3206 [1.3063, 1.9503] ns/elem | 0.739, 1.056, 1.342 | 0.518, 0.675, 1.075 | inside band (within A/A) |
| matmul 128x128x128 | 82 | 3.1497 [3.0872, 3.9117] ns/elem | 3.1154 [3.0092, 3.9407] ns/elem | 3.0969 [2.9164, 3.3977] ns/elem | 0.769, 0.989, 1.276 | 0.746, 0.983, 1.101 | inside band (within A/A) |
| adam_step (t = 7) | 2086 | 4.4934 [3.6760, 10.0740] ns/elem | 5.3854 [3.9075, 7.2259] ns/elem | 89.5615 [68.3743, 91.0248] ns/elem | 0.388, 1.199, 1.966 | 6.787, 19.932, 24.762 | kernel slower, 19.93x |
| heat1d_residual_grad 1000x9 | 26455 | 3.9881 [3.4694, 6.3577] ns/elem | 3.9568 [3.1870, 6.3265] ns/elem | 4.7994 [3.1496, 5.9557] ns/elem | 0.501, 0.992, 1.824 | 0.495, 1.203, 1.717 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 1777 | 6.1837 [3.7719, 10.3252] ns/elem | 4.9395 [3.8842, 9.8939] ns/elem | 4.8234 [3.6832, 8.9551] ns/elem | 0.376, 0.799, 2.623 | 0.357, 0.780, 2.374 | inside band (within A/A) |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
