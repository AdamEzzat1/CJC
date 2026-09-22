# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: dab5f6c41413ee535df71d2411c11b4db6074899
- dirty tree: false
- dirty paths: (none outside the record directory)
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
- unix time: 1790114917
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 15.0% max 23.2% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): svchost 21%, wmiprvse 14%, claude 12%
- last boot: 9/21/2026 9:47:14 PM (uptime 17.3 h)
- peak RSS at exit: 20744 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 33557 | 0.1530 [0.1516, 0.1602] ns/elem | 0.1544 [0.1495, 0.1689] ns/elem | 0.1944 [0.1879, 0.2008] ns/elem | 0.933, 1.009, 1.114 | 1.173, 1.271, 1.325 | 0, 0 | kernel slower, 1.27x |
| axpy | 24449 | 0.4193 [0.3867, 0.4523] ns/elem | 0.4316 [0.3870, 0.4659] ns/elem | 0.5547 [0.5246, 0.5889] ns/elem | 0.856, 1.029, 1.205 | 1.160, 1.323, 1.523 | 0, 0 | kernel slower, 1.32x |
| dot_kahan | 4777 | 3.1348 [3.0926, 3.8598] ns/elem | 3.1120 [3.0475, 3.3092] ns/elem | 3.0705 [3.0146, 3.6247] ns/elem | 0.790, 0.993, 1.070 | 0.781, 0.979, 1.172 | 0, 0 | inside band (within A/A) |
| mse | 5002 | 3.0615 [3.0092, 3.4032] ns/elem | 3.0873 [3.0474, 3.1659] ns/elem | 3.0580 [3.0106, 3.4883] ns/elem | 0.895, 1.008, 1.052 | 0.885, 0.999, 1.159 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 15267 | 1.9963 [1.8732, 2.1242] ns/elem | 1.9780 [1.8671, 2.3821] ns/elem | 1.2761 [1.2562, 1.7469] ns/elem | 0.879, 0.991, 1.272 | 0.591, 0.639, 0.933 | 0, 0 | kernel faster, 1.56x |
| matmul 128x128x128 | 155 | 3.3157 [3.0228, 3.6791] ns/elem | 3.5223 [3.0413, 3.9464] ns/elem | 3.1666 [2.8817, 3.6561] ns/elem | 0.827, 1.062, 1.306 | 0.783, 0.955, 1.209 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 3776 | 3.1278 [3.0818, 3.8442] ns/elem | 3.2918 [3.2144, 3.6309] ns/elem | 54.7514 [54.2238, 58.5336] ns/elem | 0.836, 1.052, 1.178 | 14.105, 17.505, 18.993 | 0, 0 | kernel slower, 17.50x |
| heat1d_residual_grad 1000x9 | 55555 | 2.6294 [2.3965, 3.3094] ns/elem | 2.6376 [2.4194, 3.1007] ns/elem | 2.4590 [2.2876, 2.6729] ns/elem | 0.731, 1.003, 1.294 | 0.691, 0.935, 1.115 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 5817 | 2.8738 [2.3741, 3.2460] ns/elem | 2.5044 [2.4407, 3.8306] ns/elem | 2.3584 [2.2997, 3.1920] ns/elem | 0.752, 0.871, 1.614 | 0.708, 0.821, 1.345 | 0, 0 | inside band (within A/A) |
| mse_loss_grad | 4916 | 2.8177 [2.6756, 3.4106] ns/elem | 2.8197 [2.6949, 3.1320] ns/elem | 2.4355 [2.3476, 2.7679] ns/elem | 0.790, 1.001, 1.171 | 0.688, 0.864, 1.035 | 0, 0 | inside band (within A/A) |
| mse+grad via GradGraph (status quo) | 614 | 16.6151 [15.3876, 19.9595] ns/elem | 16.0280 [15.5374, 19.6578] ns/elem | - | 0.778, 0.965, 1.278 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
