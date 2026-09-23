# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: 28c104cfad8c4d88204028764be2a08961f2caaa
- dirty tree: false
- dirty paths: (none outside the record directory)
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernels=false
- feature bruchion-kernels: true
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k11
- kernel_sha256: 5c897aa07c64ce07f1517736dda9f78247485315b95db202a5dbcd9df794432c
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~1000000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790119622
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 10.2% max 15.7% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): svchost 21%, wmiprvse 15%, claude 8%
- last boot: 9/21/2026 9:47:14 PM (uptime 18.7 h)
- peak RSS at exit: 24856 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 62893 | 0.1536 [0.1519, 0.1737] ns/elem | 0.1530 [0.1508, 0.1663] ns/elem | 0.1989 [0.1882, 0.2030] ns/elem | 0.868, 0.996, 1.095 | 1.084, 1.296, 1.336 | 0, 0 | kernel slower, 1.30x |
| axpy | 24449 | 0.4688 [0.3946, 0.5809] ns/elem | 0.4754 [0.4425, 0.5527] ns/elem | 0.6403 [0.6184, 0.6756] ns/elem | 0.762, 1.014, 1.400 | 1.065, 1.366, 1.712 | 0, 0 | kernel slower, 1.37x (within A/A) |
| dot_kahan | 4387 | 3.0892 [3.0490, 3.2482] ns/elem | 3.0683 [3.0031, 3.4176] ns/elem | 3.0739 [2.9683, 3.6891] ns/elem | 0.925, 0.993, 1.121 | 0.914, 0.995, 1.210 | 0, 0 | inside band (within A/A) |
| mse | 5230 | 3.0336 [3.0301, 3.5578] ns/elem | 3.0111 [2.9986, 3.7420] ns/elem | 3.0369 [3.0045, 3.6370] ns/elem | 0.843, 0.993, 1.235 | 0.844, 1.001, 1.200 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 11904 | 1.8908 [1.7621, 2.5131] ns/elem | 2.0643 [1.7473, 2.4301] ns/elem | 1.4630 [1.2414, 1.5948] ns/elem | 0.695, 1.092, 1.379 | 0.494, 0.774, 0.905 | 0, 0 | kernel faster, 1.29x (within A/A) |
| matmul 128x128x128 | 130 | 3.1082 [3.0373, 3.6845] ns/elem | 3.0886 [2.9936, 3.1838] ns/elem | 2.9797 [2.8426, 3.5118] ns/elem | 0.812, 0.994, 1.048 | 0.772, 0.959, 1.156 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 3921 | 3.0779 [2.9249, 5.9803] ns/elem | 2.9654 [2.8630, 4.6749] ns/elem | 54.8548 [54.2812, 73.5170] ns/elem | 0.479, 0.963, 1.598 | 9.077, 17.822, 25.135 | 0, 0 | kernel slower, 17.82x |
| heat1d_residual_grad 1000x9 | 48076 | 3.4468 [3.1725, 3.7733] ns/elem | 3.7626 [2.7086, 5.6703] ns/elem | 3.5055 [2.4068, 3.7501] ns/elem | 0.718, 1.092, 1.787 | 0.638, 1.017, 1.182 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 5405 | 3.7938 [3.3517, 4.6836] ns/elem | 3.8896 [3.2025, 4.5853] ns/elem | 3.3371 [2.7058, 3.9519] ns/elem | 0.684, 1.025, 1.368 | 0.578, 0.880, 1.179 | 0, 0 | inside band (within A/A) |
| matmul tiled 128x128x128 | 2002 | 0.3376 [0.3030, 0.4078] ns/elem | 0.3534 [0.2951, 0.3717] ns/elem | 0.5635 [0.4543, 0.6615] ns/elem | 0.724, 1.047, 1.227 | 1.114, 1.669, 2.183 | 1, 0 | kernel slower, 1.67x |
| matmul tiled 256x256x256 | 164 | 0.3512 [0.3113, 0.4639] ns/elem | 0.3061 [0.2781, 0.5075] ns/elem | 0.5539 [0.5009, 0.7301] ns/elem | 0.599, 0.872, 1.630 | 1.080, 1.577, 2.345 | 1, 0 | kernel slower, 1.58x (within A/A) |
| relu call path: bare ffi vs kernel::relu_raw | 29069 | 0.2347 [0.2168, 0.3284] ns/elem | 0.2373 [0.2119, 0.2974] ns/elem | 0.2251 [0.2142, 0.2341] ns/elem | 0.645, 1.011, 1.372 | 0.652, 0.959, 1.080 | 0, 0 | inside band (within A/A) |
| relu loop: Rust body vs bare ffi | 50000 | 0.1973 [0.1702, 0.2416] ns/elem | 0.2173 [0.1717, 0.2631] ns/elem | 0.2287 [0.2230, 0.3145] ns/elem | 0.711, 1.101, 1.546 | 0.923, 1.159, 1.848 | 0, 0 | inside band (within A/A) |
| mse_loss_grad | 4576 | 2.1243 [1.8621, 2.3390] ns/elem | 2.0396 [1.8978, 2.5416] ns/elem | 1.4207 [1.3276, 1.9900] ns/elem | 0.811, 0.960, 1.365 | 0.568, 0.669, 1.069 | 0, 0 | inside band (within A/A) |
| mse+grad via GradGraph (status quo) | 540 | 24.4017 [20.5273, 27.4229] ns/elem | 21.7395 [20.0123, 25.5018] ns/elem | - | 0.730, 0.891, 1.242 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
