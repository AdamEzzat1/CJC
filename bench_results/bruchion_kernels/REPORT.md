# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: 1ce0e1581589849ee28a6fec4a45ca21f0c3b19c
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
- unix time: 1790118944
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 14.5% max 23.5% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): claude 20%, cefsharp.browsersubprocess 9%, svchost 8%
- last boot: 9/21/2026 9:47:14 PM (uptime 18.5 h)
- peak RSS at exit: 24856 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 54054 | 0.1550 [0.1517, 0.1780] ns/elem | 0.1570 [0.1552, 0.1707] ns/elem | 0.1978 [0.1907, 0.2103] ns/elem | 0.872, 1.013, 1.125 | 1.071, 1.276, 1.386 | 0, 0 | kernel slower, 1.28x |
| axpy | 22935 | 0.3897 [0.3861, 0.4462] ns/elem | 0.3972 [0.3914, 0.4543] ns/elem | 0.5450 [0.5339, 0.5649] ns/elem | 0.877, 1.019, 1.177 | 1.196, 1.398, 1.463 | 0, 0 | kernel slower, 1.40x |
| dot_kahan | 5022 | 3.0910 [3.0863, 3.1950] ns/elem | 3.1183 [3.0375, 3.1445] ns/elem | 3.0793 [3.0074, 3.1474] ns/elem | 0.951, 1.009, 1.019 | 0.941, 0.996, 1.020 | 0, 0 | inside band (within A/A) |
| mse | 4980 | 3.0759 [3.0333, 3.1153] ns/elem | 3.0754 [3.0393, 3.2119] ns/elem | 3.0408 [3.0184, 3.0772] ns/elem | 0.976, 1.000, 1.059 | 0.969, 0.989, 1.014 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 14471 | 1.7473 [1.7203, 1.7984] ns/elem | 1.7700 [1.7302, 1.8775] ns/elem | 1.3046 [1.2415, 1.3454] ns/elem | 0.962, 1.013, 1.091 | 0.690, 0.747, 0.782 | 0, 0 | kernel faster, 1.34x |
| matmul 128x128x128 | 161 | 2.8920 [2.8649, 3.2345] ns/elem | 2.9706 [2.9023, 2.9951] ns/elem | 2.8566 [2.7814, 2.8996] ns/elem | 0.897, 1.027, 1.045 | 0.860, 0.988, 1.012 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 4302 | 3.2207 [2.8804, 3.7271] ns/elem | 3.1933 [2.8762, 3.6708] ns/elem | 55.5238 [52.8271, 68.2678] ns/elem | 0.772, 0.992, 1.274 | 14.174, 17.240, 23.701 | 0, 0 | kernel slower, 17.24x |
| heat1d_residual_grad 1000x9 | 50761 | 2.7568 [2.7064, 3.0485] ns/elem | 2.9901 [2.6775, 3.6978] ns/elem | 2.3786 [2.2114, 2.8910] ns/elem | 0.878, 1.085, 1.366 | 0.725, 0.863, 1.068 | 0, 0 | inside band (within A/A) |
| heat1d_residual_grad 8192x9 | 5408 | 3.2176 [2.7626, 4.0006] ns/elem | 2.9249 [2.6936, 3.4387] ns/elem | 2.5484 [2.2745, 2.9799] ns/elem | 0.673, 0.909, 1.245 | 0.569, 0.792, 1.079 | 0, 0 | inside band (within A/A) |
| matmul tiled 128x128x128 | 2061 | 0.2754 [0.2578, 0.3220] ns/elem | 0.2829 [0.2632, 0.3290] ns/elem | 0.4031 [0.3956, 0.4166] ns/elem | 0.817, 1.027, 1.276 | 1.228, 1.464, 1.616 | 1, 0 | kernel slower, 1.46x |
| matmul tiled 256x256x256 | 189 | 0.2652 [0.2552, 0.3483] ns/elem | 0.2689 [0.2478, 0.2996] ns/elem | 0.4138 [0.3988, 0.4581] ns/elem | 0.711, 1.014, 1.174 | 1.145, 1.561, 1.795 | 1, 0 | kernel slower, 1.56x |
| relu call path: bare ffi | kernel::relu_raw | 42016 | 0.2442 [0.2160, 0.2942] ns/elem | 0.2259 [0.2121, 0.2637] ns/elem | 0.2314 [0.2135, 0.2733] ns/elem | 0.721, 0.925, 1.221 | 0.726, 0.947, 1.265 | 0, 0 | inside band (within A/A) |
| relu loop: Rust body | bare ffi | 53191 | 0.1841 [0.1716, 0.2290] ns/elem | 0.1824 [0.1701, 0.2219] ns/elem | 0.2404 [0.2152, 0.2668] ns/elem | 0.743, 0.991, 1.293 | 0.940, 1.306, 1.555 | 0, 0 | inside band |
| mse_loss_grad | 7183 | 2.0963 [1.9511, 2.5221] ns/elem | 2.2387 [1.8986, 2.6076] ns/elem | 1.5189 [1.4762, 1.6829] ns/elem | 0.753, 1.068, 1.336 | 0.585, 0.725, 0.863 | 0, 0 | kernel faster, 1.38x (within A/A) |
| mse+grad via GradGraph (status quo) | 505 | 22.2147 [21.7581, 28.3054] ns/elem | 20.8575 [20.4587, 24.1724] ns/elem | - | 0.723, 0.939, 1.111 | - | 122, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
