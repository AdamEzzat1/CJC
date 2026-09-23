# Bruchion kernels: the CJC-side record

The Rust fallback path against the Bruchion kernel path, routed through `runtime_policy::set_bruchion_kernels`, on the same buffers, in one process. Every row's arms produced the same bits (asserted). The hard wall: nothing measured here feeds a hash, a decision, or a stable field.

## Provenance

- repo: C:/Users/adame/CJC
- branch: bruchion-kernels-m1
- commit: 59865b2aba5d5c63c7695686822eb3a18010e8bb
- dirty tree: false
- dirty paths: (none outside the record directory)
- rustc: rustc 1.97.1 (8bab26f4f 2026-07-14); host: x86_64-pc-windows-msvc
- profile: release
- target: windows-x86_64
- logical cores: 8
- runtime policy: runtime_policy: thermal=balanced threads=4 batch=128 audit=full numeric=kahan determinism=strict adaptive=true bruchion_kernel_ops=0xfff bruchion_kernels=false
- feature bruchion-kernels: true
- routing set, kernel arm: axpy, dot_kahan, relu, mse, mse_grad, mse_grad_scaled, matmul, matmul_tiled, adam_step, powi, heat1d_residual, heat1d_residual_grad
- routing set, default: dot_kahan, mse, mse_grad, matmul
- BRUCHION_KERNELS_DIR: C:\Users\adame\AppData\Local\Temp\claude\C--Users-adame-Bruchion--claude-worktrees-bruchion-cjc-kernel-substrate-fcbb2e\3fecc690-652f-4f32-9887-52dc9c9ff71a\scratchpad\k13
- kernel_sha256: 83208dd7b5cee97f112e278d59510ef30de63df0b3d951d0c21da3016ae8aec8
- f64::powi is binary exponentiation on this target: false
- elements per call (n): 65536
- protocol: iterations calibrated once on the fallback arm to ~500000 us per phase (max 1000000); 1 warm-up phase(s) per arm; 5 measured phases, arms interleaved A1/A2/B; statistic: ns per call, median [min, max] over phases
- seed: 42
- unix time: 1790176949
- launcher: bench/bruchion_kernels_bench/run.ps1 (PowerShell, load-gated)
- load gate: total CPU avg 14.2% max 17.6% over 20 s (thresholds 15% / 25%); busiest processes (% of one core): claude 26%, svchost 25%, wmiprvse 17%
- last boot: 9/21/2026 9:47:14 PM (uptime 34.6 h)
- peak RSS at exit: 93416 KiB

## Results

`median [min, max]` per arm over the measured phases; ratio band = kernel / fallback with the most conservative bounds the two bands allow; A/A band = the second fallback arm over the first. A row is **faster** only when the whole kernel band sits below 1.0 and **slower** only when it sits above; "inside band" otherwise. "within A/A" means the kernel's median ratio is no further from 1 than the A/A band reaches, so this run cannot tell the arms apart.

"allocs/call" is the number of heap allocations one call makes on the fallback arm and on the kernel arm (counted by the process's global allocator). A row marked *status quo* is not routed: it is the unfused chain the row above it replaces, timed on the fallback arms only, and it is read against that row, not as a kernel ratio.

| workload | iters/phase | fallback | fallback (A/A) | kernel | A/A band | kernel band (lo, med, hi) | allocs/call (fallback, kernel) | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| relu | 30303 | 0.1645 [0.1603, 0.1909] ns/elem | 0.1639 [0.1613, 0.1746] ns/elem | 0.2031 [0.2013, 0.2147] ns/elem | 0.845, 0.996, 1.089 | 1.055, 1.234, 1.340 | 0, 0 | kernel slower, 1.23x |
| axpy | 9689 | 0.4097 [0.3967, 0.4383] ns/elem | 0.4185 [0.4145, 0.4652] ns/elem | 0.6137 [0.5516, 0.6522] ns/elem | 0.946, 1.021, 1.173 | 1.259, 1.498, 1.644 | 0, 0 | kernel slower, 1.50x |
| dot_kahan | 2086 | 3.1288 [3.1177, 3.1381] ns/elem | 3.1445 [3.1047, 3.2609] ns/elem | 3.0921 [3.0733, 3.1072] ns/elem | 0.989, 1.005, 1.046 | 0.979, 0.988, 0.997 | 0, 0 | kernel faster, 1.01x (within A/A) |
| mse | 2452 | 3.1696 [3.1041, 3.8452] ns/elem | 3.1397 [3.1067, 3.3121] ns/elem | 3.1297 [3.0909, 3.6620] ns/elem | 0.808, 0.991, 1.067 | 0.804, 0.987, 1.180 | 0, 0 | inside band (within A/A) |
| matmul 64x17x33 | 7225 | 2.1348 [1.9015, 2.2916] ns/elem | 2.1087 [1.8224, 2.2855] ns/elem | 1.5238 [1.2874, 1.5990] ns/elem | 0.795, 0.988, 1.202 | 0.562, 0.714, 0.841 | 0, 0 | kernel faster, 1.40x |
| matmul 128x128x128 | 83 | 2.9105 [2.8898, 2.9382] ns/elem | 2.9002 [2.8838, 2.9848] ns/elem | 2.8787 [2.8203, 2.9307] ns/elem | 0.981, 0.996, 1.033 | 0.960, 0.989, 1.014 | 0, 0 | inside band (within A/A) |
| matmul 256x17x33 | 1449 | 1.8040 [1.7668, 2.0198] ns/elem | 1.7819 [1.7382, 1.8823] ns/elem | 1.2488 [1.2311, 1.3839] ns/elem | 0.861, 0.988, 1.065 | 0.610, 0.692, 0.783 | 0, 0 | kernel faster, 1.44x |
| matmul 33x64x17 | 5494 | 2.5910 [2.5625, 2.6931] ns/elem | 2.6019 [2.5400, 2.6495] ns/elem | 2.2890 [2.1847, 2.4464] ns/elem | 0.943, 1.004, 1.034 | 0.811, 0.883, 0.955 | 0, 0 | kernel faster, 1.13x |
| matmul 17x33x64 | 4897 | 2.4256 [2.4058, 2.5388] ns/elem | 2.4862 [2.4077, 2.5223] ns/elem | 2.0231 [1.9847, 2.1383] ns/elem | 0.948, 1.025, 1.048 | 0.782, 0.834, 0.889 | 0, 0 | kernel faster, 1.20x |
| matmul 32x32x32 | 5165 | 2.3885 [2.3134, 2.4474] ns/elem | 2.3671 [2.3291, 2.5210] ns/elem | 2.0335 [1.9112, 2.1527] ns/elem | 0.952, 0.991, 1.090 | 0.781, 0.851, 0.931 | 0, 0 | kernel faster, 1.17x |
| matmul 64x64x8 | 5181 | 2.6648 [2.5678, 2.7224] ns/elem | 2.6219 [2.5589, 2.6910] ns/elem | 2.2950 [2.2427, 2.3708] ns/elem | 0.940, 0.984, 1.048 | 0.824, 0.861, 0.923 | 0, 0 | kernel faster, 1.16x |
| matmul 16x16x1024 | 929 | 2.2168 [2.1660, 2.3199] ns/elem | 2.1611 [2.1252, 2.2243] ns/elem | 1.6675 [1.6512, 1.7946] ns/elem | 0.916, 0.975, 1.027 | 0.712, 0.752, 0.829 | 0, 0 | kernel faster, 1.33x |
| matmul 64x4x33 | 33112 | 1.2851 [1.2570, 1.4594] ns/elem | 1.2651 [1.2565, 1.4477] ns/elem | 0.9049 [0.8910, 0.9765] ns/elem | 0.861, 0.984, 1.152 | 0.610, 0.704, 0.777 | 0, 0 | kernel faster, 1.42x |
| matmul 64x8x33 | 20325 | 1.4319 [1.3424, 1.6111] ns/elem | 1.3603 [1.3337, 1.5476] ns/elem | 0.9816 [0.9670, 1.0839] ns/elem | 0.828, 0.950, 1.153 | 0.600, 0.686, 0.807 | 0, 0 | kernel faster, 1.46x |
| matmul 64x16x33 | 4152 | 2.2996 [2.2554, 2.5566] ns/elem | 2.3050 [2.2990, 2.5830] ns/elem | 1.5753 [1.5287, 1.5949] ns/elem | 0.899, 1.002, 1.145 | 0.598, 0.685, 0.707 | 0, 0 | kernel faster, 1.46x |
| matmul 64x24x33 | 3408 | 2.6006 [2.4209, 2.6282] ns/elem | 2.6187 [2.4112, 2.8577] ns/elem | 2.0013 [1.9303, 2.1321] ns/elem | 0.917, 1.007, 1.180 | 0.734, 0.770, 0.881 | 0, 0 | kernel faster, 1.30x |
| matmul 64x32x33 | 3121 | 2.5093 [2.4527, 2.5883] ns/elem | 2.4890 [2.4432, 2.5117] ns/elem | 2.2413 [2.0883, 2.2953] ns/elem | 0.944, 0.992, 1.024 | 0.807, 0.893, 0.936 | 0, 0 | kernel faster, 1.12x |
| matmul 64x48x33 | 1267 | 2.5960 [2.5650, 2.7822] ns/elem | 2.8752 [2.5628, 3.0331] ns/elem | 2.3421 [2.3088, 2.4559] ns/elem | 0.921, 1.108, 1.183 | 0.830, 0.902, 0.957 | 0, 0 | kernel faster, 1.11x (within A/A) |
| matmul 64x64x33 | 1328 | 2.7001 [2.6887, 2.8699] ns/elem | 2.8450 [2.7012, 2.8865] ns/elem | 2.5478 [2.5080, 2.8406] ns/elem | 0.941, 1.054, 1.074 | 0.874, 0.944, 1.056 | 0, 0 | inside band (within A/A) |
| matmul 64x128x33 | 663 | 3.2520 [2.9858, 3.8581] ns/elem | 3.2259 [3.0100, 3.8284] ns/elem | 3.2086 [2.8873, 3.9729] ns/elem | 0.780, 0.992, 1.282 | 0.748, 0.987, 1.331 | 0, 0 | inside band (within A/A) |
| matmul 8x17x33 | 49504 | 2.0099 [1.7715, 3.6080] ns/elem | 1.8629 [1.7881, 3.8688] ns/elem | 1.2762 [1.2578, 2.1738] ns/elem | 0.496, 0.927, 2.184 | 0.349, 0.635, 1.227 | 0, 0 | inside band (within A/A) |
| matmul 16x17x33 | 24752 | 1.8068 [1.7683, 1.9543] ns/elem | 1.7900 [1.7491, 1.9540] ns/elem | 1.2739 [1.2403, 1.3100] ns/elem | 0.895, 0.991, 1.105 | 0.635, 0.705, 0.741 | 0, 0 | kernel faster, 1.42x |
| matmul 32x17x33 | 13736 | 1.7880 [1.7772, 2.3301] ns/elem | 1.9151 [1.7833, 2.0046] ns/elem | 1.2867 [1.2536, 1.3284] ns/elem | 0.765, 1.071, 1.128 | 0.538, 0.720, 0.747 | 0, 0 | kernel faster, 1.39x |
| matmul 63x17x33 | 5154 | 1.8384 [1.7735, 1.9415] ns/elem | 1.7887 [1.7516, 1.9617] ns/elem | 1.2788 [1.2367, 1.2800] ns/elem | 0.902, 0.973, 1.106 | 0.637, 0.696, 0.722 | 0, 0 | kernel faster, 1.44x |
| matmul 16x16x16 | 53763 | 1.7610 [1.6665, 2.0383] ns/elem | 1.6891 [1.6570, 2.0061] ns/elem | 1.2994 [1.2178, 1.3176] ns/elem | 0.813, 0.959, 1.204 | 0.597, 0.738, 0.791 | 0, 0 | kernel faster, 1.36x |
| matmul 16x16x64 | 11337 | 2.0008 [1.9693, 2.1943] ns/elem | 2.0099 [1.9950, 2.0150] ns/elem | 1.2629 [1.1999, 1.3752] ns/elem | 0.909, 1.005, 1.023 | 0.547, 0.631, 0.698 | 0, 0 | kernel faster, 1.58x |
| matmul 16x16x256 | 2450 | 2.2718 [2.2383, 2.4410] ns/elem | 2.2682 [2.2559, 2.4667] ns/elem | 1.7498 [1.7337, 2.0364] ns/elem | 0.924, 0.998, 1.102 | 0.710, 0.770, 0.910 | 0, 0 | kernel faster, 1.30x |
| matmul 8x8x8 | 500000 | 1.4442 [1.3724, 1.5077] ns/elem | 1.4169 [1.3578, 1.5547] ns/elem | 1.0956 [1.0596, 1.2098] ns/elem | 0.901, 0.981, 1.133 | 0.703, 0.759, 0.882 | 0, 0 | kernel faster, 1.32x |
| matmul 48x17x33 | 8210 | 1.8888 [1.7887, 2.0655] ns/elem | 1.7872 [1.7788, 2.0170] ns/elem | 1.2952 [1.2619, 1.4727] ns/elem | 0.861, 0.946, 1.128 | 0.611, 0.686, 0.823 | 0, 0 | kernel faster, 1.46x |
| matmul 32x48x16 | 6858 | 2.4948 [2.4510, 2.5985] ns/elem | 2.5131 [2.4678, 2.6554] ns/elem | 2.0715 [2.0603, 2.1091] ns/elem | 0.950, 1.007, 1.083 | 0.793, 0.830, 0.861 | 0, 0 | kernel faster, 1.20x |
| matmul 63x63x63 | 604 | 2.8770 [2.7210, 3.3628] ns/elem | 2.7373 [2.7099, 3.3490] ns/elem | 2.6138 [2.5946, 3.1332] ns/elem | 0.806, 0.951, 1.231 | 0.772, 0.909, 1.151 | 0, 0 | inside band (within A/A) |
| adam_step (t = 7) | 1950 | 3.3996 [3.2126, 4.2535] ns/elem | 3.7079 [3.2593, 4.1311] ns/elem | 5.4562 [5.3477, 6.3177] ns/elem | 0.766, 1.091, 1.286 | 1.257, 1.605, 1.967 | 0, 0 | kernel slower, 1.60x |
| heat1d_residual_grad 1000x9 | 14880 | 2.6881 [2.6180, 2.8156] ns/elem | 2.6765 [2.6186, 2.7808] ns/elem | 2.3448 [2.2682, 2.4373] ns/elem | 0.930, 0.996, 1.062 | 0.806, 0.872, 0.931 | 0, 0 | kernel faster, 1.15x |
| heat1d_residual_grad 8192x9 | 2331 | 2.6979 [2.6663, 3.4963] ns/elem | 2.7509 [2.6775, 3.4846] ns/elem | 2.4513 [2.2915, 3.1109] ns/elem | 0.766, 1.020, 1.307 | 0.655, 0.909, 1.167 | 0, 0 | inside band (within A/A) |
| matmul tiled 128x128x128 | 494 | 0.2182 [0.2130, 0.2635] ns/elem | 0.2188 [0.2138, 0.2572] ns/elem | 0.4159 [0.4116, 0.5251] ns/elem | 0.812, 1.003, 1.208 | 1.562, 1.906, 2.466 | 1, 0 | kernel slower, 1.91x |
| matmul tiled 256x256x256 | 92 | 0.2841 [0.2770, 0.2947] ns/elem | 0.2861 [0.2854, 0.3791] ns/elem | 0.4273 [0.4197, 0.4902] ns/elem | 0.969, 1.007, 1.369 | 1.424, 1.504, 1.770 | 1, 0 | kernel slower, 1.50x |
| relu call path: bare ffi vs kernel::relu_raw | 22123 | 0.2410 [0.2303, 0.2780] ns/elem | 0.2476 [0.2361, 0.3054] ns/elem | 0.2444 [0.2262, 0.2933] ns/elem | 0.849, 1.027, 1.326 | 0.814, 1.014, 1.274 | 0, 0 | inside band (within A/A) |
| relu loop: Rust body vs bare ffi | 34013 | 0.1848 [0.1789, 0.2100] ns/elem | 0.1862 [0.1814, 0.2229] ns/elem | 0.2323 [0.2256, 0.2993] ns/elem | 0.864, 1.008, 1.246 | 1.074, 1.257, 1.673 | 0, 0 | kernel slower, 1.26x |
| mse_loss_grad | 2510 | 1.9706 [1.9527, 2.1741] ns/elem | 2.0132 [1.9470, 2.0817] ns/elem | 1.4151 [1.4049, 1.4575] ns/elem | 0.896, 1.022, 1.066 | 0.646, 0.718, 0.746 | 0, 0 | kernel faster, 1.39x |
| mse+grad via GradGraph, fused mean | 394 | 5.6821 [5.3515, 22.5319] ns/elem | 5.5829 [5.4759, 21.8693] ns/elem | 5.7567 [5.4361, 10.3435] ns/elem | 0.243, 0.983, 4.087 | 0.241, 1.013, 1.933 | 85, 85 | inside band (within A/A) |
| mse+grad via GradGraph, unfused (status quo) | 612 | 16.5313 [14.0686, 18.0633] ns/elem | 15.1743 [13.8747, 18.4625] ns/elem | - | 0.768, 0.918, 1.312 | - | 110, - | status quo, not routed |
| conv1d 4096x8x9 (kernel::conv1d_raw) | 1351 | 1.2928 [1.2579, 1.6005] ns/elem | 1.3155 [1.2491, 1.6449] ns/elem | - | 0.780, 1.018, 1.308 | - | 0, - | status quo, not routed |
| piml_heat_1d_train 1000x9, 20 epochs (end to end) | 640 | 2.8451 [2.7552, 3.1099] ns/elem | 2.9332 [2.7629, 3.0617] ns/elem | 2.4647 [2.4267, 2.5007] ns/elem | 0.888, 1.031, 1.111 | 0.780, 0.866, 0.908 | 13, 13 | kernel faster, 1.15x |
| piml_heat_1d_train 1000x9, 20 epochs (end to end, arguments opaque) | 1098 | 3.0325 [2.8100, 3.6319] ns/elem | 2.9188 [2.8212, 3.7660] ns/elem | 2.5011 [2.4505, 3.0721] ns/elem | 0.777, 0.962, 1.340 | 0.675, 0.825, 1.093 | 13, 13 | inside band (within A/A) |
| q_h on 20 qubits (Gate::apply, a branch per index) | 146 | 2.7973 [2.6448, 3.0112] ns/elem | 2.9008 [2.6650, 3.0075] ns/elem | - | 0.885, 1.037, 1.137 | - | 1, - | status quo, not routed |
| q_h on 20 qubits (strided pair loop, same bits) | 194 | 2.7546 [2.6180, 3.2125] ns/elem | 2.8364 [2.6267, 3.2276] ns/elem | - | 0.818, 1.030, 1.233 | - | 0, - | status quo, not routed |
| Tensor::matmul 8x8x8 (end to end) | 263157 | 2.2251 [2.2069, 2.5385] ns/elem | 2.2763 [2.1885, 2.3266] ns/elem | 1.9135 [1.8827, 2.0380] ns/elem | 0.862, 1.023, 1.054 | 0.742, 0.860, 0.923 | 8, 8 | kernel faster, 1.16x |
| Tensor::matmul 32x32x32 (end to end) | 5382 | 2.8268 [2.6290, 3.9456] ns/elem | 2.7456 [2.6473, 3.9477] ns/elem | 2.2039 [2.1789, 2.8397] ns/elem | 0.671, 0.971, 1.502 | 0.552, 0.780, 1.080 | 8, 8 | inside band (within A/A) |
| Tensor::matmul 48x17x33 (end to end) | 6596 | 2.1024 [2.0552, 2.6338] ns/elem | 2.1422 [2.0415, 2.7423] ns/elem | 1.5575 [1.3818, 2.0494] ns/elem | 0.775, 1.019, 1.334 | 0.525, 0.741, 0.997 | 8, 8 | kernel faster, 1.35x (within A/A) |
| Tensor::matmul 63x63x63 (end to end) | 552 | 3.1232 [3.0171, 3.9647] ns/elem | 3.1162 [3.0344, 3.7616] ns/elem | 2.9661 [2.8051, 3.0836] ns/elem | 0.765, 0.998, 1.247 | 0.708, 0.950, 1.022 | 8, 8 | inside band (within A/A) |
| conv2d 2x3x28x28 * 8x3x3 (kernel::conv2d_raw, one scratch per call) | 117 | 16.2860 [15.2250, 19.8433] ns/elem | 15.7718 [15.2718, 19.7352] ns/elem | - | 0.770, 0.968, 1.296 | - | 0, - | status quo, not routed |
| conv2d 2x3x28x28 * 8x3x3 (an accumulator per output, the form before) | 45 | 37.1228 [36.3596, 37.7018] ns/elem | 36.5648 [36.2663, 38.1025] ns/elem | - | 0.962, 0.985, 1.048 | - | 0, - | status quo, not routed |
| Tensor::linear [64, 128] x [32, 128] (one scratch per call) | 353 | 6.0907 [5.7926, 7.1762] ns/elem | 5.9608 [5.7444, 7.2746] ns/elem | - | 0.800, 0.979, 1.256 | - | 12, - | status quo, not routed |
| Tensor::linear [64, 128] x [32, 128] (an accumulator per output, the form before) | 204 | 9.5626 [9.4156, 12.4943] ns/elem | 10.0193 [9.5411, 11.5953] ns/elem | - | 0.764, 1.048, 1.232 | - | 10, - | status quo, not routed |
| Tensor::sum_axis [256, 64] axis 1 (one scratch per call) | 2895 | 8.3382 [7.7935, 11.0191] ns/elem | 8.1611 [7.7932, 11.9208] ns/elem | - | 0.707, 0.979, 1.530 | - | 9, - | status quo, not routed |
| Tensor::sum_axis [256, 64] axis 1 (an accumulator per output, the form before) | 1522 | 15.5852 [15.3928, 16.8080] ns/elem | 15.8053 [15.3098, 16.6329] ns/elem | - | 0.911, 1.014, 1.081 | - | 6, - | status quo, not routed |
| OptimizerState.step, SGD (ml::optimizer_step_tensors, the gradient borrowed) | 1000000 | 128.7239 [125.4083, 163.0333] ns/elem | 127.5726 [124.1557, 165.5602] ns/elem | - | 0.762, 0.991, 1.320 | - | 6, - | status quo, not routed |
| OptimizerState.step, SGD (both tensors copied, the form before) | 1000000 | 142.6882 [139.9817, 181.4255] ns/elem | 142.4932 [137.7262, 157.7543] ns/elem | - | 0.759, 0.999, 1.127 | - | 7, - | status quo, not routed |
| GradGraph zero_grad + backward + clip_grad, 2 parameters (in place) | 104166 | 1284.8439 [1259.8117, 1407.8015] ns/elem | 1268.5128 [1232.9892, 1362.9634] ns/elem | - | 0.876, 0.987, 1.082 | - | 60, - | status quo, not routed |

## Reading it

- Within-machine ratios only; never compare absolute numbers across machines or across runs on a machine whose load was not gated.
- A verdict inside the A/A band is not a result either way.
- The `adam_step` row compares the dispatch entry, where `1 - beta^t` is hoisted on both arms; `ml::adam_step`'s fallback recomputes `powf` per element, a CJC-side cost this row does not charge to either side.
- Digests (FNV-1a over the outputs' bits) are in `rows.jsonl`; per-phase timings in `phases.csv`.
