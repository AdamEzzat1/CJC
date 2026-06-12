# Phase D — silicon diagnostics report

Wall-clock + peak-RSS A/B of CANA plan choices. All subjects passed
the determinism gates (output byte-equality across AST-eval and both
arms; corpus program-hash, plan and modeled-energy identity where
applicable) BEFORE any timing was read. Ratios are arm B / arm A,
lower = B better. Bands are `median [min, max]` over the measured
phases; a verdict is only WIN/REGRESSION when the entire conservative
ratio band clears 1.0.

Protocol: 1 warm-up + 5 measured phases per arm, interleaved A/B; ~5.0 s sustained-load target per phase; fresh child process per phase; per-phase iteration count calibrated once on arm A.

Measured on one Windows machine; wall-clock and peak RSS only (CPU
frequency/temperature are out of MVP scope per the research doc §3
signal-reality audit). Within-machine deltas only — never compare
absolute numbers across machines.

## Family: selector

| subject | arms (A vs B) | plans differ | modeled B/A | iters | wall A µs/run | wall B µs/run | wall ratio B/A | wall verdict | RSS A KB | RSS B KB | RSS ratio B/A | RSS verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem_grad_a1 | baseline vs selector_rec | yes | 0.49855 | 16025 | 638.0 [424.5, 1168.6] | 426.5 [140.8, 538.3] | 0.6684 [0.1205, 1.2681] | inconclusive | 5516.0 [5504.0, 5520.0] | 5512.0 [5504.0, 5512.0] | 0.9993 [0.9971, 1.0015] | inconclusive |
| mem_grad_a2 | baseline vs selector_rec | yes | 0.49673 | 3309 | 1939.6 [1812.9, 2121.1] | 708.5 [590.4, 897.7] | 0.3653 [0.2783, 0.4952] | WIN | 5528.0 [5520.0, 5548.0] | 5520.0 [5516.0, 5532.0] | 0.9986 [0.9942, 1.0022] | inconclusive |
| mem_grad_a3 | baseline vs selector_rec | yes | 0.49628 | 608 | 5668.8 [5052.0, 6445.5] | 2103.8 [1574.6, 2471.9] | 0.3711 [0.2443, 0.4893] | WIN | 5472.0 [5464.0, 5484.0] | 5468.0 [5464.0, 5480.0] | 0.9993 [0.9964, 1.0029] | inconclusive |
| mem_grad_a4 | baseline vs selector_rec | yes | 0.49616 | 229 | 26108.1 [23103.7, 34783.4] | 9676.5 [7018.8, 10891.7] | 0.3706 [0.2018, 0.4714] | WIN | 5472.0 [5460.0, 5484.0] | 5460.0 [5460.0, 5476.0] | 0.9978 [0.9956, 1.0029] | inconclusive |
| mem_grad_a5 | baseline vs selector_rec | yes | 0.49613 | 49 | 116400.1 [62900.3, 125847.3] | 33374.6 [27451.6, 44249.8] | 0.2867 [0.2181, 0.7035] | WIN | 5460.0 [5456.0, 5484.0] | 5464.0 [5460.0, 5484.0] | 1.0007 [0.9956, 1.0051] | inconclusive |
| holdout_alloc_pulse | baseline vs selector_rec | yes | 0.49625 | 181 | 10681.6 [8731.8, 13924.2] | 3218.4 [2823.8, 3992.5] | 0.3013 [0.2028, 0.4572] | WIN | 5544.0 [5532.0, 5560.0] | 5468.0 [5460.0, 5492.0] | 0.9863 [0.9820, 0.9928] | WIN |

## Family: thermal

| subject | arms (A vs B) | plans differ | modeled B/A | iters | wall A µs/run | wall B µs/run | wall ratio B/A | wall verdict | RSS A KB | RSS B KB | RSS ratio B/A | RSS verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp_hot | baseline vs full_pinn_v2_rec | yes | 1.00000 | 1871 | 2935.3 [2474.9, 3141.5] | 3245.0 [2730.7, 3818.2] | 1.1055 [0.8693, 1.5428] | inconclusive | 5560.0 [5556.0, 5576.0] | 5556.0 [5544.0, 5572.0] | 0.9993 [0.9943, 1.0029] | inconclusive |
| grad_f90_d1_n64 | baseline vs full_pinn_v2_rec | yes | 1.00000 | 32258 | 183.6 [173.8, 197.6] | 184.8 [178.5, 204.3] | 1.0069 [0.9035, 1.1757] | inconclusive | 5588.0 [5584.0, 5592.0] | 5596.0 [5588.0, 5608.0] | 1.0014 [0.9993, 1.0043] | inconclusive |
| grad_f90_d2_n64 | baseline vs full_pinn_v2_rec | yes | 1.00000 | 3056 | 1153.1 [625.9, 1665.5] | 1072.8 [679.2, 1872.1] | 0.9304 [0.4078, 2.9911] | inconclusive | 5736.0 [5724.0, 5756.0] | 5732.0 [5704.0, 5736.0] | 0.9993 [0.9910, 1.0021] | inconclusive |
| grad_f90_d1_n256 | baseline vs full_pinn_v2_rec | yes | 1.00000 | 11286 | 478.0 [279.7, 156850.6] | 408.0 [316.5, 739.2] | 0.8535 [0.0020, 2.6434] | inconclusive | 5588.0 [5580.0, 5600.0] | 5576.0 [5572.0, 5596.0] | 0.9979 [0.9950, 1.0029] | inconclusive |
| grad_f90_d2_n256 | baseline vs full_pinn_v2_rec | yes | 1.00000 | 2318 | 3859.3 [3577.8, 4714.1] | 3365.4 [3289.3, 3630.2] | 0.8720 [0.6978, 1.0147] | inconclusive | 5748.0 [5732.0, 5760.0] | 5736.0 [5728.0, 5744.0] | 0.9979 [0.9944, 1.0021] | inconclusive |
| grad_f90_d1_n1024 | baseline vs full_pinn_v2_rec | yes | 1.00000 | 3170 | 1735.2 [1659.9, 1948.1] | 1740.9 [1446.7, 1955.0] | 1.0033 [0.7426, 1.1778] | inconclusive | 5596.0 [5580.0, 5596.0] | 5604.0 [5588.0, 5624.0] | 1.0014 [0.9986, 1.0079] | inconclusive |
| grad_f90_d2_n1024 | baseline vs full_pinn_v2_rec | yes | 1.00000 | 206 | 12416.1 [11658.7, 15135.9] | 14091.6 [12226.0, 15157.8] | 1.1349 [0.8077, 1.3001] | inconclusive | 5660.0 [5652.0, 5688.0] | 5672.0 [5640.0, 5684.0] | 1.0021 [0.9916, 1.0057] | inconclusive |

## Family: tensor

| subject | arms (A vs B) | plans differ | modeled B/A | iters | wall A µs/run | wall B µs/run | wall ratio B/A | wall verdict | RSS A KB | RSS B KB | RSS ratio B/A | RSS verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tensor_mm_n16_i50 | baseline vs selector_rec | yes | 1.00000 | 537 | 6942.4 [6415.7, 7400.1] | 7421.9 [6130.3, 8314.9] | 1.0691 [0.8284, 1.2960] | inconclusive | 10460.0 [10400.0, 10472.0] | 10488.0 [10420.0, 10492.0] | 1.0027 [0.9950, 1.0088] | inconclusive |
| tensor_ew_n32_i200 | baseline vs selector_rec | yes | 1.00000 | 136 | 33505.6 [21816.3, 41599.3] | 28087.1 [22372.1, 33262.9] | 0.8383 [0.5378, 1.5247] | inconclusive | 9856.0 [9848.0, 10060.0] | 9888.0 [9868.0, 10016.0] | 1.0032 [0.9809, 1.0171] | inconclusive |
| tensor_red_n64_i100 | baseline vs selector_rec | yes | 1.00000 | 27 | 164762.8 [160026.1, 182137.6] | 162316.9 [155811.3, 177017.9] | 0.9852 [0.8555, 1.1062] | inconclusive | 11344.0 [10656.0, 11412.0] | 10968.0 [10744.0, 10980.0] | 0.9669 [0.9415, 1.0304] | inconclusive |
| tensor_mix_n16_i50 | baseline vs selector_rec | yes | 1.00000 | 932 | 3806.8 [3274.4, 4300.7] | 3608.6 [3222.8, 3784.7] | 0.9479 [0.7494, 1.1558] | inconclusive | 10536.0 [10488.0, 10552.0] | 10468.0 [10464.0, 10480.0] | 0.9935 [0.9917, 0.9992] | WIN |
| tensor_tg_k0 | baseline vs selector_rec | yes | 1.00000 | 2050 | 3231.6 [1717.4, 3919.1] | 3239.9 [1465.1, 4876.2] | 1.0026 [0.3738, 2.8393] | inconclusive | 10512.0 [10424.0, 10544.0] | 10496.0 [10432.0, 10632.0] | 0.9985 [0.9894, 1.0200] | inconclusive |
| tensor_tg_k1 | baseline vs selector_rec | yes | 1.00000 | 1847 | 3488.4 [1904.5, 4485.8] | 3619.7 [1421.4, 1890084.1] | 1.0376 [0.3169, 992.4234] | inconclusive | 10504.0 [10460.0, 10560.0] | 10524.0 [10468.0, 10608.0] | 1.0019 [0.9913, 1.0141] | inconclusive |
| tensor_tg_k2 | baseline vs selector_rec | yes | 1.00000 | 2360 | 2690.0 [1823.5, 3146.5] | 2282.4 [1599.3, 2617.5] | 0.8485 [0.5083, 1.4354] | inconclusive | 10508.0 [10496.0, 10576.0] | 10584.0 [10524.0, 10624.0] | 1.0072 [0.9951, 1.0122] | inconclusive |
| tensor_tg_k3 | baseline vs selector_rec | yes | 1.00000 | 2624 | 2385.1 [2199.3, 2562.2] | 2845.4 [2606.6, 4116.6] | 1.1930 [1.0173, 1.8717] | REGRESSION | 10556.0 [10504.0, 10588.0] | 10552.0 [10508.0, 10700.0] | 0.9996 [0.9924, 1.0187] | inconclusive |
| tensor_tg_k4 | baseline vs selector_rec | yes | 1.00000 | 1614 | 2377.9 [2053.0, 2698.3] | 2241.3 [1959.5, 2692.1] | 0.9426 [0.7262, 1.3113] | inconclusive | 10528.0 [10480.0, 10548.0] | 10520.0 [10512.0, 10628.0] | 0.9992 [0.9966, 1.0141] | inconclusive |

## Family: nonsynthetic

| subject | arms (A vs B) | plans differ | modeled B/A | iters | wall A µs/run | wall B µs/run | wall ratio B/A | wall verdict | RSS A KB | RSS B KB | RSS ratio B/A | RSS verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| example_08_pinn_heat | baseline vs selector_rec | yes | 1.00000 | 1 | 4512635.0 [3575619.0, 5383606.0] | 3527388.0 [3289906.0, 5634113.0] | 0.7817 [0.6111, 1.5757] | inconclusive | 9452.0 [9428.0, 9576.0] | 9492.0 [9416.0, 9504.0] | 1.0042 [0.9833, 1.0081] | inconclusive |

## Modeled vs measured (the Phase D question)

| subject | modeled energy B/A | wall-clock B/A (median) | agree? |
|---|---|---|---|
| mem_grad_a1 | 0.49855 | 0.6684 | inconclusive |
| mem_grad_a2 | 0.49673 | 0.3653 | yes |
| mem_grad_a3 | 0.49628 | 0.3711 | yes |
| mem_grad_a4 | 0.49616 | 0.3706 | yes |
| mem_grad_a5 | 0.49613 | 0.2867 | yes |
| holdout_alloc_pulse | 0.49625 | 0.3013 | yes |
| fp_hot | 1.00000 | 1.1055 | inconclusive |
| grad_f90_d1_n64 | 1.00000 | 1.0069 | inconclusive |
| grad_f90_d2_n64 | 1.00000 | 0.9304 | inconclusive |
| grad_f90_d1_n256 | 1.00000 | 0.8535 | inconclusive |
| grad_f90_d2_n256 | 1.00000 | 0.8720 | inconclusive |
| grad_f90_d1_n1024 | 1.00000 | 1.0033 | inconclusive |
| grad_f90_d2_n1024 | 1.00000 | 1.1349 | inconclusive |
| tensor_mm_n16_i50 | 1.00000 | 1.0691 | inconclusive |
| tensor_ew_n32_i200 | 1.00000 | 0.8383 | inconclusive |
| tensor_red_n64_i100 | 1.00000 | 0.9852 | inconclusive |
| tensor_mix_n16_i50 | 1.00000 | 0.9479 | inconclusive |
| tensor_tg_k0 | 1.00000 | 1.0026 | inconclusive |
| tensor_tg_k1 | 1.00000 | 1.0376 | inconclusive |
| tensor_tg_k2 | 1.00000 | 0.8485 | inconclusive |
| tensor_tg_k3 | 1.00000 | 1.1930 | NO |
| tensor_tg_k4 | 1.00000 | 0.9426 | inconclusive |
| example_08_pinn_heat | 1.00000 | 0.7817 | inconclusive |

Summary: 23 subjects (0 controls with identical plans); wall-clock verdicts: 5 WIN, 1 REGRESSION, 17 inconclusive.

Hard wall: nothing in this report feeds back into compile decisions,
hashes, or profile rows.
