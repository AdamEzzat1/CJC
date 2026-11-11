# Probes: runnable evidence for the findings

Each probe is a `.cjcl` program that shows one finding from `.cjcl` source,
with no change to `crates/`. Run one through both executors with:

```bash
./target/release/cjcl parity docs/julia_comparison/probes/<probe>.cjcl
```

References come from scripts kept here. They need Python with `numpy` or
`mpmath` only, and Julia is not needed:

```bash
python -I docs/julia_comparison/probes/p6_exact.py
python -I docs/julia_comparison/probes/p_refs_mpmath.py
```

**Recorded run:** 2026-11-09, commit `31058a8`, `cjcl` v0.1.11 release build,
Windows 11 (i7-11390H), Python 3 with numpy 2.2.6 and mpmath 1.3.0. For every
probe, `cjcl parity` reported **IDENTICAL**: both executors print the same
output. That is expected, because both executors call the same Rust code here.
Parity is therefore not evidence of correctness for these findings.

| Probe | Finding | Expected (reference) | CJC output (eval = MIR) | Verdict |
|---|---|---|---|---|
| `p1_hessian_diag_stale.cjcl` | `GradGraph.hessian_diag` never re-runs the forward pass | d²(p⁴)/dp² at p=1 = **12** | **4.000000000004** | Wrong answer (P0) |
| | Control: the parameter feeds the loss directly | d²(p²)/dp² = 2 | 2.000000000002 | Correct, to the accuracy of finite differences |
| `p2_broadcast_backward.cjcl` | The backward pass does not sum a broadcast gradient back to the input's shape | grad b = `[2,2,2]`, shape `[3]` (or `[1,3]`) | `shape=[2,3]`, all `1.0` | Wrong shape and wrong values (P0) |
| `p3_svd_small_sigma.cjcl` | `svd` is computed as `eigh(AᵀA)` | σ = [2.00000000005, **5.000000413576855e-11**] (mpmath, 50 digits) | `svd`: [2.00000000005, **0.0**]; `eigh`: 5.000000413701855e-11 | `svd` reports a full-rank matrix as singular (P0). `eigh` is correct to 2.5e-11 relative. |
| `p4_erf.cjcl` | `erf` and `erfc` use Abramowitz & Stegun 7.1.26 | erf(0.5) = 0.52049987781304654; erfc(5) = 1.5374597944280349e-12 | 0.5205000163047472; 1.547817043333678e-12 | Relative error 2.7e-7 (erf(0.5)), 6.6e-4 (erfc(3)), **6.7e-3** (erfc(5)) (P1) |
| `p5_binned_permutation.cjcl` | `Tensor.sum` result depends on the order of the inputs | 3 + 2⁻⁵¹ = 3.0000000000000004 (exact, representable) | `[1, b, b]` → **3**; `[b, b, 1]` → 3.0000000000000004; `==` → **false** | The sum is not order-invariant, and one order is also wrong by 1 ulp (P0, determinism contract) |
| `p6_dmrg_heisenberg.cjcl` + `p6_exact.py` | `dmrg_heisenberg` returns an energy above the ground state | n=4: **−6.464101615137755**; n=6: −9.974308535551705; n=8: −13.499730394751575 | −5.950264027437909; −9.16015641952005; −12.522059321153153 | Relative error 8.0%, 8.2%, 7.2%. With χ=8, n=4 is exactly representable, so truncation is not the cause (P0) |
| `p7_bisect_stub.cjcl` | `bisect` has a match arm in `builtins.rs`, but no executor handles it | a root near √2 | `undefined function 'bisect'` in both executors | Dead stub (P2) |

## How to read these

- **P1 is the dangerous kind:** a correct-looking number that is wrong by 3×.
  `hessian_diag` is the only Hessian reachable from `.cjcl`. Its control case
  is correct, which is how the bug went unnoticed: tests that use `loss = p*p`
  cannot catch it.
- **P2 affects real code.** `examples/01_mlp_xor.cjcl` creates `b1` with shape
  `[1, 4]` and updates it as `b1 = b1 - gb1 * lr`. With a batch of 4, `gb1` has
  shape `[4, 4]`, so after the first step `b1` has shape `[4, 4]`. The model
  stops being "one bias per unit" without any error. This was checked by
  running a copy with `n_epochs = 2` and `print(b1.shape());` after the
  update, which prints `[4, 4]` twice:
  `sed 's/    let n_epochs = 200;/    let n_epochs = 2;/; s/^\(        b1 = b1 - gb1 \* lr;\)/\1\n        print(b1.shape());/' examples/01_mlp_xor.cjcl > /tmp/x.cjcl && cjcl run /tmp/x.cjcl`.
  The fused `mlp_layer` op
  reduces the bias gradient correctly, which is why most trainers in the repo
  (which use the fused op) never hit this.
- **P5 matters even though every run is reproducible.** The same program gives
  the same bits on every run. But the determinism contract also promises that
  a reduction does not depend on how its input is ordered or chunked (thread
  count, tiling). The binned accumulator adds values within one exponent bin
  with ordinary IEEE `+`, which is not associative. Three values are enough to
  show it.
- **P6** is not the 2-site case the tests check. The 4-site test only asserts
  `E < 0`, and the exact value is −6.46.
</content>
</invoke>
