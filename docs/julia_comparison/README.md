# CJC-Lang compared with Julia: scientific ML and quantum gap analysis

**Date:** 2026-11-09 · **CJC commit:** `31058a8` · **Phase:** plan only. No file under `crates/`, `tests/` or `lib/` was changed.

## Scope

This analysis finds where Julia and its ecosystem are more correct, more
accurate or more capable than CJC-Lang, in four areas:
- the numeric core;
- automatic differentiation;
- SciML (ODE, PDE, PINN, optimisation, linear algebra);
- quantum simulation.

For each gap it decides what CJC should adopt **without giving up
bit-identical output, executor parity, or the zero-dependency rule**.

Out of scope:
- performance comparisons;
- Julia interop or FFI;
- GPU and QPU execution.

## Method

1. **Inventory of the CJC code** ([CJC_INVENTORY.md](CJC_INVENTORY.md)).
   - Four read-only audits covered AD, SciML and numerics, the language core, and quantum.
   - I re-opened the key code myself.
   - Every row is labelled **probe** (shown by a runnable `.cjcl` program), **read** (I opened the code), or **audit**; and **cjcl**, **crate** or **stub**.
2. **Julia survey** ([JULIA_SURVEY.md](JULIA_SURVEY.md)). Current versions, maintenance status and accuracy mechanisms for 37 packages, each with a source URL. Unconfirmed items are marked [verify].
3. **Matrix** ([COMPARISON_MATRIX.md](COMPARISON_MATRIX.md)). For each Julia feature: its accuracy mechanism and whether it is determinism-compatible.
4. **Probes** ([probes/](probes/)).
   - Seven `.cjcl` programs, each run through both executors, with reference scripts (numpy, mpmath).
   - Six show a wrong answer that is reachable from `.cjcl` today.
5. **Language mechanisms** ([LANGUAGE_MECHANISMS.md](LANGUAGE_MECHANISMS.md)): which Julia mechanisms belong in the core, which in Bastion, and which nowhere.
6. **Harness design** ([ACCURACY_HARNESS_PLAN.md](ACCURACY_HARNESS_PLAN.md)): Julia as an offline oracle, double-double references, positive controls.
7. **Roadmap and issues** ([ROADMAP.md](ROADMAP.md), [ISSUES.md](ISSUES.md)), plus the findings the gate removed ([REJECTED.md](REJECTED.md)).

## Top 10 gaps

| # | Gap | Priority | Evidence |
|---|---|---|---|
| 1 | The backward pass does not reduce broadcast gradients; the XOR example's bias silently grows to [4,4] | P0 | probe P2 |
| 2 | `hessian_diag`, the only Hessian reachable from `.cjcl`, is wrong for composed losses (4 vs 12) | P0 | probe P1 |
| 3 | `svd` is computed as `eigh(AᵀA)` and reports σ = 0 for a full-rank matrix | P0 | probe P3 |
| 4 | DMRG ground-state energies are about 8% off at n ≥ 4 | P0 | probe P6 |
| 5 | Binned summation depends on input order, which contradicts the determinism contract | P0 | probe P5 |
| 6 | `erf`/`erfc` have relative error up to 6.7e-3 | P0 | probe P4 |
| 7 | `randn` uses platform libm, so results can differ across OSes; a candidate cause of the open canary failures | P0 | read |
| 8 | No exact higher-order AD (the Hessians are FD; no Taylor mode), so PINNs cannot use exact residuals | P1 | read, audit |
| 9 | ODE solvers cannot be reached from `.cjcl` (they need callbacks); no stiff solvers; the ODE adjoint is broken | P1 | audit |
| 10 | No open-system quantum simulation (Lindblad, MC wavefunction) and no general MPO | P1 | audit |

## The first harness slice to build

**H0 + S1 + S1b:**
- **H0:** check that CSV float parsing is correctly rounded.
- **S1:** elementary functions (the negative control: dmath should pass).
- **S1b:** `erf`/`erfc` against Julia `BigFloat` and mpmath (the positive control: it must fail on the tail of P4).

It is the smallest slice that compares CJC with a Julia oracle and shows a
real accuracy difference. It needs only Julia Base and SpecialFunctions, and
it checks the CSV path every later slice depends on. See ACCURACY_HARNESS_PLAN §4 and §6.

## A finding about method

Every probe showed **IDENTICAL** output in eval and MIR-exec, including the six
wrong answers. Executor parity proves the two executors agree. It does not
prove they are right, because they share the same runtime code. Accuracy
needs an outside oracle, and that is the reason for this harness.
</content>
</invoke>
