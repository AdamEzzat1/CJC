# Phase D — Silicon Diagnostics: Do Modeled Wins Appear on the Stopwatch?

**Date:** 2026-06-11 (Phase D session, follows the A+B+C triple session)
**Crate:** `bench/cana_diagnostics` (lib + bin, publish = false)
**Spec source:** `docs/cana/NEXT_ARC_RESEARCH.md` §3 (the auditor's design),
`docs/cana/HANDOFF_PHASE_D.md` §2.
**Exit criterion (roadmap §6 row D):** byte-equal outputs + reported
wall-clock/RSS deltas with confidence bands.

## 0. The question

Everything CANA/NSS measured through Phase C is MODELED energy
(executed statements + 3.0 × FP ops + heap pages, from instrumented
runs). Phase C produced the first measured modeled-energy win
(`selector_rec` mean 0.982 vs baseline; 6 named wins, best −50%
executed work). Phase D asks the only question that justifies the
stack to a user: **does any of that appear on a wall clock?** A
negative answer is load-bearing — it recalibrates the energy formula
(`FP_ENERGY_WEIGHT`, the heap term) before Phases E/F build on it.

## 1. Harness design

### 1.1 Gates before clocks (all hard errors)

| # | Gate | What it proves |
|---|---|---|
| 1 | Corpus program-hash identity | The snapshot sources in this crate ARE the corpus programs (MIR-level `program_hash` equality vs committed `profiles.cpdb`) — snapshot drift cannot silently time the wrong program. |
| 2 | Plan identity | The recomputed arm plans byte-equal the corpus rows' `pass_sequence` — we time exactly the plans Phase C measured. |
| 3 | Output determinism | AST-eval, MIR-exec(arm A), MIR-exec(arm B) outputs byte-identical — THE determinism gate (contract: FIRST, before any timing). |
| 4 | Modeled-energy reproduction | The energy ratio recomputed in THIS build matches the corpus `score` to 1e-9 — the modeled win demonstrably exists in the binary being timed, so a null stopwatch result is unambiguous. |

Gates 1/2/4 are skipped only for the non-corpus subject
(`example_08_pinn_heat`); gate 3 never. Children additionally re-check
the gate-3 output FNV on EVERY sustained-load iteration and the parent
re-verifies the child's reported FNV against the gate-3 transcript.

### 1.2 Subjects (4 families, 23 subjects)

| Family | Subjects | Arms (A vs B) | Question |
|---|---|---|---|
| `selector` | `mem_grad_a1..a5`, `holdout_alloc_pulse` | `baseline` vs `selector_rec` | Does the −50% modeled-energy win show on the stopwatch? (One mechanism — DCE of dead per-iteration allocations — not six independent wins.) |
| `thermal` | `fp_hot`, `grad_f90_d{1,2}_n{64,256,1024}` | `baseline` vs `full_pinn_v2_rec` | The original Track-3 subjects: sustained scalar FP under the thermal-aware trained stack. |
| `tensor` | `tensor_mm/ew/red/mix`, `tensor_tg_k0..4` | `baseline` vs `selector_rec` | Validates the energy formula's FP weighting on tensor dispatch. |
| `nonsynthetic` | `example_08_pinn_heat` (the shipped PINN heat-equation example) | `baseline` vs `selector_rec` | The handoff's broad-claim guard: a real workload next to the synthetics. |

Subjects whose two arms produce IDENTICAL plans are kept and labeled
CONTROLS — they measure the protocol's noise floor (expected ratio 1.0).

### 1.3 Protocol (auditor's spec)

- Iteration count calibrated ONCE on arm A (`~5 s target / single-run
  time`, clamp [1, 200k]); identical count for both arms.
- 1 warm-up phase per arm, then 5 measured phases per arm, interleaved
  A/B/A/B (both arms share the thermal environment; thermal time
  constants are seconds+, hence ~5 s sustained phases).
- **Each phase is a fresh child process.** `PeakWorkingSetSize` is
  process-monotonic, so per-arm peak RSS requires process isolation;
  fresh processes also give identical allocator/startup state.
- The child receives its gate-2-verified plan as a FILE from the
  parent and only applies it (NoGC + MIR-legality verifiers still run
  in the child). Children never plan — see §3.1.
- The child times ONLY the `iters × MirExecutor` loop; compile and
  plan application are excluded. Peak RSS is sampled twice (after plan
  application / after the loop) so the execution contribution is
  separable.
- Stats: median + [min, max] band over the 5 measured phases; ratio
  band is conservative (`lo = minB/maxA`, `hi = maxB/minA`); a verdict
  is WIN/REGRESSION only when the ENTIRE conservative band clears 1.0.
- Wall-clock + peak RSS only. CPU frequency (~1 Hz best-effort) and
  WMI temperature (`Option<f64>`, often absent) are out of MVP scope
  per the §3 signal-reality audit; within-machine deltas only.

### 1.4 The hard wall

Nothing measured here feeds back into decisions, hashes, or row
stable fields. Artifacts land in `bench_results/cana_diagnostics/`
(`REPORT.md`, `phases.csv`, `plans/*.plan`); `profiles.cpdb` is
read-only to this harness.

## 2. Verification (test-discipline contract)

`bench/cana_diagnostics/tests/test_diagnostics.rs` — 23 tests:

- **Wiring (5):** subject-list shape matches the handoff exactly (6
  named selector wins, family sizes 6/7/9/1); gate 1 over ALL 22
  corpus subjects (compile-only, cheap); gates 2+3+4 end-to-end on
  `mem_grad_a1` (recomputed plans byte-match corpus rows, modeled
  ratio < 1 reproduces); child workload cross-arm output-FNV identity
  through the plan-file protocol; the non-synthetic subject compiles
  and is excluded from corpus gates.
- **Unit (9):** median/band/ratio-band/verdict edge cases; calibration
  clamping (incl. zero-duration division guard); child-line and
  plan-file roundtrips; malformed-input rejection; output-FNV line
  boundary separation.
- **Proptest (5):** median ∈ [min, max]; band ordering; calibration
  bounds; child-line roundtrip over arbitrary measurements; plan-file
  roundtrip over arbitrary well-formed plans.
- **Bolero (3):** `parse_child_line`, `parse_plan`, and the stats
  helpers never panic on arbitrary input (NaN/inf included).

## 3. Findings before any timing (the harness itself found things)

### 3.1 Selector planning memory: 1.63 GB peak on a real program

The first harness design had each child plan its own arm. On
`example_08_pinn_heat` the `selector_rec` child peaked at **1.63 GB
RSS vs 206 MB for the baseline arm (~8×)** — and the plan-time RSS
sample showed the peak occurs DURING planning, not execution.
`PassPlanSelector` re-optimizes the whole program per candidate (≤10
per function); the corpus programs are all small, so this is the first
big-function program the selector ever planned. This is measured
evidence for the handoff §3 hardening item "per-function `optimize`
API to avoid whole-program re-optimization per candidate" — it is not
just wasteful at scale, it is a memory spike on real workloads.

(The harness was redesigned in response: the parent plans, children
apply — so child RSS measures the program, not the planner.)

**Resolution, same day:** the finding was spun off as a task and the
fix SHIPPED on master as `e3f631b`
(`cjc_mir::optimize::optimize_function_with_passes`; the selector
probe now clones one function per candidate), with plan identity
locked by a corpus gate
(`cana-ablation tests::gate_selector_rec_plans_unchanged_on_committed_corpus`).
The measurements in §4 were taken on the PRE-fix selector; the gate
proves the selected plans — and therefore the timed programs — are
identical either way.

### 3.2 Windows prefetcher warm-up

The first 1–2 child processes of a run show a ~5.4 MB startup RSS
baseline; every subsequent identical child shows ~12.6 MB — the
application prefetcher learns the exe's access pattern and maps more
image pages eagerly at startup. Arm-neutral, absorbed by the warm-up
phases (which are excluded from stats).

### 3.3 Thermal ramp is visible

Warm-up phases run measurably FASTER than measured phases (cool CPU →
turbo headroom; e.g. fp_hot 1219 µs/run warm-up vs 1413–1474 µs/run
measured). This is precisely why phases are interleaved A/B/A/B: both
arms sample the same thermal trajectory.

## 4. Results (measured 2026-06-11; full tables in `bench_results/cana_diagnostics/REPORT.md`, raw phases in `phases.csv`)

All 23 subjects passed every gate — output byte-equality held across
AST-eval and both arms on every subject (including the real PINN
example under the selector plan), and all 22 corpus subjects
reproduced their corpus scores to 1e-9. Wall-clock verdicts: **5 WIN,
1 borderline REGRESSION, 17 inconclusive**.

### 4.1 The headline: the selector wins are real on silicon — and BIGGER than modeled

| subject | modeled B/A | wall B/A median [conservative band] | verdict |
|---|---|---|---|
| mem_grad_a2 | 0.49673 | **0.3653** [0.2783, 0.4952] | WIN |
| mem_grad_a3 | 0.49628 | **0.3711** [0.2443, 0.4893] | WIN |
| mem_grad_a4 | 0.49616 | **0.3706** [0.2018, 0.4714] | WIN |
| mem_grad_a5 | 0.49613 | **0.2867** [0.2181, 0.7035] | WIN |
| holdout_alloc_pulse | 0.49625 | **0.3013** [0.2028, 0.4572] | WIN |
| mem_grad_a1 | 0.49855 | 0.6684 [0.1205, 1.2681] | inconclusive |

Five of the six named Phase C wins hold on the stopwatch with the
ENTIRE conservative band below 1.0 — including `holdout_alloc_pulse`,
the frozen-holdout program neither head ever trained on. The sixth
(`mem_grad_a1`, the smallest program at 256 loop iterations/run) is
direction-consistent (median 0.67; quick-profile runs measured
0.41–0.42) but its band crosses 1.0 under this run's noise — honest
inconclusive, not a refutation.

**The measured effect exceeds the modeled one** (wall 0.29–0.37 vs
modeled 0.496-ish). The mechanism: the eliminated statements are
per-iteration allocations (array + tuple construction), and an
allocation statement costs the interpreter ~2–3× an average statement
(Rc allocation + heap traffic), while the energy formula prices every
non-FP statement at 1. The model UNDERPRICES exactly the work the
selector removes. Recalibration direction for the formula (Phase B
follow-up): an allocation-statement weight analogous to
`FP_ENERGY_WEIGHT`.

Caveat unchanged from the handoff: all five wins are ONE mechanism
(DCE of dead per-iteration allocations), not five independent
discoveries. The frozen-holdout win generalizes the mechanism, not the
breadth.

### 4.2 RSS: plan choice does not move peak memory (expected)

Selector-family RSS ratios are ≈0.999 with bands of ±0.5% — DCE
removes executed work, not peak footprint (per-iteration allocations
are freed within the iteration; the peak never saw them).
`holdout_alloc_pulse` shows a tiny whole-band RSS win (0.9863
[0.9820, 0.9928], −1.4%) — real but marginal; do not build claims on
it.

### 4.3 Thermal + tensor families: modeled ties measure as ties

All 13 modeled-tie subjects in the `thermal` and `tensor` families
land inconclusive (bands straddle 1.0, medians scatter 0.84–1.19),
EXCEPT `tensor_tg_k3`: borderline REGRESSION at 1.1930 with band lo
1.0173 — 1.7% above the win/regress threshold, on a run where two
other phases recorded >100× outlier stalls (see §4.5). Treat as
noise-suspect and re-run before reading anything into it; a real
effect would require interpreter-level cost differences between
equal-statement-count MIR shapes, which the modeled tie can't see but
which nothing else corroborates.

The honest Track-3 conclusion: the thermal-aware stack's plans differ
from baseline on these subjects but tie on executed work, so there is
nothing for the stopwatch to find — consistent with the model,
and NOT evidence of a thermal/runtime benefit from the thermal layer.

### 4.4 The non-synthetic subject: no harm, no win

`example_08_pinn_heat` (the shipped PINN heat-equation example) under
the selector plan: output byte-identical (the gate that matters for
default-on arguments), modeled tie, wall median 0.78 with a band
crossing 1.0 → inconclusive. The selector neither helped nor hurt a
real program it never trained on; the broad-claim guard holds in both
directions.

### 4.5 Measurement honesty

This is a developer workstation, not a quiet bench. Two of 230
measured phases recorded >100× single-phase stalls (`grad_f90_d1_n256`
arm A max 156.9 ms/run vs median 0.48 ms; `tensor_tg_k1` arm B max
1.89 s/run vs median 3.6 ms — OS scheduling/AV interference). The
conservative-band verdict rule absorbs these as inflated bands →
inconclusive, never as false WIN/REGRESSION verdicts. The five selector
wins clear the bar DESPITE this noise model.

## 5. Verdict on the hypothesis ledger

| Claim | Status after Phase D |
|---|---|
| Modeled-energy wins appear on wall-clock | **MEASURED, CONFIRMED** for the selector mechanism: 5/6 named wins hold with whole-band wall-clock reductions (best median 0.287), including the frozen holdout. The 6th is direction-consistent but noise-inconclusive. |
| Energy formula calibration | **NEW FINDING:** the formula UNDERPRICES allocation statements — measured reduction (0.29–0.37) exceeds modeled (0.496) by 1.3–1.7×. Recalibration sketch: allocation-statement weight next to `FP_ENERGY_WEIGHT`. |
| Thermal-aware stack produces wall-clock benefit | **NOT OBSERVED** on the Track-3 subjects (its plans tie baseline on executed work; stopwatch agrees). |
| Selector is safe on a real program | Output byte-identical, no measurable wall/RSS change on `example_08_pinn_heat`. |
| Selector planning cost | **NEW FINDING (§3.1), FIXED same day:** 1.63 GB candidate-probing RSS peak on the real example → per-function optimize API shipped on master (`e3f631b`), plan identity corpus-gated. |

**Exit criterion (roadmap §6 row D): MET** — byte-equal outputs on
every subject + wall-clock/RSS deltas with confidence bands, reported.

What this does NOT license: any user-facing performance claim beyond
"on synthetic allocation-churn loops and one frozen holdout of the
same shape, the energy-selector plan cuts interpreter wall time
~2.7–3.5× with byte-identical output, on one machine." The selector
remains ablation-grade, not default-on; margin gating (handoff §3)
is still the next hardening step, now with a stopwatch to verify it
against.
