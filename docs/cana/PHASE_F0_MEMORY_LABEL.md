# Phase F0 — The Memory-Label Fix: Variance Unblocked, Features Now the Blocker

**Date:** 2026-06-12 (follows Phase E on `claude/stupefied-liskov-83b258`)
**Spec source:** `docs/cana/HANDOFF_PHASE_D.md` §4 (the Phase F sketch:
"per-window allocated-bytes counter at creation sites — the A1 pattern
applied to memory — a MirTraceEvent schema change → adapter + corpus
ripple").
**Exit criterion:** recorded memory-label variance clears the
trainability bar (corpus std > 0.05; the pre-fix structural ceiling was
std 0.0009 / max 0.0078).

## 0. Verdict up front

| number | before F0 | after F0 |
|---|---|---|
| memory label, corpus max | 0.0078 | **1.0000** |
| memory label, corpus std | 0.0009 | **0.1083** — exit criterion MET |
| `rec memory ~ workload` OLS R²(train) | ≈ 0 (unfittable) | 0.7683 |
| `rec memory ~ workload` OLS R²(test) | — | **0.0477 — F1's blocker** |

The label-side blindness is fixed (variance up ~120×, full [0, 1]
range). The memory HEAD (F1) remains untrainable for a new, precisely
identified reason: the STATIC feature set doesn't generalize to the new
label out-of-sample — the same information-gap pattern PINN v2 §2.1
found for thermal, now on the memory axis. See §4.

## 1. What was broken (handoff §4, verified)

The recorded memory label derived from `heap_bytes_in_use`, whose proxy
only sees `gc_alloc` objects (×4096) and executed arena-classified
`Let`s (×64) — Rc buffers (arrays, tensors, strings: the actual memory
consumers) were structurally invisible. The `mem_grad_a{1..5}` corpus
family, designed to produce a memory gradient, measured the
mechanism-exact ceiling (max 0.0078) because no `.cjcl` program could
move the label more than that.

## 2. The fix (the A1 pattern applied to memory)

**Runtime counter** (`cjc-mir-exec`): `trace_alloc_bytes`, incremented
at five curated creation sites with platform-stable MODEL prices (never
`size_of` — labels must be bit-stable across platforms):

| site | price |
|---|---|
| array literal | 16 B × elements (`ARRAY_ELEM_ALLOC_BYTES`) |
| tuple literal | 16 B × elements |
| tensor binop result (incl. broadcast) | 8 B × result elements |
| builtin results: `array_push` (+16 B), `matmul` (8 B × m·n), `adam_step` | per shape |
| tensor-method results (`abs`/`relu`/`softmax`/`transpose`/…); reductions price 0 | 8 B × elements |

Curated under-counting is the safe direction (unlisted sites make
programs look lighter, never heavier) — the same discipline as the A1
tensor-FP helpers, and the counter only ticks under `trace_enabled`
(uninstrumented runs untouched).

**Schema** (`cjc-nss`): `MirTraceEvent.alloc_bytes_in_window: u64`,
drained per window in `trace_emit`. Option-A synthetic traces carry 0
(the `thermal_intensity: 0.0` precedent).

**Adapter** (`cjc-nss::mir_adapter`): Memory pressure is now
`max(heap_term, alloc_term)` where the alloc term is the **cumulative**
per-block allocation total normalized by a new
`alloc_capacity_bytes` config knob. Cumulative, not per-window rate —
the `mem_grad` family varies iteration COUNT at fixed per-iteration
churn, so a rate term would be flat across exactly the family built to
provide the gradient; the label reads the trajectory's LAST state, so
the cumulative term turns allocation volume into signal.

**Capacity calibrated from evidence, second regen:** the first regen
shipped a guessed 64 MiB capacity and measured the corpus-max
cumulative allocation at ~4.2 MB (`mem_grad_a5` = 65,536 iterations ×
64 B — the mechanism landed exactly on its designed values), which
squashed all 158 programs into [0, 0.0625] (std 0.0068). The default
was retuned to 4 MiB — the evidence-tightened bound, same precedent as
the ablation harness's code-size cap — and the second regen produced
the §0 numbers.

## 3. What did NOT move (the safety story)

- **Plans: unchanged.** The `cana-diagnostics` plan-identity gates
  (committed-corpus byte-equality) pass before and after; the regen
  reproduces the same 6 selector wins at identical scores
  (0.49613…0.49855), mean 0.98230, parity 100%, row-hash double-run
  stable. The new label is pure signal — it does not perturb any
  current ranking decision.
- **Scores: unchanged** → both trained bundles (CPB0 thermal, CPB1
  energy) retrain to byte-identical fixed points (verified by hash
  before/after retrain).
- **Output transparency: re-proven** — instrumented vs uninstrumented
  output byte-identical on allocation-heavy programs
  (`tests/test_alloc_accounting.rs`).

## 4. F1 (the memory head) — now blocked on the FEATURE side, precisely

With label variance unblocked, the sanity pass immediately exposed the
next gap: `rec memory ~ workload` fits in-sample (R² 0.77) but does not
generalize (R²(test) 0.048; compare thermal's 0.78 after its A1 fix).
The static `allocation_bytes_estimate` does not track creation-site
allocation at the model prices — no model class can fix a feature set
that doesn't carry the information (PINN v2 §2.1's exact lesson).

**F1 prerequisite, specified:** a TypeMix-style static analysis
counting array/tuple-literal element slots and tensor-result elements ×
loop amplification — the static mirror of §2's runtime prices. This
changes `FnFeatures` → `FeatureHash` → every row hash → full corpus
regen → retrain BOTH heads → re-shadow both. A session of its own, per
the A1 precedent. Training a memory head before that would fail its
shadow gate; it was not attempted.

## 5. Verification

- `tests/test_alloc_accounting.rs` (5): exact model-price assertions
  (10-iteration churn = exactly 640 B; linear volume scaling ×8;
  tensor pricing incl. the from_vec shape-literal subtlety; scalar-FP
  programs record 0; instrumented output identity).
- `cjc-nss` 219 → 221 (+ cumulative-alloc gradient test + zero-alloc
  backward-compatibility test); full suites green: cjc-cana-nss,
  cjc-mir-exec, instrumented-transparency, fixtures parity,
  energy-selector gates, cana-diagnostics 23/23 (gates 1–4),
  cana-compress-probe 14/14 (trace formats bumped to v1 with the new
  column).
- Two corpus regens, all harness gates green both times.
