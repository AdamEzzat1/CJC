# Phase F1 — The Memory Head: Closing the Feature-Side Gap

**Date:** 2026-06-13 (follows Phase F0 on `claude/stupefied-liskov-83b258`)
**Spec source:** `docs/cana/PHASE_F0_MEMORY_LABEL.md` §4 (the F1
prerequisite: "a TypeMix-style static analysis counting array/tuple-
literal element slots and tensor-result elements × loop amplification —
the static mirror of [F0]'s runtime prices").
**Exit criterion:** a memory head whose shadow gate returns a verdict
(PROMOTE or DO-NOT-PROMOTE — both are load-bearing) against the
Phase-F0 recorded label, with the feature ablation isolating what the
new signal buys.

## 0. The pattern, third run

This is the third instance of one recurring failure mode and its fix:

| arc | what was blind | the fix | R²(test) before → after |
|---|---|---|---|
| PINN v2 (thermal) | features (type-blind) | `TypeMix` → `float_ops_estimate` | −0.05 → +0.96 |
| Phase F0 (memory, label) | the recorded LABEL (Rc-blind) | `alloc_bytes_in_window` | std 0.0009 → 0.1083 |
| **Phase F1 (memory, feature)** | features (volume-blind) | `lit_elem_slots` → `creation_alloc_bytes_estimate` | **0.019 → 0.088** (~4.6×; partial) |

The F0 signature was textbook: R²(train) 0.77 / R²(test) 0.048 — the
model could memorize which corpus programs allocate but had no feature
that *explained* it, because `alloc_sites` counts a 2-element literal
and a 774-element literal identically (1 site each), while the F0 label
prices them per element.

## 1. The feature (static mirror of F0's runtime prices)

`MemoryProxy::lit_elem_slots` (new, `crates/cjc-cana/src/memory_proxy.rs`):
total element slots across array + tuple literals — statically exact,
and exactly the literals the runtime prices per-element (struct/variant/
tensor literals price 0 on both sides). Feeds `FeatureHash` (tag
`TAG_MEMORY_PROXY`, appended after `expr_count` — so every per-function
hash changes, which is the content-addressed fingerprint working as
designed).

`PhysicalCostQuery::creation_alloc_bytes_estimate` (new,
`crates/cjc-cana/src/physical_cost.rs`): the static analog of the
window total —
`lit_slots×16 + cow_writes×16 + tensor_fp_sites×(128·8)`, all loop- and
pass-amplified like the existing byte terms so the feature and the
cumulative recorded label share scaling shape. The model constants
mirror F0's runtime prices (`ARRAY_ELEM_ALLOC_BYTES = 16`, etc.).
Additive field — the v1 closed form (`predict_physical`) never reads
it, so v1's active report hashes are unchanged.

Schema v4 (`profile_db.rs`): `FnProfile.creation_alloc` +
`CompilationProfile.estimated_creation_alloc_bytes`. No-migration
policy as before (v3 files rejected on read; regenerate).

## 2. The head (`pinn_memory_v1`, CPB2 bundle)

Eight-feature linear head on the PINN-v2 template (offline ridge,
bit-reproducible, standardize → dot with named intermediates → clamp
`[0,1]`). Features 0–5 are the workload basis the thermal head uses;
**feature 6** is `ln(1+creation_alloc_bytes_estimate)` (the volume
magnitude) and **feature 7** is creation-alloc density
`min(creation/(8·flops), 1)` (churn per unit work, exactly loop-
amplification-invariant — the `+1` epsilon trap from a first draft was
caught by `density_is_loop_amplification_invariant` and replaced with
an explicit zero-guard). Persisted as a CPB2 bundle
(`cjc-cana-compress::memory_bundle`), byte-for-byte the CPB0 design
with a `b"CPB2"` magic and an 8-element basis.

## 3. Shadow protocol (two baselines, not one)

There is no closed-form v1 memory prediction in the corpus rows (unlike
thermal, where `pinn_predicted_thermal_max` gave a v1 baseline), so the
shadow gate uses two honest baselines:

- **(a) train-mean climatology** — the no-information floor.
- **(b) the F0-era feature set** (features 0–5, no creation) under the
  *same* ridge recipe — this isolates exactly what the new feature
  buys, not what linear regression buys.

Promotion requires the head to beat **both** baselines on **both**
held-out and overall MAE. The frozen-holdout cohort is reported
separately (never trained or tuned on).

## 4. Results (measured 2026-06-13)

### 4.1 The sanity decision number — the feature helps, partially

The label is heavily zero-skewed (158 programs: mean 0.0208, std
0.1083, max 1.0 — most allocate nothing, a handful saturate). On the
8-feature memory basis vs its no-creation ablation, both fit with the
same ridge recipe on the same program split:

| feature set | R²(train) | R²(test) |
|---|---|---|
| workload only (no creation) | 0.7756 | 0.0190 |
| workload + creation (F1) | 0.8188 | **0.0882** |

R²(test) improved **~4.6×**. It did NOT fully close — absolute
held-out predictability stays low, because the FNV-test cohort
contains allocation shapes the train set underrepresents. This is an
honest partial win, not the thermal arc's −0.05 → +0.96 leap; the
memory signal is genuinely harder than the thermal one (allocation
volume depends on runtime trip counts that static loop-amplification
only approximates).

### 4.2 The shadow gate — PROMOTE, on MAE, against two baselines

MAE is the gate (not R²) because on a zero-skewed label the decision
that matters is "get the zeros right and rank the allocators," which
R² punishes disproportionately for a few hard holdout misses.

| cohort | n | mean-climatology MAE | no-creation MAE | **head MAE** | head corr |
|---|---|---|---|---|---|
| train | 114 | 0.0284 | 0.0131 | **0.0122** | +0.916 |
| held-out (FNV) | 34 | 0.0435 | 0.0420 | **0.0412** | +0.320 |
| frozen holdout | 10 | 0.0552 | 0.0303 | **0.0293** | +0.958 |
| overall | 158 | 0.0333 | 0.0204 | **0.0196** | +0.656 |

**Verdict: PROMOTE** — the head beats BOTH baselines on BOTH held-out
and overall MAE, and the standardized creation-volume coefficient is
+0.176 (> 0, the physics check: more allocation cannot predict less
memory pressure). The creation feature is load-bearing: it beats its
own ablation on every cohort and lifts held-out correlation from
+0.18 to +0.32.

**Honest texture, recorded so no external claim overstates it:** the
big wins are on train and the frozen holdout (corr +0.92/+0.96); the
FNV held-out cohort is genuinely hard (corr +0.32) and the MAE margin
there is thin (0.0412 vs 0.0420). The head is a real, shadow-verified
improvement over both baselines — but it is a modest one on the
hardest cohort, and it ships SHADOW-ONLY (attached to nothing).

### 4.3 The existing heads are undisturbed — byte-identically

The decisive safety result: retraining the thermal (CPB0) and energy
(CPB1) heads on the schema-v4 corpus reproduces the committed bundles
**byte-for-byte** (sha256 before == after). Their feature bases don't
read the new field and their labels are unchanged, so the new feature
is provably invisible to them. Both still PROMOTE (thermal held-out
MAE 0.031, energy test regret +0.0014 beating both baselines). Plans
are byte-identical: the 6 selector wins reproduce at 0.49613–0.49855,
parity 100%, row-hash double-run stable.

## 5. Safety / regen invariants

- **FeatureHash changes by design** (new feature in the fingerprint) →
  every row hash changes → full corpus regen → thermal + energy heads
  retrained and re-shadowed (their LABELS are unchanged, so quality
  must hold; the regen is mechanical, the verdicts are the check).
- Plans must stay byte-identical to F0 where the new feature doesn't
  enter a ranking decision — the v1 closed form ignores the field, and
  the selector's energy head is retrained on the same labels. Verified
  by the `cana-diagnostics` gates + the ablation corpus-plan gate.
- The memory head ships SHADOW-ONLY: a PROMOTE verdict authorizes
  nothing by itself. Activation (attaching it to `PinnPhysicalCostModel`
  as a memory axis) is a separate, later decision — exactly as the
  thermal head waited behind `--pinn-weights`.

## 6. Verdict on the hypothesis ledger

| Claim | Status after F1 |
|---|---|
| Static creation-site features close the F0 generalization gap | **PARTIALLY CONFIRMED** — R²(test) 0.019 → 0.088 (~4.6×); the feature is load-bearing (beats its ablation on every cohort) but absolute held-out predictability stays low. The memory signal is harder than thermal. |
| A memory head beats honest baselines | **MEASURED, PROMOTE** — beats train-mean climatology AND the no-creation ablation on both held-out and overall MAE; frozen-holdout corr +0.96. Ships shadow-only. |
| Adding the feature is safe for the existing heads | **CONFIRMED, byte-identically** — thermal + energy bundles retrain bit-identical; plans byte-identical; parity 100%. |

**Exit criterion (roadmap row F): MET** — a memory head exists with a
shadow verdict (PROMOTE), the feature ablation isolates its
contribution, and the existing stack is provably undisturbed.

**What this does NOT license:** activation. The head is attached to no
cost model. A memory axis in `PinnPhysicalCostModel` (analogous to
`with_thermal_head`) is a separate decision that needs (a) a use case
where memory pressure should change a pass plan, and (b) a measured
outcome effect — neither of which F1 provides or claims. The honest
one-line summary: *the compiler can now predict per-function
allocation pressure better than two baselines, shadow-verified, but
does not yet act on it.*

## 7. Next (the harder-cohort problem)

The FNV held-out corr of +0.32 says the static feature still
under-explains allocation volume on unseen shapes. The likely cause is
the one the feature can't fix: static loop-amplification (`1 + 7·depth`,
capped) is a crude stand-in for runtime trip counts, and allocation
volume is trip-count-dominated. A trip-count-aware amplification (where
countable loops contribute their actual bound, not a depth proxy) is
the natural F2 lever — it would sharpen the creation feature AND the
existing flops/bytes estimates, so it is a FeatureHash change of its
own, with the full regen+retrain ripple.
