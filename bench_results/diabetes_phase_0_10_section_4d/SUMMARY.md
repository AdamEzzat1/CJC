# Phase 0.10 §4.D — Belief-weighted per-leaf BLR prior

The original handoff hypothesis: weight each leaf's BLR prior by its Locke
BeliefScore, so noisy leaves get stronger regularization. Two blockers came
up. Part 1 fixed one of them; Part 2 (and the full hypothesis test) deferred.

## Part 1 — Locke's `?`-sentinel blindness

The diabetes-130 dataset stores missingness as the literal string `?`. Locke's
default `validate()` treats only `f64::NAN` (Float columns) as missing. So a
column like `weight` — 96.9% `?` in the actual data — shows
`missingness_score = 1.0` (= "perfect, no missingness") in Locke's per-leaf
output. That makes BeliefScore uninformative on this dataset, which blocks
any §4.D Part 2 work from being meaningful.

### Fix

`build_question_mark_null_masks(df)` at [`tests/abng/per_leaf_belief_diabetes130.rs:455`](../../tests/abng/per_leaf_belief_diabetes130.rs:455)
scans `Str` columns for `?` and builds a `NullMaskMap` (`BTreeMap<String, NullMask>`)
that Locke's `ValidateOptions.null_masks` consumes. `per_leaf_belief_with_masks`
re-maps these full-DataFrame indices to each leaf's slice index space.

### Result

| Metric | Naive (no null masks) | `?`-aware |
|---|---|---|
| `weight` recognised as `?`-heavy | NO | YES (96.9% rate) |
| Columns with `?` detected | 0 | 7 (race, gender, weight, payer_code, medical_specialty, A1Cresult, max_glu_serum-ish) |
| Min per-leaf `missingness_score` | 1.0000 | **0.9613** |
| Min per-leaf `overall` | 0.9025 | 0.8977 |

Per-leaf belief CSV (`?`-aware):

```
leaf_id,n_rows,overall,schema,missingness,drift,leakage,lineage,sample,duplication,constraint
1,9617,0.897765,0.340000,0.962121,1.000000,1.000000,1.000000,1.000000,1.000000,0.880000
2,6215,0.900305,0.360000,0.962436,1.000000,1.000000,1.000000,1.000000,1.000000,0.880000
3,2545,0.897762,0.360000,0.962098,1.000000,1.000000,1.000000,1.000000,1.000000,0.860000
4,1623,0.897662,0.360000,0.961294,1.000000,1.000000,1.000000,1.000000,1.000000,0.860000
```

Locke's belief vector now carries a real missingness signal.

## Subtler problem surfaced — leaves have *similar* belief

All four leaves' `missingness_score` is in [0.9613, 0.9625] — a 0.001 spread.
That's because the routing column (`time_in_hospital`) doesn't correlate with
data-quality variation. The 97% `?` rate in `weight` is roughly uniform
across hospital-stay durations.

**Implication for §4.D Part 2 (the actual belief-weighted prior):**
A target-MI-optimised routing may produce uniform per-leaf belief.
Belief-weighted regularization in that case is a no-op — all leaves get the
same multiplier. To get a *meaningful* §4.D Part 2 result, you need either:

1. **Use the harness's real categorical-transform routing** ([17, 9] or [17, 16]).
   Different routing features may separate leaves on quality dimensions too.
   Requires plumbing `run_trial` to return per-leaf train indices (~30 LOC).
2. **Add a quality-aware routing penalty.** Modify the MI selector to also
   reward feature-quality variation. Probably a Phase 0.11+ design.

## Part 2 — Belief-weighted per-leaf BLR prior

**Status: DEFERRED.** Two reasons:

1. **ABNG does not expose a per-leaf prior API.** `set_blr_prior` at
   [`crates/cjc-abng/src/graph.rs:1432`](../../crates/cjc-abng/src/graph.rs:1432)
   is graph-wide, one-shot, must-be-called-before-any-add_node. Storing a per-leaf
   prior would need: `set_blr_prior_for_node(node_id, ...)`, lifecycle relaxation
   (callable after `add_node`), new `AuditKind::BlrLeafPriorOverride` variant, new
   canary lock. Same multi-session profile as §4.C.

2. **As above — even with the API in place**, the routing chosen by the harness
   may produce uniform per-leaf belief on this dataset, making the experiment
   inconclusive at the Part-2 level. Demonstrating utility would require
   point #1 from the previous section (real-routing leaf indices) *before*
   the new ABNG plumbing is worth building.

## Files

- [`part1_question_marks.txt`](part1_question_marks.txt) — test run output (`?`-aware variant)
- [`per_leaf_belief_q_aware.csv`](per_leaf_belief_q_aware.csv) — 4-row per-leaf belief CSV

## Code

- `build_question_mark_null_masks` — [`tests/abng/per_leaf_belief_diabetes130.rs:455`](../../tests/abng/per_leaf_belief_diabetes130.rs:455)
- `per_leaf_belief_with_masks` — [`tests/abng/per_leaf_belief_diabetes130.rs:478`](../../tests/abng/per_leaf_belief_diabetes130.rs:478)
- `diabetes130_per_leaf_belief_with_question_marks` test — [`tests/abng/per_leaf_belief_diabetes130.rs:526`](../../tests/abng/per_leaf_belief_diabetes130.rs:526)

## Verdict

§4.D Part 1 is a concrete Locke contribution: it unblocks the per-leaf
belief signal on `?`-sentinel datasets (a common medical-data convention).
Future Locke versions could auto-detect `?` / `NA` / `-` / `NULL` / etc.
as sentinels — see also [`PHASE_0_10_SESSION_NOTES.md`](../../docs/abng/PHASE_0_10_SESSION_NOTES.md)
§"Insights" item #4.

§4.D Part 2 is *not* attempted this session — see "Status: DEFERRED" above.
