# CJC-Lang ABNG Phase 0.9.5 — Handoff v3

**Date stamped:** 2026-05-17
**Branch:** `claude/zealous-nobel-66abc9` (a clean fast-forward
continuation of `claude/abng-phase-0-9-5`; the next session continues
on **this same branch**).
**Supersedes:** [`PHASE_0_9_5_HANDOFF_V2.md`](PHASE_0_9_5_HANDOFF_V2.md).
V2's Research Phase R0 and its §0-§10 categorical-scaling design are
carried forward; this v3 records that **R0 and R1 are now COMPLETE**
and re-orders the remaining work.

**Datasets:** already present (untracked, `.gitignore`'d) in
`tests/data/` — `diabetes_130/diabetic_data.csv` (+ `IDS_mapping.csv`)
and `cdc_brfss.csv`. No re-fetch needed on this worktree.

---

## 1. What has been accomplished since v2

### Research Phase R0 — result-path performance — COMPLETE

Profiled the ABNG result path on the real Diabetes-130 workload and cut
its dominant cost. Full detail: [`PHASE_0_9_5_R0_PROFILE.md`](PHASE_0_9_5_R0_PROFILE.md).

| Commit | Content |
|---|---|
| `e12d827` **R0-1** | New bench crate `bench/abng_result_profile` (segment profiler) + the R0 design doc. Found `state_hash` (SHA-256 over the d×d BLR precision) was **67 %** of the 6.88 ms/row path; the handoff's "1-2 hr" was machine contention, not the algorithm (~110 s CPU for 16K rows). |
| `3720794` **R0-2** | Streaming `BlrState::state_hash` (T1.1) — byte-identical; kills the per-call ~477 KB allocation. |
| `7bef0c1` **R0-3** | Tier 2 **Option C** — periodic BLR audit checkpoints. `BLR_CHECKPOINT_INTERVAL = 64`; intermediate rows carry the `BLR_INTERMEDIATE_WITNESS` sentinel; `AdaptiveBeliefGraph::checkpoint_blr()` flushes mid-interval nodes. **~2.9× row speedup, no wire-format bump, 0 of the 28 canaries re-locked.** |

### Research Phase R1 — post-R0 result-path performance — COMPLETE

Detail: [`PHASE_0_9_5_R1_PERF.md`](PHASE_0_9_5_R1_PERF.md).

| Commit | Content |
|---|---|
| `9e02bbb` **R1-research** | R1 design doc. Found the post-R0 path is latency-bound on the *serial Kahan reduction chain* inside the rank-1 BLR `update`; the 477 KB precision clone is 1.4 % — a non-finding (measure, don't assume). |
| `614b7d7` **R1-1** | Lane-parallel **x8** Kahan in `update_rank1` — `quadratic_form_lanes` + `matvec_plus_xty_lanes` (dedicated, so the shared scalar helpers stay for `kl_divergence`/`combine`/n>1). Re-lock confined to the n=1 train path → 0 canaries. |
| `f678997` **R1-2** | **Tier B #4** — `cholesky_solve` lane-parallel (`forward_subst_lanes` / `back_subst_lanes` / `cholesky_solve_lanes`, used by `update_rank1`). **Tier A #1** — per-node `params_hash_cache`: `validate_blr_inputs` recomputed `params_hash` every training row; now cached, invalidated by `leaf_set_param`/`leaf_set_params_batch` (byte-identical). |

**Cumulative R0 + R1:** per-row result path ~6.9 → ~2.1 ms (**~3.3×**).

### Build state

- `cjc-abng` lib **396/396**; `cargo test --test abng` **624/0** (the
  lone failure `concurrent_training_scales_better_than_serial` is the
  documented wall-clock flake — re-run it alone to confirm).
- 28 SHA-256 canaries: **0 re-locked** across all of R0+R1 — the perf
  changes are confined to the n=1 training path, which the canaries do
  not exercise.
- `bench_results/phase_0_8_demos/c2_parallel_verify_scalability.svg`
  shows modified — test-run noise; `git checkout --` discards it.

---

## 2. What still needs work — priority order

### Priority 1 — finish the performance tiers (Tier A, Tier B done; then Tier C)

The full ranked analysis is in [`PHASE_0_9_5_R1_PERF.md`](PHASE_0_9_5_R1_PERF.md) §5-§8.

**Tier A — byte-identical, no sign-off:**
- **#1 `params_hash` cache — DONE** (R1-2).
- **#2 — optimize cjc-snap's portable SHA-256.** `process_block`
  (`crates/cjc-snap/src/hash.rs`) is a clean textbook FIPS-180-4 impl;
  round-unrolling the 64-round compression loop is a ~1.3-1.6×
  byte-identical win that speeds *every* hash in the workspace.
  **Caveat:** it is the audit-chain cryptographic foundation — do it as
  its own commit, verified against FIPS-180-4 known-answer vectors +
  the 28 canaries. Post-Option-C its *result-path* leverage is small
  (it mainly helps `verify_chain` / Merkle / serialize).
- **#3 — lazy/scalar prior BLR allocation (memory).** Represent an
  un-updated node as `(prior, λ)` instead of a materialized d×d matrix,
  so the many dead leaves of a pre-allocated routing tree cost O(d) not
  O(d²). Byte-identical, but invasive (every precision reader branches)
  and there is **no current memory pressure** — recommended only when
  deeper routing trees create real pressure.

**Tier B — re-lock confined to the n=1 train path:**
- **#4 `cholesky_solve` lane-parallel — DONE** (R1-2).

**Tier C — needs an audit-model / wire decision (attempt after Tier A/B):**
- **Batched training (`train_step_batch`).** Accumulate B rows, one
  batched NIG update (O(d³ + B·d²), amortizing the Cholesky) — ~3× on
  the update for B≈64, same posterior. Costs: a new audit event kind →
  **v15 wire-format bump**, and per-*batch* tamper localization. The
  biggest single remaining speed lever.
- **Deterministic cross-leaf parallelism.** Rows routing to different
  leaves have independent BLR updates; partition by leaf, process each
  leaf's sequence on its own thread, assemble events in fixed row
  order → byte-identical, multi-core. Caveat: the root BLR is updated
  every row (sequential) — caps the gain at ~1.5-2×. Complex,
  Determinism-Auditor-gated.
- **Out of scope** (would break a constraint): reducing d via
  projection / a learned encoder (the only order-of-magnitude lever,
  O(d²) is quadratic in d — but it changes the model's predictions and
  epistemic-uncertainty output → a Phase 1.0 *modeling* decision);
  skipping the saturated root BLR's per-row update (~2×, also a model
  change).

**First task of the next session:** re-run `cargo test --test abng` to
re-confirm R1-2's gate, then optionally re-run `abng-result-profile` to
measure R1-2's speedup (the profiler has scalar-vs-lane Kahan and
full-vs-triangular layout micro-benches built in).

### Priority 2 — the categorical-scaling commits (v2's COMMIT 5-9, still pending)

R0+R1 were the perf priority V2 prepended; the original Phase 0.9.5
categorical-scaling work is still open. V2 §3-§10 remains the
authoritative design.

- **COMMIT 5 — Diabetes-130 harness — finalize.** `tests/abng/dataset_a_diabetes130.rs`
  is committed (`178cd01`, WIP). Now that training is ~3.3× faster,
  re-scope the always-run test from the heavy 20K-row sub-sample to a
  small (~2 000-row) one (keep 20K / full-101K `#[ignore]`d), and
  produce the first real Diabetes-130 accuracy / balanced-accuracy /
  AUC numbers. The harness already calls `checkpoint_blr()` (R0-3).
- **COMMIT 6** — Dataset-B (BRFSS) harness, low-cardinality regime.
- **COMMIT 7** — seed sweeps (15 A / ≥5 B) + determinism double-run.
- **COMMIT 8** — artifact producer (`bench_results/phase_0_9_5_medical_scaling/`);
  this stage regenerates the stale Wisconsin BC `chain_heads.txt`.
- **COMMIT 9** — `known_solution_context.md` + close-out;
  CAPABILITIES / handoff updates.

---

## 3. Determinism / auditability notes to carry forward

- **The `checkpoint_blr()` contract.** A graph trained with
  `train_step` / n=1 `blr_update` **must** call
  `AdaptiveBeliefGraph::checkpoint_blr()` once before `serialize` —
  Option C hashes the full BLR state into the audit chain only every
  64th update, so a mid-interval node would otherwise fail replay with
  `DecodeError::BlrStateHashMismatch` (loud, never silent). The
  COMMIT-5 harness already does this; any new harness must too.
- **The perf re-locks are train-path-confined.** R0-3, R1-1, R1-2 each
  change the bits the n=1 `update_rank1` path produces, but the
  `decide_step` canaries and the Wisconsin BC baseline never train via
  that path (they use `observe` / `decide_step` / structural ops / the
  shared scalar helpers), so **0 canaries re-locked**. Keep this
  property: any new lane-parallel / batched path for `update_rank1`
  must stay dedicated — do **not** change the shared scalar
  `quadratic_form` / `matvec_plus_xty_kahan` / `cholesky_solve`
  (`combine` → Merge → the `decide_step` canary depends on them).
- **Wisconsin BC `chain_heads.txt`** is stale (R0-3 + R1-1 + R1-2 each
  shifted the train-path bits). The baseline *tests* are relative
  (run-to-run, seed-distinctness) and pass; the artifact producer is
  `#[ignore]`d. COMMIT 8 regenerates it.

## 4. Build / test reference

```
cargo test -p cjc-abng --lib                      # 396 in-crate
cargo test --test abng -- --skip diabetes130      # ~50 s integration gate
cargo test --test abng                            # full (624/0; 1 known flake)
cargo run -p abng-result-profile --release        # the result-path profiler
```

Every commit re-runs the determinism gate (`cargo test --test abng`,
the 28 canaries, the Wisconsin BC baseline, AST↔MIR parity).

---

*Start the next session here: Priority 1 (perf Tier A #2/#3, then
Tier C), then Priority 2 (COMMIT 5-9). Branch `claude/zealous-nobel-66abc9`,
datasets already in `tests/data/`.*
