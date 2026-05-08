# ABNG Phase 0.4 — Trigger Refinement, Audit Fixes, CLI Surface (Post-Hoc Design Note)

**Date:** 2026-05-07
**Status:** Tracks B + C SHIPPED. Track A (CLI) is the remaining piece.
**Builds on:** [Phase 0.3d](PHASE_0_3D_DESIGN.md)
**Scope:** prompt-spec refinement of every `decide_step` trigger
(Track B), post-0.3d audit fixes surfaced by independent retrospective
(Track C), and the user-facing `cjcl abng …` CLI (Track A — pending).

## Why post-hoc

The original implementation prompt (`PHASE_0_4_IMPLEMENTATION_PROMPT.md`)
described Phase 0.4 as "CLI + quality refinements" — a single chunk
expected to land as one bump (likely `\x09`). In practice the work
split into three independent tracks once the audit findings clarified:

| Track | Scope | Items | Snapshot | Audit kinds added |
|---|---|---|---|---|
| **C** | post-0.3d audit fixes (independent retrospective) | 7 (C-2.3.1..C-2.3.7) | `\x08 → \x09` (C-2.3.5 only) | `0x18`, `0x19` |
| **B** | trigger refinement to prompt-spec form | 7 (B-2.2.1..B-2.2.7) | `\x09 → \x0A` (B-2.2.7 only, absorbs B-2.2.{1,2} state in-place) | (none) |
| **A** | `cjcl abng …` CLI + JSON view + log compaction | 5 subcommands + 3 supporting builtins | (none — extends v10 in place) | `0x1A..0x1C` (planned: StatsSnapshot, Routed, ProvenanceStamped) |

Two magic-byte bumps within Phase 0.4 (`\x08 → \x09 → \x0A`) means the
"single bump per phase" goal slipped — but v10 absorbs both Stage B
state additions in-place (B-2.2.{1,2} per-node Welford + ring buffers
attached to the same magic bump as B-2.2.7's `drift_unfreeze`
threshold), so v10 is the final magic for Phase 0.4. Track A must
NOT bump again — its new audit tags `0x1A..0x1C` extend v10 in place.
Per-node `provenance_stamp_hash` is deferred to Phase 0.5 to keep
this contract.

## Track C — post-0.3d audit fixes

The audit was performed independently after Phase 0.3d shipped. It
produced 12 findings; 7 became Track C work and shipped first. The
remaining 5 are tracked as Track C-2.3.{8,9,10,11,12} and may slip
to Phase 0.5 (see "What's *not* in Phase 0.4" below).

| Item | Scope | Wire-format change | Test file |
|---|---|---|---|
| **C-2.3.1** | BLR `predict()` rename → `epistemic_leverage` (the middle tuple slot is dimensionless leverage, not variance in y-units; pre-0.4 docs misnamed it `epistemic_var`) | none (rename + doc) | (existing tests updated) |
| **C-2.3.2** | NaN/Inf input rejection at 4 boundaries: `observe`, `density_observe`, `calibration_observe`, `blr_update`. New errors `GraphError::NonFiniteInput`, `BlrError::NonFiniteInput`. | none (new error variants) | `observe_validation_tests.rs` |
| **C-2.3.3** | Replay semantic invariants: seq monotonic, Created-first, epoch match, stats_version match. 4 new `DecodeError` variants (`SeqNonMonotonic`, `MissingCreatedEvent`, `EpochMismatch`, `StatsVersionMismatch`). | none (new error variants) | `replay_invariant_tests.rs` |
| **C-2.3.4** | BLR `b<ε` clamp audit event. `BlrState::update` returns `Result<Option<f64>, BlrError>`; on clamp, `blr_update` emits `0x18 BlrNumericalRescue { reason: u8, b_pre_clamp_bits: u64 }`. Replay treats it as no-op (diagnostic-only). | new audit tag `0x18` | `blr_numerical_rescue_tests.rs` |
| **C-2.3.5** | MLP/BLR `feature_version_hash`. `BlrState` carries `[u8; 32]` stamped from the per-node MLP params hash at every BLR-init site (`set_blr_prior`, `add_node`, `force_grow`, `force_split`). `blr_update` rejects with `BlrError::FeatureVersionStale { stored, current }` when current params hash differs. New builtin `abng_reset_blr` clears posterior to prior + refreshes fvh. | snapshot **v8 → v9** | `blr_feature_version_tests.rs` |
| **C-2.3.6** | `abng_leaf_set_params_batch(g, node_id, params: Tensor[]) -> Void` + new audit kind `LeafParamsUpdatedBatch` at tag `0x19`. Atomic — if any tensor's count or shape is wrong, params are unchanged and no event appended. 6× event reduction on a 2-layer head's optimizer step. | new audit tag `0x19` | `leaf_params_batch_tests.rs` |
| **C-2.3.7** | "per-leaf" → "per-node" doc rename across architecture doc, `PHASE_0_3a_DESIGN.md`, `PHASE_0_3b_DESIGN.md`, and crate-level docs. Code was always per-node; docs were wrong. | none (doc only) | (no new tests) |

**Builtin count:** 65 → 67 (`abng_reset_blr`, `abng_leaf_set_params_batch`).

## Track B — trigger refinement

The prompt's §2.3 trigger spec listed rich criteria (Welford-smoothed
signatures, 3-window stability, KL-merge, ΔNLL split, route-entropy
grow, NIG-aware combine, drift-trip Unfreeze). Phase 0.3d-4 shipped
defensible single-threshold simplifications — Track B replaced each
with the spec form.

The implementation order matters: B-2.2.6 (NIG-aware merge math)
shipped first as a *compute-only* primitive, then B-2.2.{3,4,5} as
*compute-only* gates that don't change wire format, then B-2.2.7
bumped magic to `\x0A` and introduced the 12th `drift_unfreeze`
threshold. B-2.2.{1,2} were resequenced *after* B-2.2.7 and absorbed
their per-node state additions into the same v10 bump in-place.

| Item | Scope | Wire-format change | Test file |
|---|---|---|---|
| **B-2.2.6** | NIG-aware merge math: `BlrState::combine(&mut self, other, prior)` (sum precisions, precision-weighted-mean of means, `(a, b)` with prior subtract) and `NodeStats::combine(&mut self, other)` (Chan/Golub/LeVeque parallel Welford merge). `force_merge` and replay-side `apply_event(Merge)` both call `combine` before deactivating absorbed. | none (combines existing fields; emits extra `BlrUpdated` witness on the `into` node) | `merge_math_tests.rs` |
| **B-2.2.3** | KL-divergence gate for Merge: `BlrState::kl_divergence(&self, &other)` (closed-form for d-D Gaussian + scalar IG). `try_merge` requires both Hamming ≤ τ_merge AND posterior `KL ≤ kl_merge`. | none | (covered in `decision_tests.rs` updates) |
| **B-2.2.5** | Route-entropy gate for Grow: `route_key_entropy_at_candidate_depth(...)`. `try_grow` requires both `samples_seen ≥ grow_min` AND route-key entropy at candidate depth > `H_grow`. | none | `route_entropy_grow_tests.rs` |
| **B-2.2.4** | Bootstrap held-out ΔNLL gain for Split: `estimate_split_nll_gain(...)` uses synthetic Gaussian sampling from the BLR posterior (deterministic via SplitMix64-derived seed). `try_split` requires both `samples_seen ≥ split_min` AND ΔNLL gain ≥ `nll_split_gain`. | none | `split_nll_gate_tests.rs` |
| **B-2.2.7** | Drift-trip auto-Unfreeze: when a frozen node has both a density tracker and a drift baseline, and `drift_score(current_density) > drift_unfreeze`, `decide_step` calls `unfreeze` before the regular ladder. New 12th `DecisionPolicy.drift_unfreeze` threshold. Default `f64::MAX` keeps the gate disabled. | snapshot **v9 → v10**; `DecisionPolicy` 88B → 96B | (covered in `decision_tests.rs` updates) |
| **B-2.2.2** | 3-window ECE/σ stability ring buffers per node: `ece_history: [f64; 3]`, `ece_fill_count: u8`, `sigma_history: [f64; 3]`, `sigma_fill_count: u8`. `Maturity::from_node` flips `calibration_stable` on `max-min ≤ ECE_STABILITY_MAX_DELTA` and `uncertainty_stable` on `max/min ≤ SIGMA_STABILITY_RATIO`. | per-node +50B (absorbed into v10 in place) | `maturity_signature_tests.rs` updates |
| **B-2.2.1** | Welford-smoothed NodeSignature profiles per node: 4 × `SignatureWelford { n: u64, mean: f64, m2: f64 }` channels (prediction/uncertainty/calibration/routing). `NodeSignature::from_node` reads the four Welford means and packs canonical f64 bit patterns into `[u8; 32]`. Stability becomes lenient — small post-stability observations no longer reset the counter. `decide_step` advances all four channels per the §3.7 step 3 contract, even for frozen / inactive nodes. | per-node +96B (absorbed into v10 in place) | `maturity_signature_tests.rs` updates |

## Track A — `cjcl abng …` CLI (PENDING)

Tracking outline only — the CLI is the next concrete deliverable.

| Subcommand | Purpose | Test target |
|---|---|---|
| `cjcl abng train --config x.toml --seed 42` | Driver: observe → `decide_step` → checkpoint loop | `tests/abng/cli_tests.rs` |
| `cjcl abng inspect <model.snap> [--node ID] [--audit] [--stats] [--tree]` | Read-only viewer | same |
| `cjcl abng explain <prediction.snap>` | Lineage + abstain reason. Requires `Routed` audit events (`0x1B`). | same |
| `cjcl abng replay <model.snap> --log <audit.log> --verify` | Wrapper around `abng_replay` | same |
| `cjcl abng diff <a.snap> <b.snap>` | Topology + per-node fingerprint diff | same |

Supporting builtins (not yet shipped):
- `abng_predict_snap(g, node_id, x_idx) -> Bytes` for `cjcl abng explain`
- `abng_compact_log(g, until_seq) -> Void` for `cjcl abng explain` log compaction
- JSON snapshot adapter via `cjc_snap::snap_to_json`

Audit tags allocated for Track A (extend v10 in place, no magic bump):
`0x1A StatsSnapshot`, `0x1B Routed`, `0x1C ProvenanceStamped`.

## Decisions worth recording

### D1. Two magic bumps within Phase 0.4 — but v10 still absorbs both Stage B state additions

The pre-0.4 design intent was to consolidate every 0.4 wire-format
change into one bump (likely `\x09`). C-2.3.5's
`feature_version_hash` shipped first and bumped `\x08 → \x09`
because the audit fix was urgent. B-2.2.7's `drift_unfreeze`
12th threshold then bumped `\x09 → \x0A`. The B-2.2.{1,2} per-node
state additions (Welford + ring buffers, +146B per node) were
resequenced *after* B-2.2.7 and absorbed into the same v10 bump
in-place — so v10 is the final magic for Phase 0.4 and Track A
extends it in place. The "single bump per phase" rule is restored
once a phase is closed; mid-phase resequencing is acceptable when
the wire format ends up consolidated.

### D2. Track C ships before Track B (audit fixes have priority over feature work)

The independent post-0.3d audit surfaced 12 findings before any
trigger refinement was attempted. The audit findings ranged from
silent-data-corruption (NaN observations) to subtle invariant
holes (replay accepts reordered seqs). These ship first because
trigger refinement built on top of unsound primitives compounds the
problem. Order: C-2.3.1 (rename/doc) → C-2.3.2 (input validation) →
C-2.3.3 (replay invariants) → C-2.3.4 (clamp audit) → C-2.3.5 (fvh,
the one bump-causer) → C-2.3.6 (batch builtin) → C-2.3.7 (doc
rename), then Track B.

### D3. NIG-aware merge math is a *primitive*, not an *algorithm*

`BlrState::combine` and `NodeStats::combine` (B-2.2.6) are pure
functions of input state — sum precisions, precision-weighted means,
Chan/Golub/LeVeque parallel Welford merge. They live alongside
`update`, `predict`, `kl_divergence` in `blr.rs` / `stats.rs`.
The *decision* of whether to call combine is in `graph.rs`'s
`try_merge` (gated on Hamming + KL). This separation matches the
0.3d-3 D4 principle ("force-* mutations have minimal semantics") —
the policy decides *when*, the primitive decides *how*.

### D4. Welford signatures shift `signature_stable_calls` from strict to lenient

Pre-0.4, `NodeSignature` was a sha256-truncate of subsystem state —
*any* observation flipped the signature, so `signature_stable_calls`
reset to zero on every observation. This made `Maturity.uncertainty_stable`
unreachable in practice (the test suite had to install a policy with
`freeze_after = 1` to hit it). B-2.2.1's Welford fold makes the
signature change *gradually* — only when the running mean shifts
beyond Hamming sensitivity does the counter reset. The semantic
intent of "stable" matches the prompt's spec; the wire shape is
unchanged (still `[u8; 32]`); the read implementation in
`NodeSignature::from_node` is what changed.

### D5. Audit tags `0x18` / `0x19` are opt-in (don't fire in healthy training)

`0x18 BlrNumericalRescue` only fires on a clamp event, which only
happens when posterior `b` underflows to `< f64::EPSILON` — pathological
data. `0x19 LeafParamsUpdatedBatch` only fires when callers explicitly
use `abng_leaf_set_params_batch` (existing per-tensor `LeafParamsUpdated`
fires from `abng_leaf_set_param` as before). Healthy training
snapshots from pre-0.4 contain neither tag and replay byte-identical
through the new code paths. This was the discipline that allowed
shipping under v9/v10 instead of bumping again for these tags.

### D6. Track B compute-only items ship before Track B state-bump items

B-2.2.{3,4,5,6} all add gates / methods that compute over existing
state — no new fields, no snapshot bump. They shipped first because
they're independently testable and don't gate on the magic bump.
B-2.2.7 then introduced the 12th `DecisionPolicy.drift_unfreeze`
threshold (88B → 96B) and the magic bump. B-2.2.{1,2} resequenced
*after* B-2.2.7 piggybacked on the v10 bump for their per-node state
additions. This staging keeps each PR small and reversible.

### D7. `combine` emits one `BlrUpdated` witness on `into` (replay correctness)

When `force_merge` (or policy-driven Merge via `decide_step`) folds
absorbed's BLR posterior into `into`, the post-combine state of
`into` differs from the pre-merge state. Replay's per-node
`state_hash` matcher requires the most-recent witness event for
that node's BLR state to match the post-replay canonical bytes.
Absorbed's last `BlrUpdated` doesn't help — it's `into` whose state
changed. The implementation appends `BlrUpdated { state_hash:
into.blr.state_hash() }` to the audit log inside `force_merge` /
`apply_event(Merge)` so replay re-validates `into`'s post-combine
state. This is *not* a new audit tag — it reuses `0x0A`.

### D8. Decoder hardening for v10's longer per-node section

v9's per-node section was ~89 bytes. v10's is ~235 bytes (per-node
+50B for ring buffers + 96B for Welford channels = +146B). The
existing 0.3d-5 decoder bounds-checks (`with_capacity` capped at
remaining cursor) were re-validated against the larger layout via
`replay_invariant_tests.rs::n_events_offset` regression — bumped
200 → 346 to confirm.

## Frozen contracts established in Phase 0.4

These now sit alongside the 0.1/0.2/0.3a/b/c/0.3d invariants in
architecture-doc §7:

* Snapshot magic `\x0A` (Track A must NOT bump beyond this — audit
  tags `0x1A..0x1C` extend v10 in place)
* Audit tags `0x00..0x19` (26 kinds; `0x18`, `0x19` opt-in)
* Per-node ECE / σ ring buffers are `[f64; 3]` with saturating
  `u8` fill counts
* Per-node `SignatureWelford × 4` channels, canonical f64 bit
  patterns
* `DecisionPolicy` is exactly 12 thresholds (88B → 96B)
* `combine` math is pure (no I/O, no observability hooks); the
  audit witness comes from the caller
* BLR `feature_version_hash: [u8; 32]` is stamped at every
  BLR-init site (`set_blr_prior`, `add_node`, `force_grow`,
  `force_split`, `reset_blr`)
* All 4 input boundaries (`observe`, `density_observe`,
  `calibration_observe`, `blr_update`) reject NaN, +Inf, -Inf
* Replay validates seq monotonicity, Created-first, epoch match,
  stats_version match — in addition to all 0.3d-5 hash-chain checks

## Test-surface expansion

| Gate | Pre-0.4 (end-of-0.3d) | Post-0.4 B+C | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 227 | **252** | +25 |
| `cargo test --test abng` | 303 | **391** | +88 |
| Property tests (4 × 256 cases each) | 4 | 4 | +0 (re-tuned for 12 thresholds) |
| Bolero fuzz (4 targets) | 4 | 4 | +0 |
| `cargo test --workspace --release --lib` | 2,363 | (re-run before Track A merge) | TBD |
| `cargo test --test physics_ml --release` | 107 | (re-run before Track A merge) | TBD |

Total ABNG-direct `#[test]` markers: **643** (252 in-crate + 391
integration), plus 4 properties (× 256 cases each) and 4 fuzz targets.

New test files added in Phase 0.4:

| File | Item |
|---|---|
| `tests/abng/observe_validation_tests.rs` | C-2.3.2 |
| `tests/abng/replay_invariant_tests.rs` | C-2.3.3 |
| `tests/abng/blr_numerical_rescue_tests.rs` | C-2.3.4 |
| `tests/abng/blr_feature_version_tests.rs` | C-2.3.5 |
| `tests/abng/leaf_params_batch_tests.rs` | C-2.3.6 |
| `tests/abng/merge_math_tests.rs` | B-2.2.6 |
| `tests/abng/route_entropy_grow_tests.rs` | B-2.2.5 |
| `tests/abng/split_nll_gate_tests.rs` | B-2.2.4 |

Existing test files updated (key changes):

| File | Change |
|---|---|
| `tests/abng/decision_tests.rs` | `ok_thresholds()` 11 → 12 elements; varied-observation `decide_step_split_when_samples_high`; 3-window stability for `decide_step_auto_captures_expected_epistemic_at_uncertainty_stable` |
| `tests/abng/dispatch_p3d.rs` | `ok_thresholds_tensor` 11 → 12; `signature_changes_with_subsystem_install` updated for Welford semantics |
| `tests/abng/maturity_signature_tests.rs` | multiple tests updated for B-2.2.1 Welford signatures + B-2.2.2 ring buffers |
| `tests/abng/parity_p3a.rs` | added `parity_leaf_set_params_batch_*` tests (C-2.3.6) |
| `tests/abng/parity_p3b.rs` | added `parity_reset_blr_*` tests (C-2.3.5) |
| `tests/abng/parity_p3d.rs` | POLICY_INSTALL Tensor[11] → Tensor[12] |
| `tests/abng/replay.rs` | byte-offset comments updated for v10 per-node section size (89 → 235) |
| `tests/abng/replay_invariant_tests.rs` | `n_events_offset` updated 200 → 346 |
| `tests/prop_tests/abng_decision_props.rs` | `arb_thresholds` strategy 11 → 12 elements |

## What's *not* in Phase 0.4

### Track A (the CLI) — pending

The `cjcl abng …` subcommand surface is the largest remaining piece.
It does not bump magic — the new audit tags `0x1A..0x1C` extend v10
in place. Per-node `provenance_stamp_hash` is deferred to Phase 0.5
to keep this contract.

### Audit findings polish (Track C-2.3.{8,9,10,11,12}) — may slip to 0.5

Five lower-priority audit findings remain:

* **C-2.3.8** `abng_blr_predict_with_fallback` — read-only fallback
  walk up parent chain. Optional convenience builtin.
* **C-2.3.9** `NodeStats::canonical_bytes` 24B → 32B (append Kahan
  compensation register). Required for log compaction. **Snapshot
  bump if shipped** — conflicts with the v10 freeze contract above.
  Decision: ship only after the user explicitly OKs another bump,
  or defer to Phase 0.5.
* **C-2.3.10** Cholesky regularization design-vs-code drift.
  Doc-only correction in `PHASE_0_3b_DESIGN.md`.
* **C-2.3.11** Empty-graph chain-head wording in §2.2 of architecture
  doc.
* **C-2.3.12** Audit findings — `expected_epistemic` re-capture,
  configurable `Maturity` constants, `unfreeze_count` observability,
  `decide_step` chain-head canary.

### Property tests + fuzz targets for new audit kinds — partial coverage

Per the prompt §3.4, two additional properties were planned:

* KL-merge symmetry: `kl_divergence(a, b) ≥ 0` and equals 0 iff
  posteriors are bit-identical
* Welford-smoothed signature stability: stationary stream → Hamming
  distance to itself shrinks over time

Both are covered by the in-crate `cjc-abng --lib` Welford tests
(`stats::tests::welford_*`, `signature::tests::welford_*`) but not
yet promoted to property tests with 256-case sampling. Decision: ship
under existing property gates (4 × 256) for Phase 0.4; add new
properties in Phase 0.5 if needed.

### CLI fuzz target

Per the prompt §3.5, a fuzz target for random TOML configs feeding
`cjcl abng train` is part of Track A.

---

*This document is the post-hoc design note for Phase 0.4 Tracks B + C.
Track A (`cjcl abng …` CLI) is the next concrete deliverable; this
note will be amended once Track A ships, and the per-track status
in §1 of the architecture doc updated accordingly.*
