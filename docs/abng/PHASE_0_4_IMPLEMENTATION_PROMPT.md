# Phase 0.4 Implementation Prompt — `cjcl abng …` CLI + Quality Refinements

**Paste this whole file as the next assistant's task.**

---

You are continuing development of **ABNG** (Adaptive Belief Radix Graph)
in the CJC-Lang repository. Phase 0.3d shipped on 2026-05-07 with **303
integration + 227 in-crate tests passing**, snapshot magic `ABNG\x08`,
**65 `abng_*` builtins**, and **24 audit-kind tags `0x00..0x17`**. The
structural-decision engine (`abng_decide_step`) is functional with
defensible-but-simplified triggers.

## Step 0 — Read before doing anything

Read these files in order, **before** writing any code or even
proposing a design:

1. `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` ← source of truth (post-0.3d)
2. `docs/abng/PHASE_0_3D_DESIGN.md` (post-hoc, captures 0.3d sub-step decisions)
3. `docs/abng/PHASE_0_3D_IMPLEMENTATION_PROMPT.md` (previous phase's prompt)
4. `docs/abng/PHASE_0_3a_DESIGN.md`
5. `docs/abng/PHASE_0_3b_DESIGN.md`
6. `docs/abng/PHASE_0_3c_DESIGN.md`

Then survey the actual implementation. **Do not assume the design notes
match the code.** When they disagree, the code wins.

The architecture doc lists every frozen tag, magic byte, and field
order. **Read §7 (Do Not Change Assumptions) and §8 (Still
Undecided / Gaps) carefully** — every gap with a "0.4" deferral note
is in scope for this phase.

## Step 1 — Identify any incomplete Phase 0.3d work

Before layering Phase 0.4, scan for unfinished items:

- §8 of `ABNG_CURRENT_ARCHITECTURE.md` lists ~7 known gaps deferred to
  0.4. The decision-engine simplifications in §8.9, the merge-math
  deferral in §8.10, and the drift-trip auto-Unfreeze in §8.11 are
  the heart of Phase 0.4's quality refinements.
- Run the standard gates clean:
  - `cargo test -p cjc-abng --lib` (227 expected)
  - `cargo test --test abng` (303 expected)
  - `cargo test --test physics_ml --release` (107/0/2 ignored)
  - `cargo test --workspace --release --lib` (2,363 expected)
  - `cargo test --test prop_tests abng_decision` (4 properties pass)
  - `cargo test --test bolero_fuzz abng_decision` (4 targets pass)

If anything is broken at start, **stop and report**. Do not paper over
it.

## Step 2 — Phase 0.4 scope

Phase 0.4 has **three tracks** that can ship as separate sub-steps:

* **Track A: User-facing CLI** — `cjcl abng …` subcommands, JSON
  snapshot, log compaction, snapshot-version negotiation.
* **Track B: Quality refinements** — Welford-smoothed signatures,
  KL-merge, ΔNLL split, route-entropy grow, real NIG-aware merge
  math, drift-trip auto-Unfreeze.
* **Track C: Correctness + documentation fixes from earlier-phase
  review** (added 2026-05-07 after retrospective audit). Fixes
  semantic / naming / hardening issues that survived 0.3d. **This
  track is the highest-priority — it patches real bugs and API
  confusions that affect every existing ABNG user.** See §2.4.

Tracks A, B, and C can ship in any order. Suggested ordering:
**Track C first** (correctness before features), then **Track B**
(quality before user-facing surface), then **Track A** (CLI ships
on the cleaned-up engine). Track A is the natural last step because
it freezes the snapshot format for external users — every Track-C
or Track-B change that touches persistent state should land *before*
the format freezes.

### 2.1 Track A — `cjcl abng …` CLI

| Subcommand | Purpose | Sketch |
|---|---|---|
| `cjcl abng train --config x.toml --seed 42` | Driver for end-to-end training that wires observe → decide_step → checkpoint | Reads TOML config (seed, policy thresholds, leaf-head shape, observation source). Produces a `.abng` snapshot at the end. |
| `cjcl abng inspect model.snap [--node ID] [--audit] [--stats] [--tree]` | Read-only snapshot viewer | Prints chain head, action counts, per-node maturity / signature / BLR posterior summaries. |
| `cjcl abng explain prediction.snap` | Full RouteEvidence + per-leaf BLR coefficients + abstain reason | Requires `Routed` audit events (§8.6). |
| `cjcl abng replay model.snap --log audit.log --verify` | Run the snapshot's audit log through `replay()` and assert chain integrity | Pure verification — no graph mutation. |
| `cjcl abng diff a.snap b.snap` | Structural diff between two snapshots | Compare topology, action_counts, per-node fingerprints. |

**Additional Track A items:**
- JSON-safe snapshot view via `cjc_snap::snap_to_json` adapter
- Log compaction: squash N consecutive `*Updated` events into one
  `StatsSnapshot` (rebased delta chain)
- Snapshot-version negotiation diagnostics ("this is v3, please
  upgrade") — see architecture doc §8.8
- **Freeze the snapshot format for the CLI's lifetime** (likely as
  v9 with all known 0.4 additions consolidated into a single bump).

### 2.2 Track B — Decision-engine quality refinements

Each item improves the **signal quality** of an existing trigger
without changing the contract surface (no new audit kinds, no
snapshot bump per item). Bundle them into one snapshot bump at end of
Track B.

#### 2.2.1 Welford-smoothed `NodeSignature` profiles
Replace the lazy sha256-truncate-of-state with persistent Welford-
folded summaries per profile (prediction / uncertainty / calibration
/ routing). Each profile owns a Welford accumulator (mean+variance);
the 8-byte signature is `sha256(canonical_bytes_of_welford_state)[..8]`.

**Why:** the current strict signature changes on every observation,
which makes `signature_stable_calls` strict. Welford-smoothing makes
"stability" mean "the running summary is no longer changing
significantly" — which is what the prompt §2.2 originally specified.

**Implementation note:** new persistent state per node (~128 bytes:
4 × Welford-state). Snapshot bump.

#### 2.2.2 3-window ECE / σ stability buffers for `Maturity`

Replace the single-threshold `calibration_stable` / `uncertainty_stable`
with 3-window buffers per the prompt §2.1:
- `calibration_stable`: 3 consecutive snapshot windows of `|ΔECE| < 0.005`
- `uncertainty_stable`: epistemic σ stable to within 5% over 3 windows

Each window is one `decide_step` call. Add ring buffers
`ece_history: [f64; 3]` + `sigma_history: [f64; 3]` per node + fill
counters.

#### 2.2.3 KL-divergence gate for Merge

Add the missing `kl_merge` gate to the Merge trigger. KL between two
NIG posteriors has a closed form:
`KL[N(m1, Σ1) ‖ N(m2, Σ2)] = ½ ( log|Σ2|/|Σ1| − d + tr(Σ2⁻¹Σ1) + (m2−m1)ᵀΣ2⁻¹(m2−m1) )`.
Use the BlrState's `mean` + `precision` as Σ⁻¹. No new persistent
state needed.

#### 2.2.4 Bootstrap held-out ΔNLL gain for Split

Add the missing ΔNLL + impurity gates to the Split trigger:
- Held-out via deterministic stratified bootstrap (SplitMix64-seeded
  from `(graph.seed, node_id, decide_step_call_count)`)
- Compare NLL before vs after a hypothetical split into 2 children
- Threshold against `policy.nll_split_gain()` and `policy.impurity_min()`

This is the most computationally expensive refinement in Track B
(per-decide_step bootstrap for every Split candidate).

#### 2.2.5 Route-entropy gate for Grow

Compute Shannon entropy of the route key-byte distribution at the
candidate node's depth. Compare to `policy.h_grow()`. Requires the
codebook to be installed (skip Grow on codebook-less graphs).

#### 2.2.6 Real NIG-aware merge math

`force_merge` and policy-driven Merge currently only set
`absorbed.is_active = false`. Phase 0.4's version should:
1. Combine BLR posteriors: `Λ_into = Λ_into + Λ_absorbed`,
   `m_into = Λ_into⁻¹ (Λ_into · m_into + Λ_absorbed · m_absorbed)`,
   `a_into = a_into + a_absorbed - a_prior`,
   `b_into = b_into + b_absorbed - b_prior` (subtract prior to avoid
   double-counting).
2. Combine stats: Welford merge of (n, mean, M2) — see Chan/Golub/
   LeVeque parallel Welford combine.
3. Then mark absorbed inactive.

**Snapshot impact:** none (combines existing fields). **Test impact:**
new in-crate tests for the combine math.

#### 2.2.7 Drift-trip auto-Unfreeze inside `decide_step`

Add a 7th step to `decide_step`'s per-node ladder (after Freeze):
if `is_frozen && drift_score > policy.drift_unfreeze_threshold`,
fire Unfreeze. Requires extending `DecisionPolicy` with a 12th
threshold — but the architecture doc §3.2 says DecisionPolicy is
one-shot, so this is a snapshot-format change.

**Implementation note:** either extend `DecisionPolicy` to 12
thresholds (snapshot bump) OR hard-code a sensible default (no bump,
deferred-real). Choose with the user.

### 2.3 Track C — Correctness + documentation fixes from earlier-phase review

A retrospective audit on 2026-05-07 (post-0.3d) surfaced a set of
correctness, naming, and hardening issues that survived all prior
phases. Track C is the patch list. **Each item is a real bug or
real API confusion that affects existing users — none are
quality-of-life polish.** Fix order is roughly priority-descending.

#### 2.3.1 BLR `predict()` returns dimensionless leverage, not variance contribution
**Severity: HIGH (API correctness).** `BlrState::predict()` returns
`(mean, epistemic_var, aleatoric_var)` where `epistemic_var =
‖L⁻¹φ‖² = φᵀΛ⁻¹φ` is **dimensionless leverage**, not variance in
output units. The original 0.3b design note's predictive-variance
formula `total = aleatoric_var × (1 + epistemic_var)` treats it as
leverage internally, but the API name strongly suggests output-unit
variance. Any external consumer who computes `total = epi + ale`
(plausible mental model) gets the wrong answer.

**0.3d-2's `expected_epistemic` capture and 0.3d-4's auto-capture
also store the leverage value**, so the calibrated OOD ratio
`(epi / expected).clamp(0, 1)` works on its own terms (units cancel).
But the misnaming is load-bearing for any external use.

**Fix options (pick one):**
- **(a)** Rename the second return value to `epistemic_leverage`
  in the API + docs. Snapshot-stable; just a builtin-rename
  (`abng_blr_predict` returns same `Tensor[3]`, but documented
  fields change). Forces minor disruption but no math change.
- **(b)** Change the math: return `aleatoric_var × ‖L⁻¹φ‖²` so the
  middle slot is variance contribution in y-units. Requires
  re-capturing every existing `expected_epistemic` (since auto-capture
  formula changes), so a one-time migration step. Forces 0.3d-4 stored
  values to be invalidated.
- **(c)** Return both: change the API to `Tensor[4]` with `(mean,
  leverage, epistemic_var_in_y_units, aleatoric_var)`. Breaking change
  to the dispatch arm; clearest API.

**Recommended:** (a). It's the smallest viable correctness fix and
preserves all existing snapshots / captured `expected_epistemic`
values. Document the rename in §6.5 of the architecture doc.

#### 2.3.2 `observe()` accepts NaN/Inf — no input validation
**Severity: HIGH (silent corruption).** `AdaptiveBeliefGraph::observe(node_id, value: f64)`
does not validate `value`. A single `observe(0, f64::NAN)` call:
1. Welford updates produce NaN mean and M2 forever (no recovery)
2. NodeStats canonical bytes hash to a stable but-NaN-poisoned value
3. Replay verification PASSES (bytes are bit-identical) but the
   model is permanently broken
4. Subsequent `observe()` calls on the same node propagate NaN.

**Fix:** reject non-finite values at the boundary. Two options:
- **(a) Strict reject:** `observe()` returns `GraphError::ObserveNonFinite { value }`
  on `!value.is_finite()`. Forces user code to handle bad input.
- **(b) Canonical NaN:** quietly substitute a single canonical
  bit-pattern NaN (`f64::from_bits(0x7FF8000000000000)`) before
  applying. Preserves API but admits the foot-gun.

**Recommended:** (a). Reproducibility is sacred; surfacing bad
input to the caller matches the existing `BlrError::FeatureDimMismatch`
style. Apply the same fix to `density_observe`, `calibration_observe`,
and `blr_update`.

**Tests:** add `tests/abng/observe_validation_tests.rs` with one
test per affected method × {NaN, +Inf, -Inf}.

#### 2.3.3 Replay missing semantic invariant checks
**Severity: HIGH (security / corruption).** `replay()` validates the
hash chain but does NOT validate:
- `event.seq` is monotonic (0, 1, 2, ...)
- The first event is `Created` (and only the first)
- `event.epoch == graph.epoch` (header field)
- `event.stats_version` equals the live node's post-apply
  `stats_version`

An adversarial blob with consistent hashes but reordered seqs (or
multiple Created events, or missing Created) currently passes replay
silently. The `event.stats_version` check is the most concrete: an
attacker can swap `Updated` events for the same node and the chain
hashes still match because the per-event stats_hash is computed
from the live node state at apply time.

**Fix:** in `serialize.rs::replay()`'s event loop, after each
`apply_event()`:
```rust
if event.seq != expected_seq { return Err(DecodeError::NonMonotonicSeq { expected: expected_seq, got: event.seq }); }
if event.epoch != stored_epoch { return Err(DecodeError::EpochMismatch); }
let live_version = graph.nodes[event.node_id as usize].stats_version;
if event.stats_version != live_version { return Err(DecodeError::StatsVersionMismatch { node_id: event.node_id, at_seq: event.seq }); }
expected_seq += 1;
```

Plus a "Created must be first" precondition before the loop:
```rust
if n_events == 0 || matches!(stored_first_event_kind, AuditKind::Created) is false { return Err(DecodeError::CreatedMustBeFirst); }
```

New `DecodeError` variants: `NonMonotonicSeq`, `EpochMismatch`,
`StatsVersionMismatch`, `CreatedMustBeFirst`. All Phase 0.4 wire-format
additions; the existing `ChainMismatch` is too generic.

**Tests:** add adversarial-blob fixtures to `tests/bolero_fuzz/abng_decision_fuzz.rs`
and `tests/abng/replay.rs` for each new invariant.

#### 2.3.4 Silent `b < 0` clamp in BLR — no audit event
**Severity: MEDIUM (reproducibility).** `blr.rs:233-235`:
```rust
if b_new < f64::EPSILON {
    // Numerical floor — keep IG well-defined.
    b_new = f64::EPSILON;
}
```
This silent rescue hides where the InverseGamma posterior is
operating outside its assumptions. A user re-running training would
see identical bytes (great for replay) but no signal that the
update was rescued.

**Fix:** when the clamp fires, append a deterministic diagnostic
audit event (new tag `0x18 BlrNumericalRescue { reason: u8, b_pre_clamp_bits: u64 }`).
The event becomes part of the chain (replay still verifies). Users
who care can filter the audit log for these events to identify
unstable training regimes.

**Alternative:** make the clamp configurable via a new
`BlrPrior.b_floor_policy: enum { Clamp, Reject }` field (snapshot bump).
The Clamp default preserves current behaviour; Reject errors with
a new `BlrError::NumericalUnstable` so users opt into strict mode.

**Recommended:** ship the audit event (no policy switch). It adds
observability without breaking existing flows.

#### 2.3.5 MLP feature space drift after BLR install — no contract
**Severity: HIGH (model correctness).** `blr_features()` reads the
**current** params on each call. So if `leaf_set_param` updates
the MLP's penultimate-layer weights AFTER BLR posterior was trained,
the existing BLR posterior is now in the wrong feature space — its
`mean` and `precision` are conditional on the OLD features that
the MLP no longer produces.

There is currently NO contract enforcing one of:
1. Freeze MLP params after BLR install (forbid `leaf_set_param`)
2. Reset BLR posterior to prior after MLP update (auto-recover)
3. Version the feature hash and reject `blr_update` on stale features

**Fix (option 3, least disruptive):** add `feature_version_hash: [u8; 32]`
to `BlrState` (snapshot bump). Set on `BLR install` to
`leaf_head.params_hash`-equivalent. On `blr_update()`, recompute
the params hash and compare; if mismatched, return a new
`BlrError::FeatureVersionStale { stored, current }`. The user can
then either:
- Reset BLR to prior (`abng_reset_blr(node_id)` — new builtin) and
  retrain on the new features
- Re-train MLP and BLR jointly (existing pattern; just gates BLR
  update on params being stable)

**Tests:** `tests/abng/blr_feature_version_tests.rs`.

#### 2.3.6 `LeafParamsUpdated` event volume — add batch builtin
**Severity: MEDIUM (performance).** Each `leaf_set_param(node_id, k, t)`
call appends one `LeafParamsUpdated` event. A single optimizer step
that writes back W₁, b₁, …, W_out, b_out for an L-layer MLP fires
`2(L+1)` events — for a 2-layer head, that's 6 events per step.
A 100-epoch training loop on a 10-leaf graph with batched updates
fires ~6,000 events.

**Fix:** add `abng_leaf_set_params_batch(g, node_id, params: Tensor[])` —
takes the full param vector as one tensor, writes all in one
operation, fires ONE `LeafParamsUpdatedBatch` audit event (new tag
`0x19`). The single witness covers the post-update params hash for
the whole vector.

Dispatch surface: 67 → 68 (after Track C).

#### 2.3.7 Doc-quality: rename "per-leaf" to "per-node" throughout
**Severity: LOW (clarity).** The architecture doc and Phase 0.3a/b
design notes call the MLP head "per-leaf" — but the code calls
`init_params` for the root and every `add_node` / `force_grow` /
`force_split`. So **every node has params**, not just leaves.
Same for BLR posterior (`set_blr_prior` initializes root, then
every child via `add_node` / structural mutation).

**Fix:** find-and-replace "per-leaf MLP head" → "per-node MLP head"
and "per-leaf BLR head" → "per-node BLR head" in:
- `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` (§3.4, §6.4, §6.5)
- `docs/abng/PHASE_0_3a_DESIGN.md` (title + every body mention)
- `docs/abng/PHASE_0_3b_DESIGN.md` (title + every body mention)
- crate-level docs in `crates/cjc-abng/src/leaf_head.rs` and `blr.rs`

No code changes needed; behavior is already per-node.

#### 2.3.8 Lineage belief / inherited prior — design decision
**Severity: LOW (feature, not bug).** Each node's BLR prior is
independent — there's no "inherit parent's posterior as my prior"
mechanism. For a hierarchical / radix-tree shaped belief graph,
ancestor-conditioned prediction (e.g. "if my BLR has insufficient
evidence, fall back to my parent's") is a natural feature.

**Decision needed:** is this in 0.4 scope or 0.5? Suggest:
- 0.4: add `abng_blr_predict_with_fallback(node_id, phi, max_depth)`
  that walks up the parent chain when `node.blr.n_seen <
  fallback_threshold`. Pure read; no new audit kind.
- 0.5: full lineage belief (parent posterior used as prior at
  `add_node` time) — this requires a snapshot bump and new `BlrInitialized`
  semantics.

#### 2.3.9 `NodeStats::canonical_bytes` future-hostile for compaction
**Severity: MEDIUM (future-proofing).** `NodeStats::canonical_bytes()`
serializes only `m2.finalize()`, dropping the Kahan compensation
register. **As long as replay always reconstructs from the full
event log, this is fine** — the compensation register is rebuilt
on each `observe`. But Phase 0.4's log-compaction goal ("squash N
consecutive `*Updated` events into one `StatsSnapshot`") needs to
resume from the canonical bytes, which means the compensation
state must be canonical too.

**Fix:** when log compaction lands (Track A's `StatsSnapshot` audit
kind), also extend `NodeStats::canonical_bytes` to 32 bytes —
appending the compensation register's bit pattern. Snapshot bump.

**Alternative:** declare that compaction always re-derives a fresh
KahanAccumulator from the finalized M2 (zero compensation), and
documents the implied determinism gap (not bit-identical to a
non-compacted history). Cleaner but admits a small reproducibility
asterisk.

**Recommended:** extend canonical bytes. Reproducibility is sacred.

#### 2.3.10 Cholesky regularization — design-vs-code drift
**Severity: LOW (doc-only).** PHASE_0_3b_DESIGN.md §"Numerical
safeguards" claims Cholesky uses `f64::EPSILON` diagonal
regularization before decomposition. The actual `cholesky()`
function in `blr.rs` does NOT regularize — it errors with
`BlrError::NonPositiveDefinite` on a non-positive pivot. The code
is correct (no silent regularization → reproducibility intact);
the design note is wrong.

**Fix:** delete the regularization claim from PHASE_0_3b_DESIGN.md
and add a "Phase 0.3b post-hoc correction" note to
`ABNG_CURRENT_ARCHITECTURE.md` §6.5 explaining the divergence.

#### 2.3.11 Empty-graph chain-head wording
**Severity: LOW (doc clarity).** The original PHASE_0_1_DESIGN.md
description says an "empty graph" has chain head equal to
`genesis_hash()`. But `AdaptiveBeliefGraph::new(seed)` immediately
constructs the root and appends a `Created` event — so a
user-visible "fresh graph" has `audit_len == 1` and a non-genesis
chain head. The genesis hash is only the *internal pre-creation
state* used as the `previous_hash` of the Created event.

**Fix:** add a clarification block to `ABNG_CURRENT_ARCHITECTURE.md` §2.2:
> **Genesis vs fresh-graph head.** `genesis_hash()` is the
> well-known constant `sha256(b"ABNG-GENESIS-v1")` used as the
> `previous_hash` field of the very first `Created` event. After
> `AdaptiveBeliefGraph::new(seed)` returns, the graph already has
> one Created event applied, so `chain_head != genesis_hash()` —
> a "fresh graph" has audit length 1 and a Created-derived chain
> head. The genesis constant is purely an internal anchor.

#### 2.3.12 Independent-audit findings (additional)

The 2026-05-07 audit also surfaced these smaller items:

- **`expected_epistemic` re-capture not supported.** Once captured,
  it's frozen (one-shot per node). If the BLR posterior drifts
  significantly later (e.g. catastrophic forgetting), the captured
  reference becomes stale and the calibrated OOD ratio drifts with
  it. Phase 0.4 should add either a `force_recapture_expected_epistemic`
  builtin (test/manual) or auto-recapture inside `decide_step` when
  drift_score exceeds a threshold.

- **`Maturity` thresholds hardcoded.** `ECE_STABILITY_MAX = 0.05`
  and `UNCERTAINTY_STABLE_MIN_SAMPLES = 100` in `maturity.rs` are
  compile-time constants, not user-configurable. Phase 0.4's
  `DecisionPolicy` extension should add `ece_stability_max` and
  `uncertainty_min_samples` thresholds (12-threshold or 13-threshold
  policy depending on whether the drift_unfreeze_threshold from
  §2.2.7 also lands).

- **`Unfreeze` doesn't bump `action_counts`.** By design (architecture
  §7 #13), but this means there's no way to count un-freeze events
  programmatically. If 0.4's drift-trip auto-Unfreeze fires often,
  observability suffers. Recommend: extend `action_counts` to
  `[u64; 7]` (snapshot bump) with index 6 = Unfreeze, OR keep `[u64; 6]`
  and add a separate `unfreeze_count: u64` graph field.

- **Auto-capture races with manual capture.** If user code calls
  `abng_set_expected_epistemic` between two `decide_step` invocations,
  the second `decide_step` sees `expected_epistemic.is_some()` and
  skips its auto-capture path. This is correct (one-shot honored),
  but worth documenting in §3.7 of the architecture doc.

- **No determinism canary for `decide_step`.** The property tests
  cover decide_step monotonicity, but a single dedicated
  `decide_step_chain_head_canary` in-crate test would catch any
  regression in the engine's hash-output earlier than the property
  suite does. Recommend: add this test alongside Track B's quality
  refinements (since those touch decide_step's internals heavily).

- **`force_compress` orphans descendants** (already in §7 #12) —
  but `decide_step`-driven Compress in 0.4 should at minimum mark
  those orphans `is_active = false` to avoid them showing up in
  `node_count` for graph-level metrics. Currently they stay active.

### 2.4 New audit kinds (consolidated across all tracks)

Tag-allocation order — Track C correctness fixes get the lowest
tags so they don't depend on the bigger format additions in Track A.

| Tag | Kind | Track | Payload | Purpose |
|---|---|---|---|---|
| `0x18` | `BlrNumericalRescue` | C-2.3.4 | `{ reason: u8, b_pre_clamp_bits: u64 }` (9B) | Diagnostic when `b < ε` clamp fires |
| `0x19` | `LeafParamsUpdatedBatch` | C-2.3.6 | `{ params_hash: [u8; 32] }` (32B witness) | Single event for full-vector param writeback |
| `0x1A` | `StatsSnapshot` | A | `{ stats_canonical: [u8; 32], window_seq_range: (u64, u64) }` | Log compaction — rebases N `*Updated` into one |
| `0x1B` | `Routed` | A | `{ leaf: u32, matched_prefix: u8 }` | Opt-in explain mode (per-call trace) |
| `0x1C` | `ProvenanceStamped` | A | `{ dataset_hash, feature_transform_hash, model_hash }` (96B) | Lineage; see arch doc §6.7 |

`0x16` (Unfreeze) and `0x17` (ExpectedEpistemicCaptured) are already
allocated (Phase 0.3d). Note Track-C tags `0x18..0x19` come *before*
Track A's tags so that correctness fixes can ship independently
without depending on Track A landing.

### 2.5 New builtin surface (estimated)

**Track A:** ~12 new builtins for the CLI subcommands (each
subcommand binds to one or more `abng_cli_*` builtins).

**Track B:** ~3 new builtins:
- `abng_signature_welford_dump(g, node_id) -> Tensor` (inspect)
- `abng_kl_divergence(g, node_a, node_b) -> Float` (BLR KL helper)
- `abng_route_entropy(g, node_id) -> Float`

**Track C:** ~5 new builtins (correctness):
- `abng_leaf_set_params_batch(g, node_id, params: Tensor[]) -> Void` (§2.3.6)
- `abng_reset_blr(g, node_id) -> Void` — reset to prior, fires `BlrInitialized` (§2.3.5 fallback)
- `abng_blr_predict_with_fallback(g, node_id, phi, max_depth: i64) -> Tensor[3]` (§2.3.8 — 0.4 sub-feature)
- `abng_observe_validate(g, value) -> Bool` — pure-function probe (optional convenience)
- `abng_force_recapture_expected_epistemic(g, node_id) -> Void` — manual override of 0.3d's one-shot (§2.3.12)

Total surface after 0.4: ~85 dispatch arms (65 + Track C +
Track B + Track A).

### 2.6 Snapshot v9

**Goal: this should be the LAST snapshot bump for ABNG's pre-1.0
lifecycle.** Bundle ALL of Phase 0.4's persistent-state additions
into one bump:

```
magic                    "ABNG\x09"
... v8 layout ...
# Track C correctness state additions:
+ NodeStats: extend canonical_bytes 24B → 32B (append Kahan
  compensation register) — Track C-2.3.9
+ BlrState: gain `feature_version_hash: [u8; 32]` — Track C-2.3.5
+ DecodeError: new variants NonMonotonicSeq, EpochMismatch,
  StatsVersionMismatch, CreatedMustBeFirst — Track C-2.3.3
# Track B quality state additions:
+ per-node: Welford state for 4 NodeSignature profiles (~128 bytes)
+ per-node: ECE history [f64; 3] + σ history [f64; 3] + fill counters
+ DecisionPolicy: gain a 12th threshold (drift_unfreeze) → 96 bytes canonical
+ optional 13th threshold for ece_stability_max + 14th for
  uncertainty_min_samples (Track C-2.3.12 audit findings)
# Track A — events gain 0x1A..0x1C; per-node gains optional
# `provenance_stamp_hash: [u8; 32]` (Track A — see §6.7).
... events gain 0x18..0x1C as needed ...
```

Once shipped, **freeze the format**. Phase 0.5+ should treat v9 as
permanent. Every Track-C / Track-B persistent-state addition MUST
land before the format freezes — no v10 in the pre-1.0 lifecycle.

## Step 3 — Testing requirements

For Phase 0.4 to be considered shipped, all of these must be green:

### 3.1 Unit tests

Each new module / extension gets `#[cfg(test)] mod tests` with at
least:
- `determinism_double_run` — same input → same bytes
- `canonical_bytes_size` — pin the wire shape
- `state_hash_changes_after_*` — for any new mutation
- For each refined trigger: `triggers_when_X`, `does_not_trigger_when_Y`

### 3.2 Integration tests under `tests/abng/`

- `tests/abng/cli_tests.rs` — end-to-end driver tests for each `cjcl abng` subcommand
- `tests/abng/welford_signature_tests.rs` — Welford-smoothed signatures stable under small perturbations
- `tests/abng/merge_math_tests.rs` — NIG-aware merge correctness
- Extend `dispatch_p3d.rs` and `parity_p3d.rs` for the new builtins (Track B's 3 + any Track A inspection builtins)

### 3.3 AST↔MIR parity

Every new `abng_*` builtin must have at least one parity test that
runs a `.cjcl` snippet through both backends and asserts byte-identical
printed output.

### 3.4 Property tests — extend `tests/prop_tests/abng_decision_props.rs`

Add at least 2 new properties:
- **KL-merge symmetry:** `kl_divergence(a, b) ≥ 0` and equals 0 iff posteriors are bit-identical
- **Welford-smoothed signature stability:** observing a stationary stream, the signature's Hamming distance to itself shrinks over time

### 3.5 Bolero fuzz — extend `tests/bolero_fuzz/abng_decision_fuzz.rs`

Add at least 1 new target:
- **CLI fuzz** (Track A): random TOML configs feeding `cjcl abng train` must produce well-defined errors, never panic

### 3.6 Replay / determinism

The v8 → v9 break must:
- reject all v1..v8 magics with `BadMagic`
- byte-round-trip every test case shipping in 0.3d
- have a single `v9_magic_in_blob` + `v8_magic_rejected` pair in
  `serialize.rs::tests`

### 3.7 Regression — earlier phases

After all 0.4 changes, every gate from 0.3d-5 must still pass with
counts only **growing**, never shrinking.

### 3.8 Run order for the gate

```bash
# Fast feedback loop while developing
cargo check -p cjc-abng
cargo test -p cjc-abng --lib
cargo test --test abng

# Pre-commit regression gate
cargo test --test physics_ml --release
cargo test --workspace --release --lib

# Property + fuzz (slow; run before merge)
cargo test --test prop_tests --release
cargo test --test bolero_fuzz --release
```

## Step 4 — Hard constraints (do not violate)

These come from `ABNG_CURRENT_ARCHITECTURE.md` §7 and the project
`CLAUDE.md`. Re-read before each commit.

1. Do not weaken replay verification to make tests pass.
2. Do not introduce `HashMap`/`HashSet` in canonical paths.
3. Do not silently change audit tag bytes, snapshot encoding, or
   frozen API behavior. Tags `0x00..0x17` keep their meanings. New
   tags use `0x18..` with documented canonical payload bytes.
4. Do not change `MAGIC` without bumping the version byte.
5. Do not extend the `Value` enum.
6. Do not call `cjc_ad::GradGraph::new` inside cjc-abng.
7. Do not add FMA to any kernel that touches belief state.
8. Do not parallelize Welford / Kahan reductions without using
   `cjc_repro::BinnedAccumulator`.
9. Do not paper over a regression.
10. Do not break the `decide_step` iteration order or the trigger
    fall-through (architecture-doc §3.7 / §7 #11).
11. Do not unfreeze without a corresponding audit event (`0x16`).
12. Do not write to a frozen node's structural state.
13. Preserve ABNG's core philosophy: deterministic, auditable,
    replayable, locally inspectable, structurally adaptive.

If a feature would require breaking any constraint above, **stop**,
write a one-page note explaining the conflict, propose an alternative,
and ask before proceeding.

## Step 5 — When ambiguity arises

Always prefer:

- **Existing ABNG contract** over convenience.
- **Explicit error** over silent fallback.
- **One-shot freeze** over re-installable state.
- **Witness-only audit events** over full-payload events for
  per-batch updates (matches `*Updated` pattern).
- **Per-node arena order** over insertion-time iteration.

When a phase design note says X and the code says Y, the code wins.
Document the gap in this design note.

## Step 6 — Handoff requirement at end of Phase 0.4

When done, update:

1. `docs/abng/PHASE_0_4_DESIGN.md` — design note, post-hoc OK.
2. `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — bump phase status,
   update §1, §3, §6, §7, §8 as appropriate.
3. `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 *.md` — append a
   "Phase 0.4 amendment" section.
4. `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md` —
   prepend the new "Phase 0.4 deliverable" block.
5. `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md` — replace
   the ABNG entry with a one-line pointer to the updated project memory.

Then, if Phase 0.5 is starting, write a
`docs/abng/PHASE_0_5_IMPLEMENTATION_PROMPT.md` for the next phase.

---

## Phase 0.5+ scope (not for this session, listed for context)

After Phase 0.4, the remaining work is:

### 0.5 — Chess-RL retrofit

- Replace value head only first (Phase 0.5a)
- Then policy head (Phase 0.5b)
- Uncertainty-gated A2C/PPO bootstrap: when `epistemic > τ_abstain`,
  fall back to plain MC return instead of bootstrapping a
  confidently-wrong critic
- End-to-end determinism gate: chess-rl-v2's `weight_hash` must
  remain bit-identical for the non-ABNG comparison runs

---

*Reminder: ABNG's value isn't the math, it's the audit. Don't let
Phase 0.4's quality refinements silently break the chain.*
