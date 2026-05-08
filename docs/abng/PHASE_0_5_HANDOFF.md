# Phase 0.5 — Handoff for Next Session

**Date stamped:** 2026-05-08
**Branch (recommended):** `claude/abng-phase-0-5` (forked from `claude/abng-phase-0-4`)
**Worktree (recommended):** `C:\Users\adame\CJC\.claude\worktrees\abng-phase-0-5`
**To continue:** start a new Claude Code session inside the worktree above and paste the prompt at the end of this document (`## Recommended next-session prompt`).

---

## What's done — Phase 0.4 baseline + v11 extension

End-of-Phase-0.4 + v11 extension state (commits `29e1cca` … `2f65ccb` on
`claude/abng-phase-0-4`):

### Surface

| Property | Value |
|---|---|
| Snapshot magic | `b"ABNG\x0B"` (v11) |
| Builtin count | **73** `abng_*` arms in `dispatch.rs` |
| Audit kinds | **28** (tags `0x00..0x1B`; `0x1C` reserved for Phase 0.5) |
| `DecisionPolicy` | 14 thresholds (112 bytes) — Maturity stability constants now configurable |
| Per-node state | `NodeStats` + `BLR` + `DensityTracker` + `CalibrationBins` + `DriftBaseline` + `Maturity` ring buffers + 4 × `SignatureWelford` |
| Graph header | `action_counts: [u64; 6]` + `unfreeze_count: u64` |
| CLI surface | `cjcl abng {inspect, replay, diff, explain, train}` with `--json` on each |

### Test counts at end-of-0.4-extended (the floor — Phase 0.5 must keep these growing, never shrinking)

| Gate | Count |
|---|---:|
| `cargo test -p cjc-abng --lib` | **267** |
| `cargo test --test abng` | **442** |
| `cargo test --test prop_tests abng_decision` | **4** × 256 cases |
| `cargo test --test bolero_fuzz abng_decision` | **4** |
| `cargo test -p cjc-cli --test abng_cli_integration` | **32** |

### Frozen contracts as of v11

- `MAGIC = b"ABNG\x0B"` is the only accepted model snapshot magic.
- `b"ABNG-PRED\x01"` is the only accepted prediction snapshot magic.
- Audit-kind tag bytes `0x00..0x1B` are frozen forever.
- `ChildrenKind` codes `0..5` and `Activation` codes `0x00..0x08` are frozen.
- `action_counts` is `[u64; 6]` indexed by `ActionKind` — Unfreeze does NOT bump these (`unfreeze_count` is the separate observability counter — §7 #13).
- `N_THRESHOLDS = 14` for `DecisionPolicy`.
- `NodeStats::canonical_bytes` is 24 bytes (Phase 0.5 Item 4 changes this — see below).
- All per-node state mounting order is the contract — replay relies on it.

---

## Phase 0.5 scope — 5 items

### Item 1: Per-node `provenance_stamp_hash` + `0x1C ProvenanceStamped` (FORCES v12 bump)

**Goal.** `cjcl abng explain` currently verifies `chain_head + codebook +
leaf_head + BLR state`. This is end-to-end *for the model* but doesn't
prove anything about the *dataset* or *feature transform* that produced
the observations. Item 1 closes this gap with an opt-in
provenance-stamping mechanism.

**What to build.**

- New field `provenance_stamp_hash: [u8; 32]` on `AdaptiveBeliefNode`.
  Default `[0u8; 32]` = unstamped.
- New audit kind `AuditKind::ProvenanceStamped { node_id: NodeId,
  hash: [u8; 32] }` at tag `0x1C`. 36-byte canonical body
  (`node_id u32 BE + hash [u8; 32]`).
- New graph method
  `pub fn stamp_provenance(&mut self, node_id: NodeId, hash: [u8; 32])
  -> Result<(), GraphError>` — writes the field + emits the
  `ProvenanceStamped` event. Idempotent: repeated stamps with the same
  hash are no-op (no event).
- New builtin `abng_stamp_provenance(g, node_id, hash: String) -> Void`
  in `cjc-abng/src/dispatch.rs` (hash arg is a 64-char lowercase hex
  string for `.cjcl` ergonomics; convert to `[u8; 32]` at the boundary).
- Extend `predict_snap::pack` to include the predicting node's
  provenance stamp; bump `PRED_MAGIC` to `b"ABNG-PRED\x02"` to absorb
  the new field.
- Extend `cjcl abng explain` to print + verify provenance match.

**Locations.**

| Concern | File |
|---|---|
| AuditKind variant + tag | `crates/cjc-abng/src/audit.rs` |
| Per-node field | `crates/cjc-abng/src/node.rs` |
| Encode + decode + apply_event | `crates/cjc-abng/src/serialize.rs` |
| Graph method | `crates/cjc-abng/src/graph.rs` |
| Builtin | `crates/cjc-abng/src/dispatch.rs` |
| Prediction-snap extension | `crates/cjc-abng/src/predict_snap.rs` |
| CLI explain extension | `crates/cjc-cli/src/commands/abng.rs` |

**Tests.**

| Test type | File |
|---|---|
| In-crate unit | `crates/cjc-abng/src/audit.rs::tests` (canonical bytes), `crates/cjc-abng/src/predict_snap.rs::tests` (round-trip) |
| Integration | new `tests/abng/provenance_tests.rs` (stamp + replay + audit chain + idempotence + node out-of-range + dispatch round-trip) |
| Prop | extend `tests/prop_tests/abng_decision_props.rs` — property: arbitrary stamps round-trip through serialize/replay byte-identically |
| Fuzz | extend `tests/bolero_fuzz/abng_decision_fuzz.rs` — adversarial 0x1C payloads must not panic |
| CLI integration | extend `crates/cjc-cli/tests/abng_cli_integration.rs` — `cjcl abng explain --model` reports provenance match status |

---

### Item 2: Smart-replay using `StatsSnapshot` to fast-forward (NO magic bump)

**Goal.** Phase 0.4 ships `0x1A StatsSnapshot` as a marker only;
`apply_event` is a pure no-op for it. Phase 0.5 makes replay actually
*use* the marker to fast-forward past `*Updated` runs for that node.
This is the read-side half of log compaction.

**What to build.**

- Replay state machine: track per-node "skip until next state-changing
  event" state. When a `StatsSnapshot { node_id, stats_hash }` event is
  applied, mark `node_id` as "fast-forwarded to stats_hash"; subsequent
  `*Updated` events for that node before any state-changing
  (`*Initialized`, `Created`, etc.) event are *verified* (their
  payload's `stats_hash` must match the most recent
  `StatsSnapshot.stats_hash` for that node) but their per-node state
  is NOT re-applied — the snapshot already captures the post-state.
- Critical invariant: smart-replay must produce a `chain_head` and
  per-node state byte-identical to naive replay. The optimization
  changes the *cost*, not the *output*.
- Add a feature flag `smart_replay` (cargo feature or runtime flag)
  during initial development; promote to default once parity is proved.

**Locations.**

| Concern | File |
|---|---|
| Replay state machine | `crates/cjc-abng/src/serialize.rs` |

**Tests.**

| Test type | File |
|---|---|
| In-crate unit | `crates/cjc-abng/src/serialize.rs::tests` (smart-vs-naive byte-equality) |
| Integration | extend `tests/abng/compact_log_tests.rs` — compact_log + smart-replay vs naive-replay parity tests |
| Prop | new property in `tests/prop_tests/abng_decision_props.rs` — "compact_log + smart-replay produces same chain_head + per-node state as naive replay" (256 cases) |
| Fuzz | extend `tests/bolero_fuzz/abng_decision_fuzz.rs` — adversarial blobs with mismatched StatsSnapshot.stats_hash must error specifically (`DecodeError::StatsSnapshotMismatch` or similar) instead of panicking |

---

### Item 3: TOML `--config` files for `cjcl abng train` (NO magic bump)

**Goal.** Phase 0.4 G3.8 ships `cjcl abng train` with explicit flags
(`--seed`, `--n-obs`, `--obs-seed`, `--decide-every`, `--max-decide`,
`--out`). The `--config <x.toml>` form errors with a "Phase 0.5
deferral" message. Phase 0.5 ships the TOML form.

**What to build.**

- Hand-rolled minimal TOML parser. Subset: tables, key=value pairs,
  strings, numbers (i64 and f64), bool, arrays of these. No multiline
  strings, no inline tables, no datetime. cjc-cli has zero external
  deps — adding `toml` crate would break that contract.
- New module `crates/cjc-cli/src/toml_min.rs` (or similar) for the
  parser.
- Extend `train::run` to accept `--config <PATH>` and build the graph
  per the TOML's `[graph]`, `[codebook]`, `[leaf_head]`, `[blr_prior]`,
  `[density]`, `[calibration]`, `[decision_policy]`, `[training]`,
  `[output]` sections.
- The explicit-flag form remains supported (back-compat). Conflict
  resolution: explicit flags override TOML values.

**Locations.**

| Concern | File |
|---|---|
| TOML parser | `crates/cjc-cli/src/toml_min.rs` (new) |
| `train::run` extension | `crates/cjc-cli/src/commands/abng.rs` |

**Tests.**

| Test type | File |
|---|---|
| In-crate unit | `crates/cjc-cli/src/toml_min.rs::tests` (parser unit tests — happy path + every error variant) |
| Integration | extend `crates/cjc-cli/tests/abng_cli_integration.rs` — TOML round-trip (TOML config produces snapshot byte-identical to equivalent flag-config) |
| Prop | (optional) `tests/prop_tests` new file `cli_toml_props.rs` if the parser surface justifies — random valid TOML must round-trip without error |
| Fuzz | (optional) `tests/bolero_fuzz` new file `cli_toml_fuzz.rs` — random byte sequences must not panic the parser |

---

### Item 4: `NodeStats::canonical_bytes` 24B → 32B (FORCES v12 bump — same one as Item 1)

**Goal.** Currently `NodeStats::canonical_bytes` is 24 bytes (`n_seen
u64 BE ‖ mean f64-bits BE ‖ M2 f64-bits BE`). The Welford state has
*two* additional registers: the Kahan compensation register inside
`m2: KahanAccumulatorF64`. Today's canonical bytes drop the
compensation; replay-from-events rebuilds it (one observation at a
time), but log compaction (Item 2 + the eventual full destructive
compaction in Phase 0.6) needs to resume from canonical bytes — that
requires the compensation state too.

**What to build.**

- Extend `NodeStats::canonical_bytes` from `[u8; 24]` to `[u8; 32]`:
  append the 8-byte Kahan compensation register
  (`m2.compensation_bits().to_be_bytes()`).
- Update `Serialize::round_trip` paths.
- Update architecture doc §6.2 canonical-bytes table.
- Audit-event payload doesn't include canonical_bytes directly (only
  `stats_hash`), so the chain doesn't change shape — but `stats_hash`
  output WILL change because the input bytes are longer. This is what
  forces v12.
- Critical: keep determinism — the compensation register's bit pattern
  is platform-stable for fixed observation order.

**Locations.**

| Concern | File |
|---|---|
| `NodeStats::canonical_bytes` | `crates/cjc-abng/src/stats.rs` |
| Per-node section encode/decode | `crates/cjc-abng/src/serialize.rs` |
| `KahanAccumulatorF64::compensation_bits` accessor | `crates/cjc-repro/src/kahan.rs` (if not already exposed) |

**Tests.**

| Test type | File |
|---|---|
| In-crate unit | `crates/cjc-abng/src/stats.rs::tests` (canonical_bytes_size_32, compensation-register-stable, two-stage observe matches one-stage post-compaction-resume) |
| Integration | extend `tests/abng/replay.rs` with byte-offset adjustments for the v12 layout (per-node section size grows by 8B); extend `tests/abng/replay_invariant_tests.rs` similarly |
| Prop | extend `tests/prop_tests/abng_decision_props.rs` — observation streams round-trip through serialize/replay with the new 32B encoding |
| Fuzz | extend `tests/bolero_fuzz/abng_decision_fuzz.rs` — adversarial 32B canonical-bytes blobs (mismatched compensation vs M2) must error specifically |

---

### Item 5: Chess-RL retrofit (NO magic bump — application work)

**Goal.** Validate that ABNG is a real working architecture by
retrofitting it into the chess-rl-v2.5 PRELUDE. Replace the v2.5
value head first (smaller blast radius), then the policy head, with
ABNG-backed equivalents. End-to-end determinism gate: byte-identical
weight hash across two runs.

**Subgoals (in order).**

5.1. **Value head replacement.** `chess_rl_v2.5` currently uses an MLP
     for `V(s)`. Switch to `abng_leaf_forward` at the leaf head
     resolved via `abng_descend(state_features)`. The advantage
     estimator (`A(s, a) = Q(s, a) - V(s)`) becomes the BLR `predict`
     output's `mean`, with `epistemic_leverage` available for
     uncertainty-gating.

5.2. **Policy head replacement.** Same pattern but on the
     action-conditional features. Each leaf node gets a separate
     softmax-output MLP via `abng_leaf_forward`.

5.3. **Uncertainty-gated bootstrap.** When `ood_score(s) > τ` (high
     epistemic uncertainty), abstain or fall back to a uniform policy
     for that observation. Hyperparameter τ goes in the config.

5.4. **Determinism canary.** Run training, compare `chain_head` +
     final weight hash against existing chess-rl-v2.5 weight hash
     `9.790915694115341` — Phase 0.5 must produce its own weight hash
     and lock it in as the v2.6 canary.

**Locations.**

| Concern | File |
|---|---|
| Chess RL PRELUDE updates | `tests/chess_rl_v2/PRELUDE.cjcl` (or wherever the existing PRELUDE lives) |
| New training driver | extend or replace `tests/chess_rl_v2/test_chess_rl_v2.rs` |
| Weight hash canary | new locked-in constant in the test module |

**Tests.**

| Test type | File |
|---|---|
| Integration | `tests/chess_rl_v2/test_chess_rl_v2.rs` — extend the existing 97-test suite with v2.6 (ABNG retrofit) tests. Target: net new test count in the +20 to +30 range. |
| Determinism canary | new test in `tests/chess_rl_v2/test_chess_rl_v2.rs` — locked weight hash for v2.6, byte-equal across two runs |
| End-to-end ML gate | (existing 5-gate ladder for chess RL — see project memory `Chess RL v2.x` blocks). Phase 0.5's gate target depends on what ABNG enables. |

---

## v12 bump consolidation strategy

**Items 1 and 4 force the v12 bump. Items 2, 3, 5 do not.**

Bump magic *exactly once* in Phase 0.5. The architecture-doc §5
roadmap predicts this consolidation. Don't ship Item 1 under v12 then
Item 4 under v13 — that's two bumps where one suffices.

Recommended commit ordering:

1. **First commit:** the v12 bump infrastructure — `MAGIC: \x0B → \x0C`,
   add `0x1C ProvenanceStamped` audit kind, extend
   `NodeStats::canonical_bytes` 24B → 32B. Item 1 + Item 4 together.
   Update all test fixtures that reference byte offsets / threshold
   counts / canonical_bytes sizes.
2. **Subsequent commits** (any order):
   - Item 2 (smart-replay) — pure optimization, no wire-format change.
   - Item 3 (TOML config) — CLI-only.
   - Item 5 (Chess RL retrofit) — application work.

The decide_step canary's locked-in chain_head **may shift** under v12
because the per-node `stats_hash` now hashes 32 bytes instead of 24 —
*and the audit chain incorporates `stats_hash` in every event*. So
you'll need to recompute the canary's locked hex after the v12 bump.
This is unavoidable with Item 4 in scope.

---

## Recommended ordering across all 5 items

```
1. Item 4 (NodeStats canonical_bytes 24→32)
   ↓ same v12 bump
2. Item 1 (provenance_stamp_hash + 0x1C)
   ↓
3. Item 2 (smart-replay) — independent
   ↓
4. Item 3 (TOML config) — independent
   ↓
5. Item 5 (Chess RL retrofit) — depends on items 1-4 stable
```

---

## Verification loop (run before merge)

All five gates must pass with counts ≥ the v11 baseline:

```bash
# From the worktree root:
cargo test -p cjc-abng --lib                                # ≥ 267
cargo test --test abng                                       # ≥ 442
cargo test --test prop_tests abng_decision                   # 4 × 256
cargo test --test bolero_fuzz abng_decision                  # 4
cargo test -p cjc-cli --test abng_cli_integration            # ≥ 32
```

Plus the broader workspace gates:

```bash
cargo test --workspace --release --lib                       # full workspace
cargo test --test physics_ml --release                        # PINN canary
cargo test --test chess_rl_v2 --release                       # chess RL (after Item 5)
```

The `physics_ml` canary is the most important — if PINN tests start
failing, it means a Phase 0.5 change leaked across the cjc-abng /
cjc-ad boundary.

After all gates pass:

```bash
# Inspect the magic byte:
grep -n "const MAGIC" crates/cjc-abng/src/serialize.rs
# Should show MAGIC = b"ABNG\x0C"

# Inspect the threshold count:
grep -n "N_THRESHOLDS:" crates/cjc-abng/src/policy.rs
# Should still show 14 (Phase 0.5 doesn't add policy thresholds)

# Inspect the audit kind tag table:
grep -n "0x1C" crates/cjc-abng/src/audit.rs
# Should show ProvenanceStamped

# Inspect canonical_bytes size:
grep -n "canonical_bytes.*-> \[u8; 32\]" crates/cjc-abng/src/stats.rs
# Should match (was 24 in v11)
```

---

## Documentation updates expected

Mirror the Phase 0.4 documentation discipline:

| Doc | Update |
|---|---|
| `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` | Header (magic v12, builtin count, audit kinds 28→29 if 0x1C ships); §1 phase status table (add Phase 0.5 row); §3.2 install order (no change unless thresholds extend); §3.4 builtin surface (+1 for `abng_stamp_provenance`); §3.6 audit-kind table (add `0x1C`); §6.2 canonical-bytes table (NodeStats 24B→32B); §6.7 Lineage (now full end-to-end); §7 invariants update; Appendix A file map; Appendix B test counts |
| `docs/abng/PHASE_0_5_DESIGN.md` (NEW) | Post-hoc design note — mirror PHASE_0_4_DESIGN.md structure: per-track summary, per-item bullets, wire-format additions, test counts, "What's *not* in 0.5" section |
| `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1).md` | Append "Phase 0.5 amendment" section after the existing "Phase 0.4 Track A amendment" |
| `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md` | Prepend "Phase 0.5 deliverable" block above the existing "Phase 0.4-extended (v11) deliverable" block |
| `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md` | Update one-line ABNG entry to reflect Phase 0.5 status |

---

## Files added or substantially modified during Phase 0.4 (for context)

To know what code surface Phase 0.5 will be touching, read these files
first — they have the most recent design context:

### Heavily modified in Phase 0.4

- `crates/cjc-abng/src/audit.rs` — 28 audit kinds, frozen tag table
- `crates/cjc-abng/src/serialize.rs` — v11 magic, encode/decode for
  every per-node and per-event field, defensive bounds checks
- `crates/cjc-abng/src/graph.rs` — many: `force_merge`, `try_*`,
  `decide_step`, `compact_log`, `descend_traced`, etc.
- `crates/cjc-abng/src/maturity.rs` — `from_node_with_policy` API
- `crates/cjc-abng/src/policy.rs` — 14 thresholds incl. v11 stability
- `crates/cjc-abng/src/predict_snap.rs` — Phase 0.4 Track A module
- `crates/cjc-cli/src/commands/abng.rs` — full CLI suite (`inspect`,
  `replay`, `diff`, `explain`, `train`)

### Reference design notes

- `docs/abng/PHASE_0_4_HANDOFF.md` — predecessor of this document
- `docs/abng/PHASE_0_4_DESIGN.md` — post-hoc design note for Phase 0.4
- `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — source-of-truth contract
- `docs/abng/PHASE_0_3a_DESIGN.md` … `PHASE_0_3D_DESIGN.md` — phase
  history

---

## Recommended next-session prompt (stacked role group prompt)

Paste the following into a fresh Claude Code session inside the
`abng-phase-0-5` worktree:

```
# CJC-Lang ABNG Phase 0.5 — Stacked Role Group Prompt

## ROLE

You are a stacked systems team working inside the CJC-Lang
(Computational Jacobian Core) compiler repository, specifically inside
the ABNG (Adaptive Belief Network Graph) sub-system at
`crates/cjc-abng/`. You are continuing the ABNG roadmap from
Phase 0.4 + v11 extension (already shipped on `claude/abng-phase-0-4`,
commits `29e1cca` … `2f65ccb`).

You consist of:

1. **Lead Language Architect** — owns ABNG's external contract: which
   builtins exist, their semantics, the `Value` enum's invariance,
   the snapshot wire format. Phase 0.5 must NOT introduce any new
   `Value` variant.

2. **Compiler Pipeline Engineer** — owns the Lexer → Parser → AST →
   HIR → MIR → Exec data flow. ABNG builtins are dispatched via the
   `dispatch_abng` satellite crate from BOTH `cjc-eval` and
   `cjc-mir-exec`. Every new builtin must be reachable from both
   executors with byte-identical output (parity tests in
   `tests/abng/parity_p3*.rs` are the regression gate).

3. **Runtime Systems Engineer** — owns memory, dispatch, and ABNG's
   per-thread arena (`thread_local! BTreeMap<i64,
   AdaptiveBeliefGraph>`). Phase 0.5 expands the audit-kind enum
   (`0x1C ProvenanceStamped`), the per-node state (`provenance_stamp_hash`),
   the `NodeStats` canonical encoding (24B → 32B), and the CLI surface
   (TOML config). All of this extends snapshot v11 (forces v12 bump
   for the wire-format items).

4. **Numerical Computing Engineer** — owns deterministic floating
   point. Phase 0.5 Item 4 exposes the Kahan compensation register
   in `NodeStats::canonical_bytes`. The compensation bits MUST be
   bit-stable for fixed observation order across runs and platforms.
   Item 5 (Chess RL retrofit) is your main application validator.

5. **Determinism & Reproducibility Auditor** — enforces bit-identical
   output. Phase 0.5 introduces smart-replay (Item 2). Smart-replay
   MUST produce a `chain_head` and per-node state byte-identical to
   naive replay — the optimization changes cost, not output. The
   `decide_step` canary's locked chain_head will shift under the v12
   bump (because Item 4 changes `stats_hash`); recompute and re-lock
   it.

6. **QA Automation Engineer** — owns the 5-gate test ladder:
   - `cargo test -p cjc-abng --lib` (≥ 267)
   - `cargo test --test abng` (≥ 442)
   - `cargo test --test prop_tests abng_decision` (4 × 256)
   - `cargo test --test bolero_fuzz abng_decision` (4)
   - `cargo test -p cjc-cli --test abng_cli_integration` (≥ 32)

   No item ships without all five gates green. Plus
   `cargo test --workspace --release --lib`,
   `cargo test --test physics_ml --release`, and (after Item 5)
   `cargo test --test chess_rl_v2 --release`.

## PRIME DIRECTIVES

1. **Do not break the audit chain.** Every state mutation MUST append
   exactly one `AuditEvent` with
   `new_hash = sha256(previous_hash ‖ canonical_payload)`. Replay MUST
   reconstruct the chain bit-identically. The v12 bump is allowed (in
   fact required for Items 1 + 4); silent format changes are not.

2. **Do not introduce hidden non-determinism.** All floating-point
   reductions go through Kahan (`KahanAccumulatorF64`) or pairwise
   sum. RNG is SplitMix64 with explicit seed threading. Maps are
   `BTreeMap` only — no `HashMap` in canonical paths.

3. **Preserve `Value` enum layout.** ABNG handles cross the language
   boundary as `Value::Int(i64)`, `Value::Tensor`, `Value::String`,
   `Value::Bytes`, `Value::Array`, `Value::Bool`, `Value::Float`. No
   new `Value` variant.

4. **Consolidate the v12 magic bump.** Item 1 + Item 4 share one bump.
   Don't ship `\x0B → \x0C` for Item 1 then `\x0C → \x0D` for Item 4.

5. **Both executors must agree.** Every new builtin works in both
   `cjc-eval` (AST-walk) AND `cjc-mir-exec` (MIR register-machine)
   with identical output. Parity tests in `tests/abng/parity_p3*.rs`
   are the regression gate.

6. **Language primitives stay minimal.** Higher-level ML algorithms
   live in user `.cjcl` source — Phase 0.5 Item 5 is application-level
   work in the chess-rl-v2 PRELUDE, not new dispatch arms.

## SCOPE — Phase 0.5 5 items

Read `docs/abng/PHASE_0_5_HANDOFF.md` for the full breakdown. The five
items are:

1. **Per-node `provenance_stamp_hash` + `0x1C ProvenanceStamped`**
   audit kind. Forces v12 bump. Closes the lineage gap in
   `cjcl abng explain` (currently verifies model + codebook + leaf
   head + BLR; missing: dataset + feature transform).

2. **Smart-replay using `StatsSnapshot` to fast-forward.** No magic
   bump. Phase 0.4 ships the marker; this is the read-side
   optimization that makes it useful. Determinism contract:
   smart-replay output is byte-identical to naive replay.

3. **TOML `--config` files for `cjcl abng train`.** No magic bump.
   Hand-rolled minimal TOML parser (cjc-cli has zero external deps
   — adding `toml` crate breaks the contract).

4. **`NodeStats::canonical_bytes` 24B → 32B.** Forces v12 bump (same
   as Item 1). Appends the Kahan compensation register so log
   compaction can resume from canonical bytes.

5. **Chess-RL retrofit.** No magic bump. Replaces the v2.5 chess RL
   value head + policy head with ABNG; uncertainty-gated bootstrap;
   end-to-end determinism gate against existing chess-rl-v2 weight
   hashes. This is Phase 0.5's application validator.

## RECOMMENDED ORDER

1. Item 4 (NodeStats canonical_bytes 24→32) — must come first if v12
   bumps. Goes in the same commit as Item 1.
2. Item 1 (provenance_stamp_hash + 0x1C) — same v12 bump.
3. Item 2 (smart-replay) — independent.
4. Item 3 (TOML config) — independent.
5. Item 5 (Chess RL retrofit) — depends on Items 1-4 stable.

## TEST PLACEMENT (mirror Phase 0.4)

Put new tests in the same locations as Phase 0.4 did. Specifically:

- **In-crate unit tests** for new types/methods → `crates/cjc-abng/src/<module>.rs::tests`
  (e.g., `audit.rs::tests` for the new ProvenanceStamped canonical
  bytes; `predict_snap.rs::tests` for the v2 PRED_MAGIC round-trip;
  `stats.rs::tests` for the 32B canonical_bytes; `policy.rs::tests`
  for any new threshold accessors).
- **Integration tests** for cross-module behavior → `tests/abng/<feature>_tests.rs`
  (new files: `provenance_tests.rs`; extend
  `compact_log_tests.rs`, `replay.rs`, `replay_invariant_tests.rs`).
- **Property tests** for invariants → `tests/prop_tests/abng_decision_props.rs`
  (extend with new properties for smart-replay parity, provenance
  round-trip).
- **Fuzz targets** for adversarial blobs → `tests/bolero_fuzz/abng_decision_fuzz.rs`
  (extend with adversarial 0x1C payloads, mismatched StatsSnapshot
  hashes, 32B canonical_bytes corruption).
- **CLI integration tests** for new CLI behavior → `crates/cjc-cli/tests/abng_cli_integration.rs`
  (TOML config, provenance display in `cjcl abng explain`).
- **Chess RL retrofit tests** → `tests/chess_rl_v2/test_chess_rl_v2.rs`
  (extend the existing 97-test suite with ABNG-retrofit tests; locked
  weight hash canary for v2.6).

## DOCUMENTATION PLACEMENT (mirror Phase 0.4)

Put new docs in the same locations as Phase 0.4 did:

- **Architecture doc updates** → `docs/abng/ABNG_CURRENT_ARCHITECTURE.md`
  (header, §1 phase status table, §3.2 install order, §3.4 builtin
  surface, §3.6 audit-kind tag table, §6.2 canonical-bytes table,
  §6.7 Lineage, §7 invariants, Appendix A file map, Appendix B test
  counts).
- **Post-hoc design note** → `docs/abng/PHASE_0_5_DESIGN.md` (NEW —
  mirror `PHASE_0_4_DESIGN.md` structure: per-item summary, decisions
  worth recording, frozen contracts, test-surface expansion).
- **ADR amendment** → append "Phase 0.5 amendment" section to
  `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 ABNG Adaptive Belief
  Radix Graph (Phase 0.1).md` after the existing "Phase 0.4 Track A
  amendment".
- **Project memory** → prepend "Phase 0.5 deliverable" block to
  `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md`.
- **MEMORY.md** → update the one-line ABNG entry to reflect Phase 0.5
  status.

## VERIFICATION LOOP (run before merge)

After every item passes its own targeted tests, run all 5 ABNG-direct
gates — none should shrink:

```bash
cd C:/Users/adame/CJC/.claude/worktrees/abng-phase-0-5
cargo test -p cjc-abng --lib                  # ≥ 267
cargo test --test abng                         # ≥ 442
cargo test --test prop_tests abng_decision     # 4 × 256
cargo test --test bolero_fuzz abng_decision    # 4
cargo test -p cjc-cli --test abng_cli_integration  # ≥ 32
```

Plus the broader gates:

```bash
cargo test --workspace --release --lib
cargo test --test physics_ml --release
cargo test --test chess_rl_v2 --release   # only after Item 5
```

The `physics_ml` canary is critical — if it fails, you accidentally
leaked a change across the cjc-abng / cjc-ad boundary.

## USER PREFERENCES

- **Small, stop-and-confirm units rather than batched changes.** Pause
  after each item and summarize before moving to the next. Don't
  bundle multiple items into one mega-commit unless they share a
  v12 bump (Items 1 + 4).
- **Commit per item once gates pass.** Each numbered item is a
  logical commit; the v12 bump for Items 1 + 4 is one commit.
- **Stop-and-confirm at every magic-bump or wire-format-change
  decision.** v12 is already authorized for Items 1 + 4. Any *other*
  proposed bump (e.g., for a Phase 0.6 item slipping into 0.5) must
  pause and confirm first.
- **The user's name is Adam Ezzat.** (From auto-memory.)

## START

1. Verify the baseline by running the 5 gates above. All five should
   pass with the v11 baseline counts.
2. Read `docs/abng/PHASE_0_5_HANDOFF.md` for the full per-item
   breakdown.
3. Propose Item 4 + Item 1 (the joint v12 bump) as the first concrete
   unit of work. Ask for user confirmation before starting.
```

---

*This handoff was generated at the end of an extended Phase 0.4 +
v11-extension session. The branch `claude/abng-phase-0-5` should be
forked from `claude/abng-phase-0-4` (commit `2f65ccb`); the next
session may want to commit Phase 0.5 work as one PR per item once
each item's gates pass.*
