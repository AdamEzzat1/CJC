# Phase 0.4 — Handoff for Next Session

**Date stamped:** 2026-05-07
**Branch:** `claude/abng-phase-0-4` (forked from `abng-phase-0-3d`)
**Worktree:** `C:\Users\adame\CJC\.claude\worktrees\abng-phase-0-4`
**To continue:** start a new Claude Code session inside the worktree path above and paste this document.

---

## What's done

### Track C — all 7 items shipped (5/7 originally) ✅

| Item | Done | Wire-format change |
|---|---|---|
| C-2.3.1 BLR `predict()` rename → `epistemic_leverage` | ✅ | none (rename + doc) |
| C-2.3.2 NaN/Inf input validation at 4 boundaries | ✅ | none (new error variants) |
| C-2.3.3 Replay semantic invariants (4 new `DecodeError` variants) | ✅ | none (new error variants) |
| C-2.3.4 BLR `b<ε` clamp audit event (`0x18 BlrNumericalRescue`) | ✅ | new audit tag |
| C-2.3.5 MLP/BLR `feature_version_hash` + `abng_reset_blr` | ✅ | snapshot v8 → v9 |
| C-2.3.6 `abng_leaf_set_params_batch` + `0x19 LeafParamsUpdatedBatch` | ✅ | new audit tag |
| C-2.3.7 "per-leaf" → "per-node" doc rename | ✅ | none (doc only) |

### Track B — all 7 items shipped (4/7 compute-only first, then 3 state-bump items) ✅

| Item | Done | Wire-format change |
|---|---|---|
| B-2.2.6 NIG-aware merge math (`BlrState::combine` + `NodeStats::combine`) | ✅ | none (combines existing fields; emits extra `BlrUpdated` witness on `into`) |
| B-2.2.3 KL-divergence gate for Merge (`BlrState::kl_divergence`) | ✅ | none |
| B-2.2.5 Route-entropy gate for Grow (`route_key_entropy_at_candidate_depth`) | ✅ | none |
| B-2.2.4 Bootstrap held-out ΔNLL gain for Split (synthetic Gaussian sampling) | ✅ | none |
| B-2.2.7 Drift-trip auto-Unfreeze + `DecisionPolicy.drift_unfreeze` 12th threshold | ✅ | snapshot v9 → v10 (DecisionPolicy 88B → 96B) |
| B-2.2.2 3-window ECE/σ stability buffers per node | ✅ | per-node +50B |
| B-2.2.1 Welford-smoothed `NodeSignature` profiles per node | ✅ | per-node +96B |

**Snapshot magic:** `\x08 → \x09 (C-2.3.5) → \x0A (B-2.2.7)`. v10 absorbs both Stage B state additions in-place — no further bump needed for the items above.

**Builtin count:** 65 → 67 (`abng_reset_blr`, `abng_leaf_set_params_batch`).

**Audit tags allocated this phase:** `0x18 BlrNumericalRescue`, `0x19 LeafParamsUpdatedBatch`. Tags `0x1A..` still available for Track A.

### Test counts (cumulative since Phase 0.3d baseline 227 + 303 + 4 + 4)

| Gate | Phase 0.3d-5 | End of B-Stage-C | Δ |
|---|---:|---:|---:|
| `cargo test -p cjc-abng --lib` | 227 | **252** | **+25** |
| `cargo test --test abng` | 303 | **391** | **+88** |
| `cargo test --test prop_tests abng_decision` | 4 | 4 | +0 |
| `cargo test --test bolero_fuzz abng_decision` | 4 | 4 | +0 |

### Resolutions in `ABNG_CURRENT_ARCHITECTURE.md`

Marked ✅ RESOLVED:
- §R10 / §8.12 (BLR predict naming)
- §R11 / §8.13 (NaN/Inf validation)
- §R12 / §8.14 (replay invariants)
- §R13 / §8.15 (BLR `b<0` clamp)
- §R14 / §8.16 (MLP/BLR feature space contract)
- §R16 / §8.18 (per-leaf vs per-node naming)
- §8.10 (real merge math)
- §8.11 (drift-trip auto-Unfreeze)
- §8.17 (LeafParamsUpdated event volume)

---

## What's still pending

### Documentation finalization (LOW EFFORT, ~30 minutes)

1. **Update `ABNG_CURRENT_ARCHITECTURE.md`** — currently the §1 phase status table and §6.6 `Maturity` description don't yet reflect B-2.2.{1,2}. Add ✅ RESOLVED markers to §8.x for B-2.2.1 and B-2.2.2 specifically (the prompt's "open gaps" sections may not yet have explicit entries; check §8.2 and §8.3 since those discussed the pre-0.4 stability / signature stubs).
2. **Create `docs/abng/PHASE_0_4_DESIGN.md`** — post-hoc design note. Mirror PHASE_0_3D_DESIGN.md's structure: per-track summary (A/B/C), per-item bullet, wire-format additions, test counts.
3. **Update `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 *.md`** — append a "Phase 0.4 amendment" section. ADR-0023 is the existing ABNG ADR.
4. **Update `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md`** — prepend "Phase 0.4 deliverable" block. The current memory ends at end-of-Phase-0.3d (2026-05-07).
5. **Update `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md`** — replace the ABNG entry with a one-line pointer to the updated `project_abng.md`.

### Track A — `cjcl abng …` CLI (LARGE EFFORT, 1-2 days)

This is the biggest remaining piece. Per `PHASE_0_4_IMPLEMENTATION_PROMPT.md` §2.1:

| Subcommand | Purpose | Test target |
|---|---|---|
| `cjcl abng train --config x.toml --seed 42` | Driver: observe → decide_step → checkpoint | `tests/abng/cli_tests.rs` |
| `cjcl abng inspect <model.snap> [--node ID] [--audit] [--stats] [--tree]` | Read-only viewer | same |
| `cjcl abng explain <prediction.snap>` | Lineage + abstain reason. Requires `Routed` audit events (§8.6). | same |
| `cjcl abng replay <model.snap> --log <audit.log> --verify` | Wrapper around `abng_replay`. | same |
| `cjcl abng diff <a.snap> <b.snap>` | Topology + per-node fingerprint diff | same |

Plus per the prompt §2.5 the supporting builtins:
- `abng_predict_snap(g, node_id, x_idx) -> Bytes` for `cjcl abng explain`
- `abng_compact_log(g, until_seq) -> Void` for `cjcl abng explain` log compaction
- JSON snapshot adapter via `cjc_snap::snap_to_json`

**Scope decision needed:** the prompt §2.6 says v10 (which we just bumped) "should be the LAST snapshot bump for ABNG's pre-1.0 lifecycle" — so Track A's additions should NOT bump magic again. That means:
- New audit tags `0x1A..0x1C` (`StatsSnapshot`, `Routed`, `ProvenanceStamped`) extend v10 in place (opt-in tags don't break replay).
- New `provenance_stamp_hash: [u8; 32]` per node (if added) DOES require a bump → bump to `\x0B`.

The pragmatic compromise the next session can take: **defer the per-node provenance hash** to Phase 0.5; ship the audit tags + JSON view + CLI subcommands under v10. This keeps "one bump per phase" once the next session decides.

### Track C-2.3.{8,9,10,11,12} (audit findings) — MEDIUM EFFORT

Several minor items from the post-0.3d audit were noted in the prompt but I deferred. Decide whether to ship them in Phase 0.4 or 0.5:

- **C-2.3.8** `abng_blr_predict_with_fallback` — read-only fallback walk up parent chain. Optional convenience builtin.
- **C-2.3.9** `NodeStats::canonical_bytes` 24B → 32B (append Kahan compensation register). Required for log compaction. **Snapshot bump if shipped**.
- **C-2.3.10** Cholesky regularization design-vs-code drift. Doc-only correction in `PHASE_0_3b_DESIGN.md`.
- **C-2.3.11** Empty-graph chain-head wording in §2.2 of architecture doc.
- **C-2.3.12** Audit findings — `expected_epistemic` re-capture, configurable `Maturity` constants, `unfreeze_count` observability, `decide_step` chain-head canary. Mostly small additions.

### Workspace + property/fuzz regression gates (LOW EFFORT, but slow)

Run before merge:
```bash
cargo test --workspace --release --lib
cargo test --test physics_ml --release
cargo test --test prop_tests --release
cargo test --test bolero_fuzz --release
```

Per Phase 0.3d-5 baselines: workspace 2,363 passed, physics_ml 107/107, prop 4/256 each, fuzz 4/100 each. Phase 0.4 must keep counts growing, never shrinking.

### Add proptest properties + fuzz targets for new audit kinds + state additions (MEDIUM EFFORT)

Per prompt §3.4:
- KL-merge symmetry: `kl_divergence(a, b) ≥ 0` and equals 0 iff posteriors are bit-identical
- Welford-smoothed signature stability: stationary stream → Hamming distance to itself shrinks over time

Per prompt §3.5:
- CLI fuzz target: random TOML configs feeding `cjcl abng train` produces well-defined errors, never panics

---

## How to verify what was shipped

From the worktree root (`C:\Users\adame\CJC\.claude\worktrees\abng-phase-0-4`):

```bash
cargo test -p cjc-abng --lib                 # 252 expected
cargo test --test abng                       # 391 expected
cargo test --test prop_tests abng_decision   # 4 expected
cargo test --test bolero_fuzz abng_decision  # 4 expected
```

All four should pass cleanly.

To inspect the magic byte:
```bash
# Should show MAGIC = b"ABNG\x0A"
grep -n "const MAGIC" crates/cjc-abng/src/serialize.rs
```

To inspect the policy threshold count:
```bash
# Should show N_THRESHOLDS = 12
grep -n "N_THRESHOLDS:" crates/cjc-abng/src/policy.rs
```

To inspect the new fields on `AdaptiveBeliefNode`:
```bash
# Should show ece_history, sigma_history, welford_prediction, etc.
grep -n "pub ece_history\|pub welford_" crates/cjc-abng/src/node.rs
```

---

## Files added or substantially modified during Phase 0.4

### New files

- `tests/abng/observe_validation_tests.rs` (C-2.3.2)
- `tests/abng/replay_invariant_tests.rs` (C-2.3.3)
- `tests/abng/blr_numerical_rescue_tests.rs` (C-2.3.4)
- `tests/abng/blr_feature_version_tests.rs` (C-2.3.5)
- `tests/abng/leaf_params_batch_tests.rs` (C-2.3.6)
- `tests/abng/merge_math_tests.rs` (B-2.2.6)
- `tests/abng/route_entropy_grow_tests.rs` (B-2.2.5)
- `tests/abng/split_nll_gate_tests.rs` (B-2.2.4)
- `docs/abng/PHASE_0_4_HANDOFF.md` (this document)

### Heavily modified

- `crates/cjc-abng/src/audit.rs` — new audit kinds `0x18`/`0x19`
- `crates/cjc-abng/src/blr.rs` — `feature_version_hash`, `combine`, `kl_divergence`, `NonFiniteInput`/`FeatureVersionStale` errors, `BlrNumericalRescue` rescue path
- `crates/cjc-abng/src/graph.rs` — many: `force_merge` rewrite, `try_merge`/`try_split`/`try_grow` gate additions, `reset_blr`, `advance_stability_history`, `advance_signature_welfords`, `route_key_entropy_at_candidate_depth`, `estimate_split_nll_gain`, drift-unfreeze ladder step in `decide_step`
- `crates/cjc-abng/src/maturity.rs` — `from_node` rewrite using ring buffers; new `ECE_STABILITY_MAX_DELTA` and `SIGMA_STABILITY_RATIO` constants
- `crates/cjc-abng/src/signature.rs` — `SignatureWelford` struct; `from_node` reads Welford state; `routing_observation_value` helper
- `crates/cjc-abng/src/node.rs` — new fields: `ece_history`, `ece_fill_count`, `sigma_history`, `sigma_fill_count`, `welford_prediction`, `welford_uncertainty`, `welford_calibration`, `welford_routing`
- `crates/cjc-abng/src/policy.rs` — `N_THRESHOLDS = 12`; `drift_unfreeze` accessor
- `crates/cjc-abng/src/stats.rs` — `combine` method (Chan/Golub/LeVeque parallel Welford merge)
- `crates/cjc-abng/src/serialize.rs` — magic v10, encode/decode for all new fields, `apply_event` for `Merge` does combine + `BlrUpdated` witness, `apply_event` for `BlrInitialized` reset semantics for `reset_blr`
- `crates/cjc-abng/src/dispatch.rs` — `abng_reset_blr`, `abng_leaf_set_params_batch`
- `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — many sections updated, multiple ✅ RESOLVED markers
- `docs/abng/PHASE_0_3a_DESIGN.md` and `PHASE_0_3b_DESIGN.md` — per-leaf → per-node title + body

### Tests touched (in addition to new files above)

- `tests/abng/decision_tests.rs` — updated `ok_thresholds()` to 12 elements; updated `decide_step_split_when_samples_high` (varied observations); updated `decide_step_auto_captures_expected_epistemic_at_uncertainty_stable` (3-window stability)
- `tests/abng/dispatch_p3d.rs` — `ok_thresholds_tensor` to 12; `signature_changes_with_subsystem_install` updated for Welford semantics
- `tests/abng/maturity_signature_tests.rs` — multiple tests updated for B-2.2.1 Welford signatures + B-2.2.2 ring buffers
- `tests/abng/parity_p3a.rs` — added `parity_leaf_set_params_batch_*` tests (C-2.3.6)
- `tests/abng/parity_p3b.rs` — added `parity_reset_blr_*` tests (C-2.3.5)
- `tests/abng/parity_p3d.rs` — POLICY_INSTALL Tensor[11] → Tensor[12]
- `tests/abng/replay.rs` — byte-offset comments updated for v10 per-node section size (89 → 235)
- `tests/abng/replay_invariant_tests.rs` — `n_events_offset` updated 200 → 346
- `tests/prop_tests/abng_decision_props.rs` — `arb_thresholds` strategy 11 → 12 elements

---

## Recommended next-session prompt

When you start the next session in the new worktree, paste this:

```
I'm continuing Phase 0.4 of the ABNG project. Read
docs/abng/PHASE_0_4_HANDOFF.md for full context — it
describes what's already shipped (all of Track C +
all of Track B) and what's pending (final docs, Track A
CLI, optional Track C-2.3.{8,9,10,11,12} polish items,
plus regression gates).

Start by running the four ABNG-direct gates to verify
the baseline:
  cargo test -p cjc-abng --lib                 # 252 expected
  cargo test --test abng                       # 391 expected
  cargo test --test prop_tests abng_decision   # 4 expected
  cargo test --test bolero_fuzz abng_decision  # 4 expected

If all green, propose the next concrete unit of work
and ask for confirmation before starting. The user
prefers small, stop-and-confirm units rather than
batched changes.
```

---

*This handoff was generated near the end of an extended Phase 0.4 implementation session. The new branch `claude/abng-phase-0-4` is forked from `abng-phase-0-3d` and has uncommitted local changes — the next session may want to commit them in logical units (one per Track C / Track B item) before continuing with Track A.*
