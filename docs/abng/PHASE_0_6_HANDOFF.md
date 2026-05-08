# Phase 0.6 — Handoff for Next Session

**Date stamped:** 2026-05-08
**Branch (recommended):** `claude/abng-phase-0-6` (forked from `master` after the Phase 0.5 merge)
**Worktree (recommended):** `C:\Users\adame\CJC\.claude\worktrees\abng-phase-0-6`
**Master HEAD at handoff:** `fe0b602` (Merge ABNG Phase 0.1 → 0.5 into master)
**To continue:** start a new Claude Code session inside the worktree above and paste the prompt at the end of this document (`## Recommended next-session prompt`).

---

## What's done — Phase 0.5 baseline (current state on master)

End-of-Phase-0.5 state (commit `fe0b602` on `master` — the merge of `claude/abng-phase-0-5`):

### Surface

| Property | Value |
|---|---|
| Snapshot magic | `b"ABNG\x0C"` (v12) |
| Builtin count | **75** `abng_*` arms in `dispatch.rs` (+2 from Phase 0.4: `abng_stamp_provenance`, `abng_provenance_stamp`) |
| Audit kinds | **29** (tags `0x00..0x1C` — `0x1C ProvenanceStamped` is new in 0.5) |
| `DecisionPolicy` | 14 thresholds (112 bytes, unchanged from v11) |
| Per-node state | `NodeStats` (32B canonical) + `BLR` + `DensityTracker` + `CalibrationBins` + `DriftBaseline` + `Maturity` ring buffers + 4 × `SignatureWelford` + new `provenance_stamp_hash: [u8; 32]` |
| Graph header | `action_counts: [u64; 6]` + `unfreeze_count: u64` |
| CLI surface | `cjcl abng {inspect, replay, diff, explain, train}` with `--json`; `train` accepts `--config <PATH>.toml` (zero-dep TOML parser in `cjc-cli`) |

### Phase 0.5 demos shipped (11 distinct capabilities, 13 demo files)

**Application-layer demos (Rust API + CJC-Lang versions):**
- PINN per-region uncertainty (`tests/test_abng_pinn_uncertainty.rs` + `_cjcl.rs` + `tests/abng_demos/pinn_source.rs`)
- Tabular GP-like regression (`tests/test_abng_tabular_gp.rs` + `_cjcl.rs` + `tabular_source.rs`)
- Lineage attestation (`tests/test_abng_lineage_attestation.rs` + `_cjcl.rs` + `lineage_source.rs`)
- Chess RL v2.6 retrofit scaffold (`tests/test_chess_rl_v2_6_abng.rs`)

**Capability-only demos (CJC-Lang only):**
- OOD detection composite (`tests/test_abng_ood_detection_cjcl.rs` + `ood_source.rs`)
- Adaptive structural triggers (`tests/test_abng_adaptive_triggers_cjcl.rs` + `adaptive_source.rs`)
- Calibration / ECE (`tests/test_abng_calibration_cjcl.rs` + `calibration_source.rs`)
- Distribution-drift detection (`tests/test_abng_drift_detection_cjcl.rs` + `drift_source.rs`)
- Log compaction (`tests/test_abng_compact_log_cjcl.rs` + `compact_source.rs`)
- Maturity inspection (`tests/test_abng_maturity_inspection_cjcl.rs` + `maturity_source.rs`)

### Test counts at end-of-0.5 (the floor — Phase 0.6 must keep these passing, never shrinking)

| Gate | Count |
|---|---:|
| `cargo test -p cjc-abng --lib` | **275** |
| `cargo test --test abng` | **460** |
| `cargo test --test prop_tests abng_decision` | **6** × 256 cases |
| `cargo test --test bolero_fuzz abng_decision` | **8** |
| `cargo test -p cjc-cli --test abng_cli_integration` | **43** |
| `cargo test -p cjc-cli --lib toml_min` | **29** |
| 4 Rust application demos | 19 + 13 + 10 + 16 = **58** |
| 9 CJC-Lang demos | 9 + 9 + 9 + 9 + 9 + 10 + 9 + 7 + 10 = **81** |
| `cargo test --workspace --release --lib` | **2,440** |
| `cargo test --test physics_ml --release` | **107** (2 ignored) |
| `cargo test --test test_chess_rl_v2 --release` | **97** (2 ignored, ~12 min wall clock) |

### Locked SHA-256 canaries

| Canary | Hex | Source |
|---|---|---|
| `decide_step` chain_head (v12) | `d064fb08c546be1b9850bfa91f87f4aed95682aa4fb7f4533cf1ac4da0d87807` | `tests/abng/decide_step_canary_tests.rs` |
| Chess RL v2.6 chain_head | `27d547b8f721b6631e3cbbe5fc4de560c6f09e6cc93eaf6c9e1bf36a3db6847b` | `tests/test_chess_rl_v2_6_abng.rs` |
| Chess RL v2.6 BLR state_hash | `869b32bdf937d27ec032b789980583fa1bf5871c528a7ee1f26d1d40fb6cfabc` | `tests/test_chess_rl_v2_6_abng.rs` |
| PINN chain_head (Rust) | `30d333f1f7dca5acaa76b0e4bfdbd4a733df38c6adeda094ae69cf0e9c4e468d` | `tests/test_abng_pinn_uncertainty.rs` |
| Tabular chain_head (Rust) | `cd3f5c7be81f5966d1f41af811cc94a859b653adf9993f1d5b3e23c0a87397e6` | `tests/test_abng_tabular_gp.rs` |
| Lineage chain_head (Rust) | `789acce77a22241c2e3601bf958e978b24e4707874cdbb23a7fde9a98f0606c2` | `tests/test_abng_lineage_attestation.rs` |
| Dataset A fingerprint | `3e85d52f2508aecaaf32737edca48a644796783d6be7e6e324e6760506bc3634` | `tests/test_abng_lineage_attestation.rs` |
| Lineage chain_head (CJC-Lang) | `20f5f977cd7dcfad536fbf4be49d4b18c6ba2430b32e510713a038c94fa39b40` | `tests/test_abng_lineage_attestation_cjcl.rs` |
| PINN chain_head (CJC-Lang) | `e5d6c41daeec4b34a78ddab5086f9903d3dd56b8fd995400dd190ba8d684a64a` | `tests/test_abng_pinn_uncertainty_cjcl.rs` |
| Tabular chain_head (CJC-Lang) | `4ffacae41d76f505335218ee0479c656e059024cb7e8d6c95350bbc2af09be54` | `tests/test_abng_tabular_gp_cjcl.rs` |
| Adaptive triggers (CJC-Lang) | `d064fb08…7807` (matches Rust decide_step canary EXACTLY) | `tests/test_abng_adaptive_triggers_cjcl.rs` |
| OOD chain_head (CJC-Lang) | `85970ca5c2dbd93469fe3c849e3b15f6b32b3a593a7e69c16e1f22dcf8fd533e` | `tests/test_abng_ood_detection_cjcl.rs` |
| Calibration chain_head (CJC-Lang) | `4c625f088c72da46f756972e89adc934183f27e3ba2fa2bbb2509ba091a15a25` | `tests/test_abng_calibration_cjcl.rs` |
| Drift chain_head (CJC-Lang) | `a3a41c5b282a349d1af38637cc2207f66a9e59a20f839288c4f8e5c817ea8121` | `tests/test_abng_drift_detection_cjcl.rs` |
| Maturity chain_head (CJC-Lang) | `c7b92726459c670bd960cd27c1f307e733744bcf420f5457f9a34a1b54a1928b` | `tests/test_abng_maturity_inspection_cjcl.rs` |

**These canaries are the integrity contract for Phase 0.6.** Every one must continue to verify on every Phase 0.6 commit unless a deliberate v13 wire-format bump is shipped (in which case all canaries get re-locked simultaneously, like the v12 bump did).

### Frozen contracts as of v12

- `MAGIC = b"ABNG\x0C"` is the only accepted model snapshot magic.
- `b"ABNG-PRED\x02"` is the only accepted prediction snapshot magic.
- Audit-kind tag bytes `0x00..0x1C` are frozen forever.
- `ChildrenKind` codes `0..5` and `Activation` codes `0x00..0x08` are frozen.
- `action_counts` is `[u64; 6]` indexed by `ActionKind` — Unfreeze does NOT bump these.
- `N_THRESHOLDS = 14` for `DecisionPolicy`.
- `NodeStats::canonical_bytes` is **32 bytes** (Phase 0.5 Item 4 grew this from 24 → 32).
- All per-node state mounting order is the contract — replay relies on it.
- Per-node `provenance_stamp_hash: [u8; 32]` is the last 32 bytes of every node's per-node section.

### Honest gaps closed during Phase 0.5 + ones still open

| Gap | Status |
|---|---|
| AST↔MIR parity for new `abng_*` builtins | **Closed** — ~95 byte-equal parity assertions over ~30 builtins via 9 CJC-Lang demos |
| Application-layer demonstrability | **Closed** — 11 distinct capabilities each have a working demo |
| Cross-platform determinism CI | **Still open** — canaries are Windows-only |
| Performance benchmarks at scale | **Still open** — no wall-clock measurements at n > 200 |
| Smart-replay fast-forward optimization | **Still open** — Phase C shipped the API + tamper check; cycle-saving skip layer deferred to 0.6 |
| Adaptive triggers — fire all 6 types | **Half-open** — Merge demonstrated; Grow/Split/Prune/Compress/Freeze tested at primitive level only |
| Real-world case study (someone using ABNG to attest a real model) | **Still open** — synthetic data only |

---

## Phase 0.6 scope — 8 items

Phase 0.6 has a single overarching theme: **scale and performance, without losing the determinism contract.** The eight items are organized by impact:

1. **Cross-platform determinism CI** — a one-day project that pays off forever
2. **Performance benchmarks at scale** — establish baseline numbers before any optimization
3. **Smart-replay fast-forward optimization** — first real perf win
4. **Native batch_observe + bulk BLR update** — second perf win, hot-path specialization
5. **Adaptive triggers — fire all 6 types in scaled demos** — close the demo gap
6. **Phase 0.6 demos at scale + with noise** — prove the demos hold under realistic conditions
7. **Compiler / interpreter perf prep work** — sets up the bigger long-term wins
8. **TidyView-parity training pipeline** — start the multi-phase push toward production-grade ML training in CJC-Lang

### Item 1: Cross-platform determinism CI (NO magic bump)

**Goal.** Phase 0.5 locks 15 SHA-256 canaries on Windows. They *should* hold on Linux/macOS (Kahan, no FMA, BTreeMap iteration), but unverified. Add CI that runs the full suite on three platforms.

**What to build.**

- `.github/workflows/cross-platform-determinism.yml` — matrix build (Windows + Ubuntu + macOS)
- Job runs the 5 ABNG-direct gates + 13 demo files
- Failure mode: any platform diverges from the locked canaries → CI fails loudly, every future commit gates on this

**Locations.**

| Concern | File |
|---|---|
| CI workflow | `.github/workflows/cross-platform-determinism.yml` (new) |
| Optional cross-platform shim | `crates/cjc-snap/src/hash.rs` (verify SHA-256 returns identical bytes regardless of platform endianness) |

**Tests.** Trivially the existing test suite — the CI is the test.

**Risk:** if Linux/macOS diverges, that's a real bug to investigate. Could be platform-specific FP rounding, BTreeMap iteration drift (very unlikely with sorted keys), or a SHA-256 implementation difference. Each is a debugging exercise.

---

### Item 2: Performance benchmarks at scale (NO magic bump)

**Goal.** Establish wall-clock numbers for ABNG at realistic dataset sizes BEFORE shipping any optimization. Without baseline numbers, perf claims are unverified.

**What to build.**

- `bench/abng_micro/` — micro-benchmarks of `abng_observe`, `abng_blr_update`, `abng_blr_predict`, `abng_descend`. Measure per-op cost in nanoseconds.
- `bench/abng_vs_sklearn/` — Python-side comparison: same regression problem, ABNG vs `sklearn.gaussian_process.GaussianProcessRegressor` at n ∈ {10³, 10⁴, 10⁵}. Wall-clock + RMSE comparison.
- `bench/abng_pinn_scale/` — PINN demo at 10⁴ collocation points. Compare to a vanilla MLP-PINN (PyTorch) baseline. Convergence rate + final L2 error.
- `bench/abng_lineage_at_scale/` — lineage demo at 10⁴ rows. Wall-clock for train + stamp + predict_snap pack/unpack.

Use Criterion (already a workspace dep in `bench/`).

**Locations.**

| Concern | File |
|---|---|
| Micro-benches | `bench/abng_micro/` (new) |
| Cross-tool benches | `bench/abng_vs_sklearn/` (new — Python harness + Rust ABNG side) |
| PINN scale | `bench/abng_pinn_scale/` (new) |
| Lineage scale | `bench/abng_lineage_at_scale/` (new) |

**Tests.** Each bench is a Cargo bench target. CI runs them with deterministic seeds; results land in a `bench_results/` directory (gitignored, summary committed).

**Outcome.** A "Phase 0.6 baseline" set of numbers that all subsequent perf work measures against. Print these in the post-Phase-0.6 doc.

---

### Item 3: Smart-replay fast-forward optimization (NO magic bump)

**Goal.** Phase 0.5 (Item 2) shipped `smart_replay` API + StatsSnapshot consistency check, but explicitly deferred the cycle-saving skip-the-observe layer. Phase 0.6 ships it.

**What to build.**

- Pre-pass over the audit log to identify nodes where:
  - A `StatsSnapshot` event exists for the node, AND
  - No state-mutation events follow that snapshot for the same node
  These nodes are "fast-forwardable."
- Apply pass: for `BeliefUpdate` events on fast-forwardable nodes whose seq < snapshot_seq, **skip the `node.observe(value)` mutation** but still advance the chain hash via stored payload bytes.
- Per-event `stats_version` and `stats_hash` checks: relax for fast-forwarded events. The `StatsSnapshot`'s recorded `stats_hash` becomes the consolidated tamper check for all skipped events.
- After the main loop, for fast-forwarded nodes, install stats from the per-node section's `canonical_bytes` via `NodeStats::from_canonical_bytes()` (already added in Phase 0.5).
- Critical invariant: smart_replay output MUST stay byte-identical to naive replay. Existing parity property tests gate this.

**Locations.**

| Concern | File |
|---|---|
| Replay state machine | `crates/cjc-abng/src/serialize.rs` (extend `replay_with_options`) |
| Property test | `tests/prop_tests/abng_decision_props.rs` (extend with "smart_replay equals naive — including for compacted logs") |

**Tests.**

| Test type | File |
|---|---|
| Integration | extend `tests/abng/compact_log_tests.rs` — assert smart_replay actually skips observes (verify via instrumentation counter) |
| Prop | extend `tests/prop_tests/abng_decision_props.rs` — random observe/compact sequences must yield byte-identical smart_replay vs naive |
| Bench | `bench/abng_micro/` — measure the speedup (target: ≥ 5× for a compacted log of 10⁴ events) |

**Risk.** This relaxes per-event tamper detection for fast-forwarded events. The compensating signal: the StatsSnapshot's `stats_hash` becomes the consolidated checkpoint. Ship a doc note explaining the trade-off.

---

### Item 4: Native batch_observe + bulk BLR update (FORCES v13 bump)

**Goal.** ABNG's hot path today is `observe → blr_update → audit_chain_hash → stamp_commit`, fired once per training point. For 10⁴ training points: 10⁴ chain hashes + 10⁴ Cholesky updates. Either dominates a real workload's wall-clock.

Native batching collapses N observations into one event with O(N · d²) Cholesky update + ONE chain-hash entry.

**What to build.**

- New audit kind `AuditKind::BeliefUpdateBatch { count: u32, batch_hash: [u8; 32] }` at tag `0x1D` (next free tag).
- New builtin `abng_observe_batch(g, node_id, values: Tensor[n])` and `abng_blr_update_batch(g, node, features_2d: Tensor[n, d], y: Tensor[n])`.
- Cholesky factor maintenance for batch updates: rather than n × rank-1 updates, do one matrix-matrix multiplication (`Φ.T @ Φ` accumulation) + one `chol_solve`.
- Per-event payload includes a SHA-256 of the batch's canonical bytes, so the audit chain still detects per-row tampering of the batch.
- Bump `MAGIC = b"ABNG\x0C"` → `b"ABNG\x0D"` (v13).
- Re-lock all 15 canaries.

**Locations.**

| Concern | File |
|---|---|
| AuditKind variant | `crates/cjc-abng/src/audit.rs` |
| Encode/decode/apply_event | `crates/cjc-abng/src/serialize.rs` |
| Graph methods | `crates/cjc-abng/src/graph.rs` (`observe_batch`, `blr_update_batch`) |
| Builtins | `crates/cjc-abng/src/dispatch.rs` |
| BLR batch update math | `crates/cjc-abng/src/blr.rs` (extend `BlrState::update`) |

**Tests.**

| Test type | File |
|---|---|
| In-crate unit | `crates/cjc-abng/src/blr.rs::tests` (canonical_bytes preserved between per-row and batched paths) |
| Integration | new `tests/abng/batch_observe_tests.rs` (batch + replay + chain integrity + per-row equivalence at the canonical-bytes level) |
| Prop | extend `tests/prop_tests/abng_decision_props.rs` — random batch observations must produce same final state as per-row equivalent |
| Fuzz | extend `tests/bolero_fuzz/abng_decision_fuzz.rs` — adversarial batch sizes, including n=0 and n=u32::MAX |
| CLI | extend `crates/cjc-cli/tests/abng_cli_integration.rs` — `cjcl abng inspect` lists `BeliefUpdateBatch` events |
| Bench | `bench/abng_micro/` — `bench_observe_per_row` vs `bench_observe_batch_n=64` vs `_n=1024`. Target: ≥ 10× speedup at n=1024 |

---

### Item 5: Adaptive triggers — fire all 6 types in scaled demos (NO magic bump)

**Goal.** Phase 0.5's adaptive demo fires Merge only. Phase 0.6 demonstrates Grow, Split, Prune, Compress, Freeze in engineered workloads + locks canaries for each.

**What to build.**

For each trigger type, design a workload that fires *that specific trigger* deterministically:

- **Grow**: drive impurity > `H_grow` (entropy threshold) on a single leaf with large variance observations + sufficient evidence count
- **Split**: drive `nll_split_gain` > threshold via held-out partition test
- **Prune**: long-running stale signature on a node with `n_seen < prune_floor`, beyond `prune_grace_epochs`
- **Compress**: a sub-tree whose `tau_compress` Hamming distance falls below threshold
- **Freeze**: sustained `signature_stable_calls` count exceeds `freeze_after`

Each becomes one CJC-Lang demo, lock the canary chain_head + action_count.

**Locations.**

| Concern | File |
|---|---|
| Source | `tests/abng_demos/grow_source.rs`, `split_source.rs`, `prune_source.rs`, `compress_source.rs`, `freeze_source.rs` (add to `mod.rs`) |
| Tests | `tests/test_abng_grow_trigger_cjcl.rs`, etc. (5 new files) |

Each new demo follows the established pattern: ~9 tests including smoke (eval + mir), AST↔MIR parity, headline assertion, locked canary.

**Tests.** ~5 × 9 = ~45 net new tests.

---

### Item 6: Phase 0.6 demos at scale + with noise (NO magic bump)

**Goal.** Every Phase 0.5 demo proves the capability *exists*. Phase 0.6 adds a "scaled" sibling for each, proving the capability *works under realistic conditions*: more samples, additive noise, higher feature dimensions.

For each Phase 0.5 demo, add a `_scaled` version:

| Phase 0.5 demo | Phase 0.6 scaled version |
|---|---|
| PINN per-region uncertainty (40 samples) | `pinn_scaled_source.rs`: 10⁴ samples + Gaussian noise σ=0.01 + 2-D Burgers PDE |
| Tabular GP-like (200 samples) | `tabular_scaled_source.rs`: 10⁴ samples + 4-D input + heteroskedastic noise; sklearn comparison |
| Lineage attestation (16 rows) | `lineage_scaled_source.rs`: 10⁴ rows + realistic patient-feature richness |
| OOD detection (4-D codebook) | `ood_scaled_source.rs`: 32-D embedding-space input |
| Calibration (synthetic injection) | `calibration_scaled_source.rs`: real binary-classifier outputs (e.g., a logistic regression's predict_proba on a tabular dataset), 10³+ predictions |
| Drift detection (controlled mean shift) | `drift_scaled_source.rs`: 10⁴ streaming observations across an engineered drift schedule (gradual, abrupt, recurring) |
| Compact_log (5 obs / 3 nodes) | `compact_scaled_source.rs`: 10⁵ events, smart-replay measured speedup |
| Maturity (10 obs + 5 decide_steps) | `maturity_scaled_source.rs`: 10³ decide_step calls; observe full flag evolution |

**Critical invariant:** the EXISTING Phase 0.5 demos must continue to pass, with their canaries unchanged. The scaled versions are *additions*, not replacements. The Phase 0.5 demos test the capability at the minimum-sufficient scale; the Phase 0.6 demos test it at production-realistic scale.

**Locations.**

| Concern | File |
|---|---|
| Sources | `tests/abng_demos/<name>_scaled_source.rs` (8 new files) |
| Tests | `tests/test_abng_<name>_scaled_cjcl.rs` (8 new files) |

**Locked canaries.** Each scaled demo locks its own chain_head canary. This brings the total canary count from 15 → 23.

---

### Item 7: Compiler / interpreter perf prep work (NO magic bump for ABNG; potential larger bump elsewhere)

**Goal.** ABNG's runtime perf is bottlenecked by the CJC-Lang interpreter (5–50× slower than direct Rust). Phase 0.6 lands the *incremental* perf work; AOT compilation is Phase 0.7+.

**What to build (incremental):**

- **MIR optimizer passes**: extend `cjc-mir/src/optimize/` with:
  - LICM (loop-invariant code motion) — pull constant tensor allocations out of hot loops
  - CSE (common subexpression elimination) — observed in chess RL profile, repeated `Tensor.from_vec` calls in tight loops
  - Function inlining (small functions, < 20 MIR ops)
- **Hot-path native specialization**: extend `cjc-runtime/src/builtins.rs` with native kernels for the most expensive `abng_*` patterns identified by profiling. Candidate: the descend-route-leaf pattern (encode_prefix → descend → blr_predict).
- **Profile-guided optimization**: instrument the existing `profile_zone_*` builtins (already shipped from chess RL v2.3) to identify hot zones in the Phase 0.6 scaled demos.

**Locations.**

| Concern | File |
|---|---|
| MIR passes | `crates/cjc-mir/src/optimize/licm.rs` (new), `cse.rs` (new), `inline.rs` (new) |
| Native kernels | `crates/cjc-runtime/src/abng_kernels.rs` (new, opt-in via builtin name) |
| Profile zones | already exist; just instrument the scaled demos |

**Tests.** Each pass / kernel must:
- preserve byte-equality with the un-optimized path (parity gate)
- demonstrate measurable speedup on a benchmark

**Note.** AOT compilation (`cjcl compile foo.cjcl → foo.exe`) is **explicitly out of scope** for Phase 0.6. That's a multi-month commitment for Phase 0.7+ and depends on whether CJC-Lang's long-term identity is "research vehicle" or "production language." This phase only does the *incremental* work that doesn't commit to that direction yet.

---

### Item 8: TidyView-parity training pipeline (NEW direction; multi-phase work begins here)

**Goal.** The CJC-Lang training story currently lags TidyView's data-engine performance by orders of magnitude. v0.1.7's TidyView v3 hit ~108 ns/op on sealed lookup; ABNG training at scale is closer to milliseconds per observation. Phase 0.6 starts closing this gap.

**What to build:**

This is a *direction* more than a single deliverable. Phase 0.6's contribution is to lay the foundation:

1. **Profile ABNG training at scale** (using the existing `profile_zone_*` instrumentation from chess RL v2.3). Identify the actual hot path under a realistic workload (e.g., the lineage-scaled demo with 10⁴ rows). The hypothesis: `audit_chain_hash` and `blr_update`'s Cholesky update will dominate. Confirm or refute with measurements.
2. **Apply TidyView's lessons:**
   - **Cat-aware paths** (TidyView Phase 2): when input features are categorical, route through `Vec<u32>` codes, not `Vec<String>`. ABNG's codebook is already integer-coded; check whether the *application* layer is paying for any String allocations.
   - **Streaming set-op fast paths** (TidyView Phase 3): chunked dispatch when one or both operands are bitmask-shaped. Less directly applicable but the *pattern* generalizes — when one operand is "All" or "Empty", short-circuit.
   - **Sealed-lookup discipline** (TidyView Phase 10): ABNG's codebook is a one-shot setup. Once installed, lookups should be O(1) array index, not O(log n) BTreeMap descent. This may already be the case; verify.
3. **Build the equivalent of `cjc_data::detcoll`** for ABNG's data flow — a deterministic-by-construction collection family for whatever ABNG most-commonly accumulates over.

This item is **scoped as research, not mandatory shipping.** The Phase 0.6 deliverable is: a profile + a written analysis of what ABNG would need to match TidyView's performance discipline + at least one concrete kernel that demonstrates the pattern.

**Locations.**

| Concern | File |
|---|---|
| Profile harness | extend `bench/abng_micro/` with `profile_at_scale.rs` |
| Written analysis | `docs/abng/PHASE_0_6_TRAINING_PIPELINE.md` (new) |
| Demonstration kernel | `crates/cjc-abng/src/<kernel>.rs` (TBD based on profile) |

---

## v13 bump consolidation strategy

**Item 4 forces v13. Items 1, 2, 3, 5, 6, 7, 8 do not.**

If you do Item 4, bump magic *exactly once* in Phase 0.6 and re-lock all 15 canaries simultaneously. The same discipline as the v12 bump.

Recommended commit ordering:

1. **First commit:** Item 1 (CI) + Item 2 (benches) — establish the baseline before any optimization
2. **Second commit:** Item 3 (smart-replay fast-forward) — first measured perf win, no magic bump
3. **Third commit (joint v13 bump):** Item 4 (batch_observe) — re-lock canaries
4. **Subsequent commits:** Items 5, 6, 7 in any order — each independent
5. **Item 8** ships as a doc + one demonstration kernel; the broader push spans Phases 0.7+

If you skip Item 4 (defer to 0.7), no v13 bump is needed and Phase 0.6 stays on v12.

---

## Recommended ordering across all 8 items

```
1. Item 1 (cross-platform CI)        — gates everything else
   ↓
2. Item 2 (benches at scale)         — establishes baseline numbers
   ↓
3. Item 3 (smart-replay fast-forward) — first perf win
   ↓
4. Item 4 (batch_observe, v13 bump)  — second perf win
   ↓
5. Item 5 (all 6 trigger types demos) — independent
   ↓
6. Item 6 (scaled Phase 0.6 demos)    — independent (depends on Item 4 if using batch_observe)
   ↓
7. Item 7 (compiler / interpreter)    — independent
   ↓
8. Item 8 (TidyView-parity foundation) — depends on Items 2 + 7 stable
```

---

## Verification loop (run before merge)

All baseline gates from Phase 0.5 must continue to pass at counts ≥ the v12 floor:

```bash
# From the worktree root:
cargo test -p cjc-abng --lib                                # ≥ 275
cargo test --test abng                                       # ≥ 460
cargo test --test prop_tests abng_decision                   # ≥ 6 × 256
cargo test --test bolero_fuzz abng_decision                  # ≥ 8
cargo test -p cjc-cli --test abng_cli_integration            # ≥ 43
cargo test -p cjc-cli --lib toml_min                         # ≥ 29
```

All Phase 0.5 demos must continue to pass:

```bash
cargo test --test test_abng_pinn_uncertainty                 # 13
cargo test --test test_abng_tabular_gp                       # 10
cargo test --test test_abng_lineage_attestation              # 16
cargo test --test test_chess_rl_v2_6_abng                    # 19
cargo test --test test_abng_lineage_attestation_cjcl         # 9
cargo test --test test_abng_pinn_uncertainty_cjcl            # 9
cargo test --test test_abng_tabular_gp_cjcl                  # 9
cargo test --test test_abng_ood_detection_cjcl               # 9
cargo test --test test_abng_adaptive_triggers_cjcl           # 9
cargo test --test test_abng_calibration_cjcl                 # 10
cargo test --test test_abng_drift_detection_cjcl             # 9
cargo test --test test_abng_compact_log_cjcl                 # 7
cargo test --test test_abng_maturity_inspection_cjcl         # 10
```

Plus the broader workspace gates:

```bash
cargo test --workspace --release --lib                       # ≥ 2,440
cargo test --test physics_ml --release                       # ≥ 107
cargo test --test test_chess_rl_v2 --release                 # ≥ 97 (~12 min wall clock)
```

After all gates pass, inspect the canaries:

```bash
# Magic byte (should be \x0C if no v13 bump, \x0D if Item 4 shipped):
grep -n "const MAGIC" crates/cjc-abng/src/serialize.rs

# Audit kind tag table (should still show 0x1C, plus 0x1D if Item 4):
grep -n "0x1[CD]" crates/cjc-abng/src/audit.rs

# Locked canaries (count should match the inventory above):
grep -rn "CANARY_HEX:" tests/test_abng_*.rs tests/abng/decide_step_canary_tests.rs
```

---

## Documentation updates expected

Mirror the Phase 0.5 documentation discipline:

| Doc | Update |
|---|---|
| `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` | Update header (magic v13 if Item 4; builtin count; audit kinds 29 → 30 if `0x1D BeliefUpdateBatch` ships); §1 phase status table (add Phase 0.6 row); §3.4 builtin surface (+2-4 for batch builtins); §3.6 audit-kind tag table (add `0x1D` if Item 4); Appendix A file map; Appendix B test counts |
| `docs/abng/PHASE_0_6_DESIGN.md` (NEW) | Post-hoc design note — mirror `PHASE_0_5_DESIGN.md` structure: per-item summary, per-item bullets, wire-format additions, test counts, "What's *not* in 0.6" section |
| `docs/abng/PHASE_0_6_TRAINING_PIPELINE.md` (NEW) | Item 8 deliverable — profile output + analysis + recommended Phase 0.7+ direction |
| `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 ABNG Adaptive Belief Radix Graph (Phase 0.1).md` | Append "Phase 0.6 amendment" section after the existing "Phase 0.5 amendment" (which the Phase 0.5 ship added separately) |
| `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md` | Prepend "Phase 0.6 deliverable" block above the existing "Phase 0.5 (v12) deliverable" block |
| `~/.claude/projects/C--Users-adame-CJC/memory/MEMORY.md` | Update one-line ABNG entry to reflect Phase 0.6 status |

---

## Recommended next-session prompt (stacked role group prompt)

Paste the following into a fresh Claude Code session inside the `abng-phase-0-6` worktree:

```
# CJC-Lang ABNG Phase 0.6 — Stacked Role Group Prompt

## ROLE

You are a stacked systems team working inside the CJC-Lang
(Computational Jacobian Core) compiler repository, specifically
inside the ABNG (Adaptive Belief Network Graph) sub-system at
`crates/cjc-abng/`. You are continuing the ABNG roadmap from Phase
0.5 (already merged into master at commit `fe0b602`).

You consist of:

1. **Lead ABNG Architect** — owns ABNG's external contract: which
   builtins exist, their semantics, the `Value` enum's invariance,
   the snapshot wire format. Phase 0.6 may introduce a v13 bump
   (Item 4 batch_observe); if so, `MAGIC` becomes `b"ABNG\x0D"` and
   all 15 locked canaries get re-locked simultaneously.

2. **Compiler / Interpreter Performance Engineer** — owns the
   Lexer → Parser → AST → HIR → MIR → Exec data flow with a focus
   on perf. Phase 0.6 lands incremental MIR optimizer passes
   (LICM, CSE, function inlining) and native specialization for
   ABNG hot paths. AOT compilation is explicitly OUT OF SCOPE for
   0.6 — that's a multi-month commitment for 0.7+.

3. **Runtime Performance Engineer** — owns memory, dispatch, and
   ABNG's per-thread arena (`thread_local! BTreeMap<i64,
   AdaptiveBeliefGraph>`). Phase 0.6 ships native batch_observe +
   bulk BLR update (Item 4), the smart-replay fast-forward
   optimization (Item 3), and a new `cjc-runtime/src/abng_kernels.rs`
   module for hot-path specialization (Item 7).

4. **Numerical Computing Engineer** — owns deterministic floating
   point. Phase 0.6's batch BLR update must produce
   bit-identical canonical_bytes to the per-row equivalent path
   for any (n, d, observation_order) tuple. The Kahan compensation
   register survives batch updates intact. The 15 locked canary
   hashes are non-negotiable unless deliberately re-locked under
   the v13 bump.

5. **ML Workload Engineer** — NEW role for Phase 0.6. Owns the
   "demos at scale" deliverable (Item 6): real PINN benchmarks
   (10⁴ collocation points), real classifier calibration (not
   synthetic injection), large-N tabular benchmarks vs sklearn.
   This role makes the perf claims measurable. Plus owns the new
   demos for the 5 unfired trigger types (Item 5: Grow / Split /
   Prune / Compress / Freeze).

6. **Determinism & Reproducibility Auditor** — enforces
   bit-identical output across runs AND platforms. Phase 0.6 Item 1
   wires GitHub Actions CI on Linux + macOS + Windows, gating on
   the 15 locked canaries. Any platform divergence is a real bug
   to investigate.

7. **QA Automation Engineer** — owns the 5 ABNG-direct gates plus
   the 13 Phase 0.5 demo files (all must continue passing at the
   v12 floor or above):
   - `cargo test -p cjc-abng --lib` (≥ 275)
   - `cargo test --test abng` (≥ 460)
   - `cargo test --test prop_tests abng_decision` (≥ 6 × 256)
   - `cargo test --test bolero_fuzz abng_decision` (≥ 8)
   - `cargo test -p cjc-cli --test abng_cli_integration` (≥ 43)

   No item ships without all gates green. Plus
   `cargo test --workspace --release --lib`,
   `cargo test --test physics_ml --release`, and
   `cargo test --test test_chess_rl_v2 --release`.

## PRIME DIRECTIVES

1. **Do not break the audit chain.** Every state mutation MUST
   append exactly one `AuditEvent` with
   `new_hash = sha256(previous_hash ‖ canonical_payload)`. The v13
   bump is allowed (in fact required for Item 4); silent format
   changes are not.

2. **Do not introduce hidden non-determinism.** All FP reductions
   go through Kahan or pairwise sum. RNG is SplitMix64 with
   explicit seed threading. Maps are `BTreeMap` only. Phase 0.6
   adds the cross-platform CI requirement: Linux + macOS + Windows
   builds must all produce identical canary hashes.

3. **Preserve `Value` enum layout.** ABNG handles cross the language
   boundary as `Value::Int(i64)`, `Value::Tensor`, `Value::String`,
   `Value::Bytes`, `Value::Array`, `Value::Bool`, `Value::Float`.
   No new `Value` variant.

4. **Both executors must agree.** Every new builtin works in
   both `cjc-eval` (AST-walk) AND `cjc-mir-exec` (MIR
   register-machine) with identical output. Parity tests in
   `tests/abng/parity_p3*.rs` are the regression gate. Phase 0.6
   adds parity for `abng_observe_batch`, `abng_blr_update_batch`,
   any new builtins shipped in this phase.

5. **All 15 Phase 0.5 canaries must pass without modification on
   every Phase 0.6 commit until Item 4 ships.** Once Item 4 ships
   (v13 bump), all 15 are re-locked simultaneously in the same
   commit. No partial canary updates.

6. **Performance claims must be measured, not asserted.** Item 2
   (benchmarks at scale) ships first specifically so every
   subsequent perf claim can cite a number.

7. **Phase 0.5 demos MUST continue to pass.** The 13 demo files
   (4 Rust + 9 CJC-Lang) gate every Phase 0.6 commit. The
   `_scaled` versions in Item 6 are *additions*, not replacements.

8. **Language primitives stay minimal.** Higher-level training
   utilities live in user `.cjcl` source. Item 8 (TidyView-parity
   training) is a research direction, not a commitment to ship a
   training framework as a Rust crate.

## SCOPE — Phase 0.6 8 items

Read `docs/abng/PHASE_0_6_HANDOFF.md` for the full breakdown. The
eight items are:

1. **Cross-platform determinism CI.** GitHub Actions on Linux +
   macOS + Windows. Gates on the 15 locked canaries.

2. **Performance benchmarks at scale.** `bench/abng_micro/`,
   `bench/abng_vs_sklearn/`, `bench/abng_pinn_scale/`,
   `bench/abng_lineage_at_scale/`. Establishes baseline numbers
   before any optimization.

3. **Smart-replay fast-forward optimization.** Phase 0.5 Item 2
   shipped the API + tamper check; Phase 0.6 ships the cycle-saving
   skip-the-observe layer. Determinism contract: smart_replay
   output is byte-identical to naive replay.

4. **Native batch_observe + bulk BLR update.** New `0x1D
   BeliefUpdateBatch` audit kind, `abng_observe_batch` and
   `abng_blr_update_batch` builtins. Forces v13 bump. Re-locks all
   15 canaries.

5. **Adaptive triggers — fire all 6 types.** Phase 0.5 demonstrates
   Merge only. Phase 0.6 adds CJC-Lang demos for Grow, Split,
   Prune, Compress, Freeze, with locked canaries for each.

6. **Phase 0.6 demos at scale + with noise.** Each Phase 0.5 demo
   gets a `_scaled` sibling with 10× to 1000× more samples,
   additive noise, higher feature dim, real classifiers (vs
   synthetic injection where applicable). Phase 0.5 demos continue
   to pass unchanged.

7. **Compiler / interpreter perf prep work.** Incremental MIR
   optimizer passes (LICM, CSE, function inlining) + native
   specialization in `cjc-runtime/src/abng_kernels.rs`. AOT is
   explicitly OUT OF SCOPE for 0.6.

8. **TidyView-parity training pipeline foundation.** Profile ABNG
   training at scale, write up the analysis of what would close
   the gap to TidyView's perf discipline, ship at least one
   concrete kernel demonstrating the pattern. Multi-phase work
   begins here; Phase 0.6's deliverable is foundation, not
   completion.

## RECOMMENDED ORDER

1. Item 1 (cross-platform CI) — gates everything else
2. Item 2 (benches at scale) — establishes baselines
3. Item 3 (smart-replay fast-forward) — first perf win
4. Item 4 (batch_observe, v13 bump) — second perf win
5. Item 5 (all 6 trigger types) — independent
6. Item 6 (scaled Phase 0.6 demos) — independent
7. Item 7 (compiler / interpreter) — independent
8. Item 8 (TidyView-parity foundation) — depends on Items 2 + 7

## TEST PLACEMENT (mirror Phase 0.5)

Put new tests in the same locations as Phase 0.5 did:

- **In-crate unit tests** for new types/methods →
  `crates/cjc-abng/src/<module>.rs::tests`
- **Integration tests** for cross-module behavior →
  `tests/abng/<feature>_tests.rs` (new files allowed:
  `tests/abng/batch_observe_tests.rs`, etc.)
- **Property tests** for invariants →
  `tests/prop_tests/abng_decision_props.rs` (extend with new
  properties for batch equivalence, smart_replay parity, etc.)
- **Fuzz targets** for adversarial blobs →
  `tests/bolero_fuzz/abng_decision_fuzz.rs` (extend with
  adversarial batch sizes, malformed BatchUpdate payloads)
- **CLI integration tests** for new CLI behavior →
  `crates/cjc-cli/tests/abng_cli_integration.rs`
- **CJC-Lang demos** → `tests/abng_demos/<name>_source.rs` +
  root-level `tests/test_abng_<name>_cjcl.rs` (mirror the
  Phase 0.5 demo pattern exactly)
- **Benchmarks** → `bench/abng_micro/`, `bench/abng_vs_sklearn/`,
  `bench/abng_pinn_scale/`, `bench/abng_lineage_at_scale/` (new)

## DOCUMENTATION PLACEMENT (mirror Phase 0.5)

Put new docs in the same locations as Phase 0.5 did:

- **Architecture doc updates** →
  `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` (header, §1 phase
  status table, §3.4 builtin surface, §3.6 audit-kind tag table,
  Appendix A file map, Appendix B test counts)
- **Post-hoc design note** → `docs/abng/PHASE_0_6_DESIGN.md` (NEW
  — mirror PHASE_0_5_DESIGN.md structure)
- **Training-pipeline analysis** →
  `docs/abng/PHASE_0_6_TRAINING_PIPELINE.md` (NEW — Item 8
  deliverable)
- **ADR amendment** → append "Phase 0.6 amendment" section to
  `CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0023 ABNG Adaptive Belief
  Radix Graph (Phase 0.1).md` after the Phase 0.5 amendment
- **Project memory** → prepend "Phase 0.6 deliverable" block to
  `~/.claude/projects/C--Users-adame-CJC/memory/project_abng.md`
- **MEMORY.md** → update the one-line ABNG entry to reflect
  Phase 0.6 status

## VERIFICATION LOOP (run before merge)

Phase 0.5 baseline gates (must remain at floor or above):

```bash
cd C:/Users/adame/CJC/.claude/worktrees/abng-phase-0-6
cargo test -p cjc-abng --lib                          # ≥ 275
cargo test --test abng                                 # ≥ 460
cargo test --test prop_tests abng_decision             # ≥ 6 × 256
cargo test --test bolero_fuzz abng_decision            # ≥ 8
cargo test -p cjc-cli --test abng_cli_integration      # ≥ 43
cargo test -p cjc-cli --lib toml_min                   # ≥ 29
```

Phase 0.5 demo files (all must continue passing):

```bash
# 4 Rust application demos
cargo test --test test_abng_pinn_uncertainty           # ≥ 13
cargo test --test test_abng_tabular_gp                 # ≥ 10
cargo test --test test_abng_lineage_attestation        # ≥ 16
cargo test --test test_chess_rl_v2_6_abng              # ≥ 19

# 9 CJC-Lang demos
cargo test --test test_abng_lineage_attestation_cjcl   # ≥ 9
cargo test --test test_abng_pinn_uncertainty_cjcl      # ≥ 9
cargo test --test test_abng_tabular_gp_cjcl            # ≥ 9
cargo test --test test_abng_ood_detection_cjcl         # ≥ 9
cargo test --test test_abng_adaptive_triggers_cjcl     # ≥ 9
cargo test --test test_abng_calibration_cjcl           # ≥ 10
cargo test --test test_abng_drift_detection_cjcl       # ≥ 9
cargo test --test test_abng_compact_log_cjcl           # ≥ 7
cargo test --test test_abng_maturity_inspection_cjcl   # ≥ 10
```

Plus the broader gates:

```bash
cargo test --workspace --release --lib                 # ≥ 2,440
cargo test --test physics_ml --release                 # ≥ 107
cargo test --test test_chess_rl_v2 --release           # ≥ 97 (~12 min)
```

Phase 0.6-specific new gates (added by Items 5 + 6):

```bash
# Item 5 — 5 new trigger demos
cargo test --test test_abng_grow_trigger_cjcl
cargo test --test test_abng_split_trigger_cjcl
cargo test --test test_abng_prune_trigger_cjcl
cargo test --test test_abng_compress_trigger_cjcl
cargo test --test test_abng_freeze_trigger_cjcl

# Item 6 — 8 scaled-demo siblings
cargo test --test test_abng_pinn_scaled_cjcl
cargo test --test test_abng_tabular_scaled_cjcl
cargo test --test test_abng_lineage_scaled_cjcl
cargo test --test test_abng_ood_scaled_cjcl
cargo test --test test_abng_calibration_scaled_cjcl
cargo test --test test_abng_drift_scaled_cjcl
cargo test --test test_abng_compact_scaled_cjcl
cargo test --test test_abng_maturity_scaled_cjcl

# Item 4 — batch_observe (if v13 ships)
cargo test --test abng batch_observe_tests
```

## USER PREFERENCES

- **Small, stop-and-confirm units rather than batched changes.**
  Pause after each item and summarize before moving to the next.
  Don't bundle multiple items into one mega-commit unless they
  share a v13 bump (Item 4).
- **Commit per item once gates pass.** Each numbered item is a
  logical commit; the v13 bump for Item 4 is one commit.
- **Stop-and-confirm at every magic-bump or wire-format-change
  decision.** v13 is authorized for Item 4 only (if shipped). Any
  *other* proposed bump must pause and confirm first.
- **Performance claims must be measured.** No asserting "linear
  scaling" without a benchmark to back it up.
- **The user's name is Adam Ezzat.** (From auto-memory.)

## START

1. Verify the baseline by running the 5 gates above plus the 13
   demo files. All should pass with the v12 floor counts.
2. Read `docs/abng/PHASE_0_6_HANDOFF.md` for the full per-item
   breakdown.
3. Propose Item 1 + Item 2 (CI + baseline benches) as the first
   concrete unit of work — this is the lowest-risk, highest-leverage
   start. Both can be parallel work (Item 1 is config-file only,
   Item 2 needs Rust code).
4. Ask for user confirmation before starting.
```

---

## Files added or substantially modified during Phase 0.5 (for context)

To know what code surface Phase 0.6 will be touching, read these files
first — they have the most recent design context:

### Heavily modified in Phase 0.5

- `crates/cjc-abng/src/audit.rs` — 29 audit kinds, frozen tag table 0x00..0x1C
- `crates/cjc-abng/src/serialize.rs` — v12 magic, `replay_with_options`, `smart_replay`, ProvenanceMismatch + StatsSnapshotMismatch error variants
- `crates/cjc-abng/src/graph.rs` — `stamp_provenance` method, idempotent semantics
- `crates/cjc-abng/src/dispatch.rs` — 75 builtins (+2 from 0.4: `abng_stamp_provenance`, `abng_provenance_stamp`)
- `crates/cjc-abng/src/node.rs` — `provenance_stamp_hash: [u8; 32]` field
- `crates/cjc-abng/src/stats.rs` — `canonical_bytes` 24B → 32B (Kahan compensation), `from_canonical_bytes` constructor
- `crates/cjc-abng/src/predict_snap.rs` — PRED_MAGIC `\x01` → `\x02`, trailing `provenance_stamp_hash`
- `crates/cjc-cli/src/commands/abng.rs` — `train --config` flag, full TOML schema
- `crates/cjc-cli/src/toml_min.rs` — NEW, hand-rolled minimal TOML parser
- `crates/cjc-repro/src/kahan.rs` — `KahanAccumulatorF64::compensation_bits()` and `from_components()`

### New test surface in Phase 0.5

- `tests/abng_demos/` — 10 source files (`harness.rs`, `lineage_source.rs`, `pinn_source.rs`, `tabular_source.rs`, `ood_source.rs`, `adaptive_source.rs`, `calibration_source.rs`, `drift_source.rs`, `compact_source.rs`, `maturity_source.rs`)
- `tests/test_abng_*.rs` — 13 root-level test files
- `tests/abng/provenance_tests.rs` — NEW

### Reference design notes

- `docs/abng/PHASE_0_5_HANDOFF.md` — predecessor of this document
- `docs/abng/ABNG_CURRENT_ARCHITECTURE.md` — source-of-truth contract
- `docs/abng/PHASE_0_4_DESIGN.md` … `PHASE_0_3a_DESIGN.md` — phase history

---

*This handoff was generated at the end of an extended Phase 0.5
session that landed the v12 wire-format bump, smart_replay API,
TOML config support, chess RL v2.6 retrofit scaffold, and 9 CJC-Lang
demos covering 11 distinct ABNG capabilities. The merge to master
landed at commit `fe0b602`. The branch `claude/abng-phase-0-6`
should be forked from `master` (commit `fe0b602`); the next session
may want to commit Phase 0.6 work as one PR per item once each
item's gates pass.*
