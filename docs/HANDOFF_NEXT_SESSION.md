# Next-Session Handoff — Locke v0.7 (heavy items) → ABNG Performance Push

**Date stamped:** 2026-05-29
**Branch suggestion:** continue on `claude/adoring-shtern-01c720` or branch off at `80203df` (Locke v0.6.4 just landed, ABNG Phase 0.10 already shipped). Use `git fetch origin && git checkout claude/adoring-shtern-01c720` to land in the right place.
**Prior session result:** UCI Diabetes-130 AUC **0.6645 ± 0.0018** (n=3 seeds, full 101K) on the linear BLR head. Locke v0.6.4 shipped E9008 + E9064. Workspace clean, 629 ABNG canaries unmoved, 284 cjc-locke lib + 194 tests/locke integration green.

---

## 0. The ask

Two phases, in order:

**Phase A — Finish Locke heavy items (v0.7+).** The five items each labelled "≥ 2 sessions of focused work" in [`Locke Roadmap.md` §v0.7+ heavy](../CJC-Lang_Obsidian_Vault/07_Data_and_CLI/Locke/Locke%20Roadmap.md). These are explicit on the roadmap; they didn't fit Phase 0.10.

**Phase B — ABNG performance push.** Get throughput / memory / power beyond the §4.E baseline (213.84 s for the full 101K trial). Use the unused ABNG features the Phase 0.10 blog post (§8.4 / §10.C) catalogued, plus data-structure efficiency, plus thermal/power discipline (the [Deterministic-Green-Compute](https://adamezzat1.github.io/blog/posts/deterministic-green-compute/) framing).

The Phase B brief is **the brief that started the whole campaign** — "ABNG should be fast enough that the audit chain isn't a cost." If a 20K-row trial is sub-second wall clock and the full 101K trial is under a minute, the determinism story is stronger by default.

---

## ROLE

You are a stacked systems team inside the CJC-Lang compiler repository:

1. **Lead Language Architect** — owns language semantics, type system soundness, and feature design
2. **Compiler Pipeline Engineer** — owns Lexer → Parser → AST → HIR → MIR → Exec data flow
3. **Runtime Systems Engineer** — owns memory model, GC/NoGC boundary, dispatch, and builtins
4. **Numerical Computing Engineer** — owns deterministic BLAS, SIMD, accumulator correctness, AD, and ABNG's BLR update path
5. **Determinism & Reproducibility Auditor** — enforces bit-identical output across runs and platforms (28 SHA-256 canaries are load-bearing)
6. **QA Automation Engineer** — owns test infrastructure, parity gates, regression prevention

The Performance Auditor role gets temporary precedence in Phase B — every optimisation lands behind canary-locked benchmarks.

---

## PRIME DIRECTIVES (carry forward from CLAUDE.md)

1. **Do not break the compiler pipeline** — `Lexer → Parser → AST → [TypeChecker] → HIR → MIR → [Optimize] → Exec`.
2. **Do not introduce hidden allocations or GC usage** in NoGC-verified paths.
3. **Maintain deterministic execution** — same seed = bit-identical output. The 28 SHA-256 canaries must stay unmoved, or each new canary needs a vault note + ADR.
4. **Both executors must agree** — every change to ABNG semantics must reproduce in both `cjcl run` (eval) and `cjcl run --mir-opt` (MIR-exec).
5. **Never silently refactor unrelated systems.**

For Phase B specifically:
6. **Every performance change ships with a benchmark.** No "feels faster" — wall-clock + memory + (where measurable) Joules.
7. **Determinism canaries are load-bearing**: chain head + Merkle root must reproduce bit-for-bit across optimisations.

---

# PHASE A — Locke v0.7 heavy items

Finish the five items deferred in ADR-0036 v0.7 part 2 and the v0.7+ heavy section of the Roadmap. Each is multi-step but together they close the "Locke v0.7 fully shipped" milestone.

Order by effort × value:

## A1. Per-axis BeliefScore composition migration (v0.7 part 2)

[`docs/locke/ABNG_PER_LEAF_BELIEF.md`](locke/ABNG_PER_LEAF_BELIEF.md) and [ADR-0036](../CJC-Lang_Obsidian_Vault/13_ADRs/ADR-0036%20Locke%20v0.7%20—%20Per-axis%20BeliefScore%20Composition%20Algebra.md) shipped the *algebra* but didn't migrate the two consumers.

**Tasks:**
- Migrate `api::belief_report_from_locke_with_model` → use `compose` / `compose_many` from the algebra module. Currently the function computes per-axis scores inline via `penalty_from_findings_with_model` — should route through `algebra::compose` for traceability.
- Migrate `gate::diff_reports` → use `le_componentwise` for the partial-order diff.
- Add a property test confirming the migrated path is byte-identical to the inline path (regression gate).
- Document migration in ADR-0036 (add a v0.7 part 2 section) or open ADR-0038 if scope expands.

**Why first:** smallest scope, closes ADR-0036 cleanly, no new design decisions.

## A2. Per-value category lineage

The current `TracedDataFrame` tracks lineage at the **DataFrame** level. Categorical columns also need **per-value** lineage:

```
raw → normalized → grouped → encoded → embedding
```

**Tasks:**
- New module `crates/cjc-locke/src/per_value_lineage.rs`.
- `PerValueLineage` struct holding a chain of `(stage, transform_fn_hash, output_value)` per *distinct input value*.
- Wire into `CategoricalQualityConfig` so the existing detectors can attach lineage links to findings.
- Add `cjcl locke trace-value <col> <value>` CLI command to dump the chain.
- Tests in the existing pattern (unit + integration + proptest + bolero fuzz).

**Why now:** the diabetes-130 blog post §6 explicitly says "future Locke versions could auto-detect `?` / `NA` / `-` / `NULL`" — auto-detect is shipped (E9008), but the **trace** of what each value became after normalisation isn't yet visible. Lineage finishes the story.

## A3. Governance workflows (suppression + ownership + required-finding policies)

The largest single addition. Locke produces 142+ findings on real datasets (diabetes-130 audit). At scale this is unmanageable without suppression and policy gates.

**Tasks:**
- New `cjcl locke policy` subcommand and `.cjcl-locke.toml` config file format.
- Suppression: `suppress E9082 column=weight reason="real distinction, not typo"`.
- Owner annotations: `owner team-data-platform column=patient_nbr`.
- Required-finding policies: `require E9004==0 owner=team-data-platform`.
- Gate integration: `cjcl locke gate` checks policy and exits non-zero on violations.
- Audit log of suppression decisions (preserves Locke's content-addressed-trace property).
- Tests + ADR-0039 covering the policy DSL semantics.

**Why third:** large but well-understood; the rest of the team can use it immediately to triage diabetes-130's 142 findings.

## A4. Ontology / taxonomy consistency

Hyphen/underscore variants, common-prefix taxonomy inference, hierarchy fragmentation.

**Tasks:**
- New `detect_taxonomy_inconsistencies(df, cfg) -> Vec<ValidationFinding>` in a new `crates/cjc-locke/src/taxonomy.rs`.
- Detect hyphen/underscore variants of the same concept (e.g., `medical-specialty` vs `medical_specialty`).
- Detect common-prefix taxonomy: `diag_1` / `diag_2` / `diag_3` share prefix → flag for inference of a `diag_*` family.
- Detect hierarchy fragmentation: `admission_type_id ∈ {1..8}` mapped to free-text in `admission_type_label` columns with mismatched cardinality → flag the desync.
- New finding codes E9100-E9105 (next free range after categorical E9080-E9086 and PII E9090-E9093).

**Why fourth:** the most domain-specific; the tests need real medical / financial / sci data fixtures.

## A5. Text drift (vocabulary KS + token-entropy drift + language distribution shift)

The largest dependency-impact item — requires a tokenizer that doesn't break the zero-external-dependency rule.

**Tasks:**
- Build a minimal byte-pair-tokenizer in `crates/cjc-locke/src/tokenizer.rs` (zero deps, deterministic, vocab-frozen).
- Vocabulary KS test: detect when train-set vocabulary fingerprint diverges from test-set.
- Token entropy drift: compare distribution of token frequencies.
- Language distribution shift: detect when character n-gram distribution changes (catches a French test set when training was English).
- New finding codes E9110-E9115.

**Why last:** the tokenizer is the biggest unknown. Could be split into "ship a tokenizer" + "ship the drift detectors" across two sessions.

### A: stop-the-bleeding gate

Before shipping any of A2-A5, ensure A1 (the algebra migration) is complete. The composition algebra is the substrate for everything downstream; if A2's per-value lineage emits new finding codes, the BeliefScore axis they map to should be composable via the algebra, not inlined.

### A: pre-flight for Phase A

```bash
cargo test --workspace --release        # baseline: must be green
cargo test -p cjc-locke --lib --release  # 284
cargo test --test locke --release        # 194
cargo test --test abng --release         # 629 (no Locke changes should regress ABNG)
```

After each of A1-A5: same battery + the relevant new tests added in that step.

---

# PHASE B — ABNG performance push

The Phase 0.10 blog post §8 catalogued ABNG's architectural fingerprint *that mattered* on diabetes-130, and §10.C noted **MLP-head infrastructure already exists but unused**. Phase B leverages everything that sat unused, then adds data-structure / numerical / power-aware optimisations.

**Baseline to beat (from Phase 0.10 §4.E and §4.F):**

| Metric | Baseline | Phase B floor target | Phase B stretch target |
|---|---|---|---|
| Full 101K trial wall clock (single seed) | **213.84 s** | < 100 s | < 30 s |
| 20K trial wall clock | ~40 s | < 15 s | < 5 s |
| Multi-seed 3 × 101K | 610.77 s | < 240 s | < 90 s |
| Peak memory (estimated) | unknown — measure first | -30% | -50% |
| Mean CPU wattage (laptop power meter) | unknown — measure first | -20% | -40% |
| AUC at the same config | **0.6645 ± 0.0018** | must stay within ± 2σ (0.0036) | identical (chain head reproducible) |

**The hard constraint: AUC must reproduce bit-identically (per-seed chain head) or stay within one standard deviation, with new canary locks for any update path change.**

## B1. Wire up the unused infrastructure (§4.C MLP heads — design doc already exists)

[`docs/abng/PHASE_0_10_SECTION_4C_DESIGN.md`](abng/PHASE_0_10_SECTION_4C_DESIGN.md) maps this — `LeafHead.hidden_dims: Vec<u32>` already exists, `leaf_forward` already runs MLP forward in `cjc-ad::GradGraph`, `leaf_set_param` already canary-locks weight updates. **No new ABNG plumbing is needed**; the work is harness-side.

**Tasks:**
- Add `train_mlp_then_blr` driver in `tests/abng/dataset_a_diabetes130.rs` per the design doc's §step 1. Expected: ~250 LOC.
- New ignored test `diabetes130_subsample_trial_mlp_head`.
- Verify chain head reproducibility. If a new `AuditKind` is needed, add a canary lock (per the determinism contract).
- Measure: 20K AUC vs §4.A's 0.6312 baseline. Expected lift: +0.01 to +0.03. If lift < 0.005, ship the negative result and move on.

**Why first in Phase B:** the design doc identified this as the lowest-risk lever for AUC; if it ships AUC ≥ 0.66 at 20K, every other performance optimisation gets to ride on a stronger baseline.

## B2. The unused per-row state from Phase 0.10 §8

From the blog post's architectural fingerprint section, these features are unused or partially used:

| Feature | Current state | Phase B opportunity |
|---|---|---|
| **OOD score** | Infrastructure exists in `cjc-abng`; `run_trial` doesn't call it during eval | Wire into the trial; could give a per-row confidence score, useful for §4.D Part 2 abstention without needing per-leaf priors |
| **`leaf_set_param` / `leaf_set_params_batch`** | Already canary-locked; only invoked indirectly via §4.C path | Use them to checkpoint trained MLP weights every N steps for warm-starting variance sweeps |
| **`feature_version_hash`** | Set at `set_blr_prior`, never re-stamped | Use it to detect if MLP weights moved and the BLR posterior is stale — would unlock proper MLP-then-BLR sequencing |
| **`profile_zone_*` builtins** | Used in chess RL v2.3 (10.4 s → 1.3 s rollout speedup); ABNG harness doesn't use them | Wrap `train_step`, `descend`, `transform.transform` in zones to find the hot 80% |
| **Phase 0.9.5 result-path optimisations** | Shipped but may have regressed for the diabetes harness | Re-run the perf benchmarks (`bench/abng_result_profile`); look for `result.routing_feature_columns` hot spots |

**Tasks:**
- Add profile zones to `run_trial` first (no behaviour change, just measurement).
- Run with `--release` on full 101K. Output: per-zone time breakdown.
- The biggest zone is the first optimisation target.

## B3. Data-structure efficiency (speed + memory)

ABNG uses `BTreeMap` everywhere (correctly — determinism over speed). For Phase B, swap to determinism-preserving faster collections from cjc-data's `detcoll` module.

**Tasks:**
- **D-HARHT Memory `SealedU64Map`** — already shipped (v0.1.7, ADR-0027 or similar). Use it for: routing-prefix → leaf_id lookup (currently `BTreeMap<Vec<u8>, NodeId>`), per-leaf BLR state lookup. Expected: 2-3× speedup on the routing hot path.
- **Per-leaf BLR precision matrix layout**: currently `Vec<Vec<f64>>` (row-major). Switch to a flat `Vec<f64>` with stride-based indexing — better cache locality, fewer allocations. Use `cjc-runtime::tensor::Tensor` if the layout matches.
- **Phi vector lifetime**: currently allocated per `train_step` call. Use an arena (`bumpalo::Bump` or hand-rolled bump allocator) to reset between rows. Bonus: makes the NoGC verification cleaner.
- **MLP weights at f32 instead of f64**: BLR posterior must stay f64 (precision matters for the conjugate update), but the MLP penultimate features can be f32 in flight, upcast to f64 at the BLR boundary. Halves memory bandwidth on the MLP forward.

**Determinism gate:** every collection swap needs a canary lock — the iteration order must be provably identical. Add proptest that builds a `BTreeMap` and a `SealedU64Map` from the same `(key, value)` stream and asserts byte-identical chain heads after a fake train_step batch.

## B4. SIMD on the BLR conjugate update

The single arithmetic hot spot. `BlrState::update(phi, y)` does a rank-1 outer-product update on the precision matrix (`Λ_n + φφᵀ`) and the precision-weighted mean. Both are dot-product/AXPY-class operations.

**Constraint:** no FMA (per the determinism contract — bit-identical results). SIMD without FMA is still possible — AVX2 `_mm256_mul_pd` + `_mm256_add_pd` separately.

**Tasks:**
- Add a SIMD path behind a runtime check (`is_x86_feature_detected!("avx2")`); fall back to scalar.
- Bench: `bench/abng_micro` should show the per-step cost dropping from O(d²) f64 ops to O(d²/4) for d ~ 266.
- Property test: scalar and SIMD paths produce bit-identical chain heads on a fixed seed.

## B5. Power / thermal discipline (the "green stuff")

Reference: [`docs/abng/deterministic_green_compute_notes.md`](abng/) if it exists; the public blog post is at https://adamezzat1.github.io/blog/posts/deterministic-green-compute/.

The framing: **wall-clock improvement is also a power improvement when the CPU runs cooler.** Specific levers:

**Tasks:**
- **Batch size sweep**: the BLR update is currently per-row. Try processing rows in batches of 64/128/256, computing `Σ φφᵀ` once and folding. Same outputs, fewer matrix-allocation rounds, lower power.
- **Cache-aware iteration order**: when training rows route to many leaves, the per-leaf BLR state is in cold cache. Sort the training stream by leaf_id (post-routing) so each leaf's state stays hot in L2. Determinism is preserved because the seed-driven shuffle still happens before training; the leaf-grouped sort is a post-shuffle reorder.
- **Power measurement** (if hardware allows): use Intel Power Gadget (Windows) or `powerstat` (Linux) for Joules-per-trial. Even a rough number helps quantify the green claim.
- **Thermal margin**: monitor CPU temp during a 30-minute multi-seed sweep. The current `--release` build pushes the laptop to thermal throttle on long benches; reducing wall clock 4× should keep it below the throttle threshold and let the variance sweep run at sustained turbo.

## B6. Memory layout audit (if time)

After B1-B5, profile peak RSS. Likely candidates:

- The `cjc-data::DataFrame` cloning in `take_rows` / `transform.transform` — could the transform stream rows without materialising?
- The audit chain entries store `state_hash: [u8; 32]` per event — 32 bytes × 32K events × 12 leaves = 12 MB. Could use a `[u8; 16]` if collision risk is acceptable.
- `nodes.params` allocates even when `hidden_dims = vec![]` — small waste, but compounds.

---

## Pre-flight for the whole next session

```bash
# 1. Sync the branch
git fetch origin
git checkout claude/adoring-shtern-01c720

# 2. Verify the v0.6.4 baseline is intact
cargo test --workspace --release | grep "test result"

# Expected baseline:
#   cjc-locke --lib: 284
#   tests/locke:     194
#   tests/abng:      629
#   tests/parity:    179+
#   Other crates:    workspace clean

# 3. For Phase B specifically, run the §4.F variance baseline so chain heads are known good:
cargo test --test abng diabetes130_multi_seed_variance_full --release -- --ignored --nocapture
# Expected: AUC = 0.6645 ± 0.0018, chain heads 56af1961... / 590c3fb9... / 89e02f56...
```

## Recommended order

If only a few items fit the next session:

1. **A1** (algebra migration) — closes ADR-0036, no design work
2. **B2 profile zones** (just measurement, zero behaviour change)
3. **B1 MLP heads** (the design doc says +0.02 AUC, ~250 LOC)
4. **B3 data structures** (the biggest performance lever per hour)

A2-A5 are larger commitments — pick one per session.

## Recording results

For each Phase B optimisation, emit to `bench_results/phase_b_<lever>/`:
- `bench.md` — wall clock before/after with hardware fingerprint
- `chain_heads.txt` — per-seed chain heads, must reproduce bit-identically
- `auc_comparison.md` — AUC before/after with std/√n bounds
- `peak_memory.md` (if measured)
- `power.md` (if measured)

## Out of scope (per user instruction)

- Anything that requires changing the determinism contract beyond adding new canary-locked AuditKinds
- Architectural changes to the routing-tree algorithm (Phase B is about throughput, not accuracy beyond what MLP heads land)
- A6+ ideas that don't fit the audit-chain canary model

## Files to know

- Phase 0.10 final state: [`docs/abng/PHASE_0_10_SESSION_NOTES.md`](abng/PHASE_0_10_SESSION_NOTES.md)
- §4.C MLP-head design (for B1): [`docs/abng/PHASE_0_10_SECTION_4C_DESIGN.md`](abng/PHASE_0_10_SECTION_4C_DESIGN.md)
- Locke v0.6.4 work (just shipped): [`docs/locke/SILENT_FAILURES_AUDIT.md`](locke/SILENT_FAILURES_AUDIT.md) + ADR-0037
- Locke Roadmap: [`CJC-Lang_Obsidian_Vault/07_Data_and_CLI/Locke/Locke Roadmap.md`](../CJC-Lang_Obsidian_Vault/07_Data_and_CLI/Locke/Locke%20Roadmap.md)
- Diabetes-130 harness: [`tests/abng/dataset_a_diabetes130.rs`](../tests/abng/dataset_a_diabetes130.rs)
- ABNG architecture: [`crates/cjc-abng/src/graph.rs`](../crates/cjc-abng/src/graph.rs)
- The blog post that's the public face: https://adamezzat1.github.io/blog/posts/abng-diabetes-revisited-with-locke/

## Honest stretch goals

- **Phase A floor:** A1 + A3 (algebra migration + governance workflows) → Locke is institutionally usable at 142-finding scale
- **Phase A target:** A1 + A2 + A3 → Locke v0.7 fully shipped
- **Phase A stretch:** all five — closes the v0.7+ heavy roadmap entirely
- **Phase B floor:** B1 + B2 measurements only — quantify where the time goes
- **Phase B target:** B1 + B3 + B4 → 101K trial under 60s, AUC ≥ 0.6645 (chain heads unchanged)
- **Phase B stretch:** all six — 101K trial under 30s, multi-seed sweep under 90s, ABNG sustains 100K-row training under thermal limits

Each is a publishable result. The floor closes the open roadmap items; the target makes ABNG "fast enough that the audit chain isn't a cost"; the stretch makes the deterministic-green-compute claim quantitatively defensible.

Good luck. The substrate is solid; the levers are identified; the canary contract is the safety net.
