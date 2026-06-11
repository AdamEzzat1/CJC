# Handoff — Next Arc: From "Accurate & Promoted" to MEASURED Wins

**Date:** 2026-06-11
**Branch:** `claude/naughty-cannon-4b058d` (worktree
`C:\Users\adame\CJC\.claude\worktrees\naughty-cannon-4b058d`)
**HEAD:** `98cc712`
**Supersedes as entry point:** `docs/cana/HANDOFF_PINN_V2.md` (fully
executed — every item shipped or explicitly re-scoped by evidence)
**Roadmap source:** `docs/cana/NEXT_ARC_RESEARCH.md` (six-role research
synthesis; read it BEFORE this doc's Phase details — this handoff
sequences it, that doc evidences it)

---

## 0. THE RULE THAT HAS NOW PAID FOR ITSELF NINE TIMES

**Verify every handoff/design claim against the code before building on
it.** The prior arc caught seven implementation-blocking doc-vs-code
drifts; this arc added two more:

8. The ablation harness stamped Option-A pressure labels on the
   `_t50`/`_c80`/`_c60` recorded configs (`ends_with("_rec")` instead of
   membership in `CONFIG_IDS_RECORDED`) — 402 silently mislabeled rows.
9. The prior handoff's own "HEAD: ee7b341" predated its final commit;
   the real tip was `6980273`.

**`git log` and the code are authoritative; this doc is advice.** The
research doc's numbers were grounded by read-only agents, but its R²
expectations are HYPOTHESES (see §6 flags), not measurements.

---

## 1. State at handoff — what IS done (4 commits this session)

```
98cc712  docs(cana): six-role research synthesis (NEXT_ARC_RESEARCH.md)
9502620  feat: PINN v2 ACTIVATED — full_pinn_v2_rec config + 8 force_*
         configs + cjcl run --pinn-weights + test_pinn_v2_runner
675e049  Merge master (27c8d6f) — pure ancestry join, tree unchanged
8482c64  feat: PINN v2 — TypeMix FP feature + trained thermal head,
         shadow gate PROMOTE
```

Functional state:

- **Trained thermal head SHIPPED + ACTIVE.** `PinnThermalV2`
  (7-feature basis) attached via
  `PinnPhysicalCostModel::with_thermal_head`; identity
  `pinn_thermal_v2` v2 flows into report hashes. Shadow gate (held-out
  31 programs): v1 MAE 0.190/corr +0.18 → v2 MAE 0.021/corr +0.98.
  Plans: `full_pinn_v2_rec` differs from `full_pinn_rec` on 14/134
  programs. CLI: `cjcl run --pinn-weights PATH` (implies `--mir-opt`,
  supersedes `--thermal-aware`, hard-errors on bad bundle).
- **Corpus:** `bench_results/cana_ablation/profiles.cpdb`, schema v2,
  **2,680 rows** = 134 programs × 20 configs (5 synthetic + 7 recorded
  incl. v2 + 8 forced). Energy signal: **295 rows ≠ 1.0, range
  [0.909, 11.16]** (was 39 rows in [0.967, 1.009]). Parity 100%,
  row-hash double-run stable. Regenerate: `cargo run --release -p
  cana-ablation` (~60 s).
- **Trainer:** `bench/cana_train_pinn` modes `sanity` / `train` /
  `shadow`. Closed-form ridge (Kahan + partial-pivot Gauss, no RNG).
  Weights: `bench_results/cana_train_pinn/pinn_thermal_v2.cpb` (CPB0).
- **Honest negative, already measured:** plain linear on the divergent
  energy rows fails held-out (R²(test) ≈ −32). The energy head needs
  the §3 Phase-B treatment, not wishful re-running.
- **Quantum layer:** still ZERO outcome effect — structural
  (reorders within a decided set; one-candidate space). Phase C is the
  designed fix.
- **Blog:** PINN v2 methodology post published to the Quarto blog repo
  (`C:\Users\adame\AdamEzzat1.github.io`). Part-two opportunity after
  Phases C/D produce outcome numbers.

Gate status at HEAD (all green): `fixtures` parity,
`test_pinn_v2_runner` 4/4, `test_mir_exec_instrumented` 5/5, cjc-cana
185/185 lib, cjc-cana-compress 179 lib (+ integration suites),
cjc-cana-nss 21, cjc-nss 219 lib, cjc-mir-exec 28/28, cana-train-pinn
5/5, ablation regen asserts.

---

## 2. The roadmap (from NEXT_ARC_RESEARCH.md — sequenced, exit-criteria'd)

| Phase | Work | Goal | Exit criterion (MEASURED) |
|---|---|---|---|
| **A** | Data + safety foundation (below) | all | corpus ≥5k labels; memory std >0.05; new gates green |
| **B** | Energy head: ridge on schema-v3 features → CPB1 → shadow vs FP_ENERGY_WEIGHT=3.0 | speed | held-out R² + PROMOTE/REJECT verdict |
| **C** | `PassPlanSelector` (10 deterministic candidates/function) + selector gates + ablation config | speed (activates quantum layer) | selected plans beat ranked baseline on measured energy somewhere; parity 100% |
| **D** | `bench/cana_diagnostics` A/B harness (wall-clock + peak-RSS + best-effort thermal; determinism gate FIRST) | thermal+speed proof | byte-equal outputs + deltas with confidence bands |
| **E** | Compression prototypes: checkpoint low-rank, trace RLE, sidecar disk | memory | before/after bytes at bounded error |
| **F** | Memory head (only if Phase-A variance confirms) + NSS→runtime hint design | memory | shadow verdict |

**Parked (with reasons, don't re-litigate without new evidence):**
counterfactual plan scoring, COW-buffer compression, NSS
multi-timescale/advisory/autonomous machinery.

## 3. Phase A — recommended first-session scope, in test-discipline order

Phase A is deliberately the unglamorous one: it makes every later
phase's numbers trustworthy. Items, each small:

1. **Tensor-blindness check FIRST (highest information per hour).**
   `TypeMix` counts scalar float binops only; `tensor_heavy_ops`
   contribute ZERO to `float_ops_estimate`. Hypothesis: the thermal
   head whiffs on tensor workloads (chess-RL, physics_ml) exactly as
   v1 whiffed on scalar FP. Test before fixing: instrument 3–5
   tensor-using programs (reuse `run_program_instrumented`), compare
   recorded thermal vs head predictions. If confirmed → extend TypeMix
   (tensor ops × element estimate as FP work) = FeatureHash change =
   corpus regen + RETRAIN + re-shadow (the whole v2 pipeline re-runs in
   <10 min; it was built for this).
2. **Per-function pressure labels (schema v3).** The per-node data
   already flows (`cjc-cana-nss/src/lib.rs:462-476`); per-program MAX
   is a harness reporting choice (`bench/cana_ablation/main.rs`,
   `max_of`). Store per-fn maps → effective corpus 134 → ~500+.
   Schema bump → corpus regen (cheap, deterministic).
3. **Loop features into rows.** `countable_loop_count`,
   `max_loop_depth` exist in `CfgMetrics` but never reached
   `CompilationProfile`. Add in the same schema-v3 bump as item 2 —
   ONE regen, not two.
4. **Memory-gradient program family** (~5 programs: parametric
   alloc/iter, tensor churn, COW-write stress — sketches in
   NEXT_ARC_RESEARCH.md §2). Target: memory label std 0.0007 → >0.05.
5. **Run the EXISTING verifiers in the harness.** NoGC verifier +
   MIR legality verifier on every optimized program (~20 LOC; they
   exist, they're just not called there). Add code-size bound
   `nodes_after/nodes_before ≤ 1.5` (unroll explosion guard).
6. **Cross-profile determinism canary** (5 min): run the trainer under
   dev AND release; diff bundle bytes. Either hardens the
   bit-reproducibility claim or catches FMA/codegen drift now.
7. **Frozen holdout set.** Pick ~10 NEW programs never used in any
   training/tuning decision; commit them under a `holdout_` name
   prefix; evaluate ONLY at promotion gates. The 134-program split is
   getting worn — every head trained against it erodes it.

Also queued from the recommendations review (any session, independent):
embed default weights as a compiled-in `const` (default-on path;
`--pinn-weights` stays as override); Locke/LendingClub peak-RSS A/B as
the memory macro-benchmark; upstream fix for the cjc-quantum
wide-matrix SVD bug (checkpoint compression leans on it); root-cause
the 5 pre-existing serializer/replay failures
(`docs/cana/CANA_PHASE_1_REGRESSION_FAILURES.md`) before CPB1/schema-v3
serialization lands on top.

---

## 4. TEST DISCIPLINE CONTRACT (user-specified — applies to EVERY phase)

For each feature, in this order:

1. **Wiring tests** — prove the piece is actually connected end-to-end,
   not just unit-correct. Precedents to copy: `tests/test_pinn_v2_runner.rs`
   (CLI/executor path under the real committed bundle),
   `tests/physics_ml/grad_graph_wiring.rs` pattern (every surface
   AST↔MIR byte-equal). For schema v3: a row written by the harness
   reads back field-for-field; for the selector: selected plan reaches
   `optimize_program_with_plan` and the executor.
2. **Unit tests** — per-module behavior + edge cases (precedent:
   `type_mix.rs` 6 tests incl. the If-cond regression;
   `pinn_bundle.rs` corruption/truncation/NaN-rejection matrix).
3. **Proptests** — properties, not examples (precedent:
   `crates/cjc-cana-compress/tests/proptest_compress.rs`). Candidates:
   selector tie-break total order (any candidate set → exactly one
   winner, stable under permutation); schema v3 round-trip ∀ field
   values; per-function label maps round-trip with arbitrary BTreeMap
   contents.
4. **Bolero fuzz** — adversarial bytes/structures (precedent:
   `crates/cjc-cana-compress/tests/bolero_fuzz.rs`, which forced the
   GradGraph bounds-hardening). Candidates: CPB1 decoder (raw bytes →
   Err-not-panic); schema-v3 row decoder; selector under fuzzed
   feature structs (never panics, never emits an illegal plan).
5. **Verification loop** — run before AND after every change set:

```bash
cargo test --test fixtures --release                # THE parity gate
cargo test --test test_pinn_v2_runner --release     # v2 end-to-end (4)
cargo test --test test_mir_exec_instrumented --release  # transparency (5)
cargo test -p cjc-cana --release --lib              # 185 at handoff
cargo test -p cjc-cana-compress --release           # 179 lib + suites
cargo test -p cjc-cana-nss --release                # 21
cargo test -p cjc-nss --release --lib               # 219
cargo test -p cjc-mir-exec --release --lib          # 28
cargo test -p cana-train-pinn --release             # 5
cargo run --release -p cana-ablation                # regen + asserts (~60 s)
cargo run --release -p cana-train-pinn -- sanity    # corpus health read
```

   Never `#[ignore]` a failure. If the corpus changes (schema bump,
   TypeMix change), the regen + sanity + (if features changed) retrain
   + shadow sequence is MANDATORY before claiming anything.
6. **Documentation of changes AND experiment results** — extend
   `docs/cana/PINN_V2_DESIGN.md`-style notes: design decision → what
   was measured → verdict, including NEGATIVE results (the −32 R² and
   the quantum null are load-bearing facts; record their successors).
   Update `NEXT_ARC_RESEARCH.md` checkboxes/flags as hypotheses get
   settled. Close the session with a fresh handoff in this pattern.

---

## 5. Traps known at handoff (verified this session — will bite)

- **`PassPlan` absence semantics:** functions ABSENT from the map get
  the FULL `DEFAULT_PASS_SEQUENCE`; present-with-empty-vec get NOTHING
  (`cjc-mir/src/optimize.rs:342-356`). Selector candidates must insert
  explicit entries for every function. `force_none` relies on this.
- **Config-name matching:** never suffix-match config ids (the
  `ends_with("_rec")` bug). Membership in the const arrays only.
- **Schema version is strict:** `PROFILE_SCHEMA_VERSION` mismatch =
  read rejection, no migration. Bump → regenerate → sanity re-run.
  Same for CPB bundles (`PINN_BUNDLE_SCHEMA_VERSION`, feature-count
  check guards basis drift).
- **`FnFeatures` is `Copy` and feeds `FeatureHash`:** any new field
  changes every hash and breaks every literal constructor (two sites:
  `features.rs` extract + `physical_cost.rs` test helper — grep
  `FnFeatures {` after any change).
- **Energy formula lives in the harness**, not a crate:
  `FP_ENERGY_WEIGHT = 3.0` in `bench/cana_ablation/main.rs`. Phase B's
  shadow baseline is THAT constant.
- **rustfmt + `#[path]`:** `cana_ablation` path-includes
  `bench/cana_train_cost_model/programs.rs`; formatting one reformats
  the other. A stray workspace-wide `cargo fmt` was reverted once this
  session — don't let it sneak into a feature commit.
- **The diagnostics wall:** wall-clock/RSS/thermal measurements are
  POST-HOC DIAGNOSTICS. They never enter hashes, rows' stable fields,
  or any decision path. The auditor's design (research doc §3) starts
  with a byte-equality gate between A/B arms — keep that order.

## 6. Hypotheses vs facts (do not confuse)

| Claim | Status |
|---|---|
| Thermal head beats v1 9× held-out; 14/134 plans change | **Measured** |
| Energy rows: 295 informative, [0.909, 11.16]; plain linear fails (−32) | **Measured** |
| Ridge + loop features reaches R² 0.65–0.75 on energy | **Hypothesis** (Phase B settles it) |
| Quantum layer zero effect is structural (one-candidate space) | **Measured + code-verified** |
| Selector finds energy wins on real programs | **Hypothesis** (Phase C) |
| Thermal head is tensor-blind | **Hypothesis, high prior** (Phase A item 1) |
| Runtime scheduler saves 20–30% peak memory | **Speculation** (no design even fixed) |
| Training bit-reproducible across build profiles | **Untested** (Phase A item 6, 5 minutes) |

## 7. Housekeeping (user-level, one step each)

1. **Fast-forward master:** from the MAIN worktree (`C:\Users\adame\CJC`):
   `git merge claude/naughty-cannon-4b058d` — clean FF, ancestry
   already joined by `675e049`.
2. **Stop the stale chip session:** the "forced-plan ablation configs"
   chip (task_4373d306) was started in a parallel worktree BEFORE the
   work landed inline here (`9502620`). That session is redundant —
   stop/discard it rather than merging a second implementation.
3. The 5 serializer/replay failures and the cjc-quantum SVD upstream
   fix remain open debt (see §3 queue).

## 8. File map (this arc's surface)

```
crates/cjc-cana/src/type_mix.rs             TypeMix (scalar-FP only — see Phase A item 1)
crates/cjc-cana/src/pinn_thermal_v2.rs      trained head + features_from_query (THE basis definition)
crates/cjc-cana/src/pinn_cost_model.rs      with_thermal_head swap-in; v1/v2 identity flip
crates/cjc-cana/src/physical_cost.rs        float_ops_estimate (additive); FP_ENERGY-adjacent query build
crates/cjc-cana-compress/src/pinn_bundle.rs CPB0 codec (copy this pattern for CPB1)
crates/cjc-cana-compress/src/profile_db.rs  schema v2 rows (v3 lands here)
bench/cana_train_pinn/main.rs               sanity/train/shadow + the deterministic OLS machinery
bench/cana_ablation/main.rs                 20 configs incl. force_* + v2; energy formula; ALL regen gates
crates/cjc-mir-exec/src/lib.rs              run_program_optimized_pinn_v2*, instrumented runner
crates/cjc-cli/src/lib.rs                   --pinn-weights wiring
tests/test_pinn_v2_runner.rs                v2 end-to-end parity gate (4)
docs/cana/NEXT_ARC_RESEARCH.md              the six-role research (Phases A–F evidence)
docs/cana/PINN_V2_DESIGN.md                 v2 results record (§5–6) — extend, don't fork
bench_results/cana_ablation/profiles.cpdb   2,680-row corpus (committed)
bench_results/cana_train_pinn/pinn_thermal_v2.cpb  trained weights (committed)
```

## 9. Determinism invariants (unchanged, non-negotiable)

BTreeMap everywhere; Kahan/Binned FP reductions; no FMA; SplitMix64
seeded RNG only; FNV-1a hashing; `f64::total_cmp`; no wall-clock or
sensors in decision paths or hashes (diagnostic-only, excluded from
`row_hash`); model_id+version in report hashes for every trained or
selecting component; training offline only; shadow mode before any
trained artifact activates; legality gate retains final authority.

---

*Closing artifact of the 2026-06-10/11 sessions (8482c64..98cc712:
PINN v2 trained→promoted→activated, energy-variance groundwork,
six-role research). Next session: Phase A, starting with the
tensor-blindness check — it is the cheapest experiment with the
largest possible consequence for everything downstream.*
