# Handoff — Phase D: From Modeled Wins to Silicon Proof

**Date:** 2026-06-11 (closing the A+B+C triple session)
**Branch:** `claude/hopeful-elion-72485e` — commits `ed4793e` (Phase A),
`6563e2a` (Phase B), `029d4dd` (Phase C), plus the winner-listing
report tweak after C.
**Master:** at `ed4793e` (Phase A merged); **B+C fast-forward pending**
— one step from the MAIN worktree (`C:\Users\adame\CJC`):
`git merge claude/hopeful-elion-72485e`.
**Supersedes as entry point:** `docs/cana/HANDOFF_PHASE_A.md` (its §7
was updated in place as B and C executed; THIS doc consolidates).
**Numbers ledger:** `docs/cana/PINN_V2_DESIGN.md` §7–§10.
**Hypothesis log:** `docs/cana/NEXT_ARC_RESEARCH.md` §7.

---

## 0. THE RULE (now paid for 13+ times)

**Verify every handoff/design claim against the code before building
on it.** This session's additions to the casualty list: the thermal
LABEL was tensor-blind, not just the features (#10); the 1.5× size
bound contradicted designed unrolling on first contact (#11); the
R²-best energy fit was the regret-worst selector criterion (#12); the
memory label is structurally Rc-blind, so its variance target was
unachievable by any program (#13). `git log` and code are
authoritative; this doc is advice.

## 1. Honest project status — what has CANA/NSS actually bought the compiler?

Keep this framing straight when planning Phase D (it is the answer to
"has any of this helped yet"):

**Active in the compiler today (default or flag-reachable):**
- CANA ranking runs in the default `--mir-opt` path (functions CANA
  doesn't override fall back to the full default sequence —
  byte-identical guarantee), with the thermal-aware stack as standard
  (`b23f395`, pre-this-session) and the trained thermal head behind
  `cjcl run --pinn-weights`.
- NSS Option-B instrumentation is THE label source — every trained
  artifact (thermal v3, energy v1) exists because of it.

**Measured outcome effects so far (modeled-energy metric, corpus):**
- Ranked configs (everything before Phase C) overwhelmingly TIE the
  baseline on measured energy (148/158 ties) — their value is
  skipping predicted-useless passes and changing plans on hot
  functions, with NO demonstrated runtime/energy benefit yet.
- Phase C's `selector_rec` is the FIRST config with mean measured
  energy below baseline: **0.98230** vs incumbent 1.00329 — 6 wins
  (best −50% executed work), 16 modest regressions (worst +14%),
  parity 100%. Ablation-grade, NOT default-on.

**So: ~the entire arc to date is instrumentation + trained-advisory
infrastructure, now with ONE measured (modeled) win.** No user-facing
performance claim is defensible until Phase D measures wall-clock.
That is by design — but say it plainly in anything outward-facing.

## 2. Phase D spec (research doc §3 — the auditor's design)

New crate `bench/cana_diagnostics`. **Contract: determinism gate
FIRST** — byte-identical program outputs between A and B arms is a
hard error before any timing is read. Then:

- Protocol: 2 warm-up runs; interleaved A/B/A/B; ~5 s sustained-load
  phases (thermal time constants are seconds+); median-of-5
  wall-clock; peak RSS via `GetProcessMemoryInfo`.
- Windows signal reality: wall-clock (±100 ns) and peak RSS (±1%)
  reliable; CPU frequency ~1 Hz best-effort; CPU temperature via WMI
  is `Option<f64>` (often absent); thermal-throttle state NOT
  reliably detectable. Report frequency TRENDS within runs; never
  absolute cross-machine temperatures.
- **Hard wall: diagnostics never feed back into decisions, hashes, or
  row stable fields** (same rule as `compile_wall_micros`).

**A/B arms and subjects:**
1. Baseline-plan arm vs selector-plan arm on the 6 selector-win
   programs. NAMED (ablation report `[selector win]` lines, sorted):
   `mem_grad_a5` 0.49613, `mem_grad_a4` 0.49616,
   `holdout_alloc_pulse` 0.49625, `mem_grad_a3` 0.49628,
   `mem_grad_a2` 0.49673, `mem_grad_a1` 0.49855.
   **Read these names honestly:** all 6 are the allocation-churn
   shape (dead per-iteration allocations in a loop) — the selector's
   discovery is that DCE halves executed work where the ranked stack
   under-recommends it. ONE win mechanism, not six independent ones.
   The strong part: `holdout_alloc_pulse` is FROZEN-HOLDOUT — the
   mechanism generalized to a program neither head ever trained on.
   The weak part: synthetic dead-alloc loops are the most
   DCE-friendly shape imaginable; Phase D should ALSO time a
   non-synthetic subject before any broad claim.
   THE question: does −50% executed work appear on the stopwatch?
2. fp_hot + grad_f9* family (sustained scalar FP; thermal-aware vs
   plain) — the original Track-3 subjects.
3. Tensor family (the A1 programs) — validates the energy formula's
   FP weighting on tensor dispatch.
4. Optionally: `--pinn-weights` arm vs plain on the 38 plans-differ
   programs.

**Exit criterion (roadmap §6 row D):** byte-equal outputs + reported
wall-clock/RSS deltas with confidence bands (median-of-5 spread). A
NEGATIVE result (modeled wins don't show on the stopwatch) is a
load-bearing finding — it would redirect the energy formula (Phase B
recalibration) before anything else is built on it.

## 3. Selector hardening (queued AFTER Phase D's verdict)

- **Margin gating:** keep the ranked plan unless predicted gain
  exceeds a corpus-calibrated threshold — kills most of the 16
  regressions at the cost of some wins; tune on test, verify on
  frozen holdout.
- **Head-independent exploration configs:** add forced versions of
  the selector's candidate shapes (e.g. `force_reordered_full`) so
  training sees novel pass combinations WITHOUT the feedback loop;
  then retrain energy head → regret should improve out-of-distribution.
- Per-function `optimize` API to avoid whole-program re-optimization
  per candidate (selector currently optimizes the program once per
  distinct candidate plan per function — fine for the corpus,
  wasteful at scale).

## 4. Phases E/F (unchanged from research doc §6)

E: compression prototypes (checkpoint low-rank, trace RLE, sidecar
disk) — before/after bytes at bounded error. F: memory head — BLOCKED
on a label-side fix first (the heap proxy only sees `gc_alloc`
objects ×4096 + executed arena-`Let`s ×64; Rc buffers are invisible —
measured ceiling 0.0078). Sketch: per-window allocated-bytes counter
at creation sites (the A1 pattern applied to memory), which is a
MirTraceEvent schema change → adapter + corpus ripple.

## 5. TEST DISCIPLINE CONTRACT (unchanged) + verification loop

Wiring → unit → proptest → bolero → verification loop → docs. The
loop, updated with this session's gates:

```bash
cargo test --test fixtures --release                    # THE parity gate
cargo test --test test_pinn_v2_runner --release         # thermal v3 e2e (4)
cargo test --test test_mir_exec_instrumented --release  # transparency (5)
cargo test --test test_tensor_fp_accounting --release   # A1 wiring (6)
cargo test --test test_tensor_typemix_props --release   # A1 props/fuzz (4)
cargo test --test test_energy_head --release            # B wiring (5)
cargo test --test test_cana_energy_selector --release   # C gates (7)
cargo test -p cjc-cana --release --lib                  # 209
cargo test -p cjc-cana-compress --release               # 190 lib + suites
cargo test -p cjc-cana-nss --release                    # 21
cargo test -p cjc-nss --release --lib                   # 219
cargo test -p cjc-mir-exec --release --lib              # 28
cargo test -p cana-train-pinn --release                 # 5
cargo run --release -p cana-ablation                    # regen + asserts (~4 min)
cargo run --release -p cana-tensor-probe                # must print NOT BLIND
cargo run --release -p cana-train-pinn -- sanity        # corpus health
```

After ANY corpus-affecting change: regen → sanity → retrain (both
heads) → re-shadow → verify BOTH bundle fixed points byte-identically
(`Get-FileHash` before/after retrain).

## 6. Traps (accumulated; all verified)

- `PassPlan` absence = FULL default sequence; present-empty = nothing.
  The selector inserts explicit entries everywhere — keep it that way.
- Config-name membership tests only (never suffix/substring matching).
  KNOWN REMAINING INSTANCE: the trainer selects label rows via
  `config_id.contains("_rec")` (two sites) — currently safe, convert
  to membership lists when touched.
- `ENERGY_EXCLUDED_CONFIGS = ["selector_rec"]` is the feedback-loop
  guard — any NEW selector-driven config must be added there, or the
  energy head trains on its own decisions.
- Schema versions are strict (PROFILE v3, CPB0 thermal, CPB1 energy
  with vocabulary-dependent feature count); bump → regen → retrain.
- `FnFeatures`/`TypeMix` feed `FeatureHash` — any new field changes
  every hash; grep literal constructors after changes.
- Per-window `thermal_intensity` caps at 1.0 — Σ intensity·instr
  under-recovers FP on multi-FP-op statements (~23% on dense scalar);
  the energy label inherits this.
- Code-size cap is 16× (full unroll of 8-trip loops measures 6.24× by
  design); the regen prints the corpus max — tighten from evidence.
- Energy formula (`FP_ENERGY_WEIGHT = 3.0`) lives in
  `bench/cana_ablation/main.rs` and DEFINES the score labels — Phase D
  is what validates/recalibrates it against silicon.
- PowerShell 5.1 mangles embedded double quotes in args to native
  exes (git commit messages — use here-strings WITHOUT inner `"`).
- rustfmt + `#[path]`: cana_ablation path-includes
  `cana_train_cost_model/programs.rs`; no workspace-wide `cargo fmt`.

## 7. Hypotheses vs facts (current ledger)

| Claim | Status |
|---|---|
| Thermal head accurate incl. tensor programs (v3) | **Measured** (holdout MAE 0.19 vs v1 0.48) |
| Energy signal learnable; recipe = ln target + loops + structural, all-rows fit | **Measured** (R² 0.82 exists; regret-chosen fit ships) |
| Selector finds measured energy wins | **Measured** (6 wins, mean 0.982, +16 regressions) |
| Modeled-energy wins appear on wall-clock | **HYPOTHESIS — Phase D's entire job** |
| Selector regressions fixable by margin gating | Hypothesis (§3) |
| Memory label fixable via creation-site byte counting | Design sketch only (Phase F) |
| Training bit-reproducible across build profiles | **Measured** (dev = release, byte-identical) |

## 8. File map (this session's full surface)

```
crates/cjc-cana/src/type_mix.rs              tensor-ness propagation (A1)
crates/cjc-cana/src/memory_proxy.rs          method-call classification (A1)
crates/cjc-cana/src/physical_cost.rs         TENSOR_FP_PER_OP pricing (A1)
crates/cjc-cana/src/pinn_thermal_v2.rs       density cap, model v3 (A1)
crates/cjc-cana/src/pinn_energy_v1.rs        energy head + EnergyQuery basis (B)
crates/cjc-cana/src/plan_selector.rs         PassPlanSelector (C)
crates/cjc-cana-compress/src/profile_db.rs   schema v3 FnProfile (A2+A3)
crates/cjc-cana-compress/src/energy_bundle.rs CPB1 codec (B)
crates/cjc-mir-exec/src/lib.rs               tensor FP accounting, u64 counter (A1)
bench/cana_tensor_probe/                     A1 probe (regression check: NOT BLIND)
bench/cana_ablation/main.rs                  21 configs; families; verifier gates; winner listing
bench/cana_train_pinn/main.rs                modes: sanity/train/shadow/sanity-energy/train-energy/shadow-energy
bench_results/cana_ablation/profiles.cpdb    3,318-row v3 corpus (committed)
bench_results/cana_train_pinn/pinn_thermal_v2.cpb  thermal weights (CPB0, model v3)
bench_results/cana_train_pinn/pinn_energy_v1.cpb   energy weights (CPB1, v1)
tests/test_tensor_fp_accounting.rs           A1 wiring (6)
tests/test_tensor_typemix_props.rs           A1 props/fuzz (4)
tests/test_energy_head.rs                    B wiring (5)
tests/test_cana_energy_selector.rs           C gates (7)
```

## 9. Housekeeping

1. Master FF for B+C pending (user or agent, main worktree, one merge).
2. ABNG serializer/replay debt: chip `task_a41b1c8d` raised (5
   byte-equality failures, root-cause in `cjc-abng`, triage doc
   `CANA_PHASE_1_REGRESSION_FAILURES.md`).
3. Still queued: cjc-quantum wide-matrix SVD upstream fix; embed
   default thermal weights as compiled-in const (default-on path);
   Locke/LendingClub peak-RSS macro A/B (pairs naturally with
   Phase D's RSS plumbing).

## 10. Determinism invariants — unchanged, non-negotiable

BTreeMap everywhere; Kahan/Binned FP reductions; no FMA; SplitMix64
only; FNV-1a; `f64::total_cmp`; no wall-clock/sensors in decision
paths or hashes (Phase D's measurements are POST-HOC DIAGNOSTICS,
byte-equality-gated first); model/selector identity in report hashes;
training offline only; shadow before activation; feedback-loop guard
on every selector-driven config; legality gate final authority.

---

*Closing artifact of the 2026-06-11 A+B+C session
(`ed4793e`..`029d4dd`+). Next session: Phase D — put the 6 selector
wins on a stopwatch. A negative result there is as valuable as a
positive one: it recalibrates the energy formula before Phases E/F
build on it.*
